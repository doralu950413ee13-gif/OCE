"""White-box robustness evaluation for OCE: search for a continuous
embedding c_adv that bypasses an OCE-edited UNet's cross-attention erasure.

This targets the exact mechanism oce.py implements (see oce.py:104-121):
Orthogonal_Erase rotates only `attn2.to_v` weights with W_new = R @ W0, where
R is chosen (via the Procrustes/SVD step) to push the erased concept's
directions in to_v's *output* space away from the model's actual behaviour.

c_adv is a free tensor shaped like a CLIP text-encoder output sequence
(1, seq_len, hidden_dim) that is fed directly into the UNet's
`encoder_hidden_states` for cross-attention, bypassing the tokenizer and
text encoder entirely (a "soft prompt" / continuous-embedding attack, as
opposed to searching over discrete token strings).

This is a research tool for evaluating your own erased checkpoint's
robustness under an adaptive, white-box attacker who knows the erasure
mechanism (the standard threat model in adversarial-robustness evaluation,
matching benchmarks like UnlearnDiffAtk / Ring-A-Bell / P4D for other
erasure methods such as ESD/UCE/SPM). It requires access to BOTH the
pre-edit and post-edit weights of the same model, which you have here
since oce.py never mutates pipe.unet in place — it only writes the rotated
copy to a separate .safetensors file (oce.py:122,127-129).

CAVEATS (read before treating this as more than a skeleton):
  - Backpropagating through the full multi-step denoising loop is very
    memory-hungry (autograd keeps the whole computation graph across every
    UNet call). --num_steps defaults small for that reason; a serious run
    should either use gradient checkpointing or truncate backprop to the
    last few steps only.
  - L_target needs a *differentiable* path into CLIP, so it can't use
    transformers' CLIPProcessor (PIL/numpy, not differentiable) — see
    `clip_preprocess` below for a minimal torch-native substitute.
  - L_evasion's exact subspace definition is a design choice (see the
    module docstring above); an equally valid alternative is projecting
    onto `I - G_star` (the guide-complement used inside M_total at
    oce.py:110) instead of the raw erase subspace E.
"""
import argparse
import copy

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from safetensors.torch import load_file
from transformers import CLIPModel

from oce import build_erase_subspace

# oce.py disables autograd globally at import time (`torch.set_grad_enabled(False)`,
# oce.py:2) since Orthogonal_Erase never needs gradients. This script does, so
# re-enable it — everything except c_adv stays gradient-free via requires_grad_(False).
torch.set_grad_enabled(True)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def freeze_model(pipe):
    for m in (pipe.unet, pipe.text_encoder, pipe.vae):
        m.requires_grad_(False)
        m.eval()


def get_to_v_modules(unet):
    return [(name, module) for name, module in unet.named_modules()
            if 'attn2' in name and name.endswith('to_v')]


def encode_last_token(pipe, prompt, device):
    """Same extraction oce.py uses to build erase/guide/preserve embeddings
    (oce.py:84-89): the hidden state at the end-of-text token position."""
    t_emb = pipe.encode_prompt(prompt=prompt, device=device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False)
    last_idx = (pipe.tokenizer(prompt, padding="max_length",
                                max_length=pipe.tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt")["attention_mask"]).sum() - 2
    return t_emb[0][:, last_idx, :].squeeze(0)


def build_original_erase_subspaces(to_v_modules_original, erase_embed):
    """Recompute, per to_v layer, the exact subspace oce.py built at edit
    time from the PRE-edit weights (oce.py:107: build_erase_subspace(W0,
    erase_embs)) — this is what the rotation R tried to suppress."""
    subspaces = {}
    for name, module in to_v_modules_original:
        W0 = module.weight.detach()
        subspaces[name] = build_erase_subspace(W0, [erase_embed])  # (out_dim, 1)
    return subspaces


def init_c_adv(pipe, seed_prompt, device, dtype):
    with torch.no_grad():
        seed_emb = pipe.encode_prompt(prompt=seed_prompt, device=device,
                                       num_images_per_prompt=1,
                                       do_classifier_free_guidance=False)[0]
    return seed_emb.clone().to(dtype=dtype).requires_grad_(True)


def _clip_features(out):
    """transformers' CLIPModel.get_{text,image}_features return a plain
    tensor on some versions and a BaseModelOutputWithPooling on others."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def clip_preprocess(images, size=224):
    """Differentiable stand-in for CLIPProcessor (bilinear resize + CLIP's
    fixed normalization). images: (B,3,H,W) in [0,1]."""
    images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)
    mean = CLIP_MEAN.to(images.device, images.dtype)
    std = CLIP_STD.to(images.device, images.dtype)
    return (images - mean) / std


def decode_latents(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents).sample
    return (image / 2 + 0.5).clamp(0, 1)


def generate_with_c_adv(pipe, c_adv, num_steps, device, dtype):
    """Truncated diffusion sampling loop, kept differentiable end-to-end.
    See the memory caveat in the module docstring before raising num_steps."""
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_steps, device=device)
    shape = (1, pipe.unet.config.in_channels,
              pipe.unet.config.sample_size, pipe.unet.config.sample_size)
    latents = torch.randn(shape, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    for t in scheduler.timesteps:
        noise_pred = pipe.unet(latents, t, encoder_hidden_states=c_adv).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return decode_latents(pipe, latents)


def attack(args):
    device = args.device
    dtype = torch.float32

    # ---- pipe #1: kept at ORIGINAL (pre-edit) weights, only used to
    # recompute the erase subspace oce.py built at edit time ----
    pipe = DiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype,
                                              safety_checker=None).to(device)
    freeze_model(pipe)
    to_v_original = copy.deepcopy(get_to_v_modules(pipe.unet))

    erase_embed = encode_last_token(pipe, args.target_concept, device)
    erase_subspaces = build_original_erase_subspaces(to_v_original, erase_embed)

    # ---- now load the OCE-edited weights in place: this is the actual
    # deployed model the adversary generates images against ----
    if args.oce_weights is not None:
        edited = load_file(args.oce_weights)
        pipe.unet.load_state_dict(edited, strict=False)
    to_v_modules = dict(get_to_v_modules(pipe.unet))

    c_adv = init_c_adv(pipe, args.seed_prompt, device, dtype)
    optimizer = torch.optim.Adam([c_adv], lr=args.lr)

    clip_model = CLIPModel.from_pretrained(args.clip_id).to(device)
    clip_model.requires_grad_(False)
    clip_model.eval()

    with torch.no_grad():
        # NOTE: uses transformers' tokenizer (fine, text side isn't part of
        # the differentiable graph) but bypasses CLIPProcessor for images.
        from transformers import CLIPTokenizer
        clip_tok = CLIPTokenizer.from_pretrained(args.clip_id)
        text_ids = clip_tok([args.target_concept], return_tensors="pt", padding=True).to(device)
        target_clip_text = F.normalize(_clip_features(clip_model.get_text_features(**text_ids)), dim=-1)

    for step in range(args.attack_iters):
        optimizer.zero_grad()

        image = generate_with_c_adv(pipe, c_adv, args.num_steps, device, dtype)
        clip_size = clip_model.config.vision_config.image_size
        image_feat = F.normalize(
            _clip_features(clip_model.get_image_features(pixel_values=clip_preprocess(image, size=clip_size))), dim=-1
        )
        L_target = 1 - (image_feat @ target_clip_text.T).mean()

        L_evasion = torch.zeros((), device=device, dtype=dtype)
        if args.evasion_weight > 0:
            c_adv_eot = c_adv[:, -1, :].squeeze(0)  # crude: last-token slot, see docstring
            for name, module in to_v_modules.items():
                v = module.weight @ c_adv_eot
                E = erase_subspaces[name].to(device)
                L_evasion = L_evasion + (E.T @ v).pow(2).sum()

        loss = L_target + args.evasion_weight * L_evasion
        loss.backward()
        optimizer.step()

        print(f"[{step:03d}] L_target={L_target.item():.4f} "
              f"L_evasion={L_evasion.item():.4f} loss={loss.item():.4f}")

    torch.save(c_adv.detach().cpu(), args.save_path)
    print(f"Saved c_adv -> {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='attack_oce',
        description='White-box adversarial embedding search against an OCE-edited UNet')
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4')
    parser.add_argument('--oce_weights', type=str, required=True,
                         help='.safetensors produced by oce.py (the edited to_v weights)')
    parser.add_argument('--clip_id', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--target_concept', type=str, required=True,
                         help='the concept oce.py erased, e.g. "airplane"')
    parser.add_argument('--seed_prompt', type=str, default='a photo',
                         help='initializes c_adv from a real prompt embedding')
    parser.add_argument('--num_steps', type=int, default=4,
                         help='denoising steps unrolled through backprop (see memory caveat)')
    parser.add_argument('--attack_iters', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--evasion_weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_path', type=str, default='./c_adv.pt')
    args = parser.parse_args()

    attack(args)
