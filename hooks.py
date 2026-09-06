"""Forward hooks for recording activations of the projection matrices that
oce.py actually edits (cross-attention `attn2.to_v`), so the outputs can be
used as observation points for a downstream adversarial loss.

Note: oce.py only rotates `to_v` (see Orthogonal_Erase's target-module
collection, `name.endswith('to_v')`). `to_k` and `to_q` are never edited, so
hooking them would not reflect the erasure at all.
"""
import torch


def get_edited_module_names(unet):
    return [name for name, _ in unet.named_modules()
            if 'attn2' in name and name.endswith('to_v')]


def register_feature_hooks(unet, module_names=None, store=None, detach=True):
    """Register forward hooks on the given (or all edited) attn2.to_v modules.

    Each hook call appends the module's output tensor to store[name] — a
    forward pass through the UNet happens once per denoising step, so after
    running inference store[name] holds one activation per step.

    Returns (store, handles); call handle.remove() on each handle when done.
    """
    if module_names is None:
        module_names = get_edited_module_names(unet)
    if store is None:
        store = {name: [] for name in module_names}

    named = dict(unet.named_modules())
    handles = []
    for name in module_names:
        module = named[name]

        def make_hook(name):
            def hook(module, inputs, output):
                feat = output.detach() if detach else output
                store[name].append(feat)
            return hook

        handles.append(module.register_forward_hook(make_hook(name)))
    return store, handles


def remove_hooks(handles):
    for h in handles:
        h.remove()


if __name__ == '__main__':
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        "hf-internal-testing/tiny-stable-diffusion-pipe",
        torch_dtype=torch.float32,
        safety_checker=None,
    ).to("cpu")

    store, handles = register_feature_hooks(pipe.unet)
    print(f"Hooked modules: {list(store.keys())}")

    pipe("a photo of a cat", num_inference_steps=4, guidance_scale=1.0)

    for name, feats in store.items():
        shapes = [tuple(f.shape) for f in feats]
        print(f"{name}: {len(feats)} calls, shapes={shapes}")

    remove_hooks(handles)
