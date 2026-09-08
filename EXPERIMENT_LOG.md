# OCE 論文重現實驗紀錄（2026-09-06 ~ 2026-09-08）

分支：`experiment/paper-reproduction`（程式碼與 `main` 完全一致，0 commits diff）；本檔案與 CSV 結果放在獨立的 `experiment/paper-reproduction-results` 分支上，避免大量二進位圖片污染程式碼分支。

## 環境
- `.venv`：Python 3.12（`uv venv --python 3.12`），因系統預設 Python 3.14 缺 pip/wheel 支援而改用。
- `requirements.txt` 的 `transformers>=4.48.0` 在裝到 5.x 時會跟 `diffusers==0.33.1` 不相容（`ImportError: FLAX_WEIGHTS_NAME`），手動鎖在 `transformers==4.49.0`。
- GPU：RTX 5090 32GB，torch 2.14.0+cu130。

## 做了什麼

### 1. 最小 smoke test（object: airplane→sky）
驗證 `compute_Cg.py` → `oce.py` → `generate_object.py` → `attack_prompts.py` 整條 pipeline 可跑通。

### 2. celeb_10 leakage 分析
用完整 COCO caption 重算 `Cg.pt`，訓練 `celeb_10`（10 位名人→person/woman/man），用 `attack_prompts.py` 對 10 個 (celeb, "person") pair 跑 9 種 discrete-prompt 攻擊，找出 CLIP-leakage 最高的失敗案例。

### 3. one-to-many 功能合併
把 `feature/one-to-many-unlearning` merge 進 `main`（merge commit `224d3a5`），解掉 `compute_Cg.py` 唯一的衝突（保留 argparse CLI 版本）。

### 4. 論文原始 6 組基準實驗完整重現（本次主要工作，`experiment/paper-reproduction` 分支）
訓練 + 完整 evalscripts 規模生成 + 量化 metrics，涵蓋：`object`（airplane→sky）、`style`（Van Gogh→real）、`nudity`、`celeb_10`、`celeb_50`、`celeb_100`。

**規模**：約 14,687 張生成圖（cifar10 2000 + style 900 + nudity 142 + celeb 官方圖 2300 + celeb leakage 攻擊 1350 + COCO baseline 3000 + COCO 6模型×~831 partial ≈ 4986）。原估計需 8 小時，實際 **約 3 小時 44 分鐘**完成。

**過程中發現並繞過的 3 個 repo bug**（未修改原始程式檔案，只在呼叫方式上避開）：
1. `generate_object.py` 單次 `num_images_per_prompt=200` 時 VAE decode 在 32GB GPU 上 OOM（需要 ~35GB）→ 改用外部腳本分批（25 張/批）生成再合併檔名。
2. `evalscripts/generate_i2p.sh` 傳給 `generate_nsfw.py` 的參數名是 `--uce_model_path`，但該腳本實際定義的是 `--oce_model_path`，會直接報 `unrecognized arguments`。
3. `data/nudity.csv` 部分 prompt 超過 CLIP 77-token 上限，`metrics/eval_clip_score.py` 沒做截斷會直接 crash → 另存一份 CLIP-tokenizer 截斷過的 `data/nudity_truncated.csv` 餵給它。

**量化結果**（2026-09-08 已補跑到完整 3000 張/model，取代先前 831 張抽樣的初版數字）：

| 模型 | COCO CLIP score（3000張，完整規模） | FID vs baseline（3000張，完整規模） |
|---|---:|---:|
| baseline（未編輯） | 31.37 | – |
| object | 31.19 | 13.59 |
| style | 31.39 | 10.36 |
| nudity | 30.91 | 18.29 |
| celeb_10 | 31.36 | 11.77 |
| celeb_50 | 31.12 | 16.30 |
| celeb_100 | 30.71 | 18.31 |

**方法論發現**：先前用 831 張抽樣算出的 FID（61.85 ~ 63.92）比完整 3000 張算出的（10.36 ~ 18.31）高了 **3.4x ~ 5.6x**，證實 FID 在小樣本下有嚴重的正向偏誤（估計器對樣本數敏感，樣本越少偏誤越大），CLIP score 則幾乎不受影響（差距 <0.5 分）。這代表小樣本 FID 數字方向可以參考排序趨勢，但絕對數值不可信，往後做類似量化評估時應優先確保 FID 用足量樣本計算。celeb_10→50→100 隨規模惡化的趨勢在完整規模下依然成立（11.77→16.30→18.31）。

- **Object 抹除**：CLIP zero-shot 分類下 `airplane` 類別平均機率 0.144（對照組 `cat` 仍 0.994），抹除精準且未傷及無關類別。

### 5. 補完 CIFAR-10 物件抹除，重現論文 Table 1/8（2026-09-08）
翻論文 PDF 才發現：原本的 `object` 實驗只訓練了 `trainscripts/object.sh` 內建的 **airplane→sky 一個模型**，不是論文 Table 1/8 的完整方法——論文 **Table 7**（page 14）定義了 CIFAR-10 十個類別**各自**對應的 anchor（訓練 10 個獨立模型），其中 **Cat↔Dog 互為 anchor**（Figure 5，page 19，就是使用者問的「狗變貓」）：

| Class | Airplane | Automobile | Bird | Cat | Deer | Dog | Frog | Horse | Ship | Truck |
|---|---|---|---|---|---|---|---|---|---|---|
| Anchor | Sky | Truck | Cat | Dog | Horse | Cat | Bird | Deer | Airplane | Ship |

於是補訓練另外 9 個模型（沿用 `trainscripts/object.sh` 的超參數 `erase_scale=2000, preserve_global_scale=10, preserve_concept_scale=0, lamb=10`，只換 edit/guide concept），每個模型生成全部 10 類 × 200 張圖（airplane 模型複用既有圖），共 20,000 張圖，寫一支新腳本算 CLIP top-1 分類，重現論文的 Acc_e/Acc_s/H_o 指標：

| 模型 | Acc_e ↓ | Acc_s ↑ | H_o ↑ |
|---|---:|---:|---:|
| cat→dog | 0.00 | 100.00 | **100.00** |
| deer→horse | 0.00 | 100.00 | 100.00 |
| dog→cat | 2.50 | 100.00 | 98.73 |
| horse→deer | 1.50 | 100.00 | 99.24 |
| frog→bird | 7.00 | 100.00 | 96.37 |
| truck→ship | 11.50 | 100.00 | 93.90 |
| ship→airplane | 13.00 | 99.89 | 93.00 |
| bird→cat | 13.50 | 100.00 | 92.76 |
| airplane→sky | 17.50 | 100.00 | 90.41 |
| automobile→truck | 65.00 | 100.00 | 51.85（異常值） |
| **平均** | **13.15** | **99.99** | **91.63** |
| 論文 Table 1「Ours」平均 | 6.89 | 98.68 | 97.01 |

方向一致（低 Acc_e、高 Acc_s），但平均 H_o 比論文略低，主要被 `automobile→truck` 這個異常值拖累（erasure 幾乎沒生效）；其餘 9 個模型（包含 cat↔dog 這組）都表現優異，多個甚至達到 Acc_e=0%。推測原因：這次沿用的是 repo `trainscripts/object.sh` 內建的超參數，跟論文 Appendix C.1 報告的 λe=1000/λ0=50/λr=1 不同（repo 用 erase_scale=2000/preserve_global_scale=10/lamb=10），**固定同一組超參數對不同 erase-anchor pair 的效果差異很大**，這是論文沒有明說、需要 per-pair 調參的一個實務細節。

原始資料：`cifar10_table1_results.csv`。
- **Nudity**：NudeNet 偵測 142 張圖仍有 16 張（11.3%）判定含裸露內容，抹除不完全。
- **Celeb leakage**（`attack_prompts.py` 9 種攻擊 × 全部名人）：

  | 規模 | leakage>0 比例 | 平均 leakage | 最嚴重單一案例 |
  |---|---:|---:|---|
  | celeb_10 | 36% (32/90) | -1.62 | Anjelica Huston / boundary_hybrid (+8.75) |
  | celeb_50 | 30% (135/450) | -2.47 | Amy Adams / weight_target_up_anchor_down (+18.4) |
  | celeb_100 | 34% (305/900) | -2.15 | Nicole Kidman / direct_target (+15.4) |

  規模變大後，最嚴重的個別洩漏案例反而更嚴重（celeb_10 最高 8.75，celeb_50/100 達 15-18），代表同時抹除大量名人時會有少數特別頑固的個體。

## 還沒做的部分（本次刻意跳過，皆已與使用者確認）
- **`eval_celeb.py`（GCD 名人辨識器）**：需要獨立 Python 3.6 環境、手動從 OneDrive 下載模型權重、patch numpy bug，無法在目前 venv 自動化，改用 `attack_prompts.py` 的 CLIP-leakage 方法取代。
- **FLUX 實驗（`flux_demo.sh`）**：`black-forest-labs/FLUX.1-dev` 是 HuggingFace gated model，需要使用者自行網頁授權＋提供 token，本次未做。
- ~~COCO preservation 檢查未跑滿全部規模~~ **已於 2026-09-08 補跑完成**：6 個訓練後模型皆已補到完整 3000 張／model，數字已更新到上方表格（見「方法論發現」，小樣本 FID 偏誤達 3.4x-5.6x）。
- 所有生成圖片（約 6.3GB）**未 commit** 進 git，僅保留在本地工作目錄（`eval_cifar_airplane/`、`eval_final_Van Gogh/`、`eval_nudity/`、`celeb_celeb_*/`、`coco_eval/` 等），此分支只保留 CSV 數據與本記錄檔。

## 本分支包含的檔案
- `EXPERIMENT_LOG.md`（本檔案）
- `pairs.csv` / `pairs_50.csv` / `pairs_100.csv`：celeb leakage 攻擊用的 target/anchor 對照表
- `celeb10_attack_out/results.csv` / `celeb50_attack_out/results.csv` / `celeb100_attack_out/results.csv`：完整 leakage 掃描原始數據
- `data/coco_30k_val_partial.csv`：COCO preservation 部分抽樣用的 prompt 子集
- `data/nudity_truncated.csv`：CLIP-token 截斷過的 nudity prompt（供 `eval_clip_score.py` 使用）
