# OCE 固定置換模式的可偵測性（Detectability）實驗

## 動機

OCE 對每個被抹除的概念套用的是**固定、確定性的正交旋轉**——同一個概念每次都被導向同一個 anchor 方向。這帶出一個論文沒有直接測過的問題：**這種固定模式會不會在輸出圖片裡留下可被偵測的統計特徵？** 這跟論文/`attack_prompts.py` 測的「能不能繞過抹除、把內容找回來」（bypass robustness）是不同的角度——這裡問的是「即使拿不回內容，能不能看出這裡被動過手腳」（detectability / fingerprint）。

具體場景：CIFAR-10 物件抹除裡，`dog` 和 `cat` 互為 anchor（論文 Table 7、Figure 5）。用 dog→cat 抹除模型，輸入 prompt `"a photo of dog"`，實際生成的內容是貓。這些「本來要是狗、結果變成貓」的圖，跟「本來就正常生成的貓」，能不能被一個簡單的分類器分開？

## 資料

| 組別 | 說明 | 來源 | 張數 |
|---|---|---|---|
| **置換組**（label=1） | prompt "a photo of dog"，用 dog→cat 抹除模型生成，內容實際是貓 | `eval_cifar_full/dog/dog/` | 200 |
| **原本組 A**（label=0） | prompt "a photo of cat"，用**跟貓/狗無關**的 airplane→sky 模型生成 | `eval_cifar_full/airplane/cat/` | 200 |
| **原本組 B**（label=0） | prompt "a photo of cat"，用**完全未編輯**的原始 `CompVis/stable-diffusion-v1-4` 生成 | `eval_cifar_full/baseline_unedited/cat/` | 200 |

用兩組不同的「原本」對照（A、B）分別重跑一次，檢查「選哪個模型當對照組」會不會影響結論。B 組是更嚴謹的對照，因為除了「有沒有套用抹除」之外，其他條件（同一個底模、同解析度 512×512、同 50 步 denoising）完全一致，能最乾淨地單獨測出「抹除」這個變因的影響；A 組雖然模型本身被編輯過，但編輯的是完全無關的概念（airplane→sky），不影響 cat 的生成能力。

## 方法（四步驟）

1. **特徵抽取**：每張圖丟進 CLIP 圖像編碼器（`openai/clip-vit-base-patch32`），輸出 512 維向量，再做 L2 normalize（除以向量長度，變成單位向量，符合 CLIP embedding 慣例）。**不是**直接用原始像素訓練。
2. **標記標籤**：置換組標 1，原本組標 0。
3. **切分訓練/測試集**：80/20，**stratified**（分層抽樣，確保訓練集、測試集裡兩類比例都跟原始一致）——320 張訓練、80 張測試。
4. **訓練分類器**：在 512 維 embedding 上訓練簡單的 **Logistic Regression**（線性分類器），在測試集上驗證。

**為什麼用簡單的線性分類器，不是深度模型**：樣本數只有 400 張，訓練深度網路會嚴重過擬合。用線性探針（linear probe）測試的是「CLIP 這個沒針對此任務訓練過的通用視覺模型，其 embedding 空間裡本來就存不存在能區分兩類的方向」——如果連線性分類器都能分得很開，代表訊號非常強、幾乎是「肉眼看不出來但向量空間裡差很多」的情況；不需要更複雜的非線性模型來「硬找」出差異。

## 結果

| 對照組 | Accuracy | Precision | Recall | 測試集混淆矩陣（真\預測：genuine/swapped） |
|---|---:|---:|---:|---|
| A：airplane 模型生成的貓 | 97.50% | 100.00% | 95.00% | genuine 40/0，swapped 2/38 |
| B：完全未編輯 SD1.4 生成的貓 | 96.25% | 100.00% | 92.50% | genuine 40/0，swapped 3/37 |

兩組對照結果非常接近（97.50% vs 96.25%），**確認結論不會因為選哪個模型當「原本」對照而改變**。兩次實驗中，被分類器誤判的置換圖片高度重疊（`a photo of dog_51.png`、`a photo of dog_184.png` 在兩次都被誤判；B 組還多了 `a photo of dog_50.png`），這些都是特寫角度、看起來就是普通貓臉的圖片，肉眼完全看不出破綻。

原始資料：
- `erasure_detectability_results.csv`：兩組對照的完整數字
- `erasure_detectability_embeddings_airplane.npz` / `erasure_detectability_embeddings_baseline.npz`：對應的 CLIP embedding 原始資料（`X`: 400×512 向量、`y`: 標籤、`files`/`groups`: 對應檔名與組別）

## 結論與限制

- **結論**：OCE 的固定置換模式確實會在 CLIP embedding 空間裡留下一致、可被簡單線性分類器偵測的統計特徵，準確率達 96-97.5%，且不受「原本組」選擇影響。即使單張圖片肉眼完全看不出被動過手腳，批量統計上仍可被偵測出來。
- **限制**：
  - 樣本僅限 CIFAR-10 的 `dog↔cat` 這一組 anchor pair，尚未驗證是否適用於其他概念配對（例如 celeb→person 這種語意差異更大的配對，是否更容易或更難被偵測）。
  - 分類器是在「已知兩組各自完整資料」的情境下訓練出來的（有監督學習），實際場景中偵測方若不知道哪些圖片是置換過的，仍需要其他方式取得訓練標籤，這裡驗證的是「特徵存不存在」而非「實際可操作的偵測流程」。
  - 只測了 Logistic Regression 這一種線性探針，沒有比較其他分類器（如果想確認是否有更強的非線性可分性，可以再測 SVM/MLP）。
