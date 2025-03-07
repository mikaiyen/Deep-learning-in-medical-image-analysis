
---

# Homework 3: 深度學習於多階段 PD 與 3D MRI 影像分析

## 目錄
1. [作業概述](#作業概述)  
2. [Part 1: PD 六階段預測 (ResNet50)](#part-1-pd-六階段預測-resnet50)  
   2.1. [資料說明](#資料說明)  
   2.2. [問題與解法](#問題與解法)  
   2.3. [模型與訓練設定](#模型與訓練設定)  
   2.4. [使用步驟](#使用步驟)  
   2.5. [結果與討論](#結果與討論)  
3. [Part 2: MRI 腦部腫瘤判別](#part-2-mri-腦部腫瘤判別)  
   3.1. [資料說明](#資料說明-1)  
   3.2. [分析方法](#分析方法)  
   3.3. [模型結構](#模型結構)  
   3.4. [使用步驟](#使用步驟-1)  
   3.5. [結果與討論](#結果與討論-1)  
4. [心得與未來改進](#心得與未來改進)  

---

## 作業概述
本作業分為兩個部分：
1. **Part 1**：根據更新後的資料 (六階段 PD 標籤) 重新訓練 ResNet50 模型，並解決資料不平衡等問題後，輸出預測結果與機率分佈。  
2. **Part 2**：針對 120 份 3D MRI 影像進行「正常 / 腦部腫瘤」的二元分類，需實作四種策略 (Single Slice、Late Fusion、Early Fusion、3D CNN)。

---

## Part 1: PD 六階段預測 (ResNet50)

### 資料說明
- **train_hwk02_new.csv**：提供原始六階段 (0=normal, 1=PD stage1, 2=PD stage2, 3=PD stage3, 4=PD stage4, 5=PD stage5) 的標籤欄位 `Stage_New`。  
- **test.csv**：用於最終預測 (不含真實標籤)。  
- **影像處理**：同前作業 (SPECT 影像)，挑選適當切片並裁切後，輸入到 ResNet50 進行分類。

### 問題與解法
1. **資料不平衡**  
   - 由於 6 個階段的資料分佈不均，可能造成分類結果偏差。  
   - **解決方案**：使用 **RandomOverSampler** (或其他方法) 擴增少量階段的資料，以降低不平衡影響。

2. **資料量不足**  
   - 本身資料量相對有限，容易過擬合。  
   - **解決方案**：可配合資料增強 (Data Augmentation)，如旋轉、平移、隨機裁切等。

### 模型與訓練設定
1. **模型架構**  
   - 以 **ResNet50** 作為基底模型，並載入 **ImageNet** 預訓練權重。  
   - 修改最後全連接層 (Fully-Connected Layer)，輸出 6 個類別 (對應六階段)。  

2. **超參數**  
   - **Learning Rate**: 1e-4 或依實驗調整  
   - **Batch Size**: 16 或 32 (視 GPU 記憶體而定)  
   - **Epochs**: 10~30 (依模型收斂情況)  
   - **Optimizer**: 可使用 Adam 或 SGD  
   - **Loss Function**: CrossEntropyLoss  
   - **參數量**: 約 24,035,142 (最終模型報告中紀錄)

3. **再現性 (Reproductivity)**  
   - 亂數種子 `torch.manual_seed(...)` 固定可部分穩定結果；但 GPU 訓練仍可能存在微小浮點誤差差異。

### 使用步驟
1. **安裝套件**  
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy pandas scikit-learn imbalanced-learn matplotlib
   ```
2. **資料前處理**  
   - 讀取 `train_hwk02_new.csv`，以 `RandomOverSampler` 等進行樣本平衡。  
   - 切割訓練 / 驗證資料集 (如 80% / 20%)。  
   - 進行影像增強或縮放。  
3. **訓練模型**  
   - 在 Notebook 中逐段執行
     
4. **測試與預測**  
   - 讀取 `test.csv`，將預測結果輸出到 `ResNet50_6c.csv`。  
   - 檔案格式可包含各階段的機率值以及預測的整數標籤。

### 結果與討論
1. **平衡前後效能比較**  
   - 使用 OverSampling 後，小樣本類別的預測表現提升。  
2. **最終參數量**  
   - 約 **24,035,142**。  
3. **訓練難度**  
   - 與前作業類似，但需額外處理資料不平衡問題。  
   - 大多時間消耗在嘗試不同平衡策略與資料增強手段。

---

## Part 2: MRI 腦部腫瘤判別

### 資料說明
- **資料集**：共 120 份 3D MRI 圖像 (60 份正常、60 份有腫瘤)。  
- **目標**：二元分類 (正常 / 腫瘤)。  
- **不平衡性**：此部分資料較均勻 (60 vs. 60)，故不需特別做 OverSampling。

### 分析方法
針對 3D MRI 的特性，實作以下四種方法：
1. **Single Slice**  
   - 僅取單一關鍵切片 (如腫瘤最明顯的部分)，用預訓練模型 (VGG16 / ResNet50 / etc.) 進行 2D 分析。  

2. **Late Fusion**  
   - 各切片分別經預訓練模型提取特徵後，再將所有切片的特徵透過平均或拼接 (concatenate) 的方式合併到全連接層進行分類。  

3. **Early Fusion**  
   - 將所有切片在輸入端就整合，如將多張切片堆成多通道 (或 3D tensor) 輸入修改後的 2D CNN，從而在網路初期就學習切片之間的關聯性。  

4. **3D CNN**  
   - 以 3D 卷積操作 (3D Convolution) 處理完整 MRI 體積，能同時捕捉空間 (x, y) 和深度 (z) 資訊。  

### 模型結構
1. **Single Slice**  
   - **VGG16** (或其他預訓練 CNN) → **FC** → Output (2 類)。  
   - 輸入：1 張 2D 切片 (轉成 3-channel 或灰階擴充到 RGB)。  
2. **Late Fusion**  
   - 每張切片 → **VGG16** → 特徵向量 → 所有切片特徵 **平均或拼接** → **FC** → Output。  
3. **Early Fusion**  
   - 先將所有切片組合成多通道 (e.g., channel = slice數)，然後輸入修改後的 **2D CNN**。  
   - **FC** → Output。  
4. **3D CNN**  
   - **3D Convolution** + 3D Pooling → Flatten → FC → Output。  

### 使用步驟
1. **環境準備**  
   ```bash
   pip install torch torchvision torchaudio
   pip install numpy pandas nibabel # 或 pydicom，依 MRI 格式而定
   pip install matplotlib
   ```
2. **資料前處理**  
   - 讀取並整合 120 份 3D MRI。  
   - 若需要，將資料切片 (slicing) → 2D 影像。  
3. **執行程式**  
   - 在 Notebook 中逐段執行  
4. **輸出結果**  
   - 每個策略各自輸出分類結果或測試集預測 (正常 / 腫瘤)。  
   - 報告準確率、Precision、Recall 或 F1-score 等指標。

### 結果與討論
1. **最終參數量**  
   - 報告中提及約 **60,855,106** (可能針對最後選定的模型)。  
2. **訓練難度**  
   - 3D CNN 模型因參數量大，單個 epoch 需較久時間；若硬體資源有限，需充分利用雲端 (如 Kaggle GPU) 做訓練。  
   - 在 Single Slice / Late Fusion / Early Fusion 之間，預處理和融合策略的實作細節差異較大。  
3. **再現性**  
   - 受限於 GPU 訓練之隨機性與硬體差異，結果會略有不同，但整體表現維持一致。

---

## 心得與未來改進
- **Part 1 (PD 六階段)**  
  - 透過 OverSampling 和 Data Augmentation 改善少量類別的預測品質。  
  - 可考慮嘗試其他平衡策略 (如 SMOTE) 或更深度的模型架構。  
- **Part 2 (3D MRI 腦部腫瘤)**  
  - 3D CNN 從三維空間同時學習空間深度資訊，效果通常佳，但成本也高。  
  - Early / Late Fusion 提供切片組合的不同思路，也可再探討最適切片數量、融合方法。  
  - 進一步可嘗試 **3D U-Net** 類模型，如需做更精細的分割或定位。

---
