
---

# Homework 5: 胸部 X 光影像之多病灶偵測與標註

本作業目的為對胸部 X 光影像（Chest X-Ray）進行多病灶偵測與標註。對於每張影像中可能出現的一種或多種異常（例如：主動脈硬化、心臟肥大、肺尖肋膜增厚等），透過深度學習物件偵測模型 (如 Faster R-CNN, Mask R-CNN, RetinaNet, YOLO 等) 自動產生邊界框與疾病類別。此作業也考慮如何進行影像的強度對數轉換與對比度（色彩平衡）處理，以符合醫學影像特性與優化可視化效果。

---

## 1. 資料與作業說明

1. **資料來源**  
   - 來自義大醫院 (E-Da Hospital) 的胸部 X 光影像 (`DICOM` 格式)。  
   - 各影像有對應的**標註** (bounding boxes) 與**疾病類別**。  
   - *train.csv*：包含訓練集中所有影像檔名及其對應資訊（病灶名稱、邊界框座標等）。  
   - *test.csv*：包含測試集的影像檔名，用於最終預測（不含真實標註）。

2. **資料夾結構**  
   - `train/` (8 個子資料夾，以疾病類別區分)  
     - `aortic_atherosclerosis_calcification/`  
       - `image/`：DICOM 影像  
       - `mark/`：標註邊界框 (jpg 或其他格式)  
     - 其餘 7 個疾病子資料夾 (結構相同)。  
   - `test/`  
     - `image/`：測試影像  
   - 其餘：`README.pdf`, `train.csv`, `test.csv`。

3. **作業要求**  
   1. **影像正規化 (Normalization)**  
      - 依照題目所給的公式，對每個 DICOM 影像執行**強度對數轉換**與**對比度調整 (simplest color balance)**。  
      - 其中，`vmin = 0`, `vmax = 2.5`。  
   2. **影像調整 (可選)**  
      - 是否進行縮放 (例如 512×512) 以加速模型訓練，依個人判斷。  
   3. **深度學習模型**  
      - 使用任一物件偵測模型 (Faster R-CNN, Mask R-CNN, RetinaNet, YOLO, ...) 搭配遷移學習 (transfer learning) 進行多病灶邊界框預測。  
   4. **輸出結果**  
      - 載入測試影像，輸出每張影像中偵測到的邊界框與其疾病類別。

---

## 2. 前處理流程

### 2.1 讀取 DICOM 影像

- 使用 `pydicom` 或其他可處理 DICOM 的函式庫讀取原始影像。
- 取得以下標籤 (若檔頭提供)：  
  - `Window Center` (WC)  
  - `Window Width` (WW)  
  - `BitsStored`  
  - 判斷 `Photometric Interpretation` 與 `Pixel Intensity Relationship`。  
  - 本次作業提示：`MONOCHROME1` + `LOG` → 顯示需做對數逆轉換。

### 2.2 強度對數轉換

題目提供的程式邏輯：

1. 定義 `x[i]` 為第 i 個像素值，`N` 為總像素數。  
2. `imax = WC + (WW / 2)`  
3. `imin = WC - (WW / 2)`  
4. 如果 `x[i] < imin` → 設為 `imin`；若 `x[i] > imax` → 設為 `imax`。  
5. 經過裁切後：  
   \[
   z[i] = -\log\left(1 + \frac{x[i]}{2^{BitsStored}}\right)
   \]
6. `z[i]` 為對數轉換後的影像像素。

### 2.3 對比度調整 (simplest color balance)

1. 定義 `vmin = 0`、`vmax = 2.5`。  
2. 執行：
   \[
   c[i] = \frac{z[i] - v_{\min}}{\,v_{\max} - v_{\min}\,}
   \]
   並將 `c[i]` 限制在 [0, 1]。
3. 最終得到歸一化的灰階值 `c[i]`。

### 2.4 (可選) 影像尺寸重整

- 依照個人資源及需求，可將影像縮放至 (512×512) 或 (256×256) 等。
- 需注意縮放後邊界框座標也要對應縮放。

---

## 3. 模型與訓練策略

1. **模型選擇**  
   - 以 **Faster R-CNN** 為例：  
     - Backbone：ResNet50 / ResNet101 / MobileNet 等  
     - 使用 COCO / ImageNet 預訓練權重  
   - 其他可行：**Mask R-CNN**, **YOLOv5 / YOLOv8**, **RetinaNet**, 等。

2. **訓練資料準備**  
   - 從 `train.csv` 中讀取每筆影像資訊與對應標籤 (邊界框 x, y, w, h 與疾病類別)。  
   - DICOM → 對數轉換 → 對比度調整 → (可選) 縮放 → Tensor  
   - 將標籤轉為對應格式 (e.g., PyTorch `Target` dict for Faster R-CNN)。

3. **超參數**  
   - `batch size`：依記憶體調整，一般 2～8。  
   - `epoch`：10～30 (或更多，視收斂情況)。  
   - `learning rate`：1e-3 ~ 1e-4 (依經驗或預訓練策略)。  
   - `optimizer`：SGD / Adam。  
   - `loss`：物件偵測框架自帶 (分類 + 回歸 loss)。  

4. **資料增強** (Data Augmentation)  
   - 旋轉、平移、縮放、翻轉等。  
   - 需同時作用在影像與標籤 (bounding boxes)。

5. **訓練與驗證**  
   - 可在本地或雲端 (e.g., Kaggle) 執行，並監控 loss、mAP (mean Average Precision) 或 AP50 / AP75 等指標。

---

## 4. 測試與上線

1. **測試階段**  
   - 從 `test.csv` 中獲取測試影像清單，按同樣的前處理 (對數轉換+色彩平衡+可能的縮放)。  
   - 將每張影像餵給訓練好的偵測模型，輸出所有預測框 (x, y, w, h) 及疾病類別 + 置信度分數。  
2. **結果檔**  
   - 視作業要求，可能需輸出 CSV 或 JSON，包含每張測試影像的檢測結果 (多個疾病 bounding boxes)。

---

## 5. 結果與討論

1. **指標 / 成效**  
   - mAP (mean Average Precision)、AP per class、Recall、Precision 等。  
   - 如果實作評測程式，可計算每個類別的 AP 與平均。  
2. **視覺化**  
   - 疊加 bounding boxes 與疾病標籤至測試影像。  
   - 可使用 CAM/Grad-CAM 等解釋模型的關注區域。  
3. **困難點**  
   - 讀取與處理 DICOM 格式 + 日誌中所說的對數轉換與對比度平衡。  
   - 多病灶 (一張圖可能對應多個 disease)。  
   - 需確保縮放或增強後 bounding boxes 的正確對應關係。

---

## 6. 未來發展

- 針對更多疾病類別或更複雜病徵 (e.g., 病灶分割)。  
- 與其他醫學影像 (CT, MRI) 多模態結合，提升診斷準確度。  
- 模型架構升級：嘗試 **YOLOv8**, **Detr**, **Sparse R-CNN**, **HybridNets** 等方法。  
- 整合醫學輔助解釋介面 (Grad-CAM, Saliency Map) 提供臨床參考。  

---