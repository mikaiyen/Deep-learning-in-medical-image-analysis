
---

# Deep Learning in Medical Image Analysis – Five Homework Summaries

本系列作業聚焦於不同醫學影像的深度學習應用，從圖像分類到定位和分割，涵蓋多種模型架構與訓練策略。以下為各作業的簡要介紹：

---

## 課程資訊

- **課程名稱**: Deep Learning in Medical Image Analysis
- **授課單位**: 國立陽明交通大學 統計碩
- **授課教師**: 黃冠華
- **學期**: 2024 學年度 第 1 學期

---

## Homework 1: USPS 手寫數字辨識

**目標**  
- 分類美國郵政服務 (U.S. Postal Service) 的手寫數字資料集 (0～9)。  
- 資料為 16×16 的灰階圖，經預處理後利用多層感知器 (MLP) 進行手寫數字辨識。

**重點**  
- **資料**: `zip.train.csv`, `zip.test.csv`  
- **模型**: 三層全連接 (FC) + Batch Normalization + Dropout + ReLU Activation  
- **結果**: 訓練模型後對測試集預測，並計算分類正確率。  
- **難點**: 調整隱藏層神經元數量、活化函式、優化器等，達到穩定的訓練收斂。  

---

## Homework 2: SPECT 影像之多分類預測 (PD 病程分期)

**目標**  
- 利用單光子放射斷層掃描 (SPECT) 影像，預測帕金森氏症 (PD) 三種病程分期 (Stage 1/2/3)。  
- 結合病患的年齡與性別等輔助變數。

**重點**  
- **資料前處理**:  
  1. 選取合適的 2D slice (或使用提供的 slice index)。  
  2. 將 50×50 的區域裁切保留腦部結構。  
  3. 將灰階影像延伸為三通道以符合預訓練模型 (VGG, ResNet, ViT)。  
- **模型**:  
  - 利用預訓練 CNN/Transformer (Transfer Learning)。  
  - 在全連接層前把年齡、性別等非影像特徵合併進來。  
- **結果**: 比較不同預訓練模型 (VGG、ResNet、ViT) 在多分類 (3 類) 的準確度。

---

## Homework 3: PD 六階段預測 + MRI 腦部腫瘤判別

**Part 1**:  
- **六階段 PD 預測**: 從原本合併的三階段擴展成六階段 (0～5)；需處理資料不平衡 (e.g., OverSampling)。  
- **模型**: 基於 ResNet50 等預訓練模型，最後輸出 6 類。  

**Part 2**:  
- **MRI 腦部影像**: 120 份 3D MRI (60 正常，60 腫瘤)，要判定是否存在腦部腫瘤 (binary classification)。  
- **方法**:  
  1. **Single Slice**: 只用單張關鍵切片做 2D 分類。  
  2. **Late Fusion**: 多張切片分別提取特徵再融合後分類。  
  3. **Early Fusion**: 在輸入端把多張切片組合成多通道，再用 2D CNN 處理。  
  4. **3D CNN**: 直接以 3D 卷積學習完整立體資訊。  

---

## Homework 4: 頸動脈超音波影像分割 (FCN 與 U-Net)

**目標**  
- 針對頸動脈超音波影像 (約 300 張訓練、100 張測試)，進行頸動脈區域的語意分割。  
- **模型**:  
  - **FCN-8s** (Fully Convolutional Network)  
  - **U-Net**  

**重點**  
- **資料**: 已標註頸動脈二元 mask。  
- **訓練流程**:  
  1. 下採樣 (encoder) → 上採樣 (decoder)，或 FCN 跳躍連接。  
  2. 計算語意分割損失 (CrossEntropy, Dice Loss 等)。  
- **成果**: 產生像素級分割影像，預估頸動脈範圍。  

---

## Homework 5: 胸部 X 光影像多病灶檢測

**目標**  
- 偵測胸部 X 光影像中的多個可能病灶 (bounding boxes + disease label)。  
- 病灶包含：主動脈硬鈣化、主動脈彎曲、肺野浸潤增加、心臟肥大、脊椎側彎等。

**重點**  
1. **DICOM 影像前處理**:  
   - 對數轉換 (根據 `Pixel Intensity Relationship` 與 `MONOCHROME1`)。  
   - 色彩平衡 (simplest color balance)。  
2. **物件偵測模型**:  
   - 可使用 Faster R-CNN, YOLO, RetinaNet, Mask R-CNN…等主流架構。  
   - 載入預訓練權重進行 Transfer Learning。  
3. **輸出**:  
   - 針對測試影像產生預測框與疾病類別標籤。  

---

## 附註
由於醫學影像需保密，故無法提供原始訓練資料，如果想了解請詢問教授