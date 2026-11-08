# 🚗 Vehicle Body Type Classifier (Araç Gövde Tipi Sınıflandırma)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-ff4b4b?logo=streamlit)
![EfficientNet](https://img.shields.io/badge/Backbone-EfficientNet--B4-00A67E)
![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle)
![License](https://img.shields.io/badge/License-MIT-green)

Bu proje, derin öğrenme ve bilgisayarlı görü (computer vision) teknikleri kullanarak **8 farklı araç gövde tipini** yüksek doğrulukla tespit eden uçtan uca bir sınıflandırma sistemidir. Model, transfer learning ve ince ayar (fine-tuning) stratejileriyle eğitilmiş **EfficientNet-B4** mimarisine dayanmakta ve kullanıcı dostu bir **Streamlit** web arayüzü ile sunulmaktadır.

---

## 🎯 Model Başarı Metrikleri

Model, doğrulama veri seti üzerinde aşağıdaki başarı oranlarına ulaşmıştır:

- **Macro F1-Score:** `%96.83`
- **Genel Doğruluk (Accuracy):** `%96.70`
- **Model Boyutu:** `~68 MB` (Hafif ve hızlı çıkarım)
- **Ortalama Çıkarım Süresi (Inference Latency):** `~25-40 ms / görsel`

---

## 🏷️ Sınıflandırma Kategorileri

Model aşağıdaki 8 araç sınıfını ayırt edebilmektedir:

| Script ID | Sınıf Adı | Açıklama |
| :---: | :--- | :--- |
| **1** | **SUV** | Spor amaçlı arazi/şehir aracı |
| **2** | **VAN** | Kapalı kasa ticari / minibüs tipi araçlar |
| **3** | **STATION WAGON** | Uzatılmış tavan çizgili binek araç |
| **4** | **MICRO** | Smart vb. ultra kompakt mini şehir otomobilleri |
| **5** | **F1 ARAÇLARI** | Açık tekerlekli yarış araçları (Formula / Open-Wheel) |
| **6** | **SEDAN** | Standart 3 kutulu (motor-kabin-bagaj) binek araç |
| **7** | **HATCHBACK** | Dik bagaj kapaklı kompakt binek araç |
| **8** | **PICK UP** | Açık kasalı yük taşıma aracı |

---

## 📊 Veri Seti (Dataset)

Eğitim ve test aşamasında kullanılan dengelenmiş veri seti Kaggle üzerinde yayınlanmıştır:

🔗 **Kaggle Veri Seti:** [Vehicle Body Types Dataset](https://www.kaggle.com/datasets/mmmdpoladl/vehicle-body-types)

> **Veri Seti Özellikleri:**
> - 8 farklı sınıftan toplanmış, filtrelenmiş ve dengelenmiş araç görselleri.
> - Aşırı öğrenmeyi (overfitting) engellemek amacıyla zengin veri artırma (data augmentation) teknikleri uygulanmıştır.
> - Sınıflar arası örnek dengesizliğini çözmek için `WeightedRandomSampler` entegre edilmiştir.

---

## 🧠 Model Mimarisi ve Eğitim Stratejisi

### 1. Mimari Detayları
- **Omurga (Backbone):** ImageNet ön-eğitimli `EfficientNet-B4`
- **Giriş Çözünürlüğü:** 380 × 380 piksel
- **Sınıflandırıcı Kafa (Head):** `Dropout(p=0.4)` + `Linear(in_features, 8)`

### 2. İki Fazlı Eğitim (Two-Phase Transfer Learning)
1. **Faz 1 (Feature Extraction):**
   - EfficientNet omurga katmanları dondurulmuştur (`requires_grad=False`).
   - Sadece son sınıflandırma katmanı eğitilmiştir.
   - Optimizer: `Adam (lr=1e-3, weight_decay=1e-4)`
2. **Faz 2 (Fine-Tuning):**
   - Tüm model katmanlarının kilidi açılmıştır.
   - Düşük öğrenme oranı ve öğrenme oranı planlayıcı (scheduler) ile ince ayar yapılmıştır.
   - Optimizer: `AdamW (lr=1e-4)` + `CosineAnnealingWarmRestarts`
   - Loss: `CrossEntropyLoss` (Etiket yumuşatma: `label_smoothing=0.1`, zor sınıflar için özel sınıf ağırlıkları: `HATCHBACK`, `SUV`, `STATION_WAGON`, `SEDAN` = 2.0)

### 3. Veri Artırma (Albumentations)
- `Resize(400, 400)` & `RandomCrop(380, 380)`
- `HorizontalFlip(p=0.5)`
- `RandomBrightnessContrast(p=0.4)`
- `ShiftScaleRotate(p=0.5)`
- `CLAHE` (Histogram Eşitleme) & `GaussNoise`
- `ColorJitter`, `MotionBlur`, `Perspective` ve `CoarseDropout`

---

## 📈 Performans Grafikleri ve Hata Matrisi

### 1. Eğitim ve Doğrulama Grafikleri
Modelin eğitim ve doğrulama süreçlerindeki kayıp (loss) ve doğruluk (accuracy) eğrileri:

| Kayıp Grafiği (Loss) | Doğruluk Grafiği (Accuracy) |
| :---: | :---: |
| ![Loss Plot](loss_plot.png) | ![Accuracy Plot](acc_plot.png) |

### 2. Karmaşıklık Matrisi (Confusion Matrix)
8 araç sınıfı arasındaki normalize edilmiş tahmin doğruluğu:

<p align="center">
  <img src="confusion_matrix.png" alt="Normalized Confusion Matrix" width="600"/>
</p>

---

## 🖥️ Streamlit Web Uygulaması Özellikleri

Uygulama 4 ana modülden oluşmaktadır:

1. **Canlı Tahmin:** Kullanıcı tekli veya çoklu araç görseli yükleyebilir; tespit edilen sınıf, güven skoru (confidence score), çıkarım süresi ve tüm sınıfların olasılık dağılım grafiği anlık gösterilir.
2. **Model Özellikleri:** Model mimarisi, parametre sayıları ve kaydedilen performans grafikleri incelenebilir.
3. **Toplu Test (Batch Inference):** Bir klasördeki yüzlerce görseli otomatik olarak tarar, her birini modelden geçirir ve proje isterlerine uygun formatta `preds.txt` çıktı dosyası üretir.
4. **Performans Analizi (Test.txt Entegrasyonu):** Üretilen `preds.txt` ile cevap anahtarı niteliğindeki `True.txt` dosyasını karşılaştırarak dinamik Confusion Matrix ve Classification Report görselleştirmesi yapar.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/memmedpoladli/car-body-type-classifier.git
cd car-body-type-classifier
```

### 2. Sanal Ortam Oluşturun ve Bağımlılıkları Yükleyin
```bash
python3 -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Modeli İndirin
Eğitilmiş model dosyasını (`best_model_v6.pth`) proje ana dizinine yerleştirin.

### 4. Web Arayüzünü Başlatın
```bash
streamlit run app.py
```
Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

---

## ☁️ Google Colab ile Test Çalıştırma

Modeli Colab ortamında hızlıca toplu test etmek için:
1. `PredictionScript.txt` içeriğini Colab hücresine yapıştırın.
2. `best_model_v6.pth` dosyasını çalışma dizinine yükleyin.
3. Şu fonksiyonu çağırın:
   ```python
   Predict("/content/testdata")
   ```
4. Tahminler `/content/preds.txt` dosyasına standart formatta (`<dosya_adı> | Pred: <ID>`) yazılacaktır.

---

## 📁 Proje Dizin Yapısı

```
car-body-type-classifier/
├── app.py                     # Streamlit interaktif web uygulaması
├── model.py                   # EfficientNet-B4 model mimarisi
├── dataset.py                 # PyTorch Dataset, Albumentations & Sampler
├── train.py                   # 2-Fazlı model eğitim scripti
├── classes.json               # Sınıf isim listesi
├── requirements.txt           # Python kütüphane bağımlılıkları
├── PredictionScript.txt       # Google Colab uyumlu tahmin scripti
├── README.md                  # Proje dokümantasyonu
├── .gitignore                 # Git hariç tutma kuralları
├── acc_plot.png               # Doğruluk (Accuracy) grafiği
├── loss_plot.png              # Kayıp (Loss) grafiği
├── confusion_matrix.png       # 8x8 Normalize Karmaşıklık Matrisi
└── test_icin_ornek_dosyalar/  # Örnek doğrulama dosyaları
    ├── True.txt
    └── Preds.txt
```

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak paylaşılmıştır.
