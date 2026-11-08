# Araç Gövde Tipi Sınıflandırma

EfficientNet-B4 tabanlı derin öğrenme modeli ile **8 farklı araç gövde tipini** sınıflandıran bir Streamlit web uygulaması.

## Sınıflar

| Script ID | Sınıf |
|-----------|-------|
| 1 | SUV |
| 2 | VAN |
| 3 | STATION WAGON |
| 4 | MICRO |
| 5 | F1 ARAÇLARI (Açık Tekerlekli) |
| 6 | SEDAN |
| 7 | HATCHBACK |
| 8 | PICK UP |

## Model Performansı

- **Macro F1-Score:** %96.83  
- **Doğruluk (Accuracy):** %96.70

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Web Arayüzü

```bash
streamlit run app.py
```

### Google Colab ile Tahmin (PredictionScript.txt)

`PredictionScript.txt` dosyasını Google Colab'a kopyalayın, model ağırlık dosyasını (`best_model_v6.pth`) yükleyin ve çalıştırın:

```python
Predict("/content/testdata")
```

Tahminler `/content/preds.txt` dosyasına yazılır.

## Proje Yapısı

```
.
├── app.py                  # Streamlit web arayüzü
├── model.py                # EfficientNet-B4 model mimarisi
├── dataset.py              # Dataset ve DataLoader tanımları
├── train.py                # Model eğitim scripti
├── classes.json            # Sınıf isim listesi
├── requirements.txt        # Bağımlılıklar
├── PredictionScript.txt    # Google Colab tahmin scripti
├── test_icin_ornek_dosyalar/  # Örnek test dosyaları
│   ├── True.txt
│   └── Preds.txt
├── acc_plot.png            # Accuracy grafiği
├── loss_plot.png           # Loss grafiği
└── confusion_matrix.png    # Confusion Matrix
```

> **Not:** Model ağırlık dosyası (`best_model_v6.pth`) ve veri seti (`dataset_balanced_v6/`) boyut nedeniyle repoya dahil edilmemiştir.

## Mimari

- **Temel Model:** EfficientNet-B4 (ImageNet ön-eğitimli)
- **Giriş Çözünürlüğü:** 380×380 piksel
- **Dropout:** 0.4
- **Eğitim Stratejisi:** 2 Fazlı (Feature Extraction → Fine-Tuning)
- **Loss Fonksiyonu:** CrossEntropyLoss (Label Smoothing + Sınıf Ağırlıkları)
- **Scheduler:** CosineAnnealingWarmRestarts

