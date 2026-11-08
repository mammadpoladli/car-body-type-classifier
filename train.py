import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import json
import os
import copy

from dataset import get_dataloaders
from model import get_model

BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 10
DROPOUT_RATE = 0.4

PHASE1_EPOCHS = 5
PHASE1_LR = 0.001
PHASE2_LR = 0.0001
LABEL_SMOOTHING = 0.1


def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Kullanılan cihaz: {device}")

    train_dir = "dataset_balanced_v6/train"
    val_dir = "dataset_balanced_v6/val"

    try:
        train_loader, val_loader, classes = get_dataloaders(
            train_dir, val_dir, batch_size=BATCH_SIZE
        )
        print(f"\nSınıflar: {classes}")
        print(f"Eğitim seti: {len(train_loader.dataset)} görsel")
        print(f"Doğrulama seti: {len(val_loader.dataset)} görsel")

        with open('classes.json', 'w') as f:
            json.dump(classes, f)

    except Exception as e:
        print(f"HATA: Veri seti bulunamadı veya okunamadı: {e}")
        return

    model = get_model(num_classes=len(classes), dropout_rate=DROPOUT_RATE).to(device)

    try:
        model.load_state_dict(torch.load("best_model_v6.pth", map_location=device))
        print("\n[TRANSFER LEARNING] yüklendi. V7 eğitimi, V6'nın kaldığı yerden devam edecek.")
    except Exception as e:
        print(f"\n[DİKKAT] V6 modeli yüklenemedi, ImageNet ağırlıkları ile sıfırdan başlanıyor: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EfficientNet-B4")
    print(f"Toplam parametre: {total_params:,}")
    print(f"Eğitilebilir parametre: {trainable_params:,}")

    class_weights = []
    for c in classes:
        if c in ['HATCHBACK', 'SUV', 'STATION_WAGON', 'SEDAN']:
            class_weights.append(2.0)
        else:
            class_weights.append(1.0)
    class_weights = torch.FloatTensor(class_weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    best_f1 = 0.0
    patience_counter = 0
    best_model_wts = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n{'='*70}")
    print(f"FAZ 1: Feature Extraction (Epoch 1-{PHASE1_EPOCHS})")
    print(f"Sadece classifier katmanı eğitilir, feature katmanları dondurulmuş")
    print(f"Learning Rate: {PHASE1_LR}")
    print(f"{'='*70}")

    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=1e-4
    )

    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, PHASE1_EPOCHS, "Faz1"
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch, PHASE1_EPOCHS, "Faz1", classes
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model_v6.pth")
            print(f">>> Yeni en iyi model kaydedildi! (F1-Score: {best_f1:.4f})")
        else:
            patience_counter += 1

    print(f"\n{'='*70}")
    print(f"FAZ 2: Fine-Tuning (Epoch {PHASE1_EPOCHS+1}-{EPOCHS})")
    print(f"Tüm katmanlar eğitiliyor, düşük Learning Rate")
    print(f"Learning Rate: {PHASE2_LR}")
    print(f"{'='*70}")

    for param in model.features.parameters():
        param.requires_grad = True

    trainable_params_phase2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Eğitilebilir parametre (Faz 2): {trainable_params_phase2:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=PHASE2_LR,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    remaining_epochs = EPOCHS - PHASE1_EPOCHS
    patience_counter = 0

    for epoch in range(remaining_epochs):
        actual_epoch = PHASE1_EPOCHS + epoch

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            actual_epoch, EPOCHS, "Faz2"
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device,
            actual_epoch, EPOCHS, "Faz2", classes
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "best_model_v6.pth")
            print(f">>> Yeni en iyi model kaydedildi! (F1-Score: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    İyileşme yok ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                print(f"\n{'='*70}")
                print(f"EarlyStopping: {PATIENCE} epoch boyunca iyileşme olmadı.")
                print(f"En iyi F1-Score: {best_f1:.4f}")
                print(f"{'='*70}")
                break

    total_epochs_trained = len(train_losses)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epochs_trained+1), train_losses,
             label='Training Loss', marker='o', markersize=4, linewidth=2)
    plt.plot(range(1, total_epochs_trained+1), val_losses,
             label='Validation Loss', marker='s', markersize=4, linewidth=2)
    plt.axvline(x=PHASE1_EPOCHS, color='gray', linestyle='--', alpha=0.5, label='Faz 1→2 Geçişi')
    plt.xlabel('Epoch Sayısı')
    plt.ylabel('Loss Değeri')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_plot.png', dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epochs_trained+1), [a*100 for a in train_accs],
             label='Training Accuracy', marker='o', markersize=4, linewidth=2)
    plt.plot(range(1, total_epochs_trained+1), [a*100 for a in val_accs],
             label='Validation Accuracy', marker='s', markersize=4, linewidth=2)
    plt.axvline(x=PHASE1_EPOCHS, color='gray', linestyle='--', alpha=0.5, label='Faz 1→2 Geçişi')
    plt.xlabel('Epoch Sayısı')
    plt.ylabel('Accuracy (%)')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('acc_plot.png', dpi=150)
    plt.close()

    print("\nEn iyi model ile son Confusion Matrix hesaplanıyor...")
    model.load_state_dict(best_model_wts or torch.load("best_model_v6.pth", map_location=device))
    model.eval()

    final_preds = []
    final_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            final_preds.extend(preds.cpu().numpy())
            final_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(final_labels, final_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1)
    plt.ylabel('Gerçek Sınıflar')
    plt.xlabel('Tahmin Edilen Sınıflar')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("EĞİTİM TAMAMLANDI")
    print("=" * 70)
    print(f"En iyi Validation F1-Score (macro): {best_f1:.4f}")
    print(f"Toplam eğitilen epoch sayısı: {total_epochs_trained}")
    print("\nKaydedilen dosyalar:")
    print("  - best_model_v6.pth (en iyi model ağırlıkları)")
    print("  - classes.json (sınıf isimleri)")
    print("  - loss_plot.png (Loss grafiği)")
    print("  - acc_plot.png (Accuracy grafiği)")
    print("  - confusion_matrix.png (Confusion Matrix)")

    print("\n--- En İyi Model: Sınıf Bazlı Nihai Rapor ---")
    final_report = classification_report(
        final_labels, final_preds,
        target_names=classes,
        zero_division=0
    )
    print(final_report)

    model_size = os.path.getsize("best_model_v6.pth") / (1024 * 1024)
    print(f"\nModel dosya boyutu: {model_size:.1f} MB (limit: 95 MB)")
    if model_size > 95:
        print("⚠️ UYARI: Model boyutu 95 MB limitini aşıyor!")
    else:
        print("✅ Model boyutu limit dahilinde.")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs, phase_name):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase_name} Train]"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_loss = running_loss / len(train_loader)
    train_acc = accuracy_score(all_labels, all_preds)

    return train_loss, train_acc


def validate(model, val_loader, criterion, device, epoch, total_epochs, phase_name, classes):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase_name} Val]"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)

    val_f1_macro = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    val_f1_weighted = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
    val_precision_macro = precision_score(val_labels, val_preds, average='macro', zero_division=0)
    val_precision_weighted = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
    val_recall_macro = recall_score(val_labels, val_preds, average='macro', zero_division=0)
    val_recall_weighted = recall_score(val_labels, val_preds, average='weighted', zero_division=0)

    print(f"\n--- Epoch {epoch+1}/{total_epochs} [{phase_name}] ---")
    print(f"Doğrulama Kaybı: {val_loss:.4f}")
    print(f"Doğruluk (Acc): {val_acc:.4f}")
    print(f"[Macro]    Precision: {val_precision_macro:.4f} | Recall: {val_recall_macro:.4f} | F1-Score: {val_f1_macro:.4f}")
    print(f"[Weighted] Precision: {val_precision_weighted:.4f} | Recall: {val_recall_weighted:.4f} | F1-Score: {val_f1_weighted:.4f}")

    print(f"\nSınıf Bazlı Rapor (Epoch {epoch+1}):")
    report = classification_report(
        val_labels, val_preds,
        target_names=classes,
        zero_division=0
    )
    print(report)

    return val_loss, val_acc, val_f1_macro, val_preds, val_labels


if __name__ == "__main__":
    train()
