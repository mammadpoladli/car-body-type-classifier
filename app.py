import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import json
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from model import get_model


st.set_page_config(
    page_title="Araç Gövde Tipi Sınıflandırma",
    page_icon="🚗",
    layout="wide"
)


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0B1120;
        background-image: radial-gradient(circle at 50% 0%, #1e3a8a 0%, transparent 50%);
        color: #f1f5f9;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(59, 130, 246, 0); }
        100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
    }

    .stImage > img {
        border-radius: 16px;
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5);
        animation: fadeInUp 0.6s ease-out;
        border: 1px solid rgba(147, 197, 253, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.6), rgba(15, 23, 42, 0.8));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        animation: fadeInUp 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 50%; height: 100%;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.05), transparent);
        transform: skewX(-20deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { left: -100%; }
        20%, 100% { left: 200%; }
    }

    .pred-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 15px 0;
        background: linear-gradient(90deg, #60A5FA, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .script-id {
        font-family: 'Outfit', sans-serif;
        background-color: rgba(30, 58, 138, 0.8);
        color: #bfdbfe;
        padding: 6px 15px;
        border-radius: 50px;
        font-size: 0.95rem;
        font-weight: 600;
        border: 1px solid rgba(96, 165, 250, 0.3);
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #1E3A8A, #2563EB) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        animation: pulseGlow 2s infinite;
    }
    
    div.stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px -5px rgba(37, 99, 235, 0.5) !important;
        background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
    }
    
    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color: #60a5fa !important;
    }
    
    h1, h2, h3 {
        color: #f8fafc !important;
    }
</style>
""", unsafe_allow_html=True)

DEFAULT_CLASSES = ['F1_ARACLARI', 'HATCHBACK', 'MICRO', 'PICK_UP', 'SEDAN', 'STATION_WAGON', 'SUV', 'VAN']

CLASS_NAME_TO_SCRIPT_ID = {
    'SUV': 1, 'VAN': 2, 'STATION_WAGON': 3, 'MICRO': 4,
    'F1_ARACLARI': 5, 'SEDAN': 6, 'HATCHBACK': 7, 'PICK_UP': 8,
}

CLASS_DISPLAY_NAMES = {
    'SUV': 'SUV',
    'VAN': 'VAN',
    'STATION_WAGON': 'STATION WAGON',
    'MICRO': 'MİCRO',
    'F1_ARACLARI': 'AÇIK TEKERLEKLİ (F1)',
    'SEDAN': 'SEDAN',
    'HATCHBACK': 'HATCHBACK',
    'PICK_UP': 'PICK UP',
}


@st.cache_resource(show_spinner=False)
def load_classifier():
    try:
        with open('classes.json', 'r') as f:
            classes = json.load(f)
    except FileNotFoundError:
        classes = DEFAULT_CLASSES

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = get_model(num_classes=len(classes))

    try:
        model.load_state_dict(torch.load("best_model_v6.pth", map_location=device))
        model.eval()
        model.to(device)
        return model, classes, device
    except Exception as e:
        return None, classes, device


def predict(image, model, classes, device):
    transform = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    img_array = np.array(image.convert("RGB"))
    augmented = transform(image=img_array)
    tensor_img = augmented['image'].unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = F.softmax(outputs, dim=1)[0] * 100
    end_time = time.time()

    prob_values = probabilities.cpu().numpy()

    prob_dict = {classes[i]: prob_values[i] for i in range(len(classes))}
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

    max_class = list(sorted_probs.keys())[0]
    max_conf = sorted_probs[max_class]
    script_id = CLASS_NAME_TO_SCRIPT_ID.get(max_class, -1)

    return max_class, script_id, max_conf, sorted_probs, (end_time - start_time)


def batch_predict(test_dir, model, classes, device):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    results = []

    if not os.path.exists(test_dir):
        return None

    image_files_with_paths = []
    for root, dirs, files in os.walk(test_dir):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in valid_extensions:
                image_files_with_paths.append((fname, os.path.join(root, fname)))

    if not image_files_with_paths:
        return []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (fname, img_path) in enumerate(image_files_with_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            pred_class, script_id, conf, probs, infer_time = predict(image, model, classes, device)
            results.append({
                "Dosya": fname,
                "Yol": img_path,
                "Tahmin": CLASS_DISPLAY_NAMES.get(pred_class, pred_class),
                "ScriptID": script_id,
                "Conf": conf,
                "Probs": probs,
                "Time": infer_time,
                "line": f"{fname} | Pred: {script_id} "
            })
        except:
            results.append({"Dosya": fname, "Tahmin": "HATA", "line": f"{fname} | Pred: -1 "})

        progress_bar.progress((i + 1) / len(image_files_with_paths))
        status_text.text(f"İşleniyor: {i+1}/{len(image_files_with_paths)}")

    progress_bar.empty()
    status_text.empty()
    return results


def create_probability_chart(probs):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#0B1120')
    ax.set_facecolor('#0B1120')

    sorted_items = sorted(probs.items(), key=lambda x: x[1])
    names = [CLASS_DISPLAY_NAMES.get(k, k) for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    max_val = max(values) if values else 0
    colors = ['#60a5fa' if v == max_val else '#1e3a8a' for v in values]

    bars = ax.barh(names, values, color=colors, edgecolor='none', height=0.65)

    ax.set_xlabel('Olasılık (%)', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#e2e8f0', labelsize=9)
    ax.set_xlim(0, max(105, max_val + 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.grid(axis='x', alpha=0.1, color='white')

    for bar, val in zip(bars, values):
        if val > 0.5:
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'%{val:.1f}', va='center', color='#e2e8f0', fontsize=9)

    plt.tight_layout()
    return fig


st.title("Araç Gövde Tipi Sınıflandırma Modeli")

model, classes, device = load_classifier()

if model is None:
    st.error("Hata: Eğitilmiş model bulunamadı! Lütfen 'best_model_v6.pth' dosyasını kontrol edin.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([" Canlı Tahmin", " Model Özellikleri", "️ Toplu Test ", " Performans Analizi (Test)"])

with tab1:
    st.subheader("Görsel Yükle (Çoklu Seçim Yapabilirsiniz)")
    uploaded_files = st.file_uploader("Analiz edilecek araba fotoğraflarını seçin (Birden fazla seçebilirsiniz)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if uploaded_files:
        predict_button = st.button("Analizi Başlat", type="primary", use_container_width=True)

        if predict_button:
            st.markdown("<br>", unsafe_allow_html=True)
            for i, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns([1, 1], gap="large")

                image = Image.open(uploaded_file)
                with col1:
                    st.image(image, caption=uploaded_file.name, use_container_width=True)

                with col2:
                    with st.spinner(f"{uploaded_file.name} analiz ediliyor..."):
                        pred_class, script_id, conf, probs, infer_time = predict(image, model, classes, device)

                    display_name = CLASS_DISPLAY_NAMES.get(pred_class, pred_class)

                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div style="color: #64748b; font-size: 0.9rem; text-transform: uppercase;">Tespit Edilen Sınıf</div>
                            <div class="pred-title">{display_name}</div>
                            <span class="script-id">Script ID: {script_id}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    m1, m2 = st.columns(2)
                    m1.metric("Güven Skoru", f"%{conf:.2f}")
                    m2.metric("Çıkarım Süresi", f"{infer_time*1000:.1f} ms")

                    st.markdown("#### Olasılık Dağılımı (Tüm Sınıflar)")
                    fig_bar = create_probability_chart(probs)
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

                if i < len(uploaded_files) - 1:
                    st.markdown("<hr style='border: 1px solid rgba(147, 197, 253, 0.2); margin: 30px 0;'>", unsafe_allow_html=True)

    else:
        st.info("Lütfen analize başlamak için bir veya daha fazla görsel yükleyin.")


with tab2:
    st.header("Model Özellikleri")

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        **Mimari Detayları:**
        - **Model:** EfficientNet-B4
        - **Giriş Çözünürlüğü:** 380x380 piksel
        """)
    with col_info2:
        st.markdown("""
        **Nihai Başarı Metrikleri (V6):**
        - **Macro F1-Score:** %96.83
        - **Doğruluk (Accuracy):** %96.70
        """)

    st.markdown("---")
    st.subheader("Model Performans Grafikleri")

    try:
        if os.path.exists("loss_plot.png") and os.path.exists("acc_plot.png"):
            g1, g2 = st.columns(2)
            with g1:
                st.image("loss_plot.png", caption="Training & Validation Loss Grafiği", use_container_width=True)
            with g2:
                st.image("acc_plot.png", caption="Training & Validation Accuracy Grafiği", use_container_width=True)

        if os.path.exists("confusion_matrix.png"):
            st.markdown("#### Normalized Confusion Matrix (8x8)")
            st.image("confusion_matrix.png", caption="Hata Matrisi (Gerçek vs Tahmin)", use_container_width=True)
    except Exception as e:
        st.warning("Grafikler şu an görüntülenemiyor.")


with tab3:
    st.header("Toplu Tahmin (Batch Inference)")
    st.markdown("""
    **Toplu Test Modülü**, büyük ölçekli veri setlerinin tek seferde analiz edilmesi için tasarlanmıştır.
    Sisteme verilen kök dizindeki tüm klasör ve alt klasör hiyerarşisi otomatik olarak taranır,
    her bir görsel modelden geçirilerek proje isterlerine uygun formattaki **preds.txt** çıktı dosyasına yazılır.
    """)

    if "selected_dir" not in st.session_state:
        st.session_state.selected_dir = "test_images"

    c1, c2 = st.columns([1, 4])
    with c1:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📁 Klasör Seç (Mac)", use_container_width=True):
            import subprocess, sys
            if sys.platform == "darwin":
                script = 'tell application "System Events" to activate\n tell application "System Events" to return POSIX path of (choose folder)'
                try:
                    res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                    if res.returncode == 0 and res.stdout.strip():
                        st.session_state.selected_dir = res.stdout.strip()
                        st.rerun()
                except:
                    pass
            else:
                st.warning("Bu özellik sadece MacOS içindir.")

    with c2:
        test_dir_input = st.text_input("Hedef Klasör Yolu (Mutlak veya Göreceli Yol):", key="selected_dir")

    if st.button("Toplu Testi Başlat", type="primary"):
        with st.spinner("Görseller analiz ediliyor..."):
            results = batch_predict(test_dir_input, model, classes, device)

            if results is None:
                st.error("Belirtilen klasör bulunamadı!")
                if "batch_results" in st.session_state: del st.session_state["batch_results"]
            elif len(results) == 0:
                st.warning("Klasörde geçerli resim (.jpg, .png vb.) bulunamadı.")
                if "batch_results" in st.session_state: del st.session_state["batch_results"]
            else:
                st.session_state.batch_results = results
                st.session_state.display_count = 20
                st.success(f"{len(results)} görsel başarıyla analiz edildi.")

                lines_to_write = [r['line'] for r in results]
                try:
                    with open("preds.txt", "w", encoding="utf-8") as f:
                        for line in lines_to_write:
                            f.write(line + "\n")
                    st.info("Sonuçlar proje ana dizinine `preds.txt` olarak kaydedildi.")
                except Exception as e:
                    st.error(f"Dosya yazma hatası: {str(e)}")

    if "batch_results" in st.session_state and st.session_state.batch_results:
        results = st.session_state.batch_results
        display_limit = st.session_state.get("display_count", 20)

        with st.expander(f"Görsel Sonuçları İncele (Gösterilen: {min(display_limit, len(results))} / Toplam: {len(results)})", expanded=True):
            st.markdown("<br>", unsafe_allow_html=True)
            for idx, res in enumerate(results[:display_limit]):
                if "HATA" in res.get("Tahmin", ""):
                    st.error(f"{res['Dosya']} okunamadı.")
                    continue

                img_col, info_col = st.columns([1, 1], gap="large")

                with img_col:
                    try:
                        img = Image.open(res["Yol"])
                        st.image(img, caption=res["Dosya"], use_container_width=True)
                    except:
                        st.warning("Görsel yüklenemedi.")

                with info_col:
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div style="color: #64748b; font-size: 0.9rem; text-transform: uppercase;">Tespit Edilen Sınıf</div>
                            <div class="pred-title">{res['Tahmin']}</div>
                            <span class="script-id">Script ID: {res['ScriptID']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    m1, m2 = st.columns(2)
                    m1.metric("Güven Skoru", f"%{res['Conf']:.2f}")
                    m2.metric("Çıkarım Süresi", f"{res['Time']*1000:.1f} ms")

                    fig_bar = create_probability_chart(res['Probs'])
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

                if idx < min(len(results), display_limit) - 1:
                    st.markdown("<hr style='border: 1px solid rgba(147, 197, 253, 0.2); margin: 30px 0;'>", unsafe_allow_html=True)

            if len(results) > display_limit:
                if st.button("Daha Fazla Göster (+20) ⬇️", use_container_width=True):
                    st.session_state.display_count += 20
                    st.rerun()

        with st.expander("preds.txt Çıktısını İncele", expanded=False):
            lines_to_write = [r['line'] for r in results]
            st.code("\n".join(lines_to_write) + "\n...", language="text")

with tab4:
    st.header("Model Performans Analizi (Test.txt Entegrasyonu)")
    st.markdown("""
    Bu modül, modelin tahmin performansını detaylı olarak analiz etmek ve doğrulamak için tasarlanmıştır.
    Modelin `preds.txt` çıktı dosyası ile cevap anahtarı niteliğindeki `True.txt` dosyası karşılaştırılarak;
    doğru/yanlış tahmin sayıları, **Sınıflandırma Raporu (Classification Report)** ve **Karmaşıklık Matrisi (Confusion Matrix)** hesaplanıp ekranda görselleştirilir.
    """)

    st.markdown("<hr style='border: 1px solid rgba(147, 197, 253, 0.2);'>", unsafe_allow_html=True)

    col_file1, col_file2 = st.columns(2, gap="large")
    with col_file1:
        st.markdown("**1. Tahmin Dosyası**")
        preds_file = st.file_uploader("Oluşturduğunuz `preds.txt` dosyasını yükleyin", type=["txt"], key="preds_upload")
    with col_file2:
        st.markdown("**2. Gerçek Etiket Dosyası**")
        true_file = st.file_uploader("Cevap anahtarı olan `True.txt` dosyasını yükleyin", type=["txt"], key="true_upload")

    if preds_file is not None and true_file is not None:
        if st.button("🚀 Performans Analizini Başlat", type="primary", use_container_width=True):
            with st.spinner("Sonuçlar hesaplanıyor ve grafikler oluşturuluyor..."):
                try:
                    def read_predictions_from_upload(uploaded_file):
                        data = {}
                        content = uploaded_file.getvalue().decode("utf-8")
                        for line in content.splitlines():
                            line = line.strip()
                            if not line: continue
                            parts = line.split(" | ")
                            if len(parts) >= 2:
                                filename = parts[0].strip()
                                value = int(parts[1].split(":")[1].strip())
                                data[filename] = value
                        return data

                    pred_dict = read_predictions_from_upload(preds_file)
                    true_dict = read_predictions_from_upload(true_file)

                    common_files = sorted(set(pred_dict.keys()) & set(true_dict.keys()))

                    if len(common_files) == 0:
                        st.error("HATA: İki dosyada ortak hiçbir resim dosyası bulunamadı. Lütfen formata uygun dosyalar yüklediğinizden emin olun.")
                    else:
                        st.success(f"Başarılı! İki dosyada toplam **{len(common_files)}** ortak tahmin eşleştirildi.")

                        y_true = []
                        y_pred = []
                        for file_name in common_files:
                            y_true.append(true_dict[file_name])
                            y_pred.append(pred_dict[file_name])

                        class_labels = [1, 2, 3, 4, 5, 6, 7, 8]
                        class_names = ["1-SUV", "2-VAN", "3-STATION WAGON", "4-MICRO", "5-F1 ARAÇLARI", "6-SEDAN", "7-HATCHBACK", "8-PICK UP"]

                        correct_preds = sum(1 for t, p in zip(y_true, y_pred) if t == p)
                        wrong_preds = len(y_true) - correct_preds

                        report_dict = classification_report(y_true, y_pred, labels=class_labels, target_names=class_names, zero_division=0, output_dict=True)
                        macro_f1 = report_dict['macro avg']['f1-score']

                        st.markdown("<br><h3>🏆 Genel Performans Özeti</h3>", unsafe_allow_html=True)
                        m1, m2, m3 = st.columns(3)
                        m1.metric("✅ Doğru Tahmin Sayısı", f"{correct_preds} Dosya")
                        m2.metric("❌ Yanlış Tahmin Sayısı", f"{wrong_preds} Dosya")
                        m3.metric("🎯 Macro F1-Skoru", f"%{macro_f1 * 100:.2f}")

                        st.markdown("<hr style='border: 1px solid rgba(147, 197, 253, 0.2); margin: 20px 0;'>", unsafe_allow_html=True)

                        c1, c2 = st.columns([1.2, 1], gap="medium")

                        with c1:
                            st.markdown("### 📊 Karmaşıklık Matrisi (Confusion Matrix)")
                            cm = confusion_matrix(y_true, y_pred, labels=class_labels, normalize='true')

                            fig, ax = plt.subplots(figsize=(8, 6))
                            fig.patch.set_facecolor('#0B1120')
                            ax.set_facecolor('#0B1120')

                            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                                        xticklabels=class_names,
                                        yticklabels=class_names,
                                        linewidths=.5, ax=ax)

                            ax.set_xlabel("Tahmin Edilen (Predicted)", color='white')
                            ax.set_ylabel("Gerçek (True)", color='white')
                            ax.tick_params(colors='white')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()

                            st.pyplot(fig)

                        with c2:
                            st.markdown("### 📑 Sınıflandırma Raporu (Classification Report)")

                            df_report = pd.DataFrame(report_dict).transpose()
                            df_formatted = df_report.copy()
                            for col in ['precision', 'recall', 'f1-score']:
                                if col in df_formatted.columns:
                                    df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.4f}")

                            st.dataframe(df_formatted, use_container_width=True, height=400)

                except Exception as e:
                    st.error(f"Beklenmeyen bir hata oluştu. Lütfen dosyaların formatını kontrol edin. Hata Detayı: {str(e)}")
