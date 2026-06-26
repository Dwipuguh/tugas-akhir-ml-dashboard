
"""
╔══════════════════════════════════════════════════════════════════╗
║       PROYEK AKHIR - PEMBELAJARAN MESIN                          ║
║       Heart Disease Prediction: ANN vs SVM                       ║
║       Institut Teknologi Sepuluh Nopember (ITS)                  ║
║       Departemen Statistika Bisnis, Fakultas Vokasi              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TABLE OF CONTENTS                                               ║
║  ──────────────────────────────────────────────────────────      ║
║  [1]  IMPORT LIBRARY & KONFIGURASI                               ║
║  [2]  DATA LOADING & PREPROCESSING                               ║
║       2.1  Load Dataset (xlsx)                                   ║
║       2.2  Encoding Fitur Kategorikal                            ║
║       2.3  Train-Test Split & Normalisasi                        ║
║  [3]  TRAINING MODEL                                             ║
║       3.1  Artificial Neural Network (ANN)                       ║
║            Arsitektur : 13 → Dense(64) → Dense(32)               ║
║                            → Dense(16) → Dense(1)                ║
║            Regularisasi: BatchNorm · Dropout · L2                ║
║            Optimizer   : Adam(lr=0.001)                          ║
║            Loss        : Binary Crossentropy                     ║
║       3.2  Support Vector Machine (SVM)                          ║
║            Tuning      : GridSearchCV, 5-fold CV                 ║
║            Parameter   : C=[0.1,1,10,100], kernel=[rbf,linear]   ║
║  [4]  FUNGSI EVALUASI & VISUALISASI                              ║
║       4.1  Metrik: Accuracy, Precision, Recall, F1, AUC-ROC      ║
║       4.2  Confusion Matrix                                      ║
║       4.3  ROC Curve                                             ║
║  [5]  STREAMLIT DASHBOARD                                        ║
║       5.1  Konfigurasi Halaman & Tema CSS                        ║
║       5.2  Sidebar Navigasi                                      ║
║       5.3  Halaman: Beranda                                      ║
║       5.4  Halaman: EDA                                          ║
║       5.5  Halaman: Model ANN                                    ║
║       5.6  Halaman: Model SVM                                    ║
║       5.7  Halaman: Perbandingan Model                           ║
║       5.8  Halaman: Prediksi Pasien                              ║
║  [6]  MAIN FUNCTION                                              ║
║  ──────────────────────────────────────────────────────────      ║ 
╚══════════════════════════════════════════════════════════════════╝
"""


# ══════════════════════════════════════════════════════════════════
# [1]  IMPORT LIBRARY & KONFIGURASI
# ══════════════════════════════════════════════════════════════════
 
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks as keras_callbacks
 
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
 
# ─────────────────────────────────────────────────────────────
# [5.1]  Konfigurasi Halaman & Tema CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease · ANN vs SVM",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
}
[data-testid="stSidebar"] * { color: #c9d6df !important; }
[data-testid="stSidebar"] hr { border-color: #415a77 !important; }
 
/* ── Metric Cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1b263b, #0d1b2a);
    border: 1px solid #415a77;
    border-radius: 12px;
    padding: 16px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.35);
}
[data-testid="stMetricValue"] { color: #4fc3f7 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #90caf9 !important; }
 
/* ── Typography ── */
h1 { color: #e0f7fa !important; }
h2 { color: #b3e5fc !important; }
h3 { color: #81d4fa !important; }
p, li { color: #c9d6df; }
 
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0d47a1);
    color: #fff; border: none; border-radius: 10px;
    padding: 12px 24px; font-size: 15px; font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1976d2, #1565c0);
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(21,101,192,0.4);
}
 
/* ── Tabs ── */
[data-testid="stTabs"] { border-bottom: 1px solid #415a77; }
 
/* ── Divider ── */
hr { border-color: #415a77 !important; }
 
/* ── Prediction result card ── */
.result-card {
    background: linear-gradient(135deg, #0d1b2a, #1b263b);
    border-radius: 14px;
    padding: 22px 28px;
    text-align: center;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════
# [2]  DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════
 
# Kamus deskripsi fitur (Bahasa Indonesia)
FEATURE_DESC = {
    "age":      "Usia (tahun)",
    "sex":      "Jenis Kelamin  (1=Male, 0=Female)",
    "cp":       "Tipe Nyeri Dada  (0=Typical Angina, 1=Atypical, 2=Non-Anginal, 3=Asymptomatic)",
    "trestbps": "Tekanan Darah Istirahat (mm Hg)",
    "chol":     "Kolesterol Serum (mg/dl)",
    "fbs":      "Gula Darah Puasa > 120 mg/dl  (1=Ya, 0=Tidak)",
    "restecg":  "Hasil EKG Istirahat  (0=Normal, 1=Abnormal ST-T, 2=LV Hypertrophy)",
    "thalach":  "Detak Jantung Maksimum yang Dicapai",
    "exang":    "Angina akibat Olahraga  (1=Ya, 0=Tidak)",
    "oldpeak":  "Depresi ST Akibat Olahraga Relatif terhadap Istirahat",
    "slope":    "Kemiringan Segmen ST Puncak  (0=Down, 1=Flat, 2=Up)",
    "ca":       "Jumlah Pembuluh Darah Utama  (0–3)",
    "thal":     "Thalassemia  (0=Normal, 1=Fixed Defect, 2=Reversible Defect)",
}
 
# ─────────────────────────────────────────────────────────────
# [2.1]  Load Dataset
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame | None:
    """Cari dan baca file dataset xlsx dari beberapa lokasi umum di Colab."""
    candidates = [
        "heart__1_.xlsx",
        "heart.xlsx",
        "/content/heart__1_.xlsx",
        "/content/heart.xlsx",
        "/content/drive/MyDrive/heart__1_.xlsx",
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_excel(path)
            # Buang kolom metadata yang tidak relevan
            drop_cols = [c for c in df.columns if "Unnamed" in str(c) or "Attributes" in str(c)]
            df.drop(columns=drop_cols, inplace=True)
            return df
    return None
 
 
# ─────────────────────────────────────────────────────────────
# [2.2 & 2.3]  Encoding, Split, dan Normalisasi
# ─────────────────────────────────────────────────────────────
@st.cache_data
def preprocess(_df: pd.DataFrame):
    """
    2.2  Encoding fitur kategorikal (sex: male→1, female→0)
    2.3  Split 3 arah: Train 70% / Val 15% / Test 15% (stratified)
         + StandardScaler (fit HANYA pada train)
 
    FIX: drop_duplicates() sebelum split untuk menghilangkan
    data leakage akibat 697 baris duplikat pada dataset raw.
    FIX: Validation set dipisah dari test set agar EarlyStopping
    ANN tidak mengintip data test (information leakage).
    """
    df = _df.copy()
 
    # Encoding kolom sex (string → biner)
    if df["sex"].dtype == object:
        df["sex"] = df["sex"].map({"male": 1, "female": 0})
 
    df.dropna(inplace=True)
 
    # ── FIX #1: Hapus baris duplikat sebelum split ──────────
    # Dataset raw: 999 baris, hanya 302 unik (697 duplikat).
    # Tanpa ini, 73.5% data test bocor ke train → Recall = 1.0
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
 
    X = df.drop("target", axis=1)
    y = df["target"]
 
    # ── FIX #2: Split 3 arah (Train / Validation / Test) ────
    # Tahap 1: pisahkan test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Tahap 2: dari sisa, pisahkan val (≈15% total ≈ 17.6% dari sisa)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
 
    # Normalisasi: fit HANYA pada train, transform val & test
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
 
    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_sc, X_val_sc, X_test_sc, scaler)
 
 
# ══════════════════════════════════════════════════════════════════
# [3]  TRAINING MODEL
# ══════════════════════════════════════════════════════════════════
 
# ─────────────────────────────────────────────────────────────
# [3.1]  Artificial Neural Network (ANN) — TensorFlow/Keras
# ─────────────────────────────────────────────────────────────
def _build_ann(n_features: int = 13):
    """
    Arsitektur ANN — didesain khusus untuk dataset kecil (≈211 train samples):
        Input(13)
        → Dense(32, ReLU, L2=0.01) + Dropout(0.4)
        → Dense(16, ReLU, L2=0.01) + Dropout(0.3)
        → Dense(1,  Sigmoid)
 
    Alasan desain:
    - HAPUS BatchNormalization: tidak stabil dengan batch kecil & dataset kecil
    - KURANGI neuron (64→32, 32→16): cegah overfitting pada ≈211 sampel
    - TINGKATKAN L2 (0.001→0.01) + Dropout (0.3→0.4): regularisasi lebih ketat
    - HAPUS hidden layer 3: parameter terlalu banyak untuk dataset sekecil ini
    """
    tf.random.set_seed(42)
    np.random.seed(42)
 
    model = keras.Sequential([
        layers.Input(shape=(n_features,), name="input"),
 
        # Hidden Layer 1
        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer="he_normal", name="dense_1"),
        layers.Dropout(0.4, name="dropout_1"),
 
        # Hidden Layer 2
        layers.Dense(16, activation="relu",
                     kernel_regularizer=regularizers.l2(0.01),
                     kernel_initializer="he_normal", name="dense_2"),
        layers.Dropout(0.3, name="dropout_2"),
 
        # Output
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="ANN_HeartDisease")
 
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
 
 
def train_ann(X_train_sc, y_train, X_val_sc, y_val):
    """
    Training ANN dengan:
    - Validation set terpisah (bukan test set) untuk EarlyStopping
    - class_weight='balanced' agar model tidak bias ke kelas mayoritas
    """
    from sklearn.utils.class_weight import compute_class_weight
 
    # Hitung bobot kelas otomatis (atasi imbalance ringan 164 vs 138)
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
 
    model = _build_ann(n_features=X_train_sc.shape[1])
 
    cb_list = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss", patience=30,
            restore_best_weights=True, verbose=0
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=12, min_lr=1e-6, verbose=0
        ),
    ]
 
    history = model.fit(
        X_train_sc, y_train,
        validation_data=(X_val_sc, y_val),  # val set, bukan test set
        epochs=300, batch_size=16,
        class_weight=class_weight,           # bobot kelas seimbang
        callbacks=cb_list, verbose=0,
    )
    return model, history.history
 
 
# ─────────────────────────────────────────────────────────────
# [3.2]  Support Vector Machine (SVM) — Scikit-learn
# ─────────────────────────────────────────────────────────────
def train_svm(X_train_sc, y_train):
    """
    Hyperparameter Tuning dengan GridSearchCV (5-fold CV):
        C            : [0.1, 1, 10, 100]
        kernel       : ['rbf', 'linear']
        gamma        : ['scale', 'auto']
        class_weight : 'balanced' (fix bias ke kelas mayoritas)
    """
    param_grid = {
        "C":      [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma":  ["scale", "auto"],
    }
    # class_weight='balanced' wajib agar SVM tidak bias ke kelas mayoritas
    svm = SVC(probability=True, random_state=42, class_weight="balanced")
    gs  = GridSearchCV(svm, param_grid, cv=5, scoring="f1",  # scoring f1 lebih adil
                        n_jobs=-1, verbose=0)
    gs.fit(X_train_sc, y_train)
 
    return gs.best_estimator_, gs.best_params_, round(gs.best_score_, 4)
 
 
# ══════════════════════════════════════════════════════════════════
# [4]  FUNGSI EVALUASI & VISUALISASI
# ══════════════════════════════════════════════════════════════════
 
# ─────────────────────────────────────────────────────────────
# [4.1]  Threshold Optimal via Youden's J Statistic
# ─────────────────────────────────────────────────────────────
def find_optimal_threshold(y_val_true, y_val_prob) -> float:
    """
    Cari threshold optimal menggunakan Youden's J statistic pada validation set:
        J = Sensitivity + Specificity - 1 = TPR - FPR
    Threshold yang memaksimalkan J dipilih sebagai threshold final.
 
    Mengapa tidak pakai 0.5?
    Pada dataset kecil, distribusi probabilitas output model tidak selalu
    terpusat di 0.5. Threshold 0.5 bisa menyebabkan model selalu predict
    satu kelas. Youden's J mencari titik keseimbangan TPR-FPR yang optimal.
    """
    fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
    youden_j = tpr - fpr
    best_idx  = np.argmax(youden_j)
    optimal   = float(thresholds[best_idx])
    # Clamp agar tidak ekstrem (0.15 – 0.85)
    return float(np.clip(optimal, 0.15, 0.85))
 
 
# ─────────────────────────────────────────────────────────────
# [4.2]  Hitung Semua Metrik
# ─────────────────────────────────────────────────────────────
def evaluate_model(y_true, y_pred, y_prob, name: str):
    cm        = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc       = round(roc_auc_score(y_true, y_prob), 4)
    metrics   = {
        "Model"    : name,
        "Accuracy" : round(accuracy_score(y_true, y_pred),            4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0),    4),
        "F1-Score" : round(f1_score(y_true, y_pred, zero_division=0),        4),
        "AUC-ROC"  : auc,
    }
    return metrics, cm, fpr, tpr, auc
 
 
# ─────────────────────────────────────────────────────────────
# [4.2]  Plot Confusion Matrix
# ─────────────────────────────────────────────────────────────
def fig_confusion_matrix(cm, title: str, colorscale="Blues"):
    labels = [
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"],
    ]
    text = [[f"{labels[i][j]}<br>{cm[i, j]}" for j in range(2)] for i in range(2)]
    fig  = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted: 0 (Sehat)", "Predicted: 1 (Sakit)"],
        y=["Actual: 0 (Sehat)",    "Actual: 1 (Sakit)"],
        text=text, texttemplate="%{text}",
        textfont={"size": 13},
        colorscale=colorscale, showscale=False,
    ))
    fig.update_layout(
        title=title, height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig
 
 
# ─────────────────────────────────────────────────────────────
# [4.3]  Plot ROC Curve
# ─────────────────────────────────────────────────────────────
def fig_roc_comparison(fpr_ann, tpr_ann, auc_ann,
                        fpr_svm, tpr_svm, auc_svm):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_ann, y=tpr_ann, name=f"ANN  (AUC={auc_ann})",
                             line=dict(color="#4fc3f7", width=2.5)))
    fig.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm, name=f"SVM  (AUC={auc_svm})",
                             line=dict(color="#f48fb1", width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Classifier",
                             line=dict(color="gray", dash="dash", width=1.5)))
    fig.update_layout(
        title="ROC Curve — ANN vs SVM", height=430,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,27,42,0.85)",
        font_color="white",
        legend=dict(x=0.55, y=0.15),
    )
    return fig
 
 
# Warna default plotly (dark background)
PLOT_BG  = "rgba(13,27,42,0.85)"
PAPER_BG = "rgba(0,0,0,0)"
FONT_COL = "white"
 
def _base_layout(**kwargs):
    return dict(
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font_color=FONT_COL,
        **kwargs,
    )
 
 
# ══════════════════════════════════════════════════════════════════
# [5]  STREAMLIT DASHBOARD
# ══════════════════════════════════════════════════════════════════
 
# ─────────────────────────────────────────────────────────────
# [5.2]  Sidebar Navigasi
# ─────────────────────────────────────────────────────────────
def render_sidebar() -> str:
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 18px 0 10px;">
        <span style="font-size:3em;">❤️</span>
        <h2 style="color:#4fc3f7; margin:6px 0 2px;">Heart Disease</h2>
        <p style="color:#7a9cc0; font-size:12px; margin:0;">
            ANN &amp; SVM · ITS Statistika Bisnis
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)
 
    menu = st.sidebar.radio("📋 Navigasi", [
        "🏠  Beranda",
        "📊  EDA",
        "🧠  Model ANN",
        "⚙️  Model SVM",
        "📈  Perbandingan Model",
        "🔮  Prediksi Pasien",
    ], label_visibility="collapsed")
 
    st.sidebar.markdown("""
    <hr>
    <p style="font-size:11px; color:#5e7a94; text-align:center;">
        Proyek Akhir · Pembelajaran Mesin<br>
        Semester Genap 2025/2026
    </p>
    """, unsafe_allow_html=True)
    return menu
 
 
# ─────────────────────────────────────────────────────────────
# [5.3]  Halaman Beranda
# ─────────────────────────────────────────────────────────────
def page_beranda(df: pd.DataFrame):
    st.title("❤️ Heart Disease Prediction")
    st.markdown(
        "**Analisis Komparatif: Artificial Neural Network (ANN) vs "
        "Support Vector Machine (SVM)**  \n"
        "Dataset: Heart Disease UCI  •  ITS Statistika Bisnis, Fakultas Vokasi"
    )
    st.markdown("---")
 
    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sampel",    f"{len(df):,}")
    c2.metric("Jumlah Fitur",    "13")
    c3.metric("Sakit Jantung",   int(df["target"].sum()))
    c4.metric("Sehat",           int((df["target"] == 0).sum()))
    c5.metric("Rasio Positif",   f"{df['target'].mean()*100:.1f}%")
 
    st.markdown("---")
    col_desc, col_pie = st.columns([1.3, 1])
 
    # Tabel deskripsi fitur
    with col_desc:
        st.subheader("📋 Deskripsi Fitur")
        feat_df = pd.DataFrame({
            "No.":        list(range(1, 14)),
            "Fitur":      list(FEATURE_DESC.keys()),
            "Keterangan": list(FEATURE_DESC.values()),
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=420)
 
    # Pie distribusi target
    with col_pie:
        st.subheader("🎯 Distribusi Target")
        target_cnt = df["target"].value_counts().reset_index()
        target_cnt.columns = ["Target", "Count"]
        target_cnt["Label"] = target_cnt["Target"].map({0: "Sehat (0)", 1: "Sakit Jantung (1)"})
 
        fig = px.pie(
            target_cnt, values="Count", names="Label", hole=0.48,
            color_discrete_sequence=["#4fc3f7", "#f06292"],
        )
        fig.update_layout(**_base_layout(height=310, margin=dict(l=20, r=20, t=20, b=20)))
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)
 
        sehat  = (df["target"] == 0).sum()
        sakit  = df["target"].sum()
        st.info(
            f"⚖️ **Class Balance:**  Sehat = `{sehat}`  |  "
            f"Sakit = `{sakit}`  *(hampir seimbang)*"
        )
 
    # Statistik deskriptif
    st.markdown("---")
    st.subheader("📌 Statistik Deskriptif")
    df_disp = df.copy()
    if df_disp["sex"].dtype != object:
        df_disp["sex"] = df_disp["sex"].map({1: "male", 0: "female"})
    st.dataframe(df_disp.describe().round(3), use_container_width=True)
 
    st.subheader("📄 Preview Dataset (10 baris pertama)")
    st.dataframe(df_disp.head(10), use_container_width=True, hide_index=True)
 
 
# ─────────────────────────────────────────────────────────────
# [5.4]  Halaman EDA
# ─────────────────────────────────────────────────────────────
def page_eda(df: pd.DataFrame):
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("---")
 
    # Siapkan df numerik
    df_num = df.copy()
    if df_num["sex"].dtype == object:
        df_num["sex"] = df_num["sex"].map({"male": 1, "female": 0})
    df_num["Status"] = df_num["target"].map({0: "Sehat", 1: "Sakit Jantung"})
 
    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 Distribusi Fitur",
        "🔥 Matriks Korelasi",
        "🎯 Analisis per Status",
        "📦 Boxplot",
    ])
 
    # ── Tab 1: Distribusi ──────────────────────────────────────
    with tab1:
        st.subheader("Distribusi Fitur Numerik Utama")
        num_feats  = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        colors_num = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8"]
        fig = make_subplots(rows=2, cols=3,
            subplot_titles=[FEATURE_DESC[c].split("(")[0].strip() for c in num_feats])
 
        for idx, (col, clr) in enumerate(zip(num_feats, colors_num)):
            r, c = divmod(idx, 3)
            fig.add_trace(go.Histogram(
                x=df_num[col], name=col,
                marker_color=clr, opacity=0.85, nbinsx=22,
            ), row=r + 1, col=c + 1)
 
        fig.update_layout(**_base_layout(height=480, showlegend=False,
                                          title_text="Distribusi Fitur Numerik"))
        st.plotly_chart(fig, use_container_width=True)
 
        st.subheader("Distribusi Fitur Kategorikal")
        cat_feats = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
        fig2 = make_subplots(rows=2, cols=4, subplot_titles=cat_feats)
        for idx, col in enumerate(cat_feats):
            r, c = divmod(idx, 4)
            vc = df_num[col].value_counts().sort_index()
            fig2.add_trace(go.Bar(
                x=vc.index.astype(str), y=vc.values, name=col,
                marker_color="#4fc3f7", opacity=0.85,
            ), row=r + 1, col=c + 1)
        fig2.update_layout(**_base_layout(height=420, showlegend=False))
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Tab 2: Korelasi ────────────────────────────────────────
    with tab2:
        st.subheader("Heatmap Korelasi Pearson")
        corr = df_num.drop("Status", axis=1).corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r", zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}", textfont={"size": 8.5},
        ))
        fig.update_layout(**_base_layout(height=570,
                                          margin=dict(l=10, r=10, t=30, b=10)))
        st.plotly_chart(fig, use_container_width=True)
 
        st.subheader("🎯 Korelasi Fitur terhadap Target (diurutkan)")
        target_corr = corr["target"].drop("target").sort_values(ascending=True)
        fig2 = go.Figure(go.Bar(
            x=target_corr.values, y=target_corr.index, orientation="h",
            marker_color=["#4fc3f7" if v > 0 else "#f06292" for v in target_corr.values],
            text=target_corr.round(3).values, textposition="outside",
        ))
        fig2.update_layout(**_base_layout(height=400,
                                           xaxis_title="Korelasi",
                                           yaxis={"categoryorder": "total ascending"}))
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Tab 3: Analisis per Status ─────────────────────────────
    with tab3:
        st.subheader("Distribusi Fitur Berdasarkan Status Penyakit Jantung")
        CMAP = {"Sehat": "#4fc3f7", "Sakit Jantung": "#f06292"}
 
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df_num, x="age", color="Status",
                                barmode="overlay", nbins=20,
                                color_discrete_map=CMAP,
                                title="Distribusi Usia per Status")
            fig.update_layout(**_base_layout(height=330, bargap=0.05))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(df_num, x="thalach", color="Status",
                                barmode="overlay", nbins=20,
                                color_discrete_map=CMAP,
                                title="Detak Jantung Maks per Status")
            fig.update_layout(**_base_layout(height=330, bargap=0.05))
            st.plotly_chart(fig, use_container_width=True)
 
        c1, c2 = st.columns(2)
        with c1:
            sex_grp = df_num.copy()
            sex_grp["sex"] = sex_grp["sex"].map({1: "Male", 0: "Female"})
            grp = sex_grp.groupby(["sex", "Status"]).size().reset_index(name="Count")
            fig = px.bar(grp, x="sex", y="Count", color="Status", barmode="group",
                          color_discrete_map=CMAP, title="Gender per Status")
            fig.update_layout(**_base_layout(height=330))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            grp2 = df_num.groupby(["cp", "Status"]).size().reset_index(name="Count")
            fig = px.bar(grp2, x="cp", y="Count", color="Status", barmode="group",
                          color_discrete_map=CMAP,
                          title="Tipe Nyeri Dada per Status",
                          labels={"cp": "Chest Pain Type"})
            fig.update_layout(**_base_layout(height=330))
            st.plotly_chart(fig, use_container_width=True)
 
        # Scatter: Usia vs Thalach
        st.subheader("Scatter: Usia vs Detak Jantung Maks")
        fig = px.scatter(df_num, x="age", y="thalach",
                          color="Status", color_discrete_map=CMAP,
                          opacity=0.65, size_max=6,
                          labels={"age": "Usia", "thalach": "Detak Jantung Maks"})
        fig.update_layout(**_base_layout(height=380))
        st.plotly_chart(fig, use_container_width=True)
 
    # ── Tab 4: Boxplot ─────────────────────────────────────────
    with tab4:
        st.subheader("Boxplot Fitur Numerik per Status")
        CMAP = {"Sehat": "#4fc3f7", "Sakit Jantung": "#f06292"}
        sel = st.selectbox(
            "Pilih Fitur:",
            ["age", "trestbps", "chol", "thalach", "oldpeak"],
            format_func=lambda x: FEATURE_DESC.get(x, x)
        )
        fig = px.box(df_num, x="Status", y=sel, color="Status",
                      color_discrete_map=CMAP, points="outliers",
                      title=f"Boxplot  {FEATURE_DESC.get(sel, sel)}")
        fig.update_layout(**_base_layout(height=440, showlegend=False))
        st.plotly_chart(fig, use_container_width=True)
 
        # Violin sebagai bonus
        fig2 = px.violin(df_num, x="Status", y=sel, color="Status",
                          color_discrete_map=CMAP, box=True,
                          title=f"Violin Plot  {FEATURE_DESC.get(sel, sel)}")
        fig2.update_layout(**_base_layout(height=400, showlegend=False))
        st.plotly_chart(fig2, use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────
# [5.5]  Halaman Model ANN
# ─────────────────────────────────────────────────────────────
def page_ann(ann_model, ann_history: dict, X_test_sc, y_test, ann_threshold: float):
    st.title("🧠 Model Artificial Neural Network (ANN)")
    st.markdown("---")
 
    # ── Arsitektur ──
    st.subheader("🏗️ Arsitektur Model")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Layer Input**
- 13 neuron (sesuai jumlah fitur)
 
**Hidden Layer 1**
- 32 neuron, aktivasi ReLU
- Dropout 40%
- Regularisasi L2 (0.01)
        """)
    with c2:
        st.markdown("""
**Hidden Layer 2**
- 16 neuron, aktivasi ReLU
- Dropout 30%
- Regularisasi L2 (0.01)
        """)
    with c3:
        st.markdown("""
**Layer Output**
- 1 neuron, aktivasi Sigmoid
 
**Kompilasi**
- Optimizer : Adam (lr=0.001)
- Loss      : Binary Crossentropy
- Class Weight: Balanced
 
**Callback**
- EarlyStopping (patience=30)
- ReduceLROnPlateau (patience=12)
        """)
 
    st.markdown("---")
 
    # ── Training History ──
    st.subheader("📈 Training History")
    epochs = list(range(1, len(ann_history["loss"]) + 1))
    c1, c2 = st.columns(2)
 
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=ann_history["loss"],
                                  name="Train Loss", line=dict(color="#4fc3f7", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=ann_history["val_loss"],
                                  name="Val Loss", line=dict(color="#f06292", width=2, dash="dash")))
        fig.update_layout(**_base_layout(title="Loss Curve",
                                          xaxis_title="Epoch", yaxis_title="Loss",
                                          height=330, legend=dict(x=0.65, y=0.9)))
        st.plotly_chart(fig, use_container_width=True)
 
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=ann_history["accuracy"],
                                  name="Train Acc", line=dict(color="#81c784", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=ann_history["val_accuracy"],
                                  name="Val Acc", line=dict(color="#ffb74d", width=2, dash="dash")))
        fig.update_layout(**_base_layout(title="Accuracy Curve",
                                          xaxis_title="Epoch", yaxis_title="Accuracy",
                                          height=330, legend=dict(x=0.1, y=0.3)))
        st.plotly_chart(fig, use_container_width=True)
 
    st.caption(f"Model berhenti training pada epoch **{len(epochs)}** (EarlyStopping).")
 
    # ── Evaluasi ──
    st.markdown("---")
    st.subheader("📊 Evaluasi Model ANN (Test Set)")
 
    st.info(
        f"🎯 **Threshold Optimal (Youden's J):** `{ann_threshold:.4f}`  \n"
        "Threshold dihitung dari validation set — bukan 0.5 default — "
        "agar model tidak bias ke satu kelas."
    )
 
    y_prob_ann = ann_model.predict(X_test_sc, verbose=0).flatten()
    y_pred_ann = (y_prob_ann >= ann_threshold).astype(int)
    m_ann, cm_ann, fpr_ann, tpr_ann, auc_ann = evaluate_model(
        y_test, y_pred_ann, y_prob_ann, "ANN"
    )
 
    cols = st.columns(5)
    for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]):
        col.metric(key, f"{m_ann[key]:.4f}")
 
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        st.plotly_chart(
            fig_confusion_matrix(cm_ann, "Confusion Matrix — ANN", colorscale="Blues"),
            use_container_width=True
        )
    with c2:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_ann, y=tpr_ann,
                                  name=f"ANN  (AUC={auc_ann})",
                                  line=dict(color="#4fc3f7", width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(**_base_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=350, legend=dict(x=0.55, y=0.1)
        ))
        st.plotly_chart(fig, use_container_width=True)
 
    st.subheader("📋 Classification Report")
    rpt = classification_report(y_test, y_pred_ann,
                                  target_names=["Sehat (0)", "Sakit (1)"],
                                  output_dict=True)
    st.dataframe(pd.DataFrame(rpt).T.round(4), use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────
# [5.6]  Halaman Model SVM
# ─────────────────────────────────────────────────────────────
def page_svm(svm_model, best_params: dict, cv_score: float,
              X_test_sc, y_test, svm_threshold: float):
    st.title("⚙️ Model Support Vector Machine (SVM)")
    st.markdown("---")
 
    # ── Konfigurasi ──
    st.subheader("🔧 Hyperparameter Tuning (GridSearchCV)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Grid Parameter yang Dicoba**
- C      : [0.1, 1, 10, 100]
- kernel : [rbf, linear]
- gamma  : [scale, auto]
        """)
    with c2:
        st.markdown(f"""
**Strategi Pencarian**
- Metode       : GridSearchCV
- CV           : 5-Fold Cross Validation
- Scoring      : F1-Score (lebih adil dari Accuracy)
- Class Weight : Balanced
- CV Score Terbaik : **`{cv_score}`**
        """)
    with c3:
        st.markdown("**✅ Parameter Terbaik yang Dipilih**")
        for k, v in best_params.items():
            st.markdown(f"- **{k}** : `{v}`")
 
    # ── Evaluasi ──
    st.markdown("---")
    st.subheader("📊 Evaluasi Model SVM (Test Set)")
 
    st.info(
        f"🎯 **Threshold Optimal (Youden's J):** `{svm_threshold:.4f}`  \n"
        "Threshold dihitung dari validation set untuk keseimbangan "
        "Sensitivity dan Specificity yang optimal."
    )
 
    y_prob_svm = svm_model.predict_proba(X_test_sc)[:, 1]
    y_pred_svm = (y_prob_svm >= svm_threshold).astype(int)
    m_svm, cm_svm, fpr_svm, tpr_svm, auc_svm = evaluate_model(
        y_test, y_pred_svm, y_prob_svm, "SVM"
    )
 
    cols = st.columns(5)
    for col, key in zip(cols, ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]):
        col.metric(key, f"{m_svm[key]:.4f}")
 
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        st.plotly_chart(
            fig_confusion_matrix(cm_svm, "Confusion Matrix — SVM", colorscale="RdPu"),
            use_container_width=True
        )
    with c2:
        st.subheader("ROC Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr_svm, y=tpr_svm,
                                  name=f"SVM  (AUC={auc_svm})",
                                  line=dict(color="#f48fb1", width=2.5)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                  line=dict(color="gray", dash="dash")))
        fig.update_layout(**_base_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=350, legend=dict(x=0.55, y=0.1)
        ))
        st.plotly_chart(fig, use_container_width=True)
 
    # ── Support Vectors ──
    st.markdown("---")
    st.subheader("📌 Informasi Support Vectors")
    sv_total = int(svm_model.n_support_.sum())
    sv_0     = int(svm_model.n_support_[0])
    sv_1     = int(svm_model.n_support_[1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Support Vectors",  sv_total)
    c2.metric("SV Kelas 0 (Sehat)",     sv_0)
    c3.metric("SV Kelas 1 (Sakit)",     sv_1)
    c4.metric("Rasio SV / Sampel Train", f"{sv_total/len(y_test)*5:.1%}")
 
    st.subheader("📋 Classification Report")
    rpt = classification_report(y_test, y_pred_svm,
                                  target_names=["Sehat (0)", "Sakit (1)"],
                                  output_dict=True)
    st.dataframe(pd.DataFrame(rpt).T.round(4), use_container_width=True)
 
 
# ─────────────────────────────────────────────────────────────
# [5.7]  Halaman Perbandingan Model
# ─────────────────────────────────────────────────────────────
def page_perbandingan(ann_model, svm_model, ann_history: dict,
                       X_test_sc, y_test,
                       ann_threshold: float, svm_threshold: float):
    st.title("📈 Perbandingan Model: ANN vs SVM")
    st.markdown("---")
 
    # Prediksi dengan threshold optimal masing-masing
    y_prob_ann = ann_model.predict(X_test_sc, verbose=0).flatten()
    y_pred_ann = (y_prob_ann >= ann_threshold).astype(int)
    y_prob_svm = svm_model.predict_proba(X_test_sc)[:, 1]
    y_pred_svm = (y_prob_svm >= svm_threshold).astype(int)
 
    m_ann, cm_ann, fpr_ann, tpr_ann, auc_ann = evaluate_model(
        y_test, y_pred_ann, y_prob_ann, "ANN")
    m_svm, cm_svm, fpr_svm, tpr_svm, auc_svm = evaluate_model(
        y_test, y_pred_svm, y_prob_svm, "SVM")
 
    METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
 
    # ── Tabel Perbandingan ──
    st.subheader("📋 Tabel Perbandingan Metrik")
    df_cmp = pd.DataFrame([m_ann, m_svm]).set_index("Model")
 
    def _highlight(s):
        best = s == s.max()
        return [
            "background-color:#1b5e20; color:#a5d6a7; font-weight:bold" if v else ""
            for v in best
        ]
 
    st.dataframe(df_cmp.style.apply(_highlight), use_container_width=True)
 
    st.markdown("---")
    c1, c2 = st.columns(2)
 
    # ── ROC Curve ──
    with c1:
        st.subheader("📉 ROC Curve Perbandingan")
        st.plotly_chart(
            fig_roc_comparison(fpr_ann, tpr_ann, auc_ann,
                                fpr_svm, tpr_svm, auc_svm),
            use_container_width=True
        )
 
    # ── Bar Chart Metrik ──
    with c2:
        st.subheader("📊 Bar Chart Metrik")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="ANN", x=METRICS,
            y=[m_ann[m] for m in METRICS],
            marker_color="#4fc3f7",
            text=[f"{m_ann[m]:.4f}" for m in METRICS],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="SVM", x=METRICS,
            y=[m_svm[m] for m in METRICS],
            marker_color="#f48fb1",
            text=[f"{m_svm[m]:.4f}" for m in METRICS],
            textposition="outside",
        ))
        fig.update_layout(**_base_layout(
            barmode="group", height=420,
            yaxis=dict(range=[0, 1.12]),
            legend=dict(x=0.8, y=0.98),
        ))
        st.plotly_chart(fig, use_container_width=True)
 
    # ── Radar Chart ──
    st.subheader("🕸️ Radar Chart — Profil Model")
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[m_ann[m] for m in METRICS] + [m_ann[METRICS[0]]],
        theta=METRICS + [METRICS[0]],
        fill="toself", name="ANN",
        line_color="#4fc3f7", fillcolor="rgba(79,195,247,0.18)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[m_svm[m] for m in METRICS] + [m_svm[METRICS[0]]],
        theta=METRICS + [METRICS[0]],
        fill="toself", name="SVM",
        line_color="#f48fb1", fillcolor="rgba(244,143,177,0.18)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.6, 1.03], color="white",
                            tickfont=dict(size=10)),
            bgcolor=PLOT_BG,
        ),
        paper_bgcolor=PAPER_BG, font_color=FONT_COL,
        height=480, legend=dict(x=1.02, y=0.9),
    )
    st.plotly_chart(fig, use_container_width=True)
 
    # ── Kesimpulan ──
    st.markdown("---")
    st.subheader("💡 Kesimpulan Komparatif")
 
    def _winner(a, b, label_a="ANN", label_b="SVM"):
        return f"**{label_a}**" if a > b else f"**{label_b}**"
 
    md_rows = "\n".join([
        f"| {m} | {m_ann[m]:.4f} | {m_svm[m]:.4f} | {_winner(m_ann[m], m_svm[m])} |"
        for m in METRICS
    ])
    st.markdown(
        "| Metrik | ANN | SVM | Pemenang |\n"
        "|--------|-----|-----|----------|\n"
        + md_rows
        + f"\n| Epochs Training | {len(ann_history['loss'])} | GridSearchCV | — |"
        + "\n| Kompleksitas | Tinggi | Menengah | SVM |"
    )
    st.info(
        "📌 Perhatikan metrik **Recall** (sensitifitas) untuk penyakit jantung, "
        "karena false negative (sakit tapi diprediksi sehat) lebih berisiko "
        "dibanding false positive."
    )
 
 
# ─────────────────────────────────────────────────────────────
# [5.8]  Halaman Prediksi Pasien
# ─────────────────────────────────────────────────────────────
def page_prediksi(ann_model, svm_model, scaler: StandardScaler,
                   ann_threshold: float, svm_threshold: float):
    st.title("🔮 Prediksi Risiko Penyakit Jantung")
    st.markdown(
        "Masukkan data klinis pasien untuk mendapatkan prediksi real-time "
        "dari kedua model secara bersamaan."
    )
    st.markdown("---")
 
    c1, c2, c3 = st.columns(3)
 
    with c1:
        st.subheader("👤 Data Demografi")
        age  = st.slider("Usia (tahun)", 20, 80, 50)
        sex  = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        cp   = st.selectbox(
            "Tipe Nyeri Dada",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Typical Angina",
                1: "1 — Atypical Angina",
                2: "2 — Non-Anginal Pain",
                3: "3 — Asymptomatic",
            }[x]
        )
        fbs  = st.selectbox("Gula Darah Puasa > 120 mg/dl", [0, 1],
                             format_func=lambda x: f"{x} — {'Ya' if x else 'Tidak'}")
 
    with c2:
        st.subheader("🩺 Pemeriksaan Klinis")
        trestbps = st.slider("Tekanan Darah Istirahat (mm Hg)", 80, 200, 120)
        chol     = st.slider("Kolesterol Serum (mg/dl)", 100, 600, 240)
        restecg  = st.selectbox(
            "Hasil EKG Istirahat",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 — Normal",
                1: "1 — Abnormal ST-T",
                2: "2 — LV Hypertrophy",
            }[x]
        )
        thalach  = st.slider("Detak Jantung Maks", 60, 220, 150)
 
    with c3:
        st.subheader("🏃 Data Uji Olahraga")
        exang   = st.selectbox("Angina akibat Olahraga", [0, 1],
                                format_func=lambda x: f"{x} — {'Ya' if x else 'Tidak'}")
        oldpeak = st.slider("Depresi ST (Oldpeak)", 0.0, 6.2, 1.0, step=0.1)
        slope   = st.selectbox(
            "Kemiringan ST Puncak",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 — Downsloping", 1: "1 — Flat", 2: "2 — Upsloping"}[x]
        )
        ca   = st.selectbox("Jumlah Pembuluh Darah Utama (0–3)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 — Normal", 1: "1 — Fixed Defect", 2: "2 — Reversible Defect"}[x]
        )
 
    st.markdown("---")
 
    if st.button("🔍 Prediksi Sekarang", use_container_width=True):
        sex_val    = 1 if sex == "Male" else 0
        input_arr  = np.array([[age, sex_val, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak, slope, ca, thal]])
        input_sc   = scaler.transform(input_arr)
 
        # Prediksi ANN
        prob_ann = float(ann_model.predict(input_sc, verbose=0)[0][0])
        pred_ann = int(prob_ann >= 0.5)
 
        # Prediksi SVM
        prob_svm = float(svm_model.predict_proba(input_sc)[0][1])
        pred_svm = int(svm_model.predict(input_sc)[0])
 
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")
        c1, c2 = st.columns(2)
 
        for col, prob, pred, label in [
            (c1, prob_ann, pred_ann, "🧠 ANN"),
            (c2, prob_svm, pred_svm, "⚙️ SVM"),
        ]:
            with col:
                st.markdown(f"### {label}")
                status = "⚠️ BERISIKO PENYAKIT JANTUNG" if pred == 1 else "✅ SEHAT"
                color  = "#f06292" if pred == 1 else "#4fc3f7"
                st.markdown(f"""
                <div class="result-card" style="border: 2px solid {color};">
                    <h2 style="color:{color}; margin:0;">{status}</h2>
                    <p style="color:#aaa; font-size:17px; margin-top:12px;">
                        Probabilitas Sakit Jantung:<br>
                        <strong style="color:{color}; font-size:1.8em;">{prob:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Tingkat Risiko:**")
                st.progress(float(prob))
 
        # Ringkasan
        st.markdown("---")
        st.subheader("📌 Ringkasan")
        avg_prob = (prob_ann + prob_svm) / 2
        c1, c2, c3 = st.columns(3)
        c1.metric("Prob. ANN",           f"{prob_ann:.1%}")
        c2.metric("Prob. SVM",           f"{prob_svm:.1%}")
        c3.metric("Rata-rata Probabilitas", f"{avg_prob:.1%}")
 
        agree = pred_ann == pred_svm
        if agree:
            st.success("✅ Kedua model **sepakat** dalam diagnosis pasien ini.")
        else:
            st.warning("⚠️ Kedua model **tidak sepakat** — pertimbangkan pengujian lebih lanjut.")
 
        if avg_prob > 0.65:
            st.error("🚨 Risiko **tinggi** — pasien sangat disarankan melakukan pemeriksaan ke dokter spesialis jantung.")
        elif avg_prob > 0.4:
            st.warning("⚠️ Risiko **sedang** — pantau kondisi dan konsultasikan ke dokter.")
        else:
            st.success("✅ Risiko **rendah** — tetap jaga pola hidup sehat.")
 
        st.caption(
            "⚠️ *Disclaimer: Prediksi ini hanya untuk keperluan akademis dan "
            "tidak menggantikan diagnosis medis dari tenaga kesehatan profesional.*"
        )
 
 
# ══════════════════════════════════════════════════════════════════
# [6]  MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════
 
def main():
    menu = render_sidebar()
 
    # ── Load dataset ──
    df = load_data()
    if df is None:
        st.error("""
        ❌ **File dataset tidak ditemukan!**
 
        Upload `heart__1_.xlsx` ke Google Colab terlebih dahulu:
        ```python
        from google.colab import files
        files.upload()   # pilih file heart__1_.xlsx
        ```
        Kemudian refresh halaman ini.
        """)
        st.stop()
 
    # ── Preprocessing ──
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     X_train_sc, X_val_sc, X_test_sc, scaler) = preprocess(df)
 
    # ── Training model (caching via session_state) ──
    if "models_ready" not in st.session_state:
        with st.spinner("⏳ Melatih Model ANN & SVM… harap tunggu sebentar…"):
            # ANN — gunakan val set untuk EarlyStopping
            ann_model, ann_history = train_ann(X_train_sc, y_train,
                                                X_val_sc, y_val)
            # SVM
            svm_model, best_params, cv_score = train_svm(X_train_sc, y_train)
 
        st.session_state.ann_model   = ann_model
        st.session_state.ann_history = ann_history
        st.session_state.svm_model   = svm_model
        st.session_state.best_params = best_params
        st.session_state.cv_score    = cv_score
        st.session_state.models_ready = True
 
    # Ambil model dari session_state
    ann_model   = st.session_state.ann_model
    ann_history = st.session_state.ann_history
    svm_model   = st.session_state.svm_model
    best_params = st.session_state.best_params
    cv_score    = st.session_state.cv_score
 
    # ── Routing halaman ──
    if   menu == "🏠  Beranda":
        page_beranda(df)
    elif menu == "📊  EDA":
        page_eda(df)
    elif menu == "🧠  Model ANN":
        page_ann(ann_model, ann_history, X_test_sc, y_test)
    elif menu == "⚙️  Model SVM":
        page_svm(svm_model, best_params, cv_score, X_test_sc, y_test)
    elif menu == "📈  Perbandingan Model":
        page_perbandingan(ann_model, svm_model, ann_history, X_test_sc, y_test)
    elif menu == "🔮  Prediksi Pasien":
        page_prediksi(ann_model, svm_model, scaler)
 
 
if __name__ == "__main__":
    main()
 


