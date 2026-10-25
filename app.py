import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# =========================================================
# KONFIGURASI
# =========================================================
st.set_page_config(
    page_title="Dashboard Klasifikasi Cuaca Kolkata",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

FEATURE_COLS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "sunshine_duration",
]
TARGET_COL = "weather_code"

FEATURE_LABELS = {
    "temperature_2m_mean": "Suhu rata-rata (°C)",
    "temperature_2m_max": "Suhu maksimum (°C)",
    "temperature_2m_min": "Suhu minimum (°C)",
    "precipitation_sum": "Total presipitasi (mm)",
    "rain_sum": "Total hujan (mm)",
    "precipitation_hours": "Durasi presipitasi (jam)",
    "wind_speed_10m_max": "Kecepatan angin maksimum (km/jam)",
    "wind_gusts_10m_max": "Hembusan angin maksimum (km/jam)",
    "wind_direction_10m_dominant": "Arah angin dominan (°)",
    "sunshine_duration": "Durasi sinar matahari (detik)",
}

# Keterangan ringkas kode cuaca WMO yang muncul pada dataset.
WEATHER_LABELS = {
    0: "Cerah",
    1: "Cerah sebagian",
    2: "Berawan sebagian",
    3: "Mendung",
    45: "Berkabut",
    48: "Kabut beku",
    51: "Gerimis ringan",
    53: "Gerimis sedang",
    55: "Gerimis lebat",
    56: "Gerimis beku ringan",
    57: "Gerimis beku lebat",
    61: "Hujan ringan",
    63: "Hujan sedang",
    65: "Hujan lebat",
    66: "Hujan beku ringan",
    67: "Hujan beku lebat",
    71: "Salju ringan",
    73: "Salju sedang",
    75: "Salju lebat",
    77: "Butiran salju",
    80: "Hujan singkat ringan",
    81: "Hujan singkat sedang",
    82: "Hujan singkat lebat",
    85: "Hujan salju singkat ringan",
    86: "Hujan salju singkat lebat",
    95: "Badai petir",
    96: "Badai petir dengan hujan es ringan",
    99: "Badai petir dengan hujan es lebat",
}

# =========================================================
# CSS / TAMPILAN DASHBOARD
# =========================================================
st.markdown("""
<style>
    .stApp { background: #fbfbfe; }
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1500px; }
    .hero {
        background: linear-gradient(110deg, #5146ef 0%, #8338ec 55%, #e640a1 100%);
        padding: 25px 32px; border-radius: 0 0 22px 22px; color: white;
        margin-bottom: 18px; box-shadow: 0 8px 22px rgba(96, 65, 220, .12);
    }
    .hero h1 { color: white; font-size: 30px; font-weight: 800; margin: 0 0 12px 0; }
    .hero p { color: #f5f0ff; margin: 0; font-size: 15px; }
    .metric-card {
        background: white; border-radius: 17px; padding: 17px 20px;
        min-height: 105px; border-left: 6px solid #5146ef;
        box-shadow: 0 3px 14px rgba(25, 30, 70, .06);
        margin-bottom: 8px;
    }
    .metric-label { color: #858594; font-size: 11px; letter-spacing: .8px; text-transform: uppercase; }
    .metric-value { color: #292943; font-size: 27px; font-weight: 800; margin-top: 6px; }
    .section-title { color: #292943; font-size: 25px; font-weight: 800; margin: 10px 0 15px 0; }
    .insight {
        background: white; border-left: 5px solid #e640a1; padding: 15px 18px;
        border-radius: 12px; margin: 10px 0; box-shadow: 0 2px 10px rgba(25,30,70,.04);
        color: #37374d;
    }
    .small-note { color: #77778a; font-size: 12px; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    div[data-testid="stMetric"] {
        background: white; border-radius: 14px; padding: 12px 16px;
        border: 1px solid #efeff7;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA & TRAIN MODEL
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "kolkata_weather_jan_june_2026.csv"

@st.cache_data
def load_dataset(path, modified_time):
    data = pd.read_csv(path)
    needed = FEATURE_COLS + [TARGET_COL]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError("Kolom tidak ditemukan: " + ", ".join(missing))

    for col in FEATURE_COLS + [TARGET_COL]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    return data

@st.cache_resource
def train_model(csv_path, modified_time):
    data = load_dataset(csv_path, modified_time)
    X = data[FEATURE_COLS]
    y_original = data[TARGET_COL].astype(int)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    try:
        from sklearn.metrics import roc_auc_score
        metrics["roc_auc"] = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )
    except Exception:
        metrics["roc_auc"] = None

    return data, model, encoder, X_test, y_test, y_pred, metrics

if not CSV_PATH.exists():
    st.markdown('<div class="hero"><h1>🌦️ Dashboard Klasifikasi Cuaca Kolkata</h1><p>Analisis data cuaca menggunakan algoritma XGBoost</p></div>', unsafe_allow_html=True)
    st.error("File dataset belum ditemukan. Letakkan `kolkata_weather_jan_june_2026.csv` di folder yang sama dengan `app.py`.")
    st.info(f"Folder yang diperiksa: {BASE_DIR}")
    st.stop()

try:
    df, model, label_encoder, X_test, y_test, y_pred, metrics = train_model(
        str(CSV_PATH), CSV_PATH.stat().st_mtime
    )
except Exception as e:
    st.error(f"Dataset/model gagal diproses: {e}")
    st.stop()

# =========================================================
# HEADER & NAVIGATION
# =========================================================
st.markdown("""
<div class="hero">
  <h1>🌦️ Dashboard Klasifikasi Cuaca Kolkata</h1>
  <p>Hasil analisis dataset cuaca dengan <b>XGBoost</b> · Eksplorasi Data · Evaluasi Model · Fitur Penting · Prediksi Cuaca</p>
</div>
""", unsafe_allow_html=True)

tab_names = [
    "🏠 Ringkasan",
    "📊 Eksplorasi Data",
    "🌡️ Analisis Cuaca",
    "🎯 Evaluasi Model",
    "⭐ Fitur Penting",
    "🔮 Prediksi",
    "📘 Metodologi"
]
tabs = st.tabs(tab_names)

# =========================================================
# TAB 1: RINGKASAN
# =========================================================
with tabs[0]:
    st.markdown('<div class="section-title">💡 Ringkasan Dataset</div>', unsafe_allow_html=True)
    avg_temp = df["temperature_2m_mean"].mean()
    avg_rain = df["precipitation_sum"].mean()
    avg_wind = df["wind_speed_10m_max"].mean()
    common_code = int(df[TARGET_COL].mode().iloc[0])
    common_label = WEATHER_LABELS.get(common_code, "Kode cuaca " + str(common_code))

    metric_rows = [
        ("TOTAL DATA", f"{len(df):,}", "#5146ef"),
        ("JUMLAH FITUR", str(len(FEATURE_COLS)), "#08b987"),
        ("KELAS CUACA", str(df[TARGET_COL].nunique()), "#ff9800"),
        ("AKURASI MODEL", f"{metrics['accuracy']*100:.1f}%", "#e640a1"),
    ]
    cards = st.columns(4)
    for col, (label, value, color) in zip(cards, metric_rows):
        col.markdown(
            f'<div class="metric-card" style="border-left-color:{color}">'
            f'<div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
            unsafe_allow_html=True
        )

    second_cards = st.columns(4)
    secondary = [
        ("SUHU RATA-RATA", f"{avg_temp:.1f} °C", "#5146ef"),
        ("RATA-RATA HUJAN", f"{avg_rain:.1f} mm", "#ff9800"),
        ("ANGIN MAKS. RATA-RATA", f"{avg_wind:.1f} km/jam", "#08b987"),
        ("KODE CUACA TERBANYAK", f"{common_code} · {common_label}", "#e640a1"),
    ]
    for col, (label, value, color) in zip(second_cards, secondary):
        col.markdown(
            f'<div class="metric-card" style="border-left-color:{color}">'
            f'<div class="metric-label">{label}</div><div class="metric-value" style="font-size:21px">{value}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">💡 Temuan Utama</div>', unsafe_allow_html=True)
    top_code_counts = df[TARGET_COL].value_counts()
    top_code = int(top_code_counts.index[0])
    top_share = top_code_counts.iloc[0] / len(df) * 100
    st.markdown(
        f'<div class="insight">Kode cuaca yang paling sering muncul adalah <b>{top_code} ({WEATHER_LABELS.get(top_code, "Kode cuaca " + str(top_code))})</b>, '
        f'dengan proporsi sekitar <b>{top_share:.1f}%</b> dari data.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="insight">Rata-rata suhu tercatat <b>{avg_temp:.1f} °C</b>, sedangkan rata-rata presipitasi harian adalah <b>{avg_rain:.1f} mm</b>.</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="insight">Model XGBoost memperoleh akurasi <b>{metrics["accuracy"]*100:.1f}%</b> pada data uji. '
        'Nilai ini menggambarkan hasil pengujian pada pembagian data yang digunakan aplikasi.</div>',
        unsafe_allow_html=True
    )
    st.caption("Catatan: kode cuaca mengikuti kolom weather_code pada dataset; label merupakan keterangan ringkas kode cuaca.")

# =========================================================
# TAB 2: EKSPLORASI DATA
# =========================================================
with tabs[1]:
    st.markdown('<div class="section-title">📊 Eksplorasi Data</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Kode Cuaca")
        counts = df[TARGET_COL].astype(int).value_counts().sort_index().rename_axis("weather_code").reset_index(name="Jumlah")
        counts["Keterangan"] = counts["weather_code"].map(lambda x: WEATHER_LABELS.get(int(x), f"Kode {x}"))
        fig = px.bar(counts, x="weather_code", y="Jumlah", hover_data=["Keterangan"],
                     color="Jumlah", color_continuous_scale="Purples",
                     title="Jumlah data per weather_code")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Distribusi Suhu Rata-rata")
        fig = px.histogram(df, x="temperature_2m_mean", nbins=25,
                           color_discrete_sequence=["#7c3aed"],
                           labels={"temperature_2m_mean": "Suhu rata-rata (°C)"},
                           title="Sebaran suhu rata-rata")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Dataset")
    search = st.text_input("Cari pada tabel (tanggal/kota/kode cuaca)", "")
    view_df = df.copy()
    if "time" in view_df.columns and search:
        mask = view_df.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        view_df = view_df[mask]
    st.dataframe(view_df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Unduh data yang ditampilkan sebagai CSV",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="data_cuaca_kolkata.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 3: ANALISIS CUACA
# =========================================================
with tabs[2]:
    st.markdown('<div class="section-title">🌡️ Analisis Cuaca</div>', unsafe_allow_html=True)
    if "time" in df.columns:
        daily = df.copy()
        daily["time"] = pd.to_datetime(daily["time"], errors="coerce")
        daily = daily.dropna(subset=["time"]).sort_values("time")
        daily["Bulan"] = daily["time"].dt.to_period("M").astype(str)
        monthly = daily.groupby("Bulan", as_index=False).agg(
            Suhu=("temperature_2m_mean", "mean"),
            Presipitasi=("precipitation_sum", "sum"),
            Kecepatan_Angin=("wind_speed_10m_max", "mean")
        )
        measure = st.selectbox("Pilih parameter untuk dianalisis", ["Suhu", "Presipitasi", "Kecepatan_Angin"])
        y_label = {"Suhu": "Suhu rata-rata (°C)", "Presipitasi": "Total presipitasi (mm)", "Kecepatan_Angin": "Kecepatan angin (km/jam)"}[measure]
        fig = px.line(monthly, x="Bulan", y=measure, markers=True, title=f"Tren bulanan: {y_label}",
                      color_discrete_sequence=["#7c3aed"])
        st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Suhu Minimum, Rata-rata, dan Maksimum")
        temp_summary = pd.DataFrame({
            "Jenis Suhu": ["Minimum", "Rata-rata", "Maksimum"],
            "Suhu (°C)": [
                df["temperature_2m_min"].mean(),
                df["temperature_2m_mean"].mean(),
                df["temperature_2m_max"].mean()
            ]
        })
        fig = px.bar(temp_summary, x="Jenis Suhu", y="Suhu (°C)", color="Jenis Suhu",
                     color_discrete_sequence=["#8b5cf6", "#ec4899", "#f59e0b"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Presipitasi dan Durasi Hujan")
        fig = px.scatter(df, x="precipitation_hours", y="precipitation_sum",
                         color="weather_code", hover_data=["time"],
                         labels={"precipitation_hours": "Durasi presipitasi (jam)",
                                 "precipitation_sum": "Presipitasi (mm)", "weather_code": "Kode cuaca"},
                         title="Hubungan durasi dan jumlah presipitasi")
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4: EVALUASI MODEL
# =========================================================
with tabs[3]:
    st.markdown('<div class="section-title">🎯 Evaluasi Model XGBoost</div>', unsafe_allow_html=True)
    roc_auc = metrics["roc_auc"]
    metric_values = [
        ("ACCURACY", metrics["accuracy"], "#5146ef"),
        ("PRECISION MACRO", metrics["precision"], "#08b987"),
        ("RECALL MACRO", metrics["recall"], "#ff9800"),
        ("F1-SCORE MACRO", metrics["f1"], "#e640a1"),
    ]
    cols = st.columns(4)
    for col, (name, value, color) in zip(cols, metric_values):
        col.markdown(
            f'<div class="metric-card" style="border-left-color:{color}">'
            f'<div class="metric-label">{name}</div><div class="metric-value">{value*100:.1f}%</div></div>',
            unsafe_allow_html=True
        )
    if roc_auc is not None:
        st.markdown(
            f'<div class="metric-card" style="border-left-color:#06b6d4">'
            f'<div class="metric-label">ROC-AUC MACRO (OVR)</div>'
            f'<div class="metric-value">{roc_auc:.3f}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="insight">Metrik dihitung menggunakan data uji sebesar 20% dari dataset. Precision, recall, dan F1-score menggunakan rata-rata macro agar setiap kelas dihitung dengan bobot yang sama.</div>', unsafe_allow_html=True)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    class_names = [str(x) for x in label_encoder.classes_]
    fig = px.imshow(cm, text_auto=True, x=class_names, y=class_names,
                    labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"},
                    color_continuous_scale="Purples", aspect="auto")
    fig.update_layout(xaxis_title="Kelas Prediksi", yaxis_title="Kelas Aktual")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat classification report"):
        report = classification_report(
            y_test, y_pred, labels=np.arange(len(label_encoder.classes_)),
            target_names=class_names, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# =========================================================
# TAB 5: FITUR PENTING
# =========================================================
with tabs[4]:
    st.markdown('<div class="section-title">⭐ Fitur Penting Model</div>', unsafe_allow_html=True)
    importance = pd.DataFrame({
        "Fitur": FEATURE_COLS,
        "Kepentingan": model.feature_importances_
    }).sort_values("Kepentingan", ascending=True)
    importance["Nama fitur"] = importance["Fitur"].map(FEATURE_LABELS)
    fig = px.bar(importance, x="Kepentingan", y="Nama fitur", orientation="h",
                 color="Kepentingan", color_continuous_scale="Purples",
                 title="Feature importance XGBoost")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Nilai feature importance")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Feature importance menunjukkan kontribusi relatif fitur terhadap keputusan model, bukan hubungan sebab-akibat.")
    st.dataframe(importance.sort_values("Kepentingan", ascending=False)[["Nama fitur", "Kepentingan"]].reset_index(drop=True),
                 use_container_width=True, hide_index=True)

# =========================================================
# TAB 6: PREDIKSI
# =========================================================
with tabs[5]:
    st.markdown('<div class="section-title">🔮 Prediksi Cuaca</div>', unsafe_allow_html=True)
    st.write("Masukkan nilai parameter cuaca untuk memperkirakan kelas weather_code.")
    st.info("Nilai awal setiap input diatur berdasarkan median dataset. Kamu dapat mengubahnya sesuai skenario yang ingin diuji.")

    with st.form("prediction_form"):
        inputs = {}
        left, right = st.columns(2)
        for i, feature in enumerate(FEATURE_COLS):
            target_col = left if i < 5 else right
            series = df[feature].dropna()
            min_value = float(series.min())
            max_value = float(series.max())
            median_value = float(series.median())
            # Hindari min_value dan max_value yang sama karena number_input membutuhkan rentang.
            if min_value == max_value:
                inputs[feature] = target_col.number_input(
                    FEATURE_LABELS[feature], value=median_value, format="%.3f"
                )
            else:
                inputs[feature] = target_col.number_input(
                    FEATURE_LABELS[feature],
                    min_value=min_value,
                    max_value=max_value,
                    value=float(np.clip(median_value, min_value, max_value)),
                    format="%.3f"
                )
        submitted = st.form_submit_button("🔍 Prediksi Cuaca", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([inputs], columns=FEATURE_COLS)
        pred_encoded = int(model.predict(input_df)[0])
        pred_code = int(label_encoder.inverse_transform([pred_encoded])[0])
        pred_label = WEATHER_LABELS.get(pred_code, f"Kode cuaca {pred_code}")
        probs = model.predict_proba(input_df)[0]
        prob_df = pd.DataFrame({
            "weather_code": label_encoder.classes_.astype(int),
            "Probabilitas (%)": probs * 100
        }).sort_values("Probabilitas (%)", ascending=False)
        st.markdown(
            f'<div class="hero"><h1>🌤️ Hasil Prediksi: {pred_label}</h1>'
            f'<p>weather_code = <b>{pred_code}</b></p></div>',
            unsafe_allow_html=True
        )
        st.subheader("Probabilitas prediksi per kelas")
        prob_df["Keterangan"] = prob_df["weather_code"].map(
            lambda x: WEATHER_LABELS.get(int(x), f"Kode {x}")
        )
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        chart_df = prob_df.head(10).sort_values("Probabilitas (%)")
        fig = px.bar(chart_df, x="Probabilitas (%)", y="Keterangan", orientation="h",
                     color="Probabilitas (%)", color_continuous_scale="Purples")
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Lihat parameter input"):
            st.dataframe(input_df.rename(columns=FEATURE_LABELS), use_container_width=True)

# =========================================================
# TAB 7: METODOLOGI
# =========================================================
with tabs[6]:
    st.markdown('<div class="section-title">📘 Metodologi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight"><b>1. Pengumpulan data</b><br>Dataset cuaca Kolkata dibaca dari file CSV.</div>
    <div class="insight"><b>2. Pra-pemrosesan</b><br>Kolom fitur dan target dipilih, data numerik dikonversi, lalu baris yang memiliki nilai kosong pada kolom terpilih dihapus.</div>
    <div class="insight"><b>3. Pembagian data</b><br>Dataset dibagi menjadi 80% data latih dan 20% data uji dengan random state 42 dan stratifikasi kelas.</div>
    <div class="insight"><b>4. Pelatihan</b><br>Algoritma XGBoost dilatih menggunakan 10 fitur cuaca untuk memprediksi weather_code.</div>
    <div class="insight"><b>5. Evaluasi</b><br>Kinerja diukur menggunakan accuracy, precision macro, recall macro, F1-score macro, confusion matrix, dan ROC-AUC macro jika dapat dihitung.</div>
    """, unsafe_allow_html=True)
    st.subheader("Fitur yang digunakan")
    st.dataframe(pd.DataFrame({
        "Nama kolom": FEATURE_COLS,
        "Keterangan": [FEATURE_LABELS[x] for x in FEATURE_COLS]
    }), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown('<p class="small-note">Dashboard Klasifikasi Cuaca Kolkata · Algoritma XGBoost · Dibuat dengan Streamlit</p>', unsafe_allow_html=True)
