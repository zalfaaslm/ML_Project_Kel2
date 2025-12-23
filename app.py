import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Produksi Daging Unggas di Kota Subang",
    page_icon="🐔",
    layout="wide"
)

# =========================
# LOAD MODEL & DATA
# =========================
with open("regression_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("dataset_final.csv")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #111827; /* HITAM */
}

.sub-title {
    color: #374151; /* abu gelap */
}

.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
    color: #111827; /* <<< INI PENTING */
}

/* Paksa semua teks di card jadi hitam */
.card h1, 
.card h2, 
.card h3, 
.card p, 
.card span, 
.card div {
    color: #111827;
}

.gradient-card {
    background: linear-gradient(135deg, #22C55E, #16A34A);
    padding: 30px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 15px 30px rgba(34,197,94,0.4);
}

.metric-big {
    font-size: 44px;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.markdown("""
<div class="card">
    <div class="main-title">🐔 Prediksi Produksi Daging Unggas</div>
    <div class="sub-title">
        Aplikasi Machine Learning untuk prediksi produksi daging unggas
        dan analisis eksploratif data (EDA).
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🤖 Prediksi", "📊 EDA", "📁 Dataset"])

# =====================================================
# TAB 1 — PREDIKSI
# =====================================================
with tab1:
    st.sidebar.title("⚙️ Input Populasi Unggas")

    ayam_kampung = st.sidebar.slider("🐓 Ayam Kampung", 0, 250000, 10000, 1000)
    ayam_petelor = st.sidebar.slider("🥚 Ayam Petelor", 0, 200000, 15000, 1000)
    ayam_pedaging = st.sidebar.slider("🍗 Ayam Pedaging", 0, 300000, 30000, 3000)
    itik = st.sidebar.slider("🦆 Itik", 0, 150000, 8000, 1000)

    st.markdown("<div class='card'><h3>📊 Ringkasan Input</h3></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ayam Kampung", f"{ayam_kampung:,} ekor")
    c2.metric("Ayam Petelor", f"{ayam_petelor:,} ekor")
    c3.metric("Ayam Pedaging", f"{ayam_pedaging:,} ekor")
    c4.metric("Itik", f"{itik:,} ekor")

    st.markdown("---")

    if st.button("🚀 Jalankan Prediksi Produksi", use_container_width=True):
        input_data = np.array([[ayam_kampung, ayam_petelor ,ayam_pedaging, itik]])
        prediction = model.predict(input_data)

        st.markdown(f"""
        <div class="gradient-card">
            <div>Perkiraan Total Produksi Daging Unggas</div>
            <div class="metric-big">{prediction[0]:,.0f} kg</div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# TAB 2 — EDA
# =====================================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # -------- Line Plot 1 --------
    st.write("### Line Plot: Total Produksi per Tahun")
    fig, ax = plt.subplots()
    df.groupby("tahun")["total_produksi_daging"].sum().plot(ax=ax)
    ax.set_ylabel("Total Produksi")
    st.pyplot(fig)

    # -------- Line Plot 2 --------
    st.write("### Line Plot: Populasi Ayam Pedaging per Tahun")
    fig, ax = plt.subplots()
    df.groupby("tahun")["ayam_pedaging"].sum().plot(ax=ax)
    st.pyplot(fig)

    # -------- Box Plot 1 --------
    st.write("### Box Plot: Ayam Kampung")
    fig, ax = plt.subplots()
    sns.boxplot(y=df["ayam_kampung"], ax=ax)
    st.pyplot(fig)

    # -------- Box Plot 2 --------
    st.write("### Box Plot: Itik")
    fig, ax = plt.subplots()
    sns.boxplot(y=df["itik"], ax=ax)
    st.pyplot(fig)

    # -------- Pie Chart --------
    st.write("### Pie Chart: Komposisi Rata-rata Populasi Unggas")
    avg_pop = df[["ayam_kampung","ayam_pedaging","itik", "ayam_petelor"]].mean()
    fig, ax = plt.subplots()
    ax.pie(avg_pop, labels=avg_pop.index, autopct="%1.1f%%", startangle=90)
    st.pyplot(fig)

    # -------- Scatter Plot --------
    st.write("### Scatter Plot: Ayam Pedaging vs Produksi")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=df["ayam_pedaging"],
        y=df["total_produksi_daging"],
        ax=ax
    )
    st.pyplot(fig)

    # -------- Correlation Matrix --------
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8,6))
    corr = df[["ayam_kampung","ayam_petelor","ayam_pedaging","itik","total_produksi_daging"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# =====================================================
# TAB 3 — DATASET
# =====================================================
with tab3:
    st.subheader("📁 Preview Dataset")

    st.dataframe(df.head())
