import streamlit as st
import pickle
import numpy as np
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Prediksi Produksi Daging Unggas",
    page_icon="🐔",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================
with open("regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# PREMIUM CSS
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(180deg, #F9FAFB, #F3F4F6);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #111827;
}

.sub-title {
    font-size: 17px;
    color: #6B7280;
}

.card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
            
.card h3 {
    color: #000000;
    font-weight: 700;
}

.gradient-card {
    background: linear-gradient(135deg, #22C55E, #16A34A);
    padding: 30px;
    border-radius: 22px;
    color: white;
    box-shadow: 0 15px 30px rgba(34,197,94,0.4);
}

.metric-big {
    font-size: 46px;
    font-weight: 800;
}

.label {
    font-size: 14px;
    opacity: 0.9;
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
        Sistem prediksi berbasis Machine Learning untuk memperkirakan 
        total produksi daging unggas secara cepat dan akurat.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Parameter Populasi")
st.sidebar.caption("Geser slider untuk menyesuaikan populasi ternak")

ayam_kampung = st.sidebar.slider("🐓 Ayam Kampung", 0, 250000, 10000, 1000)
ayam_petelur = st.sidebar.slider("🥚 Ayam Petelur", 0, 300000, 50000, 5000)
ayam_pedaging = st.sidebar.slider("🍗 Ayam Pedaging", 0, 300000, 30000, 3000)
itik = st.sidebar.slider("🦆 Itik", 0, 150000, 8000, 1000)

# =========================
# SUMMARY
# =========================
st.markdown("<div class='card'><h3>📊 Ringkasan Populasi</h3></div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ayam Kampung", f"{ayam_kampung:,} ekor")
c2.metric("Ayam Petelur", f"{ayam_petelur:,} ekor")
c3.metric("Ayam Pedaging", f"{ayam_pedaging:,} ekor")
c4.metric("Itik", f"{itik:,} ekor")

# =========================
# PREDICTION
# =========================
st.markdown("---")
if st.button("🚀 Jalankan Prediksi Produksi", use_container_width=True):
    input_data = np.array([[
        ayam_kampung,
        ayam_pedaging,
        ayam_petelur,
        itik
    ]])

    prediction = model.predict(input_data)

    st.markdown(f"""
    <div class="gradient-card">
        <div class="label">Perkiraan Total Produksi Daging Unggas</div>
        <div class="metric-big">{prediction[0]:,.0f} kg</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# DATASET
# =========================
with st.expander("📁 Lihat Dataset Contoh"):
    df = pd.read_csv("dataset_final_unggas.csv")
    st.dataframe(df.head())
