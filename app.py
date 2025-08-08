import streamlit as st
import pickle
import numpy as np

# Load model dan scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Prediksi Keberhasilan Transaksi UPI")

# Input user
amount = st.number_input("Jumlah Transaksi (INR)", min_value=1)
hour = st.slider("Jam Transaksi (0-23)", 0, 23, 12)
day = st.selectbox("Hari Transaksi", ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'])

# Map hari ke angka
day_map = {'Senin': 0, 'Selasa': 1, 'Rabu': 2, 'Kamis': 3, 'Jumat': 4, 'Sabtu': 5, 'Minggu': 6}
day_num = day_map[day]

if st.button("Prediksi"):
    data = np.array([[amount, hour, day_num]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)
    
    if pred[0] == 1:
        st.success("✅ Transaksi DIPREDIKSI BERHASIL")
    else:
        st.error("❌ Transaksi DIPREDIKSI GAGAL")
