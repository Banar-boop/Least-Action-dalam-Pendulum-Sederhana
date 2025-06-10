import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pendulum import runge_kutta, calculate_energy_and_action

st.set_page_config(layout="wide")
st.title("Simulasi Pendulum – Prinsip Least Action")

with st.sidebar:
    st.header("Parameter Simulasi")
    theta0 = st.slider("Sudut Awal (derajat)", -90, 90, 30)
    L = st.slider("Panjang Tali (m)", 0.1, 2.0, 1.0)
    m = st.number_input("Massa (kg)", 0.1, 10.0, 1.0)
    tmax = st.slider("Durasi Simulasi (s)", 1, 20, 10)
    dt = 0.01
    g = 9.81

theta0_rad = np.radians(theta0)
t, theta, omega = runge_kutta(theta0_rad, L, m, g, tmax, dt)
T, V, Lagrangian, Action = calculate_energy_and_action(theta, omega, L, m, g, dt)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Grafik Sudut vs Waktu")
    fig, ax = plt.subplots()
    ax.plot(t, np.degrees(theta))
    ax.set_xlabel("Waktu (s)")
    ax.set_ylabel("Sudut (°)")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Energi Kinetik & Potensial")
    fig2, ax2 = plt.subplots()
    ax2.plot(t, T, label="Kinetik")
    ax2.plot(t, V, label="Potensial")
    ax2.legend()
    ax2.set_xlabel("Waktu (s)")
    ax2.set_ylabel("Energi (J)")
    ax2.grid(True)
    st.pyplot(fig2)

st.subheader("Lagrangian vs Waktu & Nilai Action")
fig3, ax3 = plt.subplots()
ax3.plot(t, Lagrangian, color='purple')
ax3.set_xlabel("Waktu (s)")
ax3.set_ylabel("Lagrangian (J)")
ax3.grid(True)
st.pyplot(fig3)
st.success(f"Nilai Action = {Action:.4f} J·s")
