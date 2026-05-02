import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Smart Irrigation 🌱",
    page_icon="💧",
    layout="wide"
)

# ── HEADER ───────────────────────────────
st.title("💧 Smart Irrigation System")
st.markdown("### Système intelligent de prédiction d'irrigation basé sur ML")
st.divider()

# ── SIDEBAR ──────────────────────────────
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2917/2917995.png", width=100)
st.sidebar.header("🌡️ Valeurs des capteurs")
st.sidebar.markdown("Ajuste les valeurs pour simuler les capteurs")

moisture0 = st.sidebar.slider("💧 Capteur 0", 0.0, 1.0, 0.3)
moisture1 = st.sidebar.slider("💧 Capteur 1", 0.0, 1.0, 0.6)
moisture2 = st.sidebar.slider("💧 Capteur 2", 0.0, 1.0, 0.7)
moisture3 = st.sidebar.slider("💧 Capteur 3", 0.0, 1.0, 0.5)
moisture4 = st.sidebar.slider("💧 Capteur 4", 0.0, 1.0, 0.1)
hour      = st.sidebar.slider("🕐 Heure", 0, 23, 12)

# ── MÉTRIQUES ────────────────────────────
col1, col2, col3, col4 = st.columns(4)
moyenne = (moisture0+moisture1+moisture2+moisture3+moisture4)/5
col1.metric("💧 Moyenne humidité", f"{moyenne:.2f}")
col2.metric("📉 Capteur min", f"{min(moisture0,moisture1,moisture2,moisture3,moisture4):.2f}")
col3.metric("📈 Capteur max", f"{max(moisture0,moisture1,moisture2,moisture3,moisture4):.2f}")
col4.metric("🕐 Heure", f"{hour}h00")

st.divider()

# ── BOUTON PRÉDICTION ─────────────────────
col_btn, col_res = st.columns([1, 2])

with col_btn:
    predict_btn = st.button(
        "🔮 Analyser l'irrigation",
        use_container_width=True,
        type="primary"
    )

with col_res:
    if predict_btn:
        with st.spinner("Analyse en cours..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={
                        "moisture0": moisture0,
                        "moisture1": moisture1,
                        "moisture2": moisture2,
                        "moisture3": moisture3,
                        "moisture4": moisture4,
                        "hour": hour
                    }
                )
                result = response.json()

                if result["irrigation_needed"]:
                    st.error(f"## 💧 Irrigation Nécessaire !")
                    st.error(f"**Confiance : {result['confidence']}**")
                else:
                    st.success(f"## ✅ Pas d'irrigation nécessaire")
                    st.success(f"**Confiance : {result['confidence']}**")

                st.info(f"Moyenne humidité : **{result['moisture_average']:.3f}**")

            except:
                st.error("❌ Lance d'abord l'API !")
                st.code("uvicorn src.api:app --reload")

st.divider()

# ── GRAPHIQUE ────────────────────────────
st.subheader("📊 Visualisation des capteurs")

data_viz = pd.DataFrame({
    "Capteur": ["Capteur 0","Capteur 1","Capteur 2","Capteur 3","Capteur 4"],
    "Humidité": [moisture0, moisture1, moisture2, moisture3, moisture4],
    "Seuil critique": [0.35, 0.35, 0.35, 0.35, 0.35]
})

fig = px.bar(
    data_viz, x="Capteur", y=["Humidité", "Seuil critique"],
    barmode="group",
    color_discrete_map={
        "Humidité": "#2196F3",
        "Seuil critique": "#FF5722"
    },
    title="Niveaux d'humidité vs Seuil critique (0.35)"
)
st.plotly_chart(fig, use_container_width=True)

# ── FOOTER ───────────────────────────────
st.divider()
st.markdown("**Smart Irrigation MLOps** FSSM 2024")