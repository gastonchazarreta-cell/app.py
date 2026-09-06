import math
from datetime import date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

st.set_page_config(
    page_title="Football ValueStat Predictor",
    page_icon="⚽",
    layout="wide",
)

if "historial" not in st.session_state:
    st.session_state.historial = []


# ==============================================================================
# FUNCIONES MATEMÁTICAS
# ==============================================================================

def poisson_pmf(k, lam):
    lam = max(0.001, float(lam))
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_dist(lam, max_k=8):
    p = np.array([poisson_pmf(k, lam) for k in range(max_k + 1)])
    total = p.sum()
    return p / total if total > 0 else np.ones(max_k + 1) / (max_k + 1)


def poisson_over_prob(lam, line):
    threshold = math.floor(line)
    return 1.0 - sum(poisson_pmf(k, lam) for k in range(threshold + 1))


def poisson_under_prob(lam, line):
    threshold = math.floor(line)
    return sum(poisson_pmf(k, lam) for k in range(threshold + 1))


def odds_to_prob(odds):
    return 1.0 / odds if odds and odds > 1 else 0.0


def fair_odds(prob):
    return 1.0 / prob if prob > 0 else 999.0


def expected_value(prob, odds):
    return (prob * odds) - 1.0 if prob > 0 and odds > 1 else -1.0


def market_signal(model_prob, market_prob, ev):
    edge = model_prob - market_prob
    if ev >= 0.10 and model_prob >= 0.55:
        return f"🔥 VALUE FUERTE (+{ev * 100:.1f}% EV)"
    if ev >= 0.05:
        return f"🟢 VALUE BUENO (+{ev * 100:.1f}% EV)"
    if ev > 0:
        return f"🟡 VALUE LEVE (+{ev * 100:.1f}% EV)"
    if edge <= -0.05:
        return "❌ EN CONTRA"
    return "⚪ SIN VALOR CLARO"


def analyze_market(name, model_prob, odds):
    market_prob = odds_to_prob(odds)
    fair = fair_odds(model_prob)
    ev = expected_value(model_prob, odds)
    edge = model_prob - market_prob
    return {
        "Mercado": name,
        "Prob. Modelo": model_prob,
        "Prob. Mercado": market_prob,
        "Cuota Justa": fair,
        "Cuota Mercado": odds,
        "Edge": edge,
        "EV": ev,
        "Señal": market_signal(model_prob, market_prob, ev),
    }


def calculate_over_under(tipo, total, linea):
    if "Under" in tipo:
        prob = poisson_under_prob(total, linea)
        name = "Under"
    else:
        prob = poisson_over_prob(total, linea)
        name = "Over"
    return prob, f"{name} {linea}"


# ==============================================================================
# TÍTULO
# ==============================================================================

st.title("⚽ Football ValueStat Predictor")
st.caption(
    "Modelo de probabilidades para goles, BTTS, remates, remates al arco, "
    "córners y tarjetas."
)


# ==============================================================================
# PARTIDO / ENTRADA DE DATOS MANUAL
# ==============================================================================

st.header("🏆 Partido y Equipos")

competition = st.selectbox(
    "Competición del partido",
    [
        "Liga",
        "Copa Nacional",
        "Copa Libertadores",
        "Copa Sudamericana",
        "Champions League",
        "Europa League",
        "Otro",
    ],
)

left, right = st.columns(2)

with left:
    home_team = st.text_input("Nombre Equipo Local", "Local")

with right:
    away_team = st.text_input("Nombre Equipo Visitante", "Visitante")


def render_data_inputs(prefix, title, default_vals):
    st.divider()
    st.subheader(title)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        xg = st.number_input("xG", min_value=0.0, value=default_vals[0], step=0.05, key=f"{prefix}_xg")
        xg_np = st.number_input("xG sin penal", min_value=0.0, value=default_vals[1], step=0.05, key=f"{prefix}_xg_np")

    with c2:
        xga = st.number_input("xGA", min_value=0.0, value=default_vals[2], step=0.05, key=f"{prefix}_xga")
        shots = st.number_input("Tiros totales", min_value=0.0, value=default_vals[3], step=0.5, key=f"{prefix}_shots")

    with c3:
        sot = st.number_input("Tiros al arco", min_value=0.0, value=default_vals[4], step=0.5, key=f"{prefix}_sot")
        corners = st.number_input("Córners", min_value=0.0, value=default_vals[5], step=0.5, key=f"{prefix}_corners")

    with c4:
        yellow = st.number_input("Amarillas", min_value=0.0, value=default_vals[6], step=0.1, key=f"{prefix}_yellow")
        fouls = st.number_input("Faltas", min_value=0.0, value=default_vals[7], step=0.5, key=f"{prefix}_fouls")

    return xg, xg_np, xga, shots, sot, corners, yellow, fouls


h_xg, h_xg_nonpen, h_xga, h_shots, h_sot, h_corners, h_yellow, h_fouls = render_data_inputs(
    "home", f"📊 Datos — {home_team}", [1.45, 1.30, 1.10, 14.0, 5.0, 5.5, 2.4, 13.0]
)

a_xg, a_xg_nonpen, a_xga, a_shots, a_sot, a_corners, a_yellow, a_fouls = render_data_inputs(
    "away", f"📊 Datos — {away_team}", [1.15, 1.00, 1.35, 11.0, 3.8, 4.2, 2.7, 14.2]
)


# ==============================================================================
# CÁLCULOS
# ==============================================================================

home_def_weak = max(0.20, a_xga)
away_def_weak = max(0.20, h_xga)

raw_hg = (
    0.55 + 0.42 * h_xg + 0.18 * h_xg_nonpen + 0.14 * (h_sot / 5.0)
)
raw_hg *= 0.75 + 0.25 * min(2.0, home_def_weak / 1.25)

raw_ag = (
    0.42 + 0.42 * a_xg + 0.18 * a_xg_nonpen + 0.14 * (a_sot / 5.0)
)
raw_ag *= 0.75 + 0.25 * min(2.0, away_def_weak / 1.25)

home_goals = float(np.clip(raw_hg, 0.25, 4.5))
away_goals = float(np.clip(raw_ag, 0.20, 4.5))
total_goals = home_goals + away_goals

btts_yes = (
    1
    - poisson_pmf(0, home_goals)
    - poisson_pmf(0, away_goals)
    + poisson_pmf(0, home_goals) * poisson_pmf(0, away_goals)
)

max_goals = 7
hp = poisson_dist(home_goals, max_goals)
ap = poisson_dist(away_goals, max_goals)
score_matrix = np.outer(hp, ap)

home_win = float(np.tril(score_matrix, -1).sum())
draw = float(np.trace(score_matrix))
away_win = float(np.triu(score_matrix, 1).sum())

result_probs = np.array([home_win, draw, away_win])
result_probs /= result_probs.sum()

home_shots_calc = max(1.0, 0.62 * h_shots + 0.38 * (h_shots + a_xga * 2.0))
away_shots_calc = max(1.0, 0.62 * a_shots + 0.38 * (a_shots + h_xga * 2.0))
total_shots = home_shots_calc + away_shots_calc

home_sot_calc = max(0.5, 0.70 * h_sot + 0.30 * (home_shots_calc * 0.35))
away_sot_calc = max(0.5, 0.70 * a_sot + 0.30 * (away_shots_calc * 0.35))
total_sot = home_sot_calc + away_sot_calc

home_corners_calc = max(
    1.0,
    0.70 * h_corners + 0.30 * (0.5 * h_corners + 0.5 * h_shots / 2.5),
)
away_corners_calc = max(
    1.0,
    0.70 * a_corners + 0.30 * (0.5 * a_corners + 0.5 * a_shots / 2.5),
)
total_corners = home_corners_calc + away_corners_calc

home_cards_calc = max(0.1, 0.62 * h_yellow + 0.20 * (h_fouls / 5.0))
away_cards_calc = max(0.1, 0.62 * a_yellow + 0.20 * (a_fouls / 5.0))
total_cards = home_cards_calc + away_cards_calc


# ==============================================================================
# RESUMEN
# ==============================================================================

st.divider()
st.header("📊 Predicciones del modelo")

m1, m2, m3, m4 = st.columns(4)
m1.metric("⚽ Goles", f"{total_goals:.2f}")
m2.metric("🎯 Remates", f"{total_shots:.2f}")
m3.metric("🎯 Remates al arco", f"{total_sot:.2f}")
m4.metric("🚩 Córners", f"{total_corners:.2f}")

st.info(
    f"**Proyección de goles**\n\n"
    f"{home_team}: **{home_goals:.2f}**  \n"
    f"{away_team}: **{away_goals:.2f}**  \n\n"
    f"Tarjetas esperadas: **{total_cards:.2f}**"
)


# ==============================================================================
# MAPA DE CALOR
# ==============================================================================

st.divider()
st.header("🔥 Mapa de calor — Marcadores más probables")
st.caption(
    "Eje vertical = goles del local. Eje horizontal = goles del visitante."
)

heat_max = 5
heat_matrix = score_matrix[:heat_max + 1, :heat_max + 1] * 100

fig = go.Figure(
    data=go.Heatmap(
        z=heat_matrix,
        x=[str(i) for i in range(heat_max + 1)],
        y=[str(i) for i in range(heat_max + 1)],
        text=np.vectorize(lambda x: f"{x:.1f}%")(heat_matrix),
        texttemplate="%{text}",
        hovertemplate=(
            f"{home_team}: %{{y}}<br>"
            f"{away_team}: %{{x}}<br>"
            "Probabilidad: %{z:.2f}%<extra></extra>"
        ),
        colorscale="YlOrRd",
        colorbar=dict(title="Prob."),
    )
)

fig.update_layout(
    xaxis_title=f"Goles {away_team}",
    yaxis_title=f"Goles {home_team}",
    height=500,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TOP 5 MARCADORES
# ==============================================================================

score_rows = []
for hg in range(heat_max + 1):
    for ag in range(heat_max + 1):
        score_rows.append({
            "Local": hg,
            "Visitante": ag,
            "Probabilidad": score_matrix[hg, ag],
        })

top_scores = sorted(score_rows, key=lambda x: x["Probabilidad"], reverse=True)[:5]
most_likely = top_scores[0]

st.success(
    f"🔥 **Marcador más probable:** "
    f"## {most_likely['Local']} - {most_likely['Visitante']}\n\n"
    f"Probabilidad estimada: **{most_likely['Probabilidad'] * 100:.1f}%**"
)

st.dataframe(
    pd.DataFrame([
        {
            "Ranking": i + 1,
            "Marcador": f"{x['Local']} - {x['Visitante']}",
            "Probabilidad": f"{x['Probabilidad'] * 100:.1f}%",
        }
        for i, x in enumerate(top_scores)
    ]),
    use_container_width=True,
    hide_index=True,
)


# ==============================================================================
# MERCADOS
# ==============================================================================

st.divider()
st.header("⚙️ Configuración de líneas de mercado")

c1, c2 = st.columns(2)

with c1:
    st.subheader("⚽ Goles")
    tipo_goles = st.selectbox("Mercado de Goles", ["Over (Más de)", "Under (Menos de)"])
    linea_goles = st.number_input("Línea de Goles", min_value=0.5, value=2.5, step=0.5)
    odds_goles = st.number_input("Cuota Goles", min_value=1.01, value=2.10, step=0.01)

    st.subheader("⚽ BTTS")
    odds_btts = st.number_input("Cuota BTTS Sí", min_value=1.01, value=1.85, step=0.01)

    st.subheader("🎯 Remates")
    tipo_remates = st.selectbox("Mercado Remates Totales", ["Over (Más de)", "Under (Menos de)"])
    linea_remates = st.number_input("Línea Remates Totales", min_value=0.5, value=23.5, step=0.5)
    odds_remates = st.number_input("Cuota Remates", min_value=1.01, value=1.90, step=0.01)

with c2:
    st.subheader("🎯 Remates al arco")
    tipo_sot = st.selectbox("Mercado Remates al Arco", ["Over (Más de)", "Under (Menos de)"])
    linea_sot = st.number_input("Línea Remates al Arco", min_value=0.5, value=8.5, step=0.5)
    odds_sot = st.number_input("Cuota Remates al Arco", min_value=1.01, value=1.85, step=0.01)

    st.subheader("🚩 Córners")
    tipo_corners = st.selectbox("Mercado de Córners", ["Over (Más de)", "Under (Menos de)"])
    linea_corners = st.number_input("Línea de Córners", min_value=0.5, value=9.5, step=0.5)
    odds_corners = st.number_input("Cuota Córners", min_value=1.01, value=1.90, step=0.01)

    st.subheader("🟨 Tarjetas")
    tipo_tarjetas = st.selectbox("Mercado de Tarjetas", ["Over (Más de)", "Under (Menos de)"])
    linea_tarjetas = st.number_input("Línea de Tarjetas", min_value=0.5, value=3.5, step=0.5)
    odds_tarjetas = st.number_input("Cuota Tarjetas", min_value=1.01, value=2.50, step=0.01)


# ==============================================================================
# VALUE
# ==============================================================================

p_goles, n_goles = calculate_over_under(tipo_goles, total_goals, linea_goles)
p_remates, n_remates = calculate_over_under(tipo_remates, total_shots, linea_remates)
p_sot, n_sot = calculate_over_under(tipo_sot, total_sot, linea_sot)
p_corners, n_corners = calculate_over_under(tipo_corners, total_corners, linea_corners)
p_tarjetas, n_tarjetas = calculate_over_under(tipo_tarjetas, total_cards, linea_tarjetas)

analisis = [
    analyze_market(f"{n_goles} Goles", p_goles, odds_goles),
    analyze_market("BTTS Sí", btts_yes, odds_btts),
    analyze_market(f"{n_remates} Remates", p_remates, odds_remates),
    analyze_market(f"{n_sot} Remates al Arco", p_sot, odds_sot),
    analyze_market(f"{n_corners} Córners", p_corners, odds_corners),
    analyze_market(f"{n_tarjetas} Tarjetas", p_tarjetas, odds_tarjetas),
]

st.divider()
st.header("📊 Análisis de valor vs mercado")

display_data = [
    {
        "Mercado": x["Mercado"],
        "Modelo": f"{x['Prob. Modelo'] * 100:.1f}%",
        "Mercado": f"{x['Prob. Mercado'] * 100:.1f}%",
        "Cuota justa": f"{x['Cuota Justa']:.2f}",
        "Cuota": f"{x['Cuota Mercado']:.2f}",
        "Edge": f"{x['Edge'] * 100:+.1f}%",
        "EV": f"{x['EV'] * 100:+.1f}%",
        "Señal": x["Señal"],
    }
    for x in analisis
]

st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)

mejor = max(analisis, key=lambda x: x["EV"])

if mejor["EV"] > 0:
    st.success(
        f"🎯 **Mayor EV:** {mejor['Mercado']}  \n"
        f"Probabilidad modelo: **{mejor['Prob. Modelo'] * 100:.1f}%**  \n"
        f"Cuota justa: **{mejor['Cuota Justa']:.2f}**  \n"
        f"Cuota mercado: **{mejor['Cuota Mercado']:.2f}**  \n"
        f"EV estimado: **+{mejor['EV'] * 100:.1f}%**"
    )
else:
    st.warning("No aparece ninguna selección con EV positivo.")


# ==============================================================================
# 1X2
# ==============================================================================

st.divider()
st.header("🎯 Probabilidades 1X2")

r1, r2, r3 = st.columns(3)
r1.metric(home_team, f"{result_probs[0] * 100:.1f}%")
r2.metric("Empate", f"{result_probs[1] * 100:.1f}%")
r3.metric(away_team, f"{result_probs[2] * 100:.1f}%")


# ==============================================================================
# HISTORIAL / INFORME
# ==============================================================================

st.divider()
st.header("📂 Historial")

h1, h2 = st.columns(2)

with h1:
    if st.button("📌 Guardar análisis", use_container_width=True):
        st.session_state.historial.append({
            "Fecha": date.today().isoformat(),
            "Competición": competition,
            "Local": home_team,
            "Visitante": away_team,
            "Goles esperados": round(total_goals, 2),
            "Remates": round(total_shots, 2),
            "Remates al arco": round(total_sot, 2),
            "Córners": round(total_corners, 2),
            "Tarjetas": round(total_cards, 2),
            "Marcador más probable": f"{most_likely['Local']}-{most_likely['Visitante']}",
            "Prob. marcador": f"{most_likely['Probabilidad'] * 100:.1f}%",
            "Mejor mercado": mejor["Mercado"],
            "EV": f"{mejor['EV'] * 100:+.1f}%",
        })
        st.success("Análisis guardado en la sesión.")

with h2:
    report = f"""FOOTBALL VALUESTAT PREDICTOR
==========================================

COMPETICIÓN
{competition}

PARTIDO
{home_team} vs {away_team}

PROYECCIONES
------------------------------------------
Goles esperados: {total_goals:.2f}
{home_team}: {home_goals:.2f}
{away_team}: {away_goals:.2f}
Remates: {total_shots:.2f}
Remates al arco: {total_sot:.2f}
Córners: {total_corners:.2f}
Tarjetas: {total_cards:.2f}

MARCADOR MÁS PROBABLE
------------------------------------------
{most_likely['Local']} - {most_likely['Visitante']}
Probabilidad: {most_likely['Probabilidad'] * 100:.1f}%

MEJOR VALUE
------------------------------------------
{mejor['Mercado']}
Cuota justa: {mejor['Cuota Justa']:.2f}
Cuota mercado: {mejor['Cuota Mercado']:.2f}
EV: {mejor['EV'] * 100:+.1f}%
Señal: {mejor['Señal']}
"""

    st.download_button(
        "📄 Descargar informe TXT",
        data=report,
        file_name=f"informe_{home_team}_vs_{away_team}.txt".replace(" ", "_"),
        mime="text/plain",
        use_container_width=True,
    )

if st.session_state.historial:
    st.subheader("📜 Historial de la sesión")
    st.dataframe(
        pd.DataFrame(st.session_state.historial),
        use_container_width=True,
        hide_index=True,
    )
    if st.button("🗑️ Limpiar historial"):
        st.session_state.historial = []
        st.rerun()
