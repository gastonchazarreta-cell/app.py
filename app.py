import math
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Football ValueStat Predictor",
    page_icon="⚽",
    layout="wide"
)

# ==============================================================================
# CONFIGURACIÓN / BASE DE EQUIPOS
# ==============================================================================

DEFAULT_TEAMS = {
    "Personalizado / Nuevo": [1.45, 1.30, 1.10, 14.0, 5.0, 5.5, 2.4, 13.0],
    "Boca Juniors": [1.50, 1.35, 1.00, 15.2, 5.8, 6.0, 2.8, 14.5],
    "River Plate": [1.65, 1.45, 0.95, 16.5, 6.2, 6.5, 2.2, 12.0],
    "San Lorenzo": [1.20, 1.10, 1.15, 11.5, 4.0, 4.5, 3.1, 15.0],
}

if "equipos_db" not in st.session_state:
    st.session_state.equipos_db = DEFAULT_TEAMS.copy()

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

    if total <= 0:
        return np.ones(max_k + 1) / (max_k + 1)

    return p / total


def poisson_over_prob(lam, line):
    """
    Para líneas .5:
    Over 2.5 = P(X >= 3)
    Over 3.5 = P(X >= 4)
    """
    threshold = math.floor(line)
    return 1.0 - sum(
        poisson_pmf(k, lam)
        for k in range(threshold + 1)
    )


def poisson_under_prob(lam, line):
    """
    Para líneas .5:
    Under 2.5 = P(X <= 2)
    Under 3.5 = P(X <= 3)
    """
    threshold = math.floor(line)
    return sum(
        poisson_pmf(k, lam)
        for k in range(threshold + 1)
    )


def odds_to_prob(odds):
    if odds is None or odds <= 1:
        return 0.0
    return 1.0 / odds


def fair_odds(prob):
    if prob <= 0:
        return 999.0
    return 1.0 / prob


def expected_value(prob, odds):
    """
    EV por unidad apostada.
    Ejemplo:
    prob = 0.55
    odds = 2.10
    EV = +0.155 = +15.5%
    """
    if prob <= 0 or odds <= 1:
        return -1.0

    return (prob * odds) - 1.0


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
        "Señal": market_signal(model_prob, market_prob, ev)
    }


# ==============================================================================
# TÍTULO
# ==============================================================================

st.title("⚽ Football ValueStat Predictor")

st.caption(
    "Modelo de probabilidades para goles, BTTS, remates, remates al arco, "
    "córners y tarjetas."
)


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:

    st.header("⚽ Partido")

    home_team = st.text_input(
        "Equipo Local",
        "Boca Juniors"
    )

    away_team = st.text_input(
        "Equipo Visitante",
        "San Lorenzo"
    )

    st.divider()

    st.header("🗂️ Base de equipos")

    lista_keys = list(st.session_state.equipos_db.keys())

    sel_home_db = st.selectbox(
        "Cargar datos del Local",
        lista_keys,
        key="sel_h"
    )

    sel_away_db = st.selectbox(
        "Cargar datos del Visitante",
        lista_keys,
        index=min(len(lista_keys) - 1, 3),
        key="sel_a"
    )

    st.divider()

    st.header("💾 Base de datos")

    # Exportar base
    equipos_json = json.dumps(
        st.session_state.equipos_db,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        "📥 Descargar base de equipos",
        data=equipos_json,
        file_name="equipos_football_valuestat.json",
        mime="application/json",
        use_container_width=True
    )

    # Importar base
    archivo_importado = st.file_uploader(
        "📤 Importar base de equipos",
        type=["json"]
    )

    if archivo_importado is not None:

        try:
            nueva_base = json.load(archivo_importado)

            if isinstance(nueva_base, dict):
                st.session_state.equipos_db = nueva_base
                st.success("Base importada correctamente.")
                st.rerun()
            else:
                st.error("El archivo no tiene un formato válido.")

        except Exception as e:
            st.error(f"No se pudo importar la base: {e}")


# ==============================================================================
# DATOS BASE
# ==============================================================================

h_vals = st.session_state.equipos_db[sel_home_db]
a_vals = st.session_state.equipos_db[sel_away_db]


# ==============================================================================
# 1. GOLES
# ==============================================================================

st.header("⚽ 1. Goles y resultados")

col_g1, col_g2 = st.columns(2)

with col_g1:

    st.subheader(f"Local: {home_team}")

    h_xg = st.number_input(
        "xG Local",
        min_value=0.0,
        value=float(h_vals[0]),
        step=0.05,
        key="h_xg"
    )

    h_xg_nonpen = st.number_input(
        "xG sin penaltis Local",
        min_value=0.0,
        value=float(h_vals[1]),
        step=0.05,
        key="h_xg_np"
    )

    h_xga = st.number_input(
        "xGA Local",
        min_value=0.0,
        value=float(h_vals[2]),
        step=0.05,
        key="h_xga"
    )


with col_g2:

    st.subheader(f"Visitante: {away_team}")

    a_xg = st.number_input(
        "xG Visitante",
        min_value=0.0,
        value=float(a_vals[0]),
        step=0.05,
        key="a_xg"
    )

    a_xg_nonpen = st.number_input(
        "xG sin penaltis Visitante",
        min_value=0.0,
        value=float(a_vals[1]),
        step=0.05,
        key="a_xg_np"
    )

    a_xga = st.number_input(
        "xGA Visitante",
        min_value=0.0,
        value=float(a_vals[2]),
        step=0.05,
        key="a_xga"
    )


# ==============================================================================
# 2. REMATES
# ==============================================================================

st.divider()

st.header("🎯 2. Remates y remates al arco")

col_r1, col_r2 = st.columns(2)

with col_r1:

    st.subheader(f"Local: {home_team}")

    h_shots = st.number_input(
        "Tiros totales Local",
        min_value=0.0,
        value=float(h_vals[3]),
        step=0.5,
        key="h_sh"
    )

    h_sot = st.number_input(
        "Tiros al arco Local",
        min_value=0.0,
        value=float(h_vals[4]),
        step=0.5,
        key="h_sot"
    )


with col_r2:

    st.subheader(f"Visitante: {away_team}")

    a_shots = st.number_input(
        "Tiros totales Visitante",
        min_value=0.0,
        value=float(a_vals[3]),
        step=0.5,
        key="a_sh"
    )

    a_sot = st.number_input(
        "Tiros al arco Visitante",
        min_value=0.0,
        value=float(a_vals[4]),
        step=0.5,
        key="a_sot"
    )


# ==============================================================================
# 3. CÓRNERS Y TARJETAS
# ==============================================================================

st.divider()

st.header("🚩 3. Córners y 🟨 tarjetas")

col_c1, col_c2 = st.columns(2)

with col_c1:

    h_corners = st.number_input(
        f"Córners {home_team}",
        min_value=0.0,
        value=float(h_vals[5]),
        step=0.5,
        key="h_co"
    )

    h_yellow = st.number_input(
        f"Amarillas {home_team}",
        min_value=0.0,
        value=float(h_vals[6]),
        step=0.1,
        key="h_yl"
    )

    h_fouls = st.number_input(
        f"Faltas {home_team}",
        min_value=0.0,
        value=float(h_vals[7]),
        step=0.5,
        key="h_fo"
    )


with col_c2:

    a_corners = st.number_input(
        f"Córners {away_team}",
        min_value=0.0,
        value=float(a_vals[5]),
        step=0.5,
        key="a_co"
    )

    a_yellow = st.number_input(
        f"Amarillas {away_team}",
        min_value=0.0,
        value=float(a_vals[6]),
        step=0.1,
        key="a_yl"
    )

    a_fouls = st.number_input(
        f"Faltas {away_team}",
        min_value=0.0,
        value=float(a_vals[7]),
        step=0.5,
        key="a_fo"
    )


# ==============================================================================
# CÁLCULOS DEL MODELO
# ==============================================================================

home_def_weak = max(0.20, a_xga)
away_def_weak = max(0.20, h_xga)

raw_hg = (
    0.55
    + 0.42 * h_xg
    + 0.18 * h_xg_nonpen
    + 0.14 * (h_sot / 5.0)
)

raw_hg *= (
    0.75
    + 0.25 * min(2.0, home_def_weak / 1.25)
)


raw_ag = (
    0.42
    + 0.42 * a_xg
    + 0.18 * a_xg_nonpen
    + 0.14 * (a_sot / 5.0)
)

raw_ag *= (
    0.75
    + 0.25 * min(2.0, away_def_weak / 1.25)
)


home_goals = float(np.clip(raw_hg, 0.25, 4.5))
away_goals = float(np.clip(raw_ag, 0.20, 4.5))

total_goals = home_goals + away_goals


# ==============================================================================
# BTTS
# ==============================================================================

btts_yes = (
    1
    - poisson_pmf(0, home_goals)
    - poisson_pmf(0, away_goals)
    + poisson_pmf(0, home_goals)
    * poisson_pmf(0, away_goals)
)


# ==============================================================================
# MATRIZ DE MARCADORES
# ==============================================================================

max_goals = 7

hp = poisson_dist(home_goals, max_goals)
ap = poisson_dist(away_goals, max_goals)

score_matrix = np.outer(hp, ap)


home_win = float(
    np.tril(score_matrix, -1).sum()
)

draw = float(
    np.trace(score_matrix)
)

away_win = float(
    np.triu(score_matrix, 1).sum()
)

result_probs = np.array([
    home_win,
    draw,
    away_win
])

result_probs = result_probs / result_probs.sum()


# ==============================================================================
# REMATES
# ==============================================================================

home_shots_calc = max(
    1.0,
    0.62 * h_shots
    + 0.38 * (h_shots + a_xga * 2.0)
)

away_shots_calc = max(
    1.0,
    0.62 * a_shots
    + 0.38 * (a_shots + h_xga * 2.0)
)

total_shots = home_shots_calc + away_shots_calc


# ==============================================================================
# REMATES AL ARCO
# ==============================================================================

home_sot_calc = max(
    0.5,
    0.70 * h_sot
    + 0.30 * (home_shots_calc * 0.35)
)

away_sot_calc = max(
    0.5,
    0.70 * a_sot
    + 0.30 * (away_shots_calc * 0.35)
)

total_sot = home_sot_calc + away_sot_calc


# ==============================================================================
# CÓRNERS
# ==============================================================================

home_corners_calc = max(
    1.0,
    0.70 * h_corners
    + 0.30 * (
        0.5 * h_corners
        + 0.5 * h_shots / 2.5
    )
)

away_corners_calc = max(
    1.0,
    0.70 * a_corners
    + 0.30 * (
        0.5 * a_corners
        + 0.5 * a_shots / 2.5
    )
)

total_corners = (
    home_corners_calc
    + away_corners_calc
)


# ==============================================================================
# TARJETAS
# ==============================================================================

home_cards_calc = max(
    0.1,
    0.62 * h_yellow
    + 0.20 * (h_fouls / 5.0)
)

away_cards_calc = max(
    0.1,
    0.62 * a_yellow
    + 0.20 * (a_fouls / 5.0)
)

total_cards = (
    home_cards_calc
    + away_cards_calc
)


# ==============================================================================
# RESUMEN DEL MODELO
# ==============================================================================

st.divider()

st.header("📊 Predicciones del modelo")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "⚽ Goles",
    f"{total_goals:.2f}"
)

m2.metric(
    "🎯 Remates",
    f"{total_shots:.2f}"
)

m3.metric(
    "🎯 Remates al arco",
    f"{total_sot:.2f}"
)

m4.metric(
    "🚩 Córners",
    f"{total_corners:.2f}"
)

st.info(
    f"""
**Proyección de goles**

{home_team}: **{home_goals:.2f}**  
{away_team}: **{away_goals:.2f}**

Tarjetas esperadas: **{total_cards:.2f}**
"""
)


# ==============================================================================
# MAPA DE CALOR DE MARCADORES
# ==============================================================================

st.divider()

st.header("🔥 Mapa de calor — Marcadores más probables")

st.caption(
    "Cada celda representa la probabilidad estimada de ese marcador. "
    "El eje vertical corresponde al Local y el horizontal al Visitante."
)


# Usamos 0-5 para que sea legible en celular.
heat_max = 5

heat_matrix = score_matrix[:heat_max + 1, :heat_max + 1] * 100

x_labels = [str(i) for i in range(heat_max + 1)]
y_labels = [str(i) for i in range(heat_max + 1)]

fig = go.Figure(
    data=go.Heatmap(
        z=heat_matrix,
        x=x_labels,
        y=y_labels,
        text=np.vectorize(lambda x: f"{x:.1f}%")(heat_matrix),
        texttemplate="%{text}",
        hovertemplate=(
            f"{home_team}: %{{y}}<br>"
            f"{away_team}: %{{x}}<br>"
            "Probabilidad: %{z:.2f}%"
            "<extra></extra>"
        ),
        colorscale="YlOrRd",
        colorbar=dict(
            title="Prob."
        )
    )
)

fig.update_layout(
    xaxis_title=f"Goles {away_team}",
    yaxis_title=f"Goles {home_team}",
    height=500,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==============================================================================
# TOP 5 MARCADORES
# ==============================================================================

score_rows = []

for home_g in range(heat_max + 1):
    for away_g in range(heat_max + 1):

        probability = (
            score_matrix[home_g, away_g]
        )

        score_rows.append({
            "Local": home_g,
            "Visitante": away_g,
            "Probabilidad": probability
        })


score_rows = sorted(
    score_rows,
    key=lambda x: x["Probabilidad"],
    reverse=True
)

top_scores = score_rows[:5]

most_likely = top_scores[0]

st.success(
    f"""
🔥 **Marcador más probable:**
## {most_likely["Local"]} - {most_likely["Visitante"]}

Probabilidad estimada:
**{most_likely["Probabilidad"] * 100:.1f}%**
"""
)

top_scores_df = pd.DataFrame([
    {
        "Ranking": i + 1,
        "Marcador": f"{x['Local']} - {x['Visitante']}",
        "Probabilidad": f"{x['Probabilidad'] * 100:.1f}%"
    }
    for i, x in enumerate(top_scores)
])

st.dataframe(
    top_scores_df,
    use_container_width=True,
    hide_index=True
)


# ==============================================================================
# CONFIGURACIÓN DE MERCADOS
# ==============================================================================

st.divider()

st.header("⚙️ Configuración de líneas de mercado")

col_m1, col_m2 = st.columns(2)


with col_m1:

    st.subheader("⚽ Goles")

    tipo_apuesta_goles = st.selectbox(
        "Mercado de Goles",
        ["Over (Más de)", "Under (Menos de)"],
        key="tipo_gol"
    )

    linea_goles = st.number_input(
        "Línea de Goles",
        min_value=0.5,
        value=2.5,
        step=0.5,
        key="lin_gol"
    )

    market_odds_goles = st.number_input(
        "Cuota Goles",
        min_value=1.01,
        value=2.10,
        step=0.01,
        key="cuota_gol"
    )


    st.subheader("⚽ BTTS")

    market_odds_btts = st.number_input(
        "Cuota BTTS Sí",
        min_value=1.01,
        value=1.85,
        step=0.01
    )


    st.subheader("🎯 Remates")

    tipo_apuesta_remates = st.selectbox(
        "Mercado Remates Totales",
        ["Over (Más de)", "Under (Menos de)"],
        key="tipo_rem"
    )

    linea_remates = st.number_input(
        "Línea Remates Totales",
        min_value=0.5,
        value=23.5,
        step=0.5,
        key="lin_rem"
    )

    market_odds_remates = st.number_input(
        "Cuota Remates",
        min_value=1.01,
        value=1.90,
        step=0.01,
        key="cuota_rem"
    )


with col_m2:

    st.subheader("🎯 Remates al arco")

    tipo_apuesta_sot = st.selectbox(
        "Mercado Remates al Arco",
        ["Over (Más de)", "Under (Menos de)"],
        key="tipo_sot"
    )

    linea_sot = st.number_input(
        "Línea Remates al Arco",
        min_value=0.5,
        value=8.5,
        step=0.5,
        key="lin_sot"
    )

    market_odds_sot = st.number_input(
        "Cuota Remates al Arco",
        min_value=1.01,
        value=1.85,
        step=0.01,
        key="cuota_sot"
    )


    st.subheader("🚩 Córners")

    tipo_apuesta_corners = st.selectbox(
        "Mercado de Córners",
        ["Over (Más de)", "Under (Menos de)"],
        key="tipo_cor"
    )

    linea_corners = st.number_input(
        "Línea de Córners",
        min_value=0.5,
        value=9.5,
        step=0.5,
        key="lin_cor"
    )

    market_odds_corners = st.number_input(
        "Cuota Córners",
        min_value=1.01,
        value=1.90,
        step=0.01,
        key="cuota_cor"
    )


    st.subheader("🟨 Tarjetas")

    tipo_apuesta_tarjetas = st.selectbox(
        "Mercado de Tarjetas",
        ["Over (Más de)", "Under (Menos de)"],
        key="tipo_tar"
    )

    linea_tarjetas = st.number_input(
        "Línea de Tarjetas",
        min_value=0.5,
        value=3.5,
        step=0.5,
        key="lin_tar"
    )

    market_odds_tarjetas = st.number_input(
        "Cuota Tarjetas",
        min_value=1.01,
        value=2.50,
        step=0.01,
        key="cuota_tar"
    )


# ==============================================================================
# PROBABILIDADES DE MERCADO
# ==============================================================================

def calculate_over_under(tipo, total, linea):

    if "Under" in tipo:
        prob = poisson_under_prob(total, linea)
        name = "Under"
    else:
        prob = poisson_over_prob(total, linea)
        name = "Over"

    return prob, f"{name} {linea}"


model_prob_goles, nombre_base_goles = calculate_over_under(
    tipo_apuesta_goles,
    total_goals,
    linea_goles
)

nombre_mercado_goles = f"{nombre_base_goles} Goles"


model_prob_remates, nombre_base_remates = calculate_over_under(
    tipo_apuesta_remates,
    total_shots,
    linea_remates
)

nombre_mercado_remates = f"{nombre_base_remates} Remates"


model_prob_sot, nombre_base_sot = calculate_over_under(
    tipo_apuesta_sot,
    total_sot,
    linea_sot
)

nombre_mercado_sot = f"{nombre_base_sot} Remates al Arco"


model_prob_corners, nombre_base_corners = calculate_over_under(
    tipo_apuesta_corners,
    total_corners,
    linea_corners
)

nombre_mercado_corners = f"{nombre_base_corners} Córners"


model_prob_tarjetas, nombre_base_tarjetas = calculate_over_under(
    tipo_apuesta_tarjetas,
    total_cards,
 
