import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import exp, factorial
from io import StringIO


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Football ValueStat Analyzer",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football ValueStat Analyzer")

st.caption(
    "Análisis estadístico utilizando todos los datos disponibles."
)


# ============================================================
# BASE DE COMPETICIONES
# ============================================================

COMPETITIONS = {

   "Champions League": {
        "season": "2026/27",
        "source": "UEFA / OddAlerts",
        "goals": 2.93,
        "shots": 26.40,
        "sot": 9.10,
        "corners": 9.74,
        "cards": 4.46
    },

    "Europa League": {
        "season": "2026/27",
        "source": "UEFA / FootyStats",
        "goals": 2.68,
        "shots": 24.10,
        "sot": 8.40,
        "corners": 9.20,
        "cards": 4.62
    },

    "Copa Libertadores": {
        "season": "2026",
        "source": "CONMEBOL / FootyStats",
        "goals": 2.38,
        "shots": 23.90,
        "sot": 8.10,
        "corners": 9.12,
        "cards": 4.85
    },

    "Copa Sudamericana": {
        "season": "2026",
        "source": "CONMEBOL / Sofascore",
        "goals": 2.42,
        "shots": 22.80,
        "sot": 7.90,
        "corners": 8.95,
        "cards": 5.10
    },

    "Liga Argentina": {
        "season": "2026",
        "source": "AFA / Liga Profesional",
        "goals": 2.14,
        "shots": 21.50,
        "sot": 7.10,
        "corners": 8.80,
        "cards": 5.32
    },

    "Brasileirão": {
        "season": "2026",
        "source": "CBF / FootyStats",
        "goals": 2.40,
        "shots": 24.60,
        "sot": 8.30,
        "corners": 10.15,
        "cards": 5.61
    },

    "La Liga": {
        "season": "2026/27",
        "source": "LaLiga / Statz",
        "goals": 2.53,
        "shots": 23.10,
        "sot": 8.00,
        "corners": 9.90,
        "cards": 4.75
    },

    "Serie A": {
        "season": "2026/27",
        "source": "Lega Serie A",
        "goals": 2.43,
        "shots": 25.00,
        "sot": 8.20,
        "corners": 8.30,
        "cards": 4.40
    },

    "Premier League": {
        "season": "2026/27",
        "source": "Premier League",
        "goals": 2.56,
        "shots": 26.80,
        "sot": 9.40,
        "corners": 9.14,
        "cards": 4.13
    },

    "Ligue 1": {
        "season": "2026/27",
        "source": "LFP / FBref",
        "goals": 2.61,
        "shots": 23.70,
        "sot": 8.50,
        "corners": 8.90,
        "cards": 4.05
    }
}


# ============================================================
# VALORES CONSIDERADOS COMO VACÍOS
# ============================================================

MISSING_VALUES = {
    "",
    "-",
    "--",
    "n/d",
    "nd",
    "n.a.",
    "n/a",
    "na",
    "null",
    "none",
    "nan"
}


# ============================================================
# PARSER
# ============================================================

def parse_values(
    text,
    allow_missing=True
):
    """
    Convierte el texto de ValueStat en una lista.

    Ejemplo:

    12 5 10 7 4 11

    queda:

    [12, 5, 10, 7, 4, 11]

    Si aparece - o N/D se conserva la posición
    como None.
    """

    if not text:
        return []

    normalized = (
        text
        .replace(",", " ")
        .replace(";", " ")
        .replace("|", " ")
        .replace("\t", " ")
        .replace("\n", " ")
    )

    tokens = normalized.split()

    values = []

    for token in tokens:

        clean_token = token.strip().lower()

        if (
            allow_missing
            and clean_token in MISSING_VALUES
        ):

            values.append(None)

            continue

        try:

            values.append(
                float(token)
            )

        except ValueError:

            # Si encontramos texto no numérico,
            # lo tratamos como dato faltante.
            if allow_missing:
                values.append(None)

    return values


# ============================================================
# SEPARAR EQUIPO / RIVAL
# ============================================================

def split_team_rival(values):
    """
    Interpreta:

    1 = equipo
    2 = rival
    3 = equipo
    4 = rival

    Acepta menos de 20 posiciones.
    Lo importante es que sean pares.
    """

    if not values:
        return None, None

    if len(values) < 2:
        return None, None

    # Si por algún motivo hay una posición impar,
    # descartamos solamente la última.
    if len(values) % 2 != 0:
        values = values[:-1]

    team = values[0::2]
    rival = values[1::2]

    return team, rival


# ============================================================
# LIMPIEZA
# ============================================================

def clean(values):

    if values is None:
        return []

    return [
        value
        for value in values
        if value is not None
    ]


def avg(values):

    values = clean(values)

    if not values:
        return None

    return float(
        np.mean(values)
    )


def med(values):

    values = clean(values)

    if not values:
        return None

    return float(
        np.median(values)
    )


def minimum(values):

    values = clean(values)

    if not values:
        return None

    return float(
        np.min(values)
    )


def maximum(values):

    values = clean(values)

    if not values:
        return None

    return float(
        np.max(values)
    )


def fmt(value):

    if value is None:
        return "N/D"

    if abs(
        value - round(value)
    ) < 0.01:

        return str(
            int(round(value))
        )

    return f"{value:.2f}"


def pct(value):

    if value is None:
        return "N/D"

    return f"{value:.1f}%"


def available_count(values):

    return len(
        clean(values)
    )


# ============================================================
# ESTADO DE LA MUESTRA
# ============================================================

def sample_status(values):

    count = available_count(
        values
    )

    if count >= 10:
        return "🟢 Completa"

    if count >= 7:
        return "🟡 Buena"

    if count >= 4:
        return "🟠 Parcial"

    if count >= 2:
        return "🔴 Débil"

    return "⚪ Sin datos"


# ============================================================
# OVER
# ============================================================

def over_percentage(
    values,
    line
):

    values = clean(values)

    if not values:
        return None

    return float(
        np.mean(
            np.array(values) > line
        ) * 100
    )


# ============================================================
# CONSISTENCIA
# ============================================================

def consistency_score(
    values,
    line
):

    values = clean(values)

    if not values:
        return 0

    average = avg(values)
    median = med(values)

    over = over_percentage(
        values,
        line
    )

    difference = abs(
        average - median
    )

    score = 0

    # Frecuencia
    if over >= 70:
        score += 45

    elif over >= 60:
        score += 35

    elif over >= 50:
        score += 25

    elif over >= 40:
        score += 15

    # Media vs mediana
    if difference <= 0.5:
        score += 35

    elif difference <= 1:
        score += 25

    elif difference <= 1.5:
        score += 15

    # Cantidad de partidos
    if len(values) >= 10:
        score += 20

    elif len(values) >= 7:
        score += 15

    elif len(values) >= 4:
        score += 8

    return min(
        score,
        100
    )


def consistency_label(
    values,
    line
):

    values = clean(values)

    if not values:
        return "⚪ SIN DATOS"

    average = avg(values)
    median = med(values)

    if (
        average is not None
        and median is not None
        and abs(
            average - median
        ) >= 2
    ):

        return "🔴 MEDIA INFLADA"

    score = consistency_score(
        values,
        line
    )

    if score >= 80:
        return "🟢 MUY CONSISTENTE"

    if score >= 65:
        return "🟢 CONSISTENTE"

    if score >= 45:
        return "🟡 IRREGULAR"

    return "🔴 BAJA CONFIANZA"


# ============================================================
# EXPECTATIVA
# ============================================================

def adjusted_expected(
    team_values,
    opponent_allowed,
    tournament_average=None
):

    values = []
    weights = []

    team_average = avg(
        team_values
    )

    opponent_average = avg(
        opponent_allowed
    )

    if team_average is not None:

        values.append(
            team_average
        )

        weights.append(
            0.50
        )

    if opponent_average is not None:

        values.append(
            opponent_average
        )

        weights.append(
            0.40
        )

    if tournament_average is not None:

        values.append(
            tournament_average
        )

        weights.append(
            0.10
        )

    if not values:
        return None

    return float(
        np.average(
            values,
            weights=weights
        )
    )


# ============================================================
# POISSON
# ============================================================

def poisson_probability(
    k,
    lamb
):

    if (
        lamb is None
        or lamb <= 0
    ):
        return 0

    return (
        exp(-lamb)
        * lamb ** k
        / factorial(k)
    )


def result_probabilities(
    home_expected,
    away_expected
):

    if (
        home_expected is None
        or away_expected is None
    ):

        return (
            0.3333,
            0.3333,
            0.3334,
            []
        )

    max_goals = 8

    home_probs = [
        poisson_probability(
            x,
            home_expected
        )
        for x in range(
            max_goals + 1
        )
    ]

    away_probs = [
        poisson_probability(
            x,
            away_expected
        )
        for x in range(
            max_goals + 1
        )
    ]

    home_win = 0
    draw = 0
    away_win = 0

    scores = []

    for home_score in range(
        max_goals + 1
    ):

        for away_score in range(
            max_goals + 1
        ):

            probability = (
                home_probs[
                    home_score
                ]
                *
                away_probs[
                    away_score
                ]
            )

            if home_score > away_score:

                home_win += probability

            elif home_score == away_score:

                draw += probability

            else:

                away_win += probability

            scores.append(
                (
                    f"{home_score}-{away_score}",
                    probability
                )
            )

    total = (
        home_win
        + draw
        + away_win
    )

    if total <= 0:

        return (
            0.3333,
            0.3333,
            0.3334,
            []
        )

    home_win /= total
    draw /= total
    away_win /= total

    scores.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return (
        home_win,
        draw,
        away_win,
        scores
    )


# ============================================================
# GRÁFICO 1X2
# ============================================================

def result_chart(
    home_probability,
    draw_probability,
    away_probability,
    home_name,
    away_name
):

    labels = [
        f"{home_name}\nLOCAL",
        "EMPATE",
        f"{away_name}\nVISITANTE"
    ]

    values = [
        home_probability * 100,
        draw_probability * 100,
        away_probability * 100
    ]

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    bars = ax.bar(
        labels,
        values
    )

    ax.set_ylim(
        0,
        100
    )

    ax.set_ylabel(
        "Probabilidad (%)"
    )

    ax.set_title(
        "Resultado estimado"
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold"
        )

    return fig


# ============================================================
# GRÁFICO DE DISTRIBUCIÓN
# ============================================================

def distribution_chart(
    team_values,
    rival_values,
    team_name,
    title
):

    team_plot = [
        np.nan
        if value is None
        else value
        for value in team_values
    ]

    rival_plot = [
        np.nan
        if value is None
        else value
        for value in rival_values
    ]

    total_positions = max(
        len(team_plot),
        len(rival_plot)
    )

    partidos = list(
        range(
            1,
            total_positions + 1
        )
    )

    if len(team_plot) < total_positions:

        team_plot += [
            np.nan
        ] * (
            total_positions
            - len(team_plot)
        )

    if len(rival_plot) < total_positions:

        rival_plot += [
            np.nan
        ] * (
            total_positions
            - len(rival_plot)
        )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    ax.plot(
        partidos,
        team_plot,
        marker="o",
        label=team_name
    )

    ax.plot(
        partidos,
        rival_plot,
        marker="o",
        label=f"Rival de {team_name}"
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Partido"
    )

    ax.set_ylabel(
        "Valor"
    )

    ax.set_xticks(
        partidos
    )

    ax.grid(
        alpha=0.20
    )

    ax.legend()

    return fig


# ============================================================
# CONSTRUIR DATASET
# ============================================================

def build_dataset(
    home_text,
    away_text
):

    home_values = parse_values(
        home_text,
        allow_missing=True
    )

    away_values = parse_values(
        away_text,
        allow_missing=True
    )

    # Si ambos están vacíos
    if (
        not home_values
        and not away_values
    ):

        return None

    # --------------------------------------------------------
    # Debemos tener al menos 2 posiciones:
    # equipo + rival
    # --------------------------------------------------------

    if len(home_values) < 2:

        return "ERROR_HOME"

    if len(away_values) < 2:

        return "ERROR_AWAY"

    # --------------------------------------------------------
    # Si existe una posición impar,
    # descartamos la última.
    # --------------------------------------------------------

    if len(home_values) % 2 != 0:

        home_values = home_values[:-1]

    if len(away_values) % 2 != 0:

        away_values = away_values[:-1]

    home_team, home_rival = (
        split_team_rival(
            home_values
        )
    )

    away_team, away_rival = (
        split_team_rival(
            away_values
        )
    )

    if (
        home_team is None
        or away_team is None
    ):

        return "ERROR"

    return {
        "home_team": home_team,
        "home_rival": home_rival,
        "away_team": away_team,
        "away_rival": away_rival
    }


# ============================================================
# ============================================================

def show_market(
    title,
    icon,
    data,
    home_name,
    away_name,
    lines,
    tournament_value
):

    home_team = data[
        "home_team"
    ]

    home_rival = data[
        "home_rival"
    ]

    away_team = data[
        "away_team"
    ]

    away_rival = data[
        "away_rival"
    ]

    home_average = avg(
        home_team
    )

    away_average = avg(
        away_team
    )

    home_median = med(
        home_team
    )

    away_median = med(
        away_team
    )

    home_expected = adjusted_expected(
        home_team,
        away_rival,
        tournament_value
    )

    away_expected = adjusted_expected(
        away_team,
        home_rival,
        tournament_value
    )

    total_expected = None

    if (
        home_expected is not None
        and away_expected is not None
    ):

        total_expected = (
            home_expected
            + away_expected
        )

    st.divider()

    st.header(
        f"{icon} {title}"
    )

    # ========================================================
    # RESUMEN
    # ========================================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader(
            home_name
        )

        st.metric(
            "Media",
            fmt(home_average)
        )

        st.caption(
            "Mediana: "
            + fmt(home_median)
        )

        st.caption(
            "Muestra: "
            f"{available_count(home_team)}/10"
        )

    with c2:

        st.subheader(
            "🎯 Modelo"
        )

        st.metric(
            "Total esperado",
            fmt(total_expected)
        )

        st.caption(
            f"{home_name}: "
            f"{fmt(home_expected)}"
        )

        st.caption(
            f"{away_name}: "
            f"{fmt(away_expected)}"
        )

        if tournament_value is not None:

            st.caption(
                "Baseline: "
                + fmt(tournament_value)
            )

    with c3:

        st.subheader(
            away_name
        )

        st.metric(
            "Media",
            fmt(away_average)
        )

        st.caption(
            "Mediana: "
            + fmt(away_median)
        )

        st.caption(
            "Muestra: "
            f"{available_count(away_team)}/10"
        )

    # ========================================================
    # PRODUCCIÓN / CONCESIÓN
    # ========================================================

    st.subheader(
        "⚖️ Producción vs concesión"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.write(
            f"**{home_name}** produce "
            f"**{fmt(home_average)}**"
        )

        st.write(
            "Los rivales del visitante "
            f"conceden **{fmt(avg(away_rival))}**"
        )

    with c2:

        st.write(
            f"**{away_name}** produce "
            f"**{fmt(away_average)}**"
        )

        st.write(
            "Los rivales del local "
            f"conceden **{fmt(avg(home_rival))}**"
        )

    # ========================================================
    # CALIDAD
    # ========================================================

    if len(lines) >= 2:

        reference_line = lines[1]

        st.subheader(
            "🚦 Calidad de la media"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.write(
                f"**{home_name}:** "
                f"{consistency_label(home_team, reference_line)}"
            )

        with c2:

            st.write(
                f"**{away_name}:** "
                f"{consistency_label(away_team, reference_line)}"
            )

    # ========================================================
    # TABLA
    # ========================================================

    st.subheader(
        "📋 Distribución disponible"
    )

    max_matches = max(
        len(home_team),
        len(away_team)
    )

    max_matches = max(
        max_matches,
        len(home_rival),
        len(away_rival)
    )

    table = pd.DataFrame({
        "Partido":
            range(
                1,
                max_matches + 1
            )
    })

    home_list = (
        home_team
        + [None]
        * (
            max_matches
            - len(home_team)
        )
    )

    home_rival_list = (
        home_rival
        + [None]
        * (
            max_matches
            - len(home_rival)
        )
    )

    away_list = (
        away_team
        + [None]
        * (
            max_matches
            - len(away_team)
        )
    )

    away_rival_list = (
        away_rival
        + [None]
        * (
            max_matches
            - len(away_rival)
        )
    )

    table[
        home_name
    ] = home_list

    table[
        f"Rival de {home_name}"
    ] = home_rival_list

    table[
        away_name
    ] = away_list

    table[
        f"Rival de {away_name}"
    ] = away_rival_list

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True
    )

    # ========================================================
    # GRÁFICO
    # ========================================================

    fig = distribution_chart(
        home_team,
        home_rival,
        home_name,
        f"{title} — distribución"
    )

    st.pyplot(
        fig,
        clear_figure=True
    )

    # ========================================================
    # LÍNEAS
    # ========================================================

    st.subheader(
        "🎯 Frecuencia histórica"
    )

    rows = []

    for line in lines:

        home_over = over_percentage(
            home_team,
            line
        )

        away_over = over_percentage(
            away_team,
            line
        )

        if (
            total_expected is None
        ):

            signal = "⚪"

        elif (
            total_expected >= line + 1
        ):

            signal = "🟢"

        elif (
            total_expected >= line
        ):

            signal = "🟡"

        else:

            signal = "🔴"

        rows.append({

            "Línea":
                f"+{line}",

            home_name:
                "N/D"
                if home_over is None
                else f"{home_over:.0f}%",

            away_name:
                "N/D"
                if away_over is None
                else f"{away_over:.0f}%",

            "Modelo":
                fmt(total_expected),

            "Señal":
                signal
        })

    line_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        line_df,
        hide_index=True,
        use_container_width=True
    )

# UI
# ============================================================

left, right = st.columns(
    [1, 2]
)


# ============================================================
# COLUMNA IZQUIERDA
# ============================================================

with left:

    st.header(
        "📥 Datos"
    )

    # --------------------------------------------------------
    # COMPETICIÓN
    # --------------------------------------------------------

    competition = st.selectbox(
        "🏆 Competición",
        list(
            COMPETITIONS.keys()
        )
    )

    tournament = COMPETITIONS[
        competition
    ]

    st.caption(
        f"Temporada {tournament['season']}"
    )

    st.caption(
        f"Fuente baseline: "
        f"{tournament['source']}"
    )

    with st.expander(
        "📊 Ver baseline de competición",
        expanded=False
    ):

        baseline_df = pd.DataFrame({

            "Dato": [
                "Goles",
                "Remates",
                "SOT",
                "Córners",
                "Tarjetas"
            ],

            "Promedio": [

                fmt(
                    tournament["goals"]
                ),

                fmt(
                    tournament["shots"]
                ),

                fmt(
                    tournament["sot"]
                ),

                fmt(
                    tournament["corners"]
                ),

                fmt(
                    tournament["cards"]
                )
            ]
        })

        st.dataframe(
            baseline_df,
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------------
    # EQUIPOS
    # --------------------------------------------------------

    st.subheader(
        "⚽ Equipos"
    )

    home_name = st.text_input(
        "LOCAL",
        placeholder="Barcelona"
    )

    away_name = st.text_input(
        "VISITANTE",
        placeholder="Real Madrid"
    )

    st.divider()

    st.info(
        """
        Pegá los números tal como aparecen en ValueStat.

        No hace falta que todas las estadísticas tengan
        los mismos partidos.

        Ejemplo:

        20 datos → 10 partidos

        14 datos → 7 partidos

        8 datos → 4 partidos

        Para datos faltantes podés dejar:
        - / N/D / N/A / null
        """
    )

    # ========================================================
    # GOLES
    # ========================================================

    st.subheader(
        "⚽ Goles"
    )

    home_goals_text = st.text_area(
        "LOCAL — Goles",
        height=70,
        placeholder=(
            "1 0 2 1 3 1 1 0 2 2 ..."
        )
    )

    away_goals_text = st.text_area(
        "VISITANTE — Goles",
        height=70,
        placeholder=(
            "2 1 1 0 0 1 2 2 1 0 ..."
        )
    )

    # ========================================================
    # XG
    # ========================================================

    st.subheader(
        "📈 xG — opcional"
    )

    home_xg_text = st.text_area(
        "LOCAL — xG",
        height=70,
        placeholder=(
            "1.5 0.8 2.1 1.2 - 0.9 ..."
        )
    )

    away_xg_text = st.text_area(
        "VISITANTE — xG",
        height=70,
        placeholder=(
            "1.3 1.0 1.8 - 1.4 1.1 ..."
        )
    )

    # ========================================================
    # XGA
    # ========================================================

    st.subheader(
        "🛡️ xGA — opcional"
    )

    home_xga_text = st.text_area(
        "LOCAL — xGA",
        height=70,
        placeholder=(
            "0.8 1.2 0.9 1.4 - 1.6 ..."
        )
    )

    away_xga_text = st.text_area(
        "VISITANTE — xGA",
        height=70,
        placeholder=(
            "1.0 1.1 0.8 - 1.2 0.9 ..."
        )
    )

    # ========================================================
    # REMATES
    # ========================================================

    st.subheader(
        "🎯 Remates totales"
    )

    home_shots_text = st.text_area(
        "LOCAL — Remates",
        height=70,
        placeholder=(
            "12 5 10 7 4 11 8 6 ..."
        )
    )

    away_shots_text = st.text_area(
        "VISITANTE — Remates",
        height=70,
        placeholder=(
            "10 8 7 9 12 4 6 11 ..."
        )
    )

    # ========================================================
    # SOT
    # ========================================================

    st.subheader(
        "🥅 Tiros al arco"
    )

    home_sot_text = st.text_area(
        "LOCAL — SOT",
        height=70,
        placeholder=(
            "5 2 4 3 2 5 6 2 ..."
        )
    )

    away_sot_text = st.text_area(
        "VISITANTE — SOT",
        height=70,
        placeholder=(
            "4 3 3 4 5 2 4 5 ..."
        )
    )

    # ========================================================
    # PRECISIÓN
    # ========================================================

    st.subheader(
        "🎯 Precisión de tiro %"
    )

    home_accuracy_text = st.text_area(
        "LOCAL — Precisión",
        height=70,
        placeholder=(
            "42 40 33 45 38 36 50 ..."
        )
    )

    away_accuracy_text = st.text_area(
        "VISITANTE — Precisión",
        height=70,
        placeholder=(
            "38 35 43 39 41 33 45 ..."
        )
    )

    # ========================================================
    # CÓRNERS
    # ========================================================

    st.subheader(
        "🚩 Córners"
    )

    home_corners_text = st.text_area(
        "LOCAL — Córners",
        height=70,
        placeholder=(
            "6 4 7 3 5 5 8 4 ..."
        )
    )

    away_corners_text = st.text_area(
        "VISITANTE — Córners",
        height=70,
        placeholder=(
            "5 4 6 5 7 3 4 6 ..."
        )
    )

    # ========================================================
    # TARJETAS
    # ========================================================

    st.subheader(
        "🟨 Tarjetas"
    )

    home_cards_text = st.text_area(
        "LOCAL — Amarillas",
        height=70,
        placeholder=(
            "2 3 1 4 3 2 2 5 ..."
        )
    )

    away_cards_text = st.text_area(
        "VISITANTE — Amarillas",
        height=70,
        placeholder=(
            "3 2 4 3 2 3 1 2 ..."
        )
    )

    st.divider()

    analyze_button = st.button(
        "🔎 ANALIZAR PARTIDO",
        type="primary",
        use_container_width=True
    )


# ============================================================
# ANÁLISIS
# ============================================================

if analyze_button:

    if not home_name.strip():

        st.error(
            "Ingresá el equipo local."
        )

        st.stop()

    if not away_name.strip():

        st.error(
            "Ingresá el equipo visitante."
        )

        st.stop()

    # ========================================================
    # DATOS
    # ========================================================

    input_data = {

        "Goles": (
            home_goals_text,
            away_goals_text
        ),

        "xG": (
            home_xg_text,
            away_xg_text
        ),

        "xGA": (
            home_xga_text,
            away_xga_text
        ),

        "Remates": (
            home_shots_text,
            away_shots_text
        ),

        "SOT": (
            home_sot_text,
            away_sot_text
        ),

        "Precisión": (
            home_accuracy_text,
            away_accuracy_text
        ),

        "Córners": (
            home_corners_text,
            away_corners_text
        ),

        "Tarjetas": (
            home_cards_text,
            away_cards_text
        )
    }

    parsed = {}
    errors = []

    # ========================================================
    # PROCESAR TODAS LAS ESTADÍSTICAS
    # ========================================================

    for stat_name, (
        home_text,
        away_text
    ) in input_data.items():

        # Ambos vacíos = permitido
        if (
            not home_text.strip()
            and not away_text.strip()
        ):

            continue

        dataset = build_dataset(
            home_text,
            away_text
        )

        # ----------------------------------------------------
        # ERROR DE LOCAL
        # ----------------------------------------------------

        if dataset == "ERROR_HOME":

            errors.append(
                f"{stat_name}: el LOCAL necesita "
                f"al menos un dato de equipo y uno de rival."
            )

            continue

        # ----------------------------------------------------
        # ERROR VISITANTE
        # ----------------------------------------------------

        if dataset == "ERROR_AWAY":

            errors.append(
                f"{stat_name}: el VISITANTE necesita "
                f"al menos un dato de equipo y uno de rival."
            )

            continue

        # ----------------------------------------------------
        # ERROR GENERAL
        # ----------------------------------------------------

        if dataset == "ERROR":

            errors.append(
                f"{stat_name}: no se pudieron interpretar "
                f"los datos."
            )

            continue

        # ----------------------------------------------------
        # DATASET CORRECTO
        # ----------------------------------------------------

        if dataset is not None:

            parsed[
                stat_name
            ] = dataset

    # ========================================================
    # MOSTRAR ERRORES
    # ========================================================

    if errors:

        for error in errors:

            st.error(
                error
            )

        # NO usamos st.stop()
        # porque los demás mercados pueden funcionar.
        # Esto es intencional.


    # ========================================================
    # DERECHA
    # ========================================================

    with right:

        st.header(
            f"⚽ {home_name} vs {away_name}"
        )

        st.caption(
            f"🏆 {competition}"
        )

        # ====================================================
        # CALIDAD
        # ====================================================

        st.subheader(
            "📡 Calidad de los datos"
        )

        quality_rows = []

        for stat_name in [
            "Goles",
            "xG",
            "xGA",
            "Remates",
            "SOT",
            "Precisión",
            "Córners",
            "Tarjetas"
        ]:

            if stat_name not in parsed:

                quality_rows.append({

                    "Dato":
                        stat_name,

                    "LOCAL":
                        "N/D",

                    "VISITANTE":
                        "N/D",

                    "Estado":
                        "⚪ Sin datos"
                })

                continue

            d = parsed[
                stat_name
            ]

            home_count = available_count(
                d["home_team"]
            )

            away_count = available_count(
                d["away_team"]
            )

            quality_rows.append({

                "Dato":
                    stat_name,

                "LOCAL":
                    f"{home_count}/10",

                "VISITANTE":
                    f"{away_count}/10",

                "Estado":
                    sample_status(
                        d["home_team"]
                    )
            })

        quality_df = pd.DataFrame(
            quality_rows
        )

        st.dataframe(
            quality_df,
            hide_index=True,
            use_container_width=True
        )


        # ====================================================
        # GOLES
        # ====================================================

        home_expected_goals = None
        away_expected_goals = None

        if "Goles" in parsed:

            d = parsed[
                "Goles"
            ]

            home_goals = d[
                "home_team"
            ]

            home_conceded = d[
                "home_rival"
            ]

            away_goals = d[
                "away_team"
            ]

            away_conceded = d[
                "away_rival"
            ]

            base_home = adjusted_expected(
                home_goals,
                away_conceded,
                tournament["goals"]
            )

            base_away = adjusted_expected(
                away_goals,
                home_conceded,
                tournament["goals"]
            )

            # ------------------------------------------------
            # Refinar con xG/xGA si existen
            # ------------------------------------------------

            if (
                "xG" in parsed
                and "xGA" in parsed
            ):

                xg = parsed[
                    "xG"
                ]

                xga = parsed[
                    "xGA"
                ]

                home_xg = avg(
                    xg["home_team"]
                )

                away_xg = avg(
                    xg["away_team"]
                )

                home_xga = avg(
                    xga["home_team"]
                )

                away_xga = avg(
                    xga["away_team"]
                )

                if (
                    home_xg is not None
                    and away_xga is not None
                    and base_home is not None
                ):

                    home_expected_goals = (
                        0.45 * home_xg
                        + 0.30 * away_xga
                        + 0.25 * base_home
                    )

                else:

                    home_expected_goals = (
                        base_home
                    )

                if (
                    away_xg is not None
                    and home_xga is not None
                    and base_away is not None
                ):

                    away_expected_goals = (
                        0.45 * away_xg
                        + 0.30 * home_xga
                        + 0.25 * base_away
                    )

                else:

                    away_expected_goals = (
                        base_away
                    )

            else:

                home_expected_goals = (
                    base_home
                )

                away_expected_goals = (
                    base_away
                )

            st.divider()

            st.header(
                "⚽ Goles"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    home_name,
                    fmt(
                        avg(home_goals)
                    )
                )

                st.caption(
                    "Recibe: "
                    + fmt(
                        avg(home_conceded)
                    )
                )

            with c2:

                st.metric(
                    "Modelo",
                    fmt(
                        home_expected_goals
                        + away_expected_goals
                    )
                )

                st.caption(
                    f"{home_name}: "
                    f"{fmt(home_expected_goals)}"
                )

                st.caption(
                    f"{away_name}: "
                    f"{fmt(away_expected_goals)}"
                )

            with c3:

                st.metric(
                    away_name,
                    fmt(
                        avg(away_goals)
                    )
                )

                st.caption(
                    "Recibe: "
                    + fmt(
                        avg(away_conceded)
                    )
                )


        # ====================================================
        # XG / XGA
        # ====================================================

        st.divider()

        st.header(
            "📈 xG / xGA"
        )

        if "xG" in parsed:

            d = parsed[
                "xG"
            ]

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    f"{home_name} xG",
                    fmt(
                        avg(d["home_team"])
                    )
                )

                st.caption(
                    "Datos disponibles: "
                    f"{available_count(d['home_team'])}/10"
                )

            with c2:

                st.metric(
                    f"{away_name} xG",
                    fmt(
                        avg(d["away_team"])
                    )
                )

                st.caption(
                    "Datos disponibles: "
                    f"{available_count(d['away_team'])}/10"
                )

        else:

            st.info(
                "xG no disponible. "
                "El análisis continúa normalmente."
            )

        if "xGA" in parsed:

            d = parsed[
                "xGA"
            ]

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    f"{home_name} xGA",
                    fmt(
                        avg(d["home_team"])
                    )
                )

                st.caption(
                    "Datos disponibles: "
                    f"{available_count(d['home_team'])}/10"
                )

            with c2:

                st.metric(
                    f"{away_name} xGA",
                    fmt(
                        avg(d["away_team"])
                    )
                )

                st.caption(
                    "Datos disponibles: "
                    f"{available_count(d['away_team'])}/10"
                )

        else:

            st.caption(
                "xGA no disponible."
            )


        # ====================================================
        # RESULTADO
        # ====================================================

        if (
            home_expected_goals is not None
            and away_expected_goals is not None
        ):

            st.divider()

            st.header(
                "📊 Resultado probable"
            )

            (
                p_home,
                p_draw,
                p_away,
                score_probs
            ) = result_probabilities(
                home_expected_goals,
                away_expected_goals
            )

            fig = result_chart(
                p_home,
                p_draw,
                p_away,
                home_name,
                away_name
            )

            st.pyplot(
                fig,
                clear_figure=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    f"1 — {home_name}",
                    pct(
                        p_home * 100
                    )
                )

            with c2:

                st.metric(
                    "X — Empate",
                    pct(
                        p_draw * 100
                    )
                )

            with c3:

                st.metric(
                    f"2 — {away_name}",
                    pct(
                        p_away * 100
                    )
                )

            # ----------------------------------------------
            # MARCADORES
            # ----------------------------------------------

            if score_probs:

                st.subheader(
                    "🎯 Marcadores más probables"
                )

                score_df = pd.DataFrame(
                    score_probs[:8],
                    columns=[
                        "Marcador",
                        "Probabilidad"
                    ]
                )

                score_df[
                    "Probabilidad"
                ] = (
                    score_df[
                        "Probabilidad"
                    ]
                    * 100
                ).round(1)

                st.dataframe(
                    score_df,
                    hide_index=True,
                    use_container_width=True
                )


        # ====================================================
        # REMATES
        # ====================================================

        if "Remates" in parsed:

            show_market_data = parsed[
                "Remates"
            ]

            show_market(
                title="Remates totales",
                icon="🎯",
                data=show_market_data,
                home_name=home_name,
                away_name=away_name,
                lines=[
                    4.5,
                    6.5,
                    8.5,
                    10.5,
                    12.5,
                    14.5,
                    16.5,
                    18.5,
                    20.5,
                    22.5
                ],
                tournament_value=
                    tournament["shots"]
            )


        # ====================================================
        # SOT
        # ====================================================

        if "SOT" in parsed:

            show_market(
                title="Tiros al arco",
                icon="🥅",
                data=parsed[
                    "SOT"
                ],
                home_name=home_name,
                away_name=away_name,
                lines=[
                    2.5,
                    3.5,
                    4.5,
                    5.5,
                    6.5,
                    7.5,
                    8.5,
                    9.5,
                    10.5
                ],
                tournament_value=
                    tournament["sot"]
            )


        # ====================================================
        # PRECISIÓN
        # ====================================================

        if "Precisión" in parsed:

            d = parsed[
                "Precisión"
            ]

            home_precision = avg(
                d["home_team"]
            )

            away_precision = avg(
                d["away_team"]
            )

            st.divider()

            st.header(
                "🎯 Precisión de tiro"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    home_name,
                    pct(
                        home_precision
                    )
                )

                st.caption(
                    "Muestra: "
                    f"{available_count(d['home_team'])}/10"
                )

            with c2:

                if (
                    home_precision is not None
                    and away_precision is not None
                ):

                    precision_average = (
                        home_precision
                        + away_precision
                    ) / 2

                else:

                    precision_average = None

                st.metric(
                    "Promedio",
                    pct(
                        precision_average
                    )
                )

            with c3:

                st.metric(
                    away_name,
                    pct(
                        away_precision
                    )
                )

                st.caption(
                    "Muestra: "
                    f"{available_count(d['away_team'])}/10"
                )


        # ====================================================
        # REMATES → SOT
        # ====================================================

        if (
            "Remates" in parsed
            and "SOT" in parsed
        ):

            shots = parsed[
                "Remates"
            ]

            sot = parsed[
                "SOT"
            ]

            home_shots = avg(
                shots["home_team"]
            )

            away_shots = avg(
                shots["away_team"]
            )

            home_sot = avg(
                sot["home_team"]
            )

            away_sot = avg(
                sot["away_team"]
            )

            home_ratio = None
            away_ratio = None

            if (
                home_shots is not None
                and home_shots > 0
                and home_sot is not None
            ):

                home_ratio = (
                    home_sot
                    / home_shots
                    * 100
                )

            if (
                away_shots is not None
                and away_shots > 0
                and away_sot is not None
            ):

                away_ratio = (
                    away_sot
                    / away_shots
                    * 100
                )

            st.divider()

            st.header(
                "🔬 Remates → SOT"
            )

            c1, c2 = st.columns(2)

            with c1:

                st.subheader(
                    home_name
                )

                st.write(
                    f"**{fmt(home_shots)}** remates"
                )

                st.write(
                    "↓"
                )

                st.write(
                    f"**{fmt(home_sot)}** SOT"
                )

                st.caption(
                    "Relación: "
                    + pct(
                        home_ratio
                    )
                )

            with c2:

                st.subheader(
                    away_name
                )

                st.write(
                    f"**{fmt(away_shots)}** remates"
                )

                st.write(
                    "↓"
                )

                st.write(
                    f"**{fmt(away_sot)}** SOT"
                )

                st.caption(
                    "Relación: "
                    + pct(
                        away_ratio
                    )
                )


        # ====================================================
        # CÓRNERS
        # ====================================================

        if "Córners" in parsed:

            show_market(
                title="Córners",
                icon="🚩",
                data=parsed[
                    "Córners"
                ],
                home_name=home_name,
                away_name=away_name,
                lines=[
                    3.5,
                    5.5,
                    7.5,
                    9.5,
                    11.5,
                    13.5,
                    15.5,
                    17.5
                ],
                tournament_value=
                    tournament["corners"]
            )


        # ====================================================
        # TARJETAS
        # ====================================================

        if "Tarjetas" in parsed:

            show_market(
                title="Tarjetas amarillas",
                icon="🟨",
                data=parsed[
                    "Tarjetas"
                ],
                home_name=home_name,
                away_name=away_name,
                lines=[
                    1.5,
                    2.5,
                    3.5,
                    4.5,
                    5.5,
                    6.5,
                    7.5
                ],
                tournament_value=
                    tournament["cards"]
            )


        # ====================================================
        # CENTRO DE DECISIÓN
        # ====================================================

        st.divider()

        st.header(
            "🧠 Centro de decisión"
        )

        positive_signals = []
        warnings = []

        # ----------------------------------------------------
        # REMATES
        # ----------------------------------------------------

        if "Remates" in parsed:

            d = parsed[
                "Remates"
            ]

            home_team = d[
                "home_team"
            ]

            away_team = d[
                "away_team"
            ]

            home_average = avg(
                home_team
            )

            home_median = med(
                home_team
            )

            away_average = avg(
                away_team
            )

            away_median = med(
                away_team
            )

            if (
                home_average is not None
                and home_median is not None
                and home_average
                - home_median >= 2
            ):

                warnings.append(
                    f"{home_name}: "
                    "la media de remates puede "
                    "estar inflada."
                )

            if (
                away_average is not None
                and away_median is not None
                and away_average
                - away_median >= 2
            ):

                warnings.append(
                    f"{away_name}: "
                    "la media de remates puede "
                    "estar inflada."
                )

            home_score = consistency_score(
                home_team,
                8.5
            )

            away_score = consistency_score(
                away_team,
                8.5
            )

            if home_score >= 70:

                positive_signals.append(
                    f"🎯 {home_name}: "
                    "perfil consistente de remates."
                )

            if away_score >= 70:

                positive_signals.append(
                    f"🎯 {away_name}: "
                    "perfil consistente de remates."
                )


        # ----------------------------------------------------
        # SOT
        # ----------------------------------------------------

        if "SOT" in parsed:

            d = parsed[
                "SOT"
            ]

            home_average = avg(
                d["home_team"]
            )

            home_median = med(
                d["home_team"]
            )

            away_average = avg(
                d["away_team"]
            )

            away_median = med(
                d["away_team"]
            )

            if (
                home_average is not None
                and home_median is not None
                and home_average
                - home_median >= 1.5
            ):

                warnings.append(
                    f"{home_name}: "
                    "la media de SOT está "
                    "por encima de la mediana."
                )

            if (
                away_average is not None
                and away_median is not None
                and away_average
                - away_median >= 1.5
            ):

                warnings.append(
                    f"{away_name}: "
                    "la media de SOT está "
                    "por encima de la mediana."
                )


        # ----------------------------------------------------
        # PRECISIÓN
        # ----------------------------------------------------

        if "Precisión" not in parsed:

            warnings.append(
                "No hay datos suficientes de precisión."
            )


        # ----------------------------------------------------
        # XG
        # ----------------------------------------------------

        if "xG" not in parsed:

            warnings.append(
                "No hay xG. "
                "Se utilizan goles y estadísticas de volumen."
            )


        # ----------------------------------------------------
        # XGA
        # ----------------------------------------------------

        if "xGA" not in parsed:

            warnings.append(
                "No hay xGA. "
                "La parte defensiva de xG tiene menor información."
            )


        # ----------------------------------------------------
        # MOSTRAR
        # ----------------------------------------------------

        if positive_signals:

            for signal in positive_signals:

                st.success(
                    signal
                )

        if warnings:

            for warning in warnings:

                st.warning(
                    warning
                )

        if not positive_signals:

            st.info(
                "No aparecen señales suficientemente fuertes "
                "como para considerar una tendencia clara."
            )

        st.caption(
            "La máquina utiliza solamente los datos disponibles "
            "y evita rellenar datos faltantes con ceros."
        )


        # ====================================================
        # ====================================================
        # INFORME RESUMIDO / TXT
        # ====================================================

        st.divider()

        st.header(
            "📄 Informe resumido"
        )

        report = StringIO()

        report.write(
            f"TORNEO ANALIZADO: {competition}\n\n"
        )

        report.write(
            f"PARTIDO: {home_name} vs {away_name}\n\n"
        )

        report.write(
            "PROBABILIDADES 1X2:\n"
        )

        report.write(
            f"Gana Local: {p_home * 100:.1f}%\n"
        )

        report.write(
            f"Empate: {p_draw * 100:.1f}%\n"
        )

        report.write(
            f"Gana Visitante: {p_away * 100:.1f}%\n\n"
        )

        report.write(
            "TOP 3 MARCADORES EXACTOS (MAPA DE CALOR):\n"
        )

        for score, probability in score_probs[:3]:

            home_score, away_score = score.split("-")

            report.write(
                f"Local {home_score} - "
                f"{away_score} Visitante "
                f"({probability * 100:.2f}%)\n"
            )

        report.write(
            "\n"
        )

        # ----------------------------------------------------
        # GOLES
        # ----------------------------------------------------

        total_goals_projection = None

        if (
            home_expected_goals is not None
            and away_expected_goals is not None
        ):

            total_goals_projection = (
                home_expected_goals
                + away_expected_goals
            )

        report.write(
            "CANTIDAD ESTIMADA DE GOLES:\n"
        )

        report.write(
            f"Promedio total del partido: "
            f"{fmt(total_goals_projection)} goles.\n\n"
        )

        # ----------------------------------------------------
        # MÉTRICAS TOTALES
        # ----------------------------------------------------

        report.write(
            "MÉTRICAS TOTALES DEL ENCUENTRO:\n"
        )

        projected_remates = None
        projected_sot = None
        projected_corners = None
        projected_cards = None

        if "Remates" in parsed:

            d = parsed["Remates"]

            home_projection = adjusted_expected(
                d["home_team"],
                d["away_rival"],
                tournament_average=tournament["shots"]
            )

            away_projection = adjusted_expected(
                d["away_team"],
                d["home_rival"],
                tournament_average=tournament["shots"]
            )

            if (
                home_projection is not None
                and away_projection is not None
            ):

                projected_remates = (
                    home_projection
                    + away_projection
                )

        if "SOT" in parsed:

            d = parsed["SOT"]

            home_projection = adjusted_expected(
                d["home_team"],
                d["away_rival"],
                tournament_average=tournament["sot"]
            )

            away_projection = adjusted_expected(
                d["away_team"],
                d["home_rival"],
                tournament_average=tournament["sot"]
            )

            if (
                home_projection is not None
                and away_projection is not None
            ):

                projected_sot = (
                    home_projection
                    + away_projection
                )

        if "Córners" in parsed:

            d = parsed["Córners"]

            home_projection = adjusted_expected(
                d["home_team"],
                d["away_rival"],
                tournament_average=tournament["corners"]
            )

            away_projection = adjusted_expected(
                d["away_team"],
                d["home_rival"],
                tournament_average=tournament["corners"]
            )

            if (
                home_projection is not None
                and away_projection is not None
            ):

                projected_corners = (
                    home_projection
                    + away_projection
                )

        if "Tarjetas" in parsed:

            d = parsed["Tarjetas"]

            home_projection = adjusted_expected(
                d["home_team"],
                d["away_rival"],
                tournament_average=tournament["cards"]
            )

            away_projection = adjusted_expected(
                d["away_team"],
                d["home_rival"],
                tournament_average=tournament["cards"]
            )

            if (
                home_projection is not None
                and away_projection is not None
            ):

                projected_cards = (
                    home_projection
                    + away_projection
                )

        report.write(
            f"Remates Totales Proyectados: "
            f"{fmt(projected_remates)}\n"
        )

        report.write(
            f"Remates al Arco Proyectados: "
            f"{fmt(projected_sot)}\n"
        )

        report.write(
            f"Corners Totales Estimados: "
            f"{fmt(projected_corners)}\n"
        )

        report.write(
            f"Tarjetas Totales Estimadas: "
            f"{fmt(projected_cards)}\n"
        )

        report.write(
            "\n"
        )

        # ----------------------------------------------------
        # ESTRATEGIA
        # ----------------------------------------------------

        report.write(
            "==================================================\n"
        )

        report.write(
            "ESTRATEGIA RECOMENDADA PARA TU TICKET:\n"
        )

        # Líneas orientativas:
        # se colocan por debajo de la proyección, sin inventar
        # una probabilidad que no fue calculada.

        if projected_sot is not None:

            sot_line = np.floor(
                (projected_sot - 0.5) * 2
            ) / 2

            report.write(
                f"Remates al Arco: "
                f"Sugerido Over {sot_line:.1f}\n"
            )

        else:

            report.write(
                "Remates al Arco: Sin datos suficientes\n"
            )

        if projected_corners is not None:

            corners_line = np.floor(
                (projected_corners - 0.5) * 2
            ) / 2

            report.write(
                f"Corners: "
                f"Sugerido Over {corners_line:.1f}\n"
            )

        else:

            report.write(
                "Corners: Sin datos suficientes\n"
            )

        if projected_cards is not None:

            cards_line = np.floor(
                (projected_cards - 0.5) * 2
            ) / 2

            report.write(
                f"Tarjetas: "
                f"Sugerido Over {cards_line:.1f}\n"
            )

        else:

            report.write(
                "Tarjetas: Sin datos suficientes\n"
            )

        report.write(
            "==================================================\n"
        )

        report.write(
            "Nota: este informe resume las proyecciones "
            "del modelo; no garantiza resultados.\n"
        )

        report_text = report.getvalue()

        st.download_button(
            label="⬇️ DESCARGAR INFORME TXT",
            data=report_text,
            file_name=(
                f"{home_name}_vs_"
                f"{away_name}_informe.txt"
            ),
            mime="text/plain",
            use_container_width=True
        )
