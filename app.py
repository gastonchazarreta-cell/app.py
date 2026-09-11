import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
from google import genai
from google.genai import types

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(layout="wide", page_title="Value Analytics Pro", page_icon="⚽")
st.title("⚽ Predictor Avanzado con Ajuste por Oposición e Índice Arbitral")

# --- BASE DE DATOS DE COMPETICIONES (MEDIAS DE REFERENCIA) ---
BASE_DE_COMPETICIONES = {
    "Liga Profesional (Argentina)": {
        "goles": 1.95, "tiros_tot": 22.0, "tiros_arc": 8.0, "corners": 8.70, "tarjetas": 5.20,
    },
    "Premier League (Inglaterra)": {
        "goles": 3.15, "tiros_tot": 26.5, "tiros_arc": 9.5, "corners": 10.15, "tarjetas": 3.90,
    },
    "LaLiga (España)": {
        "goles": 2.65, "tiros_tot": 24.0, "tiros_arc": 8.5, "corners": 9.95, "tarjetas": 4.60,
    },
    "Serie A (Italia)": {
        "goles": 2.55, "tiros_tot": 23.5, "tiros_arc": 8.2, "corners": 9.45, "tarjetas": 4.30,
    },
    "UEFA Champions League": {
        "goles": 3.40, "tiros_tot": 26.0, "tiros_arc": 9.8, "corners": 9.75, "tarjetas": 4.10,
    },
    "Copa Libertadores": {
        "goles": 2.35, "tiros_tot": 21.5, "tiros_arc": 7.8, "corners": 9.30, "tarjetas": 5.80,
    },
    "Copa Sudamericana": {
        "goles": 2.45, "tiros_tot": 22.0, "tiros_arc": 8.0, "corners": 9.50, "tarjetas": 5.60,
    },
}

# --- INICIALIZACIÓN DE VARIABLES EN SESSION_STATE ---
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

campos = [
    "xg_sin_penal_loc", "xg_sin_penal_vis",
    "rend_tiro_loc", "rend_tiro_vis",
    "remates_tot_loc", "remates_tot_vis",
    "remates_concedidos_loc", "remates_concedidos_vis",
    "remates_arco_loc", "remates_arco_vis",
    "corners_loc", "corners_vis",
    "corners_concedidos_loc", "corners_concedidos_vis",
    "tarjetas_loc", "tarjetas_vis",
    "tarjetas_provocadas_loc", "tarjetas_provocadas_vis",
    "promedio_arbitro"
]

for campo in campos:
    if campo not in st.session_state:
        st.session_state[campo] = 0.0

if "nombre_local" not in st.session_state:
    st.session_state["nombre_local"] = "Local"
if "nombre_visitante" not in st.session_state:
    st.session_state["nombre_visitante"] = "Visitante"

# --- FUNCIÓN PARA LIMPIAR TODO EL ESTADO ---
def limpiar_formulario():
    for campo in campos:
        st.session_state[campo] = 0.0
    st.session_state["nombre_local"] = "Local"
    st.session_state["nombre_visitante"] = "Visitante"

# --- SELECCIÓN DE COMPETICIÓN PRINCIPAL ---
st.subheader("🏆 Contexto de Competición")
liga_seleccionada = st.selectbox(
    "Selecciona la liga o torneo a analizar:",
    options=list(BASE_DE_COMPETICIONES.keys()),
)
medias_liga = BASE_DE_COMPETICIONES[liga_seleccionada]

# --- FUNCIONES DE CÁLCULO ESTADÍSTICO AVANZADO ---
def modelo_dixon_coles(lambda_loc, lambda_vis, marcador_i, marcador_j, rho=-0.06):
    """Modelo matemático Dixon-Coles para control probabilístico de goles."""
    prob_base = stats.poisson.pmf(marcador_i, lambda_loc) * stats.poisson.pmf(marcador_j, lambda_vis)
    if marcador_i == 0 and marcador_j == 0:
        return prob_base * (1 - lambda_loc * lambda_vis * rho)
    elif marcador_i == 1 and marcador_j == 0:
        return prob_base * (1 + lambda_vis * rho)
    elif marcador_i == 0 and marcador_j == 1:
        return prob_base * (1 + lambda_loc * rho)
    elif marcador_i == 1 and marcador_j == 1:
        return prob_base * (1 - rho)
    return prob_base

def analizar_partido(datos_usuario, medias):
    datos_limpios = datos_usuario.copy()
    media_eq_goles = medias["goles"] / 2
    media_eq_tiros = medias["tiros_tot"] / 2
    media_eq_arcos = medias["tiros_arc"] / 2
    media_eq_corners = medias["corners"] / 2
    media_eq_tarjetas = medias["tarjetas"] / 2

    for k in ["xg_sin_penal_loc", "xg_sin_penal_vis"]:
        if datos_limpios[k] <= 0: datos_limpios[k] = media_eq_goles
    for k in ["remates_tot_loc", "remates_tot_vis", "remates_concedidos_loc", "remates_concedidos_vis"]:
        if datos_limpios[k] <= 0: datos_limpios[k] = media_eq_tiros
    for k in ["remates_arco_loc", "remates_arco_vis"]:
        if datos_limpios[k] <= 0: datos_limpios[k] = media_eq_arcos
    for k in ["corners_loc", "corners_vis", "corners_concedidos_loc", "corners_concedidos_vis"]:
        if datos_limpios[k] <= 0: datos_limpios[k] = media_eq_corners
    for k in ["tarjetas_loc", "tarjetas_vis", "tarjetas_provocadas_loc", "tarjetas_provocadas_vis"]:
        if datos_limpios[k] <= 0: datos_limpios[k] = media_eq_tarjetas
    if datos_limpios["rend_tiro_loc"] <= 0: datos_limpios["rend_tiro_loc"] = 1.0
    if datos_limpios["rend_tiro_vis"] <= 0: datos_limpios["rend_tiro_vis"] = 1.0

    goles_esperados_loc = datos_limpios["xg_sin_penal_loc"] * datos_limpios["rend_tiro_loc"]
    goles_esperados_vis = datos_limpios["xg_sin_penal_vis"] * datos_limpios["rend_tiro_vis"]

    index_corners_loc = datos_limpios["corners_loc"] * (datos_limpios["corners_concedidos_vis"] / media_eq_corners)
    index_corners_vis = datos_limpios["corners_vis"] * (datos_limpios["corners_concedidos_loc"] / media_eq_corners)
    
    mult_l = 1.05 if datos_limpios["remates_arco_loc"] > media_eq_arcos else 0.95
    mult_v = 1.05 if datos_limpios["remates_arco_vis"] > media_eq_arcos else 0.95
    prom_corners_total = (index_corners_loc * mult_l) + (index_corners_vis * mult_v)

    index_tarjetas_loc = datos_limpios["tarjetas_loc"] * (datos_limpios["tarjetas_provocadas_vis"] / media_eq_tarjetas)
    index_tarjetas_vis = datos_limpios["tarjetas_vis"] * (datos_limpios["tarjetas_provocadas_loc"] / media_eq_tarjetas)
    prom_equipos = index_tarjetas_loc + index_tarjetas_vis

    prom_ref_arbitro = datos_limpios["promedio_arbitro"] if datos_limpios["promedio_arbitro"] > 0 else medias["tarjetas"]
    prom_tarjetas_total = (prom_equipos * 0.5) + (prom_ref_arbitro * 0.5)

    max_goles = 6
    resultados = []
    for i in range(max_goles):
        for j in range(max_goles):
            p = modelo_dixon_coles(goles_esperados_loc, goles_esperados_vis, i, j)
            resultados.append({"marcador": f"{i}-{j}", "probabilidad": p * 100})

    df_res = pd.DataFrame(resultados).sort_values(by="probabilidad", ascending=False)
    top_3 = df_res.head(3).to_dict(orient="records")

    return {
        "goles_loc": goles_esperados_loc, "goles_vis": goles_esperados_vis,
        "top_3": top_3, "corners": prom_corners_total, "tarjetas": prom_tarjetas_total,
        "datos_finales_usados": datos_limpios
    }

# --- DISEÑO DE LA INTERFAZ ---
col_ia, col_inputs = st.columns([1, 1.2])

with col_ia:
    st.subheader("🤖 Entrada Inteligente Multimodal")
    api_key_input = st.text_input("Introduce tu Gemini API Key:", type="password", value=st.session_state.api_key)
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("⚠️ **Sube capturas de pantalla separadas para cada equipo:**")
    img_local = st.file_uploader("📥 Captura del Equipo LOCAL", type=["png", "jpg", "jpeg"])
    img_vis = st.file_uploader("📥 Captura del Equipo VISITANTE", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    st.markdown("🏁 **Filtro Arbitral**")
    tiene_arbitro = st.checkbox("¿Tienes los datos del árbitro designado?", value=False)
    
    if tiene_arbitro:
        st.number_input(
            "Promedio histórico de tarjetas del árbitro:", 
            key="promedio_arbitro", 
            step=0.1, 
            min_value=0.0
        )
    else:
        st.session_state["promedio_arbitro"] = 0.0

    st.markdown("---")
    
    if img_local and img_vis and st.session_state.api_key:
        if st.button("🚀 PROCESAR AMBAS CAPTURAS", use_container_width=True):
            try:
                client = genai.Client(api_key=st.session_state.api_key)
                
                bytes_loc = img_local.getvalue()
                part_loc = types.Part.from_bytes(data=bytes_loc, mime_type=img_local.type)
                
                prompt_base = """
                Analiza esta imagen de estadísticas de fútbol de un equipo. Extrae los valores numéricos de rendimiento.
                Atención crucial: Extrae el número principal (el grande) y el número de oposición del rival (el número pequeño que se encuentra arriba o al lado de cada estadística principal). 
                Si algún campo contiene un guión (-), está en blanco o no es visible, escribe estrictamente 0.0.

                Extrae exclusivamente esta estructura exacta en texto plano clave:valor:
                {sufijo}_xg_sin_penal:valor de Goles Esperados sin penalti (o xG general)
                {sufijo}_rend_tiro:valor de rendimiento de tiro (0.0 si no existe)
                {sufijo}_remates_tot:valor de Tiros/Remates totales (Número Grande)
                {sufijo}_remates_concedidos:valor de Tiros/Remates de los rivales (Número Pequeño)
                {sufijo}_remates_arco:valor de Tiros/Remates a puerta (Número Grande)
                {sufijo}_corners:valor de Córners a favor (Número Grande)
                {sufijo}_corners_concedidos:valor de Córners de los rivales (Número Pequeño)
                {sufijo}_tarjetas:valor de Tarjetas amarillas recibidas (Número Grande)
                {sufijo}_tarjetas_provocadas:valor de Tarjetas amarillas del rival (Número Pequeño)
                """

                res_loc = client.models.generate_content(
                    model="gemini-3.6-flash", contents=[part_loc, prompt_base.format(sufijo="local")]
                )
                
                bytes_vis = img_vis.getvalue()
                part_vis = types.Part.from_bytes(data=bytes_vis, mime_type=img_vis.type)
                res_vis = client.models.generate_content(
                    model="gemini-3.6-flash", contents=[part_vis, prompt_base.format(sufijo="visitante")]
                )

                mapa_claves = {
                    "local_xg_sin_penal": "xg_sin_penal_loc", "visitante_xg_sin_penal": "xg_sin_penal_vis",
                    "local_rend_tiro": "rend_tiro_loc", "visitante_rend_tiro": "rend_tiro_vis",
                    "local_remates_tot": "remates_tot_loc", "visitante_remates_tot": "remates_tot_vis",
                    "local_remates_concedidos": "remates_concedidos_loc", "visitante_remates_concedidos": "remates_concedidos_vis",
                    "local_remates_arco": "remates_arco_loc", "visitante_remates_arco": "remates_arco_vis",
                    "local_corners": "corners_loc", "visitante_corners": "corners_vis",
                    "local_corners_concedidos": "corners_concedidos_loc", "visitante_corners_concedidos": "corners_concedidos_vis",
                    "local_tarjetas": "tarjetas_loc", "visitante_tarjetas": "tarjetas_vis",
                    "local_tarjetas_provocadas": "tarjetas_provocadas_loc", "visitante_tarjetas_provocadas": "tarjetas_provocadas_vis",
                }
                
                for respuesta_texto in [res_loc.text, res_vis.text]:
                    for linea in respuesta_texto.strip().split("\n"):
                        if ":" in linea:
                            c, v = linea.split(":")
                            c = c.strip()
                            if c in mapa_claves:
                                try:
                                    st.session_state[mapa_claves[c]] = float(v.strip())
                                except ValueError:
                                    st.session_state[mapa_claves[c]] = 0.0
                                    
                st.success("🎉 ¡Ambos equipos sincronizados! Revisa los valores a la derecha.")
            except Exception as e:
                st.error(f"Error procesando la visión artificial: {e}")
                
    elif (img_local or img_vis) and not st.session_state.api_key:
        st.warning("🔒 Introduce la clave de la API de Gemini para desbloquear la lectura automática.")

    st.button("🔄 REINICIAR Y LIMPIAR PÁGINA", on_click=limpiar_formulario, use_container_width=True)

with col_inputs:
    st.subheader("📊 Panel de Control Operativo")
    
    col_name_l, col_name_v = st.columns(2)
    with col_name_l:
        nombre_local = st.text_input("✍️ Nombre Equipo LOCAL:", key="nombre_local")
    with col_name_v:
        nombre_visitante = st.text_input("✍️ Nombre Equipo VISITANTE:", key="nombre_visitante")
        
    st.markdown("---")
    
    col_l, col_v = st.columns(2)
    with col_l:
        st.markdown(f"🏡 DATOS {nombre_local.upper()}")
        st.number_input("xG Sin Penalti", key="xg_sin_penal_loc", step=0.1, min_value=0.0)
        st.number_input("Rendimiento de Tiro", key="rend_tiro_loc", step=0.05, min_value=0.0)
        st.number_input("Remates Totales", key="remates_tot_loc", step=1.0, min_value=0.0)
        st.number_input("Remates Concedidos (Gris)", key="remates_concedidos_loc", step=1.0, min_value=0.0)
        st.number_input("Remates al Arco", key="remates_arco_loc", step=1.0, min_value=0.0)
        st.number_input("Córners Propios", key="corners_loc", step=1.0, min_value=0.0)
        st.number_input("Córners Concedidos (Gris)", key="corners_concedidos_loc", step=1.0, min_value=0.0)
        st.number_input("Tarjetas Propias", key="tarjetas_loc", step=1.0, min_value=0.0)
        st.number_input("Tarjetas Provocadas (Gris)", key="tarjetas_provocadas_loc", step=1.0, min_value=0.0)
    with col_v:
        st.markdown(f"✈️ DATOS {nombre_visitante.upper()}")
        st.number_input("xG Sin Penalti", key="xg_sin_penal_vis", step=0.1, min_value=0.0)
        st.number_input("Rendimiento de Tiro", key="rend_tiro_vis", step=0.05, min_value=0.0)
        st.number_input("Remates Totales", key="remates_tot_vis", step=1.0, min_value=0.0)
        st.number_input("Remates Concedidos (Gris)", key="remates_concedidos_vis", step=1.0, min_value=0.0)
        st.number_input("Remates al Arco", key="remates_arco_vis", step=1.0, min_value=0.0)
        st.number_input("Córners Propios", key="corners_vis", step=1.0, min_value=0.0)
        st.number_input("Córners Concedidos (Gris)", key="corners_concedidos_vis", step=1.0, min_value=0.0)
        st.number_input("Tarjetas Propias", key="tarjetas_vis", step=1.0, min_value=0.0)
        st.number_input("Tarjetas Provocadas (Gris)", key="tarjetas_provocadas_vis", step=1.0, min_value=0.0)

st.markdown("---")
btn_analizar = st.button("📊 EJECUTAR CÁLCULOS DE VALUE", use_container_width=True)

# --- PROCESAMIENTO MATEMÁTICO Y SALIDA DE ESTRATEGIA ---
if btn_analizar:
    datos_actuales = {c: st.session_state[c] for c in campos}
    res = analizar_partido(datos_actuales, medias_liga)
    df_usados = res["datos_finales_usados"]
    
    st.markdown("---")
    st.subheader(f"📈 Proyecciones Estratégicas: {nombre_local} vs {nombre_visitante}")
    
    res_c1, res_c2, res_c3, res_c4 = st.columns(4)
    with res_c1:
        st.metric(label="Goles Proyectados Ajustados", value=f"{res['goles_loc']:.2f} - {res['goles_vis']:.2f}")
    with res_c2:
        st.metric(label="Expectativa de Córners", value=f"{res['corners']:.2f}")
    with res_c3:
        st.metric(label="Expectativa de Tarjetas", value=f"{res['tarjetas']:.2f}")
    with res_c4:
        remates_arco_totales = df_usados['remates_arco_loc'] + df_usados['remates_arco_vis']
        st.metric(label="Remates al Arco Totales", value=f"{remates_arco_totales:.1f}")
        
    st.markdown("#### 🎯 Marcadores Exactos Más Probables (Dixon-Coles)")
    for idx, item in enumerate(res["top_3"]):
        st.markdown(f"Opción {idx+1}: Marcador {item['marcador']} ➡️ Probabilidad Algorítmica: {item['probabilidad']:.2f}%")

    info_arbitro = f"{df_usados['promedio_arbitro']:.2f} (Manual)" if df_usados['promedio_arbitro'] > 0 else f"{medias_liga['tarjetas']:.2f} (Media de Liga por defecto)"

    reporte_txt = f"""==================================================
📊 REPORTE AJUSTADO POR OPOSICIÓN Y CALIDAD DE RIVAL
ENCUENTRO: {nombre_local} vs {nombre_visitante}
Competición Analizada: {liga_seleccionada}
Factor Árbitro Aplicado: {info_arbitro}

MÉTRICAS BASE DEL ENCUENTRO (PROPIAS vs CONCEDIDAS):
{nombre_local.upper()} -> Córners: {df_usados['corners_loc']:.1f} (Hechos) | {df_usados['corners_concedidos_loc']:.1f} (Recibidos de Gris)
{nombre_visitante.upper()} -> Córners: {df_usados['corners_vis']:.1f} (Hechos) | {df_usados['corners_concedidos_vis']:.1f} (Recibidos de Gris)
{nombre_local.upper()} -> Tarjetas: {df_usados['tarjetas_loc']:.1f} (Recibidas) | {df_usados['tarjetas_provocadas_loc']:.1f} (Provocadas de Gris)
{nombre_visitante.upper()} -> Tarjetas: {df_usados['tarjetas_vis']:.1f} (Recibidas) | {df_usados['tarjetas_provocadas_vis']:.1f} (Provocadas de Gris)
{nombre_local.upper()} -> Remates: {df_usados['remates_tot_loc']:.1f} (Totales) | {df_usados['remates_arco_loc']:.1f} (Al Arco)
{nombre_visitante.upper()} -> Remates: {df_usados['remates_tot_vis']:.1f} (Totales) | {df_usados['remates_arco_vis']:.1f} (Al Arco)

PROYECCIONES PURAS DEL MODELO DIXON-COLES Y LÍNEAS DE APUESTA:
Goles Esperados {nombre_local}: {res['goles_loc']:.2f}
Goles Esperados {nombre_visitante}: {res['goles_vis']:.2f}
Total Córners Calculados: {res['corners']:.2f}
Total Tarjetas Calculadas: {res['tarjetas']:.2f}
Total Remates al Arco Estimados: {df_usados['remates_arco_loc'] + df_usados['remates_arco_vis']:.1f}
Total Remates Totales Estimados: {df_usados['remates_tot_loc'] + df_usados['remates_concedidos_vis']:.1f}

TOP 3 MARCADORES RECOMENDADOS PARA EVALUAR CUOTAS:
Marcador: {res['top_3'][0]['marcador']} | Probabilidad: {res['top_3'][0]['probabilidad']:.2f}%
Marcador: {res['top_3'][1]['marcador']} | Probabilidad: {res['top_3'][1]['probabilidad']:.2f}%
Marcador: {res['top_3'][2]['marcador']} | Probabilidad: {res['top_3'][2]['probabilidad']:.2f}%
==================================================
Reporte autogenerado para gestión de valor y cuotas."""

    nombre_archivo_limpio = f"{nombre_local.replace(' ', '')}vs{nombre_visitante.replace(' ', '')}_value.txt"

    st.markdown("---")
    st.download_button(
        label=f"📥 Descargar Informe: {nombre_local} vs {nombre_visitante} (.txt)",
        data=reporte_txt,
        file_name=nombre_archivo_limpio,
        mime="text/plain",
        use_container_width=True
    )
