# ============================================================
# CABECERA
# ============================================================
# Alumno: Óscar Mañero
# URL Streamlit Cloud: https://kzuzv4xpymgvptg2qlgeaj.streamlit.app/
# URL GitHub: https://github.com/omanero-droid/-scarMa-ero_BC5

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un analista de datos experto trabajando con un DataFrame de pandas llamado df que contiene el historial de escucha de Spotify de un usuario.

El dataset contiene información entre {fecha_min} y {fecha_max}.

Columnas disponibles en df:
- ts (datetime)
- master_metadata_track_name (string): nombre de la canción
- master_metadata_album_artist_name (string): artista
- master_metadata_album_album_name (string): álbum
- ms_played (int): milisegundos reproducidos
- minutes_played (float): minutos reproducidos
- hour (int): hora del día (0-23)
- weekday (int): día de la semana (0=lunes, 6=domingo)
- month (int): mes (1-12)
- year (int): año
- is_weekend (bool): si es fin de semana
- season (string): estación (invierno, primavera, verano, otoño)
- shuffle (bool): si estaba activado el modo aleatorio
- skipped (bool): si la canción fue saltada
- platform (string) (valores posibles: {plataformas})

Tu tarea es generar código Python usando pandas y plotly para responder preguntas del usuario sobre sus hábitos de escucha.

REGLAS IMPORTANTES:
- Usa SOLO el DataFrame df
- No inventes columnas
- No uses datos externos
- Siempre crea una variable llamada fig
- Usa plotly (px o go)
- El gráfico debe ser claro, con título y etiquetas

INSTRUCCIONES DE ANÁLISIS:
- Para rankings: usar groupby y sumar minutes_played o contar reproducciones
- Para evolución temporal: agrupar por month o ts
- Para patrones: usar hour, weekday o is_weekend
- Para comportamiento: usar columnas skipped y shuffle
- Para comparaciones: usar season o dividir por periodos

SIMPLIFICACIÓN DEL CÓDIGO:
- Usa operaciones simples de pandas (groupby, sum, count)
- Evita loops (for, while)
- No uses funciones complejas
- No hagas merges ni joins
- El código debe ser corto, claro y directo

TIPOS DE PREGUNTAS QUE DEBES SOPORTAR:
- Rankings (top artistas, canciones)
- Evolución temporal (por mes, tiempo)
- Patrones (horas, días, fin de semana)
- Comportamiento (skips, shuffle)
- Comparaciones (estaciones, periodos)

FORMATO DE RESPUESTA (OBLIGATORIO):

Devuelve SIEMPRE un JSON válido con esta estructura:

{{
  "tipo": "grafico",
  "codigo": "CODIGO PYTHON AQUI",
  "interpretacion": "Explicación breve del resultado"
}}

Si la pregunta no se puede responder con los datos disponibles, responde:

{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "Explicación clara"
}}

INSTRUCCIONES ADICIONALES:
- Limita los rankings a un máximo de 10 elementos
- Ordena los resultados de mayor a menor
- Trata los valores nulos de skipped como False
- Usa nombres claros en los ejes de los gráficos
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    df["ts"] = pd.to_datetime(df["ts"])

    df = df[df["master_metadata_track_name"].notna()]

    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.weekday
    df["month"] = df["ts"].dt.month
    df["year"] = df["ts"].dt.year

    df["is_weekend"] = df["weekday"].isin([5, 6])

    df["minutes_played"] = df["ms_played"] / 60000

    def get_season(month):
        if month in [12, 1, 2]:
            return "invierno"
        elif month in [3, 4, 5]:
            return "primavera"
        elif month in [6, 7, 8]:
            return "verano"
        else:
            return "otoño"

    df["season"] = df["month"].apply(get_season)

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
# La aplicación sigue una arquitectura text-to-code donde el usuario introduce una pregunta en lenguaje natural.
# Esta pregunta se envía junto con el system prompt al LLM, que no recibe los datos directamente, sino una descripción
# de la estructura del dataset. El LLM genera código Python como texto, que posteriormente se ejecuta en local con exec().
# El código trabaja sobre el DataFrame df y genera una figura (fig) que se muestra al usuario.
# El LLM no recibe los datos directamente para evitar los altos costes y mejorar la seguridad.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
# El system prompt proporciona al LLM toda la información sobre las columnas, reglas de uso y el tipo de análisis que espera.
# Por ejemplo, gracias a indicar que debe usar groupby y minutes_played, funciona correctamente una pregunta como
# "Top 5 artistas más escuchados". Si no se especificara esto, el modelo podría usar columnas erróneas o generar un código incorrecto.
# También se incluyen limitaciones para evitar errores y respuestas demasiado complejas.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
# El usuario escribe una pregunta en la interfaz de Streamlit. Esta se envía junto con el system prompt al LLM.
# El modelo devuelve un JSON con código y una interpretación. Este JSON se parsea y el código se ejecuta con exec(),
# generando una figura de Plotly. Finalmente, la app muestra el gráfico y la interpretación correcta al usuario.