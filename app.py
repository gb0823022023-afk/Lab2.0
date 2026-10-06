import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import pipeline


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Detección de Sentimiento - ISP",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    """
    Carga un modelo multilingüe de análisis de sentimiento.
    """
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )


clasificador = cargar_modelo()


# ============================================================
# FUNCIÓN DE ANÁLISIS
# ============================================================

def analizar_sentimiento(texto):
    """
    Analiza el sentimiento de un comentario.

    El modelo devuelve una puntuación de 1 a 5 estrellas.
    Se transforma posteriormente a:
        1-2 -> Negativo
        3   -> Neutral
        4-5 -> Positivo
    """

    if not texto or not str(texto).strip():
        return "Neutral", 0.0

    resultado = clasificador(str(texto)[:512])[0]

    estrellas = int(resultado["label"][0])
    confianza = float(resultado["score"])

    if estrellas <= 2:
        sentimiento = "Negativo"
    elif estrellas == 3:
        sentimiento = "Neutral"
    else:
        sentimiento = "Positivo"

    return sentimiento, confianza


# ============================================================
# TÍTULO
# ============================================================

st.title("📡 Sistema de Detección de Sentimiento")
st.subheader("Análisis de opiniones de clientes de servicios de Internet")

st.write(
    """
    Esta aplicación analiza comentarios de clientes y los clasifica
    automáticamente como **Positivos, Negativos o Neutrales**.
    """
)


# ============================================================
# MENÚ LATERAL
# ============================================================

st.sidebar.header("⚙️ Configuración")

opcion = st.sidebar.radio(
    "Seleccione una opción:",
    [
        "Analizar comentario",
        "Analizar archivo CSV",
        "Dashboard"
    ]
)


# ============================================================
# ANALIZAR UN COMENTARIO
# ============================================================

if opcion == "Analizar comentario":

    st.header("💬 Analizar comentario")

    comentario = st.text_area(
        "Ingrese el comentario del cliente:",
        placeholder=(
            "Ejemplo: La velocidad de Internet es excelente "
            "y el servicio técnico respondió rápidamente."
        ),
        height=150
    )

    if st.button("🔍 Analizar sentimiento"):

        if not comentario.strip():
            st.warning("Ingrese un comentario para analizar.")
        else:

            sentimiento, confianza = analizar_sentimiento(comentario)

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                if sentimiento == "Positivo":
                    st.success("😊 Sentimiento: POSITIVO")

                elif sentimiento == "Negativo":
                    st.error("😠 Sentimiento: NEGATIVO")

                else:
                    st.warning("😐 Sentimiento: NEUTRAL")

            with col2:

                st.metric(
                    "Confianza del modelo",
                    f"{confianza * 100:.2f}%"
                )

            st.subheader("Comentario analizado")

            st.info(comentario)


# ============================================================
# ANALIZAR CSV
# ============================================================

elif opcion == "Analizar archivo CSV":

    st.header("📂 Análisis de comentarios desde CSV")

    st.write(
        """
        El archivo CSV debe contener una columna llamada
        **comentario**.
        """
    )

    archivo = st.file_uploader(
        "Seleccione un archivo CSV",
        type=["csv"]
    )

    if archivo is not None:

        try:

            df = pd.read_csv(archivo)

            st.subheader("Vista previa")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            if "comentario" not in df.columns:

                st.error(
                    "El archivo debe contener una columna llamada "
                    "'comentario'."
                )

            else:

                if st.button("🚀 Analizar comentarios"):

                    resultados = []

                    barra = st.progress(0)

                    total = len(df)

                    for i, comentario in enumerate(
                        df["comentario"]
                    ):

                        sentimiento, confianza = \
                            analizar_sentimiento(comentario)

                        resultados.append({
                            "sentimiento": sentimiento,
                            "confianza": confianza
                        })

                        barra.progress(
                            (i + 1) / total
                        )

                    resultados_df = pd.DataFrame(resultados)

                    df["sentimiento"] = \
                        resultados_df["sentimiento"]

                    df["confianza"] = \
                        resultados_df["confianza"]

                    st.session_state["resultados"] = df

                    st.success(
                        "Análisis completado correctamente."
                    )

        except Exception as e:

            st.error(
                f"Error al procesar el archivo: {e}"
            )


# ============================================================
# DASHBOARD
# ============================================================

elif opcion == "Dashboard":

    st.header("📊 Dashboard de satisfacción")

    if "resultados" not in st.session_state:

        st.info(
            "Primero analice un archivo CSV para visualizar "
            "el dashboard."
        )

    else:

        df = st.session_state["resultados"]

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        total = len(df)

        positivos = len(
            df[df["sentimiento"] == "Positivo"]
        )

        negativos = len(
            df[df["sentimiento"] == "Negativo"]
        )

        neutrales = len(
            df[df["sentimiento"] == "Neutral"]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total comentarios",
            total
        )

        col2.metric(
            "😊 Positivos",
            positivos
        )

        col3.metric(
            "😠 Negativos",
            negativos
        )

        col4.metric(
            "😐 Neutrales",
            neutrales
        )

        st.divider()

        # ----------------------------------------------------
        # PORCENTAJES
        # ----------------------------------------------------

        porcentajes = {
            "Positivo": positivos / total * 100 if total else 0,
            "Neutral": neutrales / total * 100 if total else 0,
            "Negativo": negativos / total * 100 if total else 0
        }

        st.subheader("Distribución de sentimientos")

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots()

            etiquetas = list(porcentajes.keys())
            valores = list(porcentajes.values())

            colores = [
                "#2ecc71",
                "#f1c40f",
                "#e74c3c"
            ]

            ax.pie(
                valores,
                labels=etiquetas,
                autopct="%1.1f%%",
                colors=colores,
                startangle=90
            )

            ax.set_title("Sentimiento de clientes")

            st.pyplot(fig)

        with col2:

            fig, ax = plt.subplots()

            sns.barplot(
                x=etiquetas,
                y=valores,
                hue=etiquetas,
                palette=colores,
                legend=False,
                ax=ax
            )

            ax.set_ylabel("Porcentaje (%)")
            ax.set_xlabel("Sentimiento")
            ax.set_title("Distribución porcentual")

            st.pyplot(fig)

        # ----------------------------------------------------
        # TABLA DE RESULTADOS
        # ----------------------------------------------------

        st.subheader("📋 Resultados")

        st.dataframe(
            df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # COMENTARIOS NEGATIVOS
        # ----------------------------------------------------

        st.subheader("🚨 Comentarios negativos")

        negativos_df = df[
            df["sentimiento"] == "Negativo"
        ]

        if len(negativos_df) > 0:

            st.dataframe(
                negativos_df,
                use_container_width=True
            )

        else:

            st.success(
                "No se encontraron comentarios negativos."
            )

        # ----------------------------------------------------
        # DESCARGAR RESULTADOS
        # ----------------------------------------------------

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Descargar resultados CSV",
            data=csv,
            file_name="analisis_sentimientos.csv",
            mime="text/csv"
        )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Sistema de análisis de sentimiento para empresas "
    "proveedoras de servicios de Internet."
)