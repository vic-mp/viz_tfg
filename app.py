import altair as alt
import pandas as pd
import streamlit as st

from energy_forecast import (
    MADRID,
    MODEL_SPECS,
    build_dashboard_snapshot,
)


st.set_page_config(page_title="Forecast renovable", layout="wide")


def build_chart(series_snapshot) -> alt.Chart:
    """
    Construye el gráfico de histórico y predicción.

    Args:
        series_snapshot: salida de la predicción.

    Returns:
        alt.Chart: Gráfico Altair.
    """
    actual = series_snapshot.actual.assign(series="Real")
    forecast = series_snapshot.forecast.loc[:, ["date", "value_mw"]].assign(series="Predicción")
    chart_frame = pd.concat([actual, forecast], ignore_index=True)
    chart_frame["timestamp_local"] = chart_frame["date"].dt.tz_convert(MADRID).dt.tz_localize(None)

    forecast_start = pd.DataFrame(
        {
            "timestamp_local": [
                series_snapshot.forecast["date"].dt.tz_convert(MADRID).dt.tz_localize(None).iloc[0]
            ]
        }
    )

    line = (
        alt.Chart(chart_frame)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("timestamp_local:T", title="Hora "),
            y=alt.Y("value_mw:Q", title="Generación (MW)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Real", "Predicción"],
                    range=["#1f77b4", "#d62728"],
                ),
                title="Serie",
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(
                    domain=["Real", "Predicción"],
                    range=[[1, 0], [6, 4]],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("timestamp_local:T", title="Hora"),
                alt.Tooltip("series:N", title="Serie"),
                alt.Tooltip("value_mw:Q", title="MW", format=",.1f"),
            ],
        )
    )

    split = (
        alt.Chart(forecast_start)
        .mark_rule(color="#6b7280", strokeDash=[4, 4])
        .encode(x="timestamp_local:T")
    )

    return (line + split).properties(height=380)


def format_metrics_frame() -> pd.DataFrame:
    """
    Crea un DataFrame con las métricas para mostrarlo en la interfaz.

    Returns:
        pd.DataFrame: DataFrame con las métricas de entrenamiento.
    """
    return pd.DataFrame(
        {
            "Serie": spec.label,
            "MAE": spec.metrics.mae_scaled,
            "RMSE": spec.metrics.rmse_scaled,
            "SMAPE": spec.metrics.smape,
            "Best val_loss": spec.metrics.best_val_loss,
        }
        for spec in MODEL_SPECS.values()
    )


st.title("Forecast de generación renovable")
st.caption(
    f"Predicción horaria a 48 h de generación eólica y solar en España a partir de modelos NHITS,"
    f" Utilizando datos en directo de Red Eléctrica de España y OpenMeteo."
)

with st.sidebar:
    selected_key = st.selectbox(
        "Tecnología",
        options=list(MODEL_SPECS),
        format_func=lambda key: MODEL_SPECS[key].label,
    )
    st.button("Refrescar datos", help="Vuelve a consultar las APIs.")

try:
    with st.spinner("Descargando datos y ejecutando el modelo..."):
        series_by_key = build_dashboard_snapshot()
except Exception as exc:
    st.error("No se pudo construir el dashboard. Prueba a refrescar la página.")
    st.exception(exc)
    st.stop()

selected = series_by_key[selected_key]
selected_spec = selected.spec
local_last_actual = selected.last_actual_at.tz_convert(MADRID).tz_localize(None)
forecast_peak = selected.forecast["value_mw"].max()

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
metric_col_1.metric("Tecnología", selected_spec.label)
metric_col_2.metric("Último dato real", local_last_actual.strftime("%Y-%m-%d %H:%M"))
metric_col_3.metric("Ventana de predicción", "48 h")
metric_col_4.metric("Pico previsto (48 h)", f"{forecast_peak:,.0f} MW")

st.altair_chart(build_chart(selected), width="stretch")

st.subheader("Métricas de entrenamiento")
st.caption(
    "Métricas obtenidas tras validar el modelo después del entrenamiento."
)

st.dataframe(
    format_metrics_frame(),
    width="stretch",
    hide_index=True,
    column_config={
        "MAE": st.column_config.NumberColumn(format="%.4f"),
        "RMSE": st.column_config.NumberColumn(format="%.4f"),
        "SMAPE": st.column_config.NumberColumn(format="%.4f"),
        "Loss": st.column_config.NumberColumn(format="%.4f"),
    },
)
