import json
import logging
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import torch
from pytorch_forecasting import NHiTS, TimeSeriesDataSet

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Constantes
from forecast_config import (
    COMMUNITIES,
    COMMUNITY_NAMES,
    WEATHER_COLUMNS,
    WEATHER_SUFFIXES,
    WEATHER_VARIABLES,
)
ASSETS_DIR = Path(__file__).resolve().parent / "nhits_solar"
UTC = ZoneInfo("UTC")
MADRID = ZoneInfo("Europe/Madrid")
ENCODER_LENGTH = HISTORY_HOURS = 72
PREDICTION_LENGTH = 48
GENERATION_FETCH_DAYS = 6
WEATHER_PAST_DAYS = 5
WEATHER_FORECAST_DAYS = 3
REQUEST_TIMEOUT_SECONDS = 30
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "viz-code-renewable-forecast/1.0"})
TARGET_COLUMNS = ["eol_scaled_hourly", "sol_scaled_hourly"]


@dataclass(frozen=True)
class ModelMetrics:
    """Métricas para una serie."""

    mae_scaled: float
    rmse_scaled: float
    smape: float
    best_val_loss: float


@dataclass(frozen=True)
class ModelSpec:
    """Configuración para ejecutar un checkpoint NHITS."""

    key: str
    label: str
    checkpoint_path: Path
    target_col: str
    actual_mw_col: str
    capacity_col: str
    metrics: ModelMetrics


@dataclass
class ForecastSeries:
    """Datos preparados para una serie."""

    spec: ModelSpec
    actual: pd.DataFrame
    forecast: pd.DataFrame
    capacity_mw: float
    last_actual_at: pd.Timestamp


MODEL_SPECS = {
    "solar": ModelSpec(
        key="solar",
        label="Solar",
        checkpoint_path=ASSETS_DIR / "nhits_solar.ckpt",
        target_col="sol_scaled_hourly",
        actual_mw_col="sol_mw",
        capacity_col="sol",
        metrics=ModelMetrics(0.0126, 0.0189, 0.9140, 0.0084),
    ),
    "eolic": ModelSpec(
        key="eolic",
        label="Eólica",
        checkpoint_path=ASSETS_DIR / "nhits_eol.ckpt",
        target_col="eol_scaled_hourly",
        actual_mw_col="eol_mw",
        capacity_col="eol",
        metrics=ModelMetrics(0.0955, 0.1058, 0.3838, 0.0232),
    ),
}


def _request(
    url: str,
    params: dict[str, object],
    as_json: bool = True,
    retries: int = 4,
    backoff_seconds: float = 1.5,
) -> object:
    """
    Ejecuta una petición HTTP con reintentos.

    Args:
        url (str): URL del endpoint.
        params (dict): Parámetros de query.
        as_json (bool): Indica si la respuesta debe parsearse como JSON.
        retries (int): Número máximo de intentos.
        backoff_seconds (float): Espera entre intentos.

    Returns:
        object: Respuesta parseada como JSON o texto plano.
    """

    retriable_codes = {408, 429, 500, 502, 503, 504}

    for attempt in range(1, retries + 1):
        try:
            response = HTTP_SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code in retriable_codes and attempt < retries:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else backoff_seconds * attempt
                time.sleep(max(delay, 0.0))
                continue
            response.raise_for_status()
            return response.json() if as_json else response.text
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(backoff_seconds * attempt)

    raise RuntimeError("Failed HTTP request.")


def fetch_installed_capacity() -> pd.DataFrame:
    """
    Descarga la potencia instalada de eólica y solar.

    Returns:
        pd.DataFrame: Tabla con fecha, y capacidad eólica y solar.
    """
    today = pd.Timestamp.now(tz=UTC).tz_convert(MADRID).normalize()
    payload = _request(
        "https://apidatos.ree.es/es/datos/generacion/potencia-instalada",
        {
            "time_trunc": "day",
            "geo_trunc": "electric_system",
            "geo_limit": "peninsular",
            "geo_ids": "8741",
            "start_date": (today - pd.Timedelta(days=60)).strftime("%Y-%m-%dT%H:%M"),
            "end_date": (today + pd.Timedelta(days=3)).strftime("%Y-%m-%dT%H:%M"),
        },
    )
    if not isinstance(payload, dict):
        raise ValueError("Error downloading installed capacity from REE.")

    # Normalizar el payload.
    rows = []
    for item in payload.get("included", []):
        attrs = item.get("attributes", {})
        for value in attrs.get("values", []):
            rows.append(
                {
                    "datetime": value["datetime"],
                    "generation_type": attrs.get("title"),
                    "value": value["value"],
                }
            )
    if not rows:
        raise ValueError("Empty result for installed capacity.")

    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = (
        frame.pivot_table(index="datetime", columns="generation_type", values="value", aggfunc="first")
        .rename(columns={"E\u00f3lica": "eol", "Solar fotovoltaica": "sol"})
        .sort_index()
        .reset_index()
    )
    if not {"eol", "sol"}.issubset(frame.columns):
        raise ValueError("Installed capacity response is missing eolic or solar columns.")
    return frame.loc[:, ["datetime", "eol", "sol"]]


def _parse_generation_timestamp(val: str) -> pd.Timestamp:
    """
    Convierte timestamps del formato REE a UTC, incluyendo el cambio horario.

    Args:
        value (str): Timestamp.

    Returns:
        pd.Timestamp: Timestamp en UTC.
    """
    if not isinstance(val, str):
        return pd.NaT

    raw, ambiguous = val, "raise"
    if "2A" in raw:
        raw, ambiguous = raw.replace("2A", "02"), True

    elif "2B" in raw:
        raw, ambiguous = raw.replace("2B", "02"), False

    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT

    return parsed.tz_localize(MADRID, ambiguous=ambiguous, nonexistent="shift_forward").tz_convert(UTC)


def _fetch_generation_day(day: pd.Timestamp) -> list[dict]:
    """
    Descarga la generación horaria de REE para un día.

    Args:
        day (pd.Timestamp): Día a consultar.

    Returns:
        list[dict]: Filas horarias de generación para el día solicitado.
    """
    text = str(
        _request(
            "https://demanda.ree.es/WSvisionaMovilesPeninsulaRest/resources/demandaGeneracionPeninsula",
            {
                "callback": "angular.callbacks._7",
                "curva": "DEMANDAAU",
                "fecha": f"{day.year}-{day.month}-{day.day}",
            },
            as_json=False,
        )
    ).strip()

    # Eliminar el wrapper JSONP
    if (start := text.find("(")) != -1 and (end := text.rfind(")")) > start:
        text = text[start + 1 : end]

    rows = json.loads(text).get("valoresHorariosGeneracion") or []
    if not isinstance(rows, list):
        raise ValueError("REE generation API returned an unexpected payload.")
    return rows


def fetch_generation_range() -> pd.DataFrame:
    """
    Descarga de datos de generación de REE.

    Returns:
        pd.DataFrame: Datos con timestamps y columnas eol/sol.
    """
    today = pd.Timestamp.now(tz=UTC).tz_convert(MADRID).normalize()
    rows = []
    for day in pd.date_range(
        start=(today - pd.Timedelta(days=GENERATION_FETCH_DAYS)).date(),
        end=today.date(),
        freq="D",
    ):
        rows.extend(_fetch_generation_day(day))

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Empty REE response.")

    return frame


def build_hourly_actuals(generation_frame: pd.DataFrame, capacity_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte generación real a valores horarios y normalizados por capacidad.

    Args:
        generation_frame (pd.DataFrame): Datos de generación.
        capacity_frame (pd.DataFrame): Potencia instalada por fecha.

    Returns:
        pd.DataFrame: Histórico horario en MW normalizado.
    """
    frame = generation_frame.copy()
    frame["ts"] = frame["ts"].map(_parse_generation_timestamp)
    frame[["eol", "sol"]] = frame[["eol", "sol"]].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["ts", "eol", "sol"]).sort_values("ts")

    # Se usa la última capacidad instalada conocida para cada hora real.
    joined = pd.merge_asof(
        frame,
        capacity_frame.rename(columns={"eol": "capacity_eol", "sol": "capacity_sol"}).sort_values("datetime"),
        left_on="ts",
        right_on="datetime",
        direction="backward",
    )
    joined["date"] = joined["ts"].dt.floor("h")
    joined = joined.loc[joined["date"] < pd.Timestamp.now(tz=UTC).floor("h")]
    joined["eol_scaled_hourly"] = joined["eol"] / joined["capacity_eol"]
    joined["sol_scaled_hourly"] = joined["sol"] / joined["capacity_sol"]
    return (
        joined.groupby("date", as_index=False)
        .agg(
            eol_mw=("eol", "mean"),
            sol_mw=("sol", "mean"),
            eol_scaled_hourly=("eol_scaled_hourly", "mean"),
            sol_scaled_hourly=("sol_scaled_hourly", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def _weather_frame(name: str, payload: dict) -> pd.DataFrame:
    """
    Convierte un diccionario de datos meteorológicos en DataFrame, 
    teniendo en cuenta la comunidad autónoma.

    Args:
        name (str): Nombre de la comunidad.
        payload (dict): Diccionario de Openmeteo.

    Returns:
        pd.DataFrame: DataFrame con variables renombradas.
    """
    hourly = payload["hourly"]
    data = {"date": pd.to_datetime(hourly["time"]).tz_localize(MADRID).tz_convert(UTC)}
    for variable in WEATHER_VARIABLES:
        if variable == "snow_depth" and name == "canary_islands":
            continue
        data[f"{name}_{WEATHER_SUFFIXES[variable]}"] = pd.to_numeric(hourly[variable], errors="coerce")
    return pd.DataFrame(data)


def fetch_weather_features() -> pd.DataFrame:
    """
    Descarga variables meteorológicas de Openmeteo.

    Returns:
        pd.DataFrame: DataFrame con variables meteorológicas.
    """
    payload = _request(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": ",".join(str(COMMUNITIES[name][0]) for name in COMMUNITY_NAMES),
            "longitude": ",".join(str(COMMUNITIES[name][1]) for name in COMMUNITY_NAMES),
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": ",".join(["Europe/Madrid"] * len(COMMUNITY_NAMES)),
            "past_days": WEATHER_PAST_DAYS,
            "forecast_days": WEATHER_FORECAST_DAYS,
        },
    )
    payloads = payload if isinstance(payload, list) else [payload]
    if len(payloads) != len(COMMUNITY_NAMES):
        raise ValueError(f"Expected {len(COMMUNITY_NAMES)} communities, got {len(payloads)}.")

    # Unir las series de las diferentes comunidades por timestamp.
    frame = _weather_frame(COMMUNITY_NAMES[0], payloads[0])
    for name, community_payload in zip(COMMUNITY_NAMES[1:], payloads[1:], strict=True):
        frame = frame.merge(_weather_frame(name, community_payload), on="date", how="outer")
    frame = frame.sort_values("date").reset_index(drop=True)

    return frame.loc[:, ["date", *WEATHER_COLUMNS]]


def build_inference_frame(
    hourly_actuals: pd.DataFrame,
    weather_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Prepara el DataFrame necesario para TimeSeriesDataSet.

    Args:
        hourly_actuals (pd.DataFrame): DataFrame con datos históricos normalizados.
        weather_frame (pd.DataFrame): DataFrame con variables meteorológicas.

    Returns:
        tuple[pd.DataFrame, pd.Timestamp]: DataFrames para inferencia.
    """

    last_actual = hourly_actuals["date"].max()
    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                last_actual - pd.Timedelta(hours=ENCODER_LENGTH - 1),
                last_actual + pd.Timedelta(hours=PREDICTION_LENGTH),
                freq="h",
                tz=UTC,
            )
        }
    )
    frame = frame.merge(weather_frame.drop(columns=TARGET_COLUMNS, errors="ignore"), on="date", how="left")
    frame = (
        frame.merge(hourly_actuals[["date", *TARGET_COLUMNS]], on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # El encoder necesita targets reales completos. El decoder solo necesita valores de relleno.
    history_mask = frame["date"] <= last_actual
    history = frame.loc[history_mask, TARGET_COLUMNS]
    missing_history = history.columns[history.isna().any()].tolist()
    if missing_history:
        raise ValueError(f"Missing values historical DataFrame: {missing_history}")
    
    for target in TARGET_COLUMNS:
        frame.loc[~history_mask, target] = frame.loc[history_mask, target].iloc[-1]

    date = frame["date"]
    frame = frame.assign(
        sin_day=np.sin(2 * np.pi * date.dt.dayofyear / 365.0),
        cos_day=np.cos(2 * np.pi * date.dt.dayofyear / 365.0),
        sin_hour=np.sin(2 * np.pi * date.dt.hour / 24.0),
        cos_hour=np.cos(2 * np.pi * date.dt.hour / 24.0),
        year=date.dt.year.astype(np.int64),
        month=date.dt.month.astype(np.int64),
        hour=date.dt.hour.astype(np.int64),
        day_of_year=date.dt.dayofyear.astype(np.int64),
        time_idx=np.arange(len(frame), dtype=np.int64),
        group_id="series_0",
    )
    missing_weather = [
        column for column in WEATHER_COLUMNS
        if column not in frame or frame[column].isna().any()
    ]
    if missing_weather:
        raise ValueError(f"Missing values in weather dataframe: {missing_weather}")
    return frame, last_actual


@cache
def load_model(model_key: str) -> NHiTS:
    """
    Carga un checkpoint NHITS y lo deja en modo evaluación.

    Args:
        model_key (str): Clave del modelo.

    Returns:
        NHiTS: Modelo cargado en CPU.
    """
    model = NHiTS.load_from_checkpoint(MODEL_SPECS[model_key].checkpoint_path, map_location="cpu")
    model.eval()
    return model


def predict_scaled_series(frame: pd.DataFrame, model_key: str) -> np.ndarray:
    """
    Obtiene las predicciones de NHITS normalizadas.

    Args:
        frame (pd.DataFrame): DataFrame con encoder, decoder y las variables necesarias.
        model_key (str): Clave del modelo.

    Returns:
        np.ndarray: Predicciones normalizadas.
    """
    model = load_model(model_key)
    parameters = model.hparams.dataset_parameters

    # Se reconstruye el dataset con los parámetros guardados dentro del checkpoint.
    dataloader = TimeSeriesDataSet.from_parameters(
        parameters,
        frame.copy(),
        predict=True,
        stop_randomization=True,
    ).to_dataloader(train=False, batch_size=1, num_workers=0)
    with torch.no_grad():
        prediction = model.predict(
            dataloader,
            mode="prediction",
            trainer_kwargs={
                "accelerator": "cpu",
                "devices": 1,
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            },
        )
    scaled = (
        prediction.squeeze().detach().cpu().numpy()
        if torch.is_tensor(prediction)
        else np.asarray(prediction).squeeze()
    )
    return np.clip(scaled, a_min=0.0, a_max=None)

def _forecast_series(
    model_key: str,
    future_dates: pd.Series,
    hourly_actuals: pd.DataFrame,
    capacity_frame: pd.DataFrame,
    inference_frame: pd.DataFrame,
    last_actual: pd.Timestamp,
) -> ForecastSeries:
    """
    Construye histórico y predicción en MW para una serie.

    Args:
        model_key (str): Clave del modelo.
        future_dates (pd.Series): Fechas futuras para la predicción.
        hourly_actuals (pd.DataFrame): Histórico horario.
        capacity_frame (pd.DataFrame): Potencia instalada por fecha.
        inference_frame (pd.DataFrame): Tabla completa para inferencia.
        last_actual (pd.Timestamp): Última hora real.

    Returns:
        ForecastSeries: Serie final para crear el plot.
    """
    spec = MODEL_SPECS[model_key]
    scaled = predict_scaled_series(inference_frame, model_key)    
    capacity = pd.merge_asof(
        pd.DataFrame({"date": future_dates}).sort_values("date"),
        capacity_frame[["datetime", spec.capacity_col]].sort_values("datetime"),
        left_on="date",
        right_on="datetime",
        direction="backward",
    )[spec.capacity_col].ffill().bfill().reset_index(drop=True)
    
    if len(scaled) != len(future_dates):
        raise ValueError(
            f"{spec.label} prediction time features mismatch: "
            f"expected {len(future_dates)} points, got {len(scaled)}."
        )
    
    actual_start = last_actual - pd.Timedelta(hours=HISTORY_HOURS - 1)
    return ForecastSeries(
        spec=spec,
        actual=(
            hourly_actuals.loc[hourly_actuals["date"] >= actual_start, ["date", spec.actual_mw_col]]
            .rename(columns={spec.actual_mw_col: "value_mw"})
            .reset_index(drop=True)
        ),
        forecast=pd.DataFrame({"date": future_dates, "scaled_prediction": scaled, "value_mw": scaled * capacity}),
        capacity_mw=float(capacity.iloc[-1]),
        last_actual_at=last_actual,
    )


def build_dashboard_snapshot() -> dict[str, ForecastSeries]:
    """
    Orquestación de descarga, preparación e inferencia.

    Returns:
        dict[str, ForecastSeries]: Series de pandas preparadas para Streamlit.
    """
    capacity_frame = fetch_installed_capacity()
    hourly_actuals = build_hourly_actuals(fetch_generation_range(), capacity_frame)
    weather_frame = fetch_weather_features()
    inference_frame, last_actual = build_inference_frame(hourly_actuals, weather_frame)
    future_dates = inference_frame.loc[inference_frame["date"] > last_actual, "date"].reset_index(drop=True)
    return {
        key: _forecast_series(key, future_dates, hourly_actuals, capacity_frame, inference_frame, last_actual)
        for key in MODEL_SPECS
    }
