# Mapeo entre los nombres originales de Openmeteo y los nombres usados al entrenar.
WEATHER_SUFFIXES = {
    "temperature_2m": "tmp_2m",
    "relative_humidity_2m": "relh_2m",
    "rain": "rain",
    "snowfall": "snowfall",
    "snow_depth": "snow_depth",
    "pressure_msl": "pressure_msl",
    "cloud_cover": "cloud_cover",
    "cloud_cover_low": "cloud_cover_l",
    "cloud_cover_mid": "cloud_cover_m",
    "cloud_cover_high": "cloud_cover_h",
    "wind_speed_10m": "wind_s_10m",
    "wind_speed_100m": "wind_s_100m",
    "wind_direction_10m": "wind_dir_10m",
    "wind_direction_100m": "wind_dir_100m",
    "wind_gusts_10m": "wind_gusts_10m",
    "is_day": "is_day",
    "terrestrial_radiation": "terrestrial_rad",
    "global_tilted_irradiance": "global_tilt_irrad",
    "direct_normal_irradiance": "direct_norm_irrad",
    "diffuse_radiation": "diffuse_rad",
    "direct_radiation": "direct_rad",
    "shortwave_radiation": "shortwave_rad",
}

WEATHER_VARIABLES = list(WEATHER_SUFFIXES)

# Coordenadas de comunidades autónomas.
COMMUNITIES = {
    "andalusia": (37.4633, -4.5756),
    "aragon": (41.5195, -0.6599),
    "asturias": (43.2923, -5.9933),
    "balearic_islands": (39.5667, 2.6500),
    "basque_country": (43.0435, -2.6164),
    "canary_islands": (28.3415, -15.6670),
    "cantabria": (43.1977, -4.0293),
    "castilla_la_mancha": (39.5809, -3.0045),
    "castilla_y_leon": (41.7543, -4.7818),
    "catalonia": (41.7984, 1.5288),
    "extremadura": (39.1915, -6.1507),
    "galicia": (42.7568, -7.9105),
    "la_rioja": (42.2748, -2.5170),
    "madrid": (40.4169, -3.7033),
    "murcia": (38.0020, -1.4851),
    "navarre": (42.6671, -1.6460),
    "valencian_community": (39.4015, -0.5546),
}
COMMUNITY_NAMES = list(COMMUNITIES)

# Columnas meteorológicas uniendo todas las comunidades.
WEATHER_COLUMNS = []
for community in COMMUNITY_NAMES:
    for variable, suffix in WEATHER_SUFFIXES.items():
        if community == "canary_islands" and variable == "snow_depth":
            continue
        WEATHER_COLUMNS.append(f"{community}_{suffix}")
