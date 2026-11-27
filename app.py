import base64
import io
import json
import os
import ssl
import time
import unicodedata
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, zoom
from scipy.spatial import cKDTree
from shapely.geometry import shape

st.set_page_config(
    page_title="Geoportal Térmico 3D - Colombia 2026", layout="wide"
)

BG = "#0a1226"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; color: #f0f2f6; }}
    [data-testid="stMetric"] {{
        background-color: #111827;
        border: 1px solid #1f2937;
        padding: 12px 16px;
        border-radius: 8px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌡️ Geoportal de Análisis Térmico y Relieve 3D · Colombia (2026)")
st.markdown(
    "Fuente de datos: **IDEAM (Datos Abiertos Colombia)** · Relieve: modelo"
    " digital de elevación real · Interpolación IDW con gradiente altitudinal."
)

# ----------------------------------------------------------------------------
# CONSTANTES
# ----------------------------------------------------------------------------
RUTA_CSV = "temperaturas.csv"
RESOLUCION = 300

LON_MIN, LON_MAX = -82.5, -61.5
LAT_MIN, LAT_MAX = -8.5, 13.8

KM_GRADO = 111.0
LON0, LAT0 = -72.0, 2.5

MODO_TEMP = "Temperatura Media (°C)"
MODO_ANOM = "Anomalía vs. promedio del año (°C)"

VISTA_MAPA = "🗺️ Mapa móvil (pellizco y arrastre)"
VISTA_3D = "🏔️ Vista 3D (rotar)"

# True = mapa con MapLibre (plotly >= 5.24). Si el mapa móvil sale en blanco,
# prueba cambiando a False (usa el motor Mapbox clásico).
USAR_MAPLIBRE = hasattr(go, "Scattermap")

MESES = [
    "Ene", "Feb", "Mar", "Abr", "May", "Jun",
    "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
]
MESES_LARGO = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]

NOMBRES_FECHA = {
    "fecha", "fechaobservacion", "fechaobs", "fechahora",
    "date", "datetime", "timestamp",
}

# Cada variable define su archivo, cómo se agrega, su paleta y sus textos.
# Para sumar una variable nueva basta con añadir otra entrada aquí.
_LEY_DIV_TEMP = [
    ("#1d4ed8", "< -2.0 °C"), ("#60a5fa", "-2.0 a -0.5 °C"),
    ("#f8fafc", "-0.5 a +0.5 °C"), ("#fb923c", "+0.5 a +2.0 °C"),
    ("#991b1b", "> +2.0 °C"),
]
_PARADAS_LLUVIA = [
    (0, "#fef3c7"), (10, "#fde68a"), (25, "#e0f2fe"), (50, "#bae6fd"),
    (100, "#7dd3fc"), (200, "#38bdf8"), (300, "#2563eb"), (450, "#4338ca"),
    (600, "#581c87"),
]
VARIABLES = {
    "Temperatura": dict(
        archivo="temperaturas.csv",
        alias={"temperatura", "temp", "valor", "valorobservado"},
        unidad="°C", unidad_anom="°C", dec=1, agg_dia="mean", lapse=True,
        vmin=4.0, vmax=32.0,
        paradas=[
            (4, "#eef6ff"), (10, "#60a5fa"), (14, "#22d3ee"), (18, "#86efac"),
            (21, "#fde047"), (24, "#fb923c"), (27, "#ef4444"), (30, "#991b1b"),
            (32, "#450a0a"),
        ],
        escala_div=[
            [0.0, "#1d4ed8"], [0.25, "#60a5fa"], [0.5, "#f8fafc"],
            [0.75, "#fb923c"], [1.0, "#991b1b"],
        ],
        anom_rango=(-3.0, 3.0), anom_centro=0.0, lim_cambio=3.0,
        etq_modo="Temperatura Media", nombre_media="Temperatura media",
        kpi=["Temperatura Media", "Rango Térmico (Mín / Máx)",
             "Territorio en Calor Intenso (≥27°C)"],
        kpi_corto=["Temperatura media", "Mín / Máx", "Calor intenso (≥27 °C)"],
        kpi_cambio=["Cambio medio (B − A)", "Rango del cambio",
                    "Territorio que subió > 0.5 °C"],
        alerta=lambda t, a: t >= 27.0,
        cambio_alerta=lambda d: d > 0.5,
        nombre_alerta="% del territorio ≥ 27 °C",
        etq_extremo="Mes más cálido", extremo="max",
        etq_alerta="Mayor territorio ≥ 27 °C",
        fuente="IDEAM (Datos Abiertos Colombia · sbwg-7ju4)",
        metodo="IDW + Gradiente Altitudinal",
        plancha_paradas=[
            (4, "#1d4ed8"), (10, "#38bdf8"), (14, "#67e8f9"), (18, "#4ade80"),
            (21, "#fde047"), (24, "#f59e0b"), (27, "#f97316"), (30, "#dc2626"),
            (32, "#7f1d1d"),
        ],
        plancha_titulo="DISTRIBUCIÓN DE LA\nTEMPERATURA MEDIA\nDEL AIRE (°C)",
        plancha_titulo_anom="ANOMALÍA DE\nTEMPERATURA\nDEL AIRE (°C)",
        plancha_titulo_cambio="CAMBIO DE\nTEMPERATURA\nDEL AIRE (°C)",
        leyenda=[
            ("#1d4ed8", "< 10 °C"), ("#38bdf8", "10 – 14 °C"),
            ("#4ade80", "14 – 18 °C"), ("#fde047", "18 – 21 °C"),
            ("#f59e0b", "21 – 24 °C"), ("#f97316", "24 – 27 °C"),
            ("#dc2626", "27 – 30 °C"), ("#7f1d1d", "> 30 °C"),
        ],
        leyenda_anom=_LEY_DIV_TEMP, leyenda_cambio=_LEY_DIV_TEMP,
    ),
    "Precipitación": dict(
        archivo="precipitacion.csv",
        alias={"precipitacion", "precip", "lluvia", "valor", "valorobservado"},
        unidad="mm/mes", unidad_anom="% del promedio", dec=0, agg_dia="sum",
        lapse=False,
        vmin=0.0, vmax=600.0,
        paradas=_PARADAS_LLUVIA,
        escala_div=[
            [0.0, "#7f1d1d"], [0.25, "#f59e0b"], [0.5, "#f8fafc"],
            [0.75, "#60a5fa"], [1.0, "#1d4ed8"],
        ],
        anom_rango=(0.0, 200.0), anom_centro=100.0, lim_cambio=150.0,
        etq_modo="Precipitación", nombre_media="Lluvia media",
        kpi=["Lluvia media (mm/mes)", "Rango (Mín / Máx)",
             "Territorio con déficit (< 50 % del promedio)"],
        kpi_corto=["Lluvia media", "Mín / Máx", "Déficit (< 50 % del promedio)"],
        kpi_cambio=["Cambio medio (B − A)", "Rango del cambio",
                    "Territorio con menos lluvia en B"],
        alerta=lambda t, a: a < 50.0,
        cambio_alerta=lambda d: d < 0.0,
        nombre_alerta="% del territorio con déficit (< 50 %)",
        etq_extremo="Mes más seco", extremo="min",
        etq_alerta="Mayor territorio con déficit (< 50 % del promedio)",
        fuente="IDEAM (Datos Abiertos Colombia)",
        metodo="IDW (sin corrección altitudinal)",
        plancha_paradas=_PARADAS_LLUVIA,
        plancha_titulo="DISTRIBUCIÓN DE LA\nPRECIPITACIÓN\n(mm/mes)",
        plancha_titulo_anom="ANOMALÍA DE LA\nPRECIPITACIÓN\n(% DEL PROMEDIO)",
        plancha_titulo_cambio="CAMBIO DE LA\nPRECIPITACIÓN\n(mm/mes)",
        leyenda=[
            ("#fef3c7", "< 10 mm"), ("#e0f2fe", "10 – 50 mm"),
            ("#7dd3fc", "50 – 100 mm"), ("#38bdf8", "100 – 200 mm"),
            ("#2563eb", "200 – 300 mm"), ("#4338ca", "300 – 450 mm"),
            ("#581c87", "> 450 mm"),
        ],
        leyenda_anom=[
            ("#7f1d1d", "< 40 %"), ("#f59e0b", "40 – 80 %"),
            ("#f8fafc", "80 – 120 %"), ("#60a5fa", "120 – 160 %"),
            ("#1d4ed8", "> 160 %"),
        ],
        leyenda_cambio=[
            ("#7f1d1d", "< -90 mm"), ("#f59e0b", "-90 a -30 mm"),
            ("#f8fafc", "-30 a +30 mm"), ("#60a5fa", "+30 a +90 mm"),
            ("#1d4ed8", "> +90 mm"),
        ],
    ),
}

LUZ = dict(ambient=0.38, diffuse=0.95, specular=0.06, roughness=0.78, fresnel=0.06)
POS_LUZ = dict(x=-4000, y=4000, z=3000)

CAMARAS = {
    "Inclinada (3D)": dict(
        eye=dict(x=0.0, y=-1.05, z=0.95), center=dict(x=0.0, y=0.0, z=-0.05)
    ),
    "Desde arriba (2D)": dict(
        eye=dict(x=0.0, y=-0.02, z=1.65), center=dict(x=0.0, y=0.0, z=0.0)
    ),
    "Cordilleras (Rasante)": dict(
        eye=dict(x=0.15, y=-1.45, z=0.45), center=dict(x=0.0, y=0.05, z=-0.05)
    ),
}

CIUDADES = [
    ("Bogotá", 4.71, -74.07),
    ("Medellín", 6.25, -75.56),
    ("Cali", 3.45, -76.53),
    ("Barranquilla", 10.96, -74.80),
    ("Cartagena", 10.39, -75.48),
    ("Santa Marta", 11.24, -74.20),
    ("Bucaramanga", 7.12, -73.12),
    ("Cúcuta", 7.89, -72.50),
    ("Tunja", 5.53, -73.36),
    ("Pereira", 4.81, -75.69),
    ("Ibagué", 4.44, -75.24),
    ("Villavicencio", 4.14, -73.63),
    ("Yopal", 5.34, -72.39),
    ("Arauca", 7.08, -70.76),
    ("San José del Guaviare", 2.57, -72.64),
    ("Montería", 8.75, -75.88),
    ("Sincelejo", 9.30, -75.39),
    ("Valledupar", 10.46, -73.25),
    ("Riohacha", 11.54, -72.90),
    ("Pasto", 1.21, -77.28),
    ("Neiva", 2.93, -75.28),
    ("Florencia", 1.61, -75.61),
    ("Quibdó", 5.69, -76.66),
]

REGIONES_NATURALES = {
    "TODAS (COLOMBIA)": None,
    "Región Andina": [
        "ANTIOQUIA", "BOYACA", "CALDAS", "CUNDINAMARCA", "HUILA",
        "NORTE DE SANTANDER", "QUINDIO", "RISARALDA", "SANTANDER",
        "TOLIMA", "BOGOTA",
    ],
    "Región Caribe": [
        "ATLANTICO", "BOLIVAR", "CESAR", "CORDOBA", "LA GUAJIRA",
        "MAGDALENA", "SUCRE",
    ],
    "Región Pacífica": ["CAUCA", "CHOCO", "NARINO", "VALLE DEL CAUCA"],
    "Región Orinoquía": ["ARAUCA", "CASANARE", "META", "VICHADA"],
    "Región Amazonía": [
        "AMAZONAS", "CAQUETA", "GUAINIA", "GUAVIARE", "PUTUMAYO", "VAUPES",
    ],
}


# ----------------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------------
def a_km(lon, lat):
  return (
      (np.asarray(lon, dtype=float) - LON0) * KM_GRADO,
      (np.asarray(lat, dtype=float) - LAT0) * KM_GRADO,
  )


def _normalizar(texto):
  t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore")
  return t.decode("utf-8").lower().strip().replace(" ", "").replace("_", "")


def _idx_grid(valor, vmin, vmax, n):
  return np.clip(
      np.round((valor - vmin) / (vmax - vmin) * (n - 1)).astype(int), 0, n - 1
  )


def es_parcial(p, f_min, f_max):
  """True si el mes tiene datos incompletos (empieza tarde o termina antes)."""
  margen = pd.Timedelta(days=2)
  return bool(
      (p.start_time <= f_min <= p.end_time and f_min > p.start_time + margen)
      or (p.start_time <= f_max <= p.end_time and f_max < p.end_time - margen)
  )


def puntos_dentro(geojson, lon, lat):
  geom = shape(geojson)
  if geom.is_empty:
    return np.zeros(lon.shape, dtype=bool)
  try:
    from shapely import contains_xy

    return np.asarray(contains_xy(geom, lon.ravel(), lat.ravel())).reshape(
        lon.shape
    )
  except ImportError:
    from shapely.geometry import Point
    from shapely.prepared import prep

    pg = prep(geom)
    return np.array(
        [pg.contains(Point(x, y)) for x, y in zip(lon.ravel(), lat.ravel())]
    ).reshape(lon.shape)


def _hex(c):
  return np.array([int(c[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)


def colorear(valores):
  """Valor -> RGB (0-255) con la misma paleta que el 3D (sin matplotlib)."""
  if modo_mapa == MODO_TEMP:
    xs = np.array([t for t, _ in PARADAS_TEMP], dtype=float)
    cols = np.stack([_hex(c) for _, c in PARADAS_TEMP])
    if bandas_discretas:
      idx = np.clip(np.digitize(valores, xs) - 1, 0, len(xs) - 1)
      return cols[idx]
  else:
    lo, hi = rango_div()
    xs = np.array([p for p, _ in ESCALA_ANOM]) * (hi - lo) + lo
    cols = np.stack([_hex(c) for _, c in ESCALA_ANOM])
  return np.stack(
      [np.interp(valores, xs, cols[:, k]) for k in range(3)], axis=-1
  )


def hillshade(elev, dx, dy, exag_v, az=315.0, alt=38.0):
  """Sombreado de relieve: luz desde el noroeste (convención cartográfica)."""
  gy, gx = np.gradient(elev * exag_v, dy, dx)
  nz = 1.0 / np.sqrt(gx**2 + gy**2 + 1.0)
  nx, ny = -gx * nz, -gy * nz
  a, h = np.radians(az), np.radians(alt)
  return np.clip(
      nx * np.sin(a) * np.cos(h) + ny * np.cos(a) * np.cos(h) + nz * np.sin(h),
      0,
      1,
  )


# ----------------------------------------------------------------------------
# CARTOGRAFÍA
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_cartografia():
  with open("world_50m.json", "r", encoding="utf-8") as f:
    topo = json.load(f)

  scale = topo["transform"]["scale"]
  translate = topo["transform"]["translate"]
  arcs = topo["arcs"]

  def decodificar_arco(idx):
    invertido = idx < 0
    arco = arcs[~idx if invertido else idx]
    coords = []
    x, y = 0, 0
    for dx, dy in arco:
      x += dx
      y += dy
      coords.append([x * scale[0] + translate[0], y * scale[1] + translate[1]])
    return coords[::-1] if invertido else coords

  def construir_poligonos(geom_obj):
    polys = []
    arc_lists = (
        [geom_obj["arcs"]]
        if geom_obj["type"] == "Polygon"
        else geom_obj["arcs"]
    )
    for poly_arcs in arc_lists:
      anillos = []
      for ring in poly_arcs:
        anillo_coords = []
        for arc_idx in ring:
          seg = decodificar_arco(arc_idx)
          anillo_coords.extend(seg[:-1] if anillo_coords else seg)
        anillos.append(anillo_coords)
      polys.append(anillos)
    return polys

  col_polys = []
  vecinos_polys = []

  for g in topo["objects"]["countries"]["geometries"]:
    cid = str(g.get("id", "")).zfill(3)
    nombre = g.get("properties", {}).get("name", "")
    if cid == "170" or nombre == "Colombia":
      col_polys = construir_poligonos(g)
    else:
      polys = construir_poligonos(g)
      for poly in polys:
        if any(-84 <= p[0] <= -58 and -12 <= p[1] <= 15 for p in poly[0]):
          vecinos_polys.append(poly)

  geojson_col = {"type": "MultiPolygon", "coordinates": col_polys}
  geojson_vecinos = {"type": "MultiPolygon", "coordinates": vecinos_polys}

  if (
      not os.path.exists("deptos_colombia.json")
      or os.path.getsize("deptos_colombia.json") <= 10000
  ):
    try:
      ctx = ssl._create_unverified_context()
      url_d = "https://gist.githubusercontent.com/john-guerra/43c7656821069d00dcbc/raw/be6a6e239cd5b5b803c6e7c2ec405b793a9064dd/Colombia.geo.json"
      with urllib.request.urlopen(url_d, context=ctx, timeout=20) as r:
        contenido = r.read().decode("utf-8")
      with open("deptos_colombia.json", "w", encoding="utf-8") as f:
        f.write(contenido)
    except Exception:
      pass

  deptos_geojson = None
  if (
      os.path.exists("deptos_colombia.json")
      and os.path.getsize("deptos_colombia.json") > 10000
  ):
    with open("deptos_colombia.json", "r", encoding="utf-8") as f:
      deptos_geojson = json.load(f)

  return geojson_col, geojson_vecinos, deptos_geojson


# ----------------------------------------------------------------------------
# RELIEVE (DEM)
# ----------------------------------------------------------------------------
def descargar_dem(lons, lats, zoom_t=6):
  n = 2**zoom_t
  fx = (lons + 180.0) / 360.0 * n
  fy = (1.0 - np.arcsinh(np.tan(np.radians(lats))) / np.pi) / 2.0 * n
  x0, x1 = int(np.floor(fx.min())), int(np.floor(fx.max()))
  y0, y1 = int(np.floor(fy.min())), int(np.floor(fy.max()))

  mosaico = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256), np.float32)
  for ty in range(y0, y1 + 1):
    for tx in range(x0, x1 + 1):
      url = (
          "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/"
          f"{zoom_t}/{tx}/{ty}.png"
      )
      req = urllib.request.Request(
          url, headers={"User-Agent": "geoportal-termico/1.0"}
      )
      with urllib.request.urlopen(req, timeout=25) as r:
        rgb = np.asarray(
            Image.open(io.BytesIO(r.read())).convert("RGB"), dtype=np.float32
        )
      elev = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
      mosaico[
          (ty - y0) * 256 : (ty - y0 + 1) * 256,
          (tx - x0) * 256 : (tx - x0 + 1) * 256,
      ] = elev

  mosaico = gaussian_filter(mosaico, sigma=1.6)
  gx, gy = np.meshgrid((fx - x0) * 256.0, (fy - y0) * 256.0)
  dem = map_coordinates(mosaico, [gy, gx], order=1, mode="nearest")
  return np.clip(dem, 0, 6000)


def trazar_cadena(grid_lon, grid_lat, x1, y1, x2, y2, ancho, altura):
  dx, dy = x2 - x1, y2 - y1
  seg_len2 = dx * dx + dy * dy
  t = np.clip(((grid_lon - x1) * dx + (grid_lat - y1) * dy) / seg_len2, 0, 1)
  proj_x = x1 + t * dx
  proj_y = y1 + t * dy
  dist2 = (grid_lon - proj_x) ** 2 + (grid_lat - proj_y) ** 2
  return altura * np.exp(-dist2 / (2 * ancho**2))


def modelo_relieve_sintetico(grid_lon, grid_lat):
  elev = np.zeros_like(grid_lat)
  cordilleras = [
      (-77.7, 0.9, -76.2, 7.3, 0.42, 2600),
      (-76.7, 1.3, -75.3, 7.4, 0.46, 3600),
      (-75.7, 1.9, -72.4, 8.1, 0.56, 3500),
      (-78.6, -4.5, -77.3, 1.0, 0.65, 3700),
      (-73.75, 10.82, -73.55, 10.88, 0.34, 4100),
      (-73.05, 8.5, -72.65, 10.6, 0.30, 2300),
      (-72.1, 7.6, -69.3, 10.1, 0.48, 3300),
      (-66.5, 3.5, -61.5, 6.2, 1.10, 1100),
  ]
  for x1, y1, x2, y2, ancho, altura in cordilleras:
    elev += trazar_cadena(grid_lon, grid_lat, x1, y1, x2, y2, ancho, altura)
  for x1, y1, x2, y2, ancho, prof in [
      (-75.6, 2.2, -74.7, 8.5, 0.20, 1700),
      (-76.5, 2.5, -75.6, 7.5, 0.14, 1400),
  ]:
    elev -= trazar_cadena(grid_lon, grid_lat, x1, y1, x2, y2, ancho, prof)
  elev = np.clip(elev, 0, 5400)
  ruido = gaussian_filter(np.random.RandomState(42).normal(0, 1, elev.shape), 3)
  ruido /= ruido.std()
  elev = elev + np.clip(elev / 1800.0, 0, 1.5) * 180 * ruido
  return np.clip(gaussian_filter(elev, 1.2), 0, 5500)


# ----------------------------------------------------------------------------
# DATOS
# ----------------------------------------------------------------------------
@st.cache_data
def cargar_datos(ruta_csv, mtime):
  df = pd.read_csv(ruta_csv)
  df.columns = [str(c).strip() for c in df.columns]

  col_fecha = next(
      (c for c in df.columns if _normalizar(c) in NOMBRES_FECHA), None
  )
  if col_fecha is not None:
    df["fecha"] = pd.to_datetime(
        df[col_fecha], errors="coerce", utc=True
    ).dt.tz_localize(None)
  else:
    df["fecha"] = pd.NaT

  col_valor = next(
      (c for c in df.columns if _normalizar(c) in VAR["alias"]), None
  )
  if col_valor is None:
    raise ValueError(
        "No encuentro la columna de valores en el CSV. Debe llamarse una de:"
        f" {', '.join(sorted(VAR['alias']))}."
    )
  df["valor"] = pd.to_numeric(df[col_valor], errors="coerce")
  for col in ["latitud", "longitud"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
  df = df.dropna(subset=["latitud", "longitud", "valor"])
  if VAR["agg_dia"] == "sum":
    df = df[df["valor"] >= 0]  # descarta códigos de dato faltante (-999, etc.)
  df = df[
      (df["latitud"].between(-4.3, 12.6))
      & (df["longitud"].between(-79.2, -66.8))
  ]
  df["departamento"] = (
      df["departamento"]
      .astype(str)
      .str.upper()
      .str.strip()
      .str.normalize("NFKD")
      .str.encode("ascii", errors="ignore")
      .str.decode("utf-8")
  )
  if df["fecha"].notna().any():
    # Un valor por estación y día (lluvia: suma del día; temperatura: promedio)
    df = df.dropna(subset=["fecha"]).copy()
    df["fecha"] = df["fecha"].dt.normalize()
    dep = df.groupby(["latitud", "longitud"], as_index=False)[
        "departamento"
    ].first()
    df = (
        df.groupby(["latitud", "longitud", "fecha"], as_index=False)["valor"]
        .agg(VAR["agg_dia"])
        .merge(dep, on=["latitud", "longitud"])
    )
  return df


def estaciones_periodo(df, ini_s, fin_s):
  """Valor por estación en el periodo.

  Temperatura: promedio. Lluvia: tasa diaria media x 30.4 (mm/mes
  equivalente), exigiendo datos en al menos 70 % de los días observados.
  Columnas: latitud, longitud, valor y tasa (valor diario medio).
  """
  d = df
  hay_fecha = df["fecha"].notna().any()
  if ini_s is not None:
    d = d[
        (d["fecha"] >= pd.Timestamp(ini_s)) & (d["fecha"] <= pd.Timestamp(fin_s))
    ]
  if VAR["agg_dia"] == "mean" or not hay_fecha:
    g = d.groupby(["latitud", "longitud"], as_index=False)["valor"].mean()
    g["tasa"] = g["valor"]
    return g
  f_min, f_max = df["fecha"].min(), df["fecha"].max()
  ini = max(pd.Timestamp(ini_s), f_min) if ini_s is not None else f_min
  fin = min(pd.Timestamp(fin_s), f_max) if fin_s is not None else f_max
  dias = max((fin.normalize() - ini.normalize()).days + 1, 1)
  g = d.groupby(["latitud", "longitud"])["valor"].agg(["sum", "count"])
  g = g[g["count"] >= 0.7 * dias]
  g["tasa"] = g["sum"] / g["count"]
  g["valor"] = g["tasa"] * 30.4
  return g.reset_index()[["latitud", "longitud", "valor", "tasa"]]


@st.cache_data(show_spinner="Preparando relieve, fronteras y malla…")
def construir_base(ruta_csv, mtime, resolucion):
  lons = np.linspace(LON_MIN, LON_MAX, resolucion)
  lats = np.linspace(LAT_MIN, LAT_MAX, resolucion)
  grid_lon, grid_lat = np.meshgrid(lons, lats)

  ruta_dem = f"dem_colombia_{resolucion}.npy"
  dem = None
  if os.path.exists(ruta_dem):
    dem = np.load(ruta_dem)
  else:
    try:
      dem = descargar_dem(lons, lats)
      np.save(ruta_dem, dem)
    except Exception:
      dem = None
  dem_real = dem is not None
  if not dem_real:
    dem = modelo_relieve_sintetico(grid_lon, grid_lat)
  elev = np.clip(gaussian_filter(dem, 0.8), 0, 6000)

  geojson_col, geojson_vecinos, _ = cargar_cartografia()
  mascara_col = puntos_dentro(geojson_col, grid_lon, grid_lat)
  mascara_vec = puntos_dentro(geojson_vecinos, grid_lon, grid_lat)

  df = cargar_datos(ruta_csv, mtime)
  est = df.groupby(["latitud", "longitud"], as_index=False)[
      "departamento"
  ].first()
  _, idx = cKDTree(est[["latitud", "longitud"]].values).query(
      np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
  )
  deptos_grid = (
      est["departamento"].to_numpy(dtype=object)[idx].reshape(grid_lat.shape)
  )

  return dict(
      grid_lon=grid_lon,
      grid_lat=grid_lat,
      elev=elev,
      mascara_col=mascara_col,
      mascara_vec=mascara_vec,
      deptos_grid=deptos_grid,
      dem_real=dem_real,
  )


@st.cache_data(show_spinner="Interpolando temperatura del periodo…")
def interpolar_periodo(ruta_csv, mtime, ini_s, fin_s, resolucion, solo_estables):
  df = datos(ruta_csv, mtime, solo_estables)
  base = construir_base(ruta_csv, mtime, resolucion)
  elev = base["elev"]
  n = elev.shape[0]
  forma = elev.shape

  est = estaciones_periodo(df, ini_s, fin_s)
  if len(est) < 3:
    return None

  pts = np.column_stack([base["grid_lat"].ravel(), base["grid_lon"].ravel()])

  def idw(lat_e, lon_e, valores, k=8, p=2.0):
    k = min(k, len(valores))
    d, i = cKDTree(np.column_stack([lat_e, lon_e])).query(pts, k=k)
    if k == 1:
      d, i = d[:, None], i[:, None]
    w = 1.0 / np.maximum(d, 1e-2) ** p
    return (np.sum(w * valores[i], axis=1) / np.sum(w, axis=1)).reshape(forma)

  lat_e = est["latitud"].to_numpy()
  lon_e = est["longitud"].to_numpy()
  t = est["valor"].to_numpy()
  z_e = (
      elev[
          _idx_grid(lat_e, LAT_MIN, LAT_MAX, n),
          _idx_grid(lon_e, LON_MIN, LON_MAX, n),
      ]
      / 1000.0
  )

  pend = 0.0
  if VAR["lapse"]:
    pend = -6.0
    if len(t) >= 15 and np.ptp(z_e) > 0.5:
      pend = float(np.clip(np.polyfit(z_e, t, 1)[0], -8.0, -4.0))
    t0 = float(np.mean(t - pend * z_e))
    residuo = t - (t0 + pend * z_e)
    temp = t0 + pend * elev / 1000.0 + idw(lat_e, lon_e, residuo)
  else:
    temp = np.clip(idw(lat_e, lon_e, t), 0, None)

  anom = np.full(forma, float(VAR["anom_centro"]))
  if ini_s is not None:
    todas = estaciones_periodo(df, None, None)
    m = est.merge(todas, on=["latitud", "longitud"], suffixes=("", "_ref"))
    if VAR["lapse"]:  # temperatura: diferencia en °C
      if len(m) >= 3:
        anom = idw(
            m["latitud"].to_numpy(), m["longitud"].to_numpy(),
            (m["valor"] - m["valor_ref"]).to_numpy(),
        )
    else:  # lluvia: % del promedio diario de todos los datos cargados
      m = m[m["tasa_ref"] > 0.05]
      if len(m) >= 3:
        anom = np.clip(
            idw(
                m["latitud"].to_numpy(), m["longitud"].to_numpy(),
                (m["tasa"] / m["tasa_ref"] * 100.0).to_numpy(),
            ),
            0, 400,
        )

  return dict(temp=temp, anom=anom, n_est=len(est), pendiente=pend)


@st.cache_data
def estaciones_estables(ruta_csv, mtime, umbral=0.8):
  """Estaciones que reportan en al menos `umbral` de los meses del año."""
  d = cargar_datos(ruta_csv, mtime).dropna(subset=["fecha"])
  mes = d["fecha"].dt.to_period("M")
  total = mes.nunique()
  cuenta = d.assign(mes=mes).groupby(["latitud", "longitud"])["mes"].nunique()
  return cuenta[cuenta >= umbral * total].reset_index()[["latitud", "longitud"]]


def datos(ruta_csv, mtime, solo_estables):
  d = cargar_datos(ruta_csv, mtime)
  if solo_estables:
    d = d.merge(
        estaciones_estables(ruta_csv, mtime), on=["latitud", "longitud"]
    )
  return d


@st.cache_data(show_spinner="Calculando la evolución mensual…")
def estadisticas_mensuales(
    ruta_csv, mtime, resolucion, region, suavizado_s, solo_estables
):
  """Valor medio y % del territorio en alerta, por mes (área, no estaciones)."""
  d = datos(ruta_csv, mtime, solo_estables).dropna(subset=["fecha"])
  if d.empty:
    return None
  base = construir_base(ruta_csv, mtime, resolucion)
  m = base["mascara_col"].copy()
  if REGIONES_NATURALES[region] is not None:
    m &= np.isin(base["deptos_grid"], REGIONES_NATURALES[region])
  if not m.any():
    return None
  f_min, f_max = d["fecha"].min().normalize(), d["fecha"].max().normalize()
  filas = []
  for p in pd.period_range(f_min, f_max, freq="M"):
    r = interpolar_periodo(
        ruta_csv, mtime, str(p.start_time), str(p.end_time), resolucion,
        solo_estables,
    )
    if r is None:
      continue
    t = gaussian_filter(r["temp"], suavizado_s) if suavizado_s > 0 else r["temp"]
    an = gaussian_filter(r["anom"], suavizado_s) if suavizado_s > 0 else r["anom"]
    filas.append(
        dict(
            mes=p.start_time,
            media=float(t[m].mean()),
            pct_alerta=float(np.mean(VAR["alerta"](t[m], an[m])) * 100),
            n_est=r["n_est"],
            parcial=es_parcial(p, f_min, f_max),
        )
    )
  if not filas:
    return None
  return pd.DataFrame(filas).set_index("mes")


# ----------------------------------------------------------------------------
# DATOS DE ENTRADA + BARRA LATERAL
# ----------------------------------------------------------------------------
st.sidebar.header("🌍 Filtros Ambientales (2026)")

variable = st.sidebar.selectbox("Variable climática:", list(VARIABLES))
VAR = VARIABLES[variable]
RUTA_CSV = VAR["archivo"]
UNIDAD = VAR["unidad"]
T_MIN, T_MAX = VAR["vmin"], VAR["vmax"]
PARADAS_TEMP = VAR["paradas"]
ESCALA_TEMP = [[(t - T_MIN) / (T_MAX - T_MIN), c] for t, c in PARADAS_TEMP]
ESCALA_ANOM = VAR["escala_div"]
MODO_TEMP = f"{VAR['etq_modo']} ({UNIDAD})"
MODO_ANOM = "Anomalía vs. promedio de los datos"

if not os.path.exists(RUTA_CSV):
  st.error(
      f"No encuentro `{RUTA_CSV}` junto a app.py. Debe tener las columnas:"
      " latitud, longitud, fecha, departamento y el valor"
      f" ({', '.join(sorted(VAR['alias']))})."
  )
  st.stop()

mtime = os.path.getmtime(RUTA_CSV)
try:
  df = cargar_datos(RUTA_CSV, mtime)
except ValueError as e:
  st.error(str(e))
  st.stop()
tiene_fecha = bool(df["fecha"].notna().any())

modo_mapa = st.sidebar.radio("Visualizar:", [MODO_TEMP, MODO_ANOM])
region_sel = st.sidebar.selectbox(
    "Región Natural de Colombia:", list(REGIONES_NATURALES.keys())
)
piso_termico = "Todos los pisos térmicos"
if VAR["lapse"]:
  piso_termico = st.sidebar.selectbox(
      "Piso Térmico (Clasificación Caldas):",
      [
          "Todos los pisos térmicos",
          "Cálido (> 24 °C)",
          "Templado (17.5 – 24 °C)",
          "Frío (12 – 17.5 °C)",
          "Páramo y Nevado (< 12 °C)",
      ],
  )

# Contenedores: fijan el orden de la página. La línea de tiempo va justo
# encima del mapa para usarla con el pulgar en el celular.
cont_kpi = st.container()
cont_ctl = st.container()
cont_vista = st.container()
cont_mapa = st.container()
cont_graf = st.container()

st.sidebar.header("📅 Periodo de análisis")
ini_s, fin_s = None, None
etiqueta_periodo = "Año completo"
meses_disponibles = None
reproducir = False
modo_periodo = "Año completo"
i_mes = None
solo_estables = False
comparando = False
ini_a = fin_a = None

if tiene_fecha:
  f_min = df["fecha"].min().normalize()
  f_max = df["fecha"].max().normalize()
  meses_disponibles = pd.period_range(f_min, f_max, freq="M")

  solo_estables = st.sidebar.checkbox(
      "Comparación justa entre meses",
      value=False,
      help=(
          "Usa solo estaciones que reportan en al menos el 80% de los meses."
          " Así los cambios reflejan el clima y no cuáles estaciones enviaron"
          " datos cada mes."
      ),
  )

  def etiqueta_mes(i, largo=False):
    p = meses_disponibles[i]
    nombre = MESES_LARGO[p.month - 1] if largo else MESES[p.month - 1]
    parcial = " (parcial)" if es_parcial(p, f_min, f_max) else ""
    return f"{nombre} {p.year}{parcial}"

  with cont_ctl:
    st.markdown("##### 📅 Línea de tiempo")
    modo_periodo = st.radio(
        "Periodo:",
        ["Año completo", "Un mes", "Comparar dos meses", "Rango de fechas"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if modo_periodo == "Un mes":
      i_mes = st.select_slider(
          "Mes:",
          options=list(range(len(meses_disponibles))),
          value=len(meses_disponibles) - 1,
          format_func=etiqueta_mes,
          label_visibility="collapsed",
      )
      p = meses_disponibles[i_mes]
      ini_s, fin_s = str(p.start_time), str(p.end_time)
      etiqueta_periodo = etiqueta_mes(i_mes, largo=True)
      reproducir = st.button("▶ Reproducir el año")
    elif modo_periodo == "Comparar dos meses":
      n_m = len(meses_disponibles)
      i_a = st.select_slider(
          "Mes base (A):", options=list(range(n_m)),
          value=max(n_m - 2, 0), format_func=etiqueta_mes,
      )
      i_b = st.select_slider(
          "Mes a comparar (B):", options=list(range(n_m)),
          value=n_m - 1, format_func=etiqueta_mes,
      )
      pa, pb = meses_disponibles[i_a], meses_disponibles[i_b]
      ini_a, fin_a = str(pa.start_time), str(pa.end_time)
      ini_s, fin_s = str(pb.start_time), str(pb.end_time)
      etiqueta_periodo = (
          f"{etiqueta_mes(i_a, largo=True)} → {etiqueta_mes(i_b, largo=True)}"
      )
      comparando = True
      if i_a == i_b:
        st.caption("Elige dos meses distintos para ver diferencias.")
    elif modo_periodo == "Rango de fechas":
      rango = st.date_input(
          "Rango:",
          value=(f_min.date(), f_max.date()),
          min_value=f_min.date(),
          max_value=f_max.date(),
      )
      if isinstance(rango, (tuple, list)) and len(rango) == 2:
        d1, d2 = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
        ini_s = str(d1)
        fin_s = str(d2 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        etiqueta_periodo = f"{d1:%d/%m/%Y} – {d2:%d/%m/%Y}"
      else:
        st.info("Selecciona la fecha final del rango.")
        st.stop()
else:
  st.sidebar.warning(
      "Tu CSV no tiene una columna de fecha reconocible (`fecha` o"
      " `FechaObservacion`), mostrando promedio general."
  )

if comparando:
  modo_mapa = MODO_ANOM
  st.sidebar.caption("Comparando meses: el mapa muestra el cambio B − A.")

st.sidebar.header("🎛️ Controles del mapa")
exag = st.sidebar.slider("Exageración vertical / sombreado (×):", 2, 30, 12, step=1)
suavizado = st.sidebar.slider("Suavizado espacial:", 0.0, 1.5, 0.30, step=0.05)
bandas_discretas = st.sidebar.checkbox(
    "Bandas térmicas definidas (isotermas)", value=True
)
mostrar_deptos = st.sidebar.checkbox(
    "Líneas departamentales sobre relieve", value=True
)
mostrar_vecinos = st.sidebar.checkbox(
    "Relieve 3D de países aledaños (solo vista 3D)", value=True
)
mostrar_ciudades = st.sidebar.checkbox("Etiquetas de capitales", value=True)

# ----------------------------------------------------------------------------
# CÁLCULO
# ----------------------------------------------------------------------------
base = construir_base(RUTA_CSV, mtime, RESOLUCION)
geojson_col, _, deptos_geojson = cargar_cartografia()

if modo_mapa == MODO_ANOM and not tiene_fecha:
  st.warning("La anomalía necesita fechas en el CSV; muestro el valor medio.")
  modo_mapa = MODO_TEMP


def calcular(ini, fin):
  r = interpolar_periodo(RUTA_CSV, mtime, ini, fin, RESOLUCION, solo_estables)
  if r is None:
    return None
  t_out = gaussian_filter(r["temp"], suavizado) if suavizado > 0 else r["temp"]
  a_out = gaussian_filter(r["anom"], suavizado) if suavizado > 0 else r["anom"]
  return t_out, a_out, r["n_est"]


def mascara_filtrada(temp):
  m = base["mascara_col"].copy()
  if REGIONES_NATURALES[region_sel] is not None:
    m &= np.isin(base["deptos_grid"], REGIONES_NATURALES[region_sel])
  if piso_termico == "Cálido (> 24 °C)":
    m &= temp > 24.0
  elif piso_termico == "Templado (17.5 – 24 °C)":
    m &= (temp >= 17.5) & (temp <= 24.0)
  elif piso_termico == "Frío (12 – 17.5 °C)":
    m &= (temp >= 12.0) & (temp < 17.5)
  elif piso_termico == "Páramo y Nevado (< 12 °C)":
    m &= temp < 12.0
  return m


def rango_div():
  """Rango de la escala divergente: cambio entre meses o anomalía."""
  if comparando:
    return (-VAR["lim_cambio"], VAR["lim_cambio"])
  return VAR["anom_rango"]


def unidad_visible():
  if modo_mapa == MODO_TEMP or comparando:
    return UNIDAD
  return VAR["unidad_anom"]


def escala_actual():
  """(colorscale, cmin, cmax, título de la barra) según el modo elegido."""
  if modo_mapa == MODO_TEMP:
    if bandas_discretas:
      escala = []
      for idx_p in range(len(PARADAS_TEMP) - 1):
        v0, c0 = PARADAS_TEMP[idx_p]
        v1, _ = PARADAS_TEMP[idx_p + 1]
        p0 = (v0 - T_MIN) / (T_MAX - T_MIN)
        p1 = (v1 - T_MIN) / (T_MAX - T_MIN)
        escala.extend([[p0, c0], [max(p0, p1 - 0.002), c0]])
      escala.append([1.0, PARADAS_TEMP[-1][1]])
    else:
      escala = ESCALA_TEMP
    return escala, T_MIN, T_MAX, f"{variable} ({UNIDAD})"
  lo, hi = rango_div()
  return (
      ESCALA_ANOM, lo, hi,
      f"{'Cambio' if comparando else 'Anomalía'} ({unidad_visible()})",
  )


def lineas_lonlat():
  """Líneas departamentales (o contorno del país) como arreglos con NaN."""
  if mostrar_deptos and deptos_geojson is not None:
    polys = []
    for feat in deptos_geojson["features"]:
      g = feat["geometry"]
      if g["type"] == "Polygon":
        polys.append(g["coordinates"])
      elif g["type"] == "MultiPolygon":
        polys.extend(g["coordinates"])
  else:
    polys = geojson_col["coordinates"]

  lx, ly = [], []
  for poly in polys:
    for ring in poly:
      arr = np.asarray(ring, dtype=float)[:, :2]
      arr = arr[:: max(1, len(arr) // 1200)]
      lx += [arr[:, 0], [np.nan]]
      ly += [arr[:, 1], [np.nan]]
  if not lx:
    return None, None
  return np.concatenate(lx), np.concatenate(ly)


# ----------------------------------------------------------------------------
# FIGURA 3D
# ----------------------------------------------------------------------------
def construir_figura(
    valores, mascara_activa, modo_tactil, vista_rapida, zoom_nivel=1.0
):
  grid_lon, grid_lat, elev = base["grid_lon"], base["grid_lat"], base["elev"]
  n = elev.shape[0]
  X, Y = a_km(grid_lon, grid_lat)
  Z = elev / 1000.0 * exag

  fig = go.Figure()

  if mostrar_vecinos:
    mascara_fondo = base["mascara_vec"] | base["mascara_col"]
    fig.add_trace(
        go.Surface(
            x=X, y=Y, z=np.where(mascara_fondo, Z - 0.05 * exag, np.nan),
            surfacecolor=elev,
            colorscale=[
                [0.0, "#141b2b"], [0.45, "#2b3649"],
                [0.80, "#4b5a70"], [1.0, "#9aa9bd"],
            ],
            cmin=0, cmax=4800, showscale=False,
            lighting=LUZ, lightposition=POS_LUZ, hoverinfo="skip",
        )
    )

  escala, cmin, cmax, titulo_barra = escala_actual()

  fig.add_trace(
      go.Surface(
          x=X, y=Y, z=np.where(mascara_activa, Z, np.nan),
          surfacecolor=np.where(mascara_activa, valores, np.nan),
          customdata=np.round(np.stack([grid_lon, grid_lat, elev], axis=-1), 2),
          colorscale=escala, cmin=cmin, cmax=cmax,
          colorbar=dict(
              title=dict(text=titulo_barra, font=dict(color="white")),
              tickfont=dict(color="white"), x=0.01, len=0.6,
          ),
          lighting=LUZ, lightposition=POS_LUZ,
          hovertemplate=(
              "<b>Lat:</b> %{customdata[1]:.2f}° | <b>Lon:</b>"
              " %{customdata[0]:.2f}°<br><b>Altitud:</b>"
              " %{customdata[2]:.0f} m<br><b>Valor:</b>"
              " %{surfacecolor:.1f} " + unidad_visible() + "<extra></extra>"
          ),
      )
  )

  def altura(lon, lat):
    fi = (lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * (n - 1)
    fj = (lon - LON_MIN) / (LON_MAX - LON_MIN) * (n - 1)
    ok = np.isfinite(fi) & np.isfinite(fj)
    out = np.full(np.shape(lon), np.nan)
    out[ok] = map_coordinates(elev, [fi[ok], fj[ok]], order=1, mode="nearest")
    return out / 1000.0 * exag

  lon_l, lat_l = lineas_lonlat()
  if lon_l is not None:
    xl, yl = a_km(lon_l, lat_l)
    fig.add_trace(
        go.Scatter3d(
            x=xl, y=yl, z=altura(lon_l, lat_l) + (0.25 * exag), mode="lines",
            line=dict(color="#070b14", width=1.4),
            hoverinfo="skip", showlegend=False,
        )
    )

  if mostrar_ciudades:
    c_lon, c_lat, c_z, c_txt = [], [], [], []
    for nombre, lat, lon in CIUDADES:
      i = _idx_grid(np.array(lat), LAT_MIN, LAT_MAX, n)
      j = _idx_grid(np.array(lon), LON_MIN, LON_MAX, n)
      if mascara_activa[i, j]:
        c_lon.append(lon)
        c_lat.append(lat)
        c_z.append(
            float(altura(np.array([lon]), np.array([lat]))[0]) + (0.85 * exag)
        )
        c_txt.append(nombre)
    if c_lon:
      cx, cy = a_km(np.array(c_lon), np.array(c_lat))
      fig.add_trace(
          go.Scatter3d(
              x=cx, y=cy, z=c_z, mode="markers+text",
              marker=dict(
                  size=3.5, color="white", line=dict(color="black", width=1)
              ),
              text=c_txt, textposition="top center",
              textfont=dict(color="white", size=10.5, family="Arial Black"),
              hoverinfo="text", showlegend=False,
          )
      )

  drag_map = {
      "Rotar mapa (Tornamesa)": "turntable",
      "Mover / Desplazar": "pan",
      "Bloquear mapa (bajar página)": False,
  }

  cam_base = CAMARAS[vista_rapida]
  factor_z = 1.0 / max(zoom_nivel, 0.2)
  camara_actual = dict(
      eye=dict(
          x=cam_base["eye"]["x"] * factor_z,
          y=cam_base["eye"]["y"] * factor_z,
          z=cam_base["eye"]["z"] * factor_z,
      ),
      center=cam_base["center"],
  )

  x0, y0 = a_km(LON_MIN, LAT_MIN)
  x1, y1 = a_km(LON_MAX, LAT_MAX)
  fig.update_layout(
      height=680,
      margin=dict(l=0, r=0, t=0, b=0),
      paper_bgcolor=BG,
      showlegend=False,
      dragmode=drag_map[modo_tactil],
      uirevision=f"{vista_rapida}_{zoom_nivel}",
      scene=dict(
          dragmode=drag_map[modo_tactil],
          xaxis=dict(visible=False, range=[float(x0), float(x1)]),
          yaxis=dict(visible=False, range=[float(y0), float(y1)]),
          zaxis=dict(visible=False, range=[0, 6.2 * exag]),
          aspectmode="data",
          camera=camara_actual,
          bgcolor=BG,
      ),
  )
  return fig


# ----------------------------------------------------------------------------
# MAPA MÓVIL (raster con hillshade sobre un mapa base con gestos nativos)
# ----------------------------------------------------------------------------
def construir_mapa_movil(valores, mascara_activa):
  lons, lats = base["grid_lon"][0], base["grid_lat"][:, 0]
  elev_full = base["elev"]
  n = elev_full.shape[0]

  j0, j1 = np.searchsorted(lons, [-79.5, -66.5])
  i0, i1 = np.searchsorted(lats, [-4.5, 13.0])
  sl = (slice(i0, i1), slice(j0, j1))
  F = 3

  elev = np.clip(zoom(elev_full[sl], F, order=3), 0, None)
  val = zoom(valores[sl], F, order=1)
  alfa = np.clip(
      (zoom(mascara_activa[sl].astype(float), F, order=1) - 0.3) / 0.4, 0, 1
  )
  dx = (lons[1] - lons[0]) * KM_GRADO * 1000 / F
  dy = (lats[1] - lats[0]) * KM_GRADO * 1000 / F
  sombra = hillshade(elev, dx, dy, max(2.0, exag * 0.5))
  rgb = colorear(val) * (0.15 + 1.25 * sombra)[..., None]
  rgba = np.dstack([np.clip(rgb, 0, 255), alfa * 235]).astype(np.uint8)[::-1]

  buf = io.BytesIO()
  Image.fromarray(rgba).save(buf, format="PNG")
  uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
  esquinas = [
      [float(lons[j0]), float(lats[i1 - 1])],
      [float(lons[j1 - 1]), float(lats[i1 - 1])],
      [float(lons[j1 - 1]), float(lats[i0])],
      [float(lons[j0]), float(lats[i0])],
  ]

  Trace = go.Scattermap if USAR_MAPLIBRE else go.Scattermapbox
  clave = "map" if USAR_MAPLIBRE else "mapbox"
  escala, cmin, cmax, titulo_barra = escala_actual()

  fig = go.Figure()

  # Puntos invisibles: dan el popup al tocar y la barra de color
  paso = 4
  ii, jj = np.meshgrid(
      np.arange(0, n, paso), np.arange(0, n, paso), indexing="ij"
  )
  ok = mascara_activa[ii, jj]
  ii, jj = ii[ok], jj[ok]
  pv = valores[ii, jj]
  fig.add_trace(
      Trace(
          lon=base["grid_lon"][ii, jj],
          lat=base["grid_lat"][ii, jj],
          mode="markers",
          marker=dict(
              size=16, opacity=0, color=pv, colorscale=escala,
              cmin=cmin, cmax=cmax, showscale=True,
              colorbar=dict(
                  title=dict(text=titulo_barra, font=dict(color="white")),
                  tickfont=dict(color="white"), len=0.45, thickness=12,
                  x=0.01,
              ),
          ),
          customdata=np.column_stack([pv, elev_full[ii, jj]]),
          hovertemplate=(
              "<b>Valor:</b> %{customdata[0]:.1f} " + unidad_visible()
              + "<br><b>Altitud:</b>"
              " %{customdata[1]:.0f} m<extra></extra>"
          ),
          showlegend=False,
      )
  )

  lon_l, lat_l = lineas_lonlat()
  if lon_l is not None:
    fig.add_trace(
        Trace(
            lon=lon_l, lat=lat_l, mode="lines",
            line=dict(color="rgba(8,12,24,0.85)", width=1),
            hoverinfo="skip", showlegend=False,
        )
    )

  if mostrar_ciudades:
    c_lon, c_lat, c_txt = [], [], []
    for nombre, lat, lon in CIUDADES:
      i = _idx_grid(np.array(lat), LAT_MIN, LAT_MAX, n)
      j = _idx_grid(np.array(lon), LON_MIN, LON_MAX, n)
      if mascara_activa[i, j]:
        c_lon.append(lon)
        c_lat.append(lat)
        c_txt.append(nombre)
    if c_lon:
      fig.add_trace(
          Trace(
              lon=c_lon, lat=c_lat, mode="markers+text",
              marker=dict(size=7, color="white"),
              text=c_txt, textposition="top center",
              textfont=dict(color="white", size=11),
              hoverinfo="text", showlegend=False,
          )
      )

  fig.update_layout(
      height=640,
      margin=dict(l=0, r=0, t=0, b=0),
      paper_bgcolor=BG,
      showlegend=False,
      uirevision="mapa-movil",
      **{
          clave: dict(
              style="carto-darkmatter",
              center=dict(lat=4.6, lon=-73.3),
              zoom=4.4,
              layers=[
                  dict(
                      sourcetype="image", source=uri, coordinates=esquinas,
                      below="traces", opacity=1.0,
                  )
              ],
          )
      },
  )
  return fig


def grafico_evolucion(stats, ini, fin):
  patron = ["/" if p else "" for p in stats["parcial"]]
  fig = go.Figure()
  fig.add_trace(
      go.Bar(
          x=stats.index, y=stats["pct_alerta"], yaxis="y2",
          name=VAR["nombre_alerta"],
          marker=dict(
              color="rgba(239,68,68,0.55)",
              pattern=dict(shape=patron, fgcolor="rgba(255,255,255,0.6)"),
          ),
          hovertemplate="%{x|%b %Y}: %{y:.0f}%<extra></extra>",
      )
  )
  fig.add_trace(
      go.Scatter(
          x=stats.index, y=stats["media"], name=VAR["nombre_media"],
          mode="lines+markers", line=dict(color="#fb923c", width=3),
          marker=dict(size=7),
          hovertemplate="%{x|%b %Y}: %{y:.1f} " + UNIDAD + "<extra></extra>",
      )
  )
  if ini is not None:
    f = "%Y-%m-%d %H:%M:%S"
    fig.add_vrect(
        x0=pd.Timestamp(ini).strftime(f), x1=pd.Timestamp(fin).strftime(f),
        fillcolor="#38bdf8", opacity=0.25, line_width=0,
    )
  fig.update_layout(
      height=300, margin=dict(l=10, r=10, t=50, b=10),
      paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color="#cbd5e1"),
      title=dict(text="Evolución mensual en el territorio", font=dict(size=14)),
      legend=dict(orientation="h", y=1.18, x=0),
      xaxis=dict(gridcolor="#1f2937"),
      yaxis=dict(gridcolor="#1f2937", ticksuffix=f" {UNIDAD}"),
      yaxis2=dict(overlaying="y", side="right", range=[0, 100],
                  ticksuffix="%", showgrid=False),
  )
  return fig


# ----------------------------------------------------------------------------
# PLANCHA CARTOGRÁFICA (PNG DESCARGABLE)
# ----------------------------------------------------------------------------
def generar_infografia(valores, mascara_activa, kpis, serie, n_est):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.patheffects as pe
  import matplotlib.pyplot as plt
  from matplotlib.colors import (
      LightSource, LinearSegmentedColormap, Normalize, to_rgb,
  )
  from matplotlib.patches import Polygon, Rectangle

  lons, lats = base["grid_lon"][0], base["grid_lat"][:, 0]
  lon_w, lon_e = -79.8, -66.2
  lat_s, lat_n = -4.6, 13.0
  j0, j1 = np.searchsorted(lons, [lon_w, lon_e])
  i0, i1 = np.searchsorted(lats, [lat_s, lat_n])
  sl = (slice(i0, i1), slice(j0, j1))
  F = 3

  def prep_arr(a, orden):
    return zoom(a[sl].astype(float), F, order=orden)[::-1]

  def prep_mask(m):
    return np.repeat(np.repeat(m[sl], F, 0), F, 1)[::-1]

  elev = np.clip(prep_arr(base["elev"], 3), 0, None)
  val = prep_arr(valores, 1)
  act = prep_mask(mascara_activa)
  col = prep_mask(base["mascara_col"])
  vec = prep_mask(base["mascara_vec"]) & ~col

  if modo_mapa == MODO_TEMP:
    escala_m = [
        [(t - T_MIN) / (T_MAX - T_MIN), c] for t, c in VAR["plancha_paradas"]
    ]
    cmap = LinearSegmentedColormap.from_list("ideam", [tuple(p) for p in escala_m])
    norm = Normalize(T_MIN, T_MAX)
    titulo_mapa = VAR["plancha_titulo"]
  else:
    cmap = LinearSegmentedColormap.from_list("a", [tuple(p) for p in ESCALA_ANOM])
    lo_d, hi_d = rango_div()
    norm = Normalize(lo_d, hi_d)
    titulo_mapa = (
        VAR["plancha_titulo_cambio"] if comparando else VAR["plancha_titulo_anom"]
    )

  rgb = np.zeros(elev.shape + (3,))
  rgb[:] = to_rgb("#cce3f5")
  rgb[vec] = to_rgb("#e5ebe3")
  rgb[col & ~act] = to_rgb("#d1d5db")
  rgb[act] = cmap(norm(val[act]))[:, :3]

  dx = (lons[1] - lons[0]) * KM_GRADO * 1000 / F
  dy = (lats[1] - lats[0]) * KM_GRADO * 1000 / F
  ls = LightSource(azdeg=315, altdeg=40)
  rgb = np.clip(
      ls.shade_rgb(
          rgb, elev, fraction=0.85, blend_mode="overlay", vert_exag=6, dx=dx, dy=dy
      ),
      0,
      1,
  )

  fig = plt.figure(figsize=(10.5, 14.0), facecolor="white")

  ax = fig.add_axes([0.07, 0.24, 0.86, 0.71], facecolor="#cce3f5")
  ext = [lons[j0], lons[j1 - 1], lats[i0], lats[i1 - 1]]
  ax.imshow(rgb, extent=ext, interpolation="bilinear", aspect="auto")

  _, geojson_vec_local, _ = cargar_cartografia()
  for poly in geojson_vec_local["coordinates"]:
    for ring in poly:
      arr = np.asarray(ring, dtype=float)[:, :2]
      ax.plot(arr[:, 0], arr[:, 1], color="#64748b", lw=0.75, ls="--", alpha=0.85)

  if deptos_geojson is not None:
    for feat in deptos_geojson["features"]:
      g = feat["geometry"]
      polys_d = (
          [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
      )
      for poly in polys_d:
        for ring in poly:
          arr = np.asarray(ring, dtype=float)[:, :2]
          ax.plot(arr[:, 0], arr[:, 1], color="#1e293b", lw=0.75)

  ax.set_xlim(lons[j0], lons[j1 - 1])
  ax.set_ylim(lats[i0], lats[i1 - 1])

  xticks = np.arange(-78, -66, 3)
  yticks = np.arange(-3, 13, 3)
  ax.set_xticks(xticks)
  ax.set_yticks(yticks)
  ax.set_xticklabels(
      [f"{abs(x)}°00'00\"W" for x in xticks], fontsize=7.5, color="#1e293b"
  )
  ax.set_yticklabels(
      [f"{abs(y)}°00'00\"{'N' if y >= 0 else 'S'}" for y in yticks],
      fontsize=7.5, color="#1e293b", rotation=90, va="center",
  )
  ax.tick_params(
      top=True, bottom=True, left=True, right=True,
      labeltop=True, labelbottom=True, labelleft=True, labelright=True,
      direction="out", length=4, width=1.0, colors="#1e293b",
  )
  ax.grid(True, color="#475569", linestyle="-", linewidth=0.45, alpha=0.55)
  for spine in ax.spines.values():
    spine.set_edgecolor("#0f172a")
    spine.set_linewidth(1.8)

  estilo_pais = dict(color="#475569", fontsize=10.5, fontweight="bold", alpha=0.85)
  ax.text(-78.5, 8.6, "Panamá", **estilo_pais)
  ax.text(-69.5, 8.3, "Venezuela", **estilo_pais)
  ax.text(-68.6, -1.8, "Brasil", **estilo_pais)
  ax.text(-73.8, -3.6, "Perú", **estilo_pais)
  ax.text(-78.6, -1.4, "Ecuador", **estilo_pais)

  halo_mar = [pe.withStroke(linewidth=2.0, foreground="white")]
  ax.text(
      -76.8, 11.6, "MAR CARIBE", color="#0369a1", fontsize=9,
      fontweight="bold", style="italic", path_effects=halo_mar,
  )
  ax.text(
      -79.3, 3.8, "OCÉANO PACÍFICO", color="#0369a1", fontsize=9,
      fontweight="bold", style="italic", path_effects=halo_mar,
  )

  halo_txt = [pe.withStroke(linewidth=2.2, foreground="white")]
  if mostrar_ciudades:
    for nombre, lat, lon in CIUDADES:
      if lons[j0] < lon < lons[j1 - 1] and lats[i0] < lat < lats[i1 - 1]:
        ii = _idx_grid(np.array(lat), LAT_MIN, LAT_MAX, len(lats))
        jj = _idx_grid(np.array(lon), LON_MIN, LON_MAX, len(lons))
        if mascara_activa[ii, jj]:
          ax.plot(
              lon, lat, "o", ms=4.2, mfc="#0f172a", mec="white", mew=0.9, zorder=6
          )
          ax.text(
              lon, lat + 0.14, nombre, color="#0f172a", fontsize=7.8,
              ha="center", fontweight="bold", path_effects=halo_txt, zorder=7,
          )

  ax_sai = fig.add_axes([0.095, 0.80, 0.11, 0.125], facecolor="#cce3f5")
  ax_sai.set_xlim(-81.85, -81.25)
  ax_sai.set_ylim(12.45, 13.45)
  sai_poly = Polygon(
      [(-81.73, 12.48), (-81.69, 12.48), (-81.68, 12.59), (-81.72, 12.59)],
      closed=True, facecolor="#f97316", edgecolor="#0f172a", lw=0.8,
  )
  prov_poly = Polygon(
      [(-81.39, 13.32), (-81.34, 13.32), (-81.34, 13.39), (-81.39, 13.39)],
      closed=True, facecolor="#f97316", edgecolor="#0f172a", lw=0.8,
  )
  ax_sai.add_patch(sai_poly)
  ax_sai.add_patch(prov_poly)
  ax_sai.text(-81.70, 12.66, "SAN ANDRÉS", fontsize=5.5, ha="center", fontweight="bold")
  ax_sai.text(-81.36, 13.22, "PROVIDENCIA", fontsize=5.5, ha="center", fontweight="bold")
  ax_sai.set_xticks([])
  ax_sai.set_yticks([])
  for s in ax_sai.spines.values():
    s.set_edgecolor("#0f172a")
    s.set_linewidth(1.1)

  ax_n = fig.add_axes([0.10, 0.30, 0.06, 0.06], facecolor="none")
  ax_n.set_xlim(-1, 1)
  ax_n.set_ylim(-1, 1)
  ax_n.axis("off")
  ax_n.add_patch(Polygon([(0, 0.75), (-0.22, -0.35), (0, -0.1)], closed=True, facecolor="#0f172a"))
  ax_n.add_patch(Polygon([(0, 0.75), (0.22, -0.35), (0, -0.1)], closed=True, facecolor="white", edgecolor="#0f172a", lw=0.8))
  ax_n.text(0, 0.85, "N", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0f172a")

  ax_sc = fig.add_axes([0.09, 0.255, 0.16, 0.03], facecolor="none")
  ax_sc.set_xlim(0, 300)
  ax_sc.set_ylim(0, 1)
  ax_sc.axis("off")
  ax_sc.add_patch(Rectangle((0, 0.4), 150, 0.25, facecolor="#0f172a", edgecolor="#0f172a"))
  ax_sc.add_patch(Rectangle((150, 0.4), 150, 0.25, facecolor="white", edgecolor="#0f172a"))
  ax_sc.text(0, 0.05, "0", fontsize=6.5, ha="center")
  ax_sc.text(150, 0.05, "150", fontsize=6.5, ha="center")
  ax_sc.text(300, 0.05, "300 km", fontsize=6.5, ha="center")

  ax_caj = fig.add_axes([0.07, 0.035, 0.86, 0.175], facecolor="white")
  ax_caj.set_xlim(0, 1)
  ax_caj.set_ylim(0, 1)
  ax_caj.set_xticks([])
  ax_caj.set_yticks([])
  for s in ax_caj.spines.values():
    s.set_edgecolor("#0f172a")
    s.set_linewidth(1.8)

  for x_div in [0.29, 0.49, 0.71]:
    ax_caj.plot([x_div, x_div], [0, 1], color="#0f172a", lw=1.2)

  ax_caj.add_patch(Rectangle((0.0, 0.78), 0.29, 0.22, facecolor="#0284c7"))
  ax_caj.text(
      0.145, 0.89, "GEOPORTAL AMBIENTAL · COLOMBIA", color="white",
      fontsize=7.8, fontweight="bold", ha="center", va="center",
  )
  ax_caj.text(
      0.145, 0.56, titulo_mapa, color="#0f172a", fontsize=8.5,
      fontweight="bold", ha="center", va="center", linespacing=1.25,
  )
  ax_caj.text(
      0.145, 0.27, f"Periodo: {etiqueta_periodo}\nÁrea: {region_sel}",
      color="#334155", fontsize=7.5, ha="center", va="center", linespacing=1.3,
  )
  ax_caj.text(
      0.145, 0.08, "REPÚBLICA DE COLOMBIA · 2026", color="#0f172a",
      fontsize=7.5, fontweight="bold", ha="center", va="center",
  )

  ax_caj.text(
      0.39, 0.90, f"Leyenda ({UNIDAD})" if modo_mapa == MODO_TEMP else (f"Cambio ({UNIDAD})" if comparando else f"Anomalía ({VAR['unidad_anom']})"),
      fontsize=8.5, fontweight="bold", ha="center", color="#0f172a",
  )
  if modo_mapa == MODO_TEMP:
    rangos_ley = VAR["leyenda"]
  else:
    rangos_ley = VAR["leyenda_cambio"] if comparando else VAR["leyenda_anom"]

  y_top = 0.77
  step_y = 0.72 / len(rangos_ley)
  for idx_r, (col_hex, lbl_r) in enumerate(rangos_ley):
    yy = y_top - idx_r * step_y
    ax_caj.add_patch(
        Rectangle((0.315, yy - 0.045), 0.042, 0.065, facecolor=col_hex, edgecolor="#334155", lw=0.6)
    )
    ax_caj.text(0.372, yy - 0.015, lbl_r, fontsize=7.3, va="center", color="#1e293b")

  ax_caj.text(
      0.60, 0.90, "Convenciones", fontsize=8.5, fontweight="bold",
      ha="center", color="#0f172a",
  )
  ax_caj.plot(0.525, 0.74, "o", ms=4.8, mfc="#0f172a", mec="white", mew=0.8)
  ax_caj.text(0.55, 0.74, "Capital / Ciudad", fontsize=7.2, va="center")

  ax_caj.plot([0.508, 0.542], [0.58, 0.58], color="#1e293b", lw=1.2)
  ax_caj.text(0.55, 0.58, "Límite Departamental", fontsize=7.2, va="center")

  ax_caj.plot([0.508, 0.542], [0.42, 0.42], color="#64748b", lw=1.0, ls="--")
  ax_caj.text(0.55, 0.42, "Límite Internacional", fontsize=7.2, va="center")

  ax_caj.add_patch(Rectangle((0.508, 0.23), 0.034, 0.06, facecolor="#cce3f5", edgecolor="#64748b", lw=0.5))
  ax_caj.text(0.55, 0.26, "Océanos / Agua", fontsize=7.2, va="center")

  ax_caj.text(
      0.855, 0.90, "Información de Referencia", fontsize=8.5,
      fontweight="bold", ha="center", color="#0f172a",
  )
  meta_txt = (
      f"PROYECCIÓN: Geográfica (WGS84)\n"
      f"DATUM: MAGNA-SIRGAS\n"
      f"ESTACIONES IDEAM: {n_est} activas\n"
      f"{kpis[0][0].upper()}: {kpis[0][1]}\n"
      f"{kpis[1][0].upper()}: {kpis[1][1]}\n"
      f"{kpis[2][0].upper()}: {kpis[2][1]}\n"
      f"MÉTODO: {VAR['metodo']}"
  )
  ax_caj.text(
      0.722, 0.45, meta_txt, fontsize=6.9, va="center", ha="left",
      color="#1e293b", linespacing=1.45,
  )

  fig.text(
      0.5, 0.015,
      f"Fuente de datos: {VAR['fuente']} · Modelo"
      " Digital de Elevación: SRTM / AWS Terrain Tiles",
      ha="center", color="#475569", fontsize=7.5,
  )

  buf = io.BytesIO()
  fig.savefig(buf, format="png", dpi=220, facecolor="white", bbox_inches="tight")
  plt.close(fig)
  return buf.getvalue()


# ----------------------------------------------------------------------------
# RENDER PRINCIPAL
# ----------------------------------------------------------------------------
res = calcular(ini_s, fin_s)
if res is None:
  st.warning("No hay estaciones con datos en ese periodo. Prueba otro rango.")
  st.stop()
grid_temp, grid_anom, n_est = res
cambio = None
if comparando:
  res_a = calcular(ini_a, fin_a)
  if res_a is None:
    st.warning("No hay estaciones con datos en el mes base (A).")
    st.stop()
  cambio = grid_temp - res_a[0]
  valores = cambio
else:
  valores = grid_temp if modo_mapa == MODO_TEMP else grid_anom
mascara_activa = mascara_filtrada(grid_temp)

if not mascara_activa.any():
  st.warning(
      "Los filtros actuales no dejan ningún territorio. Ajusta región o piso térmico."
  )
  st.stop()

temps_validas = grid_temp[mascara_activa]
d_ = VAR["dec"]
if comparando:
  dif = cambio[mascara_activa]
  etiquetas_kpi = VAR["kpi_cambio"]
  kpi_vals = [
      (etiquetas_kpi[0], f"{np.nanmean(dif):+.{d_}f} {UNIDAD}"),
      (etiquetas_kpi[1], f"{np.nanmin(dif):+.{d_}f} / {np.nanmax(dif):+.{d_}f}"),
      (etiquetas_kpi[2], f"{np.mean(VAR['cambio_alerta'](dif)) * 100:.0f}%"),
      ("Estaciones usadas", f"{n_est}"),
  ]
else:
  etiquetas_kpi = VAR["kpi"]
  if VAR["lapse"] or ini_s is not None:
    pct_a = VAR["alerta"](
        grid_temp[mascara_activa], grid_anom[mascara_activa]
    )
    pct_a = f"{np.mean(pct_a) * 100:.0f}%"
  else:
    pct_a = "—"  # sin fechas no hay periodo contra el cual medir el déficit
  kpi_vals = [
      (VAR["kpi_corto"][0], f"{np.nanmean(temps_validas):.{d_}f} {UNIDAD}"),
      (
          VAR["kpi_corto"][1],
          f"{np.nanmin(temps_validas):.{d_}f} / {np.nanmax(temps_validas):.{d_}f}",
      ),
      (VAR["kpi_corto"][2], pct_a),
      ("Estaciones usadas", f"{n_est}"),
  ]

stats = (
    estadisticas_mensuales(
        RUTA_CSV, mtime, RESOLUCION, region_sel, float(suavizado), solo_estables
    )
    if tiene_fecha
    else None
)

delta_t = delta_c = None
if stats is not None and i_mes:
  ts_act = meses_disponibles[i_mes].start_time
  ts_prev = meses_disponibles[i_mes - 1].start_time
  if ts_act in stats.index and ts_prev in stats.index:
    dt = stats.loc[ts_act, "media"] - stats.loc[ts_prev, "media"]
    dc = stats.loc[ts_act, "pct_alerta"] - stats.loc[ts_prev, "pct_alerta"]
    delta_t = f"{dt:+.{VAR['dec']}f} {UNIDAD} vs. mes anterior"
    delta_c = f"{dc:+.0f} pts vs. mes anterior"

with cont_kpi:
  k1, k2, k3, k4 = st.columns(4)
  k1.metric("Periodo · Área", f"{etiqueta_periodo} · {region_sel.split(' (')[0]}")
  k2.metric(etiquetas_kpi[0], kpi_vals[0][1], delta=delta_t, delta_color="off")
  k3.metric(etiquetas_kpi[1], kpi_vals[1][1])
  k4.metric(
      etiquetas_kpi[2], kpi_vals[2][1],
      delta=delta_c, delta_color="off",
  )

# --- Tipo de vista y controles de navegación ---
modo_tactil = "Rotar mapa (Tornamesa)"
vista_rapida = "Inclinada (3D)"
zoom_nivel = 1.0
congelar_mapa = False

with cont_vista:
  vista = st.radio("Vista:", [VISTA_MAPA, VISTA_3D], horizontal=True)
  if vista == VISTA_3D:
    c_nav1, c_nav2, c_nav3 = st.columns([2, 1, 1])
    with c_nav1:
      modo_tactil = st.radio(
          "Control táctil / mouse:",
          [
              "Rotar mapa (Tornamesa)",
              "Mover / Desplazar",
              "Bloquear mapa (bajar página)",
          ],
          horizontal=True,
      )
    with c_nav2:
      vista_rapida = st.selectbox("Enfoque rápido:", list(CAMARAS.keys()))
    with c_nav3:
      zoom_nivel = st.slider("Zoom:", 0.4, 2.2, 1.0, step=0.1)
  else:
    st.caption(
        "Pellizca con dos dedos para acercar, arrastra con uno para moverte"
        " y toca el mapa para ver temperatura y altitud."
    )
    congelar_mapa = st.checkbox(
        "🔒 Congelar mapa (para bajar la página con el dedo)", value=False
    )


def renderizar(destino, vals, mask, key):
  if vista == VISTA_3D:
    fig = construir_figura(vals, mask, modo_tactil, vista_rapida, zoom_nivel)
    cfg = {"displaylogo": False, "scrollZoom": True}
    k = f"mapa3d_{key}"
  else:
    fig = construir_mapa_movil(vals, mask)
    cfg = {
        "displaylogo": False,
        "scrollZoom": True,
        "displayModeBar": False,
        "staticPlot": congelar_mapa,
    }
    k = f"mapamovil_{key}"
  destino.plotly_chart(fig, use_container_width=True, key=k, config=cfg)


with cont_mapa:
  titulo_ph = st.empty()
  mapa_ph = st.empty()

if reproducir and meses_disponibles is not None:
  for p in meses_disponibles:
    r = calcular(str(p.start_time), str(p.end_time))
    if r is None:
      continue
    t_, a_, _ = r
    m_ = mascara_filtrada(t_)
    if not m_.any():
      continue
    titulo_ph.markdown(
        f"### {MESES_LARGO[p.month - 1]} {p.year} · {np.nanmean(t_[m_]):.{VAR['dec']}f} {UNIDAD}"
    )
    renderizar(
        mapa_ph, t_ if modo_mapa == MODO_TEMP else a_, m_, f"anim_{p}"
    )
    time.sleep(0.8)
  titulo_ph.empty()

renderizar(mapa_ph, valores, mascara_activa, "principal")

with cont_graf:
  if stats is not None and len(stats) >= 2:
    st.plotly_chart(
        grafico_evolucion(stats, ini_s, fin_s),
        use_container_width=True, key="evolucion",
    )

    def nombre_mes(ts):
      return MESES_LARGO[ts.month - 1] + (
          " (parcial)" if stats.loc[ts, "parcial"] else ""
      )

    idx_ext = (
        stats["media"].idxmax() if VAR["extremo"] == "max"
        else stats["media"].idxmin()
    )
    mas_extenso = stats["pct_alerta"].idxmax()
    st.caption(
        f"{VAR['etq_extremo']}: **{nombre_mes(idx_ext)}**"
        f" ({stats.loc[idx_ext, 'media']:.{VAR['dec']}f} {UNIDAD}) ·"
        f" {VAR['etq_alerta']}: **{nombre_mes(mas_extenso)}**"
        f" ({stats.loc[mas_extenso, 'pct_alerta']:.0f}%). Calculado sobre la"
        " región elegida, sin el filtro de piso térmico. Las barras rayadas"
        " son meses con datos incompletos."
    )
  st.caption(
      "Relieve: "
      + (
          "modelo de elevación real (AWS Terrain Tiles / SRTM)."
          if base["dem_real"]
          else "⚠️ SINTÉTICO (revisa tu conexión y usa el botón de recargar)."
      )
      + (
          " La temperatura se interpola con IDW sobre los residuos del"
          " gradiente altitudinal."
          if VAR["lapse"]
          else " La lluvia se interpola con IDW (sin corrección por altitud) y"
               " se expresa en mm/mes equivalentes (promedio diario × 30,4)."
      )
      + " La anomalía compara cada periodo con el promedio de los datos"
        " cargados de la misma estación."
  )
  if modo_mapa == MODO_ANOM and not comparando:
    st.caption(
        "⚠️ Esta anomalía usa como referencia los mismos datos cargados."
        " Para medir el efecto de El Niño hace falta una referencia histórica"
        " (varios años o la normal climatológica 1991–2020)."
    )

# ----------------------------------------------------------------------------
# BARRA LATERAL: infografía y mantenimiento
# ----------------------------------------------------------------------------
st.sidebar.header("🖼️ Plancha Cartográfica")
if st.sidebar.button("Generar plancha PNG"):
  try:
    with st.spinner("Dibujando plancha cartográfica…"):
      png = generar_infografia(valores, mascara_activa, kpi_vals, None, n_est)
    nombre_png = (
        f"plancha_{_normalizar(variable)}_{_normalizar(etiqueta_periodo).replace(chr(40), chr(0)).replace(chr(41), chr(0)).replace(chr(0), chr(32)).strip().replace(chr(32), chr(45))}"
        f"_{_normalizar(region_sel.split(' (')[0])}.png"
    )
    st.session_state["infografia"] = (nombre_png, png)
  except ImportError:
    st.sidebar.error("Falta matplotlib: `pip install matplotlib`")

if "infografia" in st.session_state:
  nombre_png, png = st.session_state["infografia"]
  st.sidebar.download_button(
      "⬇️ Descargar plancha PNG",
      data=png,
      file_name=nombre_png,
      mime="image/png",
  )
  st.sidebar.caption("Se genera con los filtros del momento de pulsar el botón.")

if st.sidebar.button("♻️ Recargar relieve y caché"):
  st.cache_data.clear()
  st.rerun()
