import time
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://www.datos.gov.co/resource/sbwg-7ju4.json"
todos_los_meses = []

print("Descargando serie diaria de temperatura IDEAM (2026) mes por mes...")

periodos = [
    ("2026-01-01", "2026-02-01", "Enero"),
    ("2026-02-01", "2026-03-01", "Febrero"),
    ("2026-03-01", "2026-04-01", "Marzo"),
    ("2026-04-01", "2026-05-01", "Abril"),
    ("2026-05-01", "2026-06-01", "Mayo"),
    ("2026-06-01", "2026-07-01", "Junio"),
    ("2026-07-01", "2026-08-01", "Julio"),
    ("2026-08-01", "2026-09-01", "Agosto"),
    ("2026-09-01", "2026-10-01", "Septiembre"),
    ("2026-10-01", "2026-11-01", "Octubre"),
    ("2026-11-01", "2026-12-01", "Noviembre"),
]

for inicio, fin, nombre_mes in periodos:
  params = {
      "$select": (
          "codigoestacion, nombreestacion, departamento, municipio, latitud,"
          " longitud, date_trunc_ymd(fechaobservacion) as fecha,"
          " avg(valorobservado) as temperatura"
      ),
      "$where": (
          f"fechaobservacion >= '{inicio}T00:00:00' AND fechaobservacion <"
          f" '{fin}T00:00:00' AND valorobservado > -5 AND valorobservado < 48"
      ),
      "$group": (
          "codigoestacion, nombreestacion, departamento, municipio, latitud,"
          " longitud, fecha"
      ),
      "$limit": 25000,
  }

  try:
    resp = requests.get(url, params=params, verify=False, timeout=60)
    resp.raise_for_status()
    datos = resp.json()
    if datos:
      todos_los_meses.extend(datos)
      print(f"  ✓ {nombre_mes}: {len(datos)} registros diarios")
    else:
      print(f"  - {nombre_mes}: sin datos aún")
  except Exception as e:
    print(f"  ✗ {nombre_mes}: error ({e}), continuando...")
  time.sleep(0.5)

df = pd.DataFrame(todos_los_meses)
df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
df["temperatura"] = pd.to_numeric(df["temperatura"], errors="coerce").round(2)
df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
df = df.dropna(subset=["fecha", "latitud", "longitud", "temperatura"])

df.to_csv("temperaturas.csv", index=False, encoding="utf-8")
print(
    f"\n¡Listo! Guardadas {len(df)} observaciones diarias desde"
    f" {df['fecha'].min()} hasta {df['fecha'].max()}."
)