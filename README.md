# GLM-Radar Córdoba (RMA-1): Catálogo Convectivo y Selección de Casos Radar

Pipeline automatizado, parametrizado y reproducible para la construcción de una base de eventos convectivos a partir del sensor **GOES-16/19 GLM** (Geostationary Lightning Mapper) orientada al análisis y selección de períodos de observación del radar meteorológico **RMA-1** (UNC, Córdoba, Argentina: $-31.4413^\circ\text{ S},\ -64.1919^\circ\text{ O}$).

---

## 🎯 Objetivo

Construir un catálogo físico y reproducible de eventos convectivos electrificados que intersecten el dominio útil de RMA-1 ($d \le 240\text{ km}$) para programar y optimizar la descarga de datos radar, distinguiendo:
- **Prioridad A (`long_intense`)**: Tormentas eléctricas intensas y persistentes ($\ge 80\text{ min}$ y $\ge 50\text{ flashes/10 min}$).
- **Prioridad B (`short_intense` / `long_moderate`)**: Células intensas de corta duración o sistemas moderados persistentes.
- **Ventanas temporales de radar**: $[t_{\mathrm{start}} - 60\text{ min},\, t_{\mathrm{end}} + 60\text{ min}]$ con resolución automática del cruce de medianoche para generar listas de días calendario exactos a descargar.

---

## 🔬 Metodología y Pipeline

```text
[ NOAA S3: GOES-16/19 GLM-L2-LCFA ]
                     │
                     ▼ (Búsqueda paralela multiproceso ThreadPoolExecutor)
        [ Extracción de Flashes Nativos ] (BBox Córdoba: -66.0 a -61.8 O, -35.6 a -29.3 S)
                     │
                     ▼ (Binning temporal 10 min)
       [ Clustering Espacial: DBSCAN ] (ε = 15 km, N_min = 5 flashes)
                     │
                     ▼
        [ Tracking Temporal Continuo ] (Asignación Húngara, d_max = 30 km/ventana, gap_max = 1)
                     │
                     ▼
       [ Clasificación Eléctrica ] (Umbrales congelados: D* = 80 min, F* = 50 flashes/10 min)
                     │
                     ▼
       [ Intersección Geométrica RMA-1 ] (min(d_track, RMA1) <= 240 km)
                     │
                     ▼
     [ Selección Convectiva y Ventanas ] (Buffers ±60 min + Resolución de medianoche)
                     │
                     ▼
         [ Catálogo y Días Radar ]
```

### 1. Detección y Tracking
1. **Clustering espacial (10 min):** Agrupa flashes nativos usando DBSCAN con métrica Haversine ($\epsilon = 15\text{ km}$, $N_{\min} = 5\text{ flashes}$).
2. **Tracking temporal:** Enlaza centroides entre ventanas consecutivas usando el algoritmo de asignación lineal (Húngaro) con costo de distancia geodésica ($d_{\max} = 30\text{ km/ventana}$, tolerancia de 1 ventana faltante $\text{gap}_{\max} = 1$).
3. **Continuidad temporal:** El tracking anual procesa la secuencia cronológica continua de clusters, unificando sin costuras artificiales cualquier tormenta que cruce la medianoche de cambio de mes.

### 2. Clasificación Eléctrica (Umbrales Congelados)
Para garantizar la comparabilidad interanual sin sesgos estacionales ni recalibraciones mensuales, se aplican umbrales fijos derivados de la línea base del P75 anual:
- **Duración límite ($D^*$):** $80.0\text{ min}$
- **Tasa eléctrica máxima ($F^*$):** $50.0\text{ flashes / 10 min}$

$$\begin{array}{c|cc}
& F < 50 & F \ge 50 \\
\hline
D < 80\text{ min} & \text{short\_lower\_activity} & \text{short\_intense (Prioridad B)} \\
D \ge 80\text{ min} & \text{long\_moderate (Prioridad B)} & \text{long\_intense (Prioridad A)}
\end{array}$$

### 3. Selección Radar y Ventanas
Se seleccionan todos los tracks que cumplan $\min(d_{\text{track}, \text{RMA1}}) \le 240.0\text{ km}$ clasificados como `long_intense`, `short_intense` o `long_moderate`. Para cada evento se genera una ventana temporal $[t_{\mathrm{start}} - 1\text{ h},\, t_{\mathrm{end}} + 1\text{ h}]$, agrupando por día calendario y asignando la prioridad más alta presente en el día.

---

## 📊 Resumen Multianual Consolidado

| Métrica | Año 2021 | Año 2022 | Consistencia |
| :--- | :---: | :---: | :---: |
| **Clusters GLM (10 min)** | 42,915 | 41,009 | $-4.4\%$ |
| **Tracks totales detectados** | 7,906 | 7,447 | $-5.8\%$ |
| **Tracks dentro de 240 km RMA-1** | 4,718 ($59.7\%$) | 4,366 ($58.6\%$) | $\Delta = 1.1\text{ pp}$ |
| **Prioridad A (`long_intense`)** | 865 | 838 | $-3.1\%$ |
| **Prioridad B (`short_intense`)** | 461 | 449 | $-2.6\%$ |
| **Prioridad B (`long_moderate`)** | 461 | 420 | $-8.9\%$ |
| **Total eventos convectivos RMA-1** | **1,787** | **1,707** | $-4.5\%$ |
| **% convectivos de tracks $\le 240\text{ km}$** | **37.9%** | **39.1%** | $\Delta = 1.2\text{ pp}$ |
| **Días calendario de radar RMA-1** | **135 días** | **117 días** | $-13.3\%$ |
| **Eventos / día de radar** | **13.2** | **14.6** | Muy estable |

---

## 📁 Estructura del Proyecto

```text
GLM_radar/
├── radar_catalog_config.yaml       # Configuración desacoplada (radar, dominios, umbrales)
├── build_glm_radar_catalog.py       # Pipeline principal (descarga S3, tracking, selección)
├── validate_glm_tracks.py          # Script de validación cinemática y generación de diagnósticos
├── requirements.txt                # Dependencias de Python
├── output_2021_anual/              # Catálogo anual consolidado 2021
│   ├── diagnostic_summary.txt
│   ├── annual_summary.csv
│   ├── radar_convective_events_240km_2021.csv
│   └── radar_download_days_240km_2021.csv
├── output_2022_anual/              # Catálogo anual consolidado 2022
│   ├── diagnostic_summary.txt
│   ├── annual_summary.csv
│   ├── radar_convective_events_240km_2022.csv
│   └── radar_download_days_240km_2022.csv
└── validation_plots/               # Gráficos diagnósticos de validación de tracks
```

---

## 🚀 Instalación y Uso

### 1. Clonar e Instalar Dependencias
```bash
git clone git@github.com:LuciaSandalio/GLM_radar.git
cd GLM_radar
pip install -r requirements.txt
```

### 2. Procesar un Mes Individual
```bash
python build_glm_radar_catalog.py --year 2022 --month 12
```

### 3. Consolidar un Año Completo (con Tracking Continuo)
```bash
python build_glm_radar_catalog.py --consolidate-year 2022
```

### 4. Validar Cinemática y Generar Diagnósticos Visuales
```bash
python validate_glm_tracks.py
```

---

## ⚙️ Configuración (`radar_catalog_config.yaml`)

Todos los parámetros físicos y operativos se definen sin *hardcoding*:
```yaml
radar:
  name: "RMA-1"
  lat: -31.4413
  lon: -64.1919

tracking:
  time_bin_minutes: 10
  cluster_radius_km: 15.0
  min_flashes_cluster: 5
  max_link_distance_km: 30.0
  max_gap_windows: 1

classification:
  mode: "frozen"
  duration_threshold_min: 80.0
  intensity_threshold_flashes_10min: 50.0

selection:
  radar_intersection_radius_km: 240.0
  include_types:
    - "long_intense"
    - "short_intense"
    - "long_moderate"
```

---

## 👤 Autora
- **Lucia Elena Sandalio** ([@LuciaSandalio](https://github.com/LuciaSandalio))
- Universidad Nacional de Córdoba (UNC) / Observatorio Hidro-Meteorológico de Córdoba (OHMC).
