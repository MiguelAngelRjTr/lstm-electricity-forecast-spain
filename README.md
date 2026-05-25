# ⚡ Predicción de consumo eléctrico en España con LSTM

Proyecto de la asignatura **Redes Neuronales y Aprendizaje Profundo** — UAX.

Predice el consumo eléctrico real en España hora a hora usando una red LSTM entrenada sobre datos históricos de energía y clima (2015–2018).

---

## Estructura del proyecto

```
proyecto/
├── data/
│   ├── energy_dataset.csv
│   └── weather_features.csv
├── model/
│   ├── lstm_final.keras
│   ├── lstm_best.keras
│   ├── scaler.pkl
│   ├── features.pkl
│   ├── target_idx.pkl
│   └── window.pkl
├── 01_eda.ipynb
├── 02_model.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## Dataset

- **Fuente:** [Kaggle — Energy consumption, generation, prices and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)
- **Energía:** datos horarios de generación por fuente, consumo real y previsión del TSO (Red Eléctrica de España)
- **Clima:** temperatura, humedad, viento y nubosidad de las 5 ciudades principales de España
- **Periodo:** enero 2015 – diciembre 2018
- **Variable objetivo:** `total load actual` (MW)

---

## Instalación

```bash
# Extraer archivo .zip

# Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### 1. EDA
Abrir y ejecutar `01_eda.ipynb` para el análisis exploratorio completo.

### 2. Entrenamiento
Abrir y ejecutar `02_model.ipynb` para entrenar la LSTM y generar la carpeta `model/`.

### 3. App Streamlit
```bash
streamlit run app.py
```
Selecciona una fecha y hora del rango 2015–2018 y pulsa **Predecir consumo**.

---

## Modelo

Arquitectura LSTM de dos capas con Dropout:

```
LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
```

- **Ventana temporal:** 24 horas de historial
- **Optimizador:** Adam (lr=0.001)
- **Función de pérdida:** MSE
- **Regularización:** EarlyStopping (patience=10)

---

## Resultados

| Modelo | MAE (MW) |
|---|---|
| LSTM (nuestro) | ~415 MW |
| Previsión TSO (baseline) | ~254 MW |

El error del modelo representa aproximadamente un **1.5% sobre el consumo medio** de ~28.000 MW, resultado competitivo dado que no se dispone de los datos privilegiados que usa el TSO.

---

## Dependencias

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
streamlit
jupyter
```
