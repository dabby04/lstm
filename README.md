# Time Series Modelling with Deep Learning: Forecasting using LSTM

## Project Overview

This project explores deep learning approaches for time series modeling. This project uses an **LSTM model** for weather forecasting on the Jena Climate dataset.

The baselines were reproduced and documented. The improvement task focuses on the LSTM model, where three incremental architectural modifications were made and evaluated.

---

## Repository Structure

```
├── README.md                          # This file — project overview
├── results.md                         # Benchmark tables and experiment outcomes
├── discussion.md                      # Written answers and analysis (Task 4)
│
├── timeseries_weather_forecasting.ipynb # Task 1 — LSTM baseline notebook
│
└── lstm_experiments.ipynb             # Task 2 — LSTM incremental experiments
```

---

## Datasets

### Jena Climate (LSTM Forecasting)
- **Source:** Max Planck Institute for Biogeochemistry
- **Task:** Predict temperature 12 hours into the future given past weather readings
- **Time frame:** January 2009 – December 2016, recorded every 10 minutes
- **Features used:** 7 (pressure, temperature, vapor pressure, humidity, airtight, wind speed, wind direction)
- **Input sequence:** 720 observations (5 days of hourly readings)

---

## Models

### LSTM (Forecasting)
- Single or stacked LSTM layers followed by a Dense output layer
- Trained with mean squared error (MSE) loss
- Uses EarlyStopping and ModelCheckpoint callbacks
- **Metric:** Validation MSE loss (lower is better)

---

## Task 1: Baseline Reproduction

### LSTM (Jena Climate)
The LSTM baseline was successfully reproduced. The model achieved a best validation loss of **0.1317 MSE**, consistent with the expected behavior described in the Keras documentation. Full results and discussion are in [`results.md`](results.md).

Task 1 notebook: [`timeseries_weather_forecasting.ipynb`](timeseries_weather_forecasting.ipynb).

---

## Task 2 — Improvement Task (LSTM)

Three architectural modifications were made to the LSTM baseline, each building on the best configuration from the previous experiment. See [`lstm_experiments.ipynb`](lstm_experiments.ipynb) for the full implementation and [`results.md`](results.md) for the results and discussion.

| Experiment | Change from Previous |
|------------|----------------------|
| Exp 1 | LSTM hidden units doubled: 32 → 64 |
| Exp 2 | Second LSTM layer added (stacked) |
| Exp 3 | Input sequence length halved: 720 → 360 observations |

---

## Results and Discussion

Detailed experiment outcomes are in [`results.md`](results.md), and the written discussion answers are in [`discussion.md`](discussion.md).

---

## References

- Keras LSTM Forecasting Example: https://keras.io/examples/timeseries/timeseries_weather_forecasting/
- Jena Climate Dataset: Max Planck Institute for Biogeochemistry
