# Results and Discussion

## Overview

This file summarizes the benchmark results for the LSTM forecasting experiments. The LSTM model was first run at baseline, then modified incrementally with three architectural changes, each building on the previous configuration.

For the Transformer baseline reproduction findings, refer to the Task 1 section in `README.md`.

---

## LSTM Forecasting

### Baseline Configuration

| Parameter | Value |
|-----------|-------|
| LSTM layers | 1 |
| LSTM units | 32 |
| past (sequence length) | 720 observations (5 days) |
| future (prediction horizon) | 72 observations (12 hours) |
| learning_rate | 0.001 |
| batch_size | 256 |
| epochs (max) | 10 |

**Baseline best val_loss: 0.1317 MSE**

---

### Baseline Loss Curve

![Baseline Loss Curve](images/baseline_loss.png)

---

### Experiment Results

| Experiment | LSTM Layers | LSTM Units | past | Best Val Loss (MSE) | vs Baseline |
|------------|-------------|------------|------|---------------------|-------------|
| Baseline   | 1           | 32         | 720  | 0.1317              | —           |
| Exp 1      | 1           | 64         | 720  | 0.1473              | +11.8%      |
| Exp 2      | 2           | 64         | 720  | 0.1321              | +0.3%       |
| Exp 3      | 2           | 64         | 360  | 0.1349              | +2.4%       |

*Note: A positive "vs Baseline" percentage means the val_loss is higher than baseline (worse). Lower MSE is better.*

---

### Discussion

**Experiment 1: Double LSTM Hidden Units (32 → 64)**

Doubling the hidden units increased the model's parameter count from ~4K to ~18K but resulted in a worse `val_loss` of 0.1473. The model showed clear signs of overfitting. The best `val_loss` was achieved at Epoch 1 and never improved afterward, while training loss continued to decrease. More capacity alone did not help generalization within 10 epochs.

![Experiment 1 Loss Curve](images/exp1_loss.png)

**Experiment 2: Stack a Second LSTM Layer**

Adding a second LSTM layer significantly improved training behaviour. Unlike Experiment 1, the `val_loss` improved consistently across all 10 epochs, reaching a best of 0.1321 nearly matching the baseline. The stacked architecture helped the model generalize better, addressing the overfitting observed in Experiment 1. This was the strongest performing experiment among the three modifications.

![Experiment 2 Loss Curve](images/exp2_loss.png)

**Experiment 3: Reduce Input Sequence Length (720 → 360)**

Reducing the input sequence length from 5 days to 2.5 days of history resulted in a val_loss of 0.1349, only slightly worse than Experiment 2. This is a notable finding, the two-layer model retained most of its predictive power with half the historical context, suggesting that the most recent 2.5 days of weather data contains most of the relevant information for 12-hour temperature forecasting.

![Experiment 3 Loss Curve](images/exp3_loss.png)

---

### Cross-Experiment Comparison

From a ranking perspective (lower MSE is better), the models perform as:

1. Baseline (0.1317)
2. Experiment 2 (0.1321)
3. Experiment 3 (0.1349)
4. Experiment 1 (0.1473)

Relative to the baseline, Experiment 2 is effectively tied in performance (+0.3%), which suggests deeper sequence modeling can work when training remains stable. Experiment 3 trades a small accuracy drop (+2.4%) for a shorter input window, which may still be attractive when reducing memory/computation is a priority.

Experiment 1 is the clear outlier (+11.8%), indicating that increasing hidden units alone made optimization/generalization worse under the same training budget. Overall, depth (Exp 2) was more useful than width (Exp 1), while input reduction (Exp 3) offered a practical efficiency trade-off with limited performance cost.

---

### Overall Finding

None of the experiments improved on the baseline `val_loss` of 0.1317. The original single-layer 32-unit LSTM was the best performing configuration across all runs. This demonstrates that a simpler model is not always worse, and that increasing model complexity does not guarantee better generalization, particularly when the number of training epochs is limited.
