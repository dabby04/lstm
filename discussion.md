# Discussion

### Which model did you find easier to understand and why?

Both baselines were reproduced as part of Task 1, which provided direct context for comparing the two approaches. I considered evaluating the Transformer classification approach further, but for this project, LSTM forecasting was easier to execute end-to-end because the setup matched the forecasting objective more directly, training behavior was easier to interpret, and results were more straightforward to compare across experiments. The Transformer setup was still useful to review, but adapting a classification-oriented

The Transformer setup was still useful to review, but adapting a classification-oriented architecture to this forecasting task added extra complexity that made iteration and debugging harder. On the other hand, the LSTM pipeline was more practical for this dataset and task requirements.

### What improvement did you try, and what did you learn from it?

Three incremental architectural modifications were made to the LSTM model: doubling the hidden units from 32 to 64, stacking a second LSTM layer, and halving the input sequence length from 720 to 360 observations. The goal was to progressively build a more capable model.

The most surprising finding was that none of the modifications improved on the baseline `val_loss` of 0.1317. Adding more units, stacking layers, and reducing sequence length all seemed like reasonable improvements, but the original simple single-layer 32-unit LSTM was already the best performing configuration. This showed that in deep learning, a simpler model is not always worse and that increasing complexity without enough training time or data can actually hurt performance.
