"""
STEP 4: Continual Learning module.

Idea: once a new (previously unknown) attack cluster is confirmed,
the model should update itself using only the new samples, without
retraining from scratch and without forgetting old attacks.

This starter uses the `river` library, which is built for online/
incremental learning (simpler to start with than writing your own
replay-buffer neural network from scratch).

TODO for you:
- Replace this simple online classifier with your deep model + a
  replay buffer (keep a small sample of old attacks and mix them in
  when training on new ones, so the model doesn't forget)
- Log accuracy on OLD attacks before/after each update to prove
  "no catastrophic forgetting" for your paper
"""

import pandas as pd
from river import tree, metrics, stream

known_df = pd.read_csv("../data/processed_known.csv")
zero_day_df = pd.read_csv("../data/processed_zero_day.csv")
feature_cols = [c for c in known_df.columns if c not in ["Label", "Label_enc"]]

model = tree.HoeffdingTreeClassifier()
acc = metrics.Accuracy()

# Phase 1: train on known attacks (simulating initial deployment)
print("Phase 1: training on known attacks...")
for _, row in known_df.iterrows():
    x = {col: row[col] for col in feature_cols}
    y = row["Label_enc"]
    y_pred = model.predict_one(x)
    if y_pred is not None:
        acc.update(y, y_pred)
    model.learn_one(x, y)

print(f"Accuracy on known attacks after initial training: {acc}")

# Phase 2: "zero-day" attack starts appearing -> continually update
print("\nPhase 2: adapting to new (zero-day) attack as it arrives...")
new_label = known_df["Label_enc"].max() + 1  # assign a new class id
acc_new = metrics.Accuracy()

for _, row in zero_day_df.iterrows():
    x = {col: row[col] for col in feature_cols}
    y_pred = model.predict_one(x)
    if y_pred is not None:
        acc_new.update(new_label, y_pred)  # will be low before it learns
    model.learn_one(x, new_label)  # model incrementally learns the new attack

print(f"Accuracy on zero-day attack during continual adaptation: {acc_new}")
print("\nNext: re-check accuracy on OLD known attacks to confirm no forgetting.")
