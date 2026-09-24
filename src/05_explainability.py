"""
STEP 5: Explainability with SHAP.

Idea: for traffic flagged as "unknown / zero-day", show WHICH
features (e.g. packet size, flow duration, flags) pushed the model
toward that decision. This is what makes your project analyst-
friendly and is a strong point for the patent write-up.

TODO for you:
- Swap RandomForestClassifier for your actual deep model once ready
  (SHAP has a DeepExplainer for PyTorch models)
- Save the summary plot image for your paper's results section
"""

import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

known_df = pd.read_csv("../data/processed_known.csv")
feature_cols = [c for c in known_df.columns if c not in ["Label", "Label_enc"]]

X = known_df[feature_cols]
y = known_df["Label_enc"]

# quick baseline classifier just to demonstrate the SHAP pipeline
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X, y)
joblib.dump(clf, "../results/classifier.pkl")
print("Saved classifier to results/classifier.pkl")

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X.sample(500, random_state=42))  # sample for speed

shap.summary_plot(shap_values, X.sample(500, random_state=42), show=False)
plt.tight_layout()
plt.savefig("../results/shap_summary.png", dpi=150)
print("Saved SHAP summary plot to results/shap_summary.png")
