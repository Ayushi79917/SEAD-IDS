"""
SEAD-IDS Live Detection Dashboard (Streamlit)

Run with: streamlit run dashboard_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from river import tree

import sys
sys.path.append("./src")
from importlib import import_module
Autoencoder = import_module("02_feature_extract").Autoencoder

RESULTS = "results"
ENCODING_DIM = 64

st.set_page_config(page_title="SEAD-IDS Live Dashboard", layout="wide")
st.title("🛡️ SEAD-IDS — Live Intrusion Detection Dashboard")
st.caption("Continual Learning + Open-Set Clustering for Zero-Day Attack Detection")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(f"{RESULTS}/scaler.pkl")
    feature_cols = joblib.load(f"{RESULTS}/feature_cols.pkl")
    clf = joblib.load(f"{RESULTS}/classifier.pkl")
    label_encoder = joblib.load(f"{RESULTS}/label_encoder.pkl")

    ae = Autoencoder(input_dim=len(feature_cols), encoding_dim=ENCODING_DIM)
    ae.load_state_dict(torch.load(f"{RESULTS}/autoencoder.pt", map_location="cpu"))
    ae.eval()
    return scaler, feature_cols, clf, label_encoder, ae

if "continual_model" not in st.session_state:
    st.session_state.continual_model = tree.HoeffdingTreeClassifier()
if "known_centers" not in st.session_state:
    st.session_state.known_centers = None
if "learned_labels" not in st.session_state:
    st.session_state.learned_labels = {}

scaler, feature_cols, clf, label_encoder, ae = load_artifacts()

if st.session_state.known_centers is None:
    st.session_state.known_centers = joblib.load(f"{RESULTS}/kmeans_model.pkl")
    st.session_state.threshold = joblib.load(f"{RESULTS}/distance_threshold.pkl")

kmeans = st.session_state.known_centers
threshold = st.session_state.threshold

uploaded = st.file_uploader("Upload network traffic CSV to analyze", type=["csv"])

if uploaded:
    raw_df = pd.read_csv(uploaded, low_memory=False)
    raw_df.columns = raw_df.columns.str.strip()

    missing = [c for c in feature_cols if c not in raw_df.columns]
    if missing:
        st.error(f"Uploaded file is missing expected columns: {missing[:5]}...")
    else:
        X_scaled = scaler.transform(raw_df[feature_cols])
        with torch.no_grad():
            _, z = ae(torch.tensor(X_scaled.astype("float32")))
        z_np = z.numpy()

        _, dists = pairwise_distances_argmin_min(z_np, kmeans.cluster_centers_)
        is_unknown = dists > threshold

        preds = clf.predict(X_scaled)
        pred_labels = label_encoder.inverse_transform(preds)

        result_df = raw_df.copy()
        result_df["Distance_Score"] = dists
        result_df["Prediction"] = np.where(
            is_unknown, "⚠️ Unknown / Possible Zero-Day", pred_labels
        )

        n_unknown = is_unknown.sum()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(result_df))
        col2.metric("Flagged as Zero-Day", int(n_unknown))
        col3.metric("Known Attacks Detected", len(result_df) - int(n_unknown))

        context_cols = [c for c in
                         ["Source IP", "Src IP", "Destination IP", "Dst IP",
                          "Source Port", "Src Port", "Destination Port", "Dst Port",
                          "Protocol", "Timestamp"]
                         if c in result_df.columns]

        display_cols = context_cols + ["Prediction", "Distance_Score"]
        st.dataframe(result_df[display_cols], use_container_width=True, height=350)

        if not context_cols:
            st.caption("ℹ️ No IP/port columns found in this file — showing feature scores only. Upload a raw CIC-IDS2017-style CSV (with Source IP / Destination IP columns) to see network context alongside predictions.")

        st.subheader("📊 Detection Summary")
        chart_col1, chart_col2 = st.columns(2)
        counts = result_df["Prediction"].value_counts()
        with chart_col1:
            st.write("**Traffic by category**")
            st.bar_chart(counts)

        with chart_col2:
            st.write("**Safe vs Suspicious split**")
            safe_vs_flag = pd.Series({
                "Normal/Known": int((~is_unknown).sum()),
                "Zero-Day Flagged": int(is_unknown.sum()),
            })
            fig, ax = plt.subplots()
            ax.pie(safe_vs_flag, labels=safe_vs_flag.index, autopct="%1.1f%%",
                   colors=["#8FBF9F", "#E8A0A0"])
            ax.axis("equal")
            st.pyplot(fig)

        st.subheader("🔍 Why was this flagged? (SHAP explanation)")
        row_idx = st.number_input("Row number to explain", min_value=0,
                                   max_value=len(result_df) - 1, value=0)
        if st.button("Explain this row"):
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(pd.DataFrame([X_scaled[row_idx]], columns=feature_cols))
            fig, ax = plt.subplots()
            shap.summary_plot(shap_vals, pd.DataFrame([X_scaled[row_idx]], columns=feature_cols),
                               show=False, plot_type="bar")
            st.pyplot(fig)

        st.subheader("🧠 Continual Learning — teach the model a new attack")
        unknown_rows = result_df[is_unknown]
        if len(unknown_rows) > 0:
            new_attack_name = st.text_input("Name this new attack type (e.g. 'BotnetV2')")
            if st.button("Learn this attack now") and new_attack_name:
                for i in unknown_rows.index:
                    x = {col: raw_df.loc[i, col] for col in feature_cols}
                    st.session_state.continual_model.learn_one(x, new_attack_name)
                st.session_state.learned_labels[new_attack_name] = \
                    st.session_state.learned_labels.get(new_attack_name, 0) + len(unknown_rows)
                st.success(f"Model updated! Learned {len(unknown_rows)} samples as '{new_attack_name}'. Old attack knowledge is NOT overwritten (incremental update).")
        else:
            st.write("No unknown/zero-day rows in this file to learn from.")

        if st.session_state.learned_labels:
            st.write("**Attacks learned so far this session:**")
            st.json(st.session_state.learned_labels)
else:
    st.info("⬆️ Upload a CSV file to start detection.")
