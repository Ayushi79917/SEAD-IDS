"""
STEP 3: Open-Set Clustering.

Idea: instead of forcing every traffic sample into a known cluster,
we measure the distance of a sample from the nearest known cluster
center. If the distance is beyond a threshold, we flag it as
"unknown / possible zero-day" instead of forcing a label.

TODO for you:
- Tune DISTANCE_THRESHOLD by checking known-attack distances first
  (it should be a value most known samples fall under)
- Try DBSCAN as an alternative to KMeans + threshold
"""

import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from importlib import import_module
Autoencoder = import_module("02_feature_extract").Autoencoder

ENCODING_DIM = 32
N_CLUSTERS = 10          # roughly number of known attack types + benign
DISTANCE_THRESHOLD = None  # set this after inspecting distance distribution


def get_encoded_features(model, X):
    model.eval()
    with torch.no_grad():
        _, z = model(torch.tensor(X.astype("float32")))
    return z.numpy()


def main():
    known_df = pd.read_csv("../data/processed_known.csv")
    zero_day_df = pd.read_csv("../data/processed_zero_day.csv")
    feature_cols = [c for c in known_df.columns if c not in ["Label", "Label_enc"]]

    model = Autoencoder(input_dim=len(feature_cols), encoding_dim=ENCODING_DIM)
    model.load_state_dict(torch.load("../results/autoencoder.pt"))

    X_known = get_encoded_features(model, known_df[feature_cols].values)
    X_zero_day = get_encoded_features(model, zero_day_df[feature_cols].values)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(X_known)

    # distance of each known sample to its nearest cluster center
    _, known_dists = pairwise_distances_argmin_min(X_known, kmeans.cluster_centers_)
    print("Known sample distance stats: mean=%.3f, 95th pct=%.3f" %
          (known_dists.mean(), np.percentile(known_dists, 95)))

    threshold = DISTANCE_THRESHOLD or np.percentile(known_dists, 95)
    print(f"Using distance threshold: {threshold:.3f}")

    # Save for Member 3's dashboard to load directly (no refitting needed)
    joblib.dump(kmeans, "../results/kmeans_model.pkl")
    joblib.dump(threshold, "../results/distance_threshold.pkl")
    print("Saved kmeans_model.pkl and distance_threshold.pkl")

    # now check the held-out zero-day attack
    _, zd_dists = pairwise_distances_argmin_min(X_zero_day, kmeans.cluster_centers_)
    flagged_unknown = (zd_dists > threshold).sum()
    print(f"Zero-day samples flagged as UNKNOWN: {flagged_unknown} / {len(zd_dists)} "
          f"({100*flagged_unknown/len(zd_dists):.1f}%)")


if __name__ == "__main__":
    main()
