"""
STEP 2: Train a simple autoencoder to learn compact features from
network traffic. These compressed features are used later for
clustering and continual learning.

TODO for you:
- Once this runs, check reconstruction loss goes down over epochs
- Try changing ENCODING_DIM (16, 32, 64) and see what works best
"""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split

ENCODING_DIM = 32
EPOCHS = 20
BATCH_SIZE = 256


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


def main():
    df = pd.read_csv("../data/processed_known.csv")
    feature_cols = [c for c in df.columns if c not in ["Label", "Label_enc"]]
    X = df[feature_cols].values.astype("float32")

    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)
    X_train_t = torch.tensor(X_train)
    X_val_t = torch.tensor(X_val)

    model = Autoencoder(input_dim=X.shape[1], encoding_dim=ENCODING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train_t.size(0))
        total_loss = 0
        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = permutation[i:i + BATCH_SIZE]
            batch = X_train_t[idx]
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_recon, _ = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()
        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {total_loss:.4f} - val_loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "../results/autoencoder.pt")
    print("Saved model to results/autoencoder.pt")


if __name__ == "__main__":
    main()
