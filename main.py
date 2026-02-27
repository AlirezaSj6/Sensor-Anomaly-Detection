import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def main(data_path = "data/sensor_data.csv", out_dir = "output", contamination = 0.02, random_state = 42):
    
    # 1) Load data
    df = pd.read_csv(data_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    # 2) Features
    features = ["current", "voltage", "power"]
    X = df[features].copy()

    # 3) Scale (recommended for IsolationForest)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) Train IsolationForest
    model = IsolationForest(n_estimators = 200, contamination = contamination, random_state = random_state)
    model.fit(X_scaled)

    # 5) Predict anomalies
    # model.predict -> (1 normal, -1 anomaly)
    pred = model.predict(X_scaled)
    df["anomaly"] = (pred == -1).astype(int)
    print("Anomaly counts:")
    print(df["anomaly"].value_counts())
    # score_samples: higher = more normal, lower = more abnormal
    df["anomaly_score"] = model.score_samples(X_scaled)

    # 6) Save anomalies
    os.makedirs(out_dir, exist_ok=True)
    anomalies = df[df["anomaly"] == 1].copy()
    anomalies_path = os.path.join(out_dir, "anomalies.csv")
    anomalies.to_csv(anomalies_path, index = False)

    # 7) Plot (power over time + anomalies)
    plt.figure(figsize = (12, 5))
    plt.plot(df["timestamps"], df["power"], label = "Power")
    plt.scatter(anomalies["timestamps"], anomalies["power"], label = "Anomaly", s = 25)
    plt.title("Sensor Power with Detected Anomalies (IsolationForest)")
    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(out_dir, "power_anomalies.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print("Saved anomalies to:", anomalies_path)
    print("Saved plot to:", plot_path)
    print("Total points:", len(df), "| Anomalies:", df["anomaly"].sum())


if __name__ == "__main__":
    main()