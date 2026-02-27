import numpy as np
import pandas as pd
import os

def generate_sensor_dataset(start = "2024-01-01 00:00:00", days = 3, freq = "1min", seed = 42, out_path = "data/sensor_data.csv"):
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(start = start, periods = days * 24 * 60, freq = freq)
    n = len(timestamps)

    minutes_in_day = 24 * 60
    t = np.arange(n)
    daily_phase = 2 * np.pi * (t % minutes_in_day) / minutes_in_day

    voltage = 220 + 3*np.sin(daily_phase) + rng.normal(0, 1.2, n)
    current = 5 + 1.0*np.sin(daily_phase - 0.7) + rng.normal(0, 0.25, n)
    power = voltage * current + rng.normal(0, 5.0, n)

    df = pd.DataFrame({"timestamps": timestamps, "current": current, "voltage": voltage, "power": power})

    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)

    return df

if __name__ == "__main__":
    generate_sensor_dataset()
    print("Dataset generated.")