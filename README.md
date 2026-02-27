# Sensor Anomaly Detection

This project detects anomalies in electrical sensor data using an unsupervised machine learning approach.

## Problem Statement

Electrical systems generate time-series measurements such as current, voltage, and power.  
The objective of this project is to automatically detect abnormal operating conditions in sensor data without labeled examples.

## Methodology

1. Synthetic dataset generation with realistic electrical behavior
2. Injection of anomaly types (spikes, dips, drift, dropout)
3. Feature scaling using StandardScaler
4. Isolation Forest for unsupervised anomaly detection

## Features Used

- Current
- Voltage
- Power

## How to Run

Generate dataset:
python generate_data.py

Run anomaly detection:
python main.py

## Project Structure

sensor-anomaly-detection/
- generate_data.py
- main.py
- data/
- output/

## Model Description

Isolation Forest isolates anomalies instead of modeling normal data.  
Points that are easier to isolate are more likely to be anomalies.

## Output

- anomalies.csv (detected anomalies)
- power_anomalies.png (visualization)

## Technical Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
