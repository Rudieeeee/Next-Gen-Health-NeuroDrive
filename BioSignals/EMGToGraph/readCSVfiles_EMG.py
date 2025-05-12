import csv
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIG ---
csv_file = r'C:\Users\jacob\AAA important stuff\EE jaar 2 (2024-2025)\wheelchair_design\python_plots_EMG\EMGplot1_frontalis(eyebrow).csv'

# --- Data Storage ---
timestamps = []
current_voltages = []
average_voltages = []

# --- Read CSV ---
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            # Convert timestamp from float to datetime (optional)
            timestamp = datetime.fromtimestamp(float(row['Timestamp']))
            current = float(row['Current Voltage (V)'])
            average = float(row['Average Voltage (V)'])

            timestamps.append(timestamp)
            current_voltages.append(current)
            average_voltages.append(average)
        except ValueError:
            print("Skipping invalid row:", row)


start = datetime.strptime("11:57:26", "%H:%M:%S")
# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(timestamps, current_voltages, label='Current Voltage (V)', color='blue')
plt.plot(timestamps, average_voltages, label='Average Voltage (V)', color='green')
plt.xlabel("Time")
plt.ylabel("Voltage (V)")
plt.title("EMG Signal from CSV")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
