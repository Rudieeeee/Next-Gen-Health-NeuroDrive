import serial
import csv
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIG ---
serial_port = 'COM5'       # Replace with your actual port (e.g., '/dev/ttyUSB0')
baud_rate = 9600
output_file = r'C:\Users\jacob\AAA important stuff\EE jaar 2 (2024-2025)\wheelchair_design\python_plots_EMG\EMGplot_jaw_6.csv'
max_points = 100

# --- Initialize Serial ---
ser = serial.Serial(serial_port, baud_rate)
time.sleep(2)  # Let Arduino reset

# --- CSV Setup ---
csv_file = open(output_file, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Current Voltage (V)", "Average Voltage (V)", "Detected"])

# --- Data Buffers ---
currents = deque([0]*max_points, maxlen=max_points)
averages = deque([0]*max_points, maxlen=max_points)
detected = deque([0]*max_points, maxlen=max_points)
timestamps = deque([0]*max_points, maxlen=max_points)

# --- Plot Setup ---
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='Current Voltage (V)', color='blue')
line2, = ax.plot([], [], label='Average Voltage (V)', color='green')
line3, = ax.plot([], [], label='EMG Detected (1 OR 0)', color='red')

ax.set_ylim(0, 5)
ax.set_xlim(0, max_points)
ax.set_title("Live EMG Voltage Plot")
ax.set_xlabel("Samples")
ax.set_ylabel("Voltage (V)")
ax.legend(loc='upper right')

def update(frame):
    line = ser.readline().decode('utf-8').strip()
    parts = line.split()
    if len(parts) < 2:
        return

    try:
        current = float(parts[0])
        avg = float(parts[1])
        det = float(parts[2])
        timestamp = time.time()

        # Save to CSV
        csv_writer.writerow([timestamp, current, avg, det])

        # Update buffers
        currents.append(current)
        averages.append(avg)
        detected.append(det)

        # Update plot lines
        line1.set_data(range(len(currents)), list(currents))
        line2.set_data(range(len(averages)), list(averages))
        line3.set_data(range(len(detected)), list(detected))

    except ValueError:
        print("Malformed line:", line)

    return line1, line2, line3

ani = animation.FuncAnimation(fig, update, interval=30)

plt.tight_layout()
plt.show()

# Cleanup
csv_file.close()
ser.close()