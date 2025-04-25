import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as md
import pandas as pd

# Create figure for plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []


# This function is called periodically from FuncAnimation
def animate(i, xs, ys):

    # Read data from file
    data = pd.read_csv("EMGData.csv", header = None, parse_dates = [0])

    # Extract last 20 elements
    data = data.tail(20)

    # Extract rows
    xs = data.iloc[:,0]
    ys = data.iloc[:,1]
    
    # Draw x and y data
    ax.clear()
    ax.plot(xs, ys)

    # Format plot
    xfmt = md.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.ylabel('Signal')

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=500)
plt.show()
