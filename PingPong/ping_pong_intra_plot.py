import matplotlib.pyplot as plt
import numpy as np

# Read data
size = []
time = []
with open("ping_pong_intranode.txt") as file:
    for line in file.readlines():
        x, y = line.split('\t')
        size.append(int(x.strip()))   
        time.append(float(y.strip()))   

# Fit data
p = np.polyfit(size, time, 1)
latency = p[1]/1e-6
print(f"Latency = {latency} us")
bandwidth = 1/p[0]/1e9
print(f"Bandwidth = {bandwidth} GB/s")

# Plot data
fig, ax = plt.subplots()
ax.plot(size, time, 'bo')

x_vals = np.array(ax.get_xlim())
y_vals = p[1] + p[0] * x_vals 
ax.plot(x_vals, y_vals, '--')

ax.set(xlabel='Message Size (B)', ylabel='Time (s)',
       title='Intra-node latency')
ax.grid()

fig.savefig("ping_pong_intranode.png")
plt.show()
