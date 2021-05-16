import matplotlib.pyplot as plt
import numpy as np

# Read data
size = []
time = []
with open("pi_non-blocking_linear.txt") as file:
    for line in file.readlines():
        x, y = line.split(',')
        size.append(int(x.strip()))   
        time.append(float(y.strip()))   

# Plot data
fig, ax = plt.subplots()
ax.plot(size, time)

ax.set(xlabel='Num. processes', ylabel='Time (s)',
       title='Pi non-blocking linar')
#ax.grid()

fig.savefig("pi_non-blocking_linear.png")
plt.show()
