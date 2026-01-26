# DRAWS A GRAPH BASED ON BENCHMARK RESULTS ON OKKOS MACBOOK AIR M3 2024

import matplotlib.pyplot as plt
import numpy as np

# Data
scenarios = ['Same Images', 'Different Images']
standard_times = [3.2497, 3.2097]
powerbank_times = [3.1718, 3.2012]

x = np.arange(len(scenarios))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, standard_times, width, label='Standard Battery', color='#FF6B6B')
rects2 = ax.bar(x + width/2, powerbank_times, width, label='With Powerbank', color='#4ECDC4')

# Labels
ax.set_ylabel('Total Time (Seconds)')
ax.set_title('Inference Speed: Standard vs. Powerbank\nMacbook Air 13-inch, M3, 2024')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylim(3.0, 3.4) # Zoom in to see the difference
ax.legend()

# Add value labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()