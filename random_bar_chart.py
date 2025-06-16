import numpy as np
import matplotlib.pyplot as plt

# Number of bars
num_bars = 10

# Random data for the bar heights
values = np.random.randint(1, 20, size=num_bars)

# X locations for the bars
x_pos = np.arange(num_bars)

# Create the bar chart
plt.bar(x_pos, values, color='skyblue')
plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Random Data Bar Chart')
plt.xticks(x_pos)
plt.tight_layout()

plt.show()
