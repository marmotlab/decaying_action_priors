import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

df = pd.read_csv('imitation_data_wtw.csv', parse_dates=False)

# points = [
#     [0.0, 1.0, 0.0],   # Point 1 (x, y, z)
#     [1.0, 2.0, -0.5],  # Point 2 (x, y, z)
#     [-1.0, 0.5, 0.3],  # Point 3 (x, y, z)
#     # Add more points here if needed
# ]
points = df[['e1x', 'e1y', 'e1z']].values

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:10, 0], points[:10, 1], points[:10, 2], color='red', s=100)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()