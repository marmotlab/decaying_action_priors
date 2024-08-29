import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('imitation_data_0.2_x.csv', parse_dates=False)

fig, axes = plt.subplots(nrows=3, ncols=6)

df.base1.plot(color='g',lw=1.3, legend=True, ax=axes[0,0])
df.shoulder1.plot(color='r',lw=1.3, legend=True, ax=axes[0,1])
df.elbow1.plot(color='b',lw=1.3, legend=True, ax=axes[0,2])

# df.base4.plot(color='r',lw=1, legend=True, ax=axes[0,0])
# df.shoulder4.plot(color='b',lw=1, legend=True, ax=axes[0,1])
# df.elbow4.plot(color='g',lw=1, legend=True, ax=axes[0,2])

df.base2.plot(color='g',lw=1.3, legend=True, ax=axes[0,3])
df.shoulder2.plot(color='r',lw=1.3, legend=True, ax=axes[0,4])
df.elbow2.plot(color='b',lw=1.3, legend=True, ax=axes[0,5])

df.base4.plot(color='g',lw=1.3, legend=True, ax=axes[1,0])
df.shoulder4.plot(color='r',lw=1.3, legend=True, ax=axes[1,1])
df.elbow4.plot(color='b',lw=1.3, legend=True, ax=axes[1,2])

df.base3.plot(color='g',lw=1.3, legend=True, ax=axes[1,3])
df.shoulder3.plot(color='r',lw=1.3, legend=True, ax=axes[1,4])
df.elbow3.plot(color='b',lw=1.3, legend=True, ax=axes[1,5])


# df.base5.plot(color='g',lw=1.3, legend=True, ax=axes[2,0])
# df.shoulder5.plot(color='r',lw=1.3, legend=True, ax=axes[2,1])
# df.elbow5.plot(color='b',lw=1.3, legend=True, ax=axes[2,2])

# df.base6.plot(color='g',lw=1.3, legend=True, ax=axes[2,3])
# df.shoulder6.plot(color='r',lw=1.3, legend=True, ax=axes[2,4])
# df.elbow6.plot(color='b',lw=1.3, legend=True, ax=axes[2,5])

plt.show()