import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('imitation_data_wtw.csv', parse_dates=False)

# fig, axes = plt.subplots(nrows=4, ncols=3)
fig, axes = plt.subplots(nrows=3, ncols=1)

# df.base1.plot(color='g',lw=1.3, legend=True, ax=axes[0,0])
# df.shoulder1.plot(color='r',lw=1.3, legend=True, ax=axes[0,1])
# df.elbow1.plot(color='b',lw=1.3, legend=True, ax=axes[0,2])

# df.base2.plot(color='g',lw=1.3, legend=True, ax=axes[1,0])
# df.shoulder2.plot(color='r',lw=1.3, legend=True, ax=axes[1,1])
# df.elbow2.plot(color='b',lw=1.3, legend=True, ax=axes[1,2])

# df.base4.plot(color='g',lw=1.3, legend=True, ax=axes[3,0])
# df.shoulder4.plot(color='r',lw=1.3, legend=True, ax=axes[3,1])
# df.elbow4.plot(color='b',lw=1.3, legend=True, ax=axes[3,2])

# df.base3.plot(color='g',lw=1.3, legend=True, ax=axes[2,0])
# df.shoulder3.plot(color='r',lw=1.3, legend=True, ax=axes[2,1])
# df.elbow3.plot(color='b',lw=1.3, legend=True, ax=axes[2,2])

# df.e1x.plot(color='g',lw=1.3, legend=True, ax=axes[0])
# df.e1y.plot(color='r',lw=1.3, legend=True, ax=axes[1])
# df.e1z.plot(color='b',lw=1.3, legend=True, ax=axes[2])

# df.e2x.plot(color='r',lw=1.3, legend=True, ax=axes[0])
# df.e2y.plot(color='b',lw=1.3, legend=True, ax=axes[1])
# df.e2z.plot(color='g',lw=1.3, legend=True, ax=axes[2])

# df.e3x.plot(lw=1.3, legend=True, ax=axes[0])
# df.e3y.plot(lw=1.3, legend=True, ax=axes[1])
# df.e3z.plot(lw=1.3, legend=True, ax=axes[2])

# df.e4x.plot(lw=1.3, legend=True, ax=axes[0])
# df.e4y.plot(lw=1.3, legend=True, ax=axes[1])
# df.e4z.plot(lw=1.3, legend=True, ax=axes[2])

df.e1x_wf.plot(color='g',lw=1.3, legend=True, ax=axes[0])
df.e1y_wf.plot(color='r',lw=1.3, legend=True, ax=axes[1])
df.e1z_wf.plot(color='b',lw=1.3, legend=True, ax=axes[2])

df.e2x_wf.plot(color='r',lw=1.3, legend=True, ax=axes[0])
df.e2y_wf.plot(color='b',lw=1.3, legend=True, ax=axes[1])
df.e2z_wf.plot(color='g',lw=1.3, legend=True, ax=axes[2])

df.e3x_wf.plot(lw=1.3, legend=True, ax=axes[0])
df.e3y_wf.plot(lw=1.3, legend=True, ax=axes[1])
df.e3z_wf.plot(lw=1.3, legend=True, ax=axes[2])

df.e4x_wf.plot(lw=1.3, legend=True, ax=axes[0])
df.e4y_wf.plot(lw=1.3, legend=True, ax=axes[1])
df.e4z_wf.plot(lw=1.3, legend=True, ax=axes[2])


# df.e2x.plot(color='g',lw=1.3, legend=True, ax=axes[1,0])
# df.e2y.plot(color='r',lw=1.3, legend=True, ax=axes[1,1])
# df.e2z.plot(color='b',lw=1.3, legend=True, ax=axes[1,2])

# df.e3x.plot(color='g',lw=1.3, legend=True, ax=axes[2,0])
# df.e3y.plot(color='r',lw=1.3, legend=True, ax=axes[2,1])
# df.e3z.plot(color='b',lw=1.3, legend=True, ax=axes[2,2])

# df.e4x.plot(color='g',lw=1.3, legend=True, ax=axes[3,0])
# df.e4y.plot(color='r',lw=1.3, legend=True, ax=axes[3,1])
# df.e4z.plot(color='b',lw=1.3, legend=True, ax=axes[3,2])

# df.e1x.plot(color='g',lw=1.3, legend=True, ax=axes[2,0])
# df.shoulder3.plot(color='r',lw=1.3, legend=True, ax=axes[2,1])
# df.elbow3.plot(color='b',lw=1.3, legend=True, ax=axes[2,2])


plt.show()