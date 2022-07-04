#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

sns.set_theme()


#%%
SS = pd.read_excel("data/3dStorageSizeD.xlsx", header=None)

#%%

reserve_sumSW = SS.iloc[0:4, 0:121]


#%%
reserve_sumSW = reserve_sumSW.T

#%%
reserve_sumSW = reserve_sumSW[reserve_sumSW[2] != 0]

#%%
reserve_sumSW2 = reserve_sumSW[reserve_sumSW[1] == 0.2]
reserve_sumSW3 = reserve_sumSW[reserve_sumSW[1] == 0.3]
reserve_sumSW4 = reserve_sumSW[reserve_sumSW[1] == 0.4]
reserve_sumSW5 = reserve_sumSW[reserve_sumSW[1] == 0.5]
reserve_sumSW6 = reserve_sumSW[reserve_sumSW[1] == 0.6]
reserve_sumSW7 = reserve_sumSW[reserve_sumSW[1] == 0.7]
reserve_sumSW8 = reserve_sumSW[reserve_sumSW[1] == 0.8]
reserve_sumSW9 = reserve_sumSW[reserve_sumSW[1] == 0.9]
reserve_sumSW1 = reserve_sumSW[reserve_sumSW[1] == 1]
#%%

plt.clf()
sns.lineplot(data=reserve_sumSWnill2, x=0, y=3)
plt.title("Bus Initial Vs. Weekly Operational Cost at 0.2 bus final")
plt.xlabel("Bus Initial")
plt.ylabel("Weekly Operational Cost ($)")
plt.show()
# plt.plot(reserve_sumSWnill2[0], reserve_sumSWnill2[3])
# plt.show()
#%%
fig, axs = plt.subplots(3, 3, sharey=True)
fig.text(0.01, 0.5, 'Weekly Operational Costs ($)', va='center', rotation='vertical')
fig.suptitle('Bus Final Vs. Objective at Different Bus Initial Levels')
fig.text(0.5, 0.04, 'Bus Final', ha='center')

axs[0, 0].plot(reserve_sumSW2[0], reserve_sumSW2[3])
axs[0,0].set_title("Final: 0.2")

axs[0, 1].plot(reserve_sumSW3[0], reserve_sumSW3[3])
axs[0,1].set_title("Final: 0.3")

axs[0, 2].plot(reserve_sumSW4[0], reserve_sumSW4[3])
axs[0,2].set_title("Final: 0.4")

axs[1, 0].plot(reserve_sumSW5[0], reserve_sumSW5[3])
axs[1,0].set_title("Final: 0.5")

axs[1, 1].plot(reserve_sumSW6[0], reserve_sumSW6[3])
axs[1,1].set_title("Final: 0.6")

axs[1, 2].plot(reserve_sumSW7[0], reserve_sumSW7[3])
axs[1,2].set_title("Final: 0.7")

axs[2, 0].plot(reserve_sumSW8[0], reserve_sumSW8[3])
axs[2,0].set_title("Final: 0.8")

axs[2,1].set_visible(False)
axs[2,2].set_visible(False)

fig.tight_layout(pad=1.5)
fig.show()
# fig.savefig("BusInitialSlicesSunnySummer.png")

#%%

#%%

from matplotlib.lines import Line2D
fig = plt.figure()
ax = plt.axes(projection='3d')

summer = plt.get_cmap('viridis')
winter = plt.get_cmap('winter')
autumn = plt.get_cmap('autumn')
cloudyWinter = plt.get_cmap('summer')
fig.tight_layout(pad=2)
surf = ax.scatter(reserve_sumSW[0], reserve_sumSW[1], reserve_sumSW[3], color='green', linewidth=0, cmap=summer, label='Summer')
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d

fig.subplots_adjust(top=.85)
fig.subplots_adjust(bottom=.13)

ax.set_xlabel("Solar Size")
ax.set_ylabel("Solar Power")


ax.set_title("Solar Size Vs. Power Vs. Operational Cost", fontsize=18, pad=4, wrap=True)
ax.set_zlabel("Weekly Electricity Cost ($)")
# ax.set_zlim(400, 3200)
# ax.legend()
# ax.view_init(20, 200)
plt.savefig("3dsolarsizepower.png")
plt.show()

