#%%
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

#%%

reserve = pd.read_excel("data/3dReserve/3dReserveSunnySummer.xlsx", header=None)
reserveCloudy = pd.read_excel("data/3dReserve/3dReserveCloudySummer.xlsx", header=None)
reserveWinter = pd.read_excel("data/3dReserve/3dReserveSunnyWinter.xlsx", header=None)
reserveCloudyWinter = pd.read_excel("data/3dReserve/3dReserveCloudyWinter.xlsx", header=None)
#%%
reserve_sumCS = reserveCloudy.iloc[0:4, 0:121]
reserve_sumW = reserveWinter.iloc[0:4, 0:121]
reserve_sum = reserve.iloc[0:4, 0:121]
reserve_sumCW = reserveCloudyWinter.iloc[0:4, 0:121]


#%%
# start with sunny summer then go to cloudy summer
# initial of sunny summer has to match final of cloudy summer i.e. reserve_sumW.iloc[i,0] = reserve_sumCW.iloc[j,1]
# final of sunny summer has to match initial of cloudy summer i.e. reserve_sumW.iloc[i,1] = reserve_sumCW.iloc[j,0]
# such that reserve_sumW.iloc[i,3] + reserve_sumCW.iloc[j,3] is minimum




#%%
reserve_sumCS = reserve_sumCS.T
reserve_sumW = reserve_sumW.T
reserve_sum = reserve_sum.T
reserve_sumCW = reserve_sumCW.T

reserve_sumCS = reserve_sumCS[reserve_sumCS[3] != 0]
reserve_sumW = reserve_sumW[reserve_sumW[3] != 0]
reserve_sum = reserve_sum[reserve_sum[3] != 0]
reserve_sumCW = reserve_sumCW[reserve_sumCW[3] != 0]

#%%
from matplotlib.lines import Line2D
fig = plt.figure()
ax = plt.axes(projection='3d')
summer = plt.get_cmap('autumn')
winter = plt.get_cmap('viridis')
summer = plt.get_cmap('autumn')
winter = plt.get_cmap('winter')
autumn = plt.get_cmap('autumn')
cloudyWinter = plt.get_cmap('summer')
# fig.tight_layout(pad=3)


fig.subplots_adjust(top=.80)
fig.subplots_adjust(bottom=.13)

surf = ax.plot_trisurf(reserve_sum[0], reserve_sum[1], reserve_sum[3], color='green', linewidth=0, cmap=summer, label='Summer')
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d


surf2 = ax.plot_trisurf(reserve_sumW[0], reserve_sumW[1], reserve_sumW[3], linewidth=0,  cmap=winter, label='Winter')
surf2._facecolors2d = surf2._facecolor3d
surf2._edgecolors2d = surf2._edgecolor3d


surf3 = ax.plot_trisurf(reserve_sumCS[0], reserve_sumCS[1], reserve_sumCS[3], linewidth=0,  cmap=autumn, label='Winter')
surf3._facecolors2d = surf3._facecolor3d
surf3._edgecolors2d = surf3._edgecolor3d

surf4 = ax.plot_trisurf(reserve_sumCW[0], reserve_sumCW[1], reserve_sumCW[3], linewidth=0,  cmap=cloudyWinter, label='Winter')
surf4._facecolors2d = surf4._facecolor3d
surf4._edgecolors2d = surf4._edgecolor3d

ax.set_xlabel("Bus Ratio")
ax.set_ylabel("Reserve Ratio")

custom_lines = [
                Line2D([0], [0], color=summer(0.1), lw=4),
                Line2D([0], [0], color=winter(0.2), lw=4)]


ax.legend(custom_lines, ['Winter', 'Summer', 'Cloudy Summer', 'Cloudy Winter'])

ax.set_title("Weekly Electricity Cost Vs. Bus Start & End Levels Vs. Storage Start & End Levels", fontsize=18, wrap=True)
ax.set_zlabel("Weekly Electricity Cost ($)")
# ax.set_zlim(400, 3200)
# ax.legend()
ax.view_init(22, 65)
plt.show()

#%%


from matplotlib.lines import Line2D
fig = plt.figure()
ax = plt.axes(projection='3d')
summer = plt.get_cmap('autumn')
winter = plt.get_cmap('winter')
autumn = plt.get_cmap('autumn')
cloudyWinter = plt.get_cmap('summer')
fig.tight_layout(pad=2)
surf = ax.plot_trisurf(reserve_sum[0], reserve_sum[1], reserve_sum[3], color='green', linewidth=0, cmap=summer, label='Summer')
surf._facecolors2d = surf._facecolor3d
surf._edgecolors2d = surf._edgecolor3d


surf2 = ax.plot_trisurf(reserve_sumW[0], reserve_sumW[1], reserve_sumW[3], linewidth=0,  cmap=winter, label='Winter')
surf2._facecolors2d = surf2._facecolor3d
surf2._edgecolors2d = surf2._edgecolor3d

fig.subplots_adjust(top=.85)
fig.subplots_adjust(bottom=.13)

ax.set_xlabel("Bus Start & End SOC Ratio (%) ")
ax.set_ylabel("Reserve Start & end SOC Ratio (%)")

custom_lines = [Line2D([0], [0], color=winter(0.3), lw=4),
                Line2D([0], [0], color=summer(0.3), lw=4)]


ax.legend(custom_lines, ['Winter', 'Summer', 'Cloudy Summer', 'Cloudy Winter'])

ax.set_title("Weekly Electricity Cost Vs. Bus Start & End Levels Vs. Storage Start & End Levels", fontsize=18, pad=4, wrap=True)
ax.set_zlabel("Weekly Electricity Cost ($)")
# ax.set_zlim(400, 3200)
# ax.legend()
ax.view_init(10, 305)
plt.show()

