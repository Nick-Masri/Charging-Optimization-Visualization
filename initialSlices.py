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
SS = pd.read_excel("data/3dbusreserve/3dbusreserveSunnySummer.xlsx", header=None)

#%%
# SW = pd.read_excel()
#%%
reserve_sumSW = SS.iloc[0:4, 0:121]

#%%
reserve_sumSW = reserve_sumSW.T

#%%
reserve_sumSW = reserve_sumSW[reserve_sumSW[2] != 0]

#%%
fig, axs = plt.subplots(3, 3, sharey=True)
fig.text(0.01, 0.5, 'Weekly Operational Costs ($)', va='center', rotation='vertical')
fig.suptitle('Bus Final Vs. Objective at Different Bus Initial Levels')
fig.text(0.5, 0.04, 'Bus Final', ha='center')


# axs[0, 0].plot(reserve_sumSW2[1], reserve_sumSW2[3])
for i in range(0, 3):
    for j in range(0, 3):
        if i != 2 or j < 1:
            print(i,j)
            print(round(0.2+0.3*i+0.1*j,2))
            reserveInitial = reserve_sumSW[reserve_sumSW[0] == round(0.2+0.3*i+0.1*j,2)]
            m, b = np.polyfit(reserveInitial[1], reserveInitial[3], 1)
            axs[i, j].text(0.6, 0, 'y={}x+{}'.format(round(m), round(b)), horizontalalignment='center',
                           verticalalignment='center', transform=axs[i, j].transAxes, color='red')
            axs[i, j].set_title("Initial: {}".format(round(0.2+0.3*i+0.1*j,2)))
            axs[i, j].plot(reserveInitial[1], reserveInitial[3])


axs[2,1].set_visible(False)
axs[2,2].set_visible(False)

fig.tight_layout(pad=1.5)
fig.show()
fig.savefig("BusFinalSlicesSunnySummer.png")

