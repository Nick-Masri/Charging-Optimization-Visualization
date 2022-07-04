import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

#%%
reserve = pd.read_excel("data/sensitivityAnalysis/busLevelS3.xlsx", header=None)
reserveD = pd.read_excel("data/sensitivityAnalysis/busLevelD3.xlsx", header=None)
#%%
summary = reserve.iloc[0:4,:]
summaryD = reserveD.iloc[0:4, :]

#%%
plt.clf()
y = summary.iloc[2, 4:18].values
x = summary.iloc[0, 4:18].values

x1 = summaryD.iloc[0, 4:18].values
y1 = summaryD.iloc[2, 4:18].values
plt.figure(figsize=(5,3))
plt.plot(x1, y1, marker="o",label="Dynamic")
plt.plot(x, y, marker="o", label="Static")
plt.scatter([0, 0.1, 0.9, 1], [0, 0, 0, 0], marker="x", label="Infeasible for both", color='red')
plt.ylabel("Weekly Electricity Cost ($)", fontsize=14, loc='top')
plt.grid(b=True, which='both', linewidth=1, color="grey", linestyle=':')
plt.legend()
# plt.ylim(0, 600)
plt.xticks(ticks=[i*.1 for i in range (0,11)])

plt.xlabel("Bus Starting and End SOC (% of 675kWh)", fontsize=14)
plt.tight_layout()
plt.savefig("buslvl.eps", format='eps')

plt.show()
