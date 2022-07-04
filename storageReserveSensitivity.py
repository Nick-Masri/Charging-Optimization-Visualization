import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

#%%
reserveD = pd.read_excel("data/sensitivityAnalysis/storageLevelD3.xlsx", header=None)
reserve = pd.read_excel("data/sensitivityAnalysis/storageLevelS3.xlsx", header=None)
#%%
summary = reserve.iloc[0:4,:]
summaryD = reserveD.iloc[0:4, :]

#%%
y = summary.iloc[2, 4:].values
x = summary.iloc[0, 4:].values
plt.clf()
x1 = summaryD.iloc[0, 4:].values

y1 = summaryD.iloc[2, 4:].values
plt.plot(x1, y1, marker="o",label="Dynamic")
plt.plot(x, y, marker="o", label="Static")
plt.scatter([0, 0.05, 0.1, 0.15], [0, 0, 0, 0], marker="x", label="Infeasible for both", color='red')
plt.ylabel("Weekly Electricity Cost ($)", fontsize=14)
plt.grid(b=True, which='both', linewidth=1, color="grey", linestyle=':')
plt.legend()
plt.tight_layout(pad=2)
# plt.ylim(0, 600)
plt.xticks(ticks=[i*.1 for i in range (0,11)])
plt.xlabel("Storage Starting and End SOC (% of 1000kWh)", fontsize=14)
# plt.title("Electricity Cost Vs. Storage Start & End SOC Ratio", fontsize=18, pad=5, wrap=True)
plt.savefig("storageLVL.png")
plt.show()