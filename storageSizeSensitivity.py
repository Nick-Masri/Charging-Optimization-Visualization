import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

#%%
reserve = pd.read_excel("data/tograph/storageSizeSensitivityS2.xlsx", header=None)
reserveD = pd.read_excel("data/tograph/storageSizeSensitivityD2Chargers.xlsx", header=None)
#%%
summary = reserve.iloc[0:4,:]
summaryD = reserveD.iloc[0:4, :]

#%%
y = summary.iloc[2, 0:12].values
x = summary.iloc[0, 0:12].values
plt.clf()
x1 = summaryD.iloc[0, 0:12].values

y1 = summaryD.iloc[2, 0:12].values
# plt.plot(x1, y1, marker="o",label="Dynamic")
plt.plot(x, y, marker="o", label="Both")
# plt.scatter([0, 0.1], [0, 0,], marker="x", label="Infeasible for both", color='red')
# plt.scatter([0.9, 1], [0,0], marker="x", label="Infeasible for Static", color='orange')
plt.ylabel("Weekly Electricity Cost ($)", fontsize=14)
plt.grid(b=True, which='both', linewidth=1, color="grey", linestyle=':')
plt.legend()
plt.tight_layout(pad=2)
plt.ylim(300, 1000)
# plt.xticks(ticks=[i*.1 for i in range (0,11)])

plt.xlabel("Storage Size (kWh)", fontsize=14)
plt.title("Electricity Cost Vs. Storage Size", fontsize=18, pad=5, wrap=True)
# plt.savefig("cNumDualSensitivity.png")
plt.show()