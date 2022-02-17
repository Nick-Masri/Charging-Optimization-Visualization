import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

#%%
cNum = pd.read_excel("data/sensitivity/chargerNumSensitivity.xlsx", header=None)
cNumD = pd.read_excel("data/sensitivity/chargerNumSensitivityD.xlsx", header=None)
#%%
summary = cNum.iloc[0:4,:]
summaryD = cNumD.iloc[0:4, :]

#%%
y = summary.iloc[2, 1:6].values
x = summary.iloc[0, 1:6].values
plt.clf()
x1 = summaryD.iloc[0, 1:6].values
y1 = summaryD.iloc[2, 1:6].values
plt.plot(x1, y1, marker="o",label="Dynamic")
plt.plot(x, y, marker="o", label="Static")
plt.scatter([0, 1], [0, 0], marker="x", label="infeasible for both", color='red')
plt.ylabel("Weekly Electricity Cost ($)")
plt.grid(b=True, which='both', linewidth=1, color="grey", linestyle=':')
plt.legend(loc=7)
# plt.ylim(0, 600)
plt.xticks(ticks=[i for i in range (0,7)])
plt.xlabel("Number of Chargers Used")
plt.title("Electricity Cost Vs. Number of Chargers", fontsize=18, pad=15)
# plt.savefig("cNumDualSensitivity.png")
plt.show()