import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

#%%
reserveD = pd.read_excel("final-results/chargerPowSensitivityD3.xlsx", header=None)
reserve = pd.read_excel("final-results/chargerPowSensitivityS3.xlsx", header=None)
#%%
summary = reserve.iloc[0:4,:]
summaryD = reserveD.iloc[0:4, :]

#%%

plt.rcParams["figure.figsize"] = (6,3.15)
y = summary.iloc[2, :].values
x = summary.iloc[0, :].values
plt.clf()
x1 = summaryD.iloc[0, :].values

y1 = summaryD.iloc[2, :].values
plt.plot(x, y1, marker="o",label="Dynamic")
plt.plot(x, y, marker="o", label="Static", alpha=0.7)
# plt.scatter([0, 0.1], [0, 0,], marker="x", label="Infeasible for both", color='red')
# plt.scatter([0.9, 1], [0,0], marker="x", label="Infeasible for Static", color='orange')
plt.ylabel("Weekly Electricity Cost ($)", fontsize=14)
plt.grid(b=True, which='both', linewidth=1, color="grey", linestyle=':')
plt.legend()

# plt.xticks(ticks=[i*10+50 for i in range (0,11)])

# plt.ylim(400, 600)
plt.xlabel("Charger Power (kW)", fontsize=14)
# plt.title("Electricity Cost Vs. Solar Panel Size", fontsize=18, pad=5, wrap=True)

plt.tight_layout()
plt.savefig("cpow.eps", format='eps')
plt.show()

