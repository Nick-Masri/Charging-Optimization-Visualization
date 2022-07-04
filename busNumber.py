import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()

#%%

reserve = pd.read_excel("data/tograph/busNumberD.xlsx", header=None)
#%%
summary = reserve.iloc[0:4,:]

#%%
y = summary.iloc[2, 0:7].values
x = summary.iloc[0, 0:7].values

y[4] = 545.95
y[5] = 545.95

plt.clf()
plt.plot(x, y, marker="o", label="Dynamic")
# plt.scatter([0, 0.1], [0, 0,], marker="x", label="Infeasible for both", color='red')
plt.scatter([4, 5, 6, 7, 8, 9], [0,0,0,0,0,0], marker="x", label="Infeasible for Static", color='orange')
plt.ylabel("Weekly Electricity Cost ($)", fontsize=14)
plt.plot([10, 11], [545.9, 545.9], label="Both", color='orange', marker='o')
# plt.ylim(530, 640)
plt.grid(b=True, which='both', linewidth=1, color="grey", linestyle=':')
plt.legend()
plt.tight_layout(pad=4)


plt.xlabel("Number of Buses Used", fontsize=14)
plt.title("Electricity Cost Vs. Number of Buses Used", fontsize=18, pad=5, wrap=True)
# plt.savefig("cNumDualSensitivity.png")
plt.show()