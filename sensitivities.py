import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()
#%%
from os import listdir
from os.path import isfile, join
mypath = 'data/sensitivityAnalysis'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#%%
onlyfiles.sort()

#%%
# done:
# bus battery 2/3 chargers
# bus reserve 2/3 chargers
# cPow 2/3 chargers

summary = pd.read_excel('data/sensitivityAnalysis/busBatteryS3.xlsx', header=None)
summaryD = pd.read_excel('data/sensitivityAnalysis/busBatteryD3.xlsx', header=None)

#%%
summary = summary.iloc[0:4, :]
summaryD = summaryD.iloc[0:4, :]

#%%
y = summary.iloc[2, 13:].values
x = summary.iloc[0, 13:].values
plt.clf()

x1 = summaryD.iloc[0, 13:].values
y1 = summaryD.iloc[2, 13:].values

plt.plot(x1, y1, marker="o",label="Dynamic")
plt.plot(x, y, marker="o", label="Static")
# plt.scatter([0.00000,0.05000,0.10000,0.15000], [0, 0, 0, 0], marker="x", label="infeasible for dynamic", color='cornflowerblue')
# plt.scatter([0.90000,0.95000,1.00000], [0, 0, 0], marker="x", label="infeasible for static", color='orange')
plt.ylabel("Weekly Electricity Cost ($)")
plt.grid(b=True, which='both', linewidth=1, color="grey", linestyle=':')
plt.legend(loc=5)
plt.ylim(0, 600)
plt.xticks(ticks=[350 + 50 *i for i in range(10)])
plt.xlabel("Bus Battery (kWh)")

# plt.title("Storage Size w/ 3 Chargers", fontsize=18)
plt.savefig("busbattsize.png")
plt.tight_layout()
plt.show()
