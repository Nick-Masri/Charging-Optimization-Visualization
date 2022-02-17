# %%
# 1-3: 3, 4-8: 5*T, 9:B*T, 10: T, 11:B*D*R
'''
    1. paramValue
    2. isFeasible
    3. cost
    4. solarPowAvail(t)
    5. solarPowTotal(t)
    6. gridPowTotal(t)
    7. gridPowAvail(t)
    8. gridPowToM(t)
    9. eM(t)
    10. eB(b,t)
    11. assignment(b,d,r)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_theme()

# %%
busBatt = pd.read_excel("data/busBatterySensitivityD.xlsx", header=None)

# %%
summary = busBatt.iloc[0:3, :]
test_1 = busBatt.iloc[:, 0]
test_2 = busBatt.iloc[:, 0]

test_1 = test_1.to_frame()
test_2 = test_2.to_frame()

test_1 = test_1.T
test_2 = test_2.T

test_1 = test_1.iloc[0, 3:].reset_index(drop=True)
test_2 = test_2.iloc[0, 3:].reset_index(drop=True)

# 4 -> 4+672 = solarPowAvail

# %%
test_1_p = pd.concat([test_1.iloc[0:672],
                      test_1.iloc[672:672 * 2].reset_index(drop=True),
                      test_1.iloc[672 * 2:672 * 3].reset_index(drop=True),
                      test_1.iloc[672 * 3:672 * 4].reset_index(drop=True),
                      test_1.iloc[672 * 4:672 * 5].reset_index(drop=True)
                      ], axis=1)

test_1_eM = test_1.iloc[672 * 5:672 * 6]

test_1_eB = pd.concat([test_1.iloc[672 * 6:672 * 7].reset_index(drop=True),
                       test_1.iloc[672 * 7:672 * 8].reset_index(drop=True),
                       test_1.iloc[672 * 8:672 * 9].reset_index(drop=True),
                       test_1.iloc[672 * 9:672 * 10].reset_index(drop=True),
                       test_1.iloc[672 * 10:672 * 11].reset_index(drop=True),
                       test_1.iloc[672 * 11:672 * 12].reset_index(drop=True),
                       test_1.iloc[672 * 12:672 * 13].reset_index(drop=True),
                       test_1.iloc[672 * 13:672 * 14].reset_index(drop=True),
                       test_1.iloc[672 * 14:672 * 15].reset_index(drop=True),
                       test_1.iloc[672 * 15:672 * 16].reset_index(drop=True)
                       ], axis=1)

# %%

# x's = route
# y's = day of week
# z's = bus

test_1_aBD1 = pd.concat([test_1.iloc[672 * 16 + 10 * i:672 * 16 + 10 * (i + 1)].reset_index(drop=True)
                         for i in range(7)], axis=1)

test_1_aBD1.columns = [i for i in range(7)]

#%%
test_1_routes = []
for j in range(10):
    route = pd.concat([test_1.iloc[672 * 16 + 10 * i +70*j:672 * 16 + 10 * (i + 1)+70*j].reset_index(drop=True)
                       for i in range(7)], axis=1)
    test_1_routes.append(route)

for route in test_1_routes:
    route.columns = [i for i in range(7)]

# %%

A = np.zeros([10, 7])  # (route, day) = bus

for idx, route in enumerate(test_1_routes):
    for column in route.columns:
        print("index: {}".format(idx))
        series = route[column]
        print(series[series == 1])
        A[idx, column] = series[series == 1].index[0]

# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(A[:, 1], [i for i in range(10)])
for idx in range(10):
    ax.scatter(idx+1,[i for i in range(7)], A[idx, ], marker='o')
ax.set(title='Assignment of Buses to Routes', ylabel='Day', xlabel='Routes', zlabel='Bus')

ax.xaxis.set(ticks=range(1, 10))
ax.yaxis.set(ticks=range(1, 7))
ax.zaxis.set(ticks=range(1, 10))
fig.show()

#%%

# how many routes a bus drives in a day and
# making sure every route has at least one bus
# (2d of bus vs route for a single day)
plt.clf()
# x = buses, y = routes
plt.scatter(A[:, 0]+1, [i+1 for i in range(10)])
plt.xticks(ticks=range(1,11))
plt.yticks(ticks=range(1,11))
plt.xlabel("Bus #")
plt.title("Day 1")
plt.ylabel("Route #")
plt.grid()
plt.show()
# %%
fig= plt.figure(figsize=(20,11))
columns = 4
rows = 2

for i in range(1, 8):
    ax = fig.add_subplot(rows, columns, i)
    ax.grid(b=True,  color='b')
    plt.scatter(A[:, i-1]+1, [j+1 for j in range(10)])
    plt.xticks(ticks=range(1, 11))
    plt.yticks(ticks=range(1, 11))
    plt.xlabel("Bus #", fontsize=15)
    plt.title("Day {}".format(i), fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    plt.ylabel("Route #", fontsize=15)


plt.show()

# %%
test_1_aBD2 = pd.concat([test_1.iloc[672 * 16 + 100 + 10 * i:672 * 16 + 100 + 10 * (i + 1)].reset_index(drop=True)
                         for i in range(10)], axis=1)

# %%

test_1_aB1R = pd.concat([test_1.iloc[672 * 16 + 100 + 10 * i:672 * 16 + 100 + 10 * (i + 1)].reset_index(drop=True)
                         for i in range(10)], axis=1)

# %%
# %%

test_1_p.columns = ["solar avail", "solar power total",
                    "grid power total", "grid power available", "grid to main",
                    "energy of main"]

# %%
plt.clf()
y = df_transformed["grid to main"]
x = df_transformed.index

plt.plot(x, y)
plt.title("Grid Power To Main")
plt.xlabel("time")
# plt.ylabel("")
plt.show()

# %%
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

fig.suptitle('Bus Energy Levels')
x = energy_of_buses.index

y1 = energy_of_buses.iloc[:, 0]
y2 = energy_of_buses.iloc[:, 1]
y3 = energy_of_buses.iloc[:, 2]
y4 = energy_of_buses.iloc[:, 3]
y5 = energy_of_buses.iloc[:, 4]
ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)
ax4.plot(x, y4)
ax5.plot(x, y5)

plt.show()

# %%
plt.clf()
fig2, (ax6, ax7, ax8, ax9, ax10) = plt.subplots(5)
y6 = energy_of_buses.iloc[:, 5]
y7 = energy_of_buses.iloc[:, 6]
y8 = energy_of_buses.iloc[:, 7]
y9 = energy_of_buses.iloc[:, 8]
y10 = energy_of_buses.iloc[:, 9]

fig.suptitle('Bus Energy Levels')

ax6.plot(x, y6)
ax7.plot(x, y7)
ax8.plot(x, y8)
ax9.plot(x, y9)
ax10.plot(x, y10)

plt.show()

#%%
fig, ax = plt.subplots()

ax.bar(1, 1035, color='b', label="Static")
ax.bar(2, 904, color='orange', label="Dynamic")
ax.legend()
ax.set_ylabel("Weekly Electricity Cost ($)")
ax.set_xticks([])
# ax.set_yticks([i*50 for i in range(2)])
ax.set_title("Weekly Electricity Cost for 20 Routes and 20 Buses", fontsize=18, pad=3, wrap=True)
fig.show()

