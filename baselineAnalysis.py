# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

sns.set_style("whitegrid")
sns.set_theme()

# %%

# static = pd.read_excel("final-results/staticBaselineFinal.xlsx", header=None)
# dynamic = pd.read_excel("final-results/dynamicBaselineFinal.xlsx", header=None)

static = pd.read_excel("data/heuristic/heuristic4.xlsx", header=None)
dynamic = pd.read_excel("data/heuristic/heuristic5.xlsx", header=None)

models = [static, dynamic]

# %%


# transposing

for idx, model in enumerate(models):
    models[idx] = model.T

for idx, model in enumerate(models):
    models[idx] = model.iloc[0, 2:].reset_index(drop=True).to_frame()

# %%

models_p = []
for idx, model in enumerate(models):

    models_p.append(pd.concat([model.iloc[0:672].reset_index(drop=True),
                               model.iloc[672:672 * 2].reset_index(drop=True),
                               model.iloc[672 * 2:672 * 3].reset_index(drop=True),
                               model.iloc[672 * 3:672 * 4].reset_index(drop=True),
                               model.iloc[672 * 4:672 * 5].reset_index(drop=True)
                               ], axis=1))
models_eM = []
for idx, model in enumerate(models):
    models_eM.append(
        model.iloc[672 * 5:672 * 6].reset_index(drop=True)
    )

models_eB = []
for idx, model in enumerate(models):
    models_eB.append(
        pd.concat([model.iloc[672 * 6:672 * 7].reset_index(drop=True),
                   model.iloc[672 * 7:672 * 8].reset_index(drop=True),
                   model.iloc[672 * 8:672 * 9].reset_index(drop=True),
                   model.iloc[672 * 9:672 * 10].reset_index(drop=True),
                   model.iloc[672 * 10:672 * 11].reset_index(drop=True),
                   model.iloc[672 * 11:672 * 12].reset_index(drop=True),
                   model.iloc[672 * 12:672 * 13].reset_index(drop=True),
                   model.iloc[672 * 13:672 * 14].reset_index(drop=True),
                   model.iloc[672 * 14:672 * 15].reset_index(drop=True),
                   model.iloc[672 * 15:672 * 16].reset_index(drop=True)
                   ], axis=1)
    )

models_cU = []
for idx, model in enumerate(models):
    models_cU.append(pd.concat([model.iloc[11452:11452 + 672 * 10].reset_index(drop=True)], axis=1))

# %%
z = np.zeros((10, 672, 2))
for mod in range(2):

    x = models_cU[mod][:]

    for i in range(672):
        for j in range(10):
            z[j, i, mod] = x.iloc[i * 10 + j]

# %%

models_routes = [[], []]

for idx, model in enumerate(models):
    for j in range(10):
        route = pd.concat(
            [models[idx].iloc[672 * 16 + 10 * i + 70 * j:672 * 16 + 10 * (i + 1) + 70 * j].reset_index(drop=True)
             for i in range(7)], axis=1)

        route.columns = [i for i in range(7)]
        models_routes[idx].append(route)

# %%
# test 1 routes is a compilation of day vs route for all 10 buses
test_1_routes = []
for j in range(10):
    # routes is a set for a bus of days on y vs route on x
    route = pd.concat(
        [models[0].iloc[672 * 16 + 10 * i + 70 * j:672 * 16 + 10 * (i + 1) + 70 * j].reset_index(drop=True)
         for i in range(7)], axis=1)
    test_1_routes.append(route)

for route in test_1_routes:
    route.columns = [i for i in range(7)]

# %%

################################
# charging profile
################################

A = np.zeros([10, 7, 2])  # (bus, day) = route
for i, routes in enumerate(models_routes):
    print(i)
    for idx, route in enumerate(routes):
        for column in route.columns:
            print("index: {}".format(idx))
            series = route[column]
            print(series[series == 1])
            A[series[series == 1].index[0], column, i] = idx

# %%

# routes = [2, 9, 25, 98, 102, 134, 210, 6, 212, 69];
# 2; 29	38, 9; 23	34, 25; 29 77, 98; 59	72, 102; 61	77, 134; 68	82, 210; 40	76
# 6; 27 38, 212; 37	74, 69; 20	36

routeTimes = [[29, 38], [23, 34], [29, 77], [59, 72], [61, 77], [68, 82], [40, 76], [27, 38], [37, 74], [20, 36]]

#%%

plt.clf()
# fig, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10,10))
fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(7, 7))
for mod in range(2):
    a = axs[mod].barh(['Bus {}'.format(i + 1) for i in reversed(range(10))], [672 for i in range(10)], label='Idle',
                      edgecolor=None, linewidth=0)

    axs[mod].set_xticks(ticks=[96 * i for i in range(8)])
    axs[mod].tick_params(axis="x", pad=-5)

count = 0
count2 = 0

driveTimes = np.zeros([10, 2])


for mod in range(2):
    for bus in range(10):
        # chargeTimes = models_cU[mod].iloc[672*bus:672*(bus+1), 0]
        # chargeTimes = chargeTimes[chargeTimes != 0]

        cT = pd.DataFrame(z[bus, :, mod])
        cT = cT.loc[(cT != 0).any(axis=1)]

        for idx, row in cT.iterrows():
            b = axs[mod].barh(y='Bus {}'.format((bus + 1)), width=1, left=idx, color='orange',
                              label='Charging' if count == 0 else '', edgecolor=None, linewidth=0)
            count += 1

        for i, rt in enumerate(A[bus, :, mod]):
            print((routeTimes[int(rt)][1] - routeTimes[int(rt)][0]))
            driveTimes[bus, mod] += (routeTimes[int(rt)][1] - routeTimes[int(rt)][0])
            c = axs[mod].barh(y='Bus {}'.format(bus + 1), width=(routeTimes[int(rt)][1] - routeTimes[int(rt)][0]),
                              left=routeTimes[int(rt)][0] + 96 * i, color='r', label='Driving' if count2 == 0 else '',
                              edgecolor=None, linewidth=0)
            count2 += 1

# for mod in range(2):
#     if mod == 1:
#         print("static")
#     else:
#         print("dynamic")
#
#     for day in range(7):
#         print("###########################")
#         print("Day: {}".format(day + 1))
#         print("###########################")
#
#         for bus in range(10):
#             print("------------")
#             print("Bus: {}".format(bus + 1))
#             cT = pd.DataFrame(z[bus, 96 * day:96 * (day + 1), mod])
#             cT = cT.loc[(cT != 0).any(axis=1)]
#             rt = A[bus, day, mod]
#
#             print("Assignment: Route {}".format(round(A[bus, day, mod]) + 1))
#
#             dHours = routeTimes[int(rt)][0] / 4
#             if dHours <= 11:
#                 dayPeriod = 'AM'
#             else:
#                 dayPeriod = 'PM'
#                 dHours -= 12
#
#             dTime = datetime.timedelta(hours=dHours)
#             print("Departure Time: {} {}".format(str(dTime)[:4], dayPeriod))
#
#             rHours = routeTimes[int(rt)][1] / 4
#             if rHours <= 11:
#                 dayPeriod = 'AM'
#             else:
#                 dayPeriod = 'PM'
#                 rHours -= 12
#
#             rTime = datetime.timedelta(hours=rHours)
#             print("Return Time: {} {}".format(str(rTime)[:4], dayPeriod))
#
#             if len(cT) > 0:
#                 plugStart = cT.iloc[0].name / 4
#                 plugTime = datetime.timedelta(hours=plugStart)
#                 print("Plug in: " + (str(plugTime) + " AM" if plugStart <= 11 else str(
#                     datetime.timedelta(hours=plugStart - 12)) + " PM"))
#
#                 plugEnd = cT.iloc[-1].name / 4
#                 plugTime = datetime.timedelta(hours=plugEnd)
#                 print("Unplug: " + (
#                     str(plugTime) + " AM" if plugEnd <= 11 else str(datetime.timedelta(hours=plugEnd - 12)) + " PM"))

fig.legend([a[0], b[0], c[0]], ['Idle', 'Charging', 'Driving'], loc='lower right', bbox_to_anchor=(1, 0))

axs[0].text(0.5, 1, "Heuristic 4", size=14, ha="center",
            transform=axs[0].transAxes)

axs[1].text(0.5, 1, "Heuristic 5", size=14, ha="center",
            transform=axs[1].transAxes)

# plt.vlines(x='Day 1', ymin=18, ymax=96, color='r')
# ax.set_yticks([16 * (i) for i in range(7)])
# fig.suptitle("Operational Summary", y=0.95, size=18)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
fig.show()
fig.savefig("heuristic45.png")



# %%

import matplotlib.dates as mdates


def convertHours(hours, minutes):

    if hours >= 12:
        marker = "PM"
    else:
        marker = "AM"

    if hours == 0:
        hours = 12

    if hours >= 13:
        hours -= 12

    if minutes == 0:
        minutes = '00'
    return hours, minutes, marker

def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return hours, minutes, seconds

#%%

for day in range(7):

    plt.clf()
    # fig, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10,10))
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(13, 9))

    a = axs.bar(['Bus {}'.format(i + 1) for i in range(10)], [96], label='Idle',
                edgecolor=None, linewidth=0)

    axs.invert_yaxis()

    axs.set_yticks(ticks=[(96 / 8) * i for i in reversed(range(9))])
    labelList = []
    for i in reversed(range(8)):
        marker = " PM" if (96 / 8) * i >= 48 else " AM"
        labelHours = (96 / 8) * i

        if labelHours >= 48:
            labelHours -= 48
            marker = " PM"
        else:
            marker = " AM"

        labelList.append(str(datetime.timedelta(hours=labelHours / 4))[0:2].replace(':', '') + marker)

    labelList[-1] = "12 AM"
    labelList.insert(0, labelList[-1])
    labelList[4] = "12 PM"

    axs.set_yticklabels(labels=labelList)
    axs.tick_params(axis="y", pad=-5)

    count = 0
    count2 = 0



    mod = 1



    for bus in range(10):
        # chargeTimes = models_cU[mod].iloc[672*bus:672*(bus+1), 0]
        # chargeTimes = chargeTimes[chargeTimes != 0]
        cT = pd.DataFrame(z[bus, :96])
        cT = cT.loc[(cT != 0).any(axis=1)]

        for idx, row in cT.iterrows():
            b = axs.bar(x='Bus {}'.format((bus + 1)), height=1, bottom=idx, color='orange',
                        label='Charging' if count == 0 else '', edgecolor=None, linewidth=0)
            count += 1

        rt = A[bus, day, mod]

        cT_startEnd = [cT.iloc[0].name, cT.iloc[-1].name]

        for i, v in enumerate(cT_startEnd):
            plugStart = v / 4
            plugTime = datetime.timedelta(hours=plugStart)
            hours, minutes, seconds = convert_timedelta(plugTime)
            hours, minutes, marker = convertHours(hours,minutes)

            if i == 0:
                axs.text('Bus {}'.format(bus + 1), v+2.5, "{}:{} {}".format(hours,minutes,marker),
                     color='w', fontsize=10, fontweight='bold', ha='center', va='baseline')
            else:
                axs.text('Bus {}'.format(bus + 1), v+.25, "{}:{} {}".format(hours, minutes, marker),
                         color='w', fontsize=10, fontweight='bold', ha='center', va='baseline')

        c = axs.bar(x='Bus {}'.format(bus + 1), height=(routeTimes[int(rt)][1] - routeTimes[int(rt)][0]),
                    bottom=routeTimes[int(rt)][0], color='r', label='Driving' if count2 == 0 else '',
                    edgecolor=None, linewidth=0)

        driveStart = routeTimes[int(rt)][1]
        driveEnd = routeTimes[int(rt)][0]
        driveSTime = datetime.timedelta(hours=driveStart/4)
        driveETime = datetime.timedelta(hours=driveEnd/4)

        hours, minutes, seconds = convert_timedelta(driveSTime)
        hours, minutes, marker = convertHours(hours, minutes)

        axs.text('Bus {}'.format(bus + 1), routeTimes[int(rt)][1]-1, "{}:{} {}".format(hours, minutes, marker),
                 color='w', fontsize=10, fontweight='bold', ha='center', va='baseline')


        hours, minutes, seconds = convert_timedelta(driveETime)
        hours, minutes, marker = convertHours(hours, minutes)

        axs.text('Bus {}'.format(bus + 1), routeTimes[int(rt)][0]+2.5, "{}:{} {}".format(hours, minutes, marker),
                 color='w', fontsize=10, fontweight='bold', ha='center', va='baseline')

        routeLength= routeTimes[int(rt)][0] + (routeTimes[int(rt)][1] - routeTimes[int(rt)][0])/2
        if rt == 9:
            axs.text('Bus {}'.format(bus + 1), routeLength+.5, "Route #{}".format(round(rt+1)),
               color='b', fontsize=10, fontweight='bold', ha='center', va='baseline', backgroundcolor='lightgray')
        else:
            axs.text('Bus {}'.format(bus + 1), routeLength+.5, " Route #{}".format(round(rt+1)),
               color='b', fontsize=10, fontweight='bold', ha='center', va='baseline', backgroundcolor='lightgray')

        count2 += 1

    for day in range(1):
        print("###########################")
        print("Day: {}".format(day + 1))
        print("###########################")

        for bus in range(10):
            print("------------")
            print("Bus: {}".format(bus + 1))
            cT = pd.DataFrame(z[bus, 96 * day:96 * (day + 1)])
            cT = cT.loc[(cT != 0).any(axis=1)]
            rt = A[bus, day, mod]

            print("Assignment: Route {}".format(round(A[bus, day, mod]) + 1))

            dHours = routeTimes[int(rt)][0] / 4
            if dHours <= 11:
                dayPeriod = 'AM'
            else:
                dayPeriod = 'PM'
                dHours -= 12

            dTime = datetime.timedelta(hours=dHours)
            print("Departure Time: {} {}".format(str(dTime)[:4], dayPeriod))

            rHours = routeTimes[int(rt)][1] / 4
            if rHours <= 11:
                dayPeriod = 'AM'
            else:
                dayPeriod = 'PM'
                rHours -= 12

            rTime = datetime.timedelta(hours=rHours)
            print("Return Time: {} {}".format(str(rTime)[:4], dayPeriod))

            if len(cT) > 0:
                plugStart = cT.iloc[0].name / 4
                plugTime = datetime.timedelta(hours=plugStart)
                print("Plug in: " + (str(plugTime) + " AM" if plugStart <= 11 else str(
                    datetime.timedelta(hours=plugStart - 12)) + " PM"))

                plugEnd = cT.iloc[-1].name / 4
                plugTime = datetime.timedelta(hours=plugEnd)
                print("Unplug: " + (
                    str(plugTime) + " AM" if plugEnd <= 11 else str(datetime.timedelta(hours=plugEnd - 12)) + " PM"))

    fig.legend([a[0], b[0], c[0]], ['Idle', 'Charging', 'Driving'], loc='lower right', bbox_to_anchor=(1, 0.05))

    axs.set_title("Dynamic Assignment Operational Summary Day 1", fontsize=20, pad=20)

    # plt.vlines(x='Day 1', ymin=18, ymax=96, color='r')
    # ax.set_yticks([16 * (i) for i in range(7)])
    # fig.suptitle("Operational Summary", y=0.95, size=18)
    fig.tight_layout()
    fig.subplots_adjust(top=.90)
    fig.show()
    fig.savefig("operationalsummaryday{}.png".format(day))

# %%


######################################
# grid and solar power utilization
#######################################
plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

# fig.suptitle("Grid and Solar Power Utilization", fontsize=22)

# labeling static and dynamic

axs[0].text(0.5, 1.1, "Static", size=18, ha="center",
            transform=axs[0].transAxes)

axs[1].text(0.5, 1.1, "Dynamic", size=18, ha="center",
            transform=axs[1].transAxes)

# grid power
for i in range(2):
    axs[i].set_title("Grid Power Utilization")
    axs[i].set_ylabel("Grid Power (kWh)")
    axs[i].set_xlabel("Time Step")
    axs[i].plot([i for i in range(672)], models_p[i].iloc[:, 2], label='Grid Power Used')
    axs[i].plot([i for i in range(672)], models_p[i].iloc[:, 3], label='Grid Power Available')
    axs[i].legend(loc=1)
    axs[i].fill_between([i for i in range(672)], models_p[i].iloc[:, 2])
    percentGridStatic = round(models_p[i].iloc[:, 2].sum() / models_p[i].iloc[:, 3].sum() * 100, 2)
    axs[i].text(0.2, .8, str(percentGridStatic) + '%', size=18,
                transform=axs[i].transAxes, backgroundcolor='orange')

# # solar power
# for i in range(2):
#     axs[1, i].set_title("Solar Power Utilization")
#     axs[1, i].set_title("Solar Power Utilization")
#     axs[1, i].plot([i for i in range(672)], models_p[i].iloc[:, 0], label='Solar Power Used', color='orange')
#     axs[1, i].plot([i for i in range(672)], models_p[i].iloc[:, 1], label='Solar Power Available', color='b')
#     axs[1, i].legend(loc=1)
#     axs[1, i].set_xlabel("Time Step")
#     axs[1, i].set_ylabel("Solar Power (kWh)")
#     axs[1, i].fill_between([i for i in range(672)], models_p[i].iloc[:, 1], color='orange')
#     percentSolarStatic = round(models_p[i].iloc[:, 0].sum() / models_p[i].iloc[:, 1].sum() * 100, 2)
#     axs[1, i].text(0.2, .8, str(percentSolarStatic) + '%', size=18,
#                    transform=axs[1, i].transAxes, backgroundcolor='orange')

fig.tight_layout()
fig.savefig('gridutilization.eps', format='eps')
fig.show()

# %%
#############################
# Bar graph of costs
#############################

plt.clf()
y = [dynamic.iloc[1, :], static.iloc[1, :]]
plt.bar(x=1, label='static', height=static.iloc[1, :], width=0.2)
plt.bar(x=1.2, label='dynamic', height=dynamic.iloc[1, :], width=0.2)

plt.legend()
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False)  # labels along the bottom edge are off

plt.text(1.2, y[0].values[0] + 20, "$" + str(round(y[0].values[0])), fontweight='bold', va='center', ha='center')
plt.ylim(0, 600)
plt.text(1, y[1].values[0] + 20, "$" + str(round(y[1].values[0])), fontweight='bold', va='center', ha='center')
plt.ylabel("Weekly Operational Cost")
# plt.title("Baseline Weekly Operational Cost", fontsize=16, pad=10)
plt.tight_layout()

# plt.bar_label
plt.show()
plt.savefig('objcomp.png')
# %%
##########################################
# bar graph of peak hour plots
##########################################

# % Assume summer weekday pricing
# %   Peak:       .59002 dollars per kwh     12:00 - 18:00
# %   Partial:    .29319 dollars per kwh     08:30 - 12:00;   18:00 - 21:30
# %   Off Peak:   .22161 dollars per kwh     00:00 - 8:30;    21:30 - 24:00

# models_p[i].iloc[:, 2], label='Grid Power Used'
# off peak 0->34
# partial 34->48
# peak 48-> 72
# partial 72-> 86
# off peak 86-> 96

fig.clf()
plt.clf()
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(6.5, 4))
gridPowerUsed = [[0, 0], [0, 0], [0, 0]]
gridPowerAvail = [[0, 0], [0, 0], [0, 0]]
xs = ['Off Peak', 'Partial Peak', 'Peak']
axs[0].set_ylim(7000, 32000)
for i in range(2):
    for j in range(7):
        gridPowerUsed[0][i] += models_p[i].iloc[0 + 96 * j:34 + 96 * (j + 1), 2].sum() + models_p[i].iloc[
                                                                                         86 + 96 * j:96 + 96 * (j + 1),
                                                                                         2].sum()
        gridPowerUsed[1][i] += models_p[i].iloc[34 + 96 * j:48 + 96 * (j + 1), 2].sum() + models_p[i].iloc[
                                                                                          72 + 96 * j:86 + 96 * (j + 1),
                                                                                          2].sum()
        gridPowerUsed[2][i] += models_p[i].iloc[48 + 96 * j:72 + 96 * (j + 1), 2].sum()

        gridPowerAvail[0][i] += models_p[i].iloc[0 + 96 * j:34 + 96 * (j + 1), 3].sum() + models_p[i].iloc[
                                                                                          86 + 96 * j:96 + 96 * (j + 1),
                                                                                          2].sum()
        gridPowerAvail[1][i] += models_p[i].iloc[34 + 96 * j:48 + 96 * (j + 1), 3].sum() + models_p[i].iloc[
                                                                                           72 + 96 * j:86 + 96 * (
                                                                                                   j + 1), 2].sum()
        gridPowerAvail[2][i] += models_p[i].iloc[48 + 96 * j:72 + 96 * (j + 1), 3].sum()

    for k in range(3):
        percent = round(gridPowerUsed[k][i] / gridPowerAvail[k][i] * 100, 1)
        axs[i].text(xs[k], gridPowerUsed[k][i] + 1000, str(percent) + '%', size=12,
                    backgroundcolor='orange', ha='center')
        axs[i].text(xs[k], gridPowerUsed[k][i] + 3000, f'{round(gridPowerUsed[k][i]):,}', size=12, ha='center')

    axs[i].bar(x=xs[0], height=gridPowerUsed[0][i])
    axs[i].bar(x=xs[1], height=gridPowerUsed[1][i])
    axs[i].bar(x=xs[2], height=gridPowerUsed[2][i])

axs[0].set_title("Static", size=14, pad=20)
axs[1].set_title("Dynamic", size=14, pad=20)
axs[0].set_ylabel("Grid Power (kWh)")
fig.suptitle("Grid Power Used In Different Grid Pricing Hours")
fig.tight_layout()
fig.show()
fig.savefig('gridpowertimeuse.eps', format='eps')

# %%

############################
# energy of buses
############################

plt.clf()

fig, axs = plt.subplots(11, 2, sharex=True, figsize=(7, 9))

axs[0, 0].text(0.5, 1.4, "Static", size=14, ha="center",
               transform=axs[0, 0].transAxes)

axs[0, 1].text(0.5, 1.4, "Dynamic", size=14, ha="center",
               transform=axs[0, 1].transAxes)

# axs[9, 0].set_xticklabels([96*i for i in range(8)], rotation=45)
for i in range(2):
    for j in range(9):
        axs[j, i].sharey(axs[j + 1, i])
for i in range(2):
    for j in range(10):
        a = axs[j, i].plot([i for i in range(672)], models_eB[i].iloc[:, j], color='peru', label='Bus')

for i in range(2):
    b = axs[10, i].plot([i for i in range(672)], models_eM[i].iloc[:, :], color='b', label='Main Storage')
# fig.suptitle("Energy of buses and main storage (kWh) over Time", size=14)
fig.legend([a[0], b[0]], ['bus', 'main storage'], loc='lower right')
fig.tight_layout(pad=1)
fig.show()

fig.savefig("ebusenergy.png")

# %%

test_1_chargerUse = pd.concat([test_1.iloc[11452:11452 + 672].reset_index(drop=True)], axis=1)

# %%
cU_b1 = pd.concat([test_1_chargerUse[96 * i:96 * (i + 1)].reset_index(drop=True) for i in range(7)
                   ], axis=1)

# %%
buses = []

bus1CU = []
for i in range(7):
    bus1CU.append(pd.DataFrame(cU_b1.iloc[:, i]))

# %%

for idx, day in enumerate(bus1CU):
    day = day.loc[(day != 0).any(axis=1)]
    bus1CU[idx] = day
# %%

# %%


plt.clf()
# fig, axs = plt.subplots(5, 2, sharex=True, sharey=True, figsize=(10,10))
fig, ax = plt.subplots()
ax.bar(['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7'], [96 for i in range(7)], label='Idle',
       edgecolor=None, linewidth=0)
count = 0
for i, day in enumerate(bus1CU):
    for idx, row in day.iterrows():
        print(idx)
        ax.bar(x='Day {}'.format(i + 1), height=1, bottom=cU_b1_1.loc[idx].name, color='orange',
               label='Charging' if count == 0 else '', edgecolor=None, linewidth=0)
        count += 1

for i, rt in enumerate(A[0]):
    ax.bar(x='Day {}'.format(i + 1), height=(routeTimes[int(rt)][1] - routeTimes[int(rt)][0]),
           bottom=routeTimes[int(rt)][0], color='r', label='Driving' if i == 0 else '', edgecolor=None, linewidth=0)
ax.legend(loc='best', bbox_to_anchor=(1.05, 1))

# plt.vlines(x='Day 1', ymin=18, ymax=96, color='r')
ax.set_yticks([16 * (i) for i in range(7)])
fig.suptitle("Charging Profile of Bus 1")
fig.tight_layout()
fig.show()

# %%

from matplotlib.pyplot import figure



plt.clf()

figure(figsize=(8, 4.5))

plt.plot([i for i in range(672)], models_p[0].iloc[:, 0], color='orange', linewidth=2)
plt.xlabel("Time Step", fontsize=18)
plt.ylabel("Solar Power Available (kWh)", fontsize=18)
# plt.title("Solar Power Available Over Time", fontsize=18, pad=10)
plt.xticks([i * 96 for i in range(8)])
plt.tight_layout()
plt.savefig("solarpowavail.eps", format='eps')
plt.show()

# %%

test_1_routes = []
for j in range(10):
    route = pd.concat([test_1.iloc[672 * 16 + 10 * i + 70 * j:672 * 16 + 10 * (i + 1) + 70 * j].reset_index(drop=True)
                       for i in range(7)], axis=1)
    test_1_routes.append(route)

for route in test_1_routes:
    route.columns = [i for i in range(7)]

# %%
chargerUse = test_1.iloc[11462:]
# %%

A = np.zeros([10, 7])  # (bus, day) = route

for idx, route in enumerate(test_1_routes):
    for column in route.columns:
        print("index: {}".format(idx))
        series = route[column]
        print(series[series == 1])
        A[series[series == 1].index[0], column] = idx

# %%

# how many routes a bus drives in a day and
# making sure every route has at least one bus
# (2d of bus vs route for a single day)
plt.clf()
# x = buses, y = routes
plt.scatter(A[:, 0] + 1, [i + 1 for i in range(10)])
plt.xticks(ticks=range(1, 11))
plt.yticks(ticks=range(1, 11))
plt.xlabel("Bus #")
plt.title("Day 1")
plt.ylabel("Route #")
plt.grid()
plt.show()
# %%
fig = plt.figure(figsize=(20, 11))
columns = 4
rows = 2

for i in range(1, 8):
    ax = fig.add_subplot(rows, columns, i)
    ax.grid(b=True, color='b')
    plt.scatter(A[:, i - 1] + 1, [j + 1 for j in range(10)])
    plt.xticks(ticks=range(1, 11))
    plt.yticks(ticks=range(1, 11))
    plt.xlabel("Bus #", fontsize=15)
    plt.title("Day {}".format(i), fontsize=16)
    ax.set_aspect('equal', adjustable='box')
    plt.ylabel("Route #", fontsize=15)

plt.show()

# %%
# results = vertcat(isFeasible,cost,sPB, sPM');
staticSolar = pd.read_excel("data/baseline/staticBaselinesolarbreakdown.xlsx", header=None)
dynamicSolar = pd.read_excel("data/baseline/dynamicBaselinesolarbreakdown.xlsx", header=None)

# %%
staticSolar = staticSolar.iloc[2:, 0].reset_index(drop=True).to_frame()
dynamicSolar = dynamicSolar.iloc[2:, 0].reset_index(drop=True).to_frame()
# %%
# sPA',sPT',gPT', gPA', gPM'
staticMPB = pd.read_excel("data/baseline/staticBaselinempb.xlsx", header=None)
dynamicMPB = pd.read_excel("data/baseline/dynamicBaselinempb.xlsx", header=None)
