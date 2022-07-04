#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

sns.set_style()
sns.set_theme()


#%%
routes = pd.read_excel("data/allRoutes.xlsx", index_col='routeNum')

#%%
dynamicRouteTests = []
staticRouteTests = []
for i in range(10):
    dynamicRouteTests.append(pd.read_excel("data/routeTest/dynamicRouteTest{}.xlsx".format(i+1)))
    staticRouteTests.append(pd.read_excel("data/routeTest/staticRouteTest{}.xlsx".format(i + 1)))

#%%
dynamicCosts = []
staticCosts = []
dynamicAVGMileage = []
staticAVGMilegage = []
dynamicTotMileage = []
staticTotMileage = []

for i in range(10):
    test = dynamicRouteTests[i]
    dynamicCosts.append(test.iloc[0:10].loc[[0]].values[0][0])
    for j in range(10):
        route = test.iloc[1:11].loc[j+1].values[0]
        distance = routes.loc[[route]].distance.values[0]



