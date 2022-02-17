#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()
#%%

dynamicroutes = []
staticroutes  = []

for i in range(10):
    dynamicroutes.append(pd.read_excel("data/routeTest/dynamicRouteTest{}.xlsx".format(i+1)))
    staticroutes.append(pd.read_excel("data/routeTest/dynamicRouteTest{}.xlsx".format(i+1)))

#%%

route_df = pd.read_excel("data/allRoutes.xlsx")

#%%

route_df.drop('Unnamed: 0', axis=1, inplace=True)

#%%
route_df.set_index('routeNum')
#%%

for test in dynamicroutes:
    route_df[route_df['routeNum'] == 2].distance
d_obj = []
d_avg_rt_lngth = []


d_obj = []
d_avg_rt_lngth = []