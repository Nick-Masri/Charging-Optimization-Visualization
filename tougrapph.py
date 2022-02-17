# % Assume summer weekday pricing
# %   Peak:       .59002 dollars per kwh     12:00 - 18:00
# %   Partial:    .29319 dollars per kwh     08:30 - 12:00;   18:00 - 21:30
# %   Off Peak:   .22161 dollars per kwh     00:00 - 8:30;    21:30 - 24:00




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
import seaborn as sns
sns.set_theme()


#%%


fig, ax = plt.subplots(figsize=(6, 3))
offpeak = ax.bar([0,21.5], height=[0.22, 0.22], align='edge', label='Off Peak       $0.22', color='b', width=[8.5, 2.5])
partialpeak = ax.bar([8.5, 18], height=[0.29, 0.29], align='edge', label='Partial Peak  $0.29', color='y', width=[12-8.5,3.5])
peak = ax.bar([12], height=[0.59], align='edge', label='Peak             $0.59', color='r', width=[6])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Cost of Electricity ($/kWh)")
# ax.set_title("Time Of Use (TOU) Grid Pricing Structure", fontsize=20, pad=15)
ax.set_xticks(ticks=[0, 8.5, 12, 18, 21.5, 24])
ax.set_xticklabels(['12:00AM', '8:30AM', '12:00PM', '6:00PM', '9:30PM', '12:00AM'], rotation='25')

ax.legend()


fig.tight_layout()
plt.savefig('tou.png')

plt.show()
