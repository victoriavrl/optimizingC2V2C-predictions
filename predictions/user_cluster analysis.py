import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Your default matplotlib settings
mpl.rcParams.update({'font.size': 16})
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Charter', 'XCharter', 'Georgia', 'Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'stix'

df = pd.read_csv("data/cluster2GM_final.csv")

session_proportions = df.pivot_table(index="user_cluster", columns="SessionCluster", aggfunc="size", fill_value=0)

session_proportions = session_proportions.div(session_proportions.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(12, 6)) #https://how2matplotlib.com/matplotlib-bar-colors.html
session_proportions.plot(kind="bar", stacked=True, colormap="Set3", ax=ax)

ax.set_ylabel("Proportion of session types (%)", fontsize=12)
ax.set_xlabel("User Cluster", fontsize=12)
ax.legend(title="Session Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("figures/session_proportions_by_user_cluster_2.pdf")

