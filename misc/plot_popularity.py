import pandas as pd
from matplotlib import pyplot as plt

dl_df = pd.read_csv("../data/data_deeplearning.csv")
dls_df = pd.read_csv("../data/data_deeplearning_surveillance.csv")

cutoff_year = 2016

dl_df = dl_df[(dl_df["year"] >= cutoff_year) & (dl_df["year"] < 2022)]
dls_df = dls_df[(dls_df["year"] >= cutoff_year) & (dls_df["year"] < 2022)]

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
width = 0.3
dl_plot = ax.bar(dl_df["year"], dl_df["n"], width, color="blue", label="Deep Learning")
dls_plot = ax2.bar(dls_df["year"]+width, dls_df["n"], width, color="green", label="Deep Learning Surveillance")

plt.xticks(dl_df["year"] + width / 2, dl_df["year"])
ax.set_xlabel("Year")
ax.set_ylabel("Deep Learning Publications")
ax2.set_ylabel("Deep Learning Surveillance Publications")


plots = [dl_plot, dls_plot]
labs = [l.get_label() for l in plots]
ax.legend(plots, labs, loc=0)
plt.savefig("publications_graph.eps")
plt.show()