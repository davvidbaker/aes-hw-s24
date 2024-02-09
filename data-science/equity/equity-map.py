# %%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import contextily as ctx
import geopandas as gpd
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# path = "tl_2023_08_tract.shp"
path = "/Users/david/code/aes-hw/data-science/equity/tl_2023_08_tract.shp"
df = gpd.read_file("tl_2023_08_tract.shp")
df = df.to_crs("EPSG:4326")

df_vuln = pd.read_csv(
    "./vulnerability_scores.csv",
)

df.iloc[0, :]

denver_geoids = df_vuln["GEOIDFQ"].values
print(denver_geoids)

mask = df["GEOIDFQ"].isin(denver_geoids)
df_denver = df[mask]
df_merged = df_denver.merge(df_vuln, on="GEOIDFQ", how="left")

df_merged
# %%

f, ax = plt.subplots(1, 1, figsize=(15, 10), sharex=True, sharey=True, dpi=300)
f.tight_layout()
plt.title("Denver Vulnerability")
ax.set_axis_off()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.5, alpha=0.5)
df_merged.plot(
    "score",
    ax=ax,
    alpha=0.5,
    cmap="viridis",
    edgecolor="k",
    legend=True,
    cax=cax,
    linewidth=0.1,
)
plt.ylabel("Vulnerability (higher = more vulnerable)", fontsize=12)

df_merged["coords"] = df_merged["geometry"].apply(
    lambda x: x.representative_point().coords[:]
)
df_merged["coords"] = [coords[0] for coords in df_merged["coords"]]
for idx, row in df_merged.iterrows():
    ax.annotate(text=row["NAME"], xy=row["coords"], horizontalalignment="center", size='3')
    
plt.show()
