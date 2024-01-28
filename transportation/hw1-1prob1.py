# %%
# Problem 1 code
import matplotlib_inline
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np

matplotlib_inline.backend_inline.set_matplotlib_formats("retina")


def nice_grid(plot):
    plot.rc("axes", axisbelow=True)
    plot.grid(True, which="major", color="#DDD")
    plot.grid(True, which="minor", color="#EEE", linewidth=0.5)
    plot.minorticks_on()


df1 = pd.read_excel("AEO_LDVSales_2022.xlsx")

df1["battery electric"] *= 1e6
years_series = df1[df1.columns[0]]
years = years_series.values


def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


# curve fitting
popt, pcov = opt.curve_fit(logistic, years, df1["battery electric"], [1e6, 1, 2020])

perr = np.sqrt(np.diag(pcov)) * 2

pos_error = logistic(years, popt[0] + perr[0], popt[1] + perr[1], popt[2] + perr[2])
neg_error = logistic(years, popt[0] - perr[0], popt[1] - perr[1], popt[2] - perr[2])
plt.figure()
logistic_curve_fit = logistic(np.linspace(years[0], years[-1], 40), *popt)
plt.plot(years, df1["battery electric"], label="battery electric vehicles sold")
plt.plot(years, logistic(years, *popt), label="logistic curve fit")
plt.fill_between(
    years, pos_error, neg_error, facecolor="gray", alpha=0.3, label="uncertainty"
)
plt.ylabel("vehicles sold per year")
plt.xlabel("year")
plt.legend()
nice_grid(plt)

plt.figure()
nice_grid(plt)


def retired(year, avg_years_to_retire):
    year_sold = year - avg_years_to_retire
    if year_sold in years:
        return df1["battery electric"][np.where(years == year_sold)[0]].values[0]
    else:
        return 0


# number of electric vehicles retired, let's assume they all have 17 year life
retired_avg = [retired(y, 17) for y in years]
retired_long = [retired(y, 24) for y in years]
retired_short = [retired(y, 10) for y in years]

net_ev_change_on_road_per_year_avg = df1["battery electric"] - retired_avg
net_ev_change_on_road_per_year_long = df1["battery electric"] - retired_long
net_ev_change_on_road_per_year_short = df1["battery electric"] - retired_short
plt.plot(
    years,
    net_ev_change_on_road_per_year_avg,
)
plt.fill_between(
    years,
    net_ev_change_on_road_per_year_short,
    net_ev_change_on_road_per_year_long,
    color="springgreen",
    alpha=0.5,
    label="uncertainty in EV lifespan",
)
plt.ylabel("net battery electric LDV change on road per year")
plt.legend()
# number of EV on the road would then be integral of vehicles sold minus the integral of the number retired
num_ev_on_road = np.cumsum(net_ev_change_on_road_per_year_avg)
num_ev_on_road_long = np.cumsum(net_ev_change_on_road_per_year_long)
num_ev_on_road_short = np.cumsum(net_ev_change_on_road_per_year_short)
plt.figure()
plt.plot(years, num_ev_on_road)
plt.fill_between(
    years,
    num_ev_on_road_long,
    num_ev_on_road_short,
    color="thistle",
    alpha=0.3,
    label="uncertainty",
)
nice_grid(plt)
plt.legend()
plt.ylabel("number of battery electric LDVs on the road")
plt.xlabel("year")

# %%
df3 = pd.read_excel("TEDB40_LDVs_2022.xlsx")
df3["Vehicles Registerted"] *= 1000
index_of_2000 = np.where(df3["Year"].values == 2000)[0][0]

average_miles_traveled_since_2000 = df3.loc[
    df3.index > index_of_2000, df3.columns[3]
].values.mean(axis=0)
annual_miles_per_vehicle = df3[df3.columns[3]]
plt.figure()
plt.plot(df3["Year"], annual_miles_per_vehicle)
plt.ylabel("average annual miles per vehicle")
plt.xlabel("year")
plt.plot(
    df3["Year"],
    [average_miles_traveled_since_2000] * len(df3["Year"]),
    label="mean of average annual miles traveled since 2000",
)
plt.legend()
plt.ylim(0)
nice_grid(plt)

# %%
# https://www.edmunds.com/electric-car/articles/how-much-electricity-does-an-ev-use.html
kWh_per_mile = [0.25, 0.44, 0.63]

ev_on_road = [num_ev_on_road_short, num_ev_on_road, num_ev_on_road_long]
ev_electricity_demand_gWh = [0, 0, 0]

for i, num_vehicles_arr in enumerate(ev_on_road):
    energy_per_mile = kWh_per_mile[i]
    ev_electricity_demand_gWh[i] = (
        num_vehicles_arr * average_miles_traveled_since_2000 * kWh_per_mile[i] / 1e9
    )


plt.figure()
plt.plot(years, ev_electricity_demand_gWh[1].values)
plt.fill_between(
    years,
    ev_electricity_demand_gWh[0],
    ev_electricity_demand_gWh[2],
    alpha=0.3,
    color="xkcd:rose",
    label="uncertainty",
)
plt.xlim(2010, 2050)
plt.ylabel("electricity demand from LDV EV (TWh)")
plt.xlabel("year")
plt.legend()
nice_grid(plt)
