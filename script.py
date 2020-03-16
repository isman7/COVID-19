from pathlib import Path
from functools import wraps
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use("ggplot")

status = ["Confirmed", "Deaths", "Recovered"]


def boltzmann(xs, ymin, ymax, xc, dx):
    return ymax + (ymin - ymax) / (1 + np.exp((xs - xc) / dx))


def boltzmann_min_0(xs, ymax, xc, dx):
    return boltzmann(xs, 0, ymax, xc, dx)


def boltzmann_min_0_max_value(ymax):
    @wraps(boltzmann_min_0)
    def wrapper(xs, xc, dx):
        return boltzmann_min_0(xs, ymax, xc, dx)
    return wrapper


def read_table(status: str = "Confirmed"):
    data_csv = Path(f"csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-{status}.csv")
    data = pd.read_csv(data_csv)
    data: pd.DataFrame = (
        data
        .set_index("Country/Region")
        .drop(["Province/State", "Lat", "Long"], axis=1)
    )
    data = data.groupby("Country/Region").sum()
    data.columns = pd.to_datetime(data.columns)
    data: pd.DataFrame = data.transpose()
    data.index.name = "Timestamps"
    data.index.freq = "d"

    return data


confirmed: pd.DataFrame = read_table(status[0])
deaths: pd.DataFrame = read_table(status[1])
recovered: pd.DataFrame = read_table(status[2])
active: pd.DataFrame = confirmed - deaths - recovered

confirmed["World wo/ China"] = confirmed.loc[:, confirmed.columns != "China"].sum(axis=1)
deaths["World wo/ China"] = deaths.loc[:, deaths.columns != "China"].sum(axis=1)
recovered["World wo/ China"] = recovered.loc[:, recovered.columns != "China"].sum(axis=1)
active["World wo/ China"] = active.loc[:, active.columns != "China"].sum(axis=1)


def plot_country(country: str, ax: plt.Axes, marker: str):
    active[country].plot.line(ax=ax, marker=marker, linewidth=0, label=f"{country}: Active cases", color="black")
    confirmed[country].plot.line(ax=ax, marker=marker, linewidth=0, label=f"{country}: Confirmed cases", color="blue")
    recovered[country].plot.line(ax=ax, marker=marker, linewidth=0, label=f"{country}: Recovered cases", color="green")
    deaths[country].plot.line(ax=ax, marker=marker, linewidth=0, label=f"{country}: Deaths", color="red")


# country = "China"
# country = "Italy"
# country = "Iran"
# country = "Spain"
# country = "Korea, South"
country = "World wo/ China"

fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot(111)
plot_country(country, ax, "o")

len_index = len(confirmed.index)
x = np.arange(len_index)
x2 = np.arange(len_index*2)
long_index = confirmed.index.union(confirmed.index + len_index)

p0 = [30000, 15, 4.5]
p_opt_confirmed, p_cov_confirmed = curve_fit(
    boltzmann_min_0,
    x,
    confirmed[country].values,
    p0=p0,
    maxfev=2000,
    # bounds=([0, 0, 4], [100000, 100, 5]),
    # bounds=(0, [100000, 100, 100]),
)

b_opt_confirmed = pd.Series(boltzmann_min_0(x2, *p_opt_confirmed), index=long_index)
b_opt_confirmed.plot.line(ax=ax, linestyle="--", color="blue", label=f"{country}: Confirmed cases fitting")


p0 = [3000, 25, 2]
p_opt_deaths, p_cov_deaths = curve_fit(
    boltzmann_min_0,
    x,
    deaths[country].values,
    p0=p0,
    maxfev=2000,
    # bounds=(0, [4000, 100, 100]),
)

b_opt_deaths = pd.Series(boltzmann_min_0(x2, *p_opt_deaths), index=long_index)
b_opt_deaths.plot.line(ax=ax, linestyle="--", color="red", label=f"{country}: Deaths fitting")

y_max_recovered = p_opt_confirmed[0] - p_opt_deaths[0]

boltzmann_min_0_max_recovered = boltzmann_min_0_max_value(y_max_recovered)

p0 = [25, 2]
p_opt_recovered, p_cov_recovered = curve_fit(
    boltzmann_min_0_max_recovered,
    x,
    recovered[country].values,
    p0=p0,
    maxfev=2000,
)

b_opt_recovered = pd.Series(boltzmann_min_0_max_recovered(x2, *p_opt_recovered), index=long_index)
b_opt_recovered.plot.line(ax=ax, linestyle="--", color="green", label=f"{country}: Recovered cases fitting")

b_opt_active = b_opt_confirmed - b_opt_deaths - b_opt_recovered
b_opt_active.plot.line(ax=ax, linestyle="--", color="black", label=f"{country}: Active cases fitting")

ax.legend()
ax.set_title(country)