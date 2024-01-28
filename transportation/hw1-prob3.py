# %%
import pandas as pd

pd.options.display.float_format = "{:,.2f}".format

df = pd.DataFrame(
    index=["Ram", "Silverado", "F150"],
    data={
        "car": ["2024 Ram 1500 TRX 4WD", "2024 Chevy Silverado 4WD", "2004 Ford F150"],
        "engine displacement (L)": [5.6, 5.3, 5.4],
        "fuel type": ["gasoline", "Ethanol (E85)", "CNG"],
        "MPG gasoline equivalent": [12, 17, 13],
        "km/L gasoline equivalent": [0.0] * 3,
        "km/L ethanol": [None, None, None],
        "km/kg CNG": [None, None, None],
        "kg CO_2/L fuel": [2.29, 1.61, None],
        "kg CO_2/kg CNG": [None, None, 2.67],
        "kg CO_2/km": [0.0] * 3,
        "kg CO_2/year": [0.0] * 3,
        "kg CO_2 over 15 years": [0.0] * 3,
    },
)

df.loc[:, "km/L gasoline equivalent"] = df.loc[:, "MPG gasoline equivalent"] * 0.425144
df.loc["Silverado", "km/L ethanol"] = (
    df.loc["Silverado", "km/L gasoline equivalent"] / 1.39
)
df.loc["F150", "km/kg CNG"] = 1.609 / 2.567 * df.loc["F150", "MPG gasoline equivalent"]

df.loc["Ram", "kg CO_2/km"] = (
    df.loc["Ram", "kg CO_2/L fuel"] * 1 / df.loc["Ram", "km/L gasoline equivalent"]
)
df.loc["Silverado", "kg CO_2/km"] = (
    df.loc["Silverado", "kg CO_2/L fuel"] * 1 / df.loc["Silverado", "km/L ethanol"]
)
df.loc["F150", "kg CO_2/km"] = (
    df.loc["F150", "kg CO_2/kg CNG"] * 1 / df.loc["F150", "km/kg CNG"]
)
df.loc[:, "kg CO_2/year"] = df.loc[:, "kg CO_2/km"] * 22_000
df.loc[:, "kg CO_2 over 15 years"] = df.loc[:, "kg CO_2/year"] * 15

