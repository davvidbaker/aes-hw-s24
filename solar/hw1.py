# %%

import numpy as np
import matplotlib.pyplot as plt

sin = np.sin
cos = np.cos


def equation_of_time(julian_day):
    B = 2 * np.pi * (julian_day - 1) / 365
    return 229.2 * (
        0.000075
        + 0.001868 * cos(B)
        - 0.032077 * sin(B)
        - 0.014615 * cos(2 * B)
        - 0.04089 * sin(2 * B)
    )


day = range(1, 365)
plt.plot(day, [equation_of_time(d) for d in day])

print(equation_of_time(126))