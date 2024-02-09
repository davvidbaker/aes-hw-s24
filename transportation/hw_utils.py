import matplotlib


def nice_grid(plot_or_ax):
    if isinstance(plot_or_ax, matplotlib.axes._axes.Axes):
        plot_or_ax.xaxis.grid(True, which="major", color="#DDD")
        plot_or_ax.xaxis.grid(True, which="minor", color="#EEE", linewidth=0.5)
        plot_or_ax.yaxis.grid(True, which="major", color="#DDD")
        plot_or_ax.yaxis.grid(True, which="minor", color="#EEE", linewidth=0.5)
        plot_or_ax.minorticks_on()
    else:
        plot_or_ax.rc("axes", axisbelow=True)
        plot_or_ax.grid(True, which="major", color="#DDD")
        plot_or_ax.grid(True, which="minor", color="#EEE", linewidth=0.5)
        plot_or_ax.minorticks_on()
