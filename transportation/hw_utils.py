def nice_grid(plot):
    plot.rc("axes", axisbelow=True)
    plot.grid(True, which="major", color="#DDD")
    plot.grid(True, which="minor", color="#EEE", linewidth=0.5)
    plot.minorticks_on()
