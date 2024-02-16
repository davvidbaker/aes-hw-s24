# %%
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# 1
1 - stats.norm.cdf(7.4)

fig, ax = plt.subplots()

# %%
# 2

n = 16
dof = n - 1
mean = 16.7
var = 7.5
standard_error = 7.5**0.5 / n**0.5

bounds = stats.t.interval(0.95, dof, loc=mean, scale=standard_error)
print("interval for the mean", bounds)


chi2_upper = stats.chi2.ppf(1 - 0.975, dof)  # big -> inverted small
chi2_lower = stats.chi2.ppf(1 - 0.025, dof)  # small   -> inverted big
print("chi2_lower", chi2_lower)
print("chi2_upper", chi2_upper)

print("variance interval: ", dof * var / chi2_lower, dof * var / chi2_upper)

stats.chi2.sf(0.025, dof)
stats.chi2.sf(0.975, dof)

stats.chi2.interval(0.95, dof)

fig, ax = plt.subplots(3, figsize=(12, 6))
for dof in np.arange(15, 16, 1):
    x = np.arange(0, 50, 0.1)
    # ax.set_xlim(0, 50)
    # ax.set_ylim(0, 1)
    ax[0].plot(x, [stats.chi2.pdf(xx, dof) for xx in x], alpha=0.5, label="pdf")
    ax[1].plot(
        x, [stats.chi2.cdf(xx, dof) for xx in x], linewidth=12, alpha=0.5, label="cdf"
    )
    ax[1].plot(x, [stats.chi2.sf(xx, dof) for xx in x], alpha=0.5, label="sf")
    ax[1].plot(x, [1 - stats.chi2.sf(xx, dof) for xx in x], alpha=1, label="1 minus sf")

ps = np.arange(0, 1, 0.01)
ax[2].plot(
    ps,
    [stats.chi2.ppf(1 - p, dof) for p in ps],
    alpha=0.5,
    label="ppf",
)

acs = [0.025, 0.975]
ax[2].vlines(acs, 0, 30, alpha=0.1)
ax[2].hlines([stats.chi2.ppf(1 - a, dof) for a in acs], 0, 1, alpha=0.1)


ax[0].set_xlim(0, 50)
ax[1].set_xlim(0, 50)
ax[2].set_xlim(0, 1)
ax[2].set_ylim(0, 30)
ax[2].set_xlabel("p-value")

ax[0].legend()
ax[1].legend()
ax[2].legend()


# %%
# 3
n = 50
dof = n - 1
mean = 78
var = 64
standard_error = 8 / n**0.5

bounds_with_t_dist = stats.t.interval(0.95, dof, loc=mean, scale=standard_error)
bounds_with_norm_dist = stats.norm.interval(0.95, loc=mean, scale=standard_error)
print("bounds with t dist:", bounds_with_t_dist)
print("bounds with norm dist:", bounds_with_norm_dist)

1 - stats.norm.cdf(0.885)
alpha = (1 - stats.norm.cdf(0.885)) * 2

print(stats.norm.interval(0.62, loc=mean, scale=standard_error))
# %%
# 4

1 - stats.norm.cdf(2)

# %%
# 5
data = [176.2, 157.9, 160.1, 180.9, 165.1, 167.2, 162.9, 155.7, 166.2]
m = np.average(data)
s = np.std(data, ddof=1)
print(s)
n = 9
se = s / n**0.5

t_thresh = stats.t.ppf(0.025, 8)
print(" t thresh", t_thresh)
stats.t.sf(np.abs(-1.53), n - 1) * 2
# %%
# 6

1 - stats.norm.cdf(7.7)

stats.norm.sf(7.7)

# %%
# 7
dof = 178

p_value = stats.t.sf(np.abs(-3.1), dof) * 2


print("p_value", p_value)
stats.t.ppf(0.001125, dof)

# %%
# 8
a = np.array([20, 23, 17, 19])
b = np.array([21, 22, 19, 18])
c = np.array([24, 17, 17, 21])
d = np.array([23, 19, 17, 18])

F, p = stats.f_oneway(a, b, c, d)

print("F", F)
print("p-value", p)
