# %%
from scipy import stats
import numpy as np

# 1
1 - stats.norm.cdf(7.4)

# %%
# 2

n = 16
dof = n - 1
mean = 16.7
var = 7.5
standard_error = 7.5**0.5 / n**0.5

bounds = stats.t.interval(0.95, dof, loc=mean, scale=standard_error)
print("interval for the mean", bounds)


chi2_lower = stats.chi2.ppf(0.975, dof)
chi2_upper = stats.chi2.ppf(0.025, dof)
print("variance interval: ", dof * var / chi2_lower, dof * var / chi2_upper)

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
print("alpha", alpha)

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
a = np.array([20,23,17,19])
b = np.array([21,22,19,18])
c = np.array([24,17,17,21])
d = np.array([23,19,17,18])

F, p = stats.f_oneway(a,b,c,d)

print('F', F)
print('p-value', p)