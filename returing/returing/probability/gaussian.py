import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use('TkAgg')
import numpy as np
np.random.seed(20170430)
from scipy.stats import norm


x = np.arange(-10, 10)
g = norm(0, 1)
y = g.pdf(x)

plt.plot(x, y)
plt.interactive(False)
plt.show()

