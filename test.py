import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

lines = [
    [(10.26,2.51),(10.26,2.51)],
    [(10.26,2.51),(6.60,-2.65)],
    [(6.60,-2.65),(6.49,0.76)],
    [(6.60,-2.65),(5.78,2.35)],
    [(10.26,2.51),(6.60,-2.65)],
    [(6.60,-2.65),(4.56,6.36)],
    [(6.60,-2.65),(9.05,7.70)],
    [(10.26,2.51),(5.54,8.76)],
    [(5.54,8.76),(2.14,4.92)],
    [(5.54,8.76),(3.24,0.66)],
    [(10.26,2.51),(5.54,8.76)],
    [(5.54,8.76),(4.99,5.93)],
    [(5.54,8.76),(2.41,6.25)],
]


c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

fig, ax = pl.subplots()
# ax.set_xlim([0,8])
# ax.set_ylim([-8,8])
for l in lines:
    plt.plot([l[0][0], l[1][0]], [l[0][1],l[1][1]])
    plt.pause(3)

plt.show()
# lc = mc.LineCollection(lines, colors=c, linewidths=2)
# fig, ax = pl.subplots()
# ax.add_collection(lc)
# ax.autoscale()
# ax.margins(0.1)

plt.show()