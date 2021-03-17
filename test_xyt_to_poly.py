from shapely.geometry import Polygon
from StateEstimator import  ParticleFilter
import matplotlib.pyplot as plt

shape = Polygon([[0,0], [5,0],[5,5]])
pf = ParticleFilter(shape, 1000, (10,10))

poly = pf.xyt_to_poly(0,0,0)
print(list(poly.exterior.coords))
plt.plot(*poly.exterior.xy)
plt.show()

breakpoint()
print("Done")