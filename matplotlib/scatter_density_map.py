import os
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


## example 1


# from scipy import stats
# import numpy as np
# def measure(n):
#      "Measurement model, return two coupled measurements."
#      m1 = np.random.normal(size=n)
#      m2 = np.random.normal(scale=0.5, size=n)
#      return m1+m2, m1-m2
# m1, m2 = measure(2000)
# xmin = m1.min()
# xmax = m1.max()
# ymin = m2.min()
# ymax = m2.max()

# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# import ipdb; ipdb.set_trace()
# positions = np.vstack([X.ravel(), Y.ravel()])
# values = np.vstack([m1, m2])
# kernel = stats.gaussian_kde(values)
# Z = np.reshape(kernel(positions).T, X.shape)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#            extent=[xmin, xmax, ymin, ymax])
# ax.plot(m1, m2, 'k.', markersize=2)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])

# example2
x=bf_ious
y=scores
xy = np.vstack([x,y])
import ipdb; ipdb.set_trace()
try:
    z = gaussian_kde(xy)(xy)
except:
    pass
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots(figsize=(7,5),dpi=100)
  
scatter = ax.scatter(x,y,marker='o',c=z,edgecolors='',s=15,label='LST'
                     ,cmap='Spectral_r')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(scatter, cax=cax, label='frequency')



plt.show()
