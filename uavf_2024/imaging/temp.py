# https://users.cs.utah.edu/~tch/CS6640F2020/resources/How%20to%20draw%20a%20covariance%20error%20ellipse.pdf
# https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
 
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def graph_ellipsoid(ax, covar, center):
    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)

# your ellispsoid and center in matrix form
A = np.array([[1,0,0],[0,2,0],[0,0,2]])
center = [0,0,0]

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

graph_ellipsoid(ax,A,center)

plt.show()
plt.close(fig)
del fig