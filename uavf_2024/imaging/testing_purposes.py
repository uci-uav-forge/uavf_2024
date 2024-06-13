import numpy as np

from filterpy.stats import plot_3d_covariance
from filterpy.kalman import predict

def plot_3d_point(point, ax=None, color='r', marker='o', label=None):
    """
    Plots a 3D point on a given axis.
    
    Parameters
    ----------
    point : array-like
        3D coordinates of the point (x, y, z).
    
    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Axis to draw on. If not provided, a new 3D axis will be generated for the current figure.
    
    color : str, optional
        Color of the point.
    
    marker : str, optional
        Marker style of the point.
    
    label : str, optional
        Label for the point.
    
    Returns
    -------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The axis with the plotted point.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(point[0], point[1], point[2], color=color, marker=marker, label=label)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if label:
        ax.legend()
    
    return ax


def find_intersection(mu, sigma, p, v):
    # Ensure v is a unit vector
    v = v / np.linalg.norm(v)

    # Calculate d
    d = p - mu

    # Calculate quadratic coefficients
    a = v.T @ np.linalg.inv(sigma) @ v
    b = 2 * d.T @ np.linalg.inv(sigma) @ v
    c = d.T @ np.linalg.inv(sigma) @ d - 1

    # Solve quadratic equation
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real intersection points")

    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)

    # Calculate intersection points
    intersection1 = p + t1 * v
    intersection2 = p + t2 * v

    return intersection1, intersection2

# Example usage
mu = np.array([0, 0, 0])
sigma = np.array([[1, 0.5, 0.3], [0.5, 2, 0.2], [0.3, 0.2, 1]])
# p = np.array([2, 2, 2])
p = np.array([0,0,0])
v = np.array([1, 0, 0])

intersection1, intersection2 = find_intersection(mu, sigma, p, v)
print("Intersection points:", intersection1, intersection2)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


plot_3d_covariance(mu,sigma, ax=ax)
plot_3d_point(intersection1,ax,'b')
plot_3d_point(intersection2,ax,'g')
plot_3d_point(p,ax,'r')

plt.show()
plt.close(fig)


