import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud(ax, point, covariance):
    # Extract x, y, z coordinates of the point
    x, y, z = point

    # Plot the point cloud
    ax.scatter(x, y, z, color='blue', label='Point Cloud')

    # Calculate eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    radii = np.sqrt(eigenvalues)

    # Plot ellipsoid representing uncertainty
    for i in range(3):
        ax.plot([x, x + radii[i] * eigenvectors[i, 0]],
                [y, y + radii[i] * eigenvectors[i, 1]],
                [z, z + radii[i] * eigenvectors[i, 2]],
                color='red')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud with Covariance Visualization')

    ax.legend()
    # plt.show()

# Example usage
point = np.array([1, 2, 3])  # Example point
covariance = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 3]])  # Example covariance matrix

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


visualize_point_cloud(ax,point, covariance)
visualize_point_cloud(ax, 2*point, 2*covariance)
plt.show()
