import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# https://users.cs.utah.edu/~tch/CS6640F2020/resources/How%20to%20draw%20a%20covariance%20error%20ellipse.pdf
# https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib
def graph_ellipsoid(ax, ellipsoid_matrix, center):
    # find the rotation matrix and radii of the axes
    U, s, rotation = linalg.svd(ellipsoid_matrix)
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

# from numpy.linalg import svd
def error_ellipsoid_matrix(point, covariance_matrix):
    # Compute eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Construct diagonal matrix with eigenvalues on the diagonal
    diagonal_matrix = np.diag(eigenvalues)

    # Form error ellipsoid matrix using eigenvectors and diagonal matrix
    error_ellipsoid = np.dot(np.dot(eigenvectors, diagonal_matrix), np.linalg.inv(eigenvectors))

    return error_ellipsoid


def graph_error_ellipsoid(ax, point, covar):
    eigenvalues, eigenvectors = np.linalg.eigh(covar)
    axes_lengths = np.sqrt(eigenvalues)
    ellipsoid = np.array([[axes_lengths[0], 0, 0],
                        [0, axes_lengths[1], 0],
                        [0, 0, axes_lengths[2]]])
    graph_error_axes(ax, point, covar)
    graph_ellipsoid(ax,covar,point)


def graph_error_axes(ax, point, covariance):
    x, y, z = point
    ax.scatter(x, y, z, color='blue', label='Point Cloud')

    # Calculate eigenvalues and eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    radii = np.sqrt(eigenvalues)
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

if __name__ == "__main__":
    import filterpy
    # Example usage
    point = np.array([1, 2, 3])  # Example point
    covariance = np.array([[1, 0.5, 0],
                           [0.5, 1, 0],
                           [0, 0, 3]])  # Example covariance matrix

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    graph_error_ellipsoid(ax, point, covariance)
    # filterpy.plot_3d_covariance(point,covariance)

    plt.show()
    plt.close(fig)
    del fig
