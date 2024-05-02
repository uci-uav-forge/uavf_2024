import math
from math import cos, sin
import random
import warnings
import numpy as np
from numpy.linalg import inv
import scipy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spln
from scipy.stats import norm, multivariate_normal

def _eigsorted(cov, asc=True):
    """
    Computes eigenvalues and eigenvectors of a covariance matrix and returns
    them sorted by eigenvalue.

    Parameters
    ----------
    cov : ndarray
        covariance matrix

    asc : bool, default=True
        determines whether we are sorted smallest to largest (asc=True),
        or largest to smallest (asc=False)

    Returns
    -------
    eigval : 1D ndarray
        eigenvalues of covariance ordered largest to smallest

    eigvec : 2D ndarray
        eigenvectors of covariance matrix ordered to match `eigval` ordering.
        I.e eigvec[:, 0] is the rotation vector for eigval[0]
    """

    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()
    if not asc:
        # sort largest to smallest
        order = order[::-1]

    return eigval[order], eigvec[:, order]



def plot_3d_covariance(mean, cov, std=1.,
                       ax=None, title=None,
                       color=None, alpha=1.,
                       label_xyz=True,
                       N=60,
                       shade=True,
                       limit_xyz=True,
                       **kwargs):
    """
    Plots a covariance matrix `cov` as a 3D ellipsoid centered around
    the `mean`.

    Parameters
    ----------

    mean : 3-vector
        mean in x, y, z. Can be any type convertable to a row vector.

    cov : ndarray 3x3
        covariance matrix

    std : double, default=1
        standard deviation of ellipsoid

    ax : matplotlib.axes._subplots.Axes3DSubplot, optional
        Axis to draw on. If not provided, a new 3d axis will be generated
        for the current figure

    title : str, optional
        If provided, specifies the title for the plot

    color : any value convertible to a color
        if specified, color of the ellipsoid.

    alpha : float, default 1.
        Alpha value of the ellipsoid. <1 makes is semi-transparent.

    label_xyz: bool, default True

        Gives labels 'X', 'Y', and 'Z' to the axis.

    N : int, default=60
        Number of segments to compute ellipsoid in u,v space. Large numbers
        can take a very long time to plot. Default looks nice.

    shade : bool, default=True
        Use shading to draw the ellipse

    limit_xyz : bool, default=True
        Limit the axis range to fit the ellipse

    **kwargs : optional
        keyword arguments to supply to the call to plot_surface()
    """

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # force mean to be a 1d vector no matter its shape when passed in
    mean = np.atleast_2d(mean)
    if mean.shape[1] == 1:
        mean = mean.T

    if not(mean.shape[0] == 1 and mean.shape[1] == 3):
        raise ValueError('mean must be convertible to a 1x3 row vector')
    mean = mean[0]

    # force covariance to be 3x3 np.array
    cov = np.asarray(cov)
    if cov.shape[0] != 3 or cov.shape[1] != 3:
        raise ValueError("covariance must be 3x3")

    # The idea is simple - find the 3 axis of the covariance matrix
    # by finding the eigenvalues and vectors. The eigenvalues are the
    # radii (squared, since covariance has squared terms), and the
    # eigenvectors give the rotation. So we make an ellipse with the
    # given radii and then rotate it to the proper orientation.

    eigval, eigvec = _eigsorted(cov, asc=True)
    radii = std * np.sqrt(np.real(eigval))

    if eigval[0] < 0:
        raise ValueError("covariance matrix must be positive definite")


    # calculate cartesian coordinates for the ellipsoid surface
    u = np.linspace(0.0, 2.0 * np.pi, N)
    v = np.linspace(0.0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v)) * radii[0]
    y = np.outer(np.sin(u), np.sin(v)) * radii[1]
    z = np.outer(np.ones_like(u), np.cos(v)) * radii[2]

    # rotate data with eigenvector and center on mu
    a = np.kron(eigvec[:, 0], x)
    b = np.kron(eigvec[:, 1], y)
    c = np.kron(eigvec[:, 2], z)

    data = a + b + c
    N = data.shape[0]
    x = data[:,   0:N]   + mean[0]
    y = data[:,   N:N*2] + mean[1]
    z = data[:, N*2:]    + mean[2]

    fig = plt.gcf()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(x, y, z,
                    rstride=3, cstride=3, linewidth=0.1, alpha=alpha,
                    shade=shade, color=color, **kwargs)

    # now make it pretty!

    if label_xyz:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    if limit_xyz:
        r = radii.max()
        ax.set_xlim(-r + mean[0], r + mean[0])
        ax.set_ylim(-r + mean[1], r + mean[1])
        ax.set_zlim(-r + mean[2], r + mean[2])

    if title is not None:
        plt.title(title)

    #pylint: disable=pointless-statement
    Axes3D #kill pylint warning about unused import

    return ax


