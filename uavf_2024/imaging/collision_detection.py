import numpy as np
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from filterpy.stats import plot_3d_covariance
from filterpy.kalman import predict



def is_point_in_ellipsoid(point, mean, cov, std=1):
    """
    Check if a 3D point is inside the ellipsoid defined by a mean and covariance matrix.

    Parameters
    ----------
    point : array-like, shape (3,)
        The 3D point to check.
    mean : array-like, shape (3,)
        The mean (center) of the ellipsoid.
    cov : ndarray, shape (3, 3)
        The covariance matrix defining the ellipsoid.
    std : float, optional, default=1
        The standard deviation defining the size of the ellipsoid.

    Returns
    -------
    bool
        True if the point is inside the ellipsoid, False otherwise.
    """
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    point = np.asarray(point)
    
    if cov.shape != (3, 3):
        raise ValueError("Covariance matrix must be 3x3")
    if mean.shape != (3,):
        raise ValueError("Mean must be a 3-element vector")
    if point.shape != (3,):
        raise ValueError("Point must be a 3-element vector")
    
    diff = point - mean
    dist = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(cov)), diff))
    return dist <= std


def collision_prediction(drone_positions, current_pos, next_wp,dt=0.1,time_pred=2):
    # def collision_prediction(drone_positions: list[tuple[np.ndarray, np.ndarray], current_pos: np.ndarray, current_velocity: np.ndarray, next_wp: np.ndarray) -> tuple[time_to_collision: float, no_go_zone: np.ndarray):
    '''
    drone_position will have each tuple being (x, covariance) where x is [x,y,z,vx,vy,vz,radius] and covariance is a 7x7 matrix describing the covariance of those estimates.
    current_pos, current_velocity and next_wp are of shape (3,)
    no_go_zone should be of shape (n,3) as a list of 3d points describing a convex shape.
    dt is the discrete time step to project each ellipsoid
    time_pred is total time to predict collision with each drone.
    '''

    # P = covar
    # F = state transition

    F = np.eye(7)
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for drone_info, drone_covar in drone_positions:
        time = 0
        while (time < time_pred):
            drone_info, drone_covar = predict(x=drone_info, P=drone_covar, F=F, Q=0)
            # plot_3d_covariance(drone_info[:3],drone_covar[:3,:3], ax=ax)
            time += dt
        plot_3d_covariance(drone_info[:3],drone_covar[:3,:3], ax=ax)
        # pt_true = [2,4,2]
        # pt_false = [2,2,3.1]
        # print(is_point_in_ellipsoid(pt_true,drone_info[:3],drone_covar[:3,:3],std=1))
        # print(is_point_in_ellipsoid(pt_false,drone_info[:3],drone_covar[:3,:3],std=1))
        

    plt.show()
    plt.close(fig)

def _get_last_ellipse(mean, covar, F, time_pred, dt):
    """
        Predicts the covar and mean with a state transition of F.
        Takes time steps of dt until total time projected is time_pred.
    """
    for _ in range(int(np.ceil(time_pred/dt))):
        mean, covar = predict(x=mean, P=covar, F=F, Q=0)
    return (mean, covar)


def _get_all_polytopes(drone_positions, dt,time_pred):
    out = []

    # F = state transition
    F = np.eye(7) # assumes that all drones will move linearly according to vel
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt

    for drone_info, drone_covar in drone_positions:
        start_ellipse = drone_info, drone_covar
        end_ellipse = _get_last_ellipse(drone_info, drone_covar, F, time_pred, dt)
        print(end_ellipse)
        

    return out

def collision_prediction2(drone_positions, current_pos, next_wp,dt=0.1,time_pred=2, current_covar=None):
    # def collision_prediction(drone_positions: list[tuple[np.ndarray, np.ndarray], current_pos: np.ndarray, current_velocity: np.ndarray, next_wp: np.ndarray) -> tuple[time_to_collision: float, no_go_zone: np.ndarray):
    '''
    drone_position will have each tuple being (x, covariance) where x is [x,y,z,vx,vy,vz,radius] and covariance is a 7x7 matrix describing the covariance of those estimates.
    current_pos, current_velocity and next_wp are of shape (3,)
    no_go_zone should be of shape (n,3) as a list of 3d points describing a convex shape.
    dt is the discrete time step to project each ellipsoid
    time_pred is total time to predict collision with each drone.
    '''

    # P = covar
    # F = state transition
    F = np.eye(7) # assumes that all drones will move linearly according to vel
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt
    
    if current_covar is None:
        current_covar = np.zeros([7,7])

    for drone_info, drone_covar in drone_positions:
        time = 0
        while (time < time_pred):
            drone_info, drone_covar = predict(x=drone_info, P=drone_covar, F=F, Q=0)
            current_pos, current_covar = predict(x=current_pos, P=current_covar, F=F, Q=0)

            drone_pos, drone_pos_covar = drone_info[:3], drone_covar[:3,:3]
            if (is_point_in_ellipsoid(current_pos[:3],drone_pos, drone_pos_covar)):
                print(time)
                return True,time,_get_all_polytopes(drone_positions,dt,time_pred)
            time += dt
    return False
 

if __name__ == "__main__":
    # drone_position will have each tuple being (x, covariance) where x is [x,y,z,vx,vy,vz,radius] and covariance is a 7x7 matrix describing the covariance of those estimates.
    dt = 0.2
    pos1 = np.array([0, 0, 0, 1, 1, 1, 1]) # xyz(0,0,0) vel(1,1,1) rad=1
    pos2 = np.array([3, 3, 3, -1, -1, -1, 1]) # xyz(3,3,3) vel(-1,-1,-1) rad=1

    covar = np.eye(7)*1
    covar[2,2] = 1
    covar[5,5] = 0

    out = collision_prediction2([(pos1,covar)], pos2, None,time_pred=4)
    print(out)

   










