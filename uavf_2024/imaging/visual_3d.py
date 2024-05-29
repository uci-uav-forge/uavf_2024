import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from filterpy.stats import plot_3d_covariance
from filterpy.kalman import predict


def collision_prediction(drone_positions, current_pos, current_velocity, next_wp):
    # def collision_prediction(drone_positions: list[tuple[np.ndarray, np.ndarray], current_pos: np.ndarray, current_velocity: np.ndarray, next_wp: np.ndarray) -> tuple[time_to_collision: float, no_go_zone: np.ndarray):
    '''
    drone_position will have each tuple being (x, covariance) where x is [x,y,z,vx,vy,vz,radius] and covariance is a 7x7 matrix describing the covariance of those estimates.

    current_pos, current_velocity and next_wp are of shape (3,)

    no_go_zone should be of shape (n,3) as a list of 3d points describing a convex shape.
    '''

    dt = 0.1
    # P = covar
    # F = state transition

    F = np.eye(7)
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for drone_pos, drone_covar in drone_positions:
        time = 0
        while (time < 2):
            drone_pos, drone_covar = predict(x=drone_pos, P=drone_covar, F=F, Q=0)
            
            plot_3d_covariance(pos1[:3],P[:3,:3], ax=ax)
            

            time += dt
            # print("hi")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # drone_position will have each tuple being (x, covariance) where x is [x,y,z,vx,vy,vz,radius] and covariance is a 7x7 matrix describing the covariance of those estimates.

    dt = 0.2
    pos1 = np.array([0, 0, 0, 1, 1, 1, 1]) # xyz(0,0,0) vel(1,1,1) rad=1
    # pos2 = np.array([3, 3, 3, -1, -1, -1, 1]) # xyz(3,3,3) vel(-1,-1,-1) rad=1

    P = np.eye(7)*1
    P[2,2] = 1
    P[5,5] = 0
    F = np.eye(7)
    F[0,3] = dt
    F[1,4] = dt
    F[2,5] = dt


    # collision_prediction(drone_positions, current_pos, current_velocity, next_wp):

    # det = collision_prediction([(pos1,P)], np.array([3, 3, 3]), np.array([-1, -1, -1]), None)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for _ in range(10):
        pos1, P = predict(x=pos1, P=P, F=F, Q=0)
        print('x =', pos1)
        print('P =', P)

        plot_3d_covariance(pos1[:3],P[:3,:3], ax=ax)
    plt.show()
    plt.close(fig)

