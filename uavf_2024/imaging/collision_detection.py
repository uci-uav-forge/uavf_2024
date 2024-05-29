import numpy as np
from visual_3d import graph_error_ellipsoid
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def collide_detection(pos1, vel1, radius1, pos2, vel2, radius2):
    # Calculate relative position and relative velocity
    rel_pos = pos2 - pos1
    rel_vel = vel2 - vel1

    # Calculate parameters for quadratic equation
    a = np.dot(rel_vel, rel_vel)
    b = 2 * np.dot(rel_vel, rel_pos)
    c = np.dot(rel_pos, rel_pos) - (radius1 + radius2)**2

    # Check if collision occurs
    if a == 0:
        # Objects are not moving relative to each other
        if c <= 0:
            # Objects are already colliding
            return True, 0, pos1
        else:
            # Objects are not colliding
            return False, None, None

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        # No collision
        return False, None, None

    # Collision occurs, calculate time of collision
    t_collision = (-b - np.sqrt(discriminant)) / (2 * a)

    # Calculate position of collision
    collision_pos = pos1 + vel1 * t_collision

    return True, t_collision, collision_pos



if __name__ == "__main__":  
    # drones
    
    # self

    # Example usage
    pos1 = np.array([0, 0, 0])
    vel1 = np.array([1, 1, 1])
    radius1 = 1

    pos2 = np.array([3, 3, 3])
    vel2 = np.array([-1, -1, -1])
    radius2 = 1

    # collide, t_collision, collision_pos = collide_detection(pos1, vel1, radius1, pos2, vel2, radius2)
    # if collide:
    #     print("Collision detected at time:", t_collision)
    #     print("Collision position:", collision_pos)
    # else:
    #     print("No collision detected.")

    pos1 = np.array([0, 0, 0])
    vel1 = np.array([1, 1, 1])
    radius1 = 1

    pos2 = np.array([3, 3, 3])
    vel2 = np.array([-1, -1, -1])
    radius2 = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    graph_error_ellipsoid(ax, pos1, np.eye(3)*radius1)
    graph_error_ellipsoid(ax, pos2, np.eye(3)*radius2)

    plt.show()
    plt.close(fig)
    del fig



# Kalman filter predict step

#     # Predict future positions of drone and object
#     drone_future_pos = current_pos + current_velocity
#     object_future_pos = predict_object_position(drone_positions, next_wp)

#     # Calculate time to collision
#     time_to_collision = calculate_time_to_collision(drone_future_pos, object_future_pos, current_velocity)

#     # Check if collision point falls within the no-go zone
#     no_go_zone = np.array([])  # Placeholder, replace with actual calculation

#     return time_to_collision, no_go_zone

# # Example usage:
# drone_positions = [(np.array([0, 0, 0]), np.eye(7))]  # Example drone position with covariance
# current_pos = np.array([0, 0, 0])
# current_velocity = np.array([1, 1, 1])
# next_wp = np.array([5, 5, 5])

# time_to_collision, no_go_zone = collision_prediction(drone_positions, current_pos, current_velocity, next_wp)
# print("Time to collision:", time_to_collision)
# print("No-go zone:", no_go_zone)






