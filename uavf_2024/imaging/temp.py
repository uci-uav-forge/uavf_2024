import numpy as np

def ellipsoid_axes_lengths(covariance_matrix, confidence_level):
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Calculate chi square value for the given confidence level
    dof = len(eigenvalues)  # Degrees of freedom
    chi_square_value = np.sqrt(2 * dof) * np.sqrt(np.percentile(np.random.chisquare(dof, size=100000), 100 - confidence_level))

    # Calculate lengths of the axes of the ellipsoid
    axes_lengths = 2 * chi_square_value * np.sqrt(eigenvalues)

    return axes_lengths

# Example usage
# covariance_matrix = np.array([[5.6681, 4.6314],
#                                [4.6314, 5.5951]])  # Example covariance matrix
covariance_matrix = np.array([[1, 0, 0],
                           [0.5, 1, 0],
                           [0, 0, 3]])  # Example covariance matrix
confidence_level = 95  # Confidence level in percentage

axes_lengths = ellipsoid_axes_lengths(covariance_matrix, confidence_level)
print("Axis lengths of the ellipsoid (at 95% confidence level):", axes_lengths)



# def collide_detection(pos1, vel1, radius1, pos2, vel2, radius2):
#     # Calculate relative position and relative velocity
#     rel_pos = pos2 - pos1
#     rel_vel = vel2 - vel1

#     # Calculate parameters for quadratic equation
#     a = np.dot(rel_vel, rel_vel)
#     b = 2 * np.dot(rel_vel, rel_pos)
#     c = np.dot(rel_pos, rel_pos) - (radius1 + radius2)**2

#     # Check if collision occurs
#     if a == 0:
#         # Objects are not moving relative to each other
#         if c <= 0:
#             # Objects are already colliding
#             return True, 0, pos1
#         else:
#             # Objects are not colliding
#             return False, None, None

#     discriminant = b**2 - 4*a*c
#     if discriminant < 0:
#         # No collision
#         return False, None, None

#     # Collision occurs, calculate time of collision
#     t_collision = (-b - np.sqrt(discriminant)) / (2 * a)

#     # Calculate position of collision
#     collision_pos = pos1 + vel1 * t_collision

#     return True, t_collision, collision_pos





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
