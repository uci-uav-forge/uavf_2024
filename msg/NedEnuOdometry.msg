# odometry message for custom ROS nodes
# every array is in float32 (important!)

uint64 timestamp	                # time since system start (microseconds)	

float32[3] position_ned		        # meters
float32[3] velocity_ned		        # m/s
float32[3] euler_angle_ned	        # rad
float32[3] body_angle_rate_ned      # rad/s
float32[3] inertial_angle_rate_ned  # rad/s

float32[3] position_enu             
float32[3] velocity_enu
float32[3] euler_angle_enu
float32[3] body_angle_rate_enu
float32[3] inertial_angle_rate_enu
float32[4] quaternion_enu
