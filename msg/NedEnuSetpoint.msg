uint64 timestamp		# time since system start (microseconds)

bool is_enu
float32[3] position_setpoint		    # meters
float32[3] velocity_setpoint		    # m/s
float32[3] euler_angle_setpoint		    # rad

bool is_inertial                        # is the angular rate in inertial or body frame
float32[3] euler_angle_rate_setpoint	# rad/s