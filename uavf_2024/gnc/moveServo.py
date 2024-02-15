from pymavlink import mavutil

# Connect to the flight controller
master = mavutil.mavlink_connection('udpin:0.0.0.0:14550')

# Wait for the heartbeat msg to find the system ID
master.wait_heartbeat()

# Command a servo to move
# param1: servo number, param2: PWM value (1000-2000)
master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
    0,  # Confirmation
    1,  # Param1: Servo number
    1200,     # Param2: PWM value
    0, 0, 0, 0, 0  # Unused parameters
)
