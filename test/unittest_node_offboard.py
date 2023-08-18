from px4_offboard_mpc.node_px4_interface import PX4_Interface

def main():
    '''
    Startup:
        send offboard control mode msg 10 times,
        engage offboard mode,
        arm

    The loop goes:
        get setpoint: next objective
        get state: pos_enu, vel_enu, angle_enu, ang_rate_enu
        mpc -> next_control_and_state(state, setpoint)
        set next state
    '''

    '''
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControlNode('offboard_control_example')
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()
    '''


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)