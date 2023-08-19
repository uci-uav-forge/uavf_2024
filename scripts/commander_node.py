import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_offboard_mpc.sqp_nlmpc import derive_quad_dynamics, SQP_NLMPC
from uavf_msgs.msg import NedEnuOdometry, NedEnuSetpoint, CommanderOutput


class CommanderNode(Node):
    ''' Received input from every ongoing process
    '''

    def __init__(self, is_ENU:bool, time_step:float):
        super().__init__('commander_node')

        # Configure QoS profile according to PX4
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )


def main(args=None):
    print('Starting commander node...')
    rclpy.init(args=args)
    node = CommanderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)