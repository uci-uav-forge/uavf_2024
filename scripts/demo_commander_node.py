import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import sensor_msgs.msg
from threading import Thread
from uavf_2024.msg import GpsAltitudePosition

class CommanderNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('uavf_commander_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )

        self.global_pos_sub = self.create_subscription(
            sensor_msgs.msg.NavSatFix,
            'mavros/global_position/global',
            self.global_pos_cb,
            qos_profile)

        self.arm_client = self.create_client(mavros_msgs.srv.CommandBool, 'mavros/cmd/arming')
        
        self.mode_client = self.create_client(mavros_msgs.srv.SetMode, 'mavros/set_mode')

        self.waypoints_client = self.create_client(mavros_msgs.srv.WaypointPush, 'mavros/mission/push')
    
    def global_pos_cb(self, global_pos):
        print(global_pos)

if __name__ == '__main__':
    rclpy.init()
    node = CommanderNode()

    spinner = Thread(target = rclpy.spin, args = (node,))
    
    request = mavros_msgs.srv.WaypointPush.Request()
    request.start_index = 0
    request.waypoints = []

    spinner.start()

    res = node.arm_client.call(mavros_msgs.srv.CommandBool.Request(value = True))
    print(res.success)


    node.destroy_node()
    rclpy.shutdown()