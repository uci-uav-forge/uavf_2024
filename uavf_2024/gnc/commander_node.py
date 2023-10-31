import std_msgs.msg
import mavros_msgs.msg
import mavros_msgs.srv
import rclpy
import rclpy.node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import sensor_msgs.msg
import uavf_2024.srv

class CommanderNode(rclpy.node.Node):
    # Manages subscriptions to ROS topics and services necessary for the main GNC node. 
    def __init__(self):
        super().__init__('uavf_commander_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth = 1
        )

        self.global_pos_sub = self.create_subscription(
            sensor_msgs.msg.NavSatFix,
            'ap/geopose/filtered',
            self.global_pos_cb,
            qos_profile)
        self.got_pos = False

        self.arm_client = self.create_client(mavros_msgs.srv.CommandBool, 'mavros/cmd/arming')
        
        self.mode_client = self.create_client(mavros_msgs.srv.SetMode, 'mavros/set_mode')

        self.takeoff_client = self.create_client(mavros_msgs.srv.CommandTOL, 'mavros/cmd/takeoff')

        self.waypoints_client = self.create_client(mavros_msgs.srv.WaypointPush, 'mavros/mission/push')

        self.take_picture_client = self.create_client(uavf_2024.srv.TakePicture, 'uavf_2024/take_picture')
    
    def global_pos_cb(self, global_pos):
        self.got_pos = True
        self.last_pos = global_pos