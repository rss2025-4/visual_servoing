import math
import random
import time

import tf2_ros
from geometry_msgs.msg import Transform
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation

from libracecar.sandbox import isolate
from libracecar.test_utils import proc_manager
from libracecar.transforms import se3_to_tf, tf_to_se3
from parking_controller.main import ParkingController, parkingcontroller_config


class ParkingTest(Node):
    def __init__(self):
        super().__init__("test_parking")

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)

        self.cone_pub = self.create_publisher(ConeLocation, "/relative_cone", 1)
        self._update_timer = self.create_timer(0.02, self.update_target)

        self.cone_marker_pub = self.create_publisher(Marker, "/cone_marker", 1)

        self.start_time = time.time()

    def update_target(self):
        ang = (time.time() - self.start_time) / 4

        cone_x = math.cos(ang) * 3 + random.gauss(0, 1) / 10
        cone_y = math.sin(ang) * 3 + random.gauss(0, 1) / 10

        # cone_x = (time.time() - self.start_time) / 2 + random.gauss(0, 1) / 10
        # cone_y = random.gauss(0, 1) / 10

        map_to_cone = Transform()
        map_to_cone.translation.x = cone_x
        map_to_cone.translation.y = cone_y

        map_to_cone_mat = tf_to_se3(map_to_cone)
        self.br.sendTransform(
            [se3_to_tf(map_to_cone_mat, self.get_clock().now(), "map", "cone")]
        )
        self.draw_marker(cone_x, cone_y)

        try:
            base_link_to_cone = self.tfBuffer.lookup_transform(
                "base_link", "cone", Time()
            )
        except Exception as e:
            print("(update_target) lookup_transform failed:", e)
            return

        relative_cone = ConeLocation()
        relative_cone.x_pos = base_link_to_cone.transform.translation.x
        relative_cone.y_pos = base_link_to_cone.transform.translation.y
        self.cone_pub.publish(relative_cone)

    def draw_marker(self, x: float, y: float):
        # modified from visual_servoing/visual_servoing/visual_servoing/cone_sim_marker.py
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        self.cone_marker_pub.publish(marker)


@isolate
def test_moving():
    procs = proc_manager.new()
    procs.popen(["rviz2"])
    procs.ros_launch("racecar_simulator", "simulate.launch.xml")

    procs.ros_node_subproc(ParkingController, parkingcontroller_config(0.05))
    procs.ros_node_thread(ParkingTest)

    procs.spin()
