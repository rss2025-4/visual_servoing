import math
import os
import time

import numpy as np
import pytest
import rclpy
import tf2_ros
import tf_transformations
from geometry_msgs.msg import Pose, PoseStamped, Transform
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion
from vs_msgs.msg import ConeLocation

from libracecar.sandbox import isolate
from libracecar.test_utils import proc_manager
from libracecar.transforms import pose_2d, se3_to_tf, tf_to_se3
from parking_controller.main import ParkingController


class _ok(BaseException):
    pass


class ParkingTest(Node):
    def __init__(self, cone_x: float, cone_y: float, time_limit: float):
        super().__init__("test_parking")
        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)

        # not needed, also race
        # self.pose_pub = self.create_publisher(Pose, "pose", 1)
        # self.pose_pub.publish(pose_2d(0.0, 0.0, 0.0))

        self.cone_pub = self.create_publisher(ConeLocation, "/relative_cone", 1)
        self.cone_pub_timer = self.create_timer(0.1, self.publish_cone)

        self.static_br = tf2_ros.StaticTransformBroadcaster(self)

        map_to_cone = Transform()
        map_to_cone.translation.x = cone_x
        map_to_cone.translation.y = cone_y

        self.static_br.sendTransform(
            se3_to_tf(tf_to_se3(map_to_cone), Time(), "map", "cone")
        )

        self.finish_timer = self.create_timer(time_limit, self.finish)

    def publish_cone(self):
        try:
            t = self.tfBuffer.lookup_transform("base_link", "cone", Time())
        except Exception as e:
            print("lookup_transform failed:", e)
            return

        relative_cone = ConeLocation()
        relative_cone.x_pos = t.transform.translation.x
        relative_cone.y_pos = t.transform.translation.y
        self.cone_pub.publish(relative_cone)

    def finish(self):
        t = self.tfBuffer.lookup_transform("base_link", "cone", Time())
        assert np.allclose(t.transform.translation.x, 0.75, rtol=0.0, atol=0.1)
        assert np.allclose(t.transform.translation.y, 0.0, rtol=0.0, atol=0.1)
        raise _ok()


@pytest.mark.parametrize(
    "cone_x,cone_y,time_limit",
    [
        (0.0, 0.0, 5.0),
        (0.0, 0.5, 5.0),
        (0.0, -0.5, 5.0),
        (0.5, -0.5, 5.0),
        (-0.5, -0.5, 8.0),
        (5.0, -0.5, 10.0),
        (-5.0, -0.5, 15.0),
    ],
)
@isolate
def test_parking(cone_x: float, cone_y: float, time_limit: float):
    procs = proc_manager.new()
    # procs.popen(["rviz2"])
    procs.ros_launch("racecar_simulator", "simulate.launch.xml")

    procs.ros_node_subproc(ParkingController)
    procs.ros_node_thread(ParkingTest, cone_x, cone_y, time_limit)

    try:
        procs.spin()
    except _ok:
        return
