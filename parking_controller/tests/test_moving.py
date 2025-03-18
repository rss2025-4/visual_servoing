import math
import os
import random
import time

import numpy as np
import pytest
import rclpy
import tf2_ros
import tf_transformations
from ackermann_msgs.msg import AckermannDriveStamped
from chex import dataclass
from geometry_msgs.msg import Pose, PoseStamped, Transform
from rclpy.node import Node
from rclpy.time import Duration, Time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation

from libracecar.sandbox import isolate
from libracecar.test_utils import proc_manager
from libracecar.transforms import pose_2d, se3_to_tf, tf_to_se3
from parking_controller.main import ParkingController, parkingcontroller_config


@dataclass
class parking_metrics:
    time_took: float
    final_error_x: float
    final_error_y: float
    final_error: float
    distance_traveled: float

    def check(self):
        assert np.allclose(self.final_error_x, 0.0, rtol=0.0, atol=0.1)
        assert np.allclose(self.final_error_y, 0.0, rtol=0.0, atol=0.1)


metric_dict = dict[tuple[float, float], parking_metrics]


@pytest.fixture
def record_metric(request: pytest.FixtureRequest):
    return request.config._record_metric_storage  # type: ignore


class _ok(BaseException):
    def __init__(self, metrics: parking_metrics):
        self.metrics = metrics


class ParkingTest(Node):
    def __init__(self):
        super().__init__("test_parking")

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)

        # not needed, also race
        # self.pose_pub = self.create_publisher(Pose, "pose", 1)
        # self.pose_pub.publish(pose_2d(0.0, 0.0, 0.0))

        self.cone_pub = self.create_publisher(ConeLocation, "/relative_cone", 1)
        self._update_timer = self.create_timer(0.02, self.update_target)

        self.cone_marker_pub = self.create_publisher(Marker, "/cone_marker", 1)

        # self.static_br = tf2_ros.StaticTransformBroadcaster(self)

        self.last_drive = AckermannDriveStamped()
        self.create_subscription(
            AckermannDriveStamped, "/drive", self.drive_callback, 10
        )

        self.start_time = time.time()
        self.first_drive_time: float | None = None
        self.prev_time = self.start_time
        # self._metrics_timer = self.create_timer(0.001, self.update_metrics)

        self.distance_traveled = 0.0
        self.time_rest = 0.0

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

    def drive_callback(self, msg: AckermannDriveStamped):
        self.last_drive = msg
        if self.first_drive_time is None:
            self.first_drive_time = time.time()

    def update_metrics(self):
        cur_time = time.time()
        time_passed = cur_time - self.prev_time

        self.distance_traveled += self.last_drive.drive.speed * time_passed
        if self.last_drive.drive.speed == 0.0:
            self.time_rest += time_passed
        else:
            self.time_rest = 0.0

        if (
            self.first_drive_time is not None
            and cur_time - self.first_drive_time > 0.5
            and self.time_rest > 0.5
        ):
            t = self.tfBuffer.lookup_transform("base_link", "cone", Time())
            x_err = t.transform.translation.x - 0.75
            y_err = t.transform.translation.y

            metrics = parking_metrics(
                time_took=max(cur_time - self.first_drive_time - self.time_rest, 0.0),
                final_error_x=x_err,
                final_error_y=y_err,
                final_error=math.sqrt(x_err**2 + y_err**2),
                distance_traveled=self.distance_traveled,
            )
            raise _ok(metrics)

            return

        self.prev_time = cur_time


cone_pos = []
for x in [-2.0, -1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    # for x in [0.5]:
    for y in [-0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
        # for y in [0.2]:
        cone_pos.append((x, y))


# @pytest.mark.parametrize(
#     "cone_x,cone_y",
#     # [
#     #     (0.75, 0.0),
#     #     # (0.0, 0.5),
#     #     # (0.0, -0.5),
#     #     # (0.5, -0.5),
#     #     # (-0.5, -0.5),
#     #     # (5.0, -0.5),
#     #     # (-5.0, -0.5),
#     # ],
#     cone_pos,
# )
# @pytest.mark.timeout(20)
# def test_parking(cone_x: float, cone_y: float, record_metric: metric_dict):
#     res = run_test_parking(cone_x, cone_y)
#     record_metric[(cone_x, cone_y)] = res


@isolate
def test_moving() -> parking_metrics:
    procs = proc_manager.new()
    procs.popen(["rviz2"])
    procs.ros_launch("racecar_simulator", "simulate.launch.xml")

    procs.ros_node_subproc(ParkingController, parkingcontroller_config(0.05))
    procs.ros_node_thread(ParkingTest)

    try:
        procs.spin()
    except _ok as e:
        metrics = e.metrics
        metrics.check()
        return metrics
