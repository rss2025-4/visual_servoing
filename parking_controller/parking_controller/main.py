import math
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDrive, AckermannDriveStamped
from jax import tree_util as jtu
from rclpy.node import Node
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation, ParkingError

from libracecar.plot import plot_ctx
from libracecar.specs import path
from libracecar.utils import jit
from parking_controller.core import compute, compute_score, patheval

np.set_printoptions(precision=10, suppress=True)


class ParkingController(Node):
    """
    A controller for parking in front of a cone.
    Listens for a relative cone location and publishes control commands.
    Can be used in the simulator and on the real robot.
    """

    def __init__(self):
        super().__init__("parking_controller")

        self.declare_parameter("drive_topic", "/vesc/high_level/input/nav_0")
        DRIVE_TOPIC = self.get_parameter(
            "drive_topic"
        ).value  # set in launch file; different for simulator vs racecar
        assert isinstance(DRIVE_TOPIC, str)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, DRIVE_TOPIC, 10)
        self.error_pub = self.create_publisher(ParkingError, "/parking_error", 10)

        self.visualization_pub = self.create_publisher(Marker, "/visualization", 10)

        self.create_subscription(
            ConeLocation, "/relative_cone", self.relative_cone_callback, 1
        )

        self.parking_distance = 0.05  # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0

        self.get_logger().info("Parking Controller Initialized")

    def visualize(self, ctx: plot_ctx):
        m = Marker()
        m.type = Marker.POINTS
        m.header.frame_id = "/laser"
        m.scale.x = 0.1
        m.scale.y = 0.1
        ctx.execute(m)
        self.visualization_pub.publish(m)

    def plan(self, msg: ConeLocation) -> tuple[path, plot_ctx]:
        scorer = compute_score(
            parking_distance=self.parking_distance,
            relative_x=min(msg.x_pos, 0.1),
            relative_y=msg.y_pos,
        )
        return compute(scorer)

    def relative_cone_callback(self, msg: ConeLocation):
        print("relative_cone_callback!")
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos

        print("cone", self.relative_x, self.relative_y)

        _start_time = time.time()
        plan, ctx = self.plan(msg)
        print("Elapsed time", time.time() - _start_time)

        # print(plan)
        # print(ctx.points.x)
        # print(ctx.points.y)

        self.visualize(ctx)

        # print(plan)
        # assert False

        seg = None
        for x in plan:
            if abs(float(x.length)) > 0.01:
                seg = x
                break

        if seg is None:
            speed = 0.0
            ang = 0.0
        else:

            ang = float(seg.angle)
            if math.isnan(ang):
                print("nan!")
                return
            speed = float(seg.length)

            min_speed = 0.2
            if abs(speed) < min_speed:
                speed = min_speed if speed > 0 else -min_speed

            # speed = min(max(speed, -1.0), 1.0)
            speed = min(max(speed, -1.0), 1.0)
            # if math.isnan(ang):
            #     print("nan!")
            #     return

        print(ang, speed, flush=True)
        # assert False

        # print(self.plan())

        drive_cmd = AckermannDriveStamped()

        drive = AckermannDrive()
        drive.steering_angle = ang
        drive.speed = speed
        # drive.steering_angle = 0.0
        # drive.speed = 0.0

        drive_cmd.drive = drive

        self.drive_pub.publish(drive_cmd)
        self.error_publisher(msg)

    def error_publisher(self, msg: ConeLocation):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()
        x_pos = msg.x_pos
        y_pos = msg.y_pos
        error_msg.x_error = abs(self.parking_distance - msg.x_pos)
        error_msg.y_error = abs(msg.y_pos)
        error_msg.distance_error = error_msg.x_error**2 + error_msg.y_error**2

        self.error_pub.publish(error_msg)


def main(args=None):
    rclpy.init(args=args)
    pc = ParkingController()
    rclpy.spin(pc)
    rclpy.shutdown()
