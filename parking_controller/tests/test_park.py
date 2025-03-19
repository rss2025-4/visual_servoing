import pytest
import tf2_ros
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Transform
from rclpy.node import Node
from rclpy.time import Time
from visualization_msgs.msg import Marker
from vs_msgs.msg import ConeLocation

from libracecar.sandbox import isolate
from libracecar.test_utils import proc_manager
from libracecar.transforms import se3_to_tf, tf_to_se3
from parking_controller.main import ParkingController, parkingcontroller_config
from parking_controller.metrics import metric_tracker, parking_metrics

metric_dict = dict[tuple[float, float], parking_metrics]


@pytest.fixture
def record_metric(request: pytest.FixtureRequest):
    return request.config._record_metric_storage  # type: ignore


class _ok(BaseException):
    def __init__(self, metrics: parking_metrics):
        self.metrics = metrics


class ParkingTest(Node):
    def __init__(self, cone_x: float, cone_y: float):
        super().__init__("test_parking")
        self.cone_x = cone_x
        self.cone_y = cone_y

        self.tfBuffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tfBuffer, self)

        # not needed, also race
        # self.pose_pub = self.create_publisher(Pose, "pose", 1)
        # self.pose_pub.publish(pose_2d(0.0, 0.0, 0.0))

        self.cone_pub = self.create_publisher(ConeLocation, "/relative_cone", 1)
        self.cone_pub_timer = self.create_timer(0.1, self.publish_cone)

        self.cone_marker_pub = self.create_publisher(Marker, "/cone_marker", 1)

        self.static_br = tf2_ros.StaticTransformBroadcaster(self)

        map_to_cone = Transform()
        map_to_cone.translation.x = cone_x
        map_to_cone.translation.y = cone_y

        self.static_br.sendTransform(
            se3_to_tf(tf_to_se3(map_to_cone), Time(), "map", "cone")
        )

        self.last_drive = AckermannDriveStamped()
        self.create_subscription(
            AckermannDriveStamped, "/drive", self.drive_callback, 10
        )

        self.metrics = metric_tracker()
        self._tick_timer = self.create_timer(0.001, self.tick)

    def draw_marker(self):
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
        marker.pose.position.x = self.cone_x
        marker.pose.position.y = self.cone_y
        self.cone_marker_pub.publish(marker)

    def drive_callback(self, msg: AckermannDriveStamped):
        self.metrics.on_drive(msg)
        self.draw_marker()

    def tick(self):
        self.metrics.tick()
        if (
            self.metrics.first_drive_time is not None
            and self.metrics.since_first_drive > 0.5
            and self.metrics.time_rest > 0.5
        ):
            t = self.tfBuffer.lookup_transform("base_link", "cone", Time())
            metrics = self.metrics.finish(
                x_err=t.transform.translation.x - 0.75, y_err=t.transform.translation.y
            )
            raise _ok(metrics)

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


cone_pos = []
for x in [-2.0, -1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    # for x in [0.5]:
    for y in [-0.1, 0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]:
        # for y in [0.2]:
        cone_pos.append((x, y))


@pytest.mark.parametrize(
    "cone_x,cone_y",
<<<<<<< HEAD
    # [
    #     (0.75, 0.0),
    #     # (0.0, 0.5),
    #     # (0.0, -0.5),
    #     # (0.5, -0.5),
    #     # (-0.5, -0.5),
    #     # (5.0, -0.5),
    #     # (-5.0, -0.5),
    # ],
    cone_pos,
=======
    [
        (0.75, 0.0),
        (0.0, 0.5),
        (0.0, -0.5),
        (0.5, -0.5),
        (-0.5, -0.5),
        (5.0, -0.5),
        (-5.0, -0.5),
    ],
    # cone_pos,
>>>>>>> e89278dd36f83d74169668debc831ba2f4de4f01
)
@pytest.mark.timeout(20)
def test_parking(cone_x: float, cone_y: float, record_metric: metric_dict):
    res = run_test_parking(cone_x, cone_y)
    record_metric[(cone_x, cone_y)] = res


@isolate
def run_test_parking(cone_x: float, cone_y: float) -> parking_metrics:
    procs = proc_manager.new()
    procs.popen(["rviz2"])
    procs.ros_launch("racecar_simulator", "simulate.launch.xml")

<<<<<<< HEAD
    procs.ros_node_subproc(ParkingController)
=======
    procs.ros_node_subproc(ParkingController, parkingcontroller_config())
>>>>>>> e89278dd36f83d74169668debc831ba2f4de4f01
    procs.ros_node_thread(ParkingTest, cone_x, cone_y)

    try:
        procs.spin()
    except _ok as e:
        metrics = e.metrics
        metrics.check()
        return metrics
