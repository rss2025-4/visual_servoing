import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
import os

from rosbags.typesys import get_types_from_msg
from time import strftime, localtime

CONE_LOCATION_MSG = """
float32 x_pos
float32 y_pos       
"""
# from rosbags.typesys.types import vs_msgs__msg__ConeLocation as ConeLocation
## TODO add bagpath here
# bagpath = Path("/Users/aa/rosbags/rosbags/rosbags_lab3/rosbag_point5_speed_point5_dist")
# bagpath = Path(
#     "/Users/aa/rosbags/rosbags/rosbags_lab3/rosbag_onepointo_speed_point5_dist_left_withscore_2"
# )
# bagpath = Path("/Users/aa/rosbags/lab4_tests/lab4_line_5")
file = "lab4_line_3"  # "line_0316" #
bagpath = Path(f"/Users/aa/rosbags/lab4_tests/{file}")


typestore = get_typestore(Stores.ROS2_HUMBLE)  # ros humble


# register_types(get_types_from_msg(
#         ConeLocation, 'vs_msgs/msg/ConeLocation'))
typestore.register(get_types_from_msg(CONE_LOCATION_MSG, "vs_msgs/msg/ConeLocation"))
ConeLocation = typestore.types["vs_msgs/msg/ConeLocation"]


print(os.path.exists(bagpath))  # make sure path exists

# setpoint = 0.5  # distance setpoint


def get_data_from_bag(bagpath):
    """
    Given a topic name, reads in all the topics from the ROS bag and appends them to lists for graphing.
    """
    print("getting bag data", bagpath)

    x_error = []
    y_error = []
    dist_to_cone = []
    heading_angle = []
    time_arr = []

    topic = "/relative_cone"
    # use rosbags AnyReader to read bag at a certain path
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            relative_x = msg.x_pos
            relative_y = msg.y_pos
            angle = np.arctan2(relative_y, relative_x)
            heading_angle.append(angle)
            dist = ((relative_y**2) + (relative_x**2)) ** (1 / 2)
            time = timestamp
            dist_to_cone.append(dist)
            x_error.append(relative_x)
            y_error.append(relative_y)
            time_arr.append(time * 1e-9)

    # score_topic = "/score_metric"
    # with AnyReader([bagpath], default_typestore=typestore) as reader:
    #     connections = [x for x in reader.connections if x.topic == score_topic]
    #     for connection, timestamp, rawdata in reader.messages(connections=connections):
    #         msg = reader.deserialize(rawdata, connection.msgtype)
    #         score = msg.data
    #         time = timestamp
    #         score_arr.append(score)
    #         time_score_arr.append(time)

    return (
        np.array(time_arr),
        np.array(x_error),
        np.array(y_error),
        np.array(dist_to_cone),
        np.array(heading_angle),
    )


def plot_dist_vs_time(
    time_arr, x_error, y_error, dist_to_cone, heading_angle, save_path=None
):
    time_arr = time_arr - time_arr[0]
    print("Time", strftime("%Y-%m-%d %H:%M:%S", localtime(time_arr[0])))
    # # plot horizontal line
    # plt.axhline(y=setpoint, color="r", linestyle="-")

    # # plot time_arr against dist_to_wall_arr
    # # plt.plot(time_arr, dist_to_wall_arr, label="dist to wall")

    # # # plot error
    # # plt.plot(time_arr, dist_to_wall_arr-setpoint, label = "error")

    # # plot score
    # plt.plot(time_arr, dist_to_cone, label="Dist to Cone")
    # Create figure and primary y-axis (ax1)
    fig, ax1 = plt.subplots()

    # Plot horizontal line on ax1
    des_dist_to_cone = 0.05
    ax1.axhline(
        y=des_dist_to_cone,
        color="#ff8733",
        linestyle="--",
        label="Desired Distance to Cone (m)",
    )

    # Plot distance to wall on ax1
    # ax1.plot(time_arr, dist_to_cone, label="Distance to Cone (m)", color="b")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Distance (m)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Create secondary y-axis (ax2) for score

    ax1.plot(time_arr, x_error, "r-", label="x error")  # Green solid line for score

    ax1.plot(time_arr, y_error, "g-", label="y error")  # Green solid line for score
    ax1.plot(time_arr, dist_to_cone, label="Distance to Cone (m)", color="b")

    dist_error = dist_to_cone - des_dist_to_cone
    ax1.set_ylim([-8, 5])
    # ax2 = ax1.twinx()
    # ax2.plot(time_arr, dist_error, 'k-', label="Distance Error")  # Green solid line for score
    # ax2.set_ylabel("Distance Error", color="k")
    # ax2.tick_params(axis='y', labelcolor="k")

    ax2 = ax1.twinx()
    des_angle_to_cone = 0
    ax2.axhline(
        y=des_angle_to_cone,
        color="#808080",
        linestyle="--",
        label="Desired Angle to Cone (deg)",
    )
    ax2.plot(
        time_arr, heading_angle, "k-", label="Angle (deg)"
    )  # Green solid line for score
    ax2.set_ylabel("Angle (deg)", color="k")
    ax2.tick_params(axis="y", labelcolor="k")
    ax2.set_ylim([-3, 6])

    # ax2.set_xlim([10, 28])
    ax2.set_xlim([0, 55])

    # Handle legends separately

    ax1.legend(loc="upper left")

    plt.tight_layout()
    fig.subplots_adjust(top=0.85)
    ax2.legend(loc="lower right")
    # plt.legend()
    # show plot
    # plt.title(f"Cone Parking Controller Performance, Desired Distance: 0.75m")
    plt.title(f"Line Following Controller Performance, Desired Distance: 0.05m")

    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


def main():
    time_arr, x_error, y_error, dist_to_cone, heading_angle = get_data_from_bag(bagpath)
    save_path = f"/Users/aa/rosbags/lab4_tests/{file}_plot"
    plot_dist_vs_time(
        time_arr, x_error, y_error, dist_to_cone, heading_angle, save_path
    )


if __name__ == "__main__":
    main()
