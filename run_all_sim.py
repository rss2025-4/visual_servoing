#!/usr/bin/env python

from libracecar.sandbox import isolate
from libracecar.test_utils import proc_manager
from parking_controller.main import ParkingController


@isolate
def main():
    procs = proc_manager.new()
    procs.popen(["rviz2"])
    procs.ros_launch("racecar_simulator", "simulate.launch.xml")
    procs.ros_run("visual_servoing", "cone_sim_marker")

    procs.ros_node_subproc(ParkingController)
    procs.spin()


if __name__ == "__main__":
    main()
