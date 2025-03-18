import math
import time

import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped
from chex import dataclass


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


class metric_tracker:

    def __init__(self):
        self.start_time = time.time()
        self.first_drive_time: float | None = None
        self.prev_time = self.start_time

        self.distance_traveled = 0.0
        self.time_rest = 0.0

        self.last_drive = AckermannDriveStamped()

    @property
    def since_first_drive(self):
        if self.first_drive_time is None:
            return 0.0
        return time.time() - self.first_drive_time

    def on_drive(self, msg: AckermannDriveStamped):
        self.last_drive = msg
        if self.first_drive_time is None:
            self.first_drive_time = time.time()

    def tick(self):
        cur_time = time.time()
        time_passed = cur_time - self.prev_time

        self.distance_traveled += abs(self.last_drive.drive.speed) * time_passed
        if self.last_drive.drive.speed == 0.0:
            self.time_rest += time_passed
        else:
            self.time_rest = 0.0

    def finish(self, x_err: float, y_err: float):
        return parking_metrics(
            time_took=max(self.since_first_drive - self.time_rest, 0.0),
            final_error_x=x_err,
            final_error_y=y_err,
            final_error=math.sqrt(x_err**2 + y_err**2),
            distance_traveled=self.distance_traveled,
        )
