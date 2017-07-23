import itertools
import sys
from sweeppy import Sweep


class LIDAR():
  MAX_SAFE_DISTANCE = 300.0
  ANGLE_DIVISOR = 45.0

  def __init__(self):
    self.lidar_sensor = "/dev/ttyUSB0"
    self.sweep = None
    # self.enable_scanning = False

  def connect_to_lidar(self):
    self.sweep = Sweep(self.lidar_sensor)
    self.sweep.__enter__()

    self.sweep.set_motor_speed(2)
    self.sweep.set_sample_rate(1000)

    speed = self.sweep.get_motor_speed()
    rate = self.sweep.get_sample_rate()

    self.sweep.start_scanning()

  def get_lidar_data(self):
    # Starts scanning as soon as the motor is ready
    lidar_data = []

    # get_scans is coroutine-based generator lazily returning scans ad infinitum
    for scan in itertools.islice(sweep.get_scans(), 1):
        for sample in scan.samples:
          distance = sample.distance
          angle = math.radians(sample.angle / 1000.0) % 360.0
        #   x = math.cos(angle) * distance
        #   y = math.sin(angle) * distance
          if distance < MAX_SAFE_DISTANCE:
            lidar_data.append([distance, (angle%360.0)//ANGLE_DIVISOR])

    return lidar_data

  def shutdown(self):
    self.sweep.stop_scanning()
    self.sweep.__exit__()
