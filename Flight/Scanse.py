import itertools
import sys
import math

from sweeppy import Sweep


class LIDAR():
  MAX_SAFE_DISTANCE = 300.0
  QUADRANT_SIZE = 45.0

  def __init__(self):
    self.lidar_sensor = "/dev/serial/by-id/usb-FTDI_FT230X_Basic_UART_DO00867Q-if00-port0"
    self.sweep = None

  def connect_to_lidar(self):
    self.sweep.__enter__()
    # Starts scanning as soon as the motor is ready
    self.sweep.start_scanning()

    self.sweep.set_motor_speed(2)
    self.sweep.set_sample_rate(1000)

    speed = self.sweep.get_motor_speed()
    rate = self.sweep.get_sample_rate()

    self.sweep.start_scanning()

  def get_lidar_data(self):
    # Starts scanning as soon as the motor is ready
    lidar_data = []

    # get_scans is coroutine-based generator lazily returning scans ad infinitum
    for scan in itertools.islice(self.sweep.get_scans(), 1):
      for sample in scan.samples:
        distance = sample.distance
        angle_deg = (sample.angle / 1000.0) % 360.0
        angle_rad = math.radians(sample.angle / 1000.0)
        # x = math.cos(angle_rad) * distance
        # y = math.sin(angle_rad) * distance
        if distance < self.MAX_SAFE_DISTANCE:
          lidar_data.append([distance, ((angle_deg % 360.0) // self.QUADRANT_SIZE) ])

    return lidar_data

  def shutdown(self):
    self.sweep.stop_scanning()
    self.sweep.__exit__()
