import itertools
import sys
from sweeppy import Sweep


class LIDAR():

  def __init__(self):
    self.lidar_sensor = "/dev/ttyUSB0"
    self.sweep = None
    self.enable_scanning = False

  def connect_to_lidar(self):
    self.sweep = Sweep(self.lidar_sensor)
    self.sweep.__enter__()
    
    speed = self.sweep.get_motor_speed()
    rate = self.sweep.get_sample_rate()

    print('Motor Speed: {} Hz'.format(speed))
    print('Sample Rate: {} Hz'.format(rate))

  def get_lidar_data(self):
    # Starts scanning as soon as the motor is ready
    self.sweep.start_scanning()
    
    lidar_data = []

    # get_scans is coroutine-based generator lazily returning scans ad infinitum
    for scan in itertools.islice(self.sweep.get_scans(), 1):
      pass

    return lidar_data
    
  def shutdown(self):
    self.sweep.__exit__()

