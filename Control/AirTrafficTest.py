import unittest
from time import sleep
from AirTrafficControl import Tower, Attitude


class AttitudeTest(unittest.TestCase):
  def test_zero_rotation_quarternion(self):
    ZERO_ROTATION = [1, 0, 0, 0]
    a = Attitude(0, 0, 0)
    self.assertEqual(a.quaternion, ZERO_ROTATION)

class TowerTest(unittest.TestCase):
  def test_initialize_drone(self):
    t = Tower()
    t.initialize_drone()
    t.vehicle.close()

  def test_uptime_drone(self):
    WAIT_TIME = 5
    DECIMAL_PLACES = 2
    t = Tower()
    t.initialize_drone()
    sleep(WAIT_TIME)
    reported_uptime = t.get_uptime()
    self.assertAlmostEqual(WAIT_TIME, reported_uptime, DECIMAL_PLACES)
    t.vehicle.close()
