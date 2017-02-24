############################################
# This file contains a wrapper class
# for DroneKit related operations
# for our drone.
############################################
# Multi-Rotor Robot Design Team
# Missouri University of Science Technology
# Spring 2017
# Lucas Coon, Mark Raymond Jr.
# pylint: disable=C, F, I, R, W

from datetime import datetime, timedelta
from os import system
from time import sleep

import dronekit
import math
import time

class Attitude(dronekit.Attitude):

  def __init__(self, pitch, yaw, roll):
    self.x = pitch
    self.y = yaw
    self.z = roll
    self.pitch = math.radians(pitch)
    self.yaw = math.radians(yaw)
    self.roll = math.radians(roll)
    self.quaternion = self.get_quaternion()

  def get_quaternion(self):
    q = []

    t0 = math.cos(self.yaw * 0.5)
    t1 = math.sin(self.yaw * 0.5)
    t2 = math.cos(self.roll * 0.5)
    t3 = math.sin(self.roll * 0.5)
    t4 = math.cos(self.pitch * 0.5)
    t5 = math.sin(self.pitch * 0.5)

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    q.append(w)
    q.append(x)
    q.append(y)
    q.append(z)

    return q


class Standard_Attitudes(object):
  level = Attitude(0,0,0)
  forward = Attitude(-5,0,0) # -5 degrees.
  backward = Attitude(5,0,0)  # +5 degrees.
  left = Attitude(-5, 0, -5)
  right = Attitude(-5, 0, 5)
  
class Standard_Thrusts(object):
  hover = 0.45
  low_speed = 0.50
  med_speed = 0.55
  high_speed = 0.60

class Tower(object):
  SERIAL_PORT = "/dev/ttyS1"
  BAUD_RATE = 57600
  SIMULATOR = "127.0.0.1:14551"
  STANDARD_ATTITUDE_BIT_FLAGS = 0b11100000
  TURNING_ATTITUDE_BIT_FLAGS = 0b00000000
  STANDARD_THRUST_CHANGE = 0.05
  MAX_TURN_TIME = 5
  LAND_ALTITUDE = 0.5

  def __init__(self):
    self.start_time = 0
    self.last_thrust = 0
    self.last_attitude = None
    self.system_initialized = False
    self.vehicle_initialized = False

    self.vision_running = False
    self.avoidance_running = False
    self.stategy_running = False

    self.vehicle = None
    self.connected = False

    self.vehicle_busy = False
    self.vehicle_state = "UNKNOWN"

  def initialize_system(self):
    if not self.system_initialized:
      print("\nEnabling serial UART connection on " + self.SERIAL_PORT + " at " + self.BAUD_RATE + " baud...")
      return_code = system("stty -F " + self.SERIAL_PORT + " " + self.BAUD_RATE + " raw -echo -echoe -echok -crtscts")
      if return_code == 0:
        self.system_initialized = True
        print("\nSucessfully enabled UART connection.")

  def initialize_drone(self):
    if not self.vehicle_initialized:
      print("\nConnecting via " + self.SERIAL_PORT + " to flight controller...")
      # self.vehicle = dronekit.connect(self.SERIAL_PORT, baud=self.BAUD_RATE, wait_ready=True)
      self.vehicle = dronekit.connect(self.SIMULATOR, wait_ready=True)
      self.vehicle.mode = dronekit.VehicleMode("STABILIZE")
      self.vehicle_state = "GROUND_IDLE"
      self.connected = True
      self.vehicle_initialized = True
      self.start_time = time.time()
      self.last_attitude = [1, 0, 0, 0]
      print("\nSuccessfully connected to vehicle.")

  def arm_drone(self):
    while(not self.vehicle.armed):
      self.vehicle.armed = True

  def get_uptime(self):
    uptime = time.time() - self.start_time
    return uptime

  def set_angle_thrust(self, attitude, thrust):

    while self.vehicle.mode.name != "GUIDED_NOGPS":
      self.vehicle.mode = dronekit.VehicleMode("GUIDED_NOGPS")
    
    print("Building MAVLink message...")
    message = self.vehicle.message_factory.set_attitude_target_encode(
      0,                                 # Timestamp in milliseconds since system boot (not used).
      0,                                 # System ID
      0,                                 # Component ID
      self.STANDARD_ATTITUDE_BIT_FLAGS,       # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
      attitude.quaternion,               # Quaternions
      0,                                 # Body roll rate.
      0,                                 # Body pitch rate.
      0,                                 # Body yaw rate.
      thrust                             # Collective thrust, from 0-1.
    )
    self.vehicle.send_mavlink(message)
    self.vehicle.flush()
    self.last_attitude = attitude
    self.last_thrust = thrust
    print("Sent message.")

  def return_to_hover(self):
    self.vehicle_state = "HOVER"
    self.set_angle_thrust(Standard_Attitudes.level, Standard_Thrusts.hover)
    self.vehicle.mode = dronekit.VehicleMode("ALT HOLD")

  def takeoff(self, target_altitude, thrust):

    self.arm_drone()
    
    self.set_angle_thrust(Standard_Attitudes.level, thrust)

    while(self.vehicle.location.global_relative_frame.alt <= target_altitude):
      sleep(1)

    self.return_to_hover()
    print('Reached target altitude:{0:.2f}m'.format(self.vehicle.location.global_relative_frame.alt))

  def fly_for_time(self, duration, direction, target_velocity, should_hover_when_finished=False):
    duration = timedelta(seconds=duration)
    end_manuever = datetime.now() + duration
    self.set_angle_thrust(direction, self.last_thrust)
    while(end_manuever <= datetime.now()):
      if self.vehicle.velocity > target_velocity:
        self.set_angle_thrust(direction, self.last_thrust - self.STANDARD_THRUST_CHANGE)
      elif(self.vehicle.velocity < target_velocity):
        self.set_angle_thrust(direction, self.last_thrust + self.STANDARD_THRUST_CHANGE)
      sleep(1)
    
    if should_hover_when_finished:
     self.return_to_hover()

  def fly_for_time_thrust(self, duration, direction, thrust, should_hover_when_finished=False):
    duration = timedelta(seconds=duration)
    end_manuever = datetime.now() + duration
    self.set_angle_thrust(direction, thrust)
    while(end_manuever <= datetime.now()):
      sleep(1)

    if should_hover_when_finished:
     self.return_to_hover()

  def land(self):
    self.vehicle.mode = dronekit.VehicleMode("LAND")
    while(self.vehicle.location.global_relative_frame.alt <= self.LAND_ALTITUDE):
      sleep(1)
    else:
      self.vehicle_state = "GROUND_IDLE"

  def turn_for_time(self, direction, duration):
    if duration > self.MAX_TURN_TIME:
      return
      
    self.fly_for_time(1, Standard_Attitudes.forward, self.vehicle.velocity, False)

    print("Building MAVLink message...")

    self.vehicle.mode = dronekit.VehicleMode("ALT HOLD")

    message = self.vehicle.message_factory.set_attitude_target_encode(
      0,                                        # Timestamp in milliseconds since system boot (not used).
      0,                                        # System ID
      0,                                        # Component ID
      self.TURNING_ATTITUDE_BIT_FLAGS,          # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
      direction.quaternion,                     # Attitude quaternion.
      1,                                        # Body roll rate.
      0,                                        # Body pitch rate.
      1,                                        # Body yaw rate.
      Standard_Thrusts.low_speed                # Collective thrust, from 0-1.
    )

    self.vehicle.send_mavlink(message)
    self.vehicle.flush()

    print("Sent message.")

    sleep(duration)

    self.fly_for_time(1, self.last_attitude, self.vehicle.velocity, False)


