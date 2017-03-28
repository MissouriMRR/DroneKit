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
from copy import deepcopy
from Collision import Sonar

import RPi.GPIO as GPIO
import dronekit
import math
import os
import time
import threading

class DroneAttitude():

  def __init__(self, roll, pitch, yaw):
    self.pitch_deg = pitch
    self.yaw_deg = yaw
    self.roll_deg = roll
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


class StandardAttitudes(object):
  level = DroneAttitude(0,0,0)
  forward = DroneAttitude(0,-5,0)
  backward = DroneAttitude(0,-5,0)
  left = DroneAttitude(-5, 0, 0)
  right = DroneAttitude(5, 0, 0)

class StandardThrusts(object):
  none = 0.00
  hover = 0.50
  takeoff = 0.65
  full = 1.00

class VehicleStates(object):
  hover = "HOVER"
  flying = "FLYING"
  takeoff = "TAKEOFF"
  unknown = "UNKNOWN"
  avoidance = "AVOIDANCE"
  landed = "LANDED"

class Tower(object):
  SERIAL_PORT = "/dev/ttyS1"
  BAUD_RATE = 57600
  SIMULATOR = "127.0.0.1:14551"
  USB = "/dev/cu.usbmodem1"
  STANDARD_ATTITUDE_BIT_FLAGS = 0b00111111
  FLIP_ATTITUDE_BIT_FLAGS = 0b00111000
  STANDARD_THRUST_CHANGE = 0.05
  MAX_TURN_TIME = 5
  LAND_ALTITUDE = 0.5
  TURN_START_VELOCITY = 3
  TURN_RADIUS = 0.5 # Meters
  ANGLE_INCREMENT = 1.1
  ANGLE_DECREMENT = 0.9
  MESSAGE_SLEEP_TIME = 1
  HOVER_CIRCLE_RADIUS = 0.25
  HOVER_ADJUST_DEG = 3
  HOVER_MAX_DRIFT_TIME = 2.5
  DRIFT_CORRECT_THRESHOLD = 0.05

  def __init__(self):
    self.start_time = 0
    self.vehicle_initialized = False

    self.vision_running = False
    self.avoidance_running = False
    self.stategy_running = False

    self.vehicle = None
    self.connected = False

    self.DESIRED_ATTITUDE = StandardAttitudes.level
    self.LAST_ATTITUDE = StandardAttitudes.level
    self.DESIRED_THRUST = StandardThrusts.none
    self.LAST_THRUST = StandardThrusts.none
    self.STATE = VehicleStates.unknown

  def enable_uart(self):
      print("\nEnabling serial UART connection on " + self.SERIAL_PORT + " at " + str(self.BAUD_RATE) + " baud...")
      return_code = system("stty -F " + self.SERIAL_PORT + " " + str(self.BAUD_RATE) + " raw -echo -echoe -echok -crtscts")
      if(return_code == 0):
        print("\nSucessfully enabled UART connection.")

  def initialize(self):
    if(not self.vehicle_initialized):
      print("\nConnecting via " + self.SERIAL_PORT + " to PixHawk...")
      # self.vehicle = dronekit.connect(self.SERIAL_PORT, baud=self.BAUD_RATE, wait_ready=True)
      self.vehicle = dronekit.connect(self.SIMULATOR, wait_ready=True)

      if not self.vehicle:
        print("\nUnable to connect to vehicle.")
        return

      self.vehicle.mode = dronekit.VehicleMode("STABILIZE")
      self.STATE = VehicleStates.landed
      self.connected = True
      self.vehicle_initialized = True
      self.controller = MessageController(self)
      self.failsafes = FailsafeController(self)
      self.controller.start()
      self.failsafes.start()
      self.start_time = time.time()

      print("\nSuccessfully connected to vehicle.")

  def shutdown(self):
    self.controller.join()
    self.vehicle.close()
    self.connected = False
    self.vehicle_initialized = False
    self.start_time = 0

  def arm_drone(self):
    self.vehicle.armed = True
    while(not self.vehicle.armed):
      sleep(1)

  def disarm_drone(self):
    self.vehicle.armed = False
    while(self.vehicle.armed):
      sleep(1)

  def get_uptime(self):
    uptime = time.time() - self.start_time
    return uptime

  def constrain(self, x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

  def switch_control(func):
    def check_flight_controller_and_mode(self, *args, **kwargs):
        if not self.vehicle_initialized:
          self.initialize()
        if not self.controller:
          self.controller = MessageController(self)
          self.controller.start()
        if not self.failsafes:
          self.failsafes = FailsafeController(self)
          self.failsafes.start()
        if self.vehicle.mode.name != "GUIDED_NOGPS":
          self.vehicle.mode = dronekit.VehicleMode("GUIDED_NOGPS")
          while(self.vehicle.mode.name != "GUIDED_NOGPS"):
            sleep(1)

        return func(self, *args, **kwargs)

    return check_flight_controller_and_mode

  @switch_control
  def hover(self):
    sleep(self.HOVER_MAX_DRIFT_TIME)

    drift_distance_x = self.vehicle.velocity[0] * self.HOVER_MAX_DRIFT_TIME
    drift_distance_y = self.vehicle.velocity[1] * self.HOVER_MAX_DRIFT_TIME

    drift_distance = math.hypot(drift_distance_x, drift_distance_y)

    if(drift_distance > self.HOVER_CIRCLE_RADIUS):
      adjust_attitude = deepcopy(StandardAttitudes.level)

      if(drift_distance_x < 0):
        adjust_attitude = DroneAttitude(self.HOVER_ADJUST_DEG, adjust_attitude.pitch_deg, adjust_attitude.yaw_deg)
      elif(drift_distance_x > 0):
        adjust_attitude = DroneAttitude(-self.HOVER_ADJUST_DEG, adjust_attitude.pitch_deg, adjust_attitude.yaw_deg)
      if(drift_distance_y < 0):
        adjust_attitude = DroneAttitude(adjust_attitude.roll_deg, -self.HOVER_ADJUST_DEG, adjust_attitude.yaw_deg)
      elif(drift_distance_y > 0):
        adjust_attitude = DroneAttitude(adjust_attitude.roll_deg, self.HOVER_ADJUST_DEG, adjust_attitude.yaw_deg)

      corrected_distance_x = 0
      corrected_distance_y = 0
      corrected_distance = 0

      while(self.STATE != VehicleStates.avoidance and (math.fabs(corrected_distance - drift_distance) > -self.DRIFT_CORRECT_THRESHOLD and math.fabs(corrected_distance - drift_distance) < self.DRIFT_CORRECT_THRESHOLD)):
        corrected_distance_x = self.vehicle.velocity[0]
        corrected_distance_y = self.vehicle.velocity[1]
        corrected_distance += math.hypot(corrected_distance_x, corrected_distance_y)

        self.DESIRED_ATTITUDE = adjust_attitude
        self.DESIRED_THRUST = deepcopy(StandardThrusts.hover)
        self.controller.send_angle_thrust(self.DESIRED_ATTITUDE, self.DESIRED_THRUST)
        print("\n Correcting: " + corrected_distance + " Drifted " + drift_distance)
        sleep(1)

    self.DESIRED_ATTITUDE = deepcopy(StandardAttitudes.level)
    self.DESIRED_THRUST = deepcopy(StandardThrusts.hover)

    self.controller.send_angle_thrust(self.DESIRED_ATTITUDE, self.DESIRED_THRUST)

  @switch_control
  def takeoff(self, target_altitude):

    self.STATE = VehicleStates.takeoff

    self.arm_drone()

    initial_alt = self.vehicle.location.global_relative_frame.alt

    while((self.vehicle.location.global_relative_frame.alt - initial_alt) < target_altitude):

      self.DESIRED_ATTITUDE = deepcopy(StandardAttitudes.level)
      self.DESIRED_THRUST = deepcopy(StandardThrusts.takeoff)

      sleep(0.1)

    print('Reached target altitude:{0:.2f}m'.format(self.vehicle.location.global_relative_frame.alt))

    self.STATE = VehicleStates.hover

  @switch_control
  def fly_for_time(self, duration, direction, target_velocity):

    end_manuever = datetime.now() + timedelta(seconds=duration)

    self.STATE = VehicleStates.flying

    self.DESIRED_ATTITUDE = deepcopy(direction)
    self.DESIRED_THRUST = deepcopy(StandardThrusts.full)

    print("\n DESIRED_AT_START " + str(self.DESIRED_ATTITUDE.pitch_deg))

    while(datetime.now() < end_manuever):

      print(self.vehicle.airspeed,)
      print(self.vehicle.velocity,)

      updated_attitude = deepcopy(self.DESIRED_ATTITUDE)

      if(self.vehicle.airspeed < target_velocity):
        updated_attitude.pitch_deg -= 1
      elif(self.vehicle.airspeed > target_velocity):
        updated_attitude.pitch_deg += 1
      else:
        updated_attitude.pitch_deg = direction.pitch_deg

      updated_attitude.pitch = math.radians(self.DESIRED_ATTITUDE.pitch_deg)
      updated_attitude.quaternion = self.DESIRED_ATTITUDE.get_quaternion()

      self.DESIRED_ATTITUDE = updated_attitude

      print(self.DESIRED_ATTITUDE.pitch_deg,)

      sleep(1)

    self.STATE = VehicleStates.hover

  def land(self):
    self.vehicle.mode = dronekit.VehicleMode("LAND")
    while((self.vehicle.location.global_relative_frame.alt) >= self.LAND_ALTITUDE):
      sleep(1)
    else:
      self.STATE = VehicleStates.landed

  @switch_control
  def do_circle_turn(self, desired_angle, direction, duration):
    if(duration > self.MAX_TURN_TIME):
      return

    desired_angle = math.radians(desired_angle)

    max_angle = desired_angle
    altitude_to_hold = self.vehicle.location.global_relative_frame.alt

    end_manuever = datetime.now() + timedelta(seconds=duration)

    self.fly_for_time(1, StandardAttitudes.forward, self.TURN_START_VELOCITY)

    while(datetime.now() < end_manuever):
      change_in_time = end_manuever - datetime.now()
      current_altitude = self.vehicle.location.global_relative_frame.alt

      roll_angle = max_angle * (math.cos(self.vehicle.airspeed * change_in_time.seconds) / self.TURN_RADIUS)
      pitch_angle = max_angle * (math.sin(self.vehicle.airspeed * change_in_time.seconds) / self.TURN_RADIUS)

      roll_angle = math.degrees(roll_angle)
      pitch_angle = math.degrees(pitch_angle)

      updated_attitude = DroneAttitude(roll_angle, pitch_angle, self.LAST_ATTITUDE.yaw_deg)

      self.DESIRED_ATTITUDE = updated_attitude
      self.DESIRED_THRUST = deepcopy(StandardThrusts.full)

      print("Sent message.")

      if(current_altitude > altitude_to_hold):
        max_angle = desired_angle * self.ANGLE_INCREMENT
      elif(current_altitude < altitude_to_hold):
        max_angle = desired_angle * self.ANGLE_DECREMENT

      sleep(1)

    self.fly_for_time(1, StandardAttitudes.forward, self.vehicle.airspeed)

  def check_sonar_sensors(self):
    sonar = Sonar.Sonar(2,3, "Main")
    print("%s Measured Distance = %.1f cm" % (sonar.getName(), sonar.getDistance()))
    pass

  def check_battery_voltage(self):
    pass

class FailsafeController(threading.Thread):

  def __init__(self, atc_instance):
    self.atc = atc_instance
    self.stoprequest = threading.Event()
    super(FailsafeController, self).__init__()

  def run(self):
    while not self.stoprequest.isSet():
      if self.atc.vehicle.armed and self.atc.STATE != VehicleStates.landed:
        self.atc.check_sonar_sensors()
        self.atc.check_battery_voltage()

  def join(self, timeout=None):
    if self.atc.vehicle.armed:
      if self.atc.STATE != VehicleStates.landed:
        self.atc.land()

    self.stoprequest.set()
    GPIO.cleanup()
    super(FailsafeController, self).join(timeout)

class MessageController(threading.Thread):

  def __init__(self, atc_instance):
    self.atc = atc_instance
    self.stoprequest = threading.Event()
    super(MessageController, self).__init__()

  def run(self):
    while not self.stoprequest.isSet():
      if self.atc.vehicle.armed and self.atc.STATE != VehicleStates.landed:
        self.set_angle_thrust()

  def join(self, timeout=None):
    self.stoprequest.set()
    super(MessageController, self).join(timeout)

  def send_angle_thrust(self, attitude, thrust):
    message = self.atc.vehicle.message_factory.set_attitude_target_encode(
    0,                                 # Timestamp in milliseconds since system boot (not used).
    0,                                 # System ID
    0,                                 # Component ID
    self.atc.STANDARD_ATTITUDE_BIT_FLAGS,  # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
    attitude.quaternion,               # Attitude quaternion.
    0,                                 # Body roll rate.
    0,                                 # Body pitch rate.
    0,                                 # Body yaw rate.
    thrust                             # Collective thrust, from 0-1.
    )

    self.atc.vehicle.send_mavlink(message)
    self.atc.vehicle.commands.upload()

    self.atc.LAST_ATTIUDE = attitude
    self.atc.LAST_THRUST = thrust

    sleep(self.atc.MESSAGE_SLEEP_TIME)

  def set_angle_thrust(self):

    if self.atc.vehicle.mode.name == "GUIDED_NOGPS":

      if self.atc.STATE == VehicleStates.hover:
        # self.atc.hover()
        self.send_angle_thrust(StandardAttitudes.level, StandardThrusts.hover)
      elif self.atc.STATE == VehicleStates.unknown:
        self.send_angle_thrust(StandardAttitudes.level, 0)
      else:
        self.send_angle_thrust(self.atc.DESIRED_ATTITUDE, self.atc.DESIRED_THRUST)
