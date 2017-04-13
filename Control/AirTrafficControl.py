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
import sys
from time import sleep
from copy import deepcopy

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
  backward = DroneAttitude(0,5,0)
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
  landing = "LANDING"
  landed = "LANDED"

class Tower(object):
  SIMULATOR = "127.0.0.1:14551"
  USB = "/dev/serial/by-id/usb-3D_Robotics_PX4_FMU_v2.x_0-if00"
  USB_DEV = "/dev/cu.usbmodem1"
  STANDARD_ATTITUDE_BIT_FLAGS = 0b00111111
  FLIP_ATTITUDE_BIT_FLAGS = 0b00111000
  STANDARD_THRUST_CHANGE = 0.05
  MAX_TURN_TIME = 5
  LAND_ALTITUDE = 0.5
  TURN_START_VELOCITY = 3
  TURN_RADIUS = 0.5 # Meters
  ANGLE_INCREMENT = 1.1
  ANGLE_DECREMENT = 0.9
  HOVER_CIRCLE_RADIUS = 0.5
  HOVER_MAX_DRIFT_TIME = 1
  HOVER_VELOCITY_SAMPLES = 5000
  DRIFT_CORRECT_THRESHOLD = 0.05
  ACCEL_NOISE_THRESHOLD = 0.09
  DRIFT_COMPENSATION = 1.0
  MAX_DRIFT_COMPENSATION = 10.0
  MAX_ANGLE_ALL_AXES = 20
  BATTERY_FAILSAFE_VOLTAGE = 11.50

  def __init__(self):
    self.start_time = 0
    self.vehicle_initialized = False

    self.vehicle = None

    self.LAST_ATTITUDE = StandardAttitudes.level
    self.LAST_THRUST = StandardThrusts.none
    self.STATE = VehicleStates.unknown

  def initialize(self):

    if(not self.vehicle_initialized):

      file = open('flight_log.txt', 'w')
      sys.stdout = file

      print("\nConnecting via USB to PixHawk...")
      self.vehicle = dronekit.connect(self.USB_DEV, wait_ready=True)

      if not self.vehicle:
        print("\nUnable to connect to vehicle.")
        return

      self.vehicle.mode = dronekit.VehicleMode("STABILIZE")
      self.STATE = VehicleStates.landed
      self.vehicle_initialized = True
      self.failsafes = FailsafeController(self)
      self.failsafes.start()
      self.start_time = time.time()

      self.switch_control()
      
      print("\nSuccessfully connected to vehicle.")

  def shutdown(self):    
    self.failsafes.join()
    self.vehicle.close()
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

  def switch_control(self):
    if not self.failsafes:
      self.failsafes = FailsafeController(self)
      self.failsafes.start()
    if self.vehicle.mode.name != "GUIDED_NOGPS":
      self.vehicle.mode = dronekit.VehicleMode("GUIDED_NOGPS")
      while(self.vehicle.mode.name != "GUIDED_NOGPS"):
        sleep(1)

  def get_uptime(self):
    uptime = time.time() - self.start_time
    return uptime

  def constrain(self, x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

  def set_angle_thrust(self, attitude, thrust):
    while(self.vehicle.mode.name != "GUIDED_NOGPS"):
      sleep(1)
    
    message = self.vehicle.message_factory.set_attitude_target_encode(
      0,                                 # Timestamp in milliseconds since system boot (not used).
      0,                                 # System ID
      0,                                 # Component ID
      self.STANDARD_ATTITUDE_BIT_FLAGS,  # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
      attitude.quaternion,               # Attitude quaternion.
      0,                                 # Body roll rate.
      0,                                 # Body pitch rate.
      0,                                 # Body yaw rate.
      thrust                             # Collective thrust, from 0-1.
    )
    self.vehicle.send_mavlink(message)
    self.vehicle.commands.upload()
    self.last_attitude = attitude
    self.last_thrust = thrust

  def get_drift_distance(self):
    print("\nCollecting velocity samples...")
    samples = 0
    x_velocities = []
    y_velocities = []

    while(samples < self.HOVER_VELOCITY_SAMPLES * 1000):

      velocity_x = self.vehicle.velocity[0]
      velocity_y = self.vehicle.velocity[1]

      if -self.ACCEL_NOISE_THRESHOLD <= velocity_x <= self.ACCEL_NOISE_THRESHOLD:
        velocity_x = 0
      if -self.ACCEL_NOISE_THRESHOLD <= velocity_y <= self.ACCEL_NOISE_THRESHOLD:
        velocity_y = 0

      x_velocities.append(velocity_x)
      y_velocities.append(velocity_y)
      samples+=1
      
    average_velocity_x = sum(x_velocities) / float(len(x_velocities))
    average_velocity_y = sum(y_velocities) / float(len(y_velocities))

    drift_distance_x = average_velocity_x * self.HOVER_MAX_DRIFT_TIME
    drift_distance_y = average_velocity_y * self.HOVER_MAX_DRIFT_TIME

    drift_distance = math.hypot(drift_distance_x, drift_distance_y)

    print("\nVelocity sample collection done.")
    return drift_distance, drift_distance_x, drift_distance_y
    
  def hover(self, duration=None):
    self.set_angle_thrust(StandardAttitudes.level, StandardThrusts.hover)
    self.STATE = VehicleStates.hover

    while(duration > 0):

      drift_information = self.get_drift_distance()
      drift_distance = drift_information[0]
      drift_distance_x = drift_information[1]
      drift_distance_y = drift_information[2]

      adjust_attitude = deepcopy(StandardAttitudes.level)

      corrected_distance = 0
      correction_delta = corrected_distance - drift_distance

      if(math.fabs(drift_distance_x) > self.HOVER_CIRCLE_RADIUS or math.fabs(drift_distance_y) > self.HOVER_CIRCLE_RADIUS):
        while(not (-self.DRIFT_CORRECT_THRESHOLD <= correction_delta <= self.DRIFT_CORRECT_THRESHOLD)):

          if(drift_distance_y > 0 and math.fabs(drift_distance_y) > self.HOVER_CIRCLE_RADIUS):
            adjust_attitude.roll_deg -= self.DRIFT_COMPENSATION
            if adjust_attitude.roll_deg < -self.MAX_DRIFT_COMPENSATION:
              adjust_attitude.roll_deg = -self.MAX_DRIFT_COMPENSATION
            adjust_attitude = DroneAttitude(adjust_attitude.roll_deg, adjust_attitude.pitch_deg, adjust_attitude.yaw_deg)
            print("\Drifted right, correcting left.")
          elif(drift_distance_y < 0 and math.fabs(drift_distance_y) > self.HOVER_CIRCLE_RADIUS):
            adjust_attitude.roll_deg += self.DRIFT_COMPENSATION
            if adjust_attitude.roll_deg > self.MAX_DRIFT_COMPENSATION:
              adjust_attitude.roll_deg = self.MAX_DRIFT_COMPENSATION
            adjust_attitude = DroneAttitude(adjust_attitude.roll_deg, adjust_attitude.pitch_deg, adjust_attitude.yaw_deg)
            print("\Drifted left, correcting right.")
          if(drift_distance_x > 0 and math.fabs(drift_distance_x) > self.HOVER_CIRCLE_RADIUS):
            adjust_attitude.pitch_deg += self.DRIFT_COMPENSATION
            if adjust_attitude.pitch_deg > self.MAX_DRIFT_COMPENSATION:
              adjust_attitude.pitch_deg = self.MAX_DRIFT_COMPENSATION
            adjust_attitude = DroneAttitude(adjust_attitude.roll_deg, adjust_attitude.pitch_deg, adjust_attitude.yaw_deg)
            print("\Drifted forward, correcting backwards.")
          elif(drift_distance_x < 0 and math.fabs(drift_distance_x) > self.HOVER_CIRCLE_RADIUS):
            adjust_attitude.pitch_deg -= self.DRIFT_COMPENSATION
            if adjust_attitude.pitch_deg < -self.MAX_DRIFT_COMPENSATION:
              adjust_attitude.pitch_deg = -self.MAX_DRIFT_COMPENSATION
            adjust_attitude = DroneAttitude(adjust_attitude.roll_deg, adjust_attitude.pitch_deg, adjust_attitude.yaw_deg)
            print("\Drifted backwards, correcting forward.")

          self.set_angle_thrust(adjust_attitude, StandardThrusts.hover)

          sleep(self.HOVER_MAX_DRIFT_TIME)

          corrected_information = self.get_drift_distance()
          corrected_distance += corrected_information[0]
          correction_delta = corrected_distance - drift_distance

          if(((correction_delta > 0) and (drift_distance > 0)) and math.fabs(correction_delta) > self.DRIFT_CORRECT_THRESHOLD):
            drift_distance = correction_delta
            drift_distance_x = corrected_information[1]
            drift_distance_y = corrected_information[2]
            corrected_distance = 0
            correction_delta = corrected_distance - drift_distance

          print("\nCorrecting: " + str(corrected_distance) + " Drifted " + str(drift_distance))
          print("\nDrift X: " + str(drift_distance_x) + " Drift Y: " + str(drift_distance_y))
          print("\nDelta: " + str(correction_delta))
          print("\nVehicle Attitude: " + " Pitch: " + str(math.degrees(self.vehicle.attitude.pitch)) + " Roll: " + str(math.degrees(self.vehicle.attitude.roll)) + " Yaw: " + str(math.degrees(self.vehicle.attitude.pitch)))
          print("\nX m/s: " + str(self.vehicle.velocity[0]) + " Y m/s: " + str(self.vehicle.velocity[1]))
          print("\nAttitude Adjustments: " + "Roll: " + str(adjust_attitude.roll_deg) + " Pitch: " + str(adjust_attitude.pitch_deg) + " Yaw: " + str(adjust_attitude.yaw_deg))

      self.set_angle_thrust(StandardAttitudes.level, StandardThrusts.hover)
      sleep(self.HOVER_MAX_DRIFT_TIME)
      duration-=1

  def takeoff(self, target_altitude):
    self.arm_drone()
    self.vehicle.mode = dronekit.VehicleMode("GUIDED_NOGPS")

    while(self.vehicle.mode.name != "GUIDED_NOGPS"):
      sleep(1)

    initial_alt = self.vehicle.location.global_relative_frame.alt
    
    while((self.vehicle.location.global_relative_frame.alt - initial_alt) < target_altitude):
      self.set_angle_thrust(StandardAttitudes.level, StandardThrusts.takeoff)
      sleep(1)

    print('Reached target altitude:{0:.2f}m'.format(self.vehicle.location.global_relative_frame.alt))

    self.hover(5)
    self.land()
    self.set_angle_thrust(StandardAttitudes.level, StandardThrusts.hover)


  def fly_for_time(self, duration, direction, target_velocity, should_hover_on_finish):
    end_manuever = datetime.now() + timedelta(seconds=duration)

    self.STATE = VehicleStates.flying
    self.set_angle_thrust(direction, StandardThrusts.hover)

    while(datetime.now() < end_manuever):

      print(self.vehicle.airspeed,)
      print(self.vehicle.velocity,)

      updated_attitude = deepcopy(self.LAST_ATTITUDE)

      if(self.vehicle.airspeed < target_velocity):
        updated_attitude.pitch_deg -= 1
      elif(self.vehicle.airspeed > target_velocity):
        updated_attitude.pitch_deg += 1
      else:
        updated_attitude.pitch_deg = direction.pitch_deg

      if(updated_attitude.pitch_deg < -self.MAX_ANGLE_ALL_AXES):
        updated_attitude.pitch_deg = -self.MAX_ANGLE_ALL_AXES

      if(updated_attitude.pitch_deg > self.MAX_ANGLE_ALL_AXES):
        updated_attitude.pitch_deg = self.MAX_ANGLE_ALL_AXES

      if(updated_attitude.roll_deg < -self.MAX_ANGLE_ALL_AXES):
        updated_attitude.roll_deg = -self.MAX_ANGLE_ALL_AXES
      
      if(updated_attitude.roll_deg > self.MAX_ANGLE_ALL_AXES):
        updated_attitude.roll_deg = self.MAX_ANGLE_ALL_AXES

      if(updated_attitude.yaw_deg < -self.MAX_ANGLE_ALL_AXES):
        updated_attitude.yaw_deg = -self.MAX_ANGLE_ALL_AXES

      if(updated_attitude.yaw_deg > self.MAX_ANGLE_ALL_AXES):
        updated_attitude.yaw_deg = self.MAX_ANGLE_ALL_AXES

      updated_attitude.pitch = math.radians(updated_attitude.pitch_deg)
      updated_attitude.quaternion = updated_attitude.get_quaternion()

      self.set_angle_thrust(updated_attitude, self.LAST_THRUST)

      print(updated_attitude.pitch_deg,)

      sleep(1)
    
    if(should_hover_on_finish):
      self.hover()

  def land(self):
    self.vehicle.mode = dronekit.VehicleMode("LAND")
    self.STATE = VehicleStates.landing
    while((self.vehicle.location.global_relative_frame.alt) >= self.LAND_ALTITUDE):
      sleep(1)
    else:
      self.STATE = VehicleStates.landed

  def do_circle_turn(self, desired_angle, direction, duration):
    if(duration > self.MAX_TURN_TIME):
      return

    self.STATE = VehicleStates.flying

    desired_angle = math.radians(desired_angle)
      
    max_angle = desired_angle
    altitude_to_hold = self.vehicle.location.global_relative_frame.alt

    duration = timedelta(seconds=duration)
    end_manuever = datetime.now() + duration
    
    self.fly_for_time(1, StandardAttitudes.forward, self.TURN_START_VELOCITY, False)
    
    while(end_manuever <= datetime.now()):
      change_in_time = end_manuever - datetime.now()
      current_altitude = self.vehicle.location.global_relative_frame.alt

      roll_angle = max_angle * (math.cos(self.vehicle.airspeed * change_in_time.seconds) / self.TURN_RADIUS)
      pitch_angle = max_angle * (math.sin(self.vehicle.airspeed * change_in_time.seconds) / self.TURN_RADIUS)

      roll_angle = math.degrees(roll_angle)
      pitch_angle = math.degrees(pitch_angle)
      self.last_attitude.yaw = math.degrees(self.last_attitude.yaw)

      updated_attitude = DroneAttitude(pitch_angle, self.last_attitude.yaw, roll_angle)

      self.set_angle_thrust(updated_attitude, StandardThrusts.hover)

      print("Sent message.")

      if(current_altitude > altitude_to_hold):
        max_angle = desired_angle * self.ANGLE_INCREMENT
      elif(current_altitude < altitude_to_hold):
        max_angle = desired_angle * self.ANGLE_DECREMENT
    
      sleep(1)
      
    self.fly_for_time(1, StandardAttitudes.forward, self.vehicle.airspeed, True)

  def check_sonar_sensors(self):
    pass
  
  def check_battery_voltage(self):
    if(self.vehicle.battery.voltage * 1000 < self.BATTERY_FAILSAFE_VOLTAGE):
        self.land()

class FailsafeController(threading.Thread):

  def __init__(self, atc_instance):
    self.atc = atc_instance
    self.stoprequest = threading.Event()
    super(FailsafeController, self).__init__()

  def run(self):
    while not self.stoprequest.isSet():
      if self.atc.STATE == VehicleStates.hover or self.atc.STATE == VehicleStates.flying:
        self.atc.check_sonar_sensors()
        self.atc.check_battery_voltage()

  def join(self, timeout=None):
    if self.atc.vehicle.armed:
      if self.atc.STATE != VehicleStates.landed:
        self.atc.land()

    self.stoprequest.set()
    super(FailsafeController, self).join(timeout)

