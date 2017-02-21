############################################ 
# This file contains a wrapper class 
# for DroneKit related operations 
# for our drone.
############################################ 
# Multi-Rotor Robot Design Team
# Missouri University of Science Technology
# Spring 2017
# Lucas Coon, Mark Raymond Jr.

from datetime import datetime, timedelta
from dronekit import VehicleMode, connect, Attitude
from os import system
from time import sleep

import math

class Attitude(Attitude):

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

class Tower(object):
  STANDARD_ATTITUDE_BIT_FLAGS = 0b11100000
  TURNING_ATTITUDE_BIT_FLAGS = 0b00000000
  STANDARD_THRUST_CHANGE = 0.05
  MAX_TURN_TIME = 5
  LAND_ALTITUDE = 0.5
  STANDARD_ATTITUDES = { 
                        "LEVEL" : Attitude(0,0,0),
                        "FORWARD" : Attitude(-5,0,0), # -5 degrees.
                        "BACKWARD" : Attitude(5,0,0),  # +5 degrees.
                        "LEFT" : Attitude(-5, 0, -5),
                        "RIGHT" : Attitude(-5, 0, 5)
                       }
  STANDARD_THRUSTS = { 
                      "HOVER"      : 0.45,
                      "LOW_SPEED"  : 0.50, 
                      "MED_SPEED"  : 0.55,
                      "HIGH_SPEED" : 0.60
                    }
                      

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
      print("\nEnabling serial UART connection on /dev/ttyS1 at 57600 baud...")
      return_code = system("stty -F /dev/ttyS1 57600 raw -echo -echoe -echok -crtscts")
      if return_code == 0:
        self.system_initialized = True
        print("\nSucessfully enabled UART connection.")

  def initialize_drone(self):
    if not self.drone_initialized:
      print("Connecting via DroneKit to /dev/ttyS1...")
      self.vehicle = dronekit.connect("/dev/ttyS1", baud=57600, wait_ready=True)
      self.vehicle.mode = VehicleMode("STABILIZE")
      self.vehicle_state = "GROUND_IDLE"
      self.connected = True
      self.vehicle_initialized = True
      self.start_time = datetime.now()
      self.last_attitude = [1, 0, 0, 0]
      print("Successfully connected to vehicle.")

  def get_uptime(self):
    uptime = timedelta(self.start_time, datetime.now())
    return uptime

  def set_thrust_angle_using_vectors(self, attitude, thrust):
    print("Building MAVLink message...")
    self.vehicle.mode = VehicleMode("STABILIZE")
    message = self.vehicle.message_factory.set_attitude_target_encode(
      0,                                 # Timestamp in milliseconds since system boot (not used).
      0,                                 # System ID
      0,                                 # Component ID
      STANDARD_ATTITUDE_BIT_FLAGS,       # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
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

  def return_to_hover():
    self.vehicle_state = "HOVER"
    set_thrust_angle_using_vectors(STANDARD_ATTITUDES["LEVEL"], STANDARD_THRUSTS["HOVER"])
    self.vehicle.mode = VehicleMode("ALT HOLD")   

  def takeoff_using_vectors(self, target_altitude, thrust):
    set_thrust_angle_using_vectors(STANDARD_ATTITUDES["LEVEL"], thrust)

    while(vehicle.location.global_relative_frame.alt <= target_altitude):
      sleep(1)

    return_to_hover()
    print('Reached target altitude:{0:.2f}m'.format(vehicle.location.global_relative_frame.alt))

  def fly_for_time(self, duration, direction, target_velocity):
    duration = timedelta(seconds=duration)
    end_manuever = datetime.now() + duration
    set_thrust_angle_using_vectors(direction, self.last_thrust)
    while(end_manuever <= datetime.now()):
      if self.vehicle.velocity > target_velocity:
        set_thrust_angle_using_vectors(direction, self.last_thrust - STANDARD_THRUST_CHANGE)
      elif(self.vehicle.velocity < target_velocity):
        set_thrust_angle_using_vectors(direction, self.last_thrust + STANDARD_THRUST_CHANGE)
      sleep(1)
    return_to_hover()

  def fly_for_time_thrust(self, duration, direction, thrust):
    duration = timedelta(seconds=duration)
    end_manuever = datetime.now() + duration
    set_thrust_angle_using_vectors(direction, thrust)
    while(end_manuever <= datetime.now()):
      sleep(1)
    return_to_hover()

  def land(self):
    self.vehicle.mode = VehicleMode("LAND")
    while(vehicle.location.global_relative_frame.alt <= LAND_ALTITUDE):
      sleep(1)
    else:
      self.vehicle_state = "GROUND_IDLE"

  def turn_for_time(self, direction, duration):
    if duration > MAX_TURN_TIME:
      return

    print("Building MAVLink message...")

    self.vehicle.mode = VehicleMode("ALT HOLD")

    message = self.vehicle.message_factory.set_attitude_target_encode(
      0,                                        # Timestamp in milliseconds since system boot (not used).
      0,                                        # System ID
      0,                                        # Component ID
      TURNING_ATTITUDE_BIT_FLAGS,               # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
      attitude.quaternion,                      # Attitude quaternion.
      1,                                        # Body roll rate.
      0,                                        # Body pitch rate.
      1,                                        # Body yaw rate.
      STANDARD_THRUSTS["LOW"]                   # Collective thrust, from 0-1.
    )

    self.vehicle.send_mavlink(message)
    self.vehicle.flush()
    self.last_attitude = attitude

    print("Sent message.")

    sleep(duration)

    return_to_hover()

