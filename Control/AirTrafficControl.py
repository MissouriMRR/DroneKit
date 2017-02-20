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
from dronekit import VehicleMode, connect
from os import system

import time

class Tower(object):
  ATTITUDE_BIT_FLAGS = 0b11100000
  LAND_ALTITUDE = 1
  STANDARD_ATTITUDES = { 
                        "LEVEL" : [1, 0, 0, 0],
                        "FORWARD_5_DEG" : [0.99905, 0, -0.04362, 0] 
                       }
  STANDARD_SPEEDS = { 
                      "HOVER"      : 0.45,
                      "LOW_SPEED"  : 0.50,
                      "MED_SPEED"  : 0.55,
                      "HIGH_SPEED" : 0.60
                    }
                      

  def __init__(self):
    self.start_time = datetime.now()
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
      print("Successfully connected to vehicle.")

  def get_uptime(self):
    uptime = timedelta(self.start_time, datetime.now())
    return uptime

  def set_thrust_angle_using_vectors(self, attitude_quaternion, thrust):
    print("Building MAVLink message...")
    self.vehicle.mode = VehicleMode("STABILIZE")
    message = self.vehicle.message_factory.set_attitude_target_encode(
      0,                        # Timestamp in milliseconds since system boot (not used).
      0,                        # System ID
      0,                        # Component ID
      ATTITUDE_BIT_FLAGS,       # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
      attitude_quaternion,      # Quaternions
      0,                        # Body roll rate.
      0,                        # Body pitch rate.
      0,                        # Body yaw rate.
      thrust                    # Collective thrust, from 0-1.
    )
    self.vehicle.send_mavlink(message)
    self.vehicle.flush()
    print("Sent message.")

  def takeoff_using_vectors(self, target_altitude, thrust):
    set_thrust_angle_using_vectors(STANDARD_ATTITUDES["LEVEL"], thrust)

    while(vehicle.location.global_relative_frame.alt <= target_altitude):
      time.sleep(1)
    else:
      set_thrust_angle_using_vectors(STANDARD_ATTITUDES["LEVEL"], STANDARD_SPEEDS["HOVER"])
      print('Reached target altitude:{0:.2f}m'.format(vehicle.location.global_relative_frame.alt))
      #After we reach the target altitude in meters, break out of the loop. 
      #If you're above 1300 for a desired speed, you should ramp down to 1300 here as well.

  def land(self):
    self.vehicle.mode = VehicleMode("LAND")
      while(vehicle.location.global_relative_frame.alt <= LAND_ALTITUDE):
        time.sleep(1)


