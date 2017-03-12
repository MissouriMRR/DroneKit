from time import sleep
from AirTraffiControl import StandardAttitudes, StandardThrusts, VehicleStates

import dronekit
import threading

global DESIRED_ATTITUDE
global DESIRED_THRUST

class ControlPrimitives(threading.Thread)

  def __init__(self, atc_instance):
    self.atc = atc_instance
    self.last_attitude = None
    self.last_thrust = None
    self.stoprequest = threading.Event()

  def run(self):
    while not self.stoprequest.isSet():
      
      if atc.vehicle.armed:
        
        if atc.vehicle_state = VehicleStates.hover: 
          set_angle_thrust(StandardAttitudes.level, StandardThrusts.hover)
        else:
          set_angle_thrust(DESIRED_ATTITUDE, DESIRED_THRUST)

      continue

  def join(self, timeout=None):
    if atc.vehicle.armed():
      if atc.vehicle_state != VehicleStates.landed:
        self.atc.land()

    self.stoprequest.set()
    super(ControlThread, self).join(timeout)
          
  def set_angle_thrust(self, attitude, thrust):
    self.atc.vehicle.mode = dronekit.VehicleMode("GUIDED_NOGPS")

    while(self.atc.vehicle.mode.name != "GUIDED_NOGPS"):
    sleep(1)
    
    message = self.atc.vehicle.message_factory.set_attitude_target_encode(
    0,                                 # Timestamp in milliseconds since system boot (not used).
    0,                                 # System ID
    0,                                 # Component ID
    atc.STANDARD_ATTITUDE_BIT_FLAGS,  # Bit flags. For more info, see http://mavlink.org/messages/common#SET_ATTITUDE_TARGET.
    attitude.quaternion,               # Attitude quaternion.
    0,                                 # Body roll rate.
    0,                                 # Body pitch rate.
    0,                                 # Body yaw rate.
    thrust                             # Collective thrust, from 0-1.
    )

    self.atc.vehicle.send_mavlink(message)
    self.atc.vehicle.commands.upload()

    self.last_attitude = attitude
    self.last_thrust = thrust

    print("Sent message.")