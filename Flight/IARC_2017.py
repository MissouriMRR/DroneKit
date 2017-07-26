from AirTrafficControl import Tower, VehicleStates, StandardVelocities
from timeit import default_timer as timer

class Mission7b():
  MATCH_LENGTH = 600

  def __init__(self, **kwargs):
    self.match_start = timer()
    self.tower = Tower()
    self.tower.initialize(**kwargs)
    ##Setup ground frame passing.

  def start_match(self):
    while(self.match_start < self.MATCH_LENGTH):

      if(self.tower.STATE == VehicleStates.landed):
        self.tower.guided_takeoff(Tower.STANDARD_MATCH_ALTITUDE)
        self.tower.fly_distance(1, StandardVelocities.med, 0)
        self.tower.fly_distance(10, 0, StandardVelocities.med)
        self.tower.switch_gimbal_mode()

      if(self.tower.scanningField):
        self.tower.realsense_range_finder.send_frame_to_ground()

    self.tower.shutdown()

    
