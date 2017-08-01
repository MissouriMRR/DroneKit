from AirTrafficControl import Tower, VehicleStates, StandardVelocities
from timeit import default_timer as timer

import math
import numpy as np

class Mission7b():
  MATCH_LENGTH = 600
  VALID_ROOMBA_HEADINGS = []
  ACCEPTABLE_ANGLE_DIFFERENCE = 30

  def __init__(self, **kwargs):
    self.match_start = timer()
    self.tower = Tower()
    self.run_match(set=True)
    self.tower.initialize(frame_info_received=self.run_match.on_frame_info_received, **kwargs)

  def run_match(self, set = False):
    def on_frame_info_received(roomba_frame_data):
      if(self.tower.scanningField):
        roombas_heading_towards_vehicle = []

        for i, orientation in enumerate(roomba_frame_data['orientations']):
          angle_to_roomba_from_drone = math.atan2(orientation[0], orientation[1])
          difference = math.fabs(math.radians(self.tower.vehicle.attitude.yaw) - angle_to_roomba_from_drone)

          if difference < Mission7b.ACCEPTABLE_ANGLE_DIFFERENCE:
            roombas_heading_towards_vehicle.append((i, angle_to_roomba_from_drone, roomba_frame_data))
        
        target_roomba_i, target_roomba_angle_from_drone, target_roomba_frame_data = (0, 0, None)
        min_dist = 10

        depth = roomba_frame_data['depth']
        depth_image_area = np.prod(depth.shape[:2])
        depth_w, depth_h = (depth.shape[:2])
        X = (np.arange(depth_image_area) % self.cam.depth.shape[1]).reshape(depth_w, depth_h)
        Y = (np.arange(depth_image_area) % self.cam.depth.shape[0]).reshape(depth_h, depth_w).transpose()
        depth = (Y * depth + X)*self.depth_scale

        for i, angle_to_roomba_from_drone, roomba_frame_data in roombas_heading_towards_vehicle:
          x_center, y_center = roomba_frame_data['centers'][i]
          distance_to_roomba = depth[y_center, x_center]
          
          if distance_to_roomba <= min_dist:
            target_roomba_i, target_roomba_angle_from_drone, target_roomba_frame_data = (i, angle_to_roomba_from_drone, roomba_frame_data)
            min_dist = distance_to_roomba

        target_roomba_distance = min_dist
        # self.head_towards_roomba(target_roomba_distance, target_roomba_angle_from_drone)
        
    if(set):
      if not hasattr(self.run_match, 'on_frame_info_received'):
        self.run_match.on_frame_info_received = on_frame_info_received
    else:
      while(self.match_start < self.MATCH_LENGTH):

        if(self.tower.STATE == VehicleStates.landed):
          self.tower.guided_takeoff(Tower.STANDARD_MATCH_ALTITUDE)
          self.tower.fly_distance(1, StandardVelocities.med, 0)
          self.tower.fly_distance(10, 0, StandardVelocities.med)
          self.tower.switch_gimbal_mode()

        # if(self.tower.state == VehicleStates.hover):
        #   pass

      self.tower.shutdown()