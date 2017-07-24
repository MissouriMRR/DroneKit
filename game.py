from Fight import AirTrafficControl as ATC
import datetime


def main():
    roomba_count = 0
    end_time = datetime.datetime.now() + datetime.timedelta(minutes=10)
    t = ATC.Tower()
    t.initialize(enable_lidar = True, enable_realsense = True)
    while datetime.datetime.now() < end_time or roomba_count < 10:
        ATC.smo_guided()
        ATC.switch_gimbal_mode()
        # Send RealSense data to Chris's code to process
        # Recive the data from Chris's data
        # Identify a roomba that would make the best target
        #   based on where it is with respect to green line
        # Catch up to roomba
        ATC.switch_gimbal_mode()
        # Follow the roomba once caught up
        # Decide how to interact with it
        # Iinteract with it accordingly
        # Takeoff with a special function b/c landing on roomba
