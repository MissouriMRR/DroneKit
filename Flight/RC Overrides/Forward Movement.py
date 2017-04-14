import time
from dronekit import VehicleMode, connect
from pymavlink import mavutil
fly=False
landing = False
lat = 49.2778638
lon = -123.0633418

vehicle = connect('tcp:127.0.0.1:5763', wait_ready=True)

print "Read channels individually:"
print " Ch1: %s" % vehicle.channels['1']
print " Ch2: %s" % vehicle.channels['2']
print " Ch3: %s" % vehicle.channels['3']
print " Ch4: %s" % vehicle.channels['4']
print " Ch1: %s" % vehicle.channels['5']
print " Ch2: %s" % vehicle.channels['6']
print " Ch3: %s" % vehicle.channels['7']
print " Ch4: %s" % vehicle.channels['8']
print "Number of channels: %s" % len(vehicle.channels)

# Don't let the user try to arm until autopilot is ready
while not vehicle.is_armable:
   print(" Waiting for vehicle to initialise... (GPS={0},Battery={1})".format(vehicle.gps_0, vehicle.battery))
   time.sleep(1)

# Set vehicle mode
desired_mode = 'STABILIZE'
while vehicle.mode != desired_mode:
   vehicle.mode = VehicleMode(desired_mode)
   time.sleep(0.5)

while not vehicle.armed:
     print("Arming motors")
     vehicle.armed = True
     time.sleep(0.5)

while True:
   vehicle.channels.overrides[3] = 1500
   print " Ch3 override: %s" % vehicle.channels.overrides[3]
   if vehicle.location.global_relative_frame.alt >= 3:
     print('Reached target altitude:{0:.2f}m'.format(vehicle.location.global_relative_frame.alt))
     fly=True
     vehicle.channels.overrides[3] = 1350
     break
   else:
     print("Altitude:{0:.2f}m".format(vehicle.location.global_relative_frame.alt))
   time.sleep(0.5)

while fly==True:
    vehicle.channels.overrides[2] = 1400
    vehicle.channels.overrides[3] = 1350
    print " Ch2 override: %s" % vehicle.channels.overrides[2]
    time.sleep(10)
    fly=False
    landing = True
    break
print("landing")
while landing==True:
   vehicle.channels.overrides[2] = 1500
   vehicle.channels.overrides[3] = 1350
   print " Ch3 override: %s" % vehicle.channels.overrides[3]

   if vehicle.location.global_relative_frame.alt <= 1.0:
     print('Reached target altitude:{0:.2f}m'.format(vehicle.location.global_relative_frame.alt))
     break
   else:
     print("Altitude:{0:.2f}m".format(vehicle.location.global_relative_frame.alt))
   time.sleep(0.5)

vehicle.close()
