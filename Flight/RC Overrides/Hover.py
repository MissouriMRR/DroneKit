import time
from dronekit import VehicleMode, connect
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

vehicle = connect("/dev/ttyS1", baud=57600, wait_ready=True)

#Set the vehicle mode to stabilized. 
#This will make the vehicle compensate for any idiosynchracies.
desired_mode = 'STABILIZE'
while vehicle.mode != desired_mode:
   vehicle.mode = VehicleMode(desired_mode)
   time.sleep(0.5)

#Arm the motors.
while not vehicle.armed:
     print("Arming motors...")
     vehicle.armed = True
     time.sleep(0.5)

print("Beginning ascent...")

desired_speed = 1300
current_speed = 1000

while True:
  print "\nVehicle statistics:"
  print " Location: %s" % vehicle.location
  print " Attitude: %s" % vehicle.attitude
  print " Altitude: %s" % vehicle.location.global_relative_frame.alt
  print " Velocity: %s" % vehicle.velocity
  print " GPS: %s" % vehicle.gps_0
  print " Groundspeed: %s" % vehicle.groundspeed
  print " Airspeed: %s" % vehicle.airspeed
  print " Mount status: %s" % vehicle.mount_status
  print " Battery: %s" % vehicle.battery
  print " Rangefinder: %s" % vehicle.rangefinder
  print " Rangefinder distance: %s" % vehicle.rangefinder.distance
  print " Rangefinder voltage: %s" % vehicle.rangefinder.voltage
  print " Mode: %s" % vehicle.mode.name    # settable
  print " Armed: %s" % vehicle.armed    # settable
  cls()

  #Ramp up to the desired_speed at which point Panic should take off.
  while(current_speed < desired_speed):
    current_speed += 100
    vehicle.channels.overrides[3] = current_speed
    print(current_speed)
    print " Ch3 override: %s" % vehicle.channels.overrides[3]
    time.sleep(1)

  if vehicle.location.global_relative_frame.alt >= 1.25:
    print('Reached target altitude:{0:.2f}m'.format(vehicle.location.global_relative_frame.alt))
    #After we reach the target altitude in meters, break out of the loop. 
    #If you're above 1300 for a desired speed, you should ramp down to 1300 here as well.
    break
  else:
    print("Altitude:{0:.2f}m".format(vehicle.location.global_relative_frame.alt))
  time.sleep(0.5)


#Hover for 5 seconds.
time.sleep(5)

print("Landing vehicle...")

vehicle.mode = VehicleMode("LAND")

time.sleep(30)

print("Cleaning up...")

vehicle.close()

print("Done.")
