############################################
# This file contains an interface for
# accessing Sonar sensors on GPIO pins and
# determining distance ot objects.
############################################
# Multi-Rotor Robot Design Team
# Missouri University of Science Technology
# Spring 2017
# Innocent Niyibizi
# pylint: disable=C, F, I, R, W

import RPi.GPIO as GPIO
import time

#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

SAFE_DISTANCE = 10



#class to easily define and call methods for sonar
class Sonar:

    #default constructor
    #trigger, echo, and side are passes through and set accordingly
    def __init__(self, trigger, echo, name):
        self.trigger = trigger
        self.echo = echo
        self.name = name
        #set GPIO direction (IN / OUT)
        GPIO.setup(self.trigger, GPIO.OUT)
        GPIO.setup(self.echo, GPIO.IN)

    #returns side that sonar is on
    def getName(self):
        return self.name

    #performs operations to get the approximate distance
    #returns distance to calling function
    def getDistance(self):
        GPIO.output(self.trigger, True)

        # set Trigger after 0.01ms to LOW
        time.sleep(0.00001)
        GPIO.output(self.trigger, False)

        self.startTime = time.time()
        self.stopTime = time.time()

        # save StartTime
        while GPIO.input(self.echo) == 0:
            self.startTime = time.time()

        # save time of arrival
        while GPIO.input(self.echo) == 1:
            self.stopTime = time.time()

        # time difference between start and arrival
        self.timePassed = self.stopTime - self.startTime
        # multiply with the sonic speed (34300 cm/s)
        # and divide by 2, because there and back
        self.distance = (self.timePassed * 34300) / 2

        return self.distance

#delcaring two instances of Sonar object for testing
rightSonar = Sonar(14, 15, "Right")
leftSonar = Sonar(2, 3, "Left")

#list of sonars
sonars = [rightSonar, leftSonar]
if __name__ == '__main__':
    try:
        #Running code while true
        while True:
            #Loop to iterate through all instances of sonar in list
            for sonar in sonars:
                side = sonar.getName() #getting side of sonar
                time.sleep(0.01)    #sleeping for 0.01
                dist = sonar.getDistance() #getting distance for current sonar
                #If distance is less than safe distance then error is displayed
                #else the current distance is displayed
                if(dist<SAFE_DISTANCE):
                    print("%s Too close %.1f" % (side, dist))
                else:
                    print("%s Measured Distance = %.1f cm" % (side, dist))
    except KeyboardInterrupt:
        print("Measurement stopped by User")
        GPIO.cleanup()
