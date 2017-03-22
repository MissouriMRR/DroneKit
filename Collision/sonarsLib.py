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

from hcsr04sensor import sensor
import time

#user created sonar class for easier mapping
class Sonar:

    def __init__(self, trigger, echo):
        self.trigger = trigger
        self.echo = echo

    def getDistance(self):
        print("Ggeting")
        value = sensor.Measurement(self.trigger, self.echo)
        raw_measurement = value.raw_distance()
        self.distance = value.distance_metric(raw_measurement)
        print("The distance = {} centimeters". format(self.distance))

mainSonar = Sonar(2, 3)

try:
    while True:
        mainSonar.getDistance()
        print("distance")
except KeyboardInterrupt:
    print("Measurement stopped by User")
