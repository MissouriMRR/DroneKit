"""
This is a Python PID module heavily inspired by the Arduino PID Library
written by Brett Beauregard.
http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/
"""

import time
from enum import Enum


class Direction(Enum):
    direct = 1
    reverse = 2


class Mode(Enum):
    automatic = 1
    manual = 2


class PID(object):

    def __init__(self, kp, ki, kd, set_point, controller_direction):
        """
        The parameters specified here are those for for which we can't set up
        reliable defaults, so we need to have the user set them.
        :param kp: Proportional Tuning Parameter
        :param ki: Integral Tuning Parameter
        :param kd: Derivative Tuning Parameter
        :param set_point: The value that we want the process to be.
        :param controller_direction:
        """

        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.direction = controller_direction

        self.i_term = 0

        self.out_max = 0
        self.out_min = 0

        self.last_input = 0
        self.output = 0
        self.input = 0

        self.set_point = set_point
        self.mode = Mode.automatic

        self.set_output_limits(0, 255)
        self.sample_time = 1000
        self.controller_direction = Direction.direct
        self.set_controller_direction(controller_direction)
        self.set_tunings(self.kp, self.ki, self.kd)

        self.last_time = self.now()

    @staticmethod
    def now():
        """
        Static method to make it easy to obtain the current time
        in milliseconds.
        :return: Current time in milliseconds.
        """
        return int(round(time.time() * 1000))

    def compute(self, input):
        """
        This, as they say, is where the magic happens. This function should
        be called every time "void loop()" executes. The function will decide
        for itself whether a new PID Output needs to be computed.
        :param input: Input value for the PID controller.
        :return: Returns true when the output is computed,
        false when nothing has been done.
        """

        if self.mode is Mode.manual:
            return 0, False

        delta_time = self.now() - self.last_time

        if delta_time >= self.sample_time:
            error = self.set_point - input

            self.i_term += (self.ki * error)

            if self.i_term > self.out_max:
                self.i_term = self.out_max
            elif self.i_term < self.out_min:
                self.i_term = self.out_min

            delta_input = input - self.last_input

            self.output = self.kp * error + self.i_term - self.kd * delta_input

            if self.output > self.out_max:
                self.output = self.out_max
            elif self.output < self.out_min:
                self.output = self.out_min

            self.last_input = input
            self.last_time = self.now()

            return self.output, True
        else:
            return 0, False

    def set_tunings(self, kp, ki, kd):
        """
        This function allows the controller's dynamic performance to be
        adjusted. It's called automatically from the constructor,
        but tunings can also be adjusted on the fly during normal operation.
        :param kp: Proportional Tuning Parameter
        :param ki: Integral Tuning Parameter
        :param kd: Derivative Tuning Parameter
        """

        if kp < 0 or ki < 0 or ki < 0:
            return

        sample_time_in_sec = self.sample_time / 1000

        self.kp = kp
        self.ki = ki * sample_time_in_sec
        self.kd = kd / sample_time_in_sec

        if self.controller_direction is Direction.reverse:
            self.kp = 0 - kp
            self.ki = 0 - ki
            self.kd = 0 - kd

    def set_sample_time(self, sample_time):
        """
        Sets the period, in milliseconds, at which the calculation is
        performed.
        :param sample_time: The period, in milliseconds,
        at which the calculation is performed.
        """

        if sample_time > 0:
            ratio = sample_time / self.sample_time

            self.ki *= ratio
            self.kd /= ratio
            self.sample_time = sample_time

    def set_output_limits(self, min, max):
        """
        This function will be used far more often than set_input_limits. While
        the input to the controller will generally be in the 0-1023 range
        (which is the default already), the output will be a little different.
        Maybe they'll be doing a time window and will need 0-8000 or something.
        Or maybe they'll want to clamp it from 0-125.
        :param min: Minimum output value from the PID controller
        :param max: Maximum output value from the PID controller
        """

        if min >= max:
            return

        self.out_min = min
        self.out_max = max

        if self.mode is Mode.automatic:
            if self.output > self.out_max:
                self.output = self.out_max
            elif self.output < self.out_min:
                self.output = self.out_min

            if self.i_term > self.out_max:
                self.i_term = self.out_max
            elif self.i_term < self.out_min:
                self.i_term = self.out_min

    def set_mode(self, mode):
        """
        Allows the controller Mode to be set to manual (0) or Automatic
        (non-zero) when the transition from manual to auto occurs,
        the controller is automatically initialized.
        :param mode: The mode of the PID controller.
        Can be either manual or automatic.
        """

        if self.mode is Mode.manual and mode is Mode.automatic:
            self.initialize()

        self.mode = mode

    def initialize(self):
        """
        Does all the things that need to happen to ensure a smooth transfer
        from manual to automatic mode.
        """

        self.i_term = self.output
        self.last_input = self.input
        if self.i_term > self.out_max:
            self.i_term = self.out_max
        elif self.i_term < self.out_min:
            self.i_term = self.out_min

    def set_controller_direction(self, direction):
        """
        The PID will either be connected to a DIRECT acting process
        (+Output leads to +Input) or a REVERSE acting process
        (+Output leads to -Input.). We need to know which one,
        because otherwise we may increase the output when we should be
        decreasing. This is called from the constructor.
        :param direction: The direction of the PID controller.
        """

        if self.mode is Mode.automatic and direction is not self.direction:
            self.kp = 0 - self.kp
            self.ki = 0 - self.ki
            self.kd = 0 - self.kd

        self.direction = direction
