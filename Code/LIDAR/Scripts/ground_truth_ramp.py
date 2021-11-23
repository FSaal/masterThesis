#!/usr/bin/env python
"""Publish manual set ramp status, can be used for labeling"""
from __future__ import print_function
import atexit
from select import select
import rospy
from std_msgs.msg import Int16
import sys
import termios


class KBHit:

    def __init__(self):
        '''Creates a KBHit object that you can call to do various keyboard things.'''
        # Save the terminal settings
        self.fd = sys.stdin.fileno()
        self.new_term = termios.tcgetattr(self.fd)
        self.old_term = termios.tcgetattr(self.fd)

        # New terminal setting unbuffered
        self.new_term[3] = (self.new_term[3] & ~termios.ICANON & ~termios.ECHO)
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.new_term)

        # Support normal-terminal reset at exit
        atexit.register(self.set_normal_term)

    def set_normal_term(self):
        ''' Resets to normal terminal.  On Windows this is a no-op.'''
        termios.tcsetattr(self.fd, termios.TCSAFLUSH, self.old_term)

    def getch(self):
        ''' Returns a keyboard character after kbhit() has been called.'''
        return sys.stdin.read(1)

    def kbhit(self):
        ''' Returns True if keyboard character was hit, False otherwise.'''
        dr, dw, de = select([sys.stdin], [], [], 0)
        return dr != []


class GroundTruthRamp():
    """This node listens to keyboard input to label ramps manually"""

    def __init__(self):
        rospy.init_node('ramp_detection', anonymous=True)
        self.pub_ramp = rospy.Publisher(
            '/ground_truth_ramp', Int16, queue_size=10)
        self.c_prev = 0

    def spin(self):
        rate = 100
        r = rospy.Rate(rate)

        x = self.c_prev
        o = 'ground'
        counter = 0
        print('Label the driving situation using the following numbers:')
        print('0 = On the ground, 1 = Driving onto ramp, 3 = Driving off the ramp')
        raw_input('Press a key to start')
        kb = KBHit()

        while not rospy.is_shutdown():
            if kb.kbhit():
                c = kb.getch()
                if ord(c) == 27:  # ESC
                    break
                x, o = self.change_or_repeat(c)
                print(o)
            else:
                # Only print every second
                if counter % rate == 0:
                    print('Still in mode: {}...'.format(x))
                counter += 1
                self.pub_ramp.publish(x)
                r.sleep()

    def change_or_repeat(self, v):
        if unicode(v).isnumeric():
            n = int(v)
            if n == self.c_prev:
                out_msg = 'Already in this status. No change!'
                return int(v), out_msg
            elif 0 <= n <= 2:
                status = self.handling_logic(int(v))
                out_msg = 'Changed status to: {}'.format(status)
                self.c_prev = int(v)
                return int(v), out_msg
            else:
                out_msg = '{} is not a valid input. No change!'.format(v)
                return self.c_prev, out_msg
        else:
            out_msg = '{} is not a valid input. No change!'.format(v)
            return self.c_prev, out_msg

    def handling_logic(self, v):
        if v == 0:
            return 'Ground'
        elif v == 1:
            return 'On ramp'
        elif v == 2:
            return 'Off ramp'


if __name__ == "__main__":
    try:
        rd = GroundTruthRamp()
        rd.spin()
    except rospy.ROSInterruptException:
        pass
