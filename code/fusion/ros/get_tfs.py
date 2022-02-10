#!/usr/bin/env python

from numpy.lib.utils import source
import rospy
import numpy as np
import tf
import sys
from tf.listener import TransformListener
from tf.transformations import euler_from_quaternion

class FusionRamp():
    def __init__(self):
        rospy.init_node('tf_test', anonymous=True)
        self.tf = TransformListener()
        if len(sys.argv) != 3:
            print('Two command line arguments must be passed')
            print('First argument is source_frame and second is target_frame')
            sys.exit()

    def spin(self):
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            source_frame = sys.argv[1]
            target_frame = sys.argv[2]

            canTransform = self.tf.canTransform(target_frame, source_frame, rospy.Time.now())
            canTransform = '' if canTransform else 'not'
            print('Transformation from {} to {} frame is {} possible'.format(
                source_frame, target_frame, canTransform))

            try:
                tab = "            "
                (trans, rot) = self.tf.lookupTransform(target_frame, source_frame, rospy.Time.now())
                print('{}self.trans = {}\n{}self.quat = {}\n(Euler (rpy)):{:.3f} {:.3f} {:.3f}\n'.format(
                    tab, trans, tab, rot, *np.rad2deg(euler_from_quaternion(rot))))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                r.sleep()
            r.sleep()


if __name__ == "__main__":
    FR = FusionRamp()
    FR.spin()