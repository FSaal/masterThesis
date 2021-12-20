#!/usr/bin/env python

from numpy.lib.utils import source
import rospy
import numpy as np
import tf
from tf.listener import TransformListener
from tf.transformations import euler_from_quaternion

class FusionRamp():
    def __init__(self):
        rospy.init_node('tf_test', anonymous=True)
        self.tf = TransformListener()

    def spin(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            target_frame = '/zed2i_left_camera_optical_frame'
            source_frame = '/lidar'

            # canTransform = self.tf.canTransform(target_frame, source_frame, rospy.Time.now())
            # canTransform = '' if canTransform else 'not'
            # print('Transformation from {} to {} frame is {} possible'.format(
            #     source_frame, target_frame, canTransform))

            try:
                (trans, rot) = self.tf.lookupTransform(target_frame, source_frame, rospy.Time.now())
                print('trans = {}\nquat = {}\n(Euler (rpy)):{:.3f} {:.3f} {:.3f}\n'.format(
                    trans, rot, *np.rad2deg(euler_from_quaternion(rot))))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            r.sleep()


if __name__ == "__main__":
    FR = FusionRamp()
    FR.spin()