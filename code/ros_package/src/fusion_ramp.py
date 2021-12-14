#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float32
from ramp_detection.msg import ramp_properties


class FusionRamp(object):
    """Combines data from imu and lidar to detect ramps and measure angle and dist"""
    def __init__(self):
        # ROS stuff
        rospy.init_node('ramp_detection_fusion', anonymous=True)
        self.rate = 100

        # Subscriber and publisher
        rospy.Subscriber('/imu_ang', Float32, self.callback_imu, queue_size=10)
        rospy.Subscriber('/imu_dist', Float32, self.callback_imu2, queue_size=10)
        rospy.Subscriber('/lidar_ang', Float32, self.callback_lidar, queue_size=10)
        rospy.Subscriber('/lidar_dist', Float32, self.callback_lidar2, queue_size=10)
        self.pub_props = rospy.Publisher('/ramp_properties', ramp_properties, queue_size=1)

        # Define subscriber callback messages
        self.imu_ang = None
        self.imu_dist = None
        self.lidar_ang = None
        self.lidar_dist = None

    def callback_imu(self, msg):
        """Get imu angle"""
        self.imu_ang = msg.data

    def callback_imu2(self, msg):
        """Get length of ramp"""
        self.imu_dist = msg.data

    def callback_lidar(self, msg):
        """Get lidar angle"""
        self.lidar_ang = msg.data

    def callback_lidar2(self, msg):
        """Get distance to ramp"""
        self.lidar_dist = msg.data

    def spin(self):
        """Run node until crash or user exit"""
        # Frequency at which node runs
        r = rospy.Rate(self.rate)
        # Wait for both nodes (imu and lidar) to publish
        while self.imu_ang is None or self.lidar_ang is None:
            if rospy.is_shutdown():
                break
            r.sleep()

        while not rospy.is_shutdown():
            # Create message
            msg = ramp_properties()
            msg.ang_imu = self.imu_ang
            msg.ang_lidar = self.lidar_ang
            msg.dist_on_ramp = self.imu_dist
            msg.dist_to_ramp = self.lidar_dist

            if self.imu_ang < 2:
                # print('Not on a ramp, but lidar says there is one in {:.2f} m'.format(
                    # self.lidar_dist))
                msg.on_ramp = False
            else:
                # print('ON A RAMP with ang {:.2f} and dist {:.2f}'.format(
                    # self.imu_ang, self.imu_dist))
                msg.on_ramp = True

            self.pub_props.publish(msg)
            r.sleep()


if __name__ == "__main__":
    try:
        FR = FusionRamp()
        FR.spin()
    except rospy.ROSInterruptException:
        pass
