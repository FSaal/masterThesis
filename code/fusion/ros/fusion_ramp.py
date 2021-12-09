#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float32

# TODO: IMU and LiDAR Nodes will both publish their information (angle for IMU) and (angle, distance for LiDAR)
# If angle for IMU hit a threshold, record the covered distance from that point, until angle is going down again
# Prevent accidental IMU detections by checking if LiDAR did also detect something before
# Maybe also use calc avg angle from LiDAR before ramp and compare it to results from IMU

class FusionRamp():
    def __init__(self):
        rospy.init_node('ramp_detection_fusion', anonymous=True)
        rospy.Subscriber('/imu_ang', Float32, self.callback_imu, queue_size=10)
        rospy.Subscriber('/imu_dist', Float32, self.callback_imu2, queue_size=10)
        rospy.Subscriber('/lidar_ang', Float32, self.callback_lidar, queue_size=10)
        rospy.Subscriber('/lidar_dist', Float32, self.callback_lidar2, queue_size=10)
        self.rate = 100
        self.imu_active = False
        self.lidar_active = False

    def callback_imu(self, msg):
        self.imu_ang = msg.data
        self.imu_active = True

    def callback_imu2(self, msg):
        self.imu_dist = msg.data
        self.imu_active = True

    def callback_lidar(self, msg):
        self.lidar_ang = msg.data
        self.lidar_active = True

    def callback_lidar2(self, msg):
        self.lidar_dist = msg.data

    def spin(self):
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if not (self.imu_active and self.lidar_active):
                continue
            if self.imu_ang < 2:
                print('Not on a ramp, but lidar says there is one in {:.2f} m'.format(self.lidar_dist))
            else:
                print('ON A RAMP with ang {:.2f} and dist {:.2f}'.format(self.imu_ang, self.imu_dist))
            # print('IMU {:.2f} vs LIDAR {:.2f}'.format(self.imu_ang, self.lidar_ang))
            r.sleep()



if __name__ == "__main__":
    FR = FusionRamp()
    FR.spin()