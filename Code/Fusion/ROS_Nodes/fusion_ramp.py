#!/usr/bin/env python

import rospy

# TODO: IMU and LiDAR Nodes will both publish their information (angle for IMU) and (angle, distance for LiDAR)
# If angle for IMU hit a threshold, record the covered distance from that point, until angle is going down again
# Prevent accidental IMU detections by checking if LiDAR did also detect something before
# Maybe also use calc avg angle from LiDAR before ramp and compare it to results from IMU

class FusionRamp():
    rospy.init_node('ramp_detection_fusion', anonymous=True)

if __name__ == "__main__":
    FR = FusionRamp()
    FR.spin()
