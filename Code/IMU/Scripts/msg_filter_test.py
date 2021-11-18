#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
from ackermann_tools.msg import EGolfOdom
from std_msgs.msg import String
import message_filters


def callback(imu, odom):
    print('Synched: {}'.format(imu.header.stamp - odom.header.stamp))

def listener():
    rospy.init_node('listener')
    sub_imu = message_filters.Subscriber('/imu/data', Imu)
    sub_odom = message_filters.Subscriber('/eGolf/sensors/odometry', EGolfOdom)
    subImu = rospy.Subscriber('/imu/data', Imu, callback_imu, queue_size=1)
    subOdom = rospy.Subscriber('/eGolf/sensors/odometry', EGolfOdom, callback_odom, queue_size=1)
    # cache = message_filters.Cache()
    ts = message_filters.ApproximateTimeSynchronizer([sub_imu, sub_odom], 50, 0.02)
    # ts = message_filters.TimeSynchronizer([sub_imu, sub_odom], 50)
    ts.registerCallback(callback)
    rospy.spin()

def callback_imu(msg):
    global imu_msg
    imu_msg = msg

def callback_odom(msg):
    global odom_msg
    odom_msg = msg
    diff()

def diff():
    print('Unsynched: {}'.format(imu_msg.header.stamp - odom_msg.header.stamp))

# imu_msg = 0
# odom_msg = 0
listener()
# rospy.spin()