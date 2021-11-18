#!/usr/bin/env python

import rospy
import sys
import numpy as np
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Float32, Float32MultiArray
from tf.transformations import unit_vector, vector_norm, euler_matrix, quaternion_matrix

class ImuTransform():

    def __init__(self):
        """
        Node to transform imu msgs from imu frame to car frame and republish it
        Also calculation of car pitch angle
        """
        self.init_node = rospy.init_node('imu_transformation', anonymous=True)
        self.sub = rospy.Subscriber('/imu/data', Imu, self.callback_imu, queue_size=1)
        self.pub_imu_tf = rospy.Publisher('/imu/data/tf_new', Imu, queue_size=5)
        self.pub_pitch = rospy.Publisher('/car_angle_new', Float32, queue_size=5)
        self.pub_debug = rospy.Publisher('/debug', Float32MultiArray, queue_size=5)

        # Transformation stuff
        self.flag = False
        self.flag2 = False
        self.lin_acc = []
        self.g_car = [0, 0, 1]      # g vector in car frame
        self.rec_win_g = 100        # Recording length g vector -> 1s
        self.rec_win_fwd = 200      # Recording length forward acceleration --> 2s
        self.counter = 0

        # Moving Average Filter
        # self.imu_filt_class = FilterClass()
        # self.odom_filt_class = FilterClass()
        self.smooth_angle = FilterClass()


    def callback_imu(self, msg):
        acc_msg = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        # if not self.lin_acc and not self.flag:
        #     print('Welcome to the IMU to car alignment calibration')
        #     raw_input('In the first step the gravitational acceleration is being measured. Please stand still for {0}s.\nIf ready, press enter\n'.format(self.rec_win_g/100.0))

        # if len(self.lin_acc) < self.rec_win_g  and not self.flag:
        #     self.lin_acc.append(acc_msg)

        #     if len(self.lin_acc) == self.rec_win_g:
        #         print('Gravity acceleration measured successfully')
                # First rotation to align "g-axis" of IMU with z-axis of car
        # self.rot_mat1 = self.trafo1(acc_msg)
        # self.rot_mat1 = self.trafo1(self.lin_acc)
        # Get mount pitch angle
        pitch_angle = self.pitch_imu(acc_msg)
        pitch_angle_smooth = self.smooth_angle.moving_average(pitch_angle, 10)
        if self.counter % 100 == 0:
            print('Average of last 1s: {}'.format(pitch_angle_smooth))
        self.counter += 1
   
    def spin(self):
        self.rate = 100
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.flag2:
                pitch_angle = self.trafo1()
                self.pub_pitch.publish(pitch_angle)
            r.sleep()

    def trafo1(self, lin_acc):
        """ Find quaternion to rotate IMU frame such that only the z-axis measures the gravitational acceleration
        :param lin_acc: Linear acceleration while car stands still
        :return:        Rotation matrix
        """
        # self.g_mag = vector_norm(np.mean(lin_acc, axis=0))
        self.g_mag = vector_norm(lin_acc, axis=0)
        # print('Average linear acceleration magnitude: {}  (should ideally be 9.81)'.format(round(self.g_mag, 2)))
        # g_imu = unit_vector(np.mean(lin_acc, axis=0))
        g_imu = unit_vector(lin_acc, axis=0)
        quat = self.quat_from_vectors(g_imu, self.g_car)
        rot_mat1 = quaternion_matrix(quat)[:3, :3]
        return rot_mat1

    def quat_from_vectors(self, vec1, vec2):
        """ Find the quaternion that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return quat: A quaternion [x, y, z, w] which when applied to vec1, aligns it with vec2.
        """
        a, b = unit_vector(vec1), unit_vector(vec2)
        c = np.cross(a, b)
        d = np.dot(a, b)

        # Rotation axis
        ax = c / vector_norm(c)
        # Rotation angle
        a = np.arctan2(vector_norm(c), d)

        quat = np.append(ax*np.sin(a/2), np.cos(a/2))
        return quat

    def pitch_imu(self, lin_acc):
        """ Calculate the mount inclination angle of the IMU, the misalignment of the z-axis of the IMU pointing straight forward
        :param lin_acc: Linear acceleration vector while car stands still
        """
        # g_imu = unit_vector(np.mean(lin_acc, axis=0))
        # dot = np.dot([0, -1, 0], g_imu)
        g_imu = unit_vector(lin_acc)
        g_imu2D = self.rot3Dto2D(g_imu)
        dot = np.dot([0, -1, 0], g_imu2D)
        pitch = np.arccos(dot)
        print('Mount pitch angle in degree: {:.2f}'.format(np.degrees(pitch)))
        return np.degrees(pitch)

    def rot3Dto2D(self, g_imu):
        """ Find the rotation angle around z-axis, which makes x-axis parallel to the ground and then apply the
        rotation to the input vector
        :param g_imu:       (Normed) 3D gravitational linear acceleration vector of the IMU
        :return g_imu_2D:   Gravitational linear acceleration vector in the yz-plane (because x-value is now almost zero)
        """
        cost = 100
        # Max expected rot angle deviation in deg of x-axis from IMU not being parallel to ground plane
        ang_error = np.deg2rad(30)
        ang_precision = 0.1
        for z_angle in np.arange(-ang_error, ang_error, np.deg2rad(ang_precision)):
            rot_z = euler_matrix(0, 0, z_angle, 'sxyz')[:3,:3]
            cost_new = abs(np.inner(rot_z, g_imu).T[0])
            if cost_new < cost:
                z_angle_opt = z_angle
                cost = cost_new

        rot_z = euler_matrix(0, 0, z_angle_opt, 'sxyz')[:3,:3]
        g_imu_2D = np.inner(rot_z, g_imu).T
        return g_imu_2D

class FilterClass():
    def __init__(self):
        self.values = []
        self.sum = 0

    def moving_average(self, val, window_size):
        self.values.append(val)
        self.sum += val
        if len(self.values) > window_size:
            self.sum -= self.values.pop(0)
        return float(self.sum) / len(self.values)

if __name__ == "__main__":
    tf = ImuTransform()
    tf.spin()