#!/usr/bin/env python

import rospy
import sys
import numpy as np
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Float32, Float32MultiArray
from ackermann_tools.msg import EGolfOdom
from tf.transformations import unit_vector, vector_norm, quaternion_multiply, quaternion_conjugate, euler_matrix, quaternion_matrix, quaternion_inverse
import message_filters

class ImuTransform():

    def __init__(self):
        """
        Node to transform imu msgs from imu frame to car frame and republish it
        Also calculation of car pitch angle
        """
        self.init_node = rospy.init_node('imu_transformation')
        self.sub = rospy.Subscriber('/imu/data', Imu, self.callback, queue_size=1)
        self.pub_imu_tf = rospy.Publisher('/imu/data/tf', Imu, queue_size=5)
        self.pub_pitch = rospy.Publisher('/car_angle', Float32, queue_size=5)
        self.pub_debug = rospy.Publisher('/debug', Float32MultiArray, queue_size=5)
        
        sub_imu = message_filters.Subscriber('/imu/data/tf', Imu)
        sub_odometry = message_filters.Subscriber('/eGolf/sensors/odometry', EGolfOdom)
        ts = message_filters.ApproximateTimeSynchronizer([sub_imu, sub_odometry], 50, 0.02)
        ts.registerCallback(self.callback_sync)


        # Transformation stuff
        self.flag = False
        self.flag2 = False
        self.lin_acc = []
        self.g_car = [0, 0, 1]      # g vector in car frame
        self.rec_win_g = 100        # Recording length g vector -> 1s
        self.rec_win_fwd = 200      # Recording length forward acceleration --> 2s

        # Odometry stuff
        self.sub_odom = rospy.Subscriber('/eGolf/sensors/odometry', EGolfOdom, self.callback_odometry, queue_size=1)
        self.wheelbase = 2.631  # m
        # Might be wrong, but that is what the rosbag says
        self.f_odom = 100       # Hz
        self.vel_x_car_old = 0
        self.vel_x_car_old_filt = 0
        self.vel_x_car_old_filt_sync = 0
        self.odom_sync = 42
        self.imu_sync = 0


        # Moving Average Filter
        self.imu_filt_class = FilterClass()
        self.odom_filt_class = FilterClass()
        self.synch_msg = 42


    def callback(self, msg):
        acc_msg = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        if not self.lin_acc and not self.flag:
            print('Welcome to the IMU to car alignment calibration')
            raw_input('In the first step the gravitational acceleration is being measured. Please stand still for {0}s.\nIf ready, press enter\n'.format(self.rec_win_g/100.0))

        if len(self.lin_acc) < self.rec_win_g  and not self.flag:
            self.lin_acc.append(acc_msg)

            if len(self.lin_acc) == self.rec_win_g:
                print('Gravity acceleration measured successfully')
                # First rotation to align "g-axis" of IMU with z-axis of car
                self.rot_mat1 = self.trafo1(self.lin_acc)
                # Get mount pitch angle
                self.pitch_imu(self.lin_acc)
                # Now go to next calibration step
                self.lin_acc = []
                self.flag = True

        if self.flag:
            if not self.lin_acc:
                raw_input('\nIn the second step the yaw angle is being determined. For this please accelerate in a straight line forward.\nThe recording will automatically stop afer {0}s.\nIf ready, press enter\n'.format(self.rec_win_fwd/100.0))

            if len(self.lin_acc) < self.rec_win_fwd:
                self.lin_acc.append(acc_msg)

                if len(self.lin_acc) == self.rec_win_fwd:
                    print('Forward acceleration measured successfully')
                    # Second rotation to correct heading
                    self.tf_imu_car = self.trafo2(self.lin_acc, self.rot_mat1)
                    print('The calculation of the rotation matrix has finished\nPublishing of the transformed linear acceleration starts now!')

            else:
                self.imu_msg = msg
                self.flag2 = True

    def callback_sync(self, imu, odom):
        self.imu_sync = imu
        self.odom_sync = odom

    def callback_odometry(self, msg):
        self.odom_msg = msg
   
    def spin(self):
        self.rate = 75
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if self.flag2:
                # Transform IMU msg to car frame and publish
                acc_msg = [self.imu_msg.linear_acceleration.x, self.imu_msg.linear_acceleration.y, self.imu_msg.linear_acceleration.z]
                vel_msg = [self.imu_msg.angular_velocity.x, self.imu_msg.angular_velocity.y, self.imu_msg.angular_velocity.z]
                acc_msg_tf = np.inner(self.tf_imu_car, acc_msg).T
                vel_msg_tf = np.inner(self.tf_imu_car, vel_msg).T
                self.imu_msg.angular_velocity.x = vel_msg_tf[0]
                self.imu_msg.angular_velocity.y = vel_msg_tf[1]
                self.imu_msg.angular_velocity.z = vel_msg_tf[2]
                self.imu_msg.linear_acceleration.x = acc_msg_tf[0]
                self.imu_msg.linear_acceleration.y = acc_msg_tf[1]
                self.imu_msg.linear_acceleration.z = acc_msg_tf[2]
                self.pub_imu_tf.publish(self.imu_msg)
                self.acc_x_imu = acc_msg_tf[0]
                
                # Perform calculation of car pitch angle
                car_angle = self.pitch_car()
                self.pub_pitch.publish(car_angle)

                r.sleep()

    def vel_from_odom(self, odom_msg):
        alpha = (odom_msg.rear_left_wheel_speed - odom_msg.rear_right_wheel_speed) / self.wheelbase
        yaw = alpha * 1.0 / self.f_odom
        # 3.6 to convert from km/h to m/s
        vel_x_car = ((odom_msg.rear_left_wheel_speed + odom_msg.rear_right_wheel_speed) / 2) * np.cos(yaw) / 3.6
        return vel_x_car

    def pitch_car(self):
        # print('Unsynched: {}'.format(self.imu_msg.header.stamp - self.odom_msg.header.stamp))
        # print('Synched: {}'.format(self.synch_msg))

        # Smooth (soon filter) both subscribed topics (IMU and odometry)
        vel_x_car = self.vel_from_odom(self.odom_msg)
        vel_x_car_filt = self.odom_filt_class.moving_average(vel_x_car, 50)
        acc_x_imu_filt = self.imu_filt_class.moving_average(self.acc_x_imu, 50)

        # Original data (for debug purposes, will be removed in later versions)
        # acc_x_car = (vel_x_car - self.vel_x_car_old) / (1.0 / self.rate)
        # self.vel_x_car_old = vel_x_car
        # car_angle = np.degrees(np.arcsin((self.acc_x_imu - acc_x_car) / self.g_mag))

        # Car acceleration from car velocity
        acc_x_car_filt = (vel_x_car_filt - self.vel_x_car_old_filt) / (1.0 / self.rate)
        self.vel_x_car_old_filt = vel_x_car_filt

        # with synched
        if self.odom_sync == 42:
            vel_x_car_sync = 0
            acc_x_imu_filt_sync = 0
        else:
            vel_x_car_sync = self.vel_from_odom(self.odom_sync)
            acc_x_imu_filt_sync = self.imu_filt_class.moving_average(self.imu_sync.linear_acceleration.x, 50)

        vel_x_car_filt_sync = self.odom_filt_class.moving_average(vel_x_car_sync, 50)
        acc_x_car_filt_sync = (vel_x_car_filt_sync - self.vel_x_car_old_filt_sync) / (1.0 / self.rate)
        self.vel_x_car_old_filt_sync = vel_x_car_filt_sync
        car_angle_filt_sync = np.degrees(np.arcsin((acc_x_imu_filt_sync - acc_x_car_filt_sync) / self.g_mag))


        # Actual calculation of the car pitch angle (using smoothed data)
        car_angle_filt = np.degrees(np.arcsin((acc_x_imu_filt - acc_x_car_filt) / self.g_mag))

        # # Debug part
        array = [self.acc_x_imu, acc_x_imu_filt, vel_x_car, vel_x_car_filt, acc_x_car_filt,
                car_angle_filt, car_angle_filt_sync, self.imu_msg.angular_velocity.y]
        self.someVals = Float32MultiArray(data=array)
        self.pub_debug.publish(self.someVals)

        return car_angle_filt

    def trafo1(self, lin_acc):
        """ Find quaternion to rotate IMU frame such that only the z-axis measures the gravitational acceleration
        :param lin_acc: Linear acceleration while car stands still
        :return:        Rotation matrix
        """
        self.g_mag = vector_norm(np.mean(lin_acc, axis=0))
        print('Average linear acceleration magnitude: {}  (should ideally be 9.81)'.format(round(self.g_mag, 2)))
        g_imu = unit_vector(np.mean(lin_acc, axis=0))
        quat = self.quat_from_vectors(g_imu, self.g_car)
        rot_mat1 = quaternion_matrix(quat)[:3, :3]
        return rot_mat1

    def trafo2(self, lin_acc, rot_mat1):
        """ Find rotation matrix to correct yaw angle error and combine with previous rotation matrix
        :param lin_acc:             Linear acceleration while car accelerates
        :param rot_mat1:            Rotation matrix from the first transform step
        :return rot_mat_imu_car:    Rotation matrix to transform IMU frame to car frame
        """
        # Apply first rotation (trafo1)
        lin_acc_rot1 = np.inner(rot_mat1, lin_acc).T
        # Get second rotation
        z_angle = self.find_z_angle(lin_acc_rot1)
        # Second rotation matrix for yaw correction
        rot_mat2 = euler_matrix(0, 0, z_angle, 'sxyz')[:3, :3]

        rot_mat_imu_car = np.matmul(rot_mat2, rot_mat1)
        return rot_mat_imu_car

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

    def find_z_angle(self, lin_acc_rot):
        """ Find the yaw angle to align IMU x-axis with x-axis of car
        :param lin_acc_rot: Linear acceleration while car accelerates after first rotation
        :return ang_opt:    Rotation angle around z-axis in rad
        """
        x_mean_old = 0
        # Precision of returned rotation angle in degree
        ang_precision = 0.1
        # Max expected yaw angle deviation of IMU from facing straight forward (expected to be 90)
        ang_error = np.deg2rad(180)

        # Find rotation angle around z-axis which maximizes the measurements of x-axis
        for i in np.arange(-ang_error, ang_error, np.deg2rad(ang_precision)):
            # Apply rotation of i rad around z-axis
            rot_mat = euler_matrix(0, 0, i, 'sxyz')[:3,:3]
            z_rot = np.inner(rot_mat, lin_acc_rot).T

            x_mean = np.mean(z_rot[:,0])
            if x_mean > x_mean_old:
                x_mean_old = x_mean
                ang_opt = i
        print('Yaw angle was corrected by {} degree'.format(np.degrees(ang_opt)))
        return ang_opt

    def pitch_imu(self, lin_acc):
        """ Calculate the mount inclination angle of the IMU, the misalignment of the z-axis of the IMU pointing straight forward
        :param lin_acc: Linear acceleration vector while car stands still
        """
        g_imu = unit_vector(np.mean(lin_acc, axis=0))
        g_imu2D = self.rot3Dto2D(g_imu)
        dot = np.dot([0, -1, 0], g_imu2D)
        pitch = np.arccos(dot)
        print('Mount pitch angle in degree: {}'.format(round(np.degrees(pitch), 3)))

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