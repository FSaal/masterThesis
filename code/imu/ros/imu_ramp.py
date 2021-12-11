#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import message_filters
import rospy
from ackermann_tools.msg import EGolfOdom
from tf.transformations import (euler_from_quaternion, euler_matrix, quaternion_about_axis,
                                quaternion_matrix, quaternion_multiply, unit_vector, vector_norm)
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray
from message_filters import Cache, Subscriber
# import message_filter


class ImuRampDetect(object):
    """Calculate car pitch angle using imu, decide whether or not car is on
    ramp and if so, measure distance and angle of ramp
    """
    def __init__(self):
        # ROS stuff
        rospy.init_node('imu_transformation', anonymous=True)
        # imu_topic = '/imu/data'
        imu_topic = '/zed2i/zed_node/imu/data'
        self.sub_imu = rospy.Subscriber(imu_topic, Imu, self.callback_imu, queue_size=10)
        self.sub_odom = rospy.Subscriber(
            '/eGolf/sensors/odometry', EGolfOdom, self.callback_odom, queue_size=1)
        self.pub_pitch = rospy.Publisher('/car_angle_new', Float32, queue_size=5)
        self.pub_debug = rospy.Publisher('/debug', Float32MultiArray, queue_size=5)
        self.rate = 100             # Because both imu and odom publish with 100 Hz
        self.is_subbed = False      # True if imu topic has started publishing

        # Variables for transformation from imu to car frame
        self.z_calibrated = False   # True after first imu car frame alignment (pitch, roll)
        self.is_calibrated = False  # True after second alignment (heading)
        self.buffer = []            # Buffer used for calibration
        self.quat1 = [0, 0, 0, 1]   # Init quaternion used for first rotation
        self.rec_win_g = 1          # Recording length g vector [s]
        self.rec_win_fwd = 2        # Recording length forward acceleration [s]

        # Variables for calculation of car pitch angle
        self.wheelbase = 2.631      # eGolf wheelbase [m]
        self.vel_x_car_filt_old = 0 # Initialize previous car velocity
                                    # (for calc of acceleration)
        self.angle_est = 0          # Initialize previous car pitch angle
                                    # (for complementary filter)
        self.dist = 0               # Travelled distance
        self.lin_acc_msg = 0

        # Moving Average Filter
        self.imu_filt_class = FilterClass()
        self.odom_filt_class = FilterClass()
        # Window length, results in a delay of win_len/200 [s]
        self.win_len = 50

        self.sub = message_filters.Subscriber('/zed2i/zed_node/imu/data', Imu)
        self.cache = message_filters.Cache(self.sub, 400)

    def callback_imu(self, msg):
        # print('callback')
        """Get msg from imu"""
        self.lin_acc_msg = [
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.ang_vel_msg = [
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        self.is_subbed = True

    def callback_odom(self, msg):
        """Get msg from odometer"""
        self.odom_msg = msg

    def spin(self):
        """Run node until crash or user exit"""
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            # Wait until imu msgs are received
            if not self.is_subbed:
                continue
            # Get transformation from imu frame to car frame
            if not self.is_calibrated:
                rot_mat = self.align_imu()
            else:
                print(rot_mat)
                # Transform
                lin_acc, ang_vel = self.transform_imu(rot_mat)
                # Calculate pitch angle
                car_angle = self.pitch_car(lin_acc[0], ang_vel[1])
                print(car_angle)
                self.is_ramp(car_angle)
                print(self.covered_distance())
                self.pub_pitch.publish(car_angle)
            r.sleep()

    def covered_distance(self):
        """How far has car driven"""
        # Car velocity
        v = self.vel_from_odom(self.odom_msg)

        # Get covered distance by integrating with respect to time
        self.dist += v * 1.0/self.rate
        return self.dist

    def is_ramp(self, car_angle):
        """Checks if car is on ramp"""
        if car_angle > 3:
            print('ON A RAMP')
            print(self.covered_distance())
            return True
        return False

    def align_imu(self):
        # First rotation, collect msgs for 1 s
        if self.samples_collected(1) and not self.z_calibrated:
            # Quaternion to align z-axes of imu and car
            self.quat1 = self.trafo1()
            # Reset buffer, to allow for forward acceleration measurement
            self.buffer = []
            # Prevent repeated execution
            print('Accelerate forward to complete calibration')
            self.z_calibrated = True

        # Second (final) rotation after first rotation was calculated
        if self.samples_collected(1) and self.z_calibrated and self.is_car_accelerating():
            # Rotation matrix to transform from imu to car frame
            tf_imu_car = self.trafo2()
            self.is_calibrated = True
            return tf_imu_car

    def samples_collected(self, duration, buffer=None):
        buffer = self.buffer if buffer is None else buffer
        if len(self.buffer) < duration * self.rate:
            self.buffer.append(self.lin_acc_msg)
            return False
        return True

    def is_car_accelerating(self):
        """Detects if car is accelerating (True) or not (False)"""
        # Get transformation matrix for z-axes alignment
        rot_mat1 = quaternion_matrix(self.quat1)[:3, :3]
        # Apply transformation, such that z-axes of imu and car are aligned
        lin_acc_rot = np.inner(rot_mat1, self.lin_acc_msg).T

        # Whatever
        lin_acc_xy = vector_norm(lin_acc_rot[:2])
        # TODO get threshold from imu data (not arbitrary)
        if lin_acc_xy > 0.4:
            return True
        return False

    def trafo1(self):
        """First rotation to align imu measured g-vector with car z-axis (up)"""
        # Take average ov
        lin_acc_avg = np.mean(self.buffer, axis=0)
        # Magnitude of measured g vector (should be around 9.81)
        self.g_mag = vector_norm(lin_acc_avg)
        print('Average linear acceleration magnitude: {:.2f}'.format(self.g_mag))
        g_imu = unit_vector(lin_acc_avg)
        quat = self.quat_from_vectors(g_imu, (0, 0, 1))
        r, p, y = euler_from_quaternion(quat)
        # Apply first rotation (trafo1) for z axis alignment
        print(np.rad2deg([r, p, y]))
        return quat

    @staticmethod
    def quat_from_vectors(vec1, vec2):
        """Quaternion that aligns vec1 to vec2"""
        a, b = unit_vector(vec1), unit_vector(vec2)
        c = np.cross(a, b)
        d = np.dot(a, b)

        # Rotation axis
        axis = c / vector_norm(c)
        # Rotation angle (rad)
        ang = np.arctan2(vector_norm(c), d)

        # Quaternion ([x,y,z,w])
        quat = np.append(axis*np.sin(ang/2), np.cos(ang/2))
        return quat

    def trafo2(self):
        """Second rotation to align imu with car frame

        Use previously rotated data to correct remaining yaw angle error
        and combine with previous rotation matrix to get the final rotation matrix

        :param lin_acc:             Linear acceleration while car accelerates
        :param quat1:               Quaternion from the first transform step
        :return rot_mat_imu_car:    Rotation matrix to transform imu frame to car frame
        """
        rot_mat1 = quaternion_matrix(self.quat1)[:3, :3]
        # Apply first rotation (trafo1) for z axis alignment
        lin_acc_rot1 = np.inner(rot_mat1, self.buffer).T
        # Get angle for second rotation (axis is z)
        z_angle = self.find_z_angle(lin_acc_rot1)
        # Get second quaternion for yaw correction
        quat2 = quaternion_about_axis(z_angle, (0, 0, 1))
        # Apply second rotation to first (always in reverse order)
        quat = quaternion_multiply(quat2, self.quat1)
        r,p,y = euler_from_quaternion(quat)
        print(np.rad2deg([r,p,y]))
        # Get rotation matrix from quaternion
        rot_mat_imu_car = quaternion_matrix(quat)[:3, :3]
        return rot_mat_imu_car

    @staticmethod
    def find_z_angle(lin_acc_rot):
        """Yaw angle to align imu x-axis with x-axis of car
        :param lin_acc_rot: Linear acceleration while car accelerates,
                            after first rotation (z-axes aligned)
        :return ang_opt:    Rotation angle around z-axis in rad
        """
        x_mean_old = 0
        # Precision of returned rotation angle in degree
        ang_precision = 0.1
        # Max expected yaw angle deviation of imu from facing straight forward (expected to be -90)
        ang_error = np.deg2rad(180)

        # Find rotation angle around z-axis which maximizes the measurements of x-axis
        for i in np.arange(-ang_error, ang_error, np.deg2rad(ang_precision)):
            # Apply rotation of i rad around z-axis
            rot_mat = euler_matrix(0, 0, i, 'sxyz')[:3, :3]
            z_rot = np.inner(rot_mat, lin_acc_rot).T
            x_mean = np.mean(z_rot[:, 0])
            if x_mean > x_mean_old:
                x_mean_old = x_mean
                ang_opt = i

        return ang_opt

    def transform_imu(self, rot_mat):
        """Transforms imu msgs from imu to car frame"""
        lin_acc_tf = np.inner(rot_mat, self.lin_acc_msg)
        ang_vel_tf = np.inner(rot_mat, self.ang_vel_msg)

        return lin_acc_tf, ang_vel_tf

    def vel_from_odom(self, odom_msg):
        """Car velocity from wheel speeds"""
        alpha = (odom_msg.rear_left_wheel_speed - odom_msg.rear_right_wheel_speed) / self.wheelbase
        yaw = alpha * 1.0 / self.rate
        # 3.6 to convert from km/h to m/s
        vel_x_car = ((odom_msg.rear_left_wheel_speed + odom_msg.rear_right_wheel_speed)
                     / 2) * np.cos(yaw) / 3.6
        return vel_x_car

    def pitch_car(self, acc_x_imu, vel_y_imu):
        """Get pitch"""
        # Low pass filter both subscribed topics (imu acc and odometry vel)
        vel_x_car = self.vel_from_odom(self.odom_msg)
        vel_x_car_filt = self.odom_filt_class.moving_average(vel_x_car, self.win_len)
        acc_x_imu_filt = self.imu_filt_class.moving_average(acc_x_imu, self.win_len)

        # Car acceleration from car velocity
        acc_x_car = (vel_x_car_filt - self.vel_x_car_filt_old) / (1.0 / self.rate)
        self.vel_x_car_filt_old = vel_x_car_filt

        # Car pitch angle (imu acc + odom only)
        car_angle_filt = np.degrees(np.arcsin((acc_x_imu_filt - acc_x_car) / self.g_mag))

        # Car pitch angle (imu acc + odom + imu angular vel --> Complementary filter)
        self.angle_est = self.complementary_filter(self.angle_est, vel_y_imu, car_angle_filt, 0.01)

        return self.angle_est

    def complementary_filter(self, angle, gyr, acc_angle, alpha):
        """Sensor fusion imu gyroscope with accelerometer to estimate car pitch angle

        Uses gyroscope data on the short term and the from accelerometer+odometry
        calculated pitch angle in the long term (because gyroscope is not drift free,
        but accelerometer is) to estimate the car pitch angle

        :param angle:       Previous angle estimation
        :param gyr:         y-axis gyroscope data (angular velocity)
        :param acc_angle:   Car angle from accelerometer+odometry calculation (low pass filtered)
        :param alpha:       Time constant response time [0-1], 0: use only gyroscope,
                            1: use only accelerometer
        :return angle_est:  Estimation of current angle
        """
        if np.isnan(angle):
            angle = 0
        angle_est = (1-alpha)*(angle + gyr*(1.0/self.rate)) + alpha*acc_angle
        return angle_est


class FilterClass(object):
    """Filters a signal"""
    def __init__(self):
        self.values = []
        self.sum = 0

    def moving_average(self, val, window_size):
        """Moving average filter, acts as lowpass filter

        :param val:         Measured value (scalar)
        :param window_size: Window size, how many past values should be considered
        :return filtered:   Filtered signal
        """
        self.values.append(val)
        self.sum += val
        if len(self.values) > window_size:
            self.sum -= self.values.pop(0)
        return self.sum / len(self.values)


if __name__ == "__main__":
    try:
        IRD = ImuRampDetect()
        IRD.spin()
    except rospy.ROSInterruptException:
        pass
