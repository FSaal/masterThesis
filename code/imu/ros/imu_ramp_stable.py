#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import rospy
from ackermann_tools.msg import EGolfOdom
from tf.transformations import (euler_matrix, quaternion_about_axis, quaternion_matrix,
                                quaternion_multiply, unit_vector, vector_norm)
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray


class ImuRampDetect():
    """Calculate car pitch angle using IMU, decide whether or not car is on
    ramp and if so, measure distance and angle of ramp
    """
    def __init__(self):
        # ROS stuff
        rospy.init_node('imu_transformation', anonymous=True)
        self.sub_imu = rospy.Subscriber('/zed2i/zed_node/imu/data', Imu, self.callback_imu, queue_size=1)
        self.sub_odom = rospy.Subscriber(
            '/eGolf/sensors/odometry', EGolfOdom, self.callback_odom, queue_size=1)
        self.pub_pitch = rospy.Publisher('/imu_ang', Float32, queue_size=5)
        self.pub_dist = rospy.Publisher('/imu_dist', Float32, queue_size=5)
        self.pub_debug = rospy.Publisher('/debug', Float32MultiArray, queue_size=5)
        self.rate = 100             # Because both IMU and odom publish with 100 Hz

        # Variables for transformation from IMU to car frame
        self.flag = False           # True after first imu car frame alignment (pitch, roll)
        self.is_calibrated = False  # True after second alignment (heading)
        self.lin_acc = []           # Buffer used for calibration
        self.g_car = (0, 0, 1)      # g vector in car frame
        self.quat1 = [0, 0, 0, 1]   # Init quaternion used for first rotation
        self.rec_win_g = 1          # Recording length g vector [s]
        self.rec_win_fwd = 2        # Recording length forward acceleration [s]

        # Variables for calculation of car pitch angle
        self.wheelbase = 2.631      # eGolf wheelbase [m]
        self.vel_x_car_filt_old = 0 # Initialize previous car velocity (for calc of acceleration)
        self.angle_est = 0          # Initialize previous car pitch angle (for complementary filter)
        self.dist = 0               # Travelled distance

        # Moving Average Filter
        self.imu_filt_class = FilterClass()
        self.odom_filt_class = FilterClass()
        # Window length, results in a delay of win_len/200 [s]
        self.win_len = 50

    def callback_imu(self, msg):
        """Get msg from IMU"""
        self.lin_acc_msg = [
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.ang_vel_msg = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    def callback_odom(self, msg):
        """Get msg from odometer"""
        self.odom_msg = msg

    def spin(self):
        """Run node until crash or user exit"""
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if not self.is_calibrated:
                rot_mat = self.align_imu()
            else:
                # Transform
                lin_acc, ang_vel = self.transform_imu(rot_mat)
                car_angle = self.pitch_car(lin_acc[0], ang_vel[1])
                # print(car_angle)
                if self.is_ramp(car_angle):
                    # Calculate covered distance when on a ramp
                    dist = self.covered_distance()
                    print('On ramp of angle {:.2f} and driven {:.2f} m so far.')
                else:
                    dist = 0
                self.pub_pitch.publish(car_angle)
                self.pub_dist.publish(dist)
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
        """Aligns IMU Frame with car frame"""
        # First calibration step
        # Print calibration msg 1 once and wait for user to start
        if not self.lin_acc and not self.flag:
            print('Welcome to the IMU to car alignment calibration')
            print('In the first step the gravitational acceleration is being measured.')
            # rospy.sleep(0.5)
            raw_input('Please stand still for {}s.\nIf ready, press enter'.format(self.rec_win_g))
            # Convert window length from [s] to samples
            self.rec_win_g = self.rec_win_g * self.rate
        # Collect samples
        if len(self.lin_acc) < self.rec_win_g and not self.flag:
            self.lin_acc.append(self.lin_acc_msg)
            # Take average of recording and calculate first rotmat
            if len(self.lin_acc) == self.rec_win_g:
                print('Gravity acceleration measured successfully')
                # First rotation to align "g-axis" of IMU with z-axis of car
                self.quat1 = self.trafo1(self.lin_acc)
                # Now go to next calibration step
                self.lin_acc = []
                self.flag = True

        # Second calibration step
        if self.flag:
            # Print calibration msg 2 once and wait for user to start
            if not self.lin_acc:
                print('\nIn the second step the yaw angle is being determined.')
                print('For this please accelerate in a straight line forward.')
                # rospy.sleep(1)
                raw_input('The recording will automatically stop after {}s.'.format(
                    self.rec_win_fwd))
                print('If ready, press enter')
                # Convert window length from [s] to samples
                self.rec_win_fwd = self.rec_win_fwd*self.rate
            # Collect samples
            if len(self.lin_acc) < self.rec_win_fwd:
                self.lin_acc.append(self.lin_acc_msg)
                # Take average of recording and calculate final rotmat
                if len(self.lin_acc) == self.rec_win_fwd:
                    print('Forward acceleration measured successfully')
                    # Second rotation to correct heading
                    tf_imu_car = self.trafo2(self.lin_acc, self.quat1)
                    # Do not call method again
                    self.is_calibrated = True
                    return tf_imu_car

    def trafo1(self, lin_acc):
        """Rotation to align IMU measured g-vector with car z-axis (up)
        :param lin_acc: Linear acceleration while car stands still
        :return:        Quaternion
        """
        # Take average over 2 s
        lin_acc_avg = np.mean(lin_acc, axis=0)
        # Magnitude of measured g vector (should be around 9.81)
        self.g_mag = vector_norm(lin_acc_avg)
        print('Average linear acceleration magnitude: {:.2f}'.format(self.g_mag))
        g_imu = unit_vector(lin_acc_avg)
        quat = self.quat_from_vectors(g_imu, self.g_car)
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

    def trafo2(self, lin_acc, quat1):
        """Second rotation to align IMU with car frame
        Use previously rotated data to correct remaining yaw angle error
        and combine with previous rotation matrix to get the final rotation matrix
        :param lin_acc:             Linear acceleration while car accelerates
        :param quat1:               Quaternion from the first transform step
        :return rot_mat_imu_car:    Rotation matrix to transform IMU frame to car frame
        """
        # Apply first rotation (trafo1)
        lin_acc_rot1 = np.inner(quaternion_matrix(quat1)[:3, :3], lin_acc).T
        # Get angle for second rotation (axis is z)
        z_angle = self.find_z_angle(lin_acc_rot1)

        # Get second quaternion for yaw correction
        quat2 = quaternion_about_axis(z_angle, (0, 0, 1))
        # Apply second rotation to first (always in reverse order)
        quat = quaternion_multiply(quat2, self.quat1)
        # Get rotation matrix from quaternion
        rot_mat_imu_car = quaternion_matrix(quat)[:3, :3]
        return rot_mat_imu_car

    @staticmethod
    def find_z_angle(lin_acc_rot):
        """Yaw angle to align IMU x-axis with x-axis of car
        :param lin_acc_rot: Linear acceleration while car accelerates,
                            after first rotation (z-axes aligned)
        :return ang_opt:    Rotation angle around z-axis in rad
        """
        x_mean_old = 0
        # Precision of returned rotation angle in degree
        ang_precision = 0.1
        # Max expected yaw angle deviation of IMU from facing straight forward (expected to be -90)
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
        """Transforms IMU msgs from IMU to car frame"""
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
        # Low pass filter both subscribed topics (IMU acc and odometry vel)
        vel_x_car = self.vel_from_odom(self.odom_msg)
        vel_x_car_filt = self.odom_filt_class.moving_average(vel_x_car, self.win_len)
        acc_x_imu_filt = self.imu_filt_class.moving_average(acc_x_imu, self.win_len)

        # Car acceleration from car velocity
        acc_x_car = (vel_x_car_filt - self.vel_x_car_filt_old) / (1.0 / self.rate)
        self.vel_x_car_filt_old = vel_x_car_filt

        # Car pitch angle (imu acc + odom only)
        car_angle_filt = np.degrees(np.arcsin((acc_x_imu_filt - acc_x_car) / self.g_mag))

        # Car pitch angle (imu acc + odom + imu angular vel --> Complementary filter)
        self.angle_est = self.complementary_filter(self.angle_est, vel_y_imu, car_angle_filt, 0.025)

        return self.angle_est

    def complementary_filter(self, angle, gyr, acc_angle, alpha):
        """Sensor fusion IMU gyroscope with accelerometer to estimate car pitch angle
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


class FilterClass():
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
        return float(self.sum) / len(self.values)


if __name__ == "__main__":
    IRD = ImuRampDetect()
    IRD.spin()