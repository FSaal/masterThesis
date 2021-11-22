#!/usr/bin/env python

import numpy as np
from math import isnan
import rospy
import sys
from ackermann_tools.msg import EGolfOdom
from sensor_msgs.msg import Imu
from std_msgs.msg import String, Float32, Float32MultiArray
from tf.transformations import unit_vector, vector_norm, quaternion_multiply, quaternion_conjugate, euler_matrix, quaternion_matrix, quaternion_inverse

class ImuTransform():

    def __init__(self):
        """
        Node to transform imu msgs from imu frame to car frame and republish it
        Also calculation of car pitch angle
        """
        self.init_node = rospy.init_node('imu_transformation', anonymous=True)
        self.sub_imu = rospy.Subscriber('/imu/data', Imu, self.callback_imu, queue_size=1)
        self.sub_odom = rospy.Subscriber('/eGolf/sensors/odometry', EGolfOdom, self.callback_odom, queue_size=1)
        self.pub_pitch = rospy.Publisher('/car_angle_new', Float32, queue_size=5)
        self.pub_debug = rospy.Publisher('/debug', Float32MultiArray, queue_size=5)

        # Variables for transformation from IMU to car frame
        self.flag = False           # True after first imu car frame alignment (pitch, roll)
        self.is_calibrated = False  # True after second alignment (heading)
        self.lin_acc = []           # Buffer used for calibration
        self.g_car = [0, 0, 1]      # g vector in car frame
        self.rec_win_g = 1          # Recording length g vector [s]
        self.rec_win_fwd = 2        # Recording length forward acceleration [s]
        self.f_imu = 100            # [Hz]
        self.f_odom = 100           # [Hz]

        # Variables for calculation of car pitch angle
        self.wheelbase = 2.631      # eGolf wheelbase [m]
        self.vel_x_car_filt_old = 0 # Initialize previous car velocity (for calc of acceleration)
        self.angle_est = 0          # Initialize previous car pitch angle (for complementary filter)
        # Moving Average Filter
        self.imu_filt_class = FilterClass()
        self.odom_filt_class = FilterClass()
        self.win_len = 50           # Window length, results in a delay of win_len/200 [s]
        # Outlier filter
        self.buffer = []
        self.last_change = 0
        self.s = 0

    def callback_imu(self, msg):
        self.lin_acc_msg = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        self.ang_vel_msg = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    def callback_odom(self, msg):
        self.odom_msg = msg

    def spin(self):
        self.rate = 100
        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            if not self.is_calibrated:
                self.rot_mat = self.align_imu()
            else:
                # Transform
                lin_acc, ang_vel = self.transform_imu()
                car_angle = self.pitch_car(lin_acc[0], ang_vel[1])
                print(car_angle)
                self.is_ramp(car_angle)
                # print(self.covered_distance())
                # self.pub_pitch.publish(car_angle)
            r.sleep()
     
    def covered_distance(self):
        # Car velocity
        v = self.vel_from_odom(self.odom_msg)
        
        # Get covered distance by integrating with respect to time
        self.s += v * 1.0/self.rate
        return self.s
     
    def is_ramp(self, car_angle):
        # TODO: Move to fusion node
         if 3 < car_angle:
             print('ON A RAMP')
             print(self.covered_distance())
             return True
         else:
             return False
            
    def align_imu(self):
        # Print calibration msg 1 once and wait for user to start
        if not self.lin_acc and not self.flag:
            print('Welcome to the IMU to car alignment calibration')
            print('In the first step the gravitational acceleration is being measured.')
            raw_input('Please stand still for {}s.\nIf ready, press enter'.format(self.rec_win_g))
            # Convert window length from [s] to samples
            self.rec_win_g = self.rec_win_g * self.rate

        # Collect samples
        if len(self.lin_acc) < self.rec_win_g and not self.flag:
            self.lin_acc.append(self.lin_acc_msg)
            
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
            # Print calibration msg 2 once and wait for user to start
            if not self.lin_acc:
                print('In the second step the yaw angle is being determined.')
                print('For this please accelerate in a straight line forward.')
                raw_input('The recording will automatically stop after {}s.'.format(self.rec_win_fwd))
                print('If ready, press enter')
                # Convert window length from [s] to samples
                self.rec_win_fwd = self.rec_win_fwd*self.rate

            # Collect samples
            if len(self.lin_acc) < self.rec_win_fwd:
                self.lin_acc.append(self.lin_acc_msg)

                if len(self.lin_acc) == self.rec_win_fwd:
                    print('Forward acceleration measured successfully')
                    # Second rotation to correct heading
                    tf_imu_car = self.trafo2(self.lin_acc, self.rot_mat1)
                    # Do not call method
                    self.is_calibrated = True
                    return tf_imu_car
                
    def trafo1(self, lin_acc):
        """Rotation to align IMU measured g-vector with car z-axis (up)
                
        :param lin_acc: Linear acceleration while car stands still
        :return:        Rotation matrix
        """
        # Take average over 2 s
        lin_acc_avg = np.mean(lin_acc, axis=0)
        # Magnitude of measured g vector (should be around 9.81)
        self.g_mag = vector_norm(lin_acc_avg)
        print('Average linear acceleration magnitude: {:.2f}'.format(self.g_mag))
        g_imu = unit_vector(lin_acc_avg)
        quat = self.quat_from_vectors(g_imu, self.g_car)
        rot_mat1 = quaternion_matrix(quat)[:3, :3]
        return rot_mat1

    def quat_from_vectors(self, vec1, vec2):
        """Quaternion that aligns vec1 to vec2
        """
        a, b = unit_vector(vec1), unit_vector(vec2)
        c = np.cross(a, b)
        d = np.dot(a, b)

        # Rotation axis
        ax = c / vector_norm(c)
        # Rotation angle (rad)
        a = np.arctan2(vector_norm(c), d)

        # Quaternion ([x,y,z,w])
        quat = np.append(ax*np.sin(a/2), np.cos(a/2))
        return quat

    def trafo2(self, lin_acc, rot_mat1):
        """Second rotation to align IMU with car frame
        
        Use previously rotated data to correct remaining yaw angle error 
        and combine with previous rotation matrix to get the final rotation matrix

        :param lin_acc:             Linear acceleration while car accelerates
        :param rot_mat1:            Rotation matrix from the first transform step
        :return rot_mat_imu_car:    Rotation matrix to transform IMU frame to car frame
        """
        # Apply first rotation (trafo1)
        lin_acc_rot1 = np.inner(rot_mat1, lin_acc).T
        # Get second rotation
        z_angle = self.find_z_angle(lin_acc_rot1)
        # Get second rotation matrix for yaw correction
        rot_mat2 = euler_matrix(0, 0, z_angle, 'sxyz')[:3, :3]

        # Concatenation of rotation matrices is in reverse order
        rot_mat_imu_car = np.matmul(rot_mat2, rot_mat1)
        return rot_mat_imu_car

    def find_z_angle(self, lin_acc_rot):
        """Yaw angle to align IMU x-axis with x-axis of car

        :param lin_acc_rot: Linear acceleration while car accelerates, after first rotation (z-axes aligned)
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
            rot_mat = euler_matrix(0, 0, i, 'sxyz')[:3,:3]
            z_rot = np.inner(rot_mat, lin_acc_rot).T
            x_mean = np.mean(z_rot[:,0])
            if x_mean > x_mean_old:
                x_mean_old = x_mean
                ang_opt = i

        return ang_opt
                
    def transform_imu(self):
        """Transforms IMU msgs from IMU to car frame"""
        lin_acc_tf = np.inner(self.rot_mat, self.lin_acc_msg)
        ang_vel_tf = np.inner(self.rot_mat, self.ang_vel_msg)
        
        return lin_acc_tf, ang_vel_tf

    def vel_from_odom(self, odom_msg):
        """Car velocity from wheel speeds"""
        alpha = (odom_msg.rear_left_wheel_speed - odom_msg.rear_right_wheel_speed) / self.wheelbase
        yaw = alpha * 1.0 / self.f_odom
        # 3.6 to convert from km/h to m/s
        vel_x_car = ((odom_msg.rear_left_wheel_speed + odom_msg.rear_right_wheel_speed) / 2) * np.cos(yaw) / 3.6
        return vel_x_car

    # TODO: Check if tilt of car due to acceleration is really taken into account (should be)
    def pitch_car(self, acc_x_imu, vel_y_imu):
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
        self.angle_est = self.complementary_filter(self.angle_est, vel_y_imu, car_angle_filt, 0.01)

        # Car pitch angle (imu acc + odom + imu angular vel --> Outlier filter)
        # Problem: angle_filt is delayed by about 0.5s, whereas vel_y_imu is not
        # ang_vel_delayed = vel_y_imu
        # car_angle_filt_hyst = self.outlier_filter(car_angle_filt, acc_x_imu_filt, 10, 3, 1)

        # # Debug part
        # array = [self.acc_x_imu, acc_x_imu_filt, vel_x_car, vel_x_car_filt, acc_x_car,
        #         car_angle_filt, self.angle_est, car_angle_filt_hyst]
        # self.someVals = Float32MultiArray(data=array)
        # self.pub_debug.publish(self.someVals)

        return self.angle_est

    def gyr_change_detection(self, ang_vel, win_len, threshold_upper, threshold_lower):
        """Returns True if sth happend
        :param ang_vel:         Angular velocity
        :param win_len:         Window length over which the angle difference from start to finish is being calculated
        :param threshold_upper: Angular changes per second above this threshold are ignored.
        :param threshold_lower: Angular changes per second below this threshold are ignored.
        :return change_detected: Returns true if treshold_lower < angular change < threshold_upper (reasonable change detected), else False
        """
        # Convert threshold to deg/s to deg/win_len
        threshold_upper_norm = threshold_upper * win_len/float(self.f_imu)
        threshold_lower_norm = threshold_lower * win_len/float(self.f_imu)

        # Add ang vel to buffer
        self.buffer.append(ang_vel)
        # print(len(self.buffer))
        if len(self.buffer) == win_len:
            ang_diff = self.buffer[-1] - self.buffer[0]
            # print(ang_diff)
            self.buffer.pop(0)
            if threshold_lower_norm < abs(ang_diff) < threshold_upper_norm:
                return True
        return False

    def outlier_filter(self, car_angle, ang_vel, win_len, threshold_upper, threshold_lower):
        """Only use acc data if gyr data also detected a change, if not repeat last value
        :param car_angle:       Car angle calculated using IMU acc + odom
        :param change_detected: Boolean, True when threshold_lower < gyr change < threshold_upper
        """
        # Delay ang_vel
        # self.ang_vel_delay.append(ang_vel)
        change_detected = self.gyr_change_detection(ang_vel, win_len, threshold_upper, threshold_lower)
        if change_detected:
            print('Yes Change')
            self.last_change = car_angle
            return car_angle
        else:
            print('No Change')
            return self.last_change

    def complementary_filter(self, angle, gyr, acc_angle, alpha):
        """Sensor fusion IMU gyroscope with accelerometer to estimate car pitch angle

        Uses gyroscope data on the short term and the from accelerometer+odometry calculated pitch angle in the long term
        (because gyroscope is not drift free, but accelerometer is) to estimate the car pitch angle

        :param angle:       Previous angle estimation
        :param gyr:         y-axis gyroscope data (angular velocity)
        :param acc_angle:   Car angle from accelerometer+odometry calculation (low pass filtered)
        :param alpha:       Time constant response time [0-1], 0: use only gyroscope, 1: use only accelerometer
        :return angle_est:  Estimation of current angle
        """
        if isnan(angle):
            angle = 0
        angle_est = (1-alpha)*(angle + gyr*(1.0/self.f_imu)) + alpha*acc_angle
        return angle_est



class FilterClass():
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
    tf = ImuTransform()
    tf.spin()