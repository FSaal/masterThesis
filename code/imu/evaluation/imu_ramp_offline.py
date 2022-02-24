#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import rospy
from ackermann_tools.msg import EGolfOdom
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, Float32MultiArray
from tf.transformations import (
    euler_from_quaternion,
    quaternion_matrix,
    quaternion_multiply,
    unit_vector,
    vector_norm,
)


class ImuRampDetect(object):
    """Calculate car pitch angle using imu, decide whether or not car is on
    ramp and if so, measure distance and angle of ramp
    """

    def __init__(self, rate):
        self.rate = rate
        self.car_angle_gyr = 0

        # # Variables for transformation from imu to car frame
        # self.buffer = []  # Buffer used to collect imu msgs to calculate average
        # self.g_mag = None  # Define gravity magnitude
        # self.quat1 = None  # Define quaternion used for first rotation
        # self.z_calibrated = False  # True after first imu car frame alignment (pitch, roll)
        # self.is_calibrated = False  # True after second alignment (heading)

        # Variables for calculation of car pitch angle
        self.wheelbase = 2.631  # eGolf wheelbase [m]
        self.vel_x_car_filt_old = 0  # Initialize previous car velocity
        # (for calc of acceleration)
        self.angle_est = 0  # Initialize previous car pitch angle
        self.angle_est2 = 0
        # (for complementary filter)
        self.dist = 0  # Travelled distance

        # Moving Average Filter
        self.imu_filt_class = FilterClass()
        self.odom_filt_class = FilterClass()
        # Window length, results in a delay of win_len/200 [s]
        self.win_len = 50

    def spin(self, lin_acc, ang_vel, odom, rot_mat):
        self.lin_acc_msg = lin_acc
        self.ang_vel_msg = ang_vel
        self.odom_msg = odom

        # Transform
        lin_acc, ang_vel = self.transform_imu(rot_mat)
        # Calculate pitch angle
        car_angles = self.gravity_method(lin_acc[0], ang_vel[1])
        return car_angles

    def align_imu(self, lin_acc):
        """Get rotation matrix to align imu frame with car frame"""
        # First rotation
        # Quaternion to align z-axes of imu and car (use first second)
        self.quat1 = self.trafo1(lin_acc[: self.rate])

        # Second (final) rotation
        # Assumes acceleration occurs between 1-3 s
        tf_imu_car = self.trafo2(lin_acc[self.rate : 3 * self.rate])
        return tf_imu_car

    def trafo1(self, lin_acc):
        """First rotation to align imu measured g-vector with car z-axis (up)"""
        # Take average to reduce influence of noise
        lin_acc_avg = np.mean(lin_acc, axis=0)
        # Magnitude of measured g vector (should be around 9.81)
        self.g_mag = vector_norm(lin_acc_avg)

        # Get quaternion to rotate g_imu onto car z-axis
        quat = self.quat_from_vectors(lin_acc_avg, (0, 0, 1))
        return quat

    def trafo2(self, lin_acc):
        """Second rotation to align imu with car frame"""
        # Take average to reduce influence of noise
        lin_acc_avg = np.mean(lin_acc, axis=0)
        # Get rotation matrix from first alignment
        rot_mat1 = quaternion_matrix(self.quat1)[:3, :3]
        # Apply first rotation (trafo1) for z axis alignment
        lin_acc_rot1 = np.inner(rot_mat1, lin_acc_avg)

        # Ignore gravity (because this is already aligned)
        lin_acc_rot1[2] = 0
        # Get quaternion to rotate, such that forward acc is only measured by x-axis
        quat2 = self.quat_from_vectors(lin_acc_rot1, (1, 0, 0))

        # Apply second rotation to first (always in reverse order) to get final rotation
        quat = quaternion_multiply(quat2, self.quat1)
        # Get euler angles to show the difference between the two frames
        euler_angles = euler_from_quaternion(quat)
        print(
            "Correct angles by (rpy in deg): {:.2f} {:.2f} {:.2f}".format(
                *[np.rad2deg(x) for x in euler_angles]
            )
        )

        # Get rotation matrix from quaternion
        rot_mat_imu_car = quaternion_matrix(quat)[:3, :3]
        return rot_mat_imu_car

    @staticmethod
    def quat_from_vectors(vec1, vec2):
        """Gets quaternion to align vector 1 with vector 2"""
        # Make sure both vectors are unit vectors
        v1_uv, v2_uv = unit_vector(vec1), unit_vector(vec2)
        cross_prod = np.cross(v1_uv, v2_uv)
        dot_prod = np.dot(v1_uv, v2_uv)

        # Rotation axis
        axis = cross_prod / vector_norm(cross_prod)
        # Rotation angle (rad)
        ang = np.arccos(dot_prod)

        # Quaternion ([x,y,z,w])
        quat = np.append(axis * np.sin(ang / 2), np.cos(ang / 2))
        return quat

    def transform_imu(self, rot_mat):
        """Transforms imu msgs from imu to car frame"""
        lin_acc_tf = np.inner(rot_mat, self.lin_acc_msg)
        ang_vel_tf = np.inner(rot_mat, self.ang_vel_msg)

        return lin_acc_tf, ang_vel_tf

    def vel_from_odom(self, odom_msg):
        """Car velocity from wheel speeds"""
        alpha = (odom_msg[2] - odom_msg[3]) / self.wheelbase
        yaw = alpha * 1.0 / self.rate
        # 3.6 to convert from km/h to m/s
        vel_x_car = ((odom_msg[2] + odom_msg[3]) / 2) * np.cos(yaw) / 3.6
        return vel_x_car

    def gravity_method(self, acc_x_imu, vel_y_imu):
        """Calculate pitch angle of car using the gravity method"""
        # Low pass filter both subscribed topics (imu acc and odometry vel)
        vel_x_car = self.vel_from_odom(self.odom_msg)
        vel_x_car_filt = self.odom_filt_class.moving_average(vel_x_car, self.win_len)
        acc_x_imu_filt = self.imu_filt_class.moving_average(acc_x_imu, self.win_len)

        # Car acceleration from car velocity
        acc_x_car = (vel_x_car_filt - self.vel_x_car_filt_old) / (1 / self.rate)
        self.vel_x_car_filt_old = vel_x_car_filt

        # Car pitch angle using only accelerometer
        car_angle_acc = np.arcsin(acc_x_imu_filt / self.g_mag)
        # Car pitch angle using only gyroscope
        self.car_angle_gyr = self.car_angle_gyr - vel_y_imu * (1.0 / self.rate)

        # Car pitch angle using accelerometer method
        car_angle_odom = np.arcsin((acc_x_imu_filt - acc_x_car) / self.g_mag)

        # Car pitch angle (imu acc + odom + imu angular vel --> Complementary filter)
        self.angle_est = self.complementary_filter(self.angle_est, vel_y_imu, car_angle_acc, 0.01)
        self.angle_est2 = self.complementary_filter(
            self.angle_est2, vel_y_imu, car_angle_acc, 0.005
        )

        return [
            car_angle_acc,
            self.car_angle_gyr,
            car_angle_odom,
            self.angle_est,
            self.angle_est2,
        ]

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
        angle_est = (1 - alpha) * (angle + gyr * (1.0 / self.rate)) + alpha * acc_angle
        return angle_est

    def covered_distance(self):
        """How far has car driven"""
        # Car velocity
        v = self.vel_from_odom(self.odom_msg)

        # Get covered distance by integrating with respect to time
        self.dist += v * 1.0 / self.rate
        return self.dist

    def is_ramp(self, car_angle):
        """Checks if car is on ramp"""
        if car_angle > 3:
            print("ON A RAMP")
            print(self.covered_distance())
            return True
        return False


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
