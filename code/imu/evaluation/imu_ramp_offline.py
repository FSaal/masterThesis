#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import rospy
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

    def __init__(self, rate, is_odom_available=False):
        self.rate = rate
        self.car_angle_gyr = 0

        # Variables for calculation of car pitch angle
        self.track_width = 1.52  # eGolf track width [m]
        self.vel_x_car_filt_old = 0  # Initialize previous car velocity
        # (for calc of acceleration)
        self.angle_est = 0  # Initialize previous car pitch angle
        self.angle_est2 = 0  # Initialize previous car pitch angle
        # (for complementary filter)
        self.dist_car = 0  # Travelled distance
        self.v_imu = 0  #
        self.dist_imu = 0
        self.is_odom_available = is_odom_available

        # Moving Average Filter
        self.imu_filt_class = FilterClass()
        self.odom_filt_class = FilterClass()

        # Ramp properties
        self.ramp_props_class = RampProperties(self.rate)

    def spin(self, lin_acc, ang_vel, odom, rot_mat, gyr_bias):
        # Make sensor msgs public
        self.lin_acc_msg = lin_acc
        self.ang_vel_msg = ang_vel
        self.odom_msg = odom

        # Transform
        lin_acc, ang_vel = self.transform_imu(rot_mat, gyr_bias)

        # Calculate pitch angle
        # Calculate pitch angle using different methods
        car_angle_acc = self.pitch_from_acc(lin_acc[0])
        car_angle_gyr = self.pitch_from_gyr(ang_vel[1])
        car_angle_grav = self.pitch_from_grav(lin_acc[0], 0.5)
        car_angle_compl = self.complementary_filter(ang_vel[1], lin_acc[0], 0.99)
        car_angle_compl_grav = self.complementary_filter_grav(ang_vel[1], car_angle_grav, 0.99)

        # if self.ramp_props_class.is_ramp(np.rad2deg(car_angle_compl_grav)):
        #     self.travel_distance_odom(lin_acc[0], car_angle_compl_grav)
        # else:
        #     dist = 0

        angles = [
            car_angle_acc,
            car_angle_gyr,
            car_angle_grav,
            car_angle_compl,
            car_angle_compl_grav,
        ]
        # Calculate average ramp angle
        avg_ramp_angles = []
        for angle in angles:
            rp = RampProperties(self.rate)
            avg_ramp_angle = rp.ramp_angle_est(np.rad2deg(angle), 4, 2, 0.5)
            avg_ramp_angles.append(avg_ramp_angle)
        # ramp_angle = self.ramp_props_class.ramp_angle_est(
        #     np.rad2deg(car_angle_compl_grav), 4, 2, 0.5
        # )

        return (
            lin_acc,
            ang_vel,
            angles,
            avg_ramp_angles,
        )

    def align_imu(self, lin_acc, ang_vel, odom):
        """Get rotation matrix to align imu frame with car frame"""
        # First estimate: Car accelerates after 1s
        start_idx = 1 * self.rate
        for i in range(2):
            # First rotation
            # Quaternion to align z-axes of imu and car
            self.quat1 = self.trafo1(lin_acc[:start_idx])

            # Index where car starts accelerating
            start_idx = self.car_starts_driving(lin_acc, odom)
            # Add a small buffer to make sure that car is really still before
            start_idx -= int(0.25 * self.rate)
        print("Car starts driving after {:.2f} s".format(start_idx / self.rate))

        # Get gyroscope bias
        gyr_bias = np.mean(ang_vel[:start_idx], axis=0)
        print("Gyroscope bias: {}".format(gyr_bias))

        # Second (final) rotation
        # Assumes acceleration occurs for 1 s
        tf_imu_car = self.trafo2(lin_acc[start_idx : start_idx + (1 * self.rate)])

        return tf_imu_car, gyr_bias

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

    def car_starts_driving(self, lin_acc, odom, acc_thresh=0.2):
        """Check if car starts accelerating

        Args:
            acc_thresh (float, optional): Magnitude in xy-plane to surpass. Defaults to 0.2.

        Returns:
            bool: Has car started driving?
        """
        # Use accelerometer if odom is not available
        if not np.any(odom):
            # Rotation matrix to align imu z-axis with car z-axis
            align_z = quaternion_matrix(self.quat1)[:3, :3]
            # Apply rotation
            lin_acc_z_aligned = np.inner(align_z, lin_acc).T
            # Check if an acceleration occurs on x or y-axis
            acc_xy_plane = np.linalg.norm(lin_acc_z_aligned[:, :2], axis=1)
            # Get first index where acceleration surpasses threshold
            start_idx = np.argmax(acc_xy_plane > acc_thresh)
        # Use odometer data if available
        else:
            self.is_odom_available = True
            # Index where wheel speed is not zero anymore
            start_idx = np.flatnonzero(odom[:, 2])[0]
        return start_idx

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

    def transform_imu(self, rot_mat, gyr_bias):
        """Transforms imu msgs from imu to car frame"""
        # Remove gyroscope bias
        self.ang_vel_msg -= gyr_bias
        lin_acc_tf = np.inner(rot_mat, self.lin_acc_msg)
        ang_vel_tf = np.inner(rot_mat, self.ang_vel_msg)

        return lin_acc_tf, ang_vel_tf

    def vel_from_odom(self, odom_msg):
        """Car velocity from wheel speeds"""
        alpha = (odom_msg[2] - odom_msg[3]) / self.track_width
        yaw = alpha * 1 / self.rate
        # 3.6 to convert from km/h to m/s
        vel_x_car = ((odom_msg[2] + odom_msg[3]) / 2) * np.cos(yaw) / 3.6
        return vel_x_car

    def pitch_from_acc(self, acc_x_imu):
        """Calculate pitch angle of car using only the accelerometer"""
        return np.arcsin(acc_x_imu / self.g_mag)

    def pitch_from_gyr(self, ang_vel_y):
        """Calculate pitch angle of car using only the gyroscope"""
        self.car_angle_gyr -= ang_vel_y * (1 / self.rate)
        return self.car_angle_gyr

    def pitch_from_grav(self, acc_x_imu, win):
        """Calculate pitch angle of car using the gravity method"""
        # Calculate car acceleration from odom (if available)
        # Low pass filter and odometry vel
        vel_x_car = self.vel_from_odom(self.odom_msg)
        vel_x_car_filt = self.odom_filt_class.moving_average(vel_x_car, win, self.rate)
        # Car acceleration from car velocity
        acc_x_car = (vel_x_car_filt - self.vel_x_car_filt_old) / (1 / self.rate)
        self.vel_x_car_filt_old = vel_x_car_filt
        # Low-pass filter linear acceleration measured by imu
        acc_x_imu_filt = self.imu_filt_class.moving_average(acc_x_imu, win, self.rate)

        # Car pitch angle (imu acc + odom only)
        car_angle_filt = np.arcsin((acc_x_imu_filt - acc_x_car) / self.g_mag)
        return car_angle_filt

    def complementary_filter(self, gyr, acc_x_imu, K):
        """Sensor fusion imu gyroscope with accelerometer to estimate car pitch angle

        Uses gyroscope data on the short term and the from accelerometer
        calculated pitch angle in the long term (because gyroscope is not drift free,
        but accelerometer is) to estimate the car pitch angle

        :param angle:       Previous angle estimation
        :param gyr:         y-axis gyroscope data (angular velocity)
        :param acc_angle:   Car angle from accelerometer+odometry calculation (low pass filtered)
        :param K:           Time constant response time [0-1], 1: use only gyroscope,
                            0: use only accelerometer
        :return angle_est:  Estimation of current angle
        """
        acc_angle = np.arcsin(acc_x_imu / self.g_mag)
        self.angle_est = K * (self.angle_est - gyr / self.rate) + (1 - K) * acc_angle
        return self.angle_est

    def complementary_filter_grav(self, gyr, acc_angle, K):
        """Sensor fusion imu gyroscope with accelerometer+odom to estimate car pitch angle"""
        self.angle_est = K * (self.angle_est - gyr / self.rate) + (1 - K) * acc_angle
        return self.angle_est

    def travel_distance_odom(self, acc_x_imu, car_angle_est):
        """Distance along x-axis which the car has travelled so far"""
        if self.is_odom_available:
            v_car = self.vel_from_odom(self.odom_msg)
            self.dist += v_car * (1 / self.rate)
        else:
            acc_free_off_g = acc_x_imu - np.sin(car_angle_est) * self.g_mag
            self.v_imu += acc_free_off_g * (1 / self.rate)
            self.dist += self.v_imu * (1 / self.rate)
        return self.dist

    def travel_distance_imu(self, acc_x_imu, car_angle_est):
        """Distance along x-axis which the car has travelled so far"""
        acc_free_off_g = acc_x_imu - np.sin(car_angle_est) * self.g_mag
        self.v_imu += acc_free_off_g * (1 / self.rate)
        self.dist_imu += self.v_imu * (1 / self.rate)
        return self.dist_imu

    def is_ramp(self, car_angle, ramp_thresh=2):
        """Checks if car is on ramp"""
        # Convert angle from radian to degree
        car_angle = np.rad2deg(car_angle)
        if np.abs(car_angle) > ramp_thresh:
            return True
        return False


class FilterClass(object):
    """Filters a signal"""

    def __init__(self):
        self.values = []
        self.sum = 0

    def moving_average(self, val, win, f):
        """Moving average filter, acts as lowpass filter

        :param val:         Measured value (scalar)
        :param window_size: Window size, how many past values should be considered
        :return filtered:   Filtered signal
        """
        # Convert window size from s to samples
        win = int(win * f)
        self.values.append(val)
        self.sum += val
        if len(self.values) > win:
            self.sum -= self.values.pop(0)
        return self.sum / len(self.values)


class RampProperties(object):
    def __init__(self, f):
        self.f = f
        self.ang_buffer = [0]
        self.counter = 0
        self.max_ang = 0
        self.done = False
        self.ang_buffer2 = []

    def ramp_angle_est(self, angle, min_ang, win, buf):
        """Estimate max angle of the ramp

        Args:
            angle (list): Road grade angle estimation over time [deg]
            min_ang (float): Minimum angle to be considered a peak
            win (float): Window length, in which peak will count as global [s]
            buf (float): Window length, before and after peak to use for smoothing [s]
            f (int): Frequency in Hz

        Returns:
            float: Estimated maximum angle of the ramp [deg]
        """
        # Only run until max angle has been found
        if self.done:
            return
        # Convert window length from sec to samples
        win *= self.f
        buf = int(buf * self.f)

        # Iterate through all angles (real time simulation)
        if len(self.ang_buffer) > 2 * win:
            self.ang_buffer.pop(0)
        self.counter += 1
        # Set new max angle and reset counter
        if np.abs(angle) > np.max(np.abs(self.ang_buffer)):
            self.max_ang = np.abs(angle)
            self.counter = 0
        self.ang_buffer.append(angle)
        # Return if no new max angle for a while and angle big enough
        if self.counter > win and self.max_ang > min_ang:
            # Get some val before and after peak
            max_ang_est = np.mean(self.ang_buffer[-self.counter - buf : -self.counter + buf])
            self.done = True
            print("The ramp has an estimated peak angle of {}".format(max_ang_est))
            return max_ang_est

    def is_ramp(self, angle, ramp_thresh=2):
        """Checks if car is on ramp"""
        # TODO: Prevent accidental detection during acceleration phase
        # Convert angle from radian to degree
        if np.abs(angle) > ramp_thresh:
            return True
        return False
