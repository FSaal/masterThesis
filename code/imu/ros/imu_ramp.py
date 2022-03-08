#!/usr/bin/env python

from __future__ import division, print_function
import numpy as np
import rospy
from ackermann_tools.msg import EGolfOdom
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
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

    def __init__(self):
        # ROS stuff
        rospy.init_node("imu_ramp_detection", anonymous=True)
        #! Select imu topic
        # imu_topic = '/imu/data'
        imu_topic = "/zed2i/zed_node/imu/data"
        #! Select if odom readings are available
        self.is_odom_available = False
        self.rate = 100

        # Subscriber and publisher
        rospy.Subscriber(imu_topic, Imu, self.callback_imu, queue_size=10)
        rospy.Subscriber("/eGolf/sensors/odometry", EGolfOdom, self.callback_odom, queue_size=1)
        self.pub_pitch_compl = rospy.Publisher("/car_angle_compl", Float32, queue_size=5)
        self.pub_pitch_grav = rospy.Publisher("/car_angle_grav", Float32, queue_size=5)

        # Define subscriber callback messages
        self.lin_acc_msg = None
        self.ang_vel_msg = None
        self.odom_msg = None

        # Variables for transformation from imu to car frame
        self.buffer = []  # Buffer used to collect imu msgs to calculate average
        self.buffer2 = []
        self.g_mag = None  # Define gravity magnitude
        self.gyr_bias = None  # Gyroscope bias at start
        self.quat1 = None  # Define quaternion used for first rotation
        self.z_calibrated = False  # True after first imu car frame alignment (pitch, roll)
        self.is_calibrated = False  # True after second alignment (heading)

        # Variables for calculation of car pitch angle
        self.track_width = 1.52  # eGolf track width [m]
        self.imu_acc_x_old = 0  # Initialize previous imu acceleration
        self.vel_x_car_filt_old = 0  # Initialize previous car velocity
        # (for calc of acceleration)
        self.angle_est = 0  # Initialize previous car pitch angle
        self.angle_est2 = 0  # Initialize previous car pitch angle
        # (for complementary filter)
        self.dist = 0  # Travelled distance
        self.v_imu = 0

        # Moving Average Filter
        self.imu_filt_class = FilterClass()
        self.imu_filt_class = FilterClass()
        self.odom_filt_class = FilterClass()
        # Window length, results in a delay of win_len/200 [s]
        self.win_len = 50

    def callback_imu(self, msg):
        """Get msg from imu"""
        self.lin_acc_msg = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ]
        self.ang_vel_msg = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]

    def callback_odom(self, msg):
        """Get msg from odometer"""
        self.odom_msg = msg

    def spin(self):
        """Run node until crash or user exit"""
        # Frequency at which node runs
        r = rospy.Rate(self.rate)
        # Wait until imu msgs
        while self.lin_acc_msg is None:
            if rospy.is_shutdown():
                break
            r.sleep()

        while not rospy.is_shutdown():
            # Get transformation from imu frame to car frame
            if not self.is_calibrated:
                rot_mat = self.align_imu()
            else:
                # Transform
                lin_acc, ang_vel = self.transform_imu(rot_mat)

                # Calculate pitch angle using different methods
                car_angle_grav = self.pitch_from_grav(lin_acc[0])
                car_angle_compl = self.complementary_filter(ang_vel[1], lin_acc[0], 0.99)
                car_angle_compl_grav = self.complementary_filter_grav(
                    ang_vel[1], car_angle_grav, 0.99
                )
                if self.is_ramp(car_angle_compl_grav):
                    self.travel_distance_odom(lin_acc[0], car_angle_compl_grav)
                angs = np.asarray([car_angle_grav, car_angle_compl, car_angle_compl_grav])
                print(np.rad2deg(angs))
            r.sleep()

    def align_imu(self):
        """Get rotation matrix to align imu frame with car frame"""
        # First rotation
        if not self.z_calibrated:
            # Collect imu msgs for some time
            is_list_filled, lin_acc_still = self.collect_messages(1, "lin_acc")
            _, ang_vel_still = self.collect_messages(1, "ang_vel", self.buffer2)
            if is_list_filled:
                # Quaternion to align z-axes of imu and car
                self.quat1 = self.trafo1(lin_acc_still)
                # Get gyroscope bias
                self.gyr_bias = np.mean(ang_vel_still, axis=0)
                print("Gyroscope bias: {}".format(self.gyr_bias))
                # Reset buffer, to allow for forward acceleration measurement
                self.buffer = []
                # Prevent repeated execution
                self.z_calibrated = True
                print("Accelerate forward to complete calibration")

        # Second (final) rotation, only execute after first rotation was
        # calculated and car starts accelerating forward
        if self.z_calibrated and self.car_starts_driving():
            # Collect imu msgs for some time
            is_list_filled, lin_acc = self.collect_messages(0.5, "lin_acc")
            if is_list_filled:
                # Rotation matrix to transform from imu to car frame
                tf_imu_car = self.trafo2(lin_acc)
                self.is_calibrated = True
                return tf_imu_car

    def trafo1(self, lin_acc):
        """First rotation to align imu measured g-vector with car z-axis (up)

        Args:
            lin_acc (list): 1x3 Linear acceleration vector

        Returns:
            list: 1x4 Quaternion to align z-axes of imu and car
        """
        # Take average to reduce influence of noise
        lin_acc_avg = np.mean(lin_acc, axis=0)
        # Magnitude of measured g vector (should be around 9.81)
        self.g_mag = vector_norm(lin_acc_avg)

        # Get quaternion to rotate g_imu onto car z-axis
        quat = self.quat_from_vectors(lin_acc_avg, (0, 0, 1))
        return quat

    def trafo2(self, lin_acc):
        """Second rotation to align imu with car frame

        Args:
            lin_acc (list): 1x3 Linear acceleration with aligned z-axis

        Returns:
            ndarray: 3x3 Rotation matrix
        """
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

    def collect_messages(self, duration, msg_type, buffer=None):
        """Listens for msgs and appends them to list for the desired duration"""
        # Define variable where data will be stored
        buffer = self.buffer if buffer is None else buffer
        # Get either linear acceleration or angular velocity
        if msg_type == "lin_acc":
            msg = self.lin_acc_msg
        elif msg_type == "ang_vel":
            msg = self.ang_vel_msg
        # Add data to list for certain duration [s]
        if len(buffer) < duration * self.rate:
            buffer.append(msg)
            return False, buffer
        # Signal that list is full
        return True, buffer

    def car_starts_driving(self, acc_thresh=0.2):
        """Check if car starts accelerating

        Args:
            acc_thresh (float, optional): Magnitude in xy-plane to surpass. Defaults to 0.2.

        Returns:
            bool: Has car started driving?
        """
        if not self.is_odom_available:
            # Rotation matrix to align imu z-axis with car z-axis
            align_z = quaternion_matrix(self.quat1)[:3, :3]
            # Apply rotation
            lin_acc_z_aligned = np.inner(align_z, self.lin_acc_msg)
            # Check if an acceleration occurs on x or y-axis
            acc_xy_plane = np.linalg.norm(lin_acc_z_aligned[:2])
            if acc_xy_plane > acc_thresh:
                return True
        else:
            # Check if wheels are turning
            if self.odom_msg.rear_left_wheel_speed > 0:
                return True
        return False

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
        # Remove gyro bias
        self.ang_vel_msg -= self.gyr_bias
        ang_vel_tf = np.inner(rot_mat, self.ang_vel_msg)

        return lin_acc_tf, ang_vel_tf

    def vel_from_odom(self, odom_msg):
        """Car velocity from wheel speeds"""
        alpha = (
            odom_msg.rear_left_wheel_speed - odom_msg.rear_right_wheel_speed
        ) / self.track_width
        yaw = alpha * 1.0 / self.rate
        # 3.6 to convert from km/h to m/s
        vel_x_car = (
            ((odom_msg.rear_left_wheel_speed + odom_msg.rear_right_wheel_speed) / 2)
            * np.cos(yaw)
            / 3.6
        )
        return vel_x_car

    def pitch_from_grav(self, acc_x_imu):
        """Calculate pitch angle of car using the gravity method"""
        # Calculate car acceleration from odom (if available)
        acc_x_car = 0
        if self.is_odom_available:
            # Low pass filter and odometry vel
            vel_x_car = self.vel_from_odom(self.odom_msg)
            vel_x_car_filt = self.odom_filt_class.moving_average(vel_x_car, self.win_len)
            # Car acceleration from car velocity
            acc_x_car = (vel_x_car_filt - self.vel_x_car_filt_old) / (1 / self.rate)
            self.vel_x_car_filt_old = vel_x_car_filt
        # Low-pass filter linear acceleration measured by imu
        acc_x_imu_filt = self.imu_filt_class.moving_average(acc_x_imu, self.win_len)

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
        self.angle_est2 = K * (self.angle_est2 - gyr / self.rate) + (1 - K) * acc_angle
        return self.angle_est2

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
