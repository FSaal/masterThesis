#!/usr/bin/env python

from __future__ import division, print_function
import sys
import numpy as np
import pcl
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from tf.transformations import (
    euler_from_quaternion,
    euler_matrix,
    unit_vector,
    vector_norm,
)


class VisualDetection(object):
    """Detect ramps using LIDAR, get angle and distance to ramp"""
    def __init__(self):
        # ROS stuff
        rospy.init_node('lidar_ramp_detection', anonymous=True)
        # Frequency of node [Hz]
        self.rate = 10

        # Evaluating command line argument
        # Select lidar topic (velodyne or robosense), velodyne is default
        lidar_topic = '/velodyne_points'
        if len(sys.argv) == 1:
            pass
        elif sys.argv[1] == 'velodyne':
            pass
        elif sys.argv[1] == 'robosense':
            lidar_topic = '/right/rslidar_points'
        else:
            print("Wrong argument \'{}\'".format(sys.argv[1]))
            print('Either enter \'velodyne\' or \'robosense\' to select lidar topic.')

        # Subscriber and publisher
        rospy.Subscriber(lidar_topic, PointCloud2, self.callback_lidar, queue_size=10)
        self.pub_angle = rospy.Publisher('/lidar_ang', Float32, queue_size=10)
        self.pub_distance = rospy.Publisher('/lidar_dist', Float32, queue_size=10)

        # Define subscriber callback message
        self.cloud = None

        # Flag for lidar car frame alignment, True after tf has been calculated
        self.is_calibrated = False

        # Moving average filter
        self.ang_filter = FilterClass()
        self.dist_filter = FilterClass()

    def callback_lidar(self, msg):
        """Get msg from LIDAR"""
        self.cloud = msg

    def spin(self):
        """Run node until crash or user exit"""
        # Frequency at which node runs
        r = rospy.Rate(self.rate)
        # Wait until lidar msgs are received
        while self.cloud is None:
            if rospy.is_shutdown():
                break
            r.sleep()

        while not rospy.is_shutdown():
            # Convert PointCloud2 msg to numpy array
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.cloud, remove_nans=True)

            # Get transformation from lidar to car frame
            if not self.is_calibrated:
                # Get euler angles (roll and pitch) to align lidar with car frame
                roll, pitch, height = self.align_lidar(pc_array)
                self.is_calibrated = True

            # Apply lidar to car frame transformation (adjust yaw angle manually)
            pc_array_tf = self.transform_pc(pc_array, rpy=(0, pitch, np.pi),
                                            translation_xyz=(1.14, 0.005, -height))

            # Filter unwanted points (to reduce point cloud size) with passthrough filter
            pc_array_cut = self.reduce_pc(pc_array_tf, (0, 30), (-2, 2), (-1, 2))
            # Convert numpy array to pcl object
            pc_cut = self.array_to_pcl(pc_array_cut)
            # Downsample point cloud using voxel filter to further decrease size
            pc_small = self.voxel_filter(pc_cut, 0.1)

            # Perform RANSAC until no new planes are being detected
            ramp_angle, ramp_distance = self.plane_detection(pc_small, 20, 4)

            # Smooth signals
            avg_angle = self.ang_filter.moving_average(ramp_angle, 5)
            avg_dist = self.dist_filter.moving_average(ramp_distance, 5)

            # Publish angle and distance of/to ramp
            self.pub_angle.publish(avg_angle)
            self.pub_distance.publish(avg_dist)
            r.sleep()

    def align_lidar(self, pc_array):
        """Calculate roll and pitch angle to align Lidar with car frame"""
        # Convert numpy array to pcl point cloud
        pc = self.array_to_pcl(pc_array)

        # Get normal vector of ceiling plane
        ceiling_vec_lidar, _ = self.ground_or_ceiling_detection(pc)
        # Normal vector ground plane in car frame
        # (assuming car stands on a flat surface)
        ground_vec_car = [0, 0, 1]

        # Quaternion to align lidar vec to car vec
        quat = self.quat_from_vectors(ceiling_vec_lidar, ground_vec_car)
        # Calculate euler angles
        roll, pitch, _ = euler_from_quaternion(quat)
        # Apply rotation
        rot = euler_matrix(roll, pitch, 0, 'sxyz')[:3, :3]
        pc_array_tf = np.inner(pc, rot)

        # Detect ground plane after applied rotation to measure distance to ground
        # Cut ceiling points, ground is always below LIDAR
        pc_array_wo_ceiling = pc_array_tf[pc_array_tf[:, 2] < 0]
        # Convert numpy array to pcl point cloud
        pc_wo_ceiling = self.array_to_pcl(pc_array_wo_ceiling)
        # Get distance to ground
        _, height = self.ground_or_ceiling_detection(pc_wo_ceiling)

        print('\n__________LIDAR__________')
        print('Euler angles in deg to tf lidar to car frame:')
        print('Roll: {:.2f}\nPitch: {:.2f}'.format(
            *np.degrees([roll, pitch])))
        print('Distance to ground: {:.2f} m\n'.format(height))
        return (roll, pitch, height)

    def ground_or_ceiling_detection(self, pc):
        """Detect ground or ceiling plane and get normal vector and distance to ground"""
        # Initialize
        ground_vec = [0, 0, 0]
        # Prevent accidental wall detection
        while not abs(ground_vec[2]) > 0.7:
            # Get most dominant plane
            inliers_idx, coefficients = self.ransac(pc)

            # Split point cloud in inliers and outliers of plane
            plane = pc.extract(inliers_idx)
            pc = pc.extract(inliers_idx, negative=True)

            # Remove 4th plane coefficient (translation) to get normal vector
            ground_vec = coefficients[:-1]
            # Get distance to ground (from lidar)
            dist_to_ground = np.mean(plane, axis=0)[2]
        return ground_vec, dist_to_ground

    @staticmethod
    def array_to_pcl(pc_array):
        """Get pcl point cloud from numpy array"""
        pc = pcl.PointCloud()
        pc.from_array(pc_array.astype('float32'))
        return pc

    @staticmethod
    def quat_from_vectors(vec1, vec2):
        """Quaternion that aligns vec1 to vec2"""
        # Make sure both vectors are unit vectors
        v1_uv, v2_uv = unit_vector(vec1), unit_vector(vec2)
        cross_prod = np.cross(v1_uv, v2_uv)
        dot_prod = np.dot(v1_uv, v2_uv)

        # Rotation axis
        axis = cross_prod / vector_norm(cross_prod)
        # Rotation angle (rad)
        ang = np.arctan2(vector_norm(cross_prod), dot_prod)

        # Quaternion ([x,y,z,w])
        quat = np.append(axis*np.sin(ang/2), np.cos(ang/2))
        return quat

    @staticmethod
    def transform_pc(pc, rpy=(0, 0, 0), translation_xyz=(1.7, 0, 1.7)):
        """Transformation from Lidar frame to car frame. Rotation in rad and translation in m."""
        # Extract euler angles
        roll, pitch, yaw = rpy
        # Extract translations
        transl_x, transl_y, transl_z = translation_xyz

        # Rotation matrix
        rot = euler_matrix(roll, pitch, yaw, 'sxyz')[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc, rot)

        # Translation to rear wheel axis
        translation_to_base_link = [transl_x, transl_y, transl_z]
        # Translation to front of the car
        translation_to_front = [-3.5, 0, 0]
        # Combine both translations
        translation = np.add(translation_to_base_link, translation_to_front)
        # Combine rotation and translation
        pc_tf += translation
        return pc_tf

    @staticmethod
    def reduce_pc(pc, x_range, y_range, z_range):
        """Removes points outside of box"""
        # Filter array
        pc_cut = pc[
            (pc[:, 0] > x_range[0])
            & (pc[:, 0] < x_range[1])
            & (pc[:, 1] > y_range[0])
            & (pc[:, 1] < y_range[1])
            & (pc[:, 2] > z_range[0])
            & (pc[:, 2] < z_range[1])
            ]
        return pc_cut

    @staticmethod
    def voxel_filter(pc, leaf_size):
        """Downsample point cloud using voxel filter"""
        vgf = pc.make_voxel_grid_filter()
        # Leaf_size is the length of the side of the voxel cube in m
        vgf.set_leaf_size(leaf_size, leaf_size, leaf_size)
        pc_filtered = vgf.filter()
        return pc_filtered

    def plane_detection(self, pc, min_points, max_planes):
        """Detects all planes in point cloud

        Iteratively detects most dominant plane and corresponding normal vector
        of the point cloud until either max_planes amount of iterations have
        been performed or if not enough points (min_points) are left.
        After each detection the plane gets removed from the point cloud.

        :param pc:          [PCL] Point cloud
        :param min_points:  [Int] Min number of points left before exiting
        :param max_planes:  [Int] Max number of planes to detect before exiting
        :return:            (ramp angle, distance), (0,0) if no ramp detected
        """
        # Count number of iterations
        counter = 0
        # Standard values for ramp angle and distance if no detection
        ramp_stats = (0, 0)
        # Detect planes until ramp found or conditions not met anymore
        while pc.size > min_points and counter < max_planes:
            # Detect most dominate plane and get inliers and normal vector
            inliers_idx, coefficients = self.ransac(pc)
            # Normal vector of plane
            n_vec = coefficients[:-1]

            # Split pointcloud in outliers of plane and inliers
            plane = pc.extract(inliers_idx)
            pc = pc.extract(inliers_idx, negative=True)

            # Exit if plane is empty (RANSAC did not find anything)
            if plane.size == 0:
                print('EMPTY')
                return ramp_stats

            # Ignore planes parallel to the side or front walls
            if self.is_plane_near_ground(n_vec):
                # Check if ramp conditions are fullfilled
                is_ramp, ramp_ang, ramp_dist = self.ramp_detection(
                    plane, n_vec, (3, 9), (2, 6))
                # Ramp conditions met
                if is_ramp:
                    return (ramp_ang, ramp_dist)
                # Ground
                else:
                    pass
            # Wall or sth
            else:
                pass
            counter += 1
        return ramp_stats

    def ramp_detection(self, plane, n_vec, angle_range, width_range):
        """Checks if conditions to be considered a ramp are fullfilled.

        The following values of the plane are being calculated and
        checked whether they lie within the desired range:
        - angle (calculated between normal vec of ground_plane and this plane)
        - width
        If all conditions are met return True, else False
        """
        # Convert pcl plane to numpy array
        plane_array = plane.to_array()

        # Calculate angle [deg] between normal vector of plane and ground
        angle = self.angle_calc([0, 0, 1], n_vec)
        # Get ramp width (difference between y-values)
        width = max(plane_array[:, 1]) - min(plane_array[:, 1])
        # Ramp distance (average x-value of nearest points of the plane)
        n_nearest = 10
        dist = np.mean(np.sort(plane_array[:n_nearest, 0]))

        # Assert ramp angle and width thresholds
        if (angle_range[0] <= angle <= angle_range[1]
                and width_range[0] <= width <= width_range[1]):
            return True, angle, dist
        return False, angle, dist

    @staticmethod
    def ransac(pc):
        """Find inliers and normal vector of dominant plane"""
        # 50?
        seg = pc.make_segmenter_normals(50)
        # Doubles the speed if True
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        # How close a point must be to model to be considered inlier
        seg.set_distance_threshold(0.11)
        # normal_distance_weight?
        seg.set_normal_distance_weight(0.01)
        # How many tries
        seg.set_max_iterations(100)
        inliers_idx, coefficients = seg.segment()
        return inliers_idx, coefficients

    @staticmethod
    def is_plane_near_ground(v, threshold=0.8):
        """Returns True if plane is on the ground (and false if e.g. side wall)"""
        # z-axis points up
        return abs(v[2]) > threshold

    @staticmethod
    def angle_calc(v1, v2, degrees=True):
        """Calculate angle between two vectors (planes)"""
        # Assuming both vectors can be rotated alongside one axis to be aligned
        dot = np.dot(v1, v2)

        # Make sure arccos is defined (dot=[0,1]) (should always be the case because
        # v1 and v2 are unit vectors, but probably due to rounding errors not always true)
        if dot <= 1:
            angle = np.arccos(dot)
        else:
            angle = 0

        if degrees is True:
            return np.degrees(angle)
        return angle


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
        # Only add to average if detection
        # Because value is zero if no ramp was detected
        if val != 0:
            self.values.append(val)
            self.sum += val
            # Limit number of values used for filtering
            if len(self.values) > window_size:
                self.sum -= self.values.pop(0)
        # Remove oldest from list if no detection
        else:
            if self.values:
                self.sum -= self.values.pop(0)
        # Prevent division by zero
        if len(self.values) != 0:
            return self.sum / len(self.values)
        else:
            return 0


if __name__ == "__main__":
    try:
        VD = VisualDetection()
        VD.spin()
    except rospy.ROSInterruptException:
        pass
