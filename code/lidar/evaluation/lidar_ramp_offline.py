#!usr/bin/env python

import numpy as np
import pcl
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
from tf.transformations import (
    euler_from_quaternion,
    euler_matrix,
    unit_vector,
    vector_norm,
    quaternion_matrix,
    euler_from_matrix,
)

# My ROS Node
class VisualDetection(object):
    def __init__(self, ramp_start_x):
        self.ramp_data = [[] for i in range(5)]
        self.arr = np.zeros((1, 3))
        # x coordinates where ramp starts
        self.ramp_start = ramp_start_x
        self.is_calibrated = False

    def spin(self, cloud, pose):
        self.cloud = cloud
        self.pose = pose
        # Convert PointCloud2 msg to numpy array
        pc_array = pointcloud2_to_xyz_array(self.cloud, remove_nans=True)

        # Get transformation from lidar to car frame
        if not self.is_calibrated:
            # Get euler angles (roll and pitch) to align lidar with
            # car frame as well as distance of lidar to the ground
            self.roll, self.pitch, self.height = self.align_lidar(pc_array)
            self.is_calibrated = True

        # Apply lidar to car frame transformation
        pc_array_tf = self.transform_pc(
            pc_array, rpy=(self.roll, self.pitch, 0), translation_xyz=(-2.36, -0.05, -self.height)
        )

        # Reduce point cloud size with a passthrough filter
        pc_array_cut = self.reduce_pc(pc_array_tf, (0, 30), (-2, 2), (-1, 2))

        # Convert numpy array to pcl object
        pc_cut = self.array_to_pcl(pc_array_cut)

        # Downsample point cloud using voxel filter to further decrease size
        pc_small = self.voxel_filter(pc_cut, 0.1)

        # Perform RANSAC until no new planes are being detected
        plane_coor, data = self.plane_detection(pc_small, 20, 4)
        return plane_coor, data, self.relative_to_absolute(pc_array_cut)

    def align_lidar(self, pc_array):
        """Calculates roll and pitch angle of lidar relative to car
        as well as lidar distance to ground.

        Args:
            pc_array (ndarray): Nx3 array of pointcloud with N points

        Returns:
            (float, float, float): Roll, pitch angle in radians,
            lidar distance to ground in m
        """
        # Convert numpy array to pcl point cloud
        pc = self.array_to_pcl(pc_array)
        # Reduce point cloud size a bit by downsampling
        pc_small = self.voxel_filter(pc, 0.05)

        # Get rotation to make ground perpendicular to car z axis
        rot, lidar_height = self.get_ground_plane(pc_small)
        # Calculate euler angles from rotation matrix
        roll, pitch, _ = euler_from_matrix(rot)

        # Display calculated transform
        print("\n__________LIDAR__________")
        print("Euler angles in deg to tf lidar to car frame:")
        print("Roll: {:.2f}\nPitch: {:.2f}".format(np.rad2deg(roll), np.rad2deg(pitch)))
        print("Lidar height above ground: {:.2f} m\n".format(lidar_height))
        return (roll, pitch, lidar_height)

    def get_ground_plane(self, pc, max_iter=10):
        """Calculates rotation matrix to make z axis of car perpendicular
        to the ground plane.

        Args:
            pc (pcl): Full point cloud
            max_iter (int, optional): Allowed tries to detect ground
            before exiting. Defaults to 10.

        Raises:
            RuntimeError: If ground plane has not been found after max_iter tries.

        Returns:
            (ndarray, float): 3x3 rotation matrix, lidar distance to ground in m
        """
        counter = 0
        # Extract different planes using RANSAC until ground has been identified
        while True:
            # Get most dominant plane
            inliers_idx, coefficients = self.ransac(pc)

            # Split point cloud in inliers and outliers of plane
            plane = pc.extract(inliers_idx)
            pc = pc.extract(inliers_idx, negative=True)
            # Calculate normal vector of plane
            est_ground_vec = coefficients[:-1]

            # Test if plane could be the ground
            is_ground = self.test_ground_estimation(plane, est_ground_vec)

            # Exit if ground has been detected
            if is_ground:
                # Get rotation to align plane with ground
                rot = self.level_plane(est_ground_vec)
                # Apply rotation
                plane_tf = np.inner(plane, rot)

                # Calculate estimated distance from lidar to ground
                dist_to_ground = np.mean(plane_tf, axis=0)[2]
                return rot, dist_to_ground

            # Prevent infinite loop
            counter += 1
            if counter == max_iter:
                raise RuntimeError("No ground could be detected.")

    def test_ground_estimation(self, plane, est_ground_vec, lidar_height=1):
        """Tests if detected plane fullfills conditions to be considered the ground.

        Args:
            plane (pcl): Point cloud of plane
            est_ground_vec (list): Normal vector of plane
            lidar_height (float, optional): Height at which lidar is mounted
            above ground (guess conservatively (low)). Defaults to 1 m.

        Returns:
            bool: Is plane the ground plane?
        """
        # Get rotation to align plane with ground
        rot = self.level_plane(est_ground_vec)
        # Apply rotation
        plane_tf = np.inner(plane, rot)

        # Calculate estimated distance from lidar to ground
        dist_to_ground = np.mean(plane_tf, axis=0)[2]

        # Is plane not a side wall (assumption only true if
        # roll angle is below 45 deg)
        is_not_sidewall = abs(est_ground_vec[2]) > 0.7
        # Is detected plane well below lidar?
        is_not_ceiling = dist_to_ground < -lidar_height

        # Check if both conditions are fullfilled
        if is_not_sidewall and is_not_ceiling:
            return True
        return False

    def level_plane(self, plane_normal, roll0=False):
        """Get rotation (matrix) to make plane perpendicular to z axis of car"""
        # Normal vector ground plane in car frame
        # (assuming car stands on a flat surface)
        ground_vec = (0, 0, 1)

        # Get rotation to align detected plane with real ground plane
        quat = self.quat_from_vectors(plane_normal, ground_vec)
        # Calculate euler angles
        roll, pitch, _ = euler_from_quaternion(quat)
        # Ignore roll angle if roll=0 has been measured
        if roll0:
            roll = 0
        # Get rotation matrix (ignoring yaw angle)
        rot = euler_matrix(roll, pitch, 0, "sxyz")[:3, :3]
        return rot

    @staticmethod
    def array_to_pcl(pc_array):
        """Get pcl point cloud from numpy array"""
        pc = pcl.PointCloud()
        pc.from_array(pc_array.astype("float32"))
        return pc

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
        ang = np.arctan2(vector_norm(cross_prod), dot_prod)

        # Quaternion ([x,y,z,w])
        quat = np.append(axis * np.sin(ang / 2), np.cos(ang / 2))
        return quat

    @staticmethod
    def transform_pc(pc, rpy=(0, 0, 0), translation_xyz=(1.7, 0, 1.7)):
        """Transformation from lidar frame to car frame.
        Rotation in rad and translation in m."""
        # Extract euler angles
        roll, pitch, yaw = rpy
        # Extract translations
        transl_x, transl_y, transl_z = translation_xyz

        # Rotation matrix
        rot = euler_matrix(roll, pitch, yaw, "sxyz")[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc, rot)

        # Translation to front of the car
        translation_to_front = [transl_x, transl_y, transl_z]
        # Combine rotation and translation
        pc_tf += translation_to_front
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
        """Detects all planes in point cloud"""
        # Count number of iterations
        counter = 0
        # Standard values for ramp angle and distance if no detection
        ramp_stats = ([], [])
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
                return ramp_stats

            # Ignore planes parallel to the side or front walls
            if self.is_plane_near_ground(n_vec):
                # Check if ramp conditions are fullfilled
                is_ramp, data = self.ramp_detection(plane, n_vec, (3, 9), (2, 6))
                # Ramp conditions met
                if is_ramp:
                    plane_global = self.relative_to_absolute(plane)
                    return (plane_global, data)
            counter += 1
        return ramp_stats

    def relative_to_absolute(self, pc):
        """Transforms relative lidar data to absolute by adding translation rotating"""
        # pc_arr = pc.to_array()
        pc_arr = np.array(pc)

        # Odometer
        pos = self.pose.position
        translation = [pos.x, pos.y, pos.z]
        ori = self.pose.orientation
        quat = [ori.x, ori.y, ori.z, ori.w]
        roll, pitch, yaw = euler_from_quaternion(quat)

        # Rotation matrix
        rot = quaternion_matrix(quat)[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc_arr, rot)

        # Combine rotation and translation
        pc_tf += translation
        return pc_tf

    def ramp_detection(self, plane, n_vec, angle_range, width_range):
        """Checks if conditions to be considered a ramp are fullfilled."""
        # Convert pcl plane to numpy array
        plane_array = plane.to_array()

        # Calculate angle [deg] between normal vector of plane and ground
        angle = self.angle_calc([0, 0, 1], n_vec)
        # Calcualte ramp width (difference between y-values)
        width = max(plane_array[:, 1]) - min(plane_array[:, 1])
        # Ramp distance (average x-value of nearest points of the plane)
        n_nearest = 10
        # Sort array by x-values
        plane_array_sort = np.sort(plane_array[:, 0])
        dist = np.mean(plane_array_sort[:n_nearest])
        # Calculate ramp length (difference between x-values)
        length = np.mean(plane_array_sort[-n_nearest:]) - np.mean(plane_array_sort[:n_nearest])

        # True distance to ramp using odometer from hdl_slam
        true_dist = self.ramp_start - self.pose.position.x

        # Assert ramp angle and width thresholds
        if angle_range[0] <= angle <= angle_range[1] and width_range[0] <= width <= width_range[1]:
            return True, [angle, width, dist, true_dist, length]
        return False, [angle, width, dist, true_dist, length]

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
