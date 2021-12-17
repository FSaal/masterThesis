#!/usr/bin/env python
"""Calculates the estimated average roll, pitch angle of LIDAR and
also the distance from the lidar to the ground for multiple rosbags (offline)"""

from __future__ import division, print_function
import os
import glob
import numpy as np
import pcl
import rosbag
import ros_numpy
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Float32
from tf.transformations import (
    euler_from_matrix,
    euler_from_quaternion,
    euler_matrix,
    unit_vector,
    vector_norm,
)

class AlignGround():
    def __init__(self, pc_msg, show_plane=False):
        self.pc_msg = pc_msg
        self.show_plane = show_plane

    def align_lidar(self):
        """Calculate roll and pitch angle to align Lidar with car frame"""
        # Convert lidar msg to numpy array
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(
            self.pc_msg, remove_nans=True)
        # Convert numpy array to pcl point cloud
        pc = self.array_to_pcl(pc_array)
        # Reduce point cloud size a bit by downsampling
        pc_small = self.voxel_filter(pc, 0.05)

        # Rotation to make ground perpendicular to car z axis
        rot, lidar_height = self.get_ground_plane(pc_small)
        roll, pitch, _ = euler_from_matrix(rot)

        return (roll, pitch, lidar_height)

    def get_ground_plane(self, pc, max_iter=10):
        """123"""
        counter = 0
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

                # Publish ground to check manually if it is ground
                if self.show_plane:
                    self.publish_pc(plane, 'hey')

                return rot, dist_to_ground

            # Prevent infinite loop
            counter += 1
            if counter == max_iter:
                raise RuntimeError ('Ground could not be detected')

    def test_ground_estimation(self, plane, est_ground_vec, lidar_height=1):
        """Tests whether or not plane is ground (True) or not (False)
        :param: plane:          [PCL] Point cloud
        :param: est_ground_vec  [List]
        :param: lidar_height    [Float]
        :return:                [Bool]
        """
        # Get rotation to align plane with ground
        rot = self.level_plane(est_ground_vec)
        # Apply rotation
        plane_tf = np.inner(plane, rot)

        # Calculate estimated distance from lidar to ground
        dist_to_ground = np.mean(plane_tf, axis=0)[2]

        # Check if both conditions are fullfilled
        # Is plane not a side wall (assumption only true if
        # roll angle is below 45 deg)
        is_not_sidewall = abs(est_ground_vec[2]) > 0.7
        # Lidar is mounted 1m above --> ground must be lower
        is_not_ceiling = dist_to_ground < -lidar_height

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
        rot = euler_matrix(roll, pitch, 0, 'sxyz')[:3, :3]
        return rot

    @staticmethod
    def array_to_pcl(pc_array):
        """Get pcl point cloud from numpy array"""
        pc = pcl.PointCloud()
        pc.from_array(pc_array.astype('float32'))
        return pc

    @staticmethod
    def quat_from_vectors(vec1, vec2):
        """Quaternion that aligns vector 1 to vector 2"""
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
    def voxel_filter(pc, leaf_size):
        """Downsample point cloud using voxel filter"""
        vgf = pc.make_voxel_grid_filter()
        # Leaf_size is the length of the side of the voxel cube in m
        vgf.set_leaf_size(leaf_size, leaf_size, leaf_size)
        pc_filtered = vgf.filter()
        return pc_filtered

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

    def publish_pc(self, pc, pub_name):
        """Publishes a point cloud from point list"""
        # Make sure that pc is a list
        if isinstance(pc, list):
            pass
        # Convert to list if numpy array
        elif isinstance(pc, np.ndarray):
            pc = list(pc)
        # Convert to list if pcl point cloud
        else:
            pc = pc.to_list()

        # Initialize pc2 msg
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = '/velodyne'
        # Convert point cloud list to pc2 msg
        pc = pc2.create_cloud_xyz32(header, pc)
        # Publish message
        rospy.Publisher(pub_name, PointCloud2, queue_size=10).publish(pc)



def filelist_from_dir(dir, extension):
    """Returns a list of files in the directory"""
    # Go to the directory
    os.chdir(dir)
    # Get list of files which have specified extension
    lst = glob.glob('*.{}'.format(extension))
    return lst

#! Change path to directory which contains the rosbags
path = "/home/user/rosbags/final"
# Get file list
bag_lst = filelist_from_dir(path, 'bag')

#! Change lidar topic
lidar_topic = '/velodyne_points'

#! Publish pointcloud of detected ground?
show_ground = True

# Define list to collect values
props = []

rospy.init_node('check_ground')
while not rospy.is_shutdown():
    for b in bag_lst:
        bag = rosbag.Bag(os.path.join(path, b))
        # Get first message and calculate stuff
        for topic,msg,t in bag.read_messages(topics=lidar_topic):
            roll, pitch, height = AlignGround(msg, show_ground).align_lidar()
            print('Roll: {:.2f}\nPitch: {:.2f}\nHeight: {:.2f}'.format(
                np.rad2deg(roll), np.rad2deg(pitch), height
            ))
            print('Bag: {}\n'.format(b))
            # Add values to list to get average later
            props.append((roll, pitch, height))
            # Stop after first message
            break
    break

# Calculate average of all bags for a more precise output
means = np.mean(np.asarray(props), axis=0)
print('Average angles / height of all rosbags')
print('Roll: {:.2f}\nPitch: {:.2f}\nHeight: {:.2f}'.format(
    np.rad2deg(means[0]), np.rad2deg(means[1]), means[2]
))
