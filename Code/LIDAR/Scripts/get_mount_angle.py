#!/usr/bin/env python
"""Calculates the estimated average pitch angle of rosbags (offline)"""

import rosbag
from tf.transformations import euler_from_quaternion, unit_vector, vector_norm
import ros_numpy
import pcl
import numpy as np
import glob
import os

class pitchCalc():
    def align_lidar(self, pc_msg):
        """Calculate roll and pitch angle to align Lidar with car frame"""
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pc_msg, remove_nans=True)

        # Convert numpy array to pcl point cloud
        pc = pcl.PointCloud()
        pc.from_array(pc_array.astype('float32'))

        # Get normal vector of ground plane
        ground_vec_lidar = self.ground_detection(pc)
        # Normal vector ground plane in car frame
        ground_vec_car = [0, 0, 1]

        # Quaternion to align lidar vec to car vec
        quat = self.quat_from_vectors(ground_vec_lidar, ground_vec_car)
        # Calculate euler angles
        roll, pitch, yaw = euler_from_quaternion(quat)

        # print('Euler angles in deg to tf lidar to car frame:')
        # print('Roll: {:05.2f}\nPitch: {:05.2f}\nYaw: {:05.2f}'.format(
            # np.degrees(roll), np.degrees(pitch), np.degrees(yaw)))
        return pitch

    def ground_detection(self, pc):
        """Detect ground plane and get normal vector"""
        # Get most dominant plane (assume that this is the ground plane)
        indices, coefficients = self.ransac(pc)
        ground_vec = coefficients[:-1]
        return ground_vec

    def ransac(self, pc):
        """Finds inliers and normal vector of dominant plane"""
        # 50?
        seg = pc.make_segmenter_normals(50)
        # Doubles the speed if True
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        # How close a point must be to model to be considered inlier
        seg.set_distance_threshold(0.01)
        # normal_distance_weight?
        seg.set_normal_distance_weight(0.01)
        # How many tries
        seg.set_max_iterations(100)
        indices, coefficients = seg.segment()
        return indices, coefficients

    def quat_from_vectors(self, vec1, vec2):
        """Quaternion that aligns vec1 to vec2"""
        a, b = unit_vector(vec1), unit_vector(vec2)
        c = np.cross(a, b)
        d = np.dot(a, b)

        # Rotation axis
        ax = c / vector_norm(c)
        # Rotation angle
        a = np.arctan2(vector_norm(c), d)

        return np.append(ax*np.sin(a/2), np.cos(a/2))

# Get list of bag files
path = "/home/user/rosbags/big/evaluation"
os.chdir(path)
bag_list = glob.glob('*.bag')
# Angle list
ang_lst = []

for b in bag_list:
    # Load bag
    bag = rosbag.Bag(os.path.join(path, b))
    # Get first message
    for topic,msg,t in bag.read_messages(topics='/right/rslidar_points'):
        # print(msg)
        inst = pitchCalc()
        pitch_angle = inst.align_lidar(msg)
        print("The estimated pitch angle for bag {} is {}".format(b, np.rad2deg(pitch_angle)))
        ang_lst.append(pitch_angle)
        break

    avg_angle = np.mean(ang_lst)
print('Estimated angle: {} deg ({} rad)'.format(np.rad2deg(avg_angle), avg_angle))