#!/usr/bin/env python

# Limit CPU usage (of numpy)
# ! must be called before importing numpy
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from tf.transformations import euler_from_quaternion, euler_matrix, unit_vector, vector_norm, quaternion_from_matrix
from sensor_msgs.msg import PointCloud2
import ros_numpy
import rospy
import pcl
import numpy as np

"""Calculates the pitch and roll angle of lidar (online)"""
class VisualDetection():

    def __init__(self):
        rospy.init_node('visual_detection', anonymous=True)
        # * Change depending on rosbag
        lidar_topic = ['velodyne_points', '/right/rslidar_points']
        # Set this to False if velodyne is used
        self.is_robosense = True
        if self.is_robosense:
            lidar_topic = lidar_topic[1]
        else:
            lidar_topic = lidar_topic[0]
        self.sub_lidar = rospy.Subscriber(lidar_topic, PointCloud2, self.callback_lidar, queue_size=10)
        self.flag = False
        
    def callback_lidar(self, msg):
        self.cloud = msg
        self.flag = True

    def spin(self):
        # Robosense Lidar has a rate of 10 Hz
        rate = 10
        r = rospy.Rate(rate)
        while not rospy.is_shutdown():
            if not self.flag:
                continue
            # Convert PointCloud2 msg to numpy array
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.cloud, remove_nans=True)

            # Get transformation from lidar to car frame
            self.align_lidar(pc_array)
            r.sleep()

    def align_lidar(self, pc_array):
        """Calculate roll and pitch angle to align Lidar with car frame"""
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

        print('Pitch angle: {:05.2f} deg (roll angle: {:05.2f} deg)'.format(
            np.degrees(pitch), np.degrees(roll)))
        return [roll, pitch]

    def ground_detection(self, pc):
        """Detect ground plane and get normal vector"""
        # Get most dominant plane (assume that this is the ground plane)
        indices, coefficients = self.ransac(pc)
        ground_vec = coefficients[:-1]
        return ground_vec

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


if __name__ == "__main__":
    try:
        vd = VisualDetection()
        vd.spin()
    except rospy.ROSInterruptException:
        pass