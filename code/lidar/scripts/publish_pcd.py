#!/usr/bin/env python
""" Publishes a pcd file as pc2 msg
 Also publishes a trimmed pointcloud of the groundpoints and ramp region"""

from __future__ import print_function
import os
import sys
import pcl
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from tf.transformations import euler_matrix


class PcdVis():
    def __init__(self):
        rospy.init_node('pcd_vis')
        # Point cloud publisher
        self.pcd_pub = rospy.Publisher('/lidar_slam_pcl', PointCloud2, queue_size=10)
        self.pcd_pub2 = rospy.Publisher('/groundish', PointCloud2, queue_size=10)
        self.header = Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = '/map_lidar'
        self.pub_ramp_flag = False
        self.pub_ramps_flag = False

        # Frequency of Node [Hz]
        self.rate = 1

        # Check if exactly one command line argument was given
        if len(sys.argv) == 2:
            # pcd filepath
            self.pcd_filepath = sys.argv[1]
        else:
            print('A command line argument with path to pcd file is necessary')
            sys.exit()

    def does_file_exist(self):
        if os.path.isfile(self.pcd_filepath):
            extension = self.pcd_filepath.split('.')[-1]
            if extension != 'pcd':
                print('File must be a .pcd file not a .{}'.format(extension))
                return False
            return True
        else:
            print('{} does not exist'.format(self.pcd_filepath))
            return False

    def spin(self):
        # Quit if file does not exist or wrong extension
        if not self.does_file_exist:
            sys.exit()

        # Load pcd file
        pc_map = pcl.load(self.pcd_filepath)
        # Downsample pc
        pc_smol = self.voxel_filter(pc_map, 0.1)
        # Convert pcl object to list
        pc_map_lst = pc_smol.to_list()
        # Remove non ground points
        pc_ground = self.ground_only(pc_map_lst)
        # Get PointCloud2 msg from list
        pc = pc2.create_cloud_xyz32(self.header, pc_map_lst)
        print('Publishing pcd pointcloud on topic: "/lidar_slam_pcl"')
        # Frequency of Node [Hz]
        r = rospy.Rate(self.rate)

        # Static publisher of the point cloud map
        while not rospy.is_shutdown():
            # Publish map
            self.pcd_pub.publish(pc)
            # Publish ground seperately
            self.pcd_pub2.publish(pc_ground)
            r.sleep()

    def ground_only(self, pc_msg):
        pc_array = np.array(pc_msg)
        # Remove points with z > 2 --> leaves mostly the ground points
        pc_cut = pc_array[pc_array[:, 2] < 0.5]
        # pc_array_rot = self.transform_pc(pc_array, yaw=np.deg2rad(0))

        # Convert numpy array to pointcloud msg
        pc_ground = pc2.create_cloud_xyz32(self.header, list(pc_cut))
        return pc_ground

    def voxel_filter(self, pc, leaf_size):
        """Downsample point cloud using voxel filter"""
        vgf = pc.make_voxel_grid_filter()
        # Leaf_size is the length of the side of the voxel cube in m
        vgf.set_leaf_size(leaf_size, leaf_size, leaf_size)
        pc_filtered = vgf.filter()
        return pc_filtered

    def transform_pc(self, pc, roll=0, pitch=0, yaw=0):
        """Transformation from Lidar frame to car frame. Rotation in rad and translation in m."""
        # Rotation
        rot = euler_matrix(roll, pitch, yaw, 'sxyz')[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc, rot)
        return pc_tf


if __name__ == '__main__':
    PV = PcdVis()
    PV.spin()
