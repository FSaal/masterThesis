#!/usr/bin/env python
""" Publishes the map from the hdl_slam of the rosbag
 Also publishes a trimmed pointcloud of the groundpoints and ramp region"""

from __future__ import print_function
import os
import glob
import pcl
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from tf.transformations import euler_matrix


class EvalMap():
    def __init__(self):
        rospy.init_node('pcd_vis')
        # Point cloud publisher
        self.pcd_pub = rospy.Publisher('/world_map', PointCloud2, queue_size=10)
        self.pcd_pub2 = rospy.Publisher('/groundish', PointCloud2, queue_size=10)
        self.ramp_pub = rospy.Publisher('/ramp_region', PointCloud2, queue_size=10)
        self.header = Header()
        self.header.stamp = rospy.Time.now()
        self.header.frame_id = '/map_lidar'
        self.pub_ramp_flag = False
        self.pub_ramps_flag = False

    def spin(self):
        # Ask user to select bag
        pc_map, region = self.select_bag()
        # Make sure region was specified
        if region:
            print('Ramp region is published as separate point cloud')
            self.pub_ramp_flag = True
        else:
            print('Ramp region does not seem to be specified yet')

        # Static publisher of the point cloud map
        while not rospy.is_shutdown():
            # Downsample pc
            pc_smol = self.voxel_filter(pc_map, 0.1)
            # Convert pcl object to list
            pc_map_lst = pc_smol.to_list()
            # Get PointCloud2 msg from list
            pc = pc2.create_cloud_xyz32(self.header, pc_map_lst)
            # Publish map
            self.pcd_pub.publish(pc)

            # Publish ground seperately
            pc_ground = self.ground_only(pc_map_lst)
            self.pcd_pub2.publish(pc_ground)

            if self.pub_ramp_flag:
                # Publish ramp region seperately
                pc_ramp = self.get_ramp_region(pc_map_lst, region)
                self.ramp_pub.publish(pc_ramp)

    def ground_only(self, pc_msg):
        pc_array = np.array(pc_msg)
        # Remove points with z > 2 --> leaves mostly the ground points
        pc_cut = pc_array[pc_array[:, 2] < 0.5]
        # pc_array_rot = self.transform_pc(pc_array, yaw=np.deg2rad(0))

        # Convert numpy array to pointcloud msg
        pc_ground = pc2.create_cloud_xyz32(self.header, list(pc_cut))
        return pc_ground

    def get_ramp_region(self, pc_msg, ramps):
        """Cut all points which are not ramps and publish only the ramp points"""
        # Convert from list to numpy array
        pc_array = np.array(pc_msg)
        # First initialize, that every index/point is not a ramp
        ramps_indices = np.zeros(len(pc_array), dtype=bool)

        # Check if more than one ramp has been specified
        if np.asarray(ramps).ndim > 2:
            # At least two ramps
            for ramp in ramps:
                x_range, y_range = ramp
                ramp_indices = self.cut_pc(pc_array, x_range, y_range)
                ramps_indices += ramp_indices
        # Only one ramp
        else:
            x_range, y_range = ramps
            ramp_indices = self.cut_pc(pc_array, x_range, y_range)
            ramps_indices += ramp_indices

        # Trim down to ramp region
        pc_cut = pc_array[ramps_indices]

        # Convert numpy array to pointcloud msg
        pc_ramp = pc2.create_cloud_xyz32(self.header, list(pc_cut))
        return pc_ramp

    def cut_pc(self, pc_array, x_range, y_range):
        """Pointcloud row indices of points, which are on the ramp"""
        ramp_idx = (
            (pc_array[:, 2] < 2) &
            (pc_array[:, 0] > x_range[0]) &
            (pc_array[:, 0] < x_range[1]) &
            (pc_array[:, 1] > y_range[0]) &
            (pc_array[:, 1] < y_range[1])
        )
        return ramp_idx

    def voxel_filter(self, pc, leaf_size):
        """Downsample point cloud using voxel filter"""
        vgf = pc.make_voxel_grid_filter()
        # Leaf_size is the length of the side of the voxel cube in m
        vgf.set_leaf_size(leaf_size, leaf_size, leaf_size)
        pc_filtered = vgf.filter()
        return pc_filtered

    def select_bag(self):
        # Get list of bag files
        path = "/home/user/rosbags/final/slam"
        os.chdir(path)
        map_lst = glob.glob('*.pcd')
        # Sort list alphabetically
        map_lst.sort()

        # Map list

        # Ask user what map to load
        print("The map of which bag do you want to see?")
        for i, v in enumerate(map_lst):
            print('Enter {} for {}'.format(i, v))
        while True:
            idx = raw_input("Enter now: ")
            # Make sure number is right
            if idx.isdigit() and -1 < int(idx) < len(map_lst):
                i = int(idx)
                break
            print("Please enter a valid number")

        map_filename = os.path.join(path, map_lst[i])
        print(map_filename)
        pc_map = pcl.load(map_filename)

        region = [
        [],
        # d_d2r2s_odom
        [
            [[25, 36], [-2.6, 1.4]],
            [[26, 36], [37, 41]],
            ],
        # d_e2q (width and length are swapped)
        [[24.5, 28.5], [3.3, 15]],
        [],
        # u_c2s_half
        [[20.3, 33], [-0.9, 2.8]],
        [],
        [],
        [],
        # u_d2e
        [[32.5, 44], [2, 5.5]],
        [],
        # u_s2c_half (hard because ramp is not straight)
        [[42, 56], [-2.2, 2]],
        ]

        return pc_map, region[i]

    def transform_pc(self, pc, roll=0, pitch=0, yaw=0):
        """Transformation from Lidar frame to car frame. Rotation in rad and translation in m."""
        # Rotation
        rot = euler_matrix(roll, pitch, yaw, 'sxyz')[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc, rot)
        return pc_tf


if __name__ == '__main__':
    EM = EvalMap()
    EM.spin()
