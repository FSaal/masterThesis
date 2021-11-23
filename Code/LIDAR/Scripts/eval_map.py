#!/usr/bin/env python

import pcl 
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import os
import glob
import numpy as np
from tf.transformations import euler_matrix

""" Publishes the map from the hdl_slam of the rosbag
 Also publishes a trimmed pointcloud of the groundpoints and ramp region"""
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

    def spin(self):
        # Ask user to select bag
        pc_map, region = self.select_bag()
        # Make sure region was specified
        if region:
            x_range, y_range = region
            self.pub_ramp_flag = True


        # Static publisher of the point cloud map
        while not rospy.is_shutdown():
            # Convert pcl object to list
            pc_map_lst = pc_map.to_list()
            # Get PointCloud2 msg from list
            pc = pc2.create_cloud_xyz32(self.header, pc_map_lst)
            # Publish map
            self.pcd_pub.publish(pc)

            # Publish ground seperately
            pc_ground = self.ground_only(pc_map_lst)
            self.pcd_pub2.publish(pc_ground)

            if self.pub_ramp_flag:
                # Publish ramp region seperately
                pc_ramp = self.ramp_region(pc_map_lst, x_range, y_range)
                self.ramp_pub.publish(pc_ramp)

    def ground_only(self, pc_msg):
        pc_array = np.array(pc_msg)
        # Remove points with z > 0.5 --> leaves mostly the ground points
        pc_cut = pc_array[pc_array[:,2] < 0.5]
        # pc_array_rot = self.transform_pc(pc_array, yaw=np.deg2rad(0))

        # Convert numpy array to pointcloud msg 
        pc_ground = pc2.create_cloud_xyz32(self.header, list(pc_cut))
        return pc_ground

    def ramp_region(self, pc_msg, x_range, y_range):
        pc_array = np.array(pc_msg)
        # Trim down to ramp region
        pc_cut = pc_array[
            (pc_array[:,2] < 0.5) & 
            (pc_array[:,0] > x_range[0]) & 
            (pc_array[:,0] < x_range[1]) &
            (pc_array[:,1] > y_range[0]) & 
            (pc_array[:,1] < y_range[1])
            ] 

        # Convert numpy array to pointcloud msg 
        pc_ramp = pc2.create_cloud_xyz32(self.header, list(pc_cut))
        return pc_ramp

    def select_bag(self):
        # Get list of bag files
        path = "/home/user/rosbags/big/evaluation"
        os.chdir(path)
        bag_list = glob.glob('*.bag')
        # Remove hdl bags (by check if something has been added to number before .bag)
        bag_list = [b for b in bag_list if b[-6:-4].isdigit()]
        # Sort list alphabetically
        bag_list.sort()

        # Map list
        map_list = []
        for b in bag_list:
            bag_name, ext = b.split(".")
            map_name = bag_name + "_map.pcd"
            map_list.append(map_name)

        # Ask user what map to load
        print("The map of which bag do you want to see?")
        for i,v in enumerate(map_list):
            print('Enter {} for {}'.format(i, v))
        while True:
            idx = raw_input("Enter now: ")
            # Make sure number is right
            if idx.isdigit() and -1 < int(idx) < len(map_list):
                i = int(idx)
                break
            print("Please enter a valid number")

        map_filename = os.path.join(path, map_list[i])
        print(map_filename)
        pc_map = pcl.load(map_filename)

        region = [
            [[24, 36], [-2.6, 1.5]], 
            [],
            [],
            [[20, 32], [-1.4, 2.5]],
            [], 
            [],
            [[14.5, 24], [1.5, 5.8]]
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
    em = EvalMap()
    em.spin()