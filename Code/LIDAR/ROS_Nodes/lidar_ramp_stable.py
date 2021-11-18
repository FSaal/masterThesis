#!/usr/bin/env python

# Limit CPU usage (of numpy)
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from tf.transformations import euler_from_quaternion, euler_matrix, unit_vector, vector_norm, quaternion_from_matrix
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Float32, Float32MultiArray
import ros_numpy
import rospy
import pcl
import numpy as np


class VisualDetection():

    def __init__(self):
        rospy.init_node('visual_detection', anonymous=True)
        # self.sub_lidar = rospy.Subscriber('/rslidar_points', PointCloud2, self.callback_lidar, queue_size=10)
        self.sub_lidar = rospy.Subscriber('/right/rslidar_points', PointCloud2, self.callback_lidar, queue_size=10)
        self.pub_angle = rospy.Publisher(
            '/ramp_angle_lidar', Float32, queue_size=10)
        self.flag = False
        self.calibrated = False
        self.rp = [0, 0]
        self.perf1 = PerformanceMeasure()
        self.perf2 = PerformanceMeasure()

    def callback_lidar(self, msg):
        self.cloud = msg
        self.flag = True

    def spin(self):
        # Robosense Lidar has a rate of 10 Hz
        rate = 10
        r = rospy.Rate(rate)
        while not rospy.is_shutdown():
            if self.flag == True:
                start_init = rospy.get_time()
                # Convert PointCloud2 msg to numpy array
                pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(
                    self.cloud, remove_nans=True)

                # Get transformation from lidar to car frame
                if not self.calibrated:
                    # Get euler angles to align lidar with car frame
                    self.rp = self.align_lidar(pc_array)
                    self.calibrated = True

                # Apply base-link-lidar transformation
                pc_array_tf = self.transform_pc(
                    pc_array, roll=self.rp[0], pitch=self.rp[1])
                # pc_array_tf = self.transform_pc(pc_array, roll=self.rp[0], pitch=self.rp[1], yaw=-0.14)

                # Filter unwanted points (to reduce point cloud size) with passthrough filter
                side_limit = 2.5
                pc_array_cut = self.reduce_pc(
                    pc_array_tf, 2, 42, -side_limit, side_limit, -1, 1.5)
                # Convert numpy array to pcl object
                pc_cut = pcl.PointCloud()
                pc_cut.from_array(pc_array_cut.astype('float32'))

                # Downsample point cloud using voxel filter to further decrease size
                pc_small = self.voxel_filter(pc_cut, 0.1)

                # Perform RANSAC until no new planes are being detected
                # start = rospy.get_time()
                self.detect_all_planes(pc_small, 100, 4, 3)
                # self.perf1.performance_calc(start)
                self.perf2.performance_calc(start_init, 'total\n')
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
        # Euler angles
        roll, pitch, yaw = euler_from_quaternion(quat)
        print('Euler angles in deg to tf lidar to car frame:')
        print('Roll: {:05.2f}\nPitch: {:05.2f}\nYaw: {:05.2f}'.format(
            np.degrees(roll), np.degrees(pitch), np.degrees(yaw)))
        return [roll, pitch]

    def ground_detection(self, pc):
        """Detect ground plane and get normal vector"""
        # Assume most dominant plane is the ground plane
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

        quat = np.append(ax*np.sin(a/2), np.cos(a/2))
        return quat

    def transform_pc(self, pc, roll=0, pitch=0, yaw=0, transl_x=1.753, transl_y=0, transl_z=1.156):
        """Transformation from Lidar frame to car frame. Rotation in rad and translation in m."""
        # Rotation
        rot = euler_matrix(roll, pitch, yaw, 'sxyz')[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc, rot)
        # Translation
        translation = [transl_x, transl_y, transl_z]
        pc_tf += translation
        return pc_tf

    def reduce_pc(self, pc, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
        pc_cut = pc[(pc[:, 0] > x_lower) & (pc[:, 0] < x_upper) & (pc[:, 1] > y_lower) & (
            pc[:, 1] < y_upper) & (pc[:, 2] > z_lower) & (pc[:, 2] < z_upper)]
        return pc_cut

    def voxel_filter(self, pc, leaf_size):
        """Downsample point cloud using voxel filter"""
        vg = pc.make_voxel_grid_filter()
        # Leaf_size is the length of the side of the voxel cube in m
        vg.set_leaf_size(leaf_size, leaf_size, leaf_size)
        pc_filtered = vg.filter()
        # print('Reduced size from {} to {}'.format(pc.size, pc_filtered.size))
        return pc_filtered

    def detect_all_planes(self, pc, cutoff, max_planes, ramp_ang):
        """Detects all planes in point cloud until threshold of amount of left points is reached

        :param pc:          Point cloud
        :param cutoff:      Iteration stops if cutoff amount of points are left in reduced pc
        :max_planes:        Max number of planes to detect before exiting
        :ramp_ang:          Minimum pitch angle [deg] of plane to be considered a ramp
        """
        ground_vector = None
        counter = 0

        while pc.size > cutoff and counter < max_planes:
            # Detect most dominate plane
            indices, coefficients = self.ransac(pc)
            # Normal vector of plane
            normal_vector = coefficients[:-1]

            # Split pointcloud in inliers and outliers of plane
            pc, plane = self.split_pc(pc, indices)

            # Exit if plane is empty
            if not plane:
                print('ERROR: No plane could be detected')
                return

            # Identify what type of plane was detected (e.g. xy, yz...)
            plane_type = self.nv_simplifier(normal_vector, 0.7)

            if plane_type == 2:
                if ground_vector is None:
                    ground_vector = normal_vector
                    self.pub_angle.publish(0)
                # Either ground is detected again or potential ramp
                else:
                    # Calculate angle [deg] between new and previously recorded normal vector of ground
                    angle = self.angle_calc(ground_vector, normal_vector)
                    self.pub_angle.publish(angle)
                    # Angle threshold [deg] to be considered a ramp
                    if angle > ramp_ang:
                        # Calculate distance to ramp based on nearest (lowest x value) point of ramp plane
                        plane_array = np.array(plane)
                        ramp_dist = min(plane_array[:, 0])
                        plane_height = max(plane_array[:, 2]) - min(plane_array[:, 2])

                        min_ramp = 0.9
                        max_ramp = 1.5
                        if min_ramp < plane_height < max_ramp:
                            print('Possible ramp in {:05.2f}m with angle {:05.2f} deg'.format(
                                ramp_dist, angle))
                        return
            counter += 1

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

    def split_pc(self, pc, inliers):
        """Extract detected plane from point cloud and split into two pcs

        :param pc:              Point cloud
        :param inliers:         Indices of inliers of plane
        :return pc_outliers:    PCL point cloud object w/o plane inliers
        :return detected_plane: List of inlier points
        """
        detected_plane = [pc[i] for i in inliers]
        # Point cloud of detected plane (inliers)
        # pc_inliers = pc.extract(inliers)
        # Point cloud of outliers
        outlier_indices = list(
            set(np.arange(pc.size)).symmetric_difference(inliers))
        pc_outliers = pc.extract(outlier_indices)

        return pc_outliers, detected_plane

    def nv_simplifier(self, v, threshold):
        """Identifies plane type by evaluating normal vector of plane"""
        # yz-plane (wall ahead)
        if abs(v[0]) > threshold:
            return 0
        # xz-plane (side wall)
        elif abs(v[1]) > threshold:
            return 1
        # xy-plane (ground)
        elif abs(v[2]) > threshold:
            return 2
        else:
            return 3

    def angle_calc(self, v1, v2, degrees=True):
        """Calculate angle between two vectors (planes)"""
        # Assuming both vectors can be rotated alongside one axis to be aligned
        dot = np.dot(v1, v2)
        if dot <= 1:
            angle = np.arccos(dot)
        else:
            print('ERROR: dot product > 1')
            angle = 0

        if degrees is True:
            return np.degrees(angle)
        else:
            return angle

    def publish_pc(self, pc_list, publisher):
        """Publishes a point cloud from point list"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = '/rslidar'
        pc = pc2.create_cloud_xyz32(header, pc_list)
        publisher.publish(pc)


class PerformanceMeasure():
    def __init__(self):
        self.total_time = 0
        self.counter = 1

    def performance_calc(self, start_time, name=""):
        end = rospy.get_time()
        duration = end - start_time
        self.total_time += duration
        avg_time = self.total_time/self.counter
        if not name:
            print('Took {:.5f}s and on average {:.5}s which is {:5.4}Hz'.format(
                duration, avg_time, 1/avg_time))
        else:
            print('Took {:.5f}s and on average {:.5}s which is {:5.4}Hz - {}'.format(
                duration, avg_time, 1/avg_time, name))
        self.counter += 1


if __name__ == "__main__":
    try:
        vd = VisualDetection()
        vd.spin()
    except rospy.ROSInterruptException:
        pass


"""
Information about the rosbag hdl_odom_localize_only.bag:
50s - 60s driving around corner with straight wall
75s - 85s driving around other corner with straight wall (to the right is the ramp, but not captured by lidar)
115s - a ramp is being captured!
165s - 175s driving to straight wall
"""
