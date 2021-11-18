#!/usr/bin/env python

# Limit CPU usage (of numpy)
# ! must be called before importing numpy
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
import math
import scipy.stats as st

class VisualDetection():

    def __init__(self):
        rospy.init_node('visual_detection', anonymous=True)
        # * Change depending on rosbag
        lidar_topic = ['/rslidar_points', '/right/rslidar_points']
        self.sub_lidar = rospy.Subscriber(lidar_topic[1], PointCloud2, self.callback_lidar, queue_size=10)
        self.pub_angle = rospy.Publisher('/ramp_angle_lidar', Float32, queue_size=10)
        self.flag = False
        self.calibrated = False
        self.rp = [0, 0]
        self.perf1 = PerformanceMeasure()
        self.perf2 = PerformanceMeasure()
        self.flag2 = False
        self.detect_counter = []
        self.detect_counter2 = []
        
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
            # start_init = rospy.get_time()
            # Convert PointCloud2 msg to numpy array
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.cloud, remove_nans=True)

            # Get transformation from lidar to car frame
            if not self.calibrated:
                # Get euler angles to align lidar with car frame
                self.rp = self.align_lidar(pc_array)
                self.calibrated = True

            # Apply lidar to car frame transformation
            # * Change depending on rosbag
            yaw = [0, -0.14, 0.07]
            pc_array_tf = self.transform_pc(pc_array, roll=self.rp[0], pitch=self.rp[1], yaw=yaw[2])

            # Filter unwanted points (to reduce point cloud size) with passthrough filter
            # TODO: Check if Points further than 30m have intensity less than 10%
            # * Max Range of lidar is 100m (30m @ 10% NIST)
            pc_array_cut = self.reduce_pc(pc_array_tf, 0, 30, -3.5, 3.5, -1, 1.5)
                
            # Convert numpy array to pcl object
            pc_cut = pcl.PointCloud()
            pc_cut.from_array(pc_array_cut.astype('float32'))

            # Downsample point cloud using voxel filter to further decrease size
            pc_small = self.voxel_filter(pc_cut, 0.1)
            # self.publish_pc(pc_small.to_list(), 'pc')

            # Perform RANSAC until no new planes are being detected
            ang, dist = self.plane_detection(pc_small, 100, 4)
            self.ramp_detection_confidence(ang, 10, 5, dist)
            # self.perf1.performance_calc(start_init, 'total\n')
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

        print('Euler angles in deg to tf lidar to car frame:')
        print('Roll: {:05.2f}\nPitch: {:05.2f}\nYaw: {:05.2f}'.format(
            np.degrees(roll), np.degrees(pitch), np.degrees(yaw)))
        return [roll, pitch]

    # TODO: Yaw Calibration
    def yaw_correction(self, pc_array):
        pc_array2 = pc_array[(pc_array[:,2] > 0.2) & (pc_array[:,2] < 0.7)]
        pc_array = pc_array2
        print('Are you ready to go?')
        if not self.flag2:
            self.first_pc = pc_array
            # Wait for n seconds to get second pcl
            self.flag2 = True
        else:
            second_pc = pc_array
            pcfir = pcl.PointCloud()
            pcfir.from_array(self.first_pc.astype('float32'))
            pcsec = pcl.PointCloud()
            pcsec.from_array(second_pc.astype('float32'))
            icp = pcfir.make_IterativeClosestPoint()
            converged, transf, estimate, fitness = icp.icp(pcfir, pcsec)
            print(transf[:-1,3])

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

    def transform_pc(self, pc, roll=0, pitch=0, yaw=0, transl_x=1.753, transl_y=0, transl_z=1.156):
        """Transformation from Lidar frame to car frame. Rotation in rad and translation in m."""
        # Rotation matrix
        rot = euler_matrix(roll, pitch, yaw, 'sxyz')[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc, rot)
        # Translation
        translation = [transl_x, transl_y, transl_z]
        # Combine rotation and translation
        pc_tf += translation
        return pc_tf

    def reduce_pc(self, pc, x_lower, x_upper, y_lower, y_upper, z_lower, z_upper):
        """Removes points outside of box"""
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

    def plane_detection(self, pc, min_points, max_planes):
        """Detects all planes in point cloud

        Iteratively detects most dominant plane and corresponding normal vector of the
        point cloud until either max_planes amount of iterations have been performed or
        if not enough points (min_points) are left. After each detection the plane gets
        removed from the point cloud.

        :param pc:          Point cloud (PCL)
        :param min_points:  At least min_points must be left to continue detection
        :max_planes:        Max number of planes to detect before exiting (to limit computation time)
        """
        # Ground vector
        g_vec = None
        counter = 0
        while pc.size > min_points and counter < max_planes:
            # Detect most dominate plane and get inliers and normal vector
            indices, coefficients = self.ransac(pc)
            n_vec = coefficients[:-1]

            # Split pointcloud in inliers and outliers of plane
            pc, plane = self.split_pc(pc, indices)

            # Exit if plane is empty
            if not plane:
                print('ERROR: No plane could be detected')
                return 0,0

            # Ignore walls to the side or in front
            if self.is_plane_near_ground(n_vec):
                # First ground like detection is most probably the ground
                if g_vec is None:
                    g_vec = n_vec
                # Either ground is detected again or potential ramp
                else:
                    # self.pub_angle.publish(angle)
                    if self.ramp_detection(
                        plane, g_vec, n_vec, 3, 10, 0.5, 1.5, 2, 6, 1, 10):
                        return self.angle, self.d
                    else:  
                        # self.angle = 0
                        # self.pub_angle.publish(0)
                        continue
                    # return
            else:
                print(n_vec)
            counter += 1
        return 0, 0

    def get_some_stats(self, data):
        mean = np.mean(data)
        std_dev = np.std(data)
        var = np.var(data)
        median = np.median(data)

    def ramp_detection_confidence(self, angle, win, min_count, dist):
        # Populate list with calculated angles
        self.detect_counter.append(angle)
        self.detect_counter2.append(dist)

        # Possible ramp angles (not 0 deg)
        angles = [i for i in self.detect_counter if i != 0]
        dists = [i for i in self.detect_counter2 if i != 0]
        if len(angles) > min_count:
            s_mean = np.mean(dists)
            x_mean = np.mean(angles)
            std_dev = np.std(angles)
            n = len(angles)
            a,b = st.t.interval(alpha = 0.95, df=len(angles)-1, loc=np.mean(angles), scale=st.sem(angles))
            # print('With a probability of alpha, the angle lies in the range of {}, {}'.format(a,b))
            # print('Angle is probably {} or {}'.format(np.mean(angles), np.median(angles)))

        # Confidence
        c,d = st.t.interval(alpha = 0.95, df=len(angles)-1, loc=np.mean(angles), scale=st.sem(angles))
        conf = (len(self.detect_counter) - len(angles)) / float(len(self.detect_counter))
        if len(angles) > min_count:
            conf = 1 - (len(self.detect_counter) - len(angles)) / float(len(self.detect_counter))
        else:
            x_mean = 0
            s_mean = 0
        print('With a conf of {}% the angle is {} and in {}m'.format(conf*100, x_mean, s_mean))
            
        # Remove oldest angle from list
        if len(self.detect_counter) > win:
            self.detect_counter.pop(0)

    def ramp_detection(
            self, plane, g_vec, n_vec, min_angle, max_angle, 
            min_height, max_height, min_width, max_width, 
            min_length, max_length, logging=False):
        """Checks if conditions to be considered a ramp are fullfilled.
        
        The following values of the plane are being calculated and 
        checked whether they lie within the desired range:
        - angle (calculated between normal vec of ground_plane and this plane)
        - height (relative height difference between furthest and nearest point)
        - width
        - length
        If all conditions are met return True, else False
        """
        # Convert pcl plane to numpy array
        plane_array = np.array(plane)

        # Calculate angle [deg] between new and previously recorded normal vector of ground
        angle = self.angle_calc(g_vec, n_vec)
        # Assert ramp angle threshold
        if min_angle <= angle <= max_angle:
            if logging: print('ANGLE PASSED')
            pass
        else:
            if logging: print('Angle wrong with {}'.format(angle))
            return False
        
        # Get ramp height (Difference between z-values of furthest and nearest point)
        height = max(plane_array[:, 2]) - min(plane_array[:, 2])
        # Assert ramp height threshold
        if min_height <= height <= max_height:
            if logging: print('HEIGHT PASSED')
            pass
        else:
            if logging: print('Height wrong with {}'.format(height))
            return False

        # Get ramp width (Difference between y-values)
        width = max(plane_array[:, 1]) - min(plane_array[:, 1])
        # Assert ramp width threshold
        if min_width <= width <= max_width:
            if logging: print('WIDTH PASSED')
            pass
        else:
            if logging: print('Width wrong with {}'.format(width))
            return False

        # Length in x direction 
        x = max(plane_array[:, 0]) - min(plane_array[:, 0])
        # Get ramp length (using pythagorean theorem)
        length = math.sqrt(x**2 + height**2)
        # Assert ramp length threshold
        if min_length <= length <= max_length:
            if logging: print('Length passed')
            pass
        else:
            if logging: print('Length wrong with {}'.format(length))
            return False

        # Ramp distance (x-value of nearest point of the plane)
        dist = min(plane_array[:,0])

        self.angle = angle
        self.d = dist
        if logging:
            print('Possible ramp in {:05.2f}m with angle {:05.2f}deg, height {:05.2f}m, width {:05.2f}m and length {:05.2f}m'.format(
            dist, angle, height, width, length))        
        return True

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

        :param pc:              Point cloud (PCL)
        :param inliers:         Indices of inliers of plane
        :return pc_outliers:    PCL point cloud object w/o plane inliers
        :return detected_plane: List of inlier points
        """
        # Get point cooridnates of plane
        detected_plane = [pc[i] for i in inliers]
        # Point cloud of detected plane (inliers)
        # pc_inliers = pc.extract(inliers)

        # Point cloud of outliers
        outlier_indices = list(set(np.arange(pc.size)).symmetric_difference(inliers))
        pc_outliers = pc.extract(outlier_indices)

        return pc_outliers, detected_plane

    def is_plane_near_ground(self, v, threshold=0.8):
        """Returns True if plane is on the ground (and false if e.g. side wall)"""
        return abs(v[2]) > threshold
        # # yz-plane (wall ahead)
        # if abs(v[0]) > threshold:
        #     return 'front'
        # # xz-plane (side wall)
        # elif abs(v[1]) > threshold:
        #     return 'side'
        # # xy-plane (ground)
        # elif abs(v[2]) > threshold:
        #     return 'ground'
        # else:
        #     return 'weird'     

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

    def publish_pc(self, pc_list, pub_name):
        """Publishes a point cloud from point list"""
        # Initialize pc2 msg
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = '/rslidar'
        # Convert point cloud list to pc2 msg
        pc = pc2.create_cloud_xyz32(header, pc_list)
        # Publish message
        rospy.Publisher(pub_name, PointCloud2, queue_size=10).publish(pc)

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

# Unused Code

# Remove outliers
# outrem = pc_small.make_RadiusOutlierRemoval()
# outrem.set_radius_search(0.10)
# outrem.set_MinNeighborsInRadius(5)
# pc_smaller = outrem.filter()
# print(pc_cut.size, pc_smaller.size)
# fil = pc_small.make_statistical_outlier_filter()
# fil.set_mean_k(50)
# fil.set_std_dev_mul_thresh(1.0)
# pc_smaller = fil.filter()
# print(pc_small.size, pc_smaller.size)

# icp
# if not self.flag2:
#     self.first_pc = pc_small
#     # Wait for n seconds to get second pcl
#     self.flag2 = True
# else:
#     icp = self.first_pc.make_IterativeClosestPoint()
#     converged, transf, estimate, fitness = icp.icp(self.first_pc, pc_small)
#     print(transf[:-1,3])