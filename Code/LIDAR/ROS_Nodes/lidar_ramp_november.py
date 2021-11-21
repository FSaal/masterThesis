#!/usr/bin/env python

# Limit CPU usage (of numpy)
# ! must be called before importing numpy
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from tf.transformations import euler_from_quaternion, euler_matrix, unit_vector, vector_norm
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
        # * Change lidar topic depending on used lidar
        lidar_topic = '/right/rslidar_points'
        self.sub_lidar = rospy.Subscriber(lidar_topic, PointCloud2, self.callback_lidar, queue_size=10)
        self.pub_angle = rospy.Publisher('/ramp_angle_lidar', Float32, queue_size=10)
        self.pub_stuff = rospy.Publisher('/some_stats', Float32MultiArray, queue_size=10)
        # Gets set True if lidar topic has started publishing
        self.subbed_lidar = False
        # Gets set True if initial tf from lidar to car frame has been performed
        self.calibrated = False
        self.buffer = []
        self.dist_buffer = []
        self.ang_filter = FilterClass()
        self.dist_filter = FilterClass()
        
    def callback_lidar(self, msg):
        self.cloud = msg
        self.subbed_lidar = True

    def spin(self):
        # Robosense Lidar has a rate of 10 Hz, set rate of node to the same
        r = rospy.Rate(10)

        while not rospy.is_shutdown():
            # Wait until lidar msgs are received
            if not self.subbed_lidar:
                continue
            # Convert PointCloud2 msg to numpy array
            pc_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.cloud, remove_nans=True)

            # Get transformation from lidar to car frame
            if not self.calibrated:
                # Get euler angles (roll and pitch) to align lidar with car frame
                self.rp = self.align_lidar(pc_array)
                self.calibrated = True

            # Apply lidar to car frame transformation (adjust yaw angle manually)
            pc_array_tf = self.transform_pc(pc_array, roll=self.rp[0], pitch=self.rp[1], 
                yaw=0, transl_x=-1.944)

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
            ramp_angle, ramp_distance = self.plane_detection(pc_small, 100, 4)
            
            # Smooth signals
            avg_angle = self.ang_filter.moving_average(ramp_angle, 5)
            avg_dist = self.dist_filter.moving_average(ramp_distance, 5)

            print('{:.2f} vs {:.2f}'.format(avg_angle, avg_dist))
            r.sleep()
            
    def align_lidar(self, pc_array):
        """Calculate roll and pitch angle to align Lidar with car frame"""
        # Convert numpy array to pcl point cloud
        pc = pcl.PointCloud()
        pc.from_array(pc_array.astype('float32'))

        # Get normal vector of ground plane
        ground_vec_lidar = self.ground_detection(pc)
        # Normal vector ground plane in car frame 
        # (assuming car stands on a flat surface)
        ground_vec_car = [0, 0, 1]

        # Quaternion to align lidar vec to car vec
        quat = self.quat_from_vectors(ground_vec_lidar, ground_vec_car)
        # Calculate euler angles
        roll, pitch, yaw = euler_from_quaternion(quat)

        print('Euler angles in deg to tf lidar to car frame:')
        print('Roll: {:05.2f}\nPitch: {:05.2f}'.format(
            np.degrees(roll), np.degrees(pitch)))
        return [roll, pitch]

    def ground_detection(self, pc):
        """Detect ground plane and get normal vector"""
        # Get most dominant plane (assume that this is the ground plane)
        indices, coefficients = self.ransac(pc)
        # Remove 4th plane coefficient (translation) to get normal vector
        ground_vec = coefficients[:-1]
        return ground_vec

    def quat_from_vectors(self, vec1, vec2):
        """Quaternion that aligns vec1 to vec2"""
        # Normalize vectors
        a, b = unit_vector(vec1), unit_vector(vec2)
        c = np.cross(a, b)
        d = np.dot(a, b)

        # Rotation axis
        ax = c / vector_norm(c)
        # Rotation angle
        a = np.arctan2(vector_norm(c), d)
        # Quaternion
        quat = np.append(ax*np.sin(a/2), np.cos(a/2))
        return quat

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
        pc_cut = pc[
            (pc[:, 0] > x_lower) & 
            (pc[:, 0] < x_upper) & 
            (pc[:, 1] > y_lower) & 
            (pc[:, 1] < y_upper) & 
            (pc[:, 2] > z_lower) & 
            (pc[:, 2] < z_upper)
            ]
        return pc_cut

    def voxel_filter(self, pc, leaf_size):
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
        # Ground vector
        g_vec = None
        # Count number of iterations
        counter = 0
        # Standard values for ramp angle and distance if no detection
        ramp_stats = (0,0)
        while pc.size > min_points and counter < max_planes:
            # Detect most dominate plane and get inliers and normal vector
            indices, coefficients = self.ransac(pc)
            # Normal vector of plane
            n_vec = coefficients[:-1]

            # Split pointcloud in outliers of plane and inliers
            pc, plane = self.split_pc(pc, indices)

            # Exit if plane is empty (RANSAC did not find anything)
            if not plane:
                return ramp_stats
    
            # Ignore planes parallel to the side or front walls
            if self.is_plane_near_ground(n_vec):
                # First ground like detection is most probably the ground
                if g_vec is None:
                    g_vec = n_vec
                # Either ground is detected again or potential ramp
                else:
                    # Check if ramp conditions are fullfilled
                    is_ramp, ramp_ang, ramp_dist =  self.ramp_detection(
                        plane, g_vec, n_vec, 3, 8, 2, 6)
                    # Ramp conditions met
                    if is_ramp:
                        return (ramp_ang, ramp_dist)
                    else:  
                        continue
            counter += 1
        return ramp_stats

    def ramp_detection(
            self, plane, g_vec, n_vec, min_angle, max_angle, 
            min_width, max_width, logging=False):
        """Checks if conditions to be considered a ramp are fullfilled.
        
        The following values of the plane are being calculated and 
        checked whether they lie within the desired range:
        - angle (calculated between normal vec of ground_plane and this plane)
        - width
        If all conditions are met return True, else False
        """
        # Convert pcl plane to numpy array
        plane_array = np.array(plane)

        # Calculate angle [deg] between normal vector of plane and ground
        angle = self.angle_calc(g_vec, n_vec)
        # Get ramp width (difference between y-values)
        width = max(plane_array[:, 1]) - min(plane_array[:, 1])
        # Ramp distance (average x-value of nearest points of the plane)
        n_nearest = 10
        dist = np.mean(np.sort(plane_array[:n_nearest,0]))

        # Assert ramp angle and width thresholds 
        if min_angle <= angle <= max_angle and min_width <= width <= max_width:
            return True, angle, dist
        else:
            return False, angle, dist

    def ransac(self, pc):
        """Find inliers and normal vector of dominant plane"""
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

        :param pc:              [PCL] Point cloud
        :param inliers:         [List] Indices of inliers of plane
        :return pc_outliers:    [PCL] Point cloud w/o plane inliers
        :return detected_plane: [List] Inlier points
        """
        # Get point cooridnates of plane
        detected_plane = [pc[i] for i in inliers]
        # Point cloud of detected plane (inliers)
        # pc_inliers = pc.extract(inliers)

        # Point cloud of outliers, difference between whole pc and plane inliers
        outlier_indices = list(set(np.arange(pc.size)).symmetric_difference(inliers))
        pc_outliers = pc.extract(outlier_indices)

        return pc_outliers, detected_plane

    def is_plane_near_ground(self, v, threshold=0.8):
        """Returns True if plane is on the ground (and false if e.g. side wall)"""
        # z-axis points up
        return abs(v[2]) > threshold

    def angle_calc(self, v1, v2, degrees=True):
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
        
class FilterClass():
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
            if len(self.values) > 0:
                self.sum -= self.values.pop(0)
        # Prevent division by zero
        try:
            return float(self.sum) / len(self.values)
        except ZeroDivisionError:
            return 0

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