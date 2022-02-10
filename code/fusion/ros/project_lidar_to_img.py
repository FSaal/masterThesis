#!/usr/bin/env python

from __future__ import division, print_function
from cv_bridge import CvBridge
import numpy as np
import pcl
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
import rospy
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header, Float32
import cv2
from tf.transformations import (
    euler_from_matrix,
    euler_from_quaternion,
    euler_matrix,
    unit_vector,
    quaternion_matrix,
    vector_norm,
)


class VisualDetection(object):
    """Detects ramps using lidar, gets angle and distance to ramp
    and projects lidar points into camera image (tf necessary)"""

    def __init__(self):
        """Initialize class variables"""
        # ROS stuff
        rospy.init_node('lidar_ramp_detection', anonymous=True)
        # Frequency of node [Hz]
        self.rate = 10
        #! Change lidar topic depending on used lidar
        self.is_robos = False
        if self.is_robos:
            lidar_topic = '/right/rslidar_points'
        else:
            lidar_topic = '/velodyne_points'
        camera_topic = '/zed2i/zed_node/left/image_rect_color'

        # Subscriber and publisher
        rospy.Subscriber(lidar_topic, PointCloud2, self.callback_lidar, queue_size=10)
        rospy.Subscriber(camera_topic, Image, self.callback_cam, queue_size=10)
        self.pub_angle = rospy.Publisher('/lidar_ang', Float32, queue_size=10)
        self.pub_distance = rospy.Publisher('/lidar_dist', Float32, queue_size=10)
        self.pub_img = rospy.Publisher('/img', Image, queue_size=10)

        # Define subscriber callback message
        self.cloud = None
        # Flag for lidar car frame alignment, True after tf has been calculated
        #! Change this again
        self.is_calibrated = True

        old_setup = True

        if old_setup:
            # Old recordings (2021)
            self.translation = (1.14-3.5, 0.005, 1.8753)
            self.rotation = (0, 0.0083, 0)
            self.trans = [0.2181620651024292, -0.4272590906175757, -0.322603629076077]
            self.quat = [-0.38795911077076534, 0.508647571298185, -0.6101047218148247, -0.46748005840000295]
        else:
            # New recordings (2022) [m]
            self.translation = (-2.36, 0, 1.8375)
            # RPY [rad]
            self.rotation = (0, 0, 0)
            self.trans = [0.05999999999999983, -0.3898697311173096, -0.33652837214493014]
            self.quat = [-0.4473902184573541, 0.4473902184573541, -0.5475782979891378, -0.5475782979891378]

        # Moving average filter
        self.ang_filter = FilterClass()
        self.dist_filter = FilterClass()

    def callback_lidar(self, msg):
        """Get msg from lidar"""
        self.cloud = msg

    def callback_cam(self, msg):
        self.img = msg

    def spin(self):
        """Run node until crash or user exit"""
        # Frequency at which node runs
        r = rospy.Rate(self.rate)
        # Wait until lidar msgs are received
        while self.cloud is None:
            if rospy.is_shutdown():
                break
            r.sleep()

        while not rospy.is_shutdown():
            # Convert PointCloud2 msg to numpy array
            pc_array = pointcloud2_to_xyz_array(self.cloud, remove_nans=True)

            # Get transformation from lidar to car frame
            if not self.is_calibrated:
                # Get euler angles (roll and pitch) to align lidar with
                # car frame as well as distance of lidar to the ground
                roll, pitch, height = self.align_lidar(pc_array)
                self.is_calibrated = True

            # Apply lidar to car frame transformation
            pc_array_tf = self.transform_pc(pc_array, rpy=self.rotation,
                                            translation_xyz=self.translation)

            # Original cloud after transformation
            self.publish_pc(pc_array_tf, 'pc_big')

            # Reduce point cloud size with a passthrough filter
            pc_array_cut = self.reduce_pc(pc_array_tf, (0, 30), (-2, 2), (-1, 2))

            # Convert numpy array to pcl object
            pc_cut = self.array_to_pcl(pc_array_cut)
            self.publish_pc(pc_cut, 'pc_small_before')

            # Downsample point cloud using voxel filter to further decrease size
            pc_small = self.voxel_filter(pc_cut, 0.1)
            self.publish_pc(pc_small, 'pc_small')

            # Perform RANSAC until no new planes are being detected
            ramp_angle, ramp_distance = self.plane_detection(pc_small, 20, 4)

            # Smooth signals
            avg_angle = self.ang_filter.moving_average(ramp_angle, 5)
            avg_dist = self.dist_filter.moving_average(ramp_distance, 5)

            print('angle: {:.2f}, dist: {:.2f}'.format(avg_angle, avg_dist))

            self.pub_angle.publish(avg_angle)
            self.pub_distance.publish(avg_dist)
            r.sleep()

    def project(self, pc_array, color=[255, 255, 0, 255]):
        # Convert image to numpy array (grayscale?)
        img_cv = CvBridge().imgmsg_to_cv2(self.img, desired_encoding="rgba8")

        # Transformation from lidar to camera left optical (using output from tf node)


        # Lidar points
        # Remove points behind car (because not visible in camera frame)
        # pc_array = pc_array[pc_array[:, 0] > 0]

        # Get rotation vector from lidar frame to camera frame
        # Convert quaternion to rotation matrix
        rotMat = quaternion_matrix(self.quat)[:3, :3]
        # Convert rotation matrix to rotation vector
        rvec, _ = cv2.Rodrigues(rotMat)

        # Translation vector from lidar to camera frame
        tvec = np.asarray(self.trans)

        # Camera matrix (K)
        cameraMatrix = np.array(
            [
                [519.9877319335938, 0.0, 630.9716796875],
                [0.0, 519.9877319335938, 352.2940673828125],
                [0.0, 0.0, 1.0],
            ]
        )

        # Distortion coefficients (D)
        distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Project 3D lidar points onto 2D plane
        # Projection of 3D points onto 2D plane
        lidar_img_points, _ = cv2.projectPoints(pc_array, rvec, tvec, cameraMatrix, distCoeffs)
        lidar_img_points = lidar_img_points.reshape(len(lidar_img_points), 2)

        # Remove points which do not lie inside image plane
        lidar_img_points_small, indices = self.cut_pc(lidar_img_points, img_cv.shape[1], img_cv.shape[0])
        # Convert to int
        lidar_img_points_small = lidar_img_points_small.astype(np.int16)
        # Swap columns
        lidar_img_points_small[:, [0, 1]] = lidar_img_points_small[:, [1, 0]]

        # Change sth
        # img_cv[460:500, 1100:1200] = np.array([255, 255, 255, 255], dtype=np.uint8)
        for point in lidar_img_points_small:
            img_cv[point[0]-1:point[0]+1, point[1]-1:point[1]+1] = np.array(color, dtype=np.uint8)
        # Convert back to img msg
        img_new = CvBridge().cv2_to_imgmsg(img_cv, "rgba8")
        self.pub_img.publish(img_new)

    def cut_pc(self, pc2D, x1=1280, y1=720, idx=None):
        """Cuts points of 2D pointcloud which lie outside of camera image"""
        if idx is None:
            idx = (
                (pc2D[:, 0] > 0) & (pc2D[:, 0] < x1) & (pc2D[:, 1] > 0) & (pc2D[:, 1] < y1)
            )
            pc_smol = pc2D[idx]
            return pc_smol, idx
        return pc2D[idx]

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
        print('\n__________LIDAR__________')
        print('Euler angles in deg to tf lidar to car frame:')
        print('Roll: {:.2f}\nPitch: {:.2f}'.format(
            np.rad2deg(roll), np.rad2deg(pitch)))
        print('Lidar height above ground: {:.2f} m\n'.format(lidar_height))
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
                raise RuntimeError ('No ground could be detected.')

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
        """Gets quaternion to align vector 1 with vector 2"""
        # Make sure both vectors are unit vectors
        v1_uv, v2_uv = unit_vector(vec1), unit_vector(vec2)
        cross_prod = np.cross(v1_uv, v2_uv)
        dot_prod = np.dot(v1_uv, v2_uv)

        # Rotation axis
        axis = cross_prod / vector_norm(cross_prod)
        # Rotation angle (rad)
        ang = np.arccos(dot_prod)

        # Quaternion ([x,y,z,w])
        quat = np.append(axis*np.sin(ang/2), np.cos(ang/2))
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
        rot = euler_matrix(roll, pitch, yaw, 'sxyz')[:3, :3]
        # Apply rotation
        pc_tf = np.inner(pc, rot)

        # Translation to rear wheel axis
        translation_to_base_link = [transl_x, transl_y, transl_z]
        # Translation to front of the car
        # translation_to_front = [-3.5, 0, 0]
        translation_to_front = [0, 0, 0]
        # Combine both translations
        translation = np.add(translation_to_base_link, translation_to_front)
        # Combine rotation and translation
        pc_tf += translation
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
        # Count number of iterations
        counter = 0
        # Standard values for ramp angle and distance if no detection
        ramp_stats = (0, 0)

        self.down_ramp_detection(pc)
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
                print('EMPTY')
                return ramp_stats

            # Ignore planes parallel to the side or front walls
            if self.is_plane_near_ground(n_vec):
                # Check if ramp conditions are fullfilled
                is_ramp, ramp_ang, ramp_dist, data = self.ramp_detection(
                    plane, n_vec, (3, 9), (2, 6))
                # Ramp conditions met
                if is_ramp:
                    print('RAMP')
                    self.publish_pc(plane, 'ramp')

                    # Transform plane back to original frame
                    plane_tf = self.transform_pc(data, rpy=tuple(-np.array(self.rotation)),
                                                 translation_xyz=tuple(-np.array(self.translation)))
                    # Project points onto camera frame
                    # t0 = rospy.get_time()
                    self.project(plane_tf, color=[255, 0, 0, 255])
                    # print(rospy.get_time() - t0)

                    return (ramp_ang, ramp_dist)
                # Ground
                else:
                    print('GROUND')
                    # Transform plane back to original frame
                    plane_tf = self.transform_pc(plane.to_array(), rpy=tuple(-np.array(self.rotation)),
                                                 translation_xyz=tuple(-np.array(self.translation)))
                    # Project points onto camera frame
                    # t0 = rospy.get_time()
                    self.project(plane_tf, [0, 255, 255, 255])
                    # print(rospy.get_time() - t0)

                    # print('Ground ({:.2f} deg)'.format(ramp_ang))
                    self.publish_pc(plane, 'ground')
            else:
                # print('WALL')
                pass
            counter += 1
        return ramp_stats

    def down_ramp_detection(self, plane):
        """Checks if conditions to be considered a downwards ramp are met."""
        # Convert pcl plane to numpy array
        plane_array = plane.to_array()

        # Area 7 m in front of car
        front_area = plane_array[
            (plane_array[:, 0] > 4) &
            (plane_array[:, 0] < 7)]
        # print(max(front_area[:, 2]), min(front_area[:, 2]))
        front_area_ground = front_area[
            (front_area[:, 2] < 0.2) &
            (front_area[:, 2] > -0.2)
        ]
        self.publish_pc(front_area_ground, 'spot')
        # print(front_area_ground.shape[0])
        # print(plane_array.shape, front_area.shape, front_area_ground.shape)

        return False

    def ramp_detection(self, plane, n_vec, angle_range, width_range, n_nearest=20):
        """Checks if conditions to be considered a ramp are fullfilled.

        The following values of the plane are being calculated and
        checked whether they lie within the desired range:
        - angle (calculated between normal vec of ground_plane and this plane)
        - width

        If all conditions are met return True, else False
        """
        # Convert pcl plane to numpy array
        plane_array = plane.to_array()

        # Calculate angle [deg] between normal vector of plane and ground
        angle = self.angle_calc([0, 0, 1], n_vec)
        # Get ramp width (difference between highest and lowest y-values)
        # Sort y vector from lowest to highest
        y_points_sorted = np.sort(plane_array[:, 1])
        width = np.mean(y_points_sorted[-n_nearest:]) - np.mean(y_points_sorted[:n_nearest])
        # Ramp distance (median x-value of nearest points of the plane)
        dist = np.median(np.sort(plane_array[:, 0])[:n_nearest])

        # Assert ramp angle and width thresholds
        if (angle_range[0] <= angle <= angle_range[1]
                and width_range[0] <= width <= width_range[1]):
            return True, angle, dist, plane_array
        return False, angle, dist, plane_array

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

    # @staticmethod
    def publish_pc(self, pc, pub_name):
        """Publishes a point cloud from point list"""
        # # Make sure that pc is a list
        # if isinstance(pc, list):
        #     pass
        # # Convert to list if numpy array
        # elif isinstance(pc, np.ndarray):
        #     pc = list(pc)
        # # Convert to list if pcl point cloud
        # else:
        #     pc = pc.to_list()

        # # Initialize pc2 msg
        # header = Header()
        # header.stamp = rospy.Time.now()
        # if self.is_robos:
        #     header.frame_id = '/rslidar'
        # else:
        #     header.frame_id = '/velodyne'
        # # Convert point cloud list to pc2 msg
        # pc = pc2.create_cloud_xyz32(header, pc)
        # # Publish message
        # rospy.Publisher(pub_name, PointCloud2, queue_size=10).publish(pc)
        pass


class FilterClass(object):
    """Filters a signal"""

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
            if self.values:
                self.sum -= self.values.pop(0)
        # Prevent division by zero
        if len(self.values) != 0:
            return self.sum / len(self.values)
        else:
            return 0


class PerformanceMeasure(object):
    """Can be used to measure time between two statements"""

    def __init__(self):
        self.total_time = 0
        self.counter = 1

    def performance_calc(self, start_time, name=""):
        """Prints the time"""
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
        VD = VisualDetection()
        VD.spin()
    except rospy.ROSInterruptException:
        pass
