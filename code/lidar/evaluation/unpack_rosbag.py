#!usr/bin/env python

import numpy as np
import rosbag
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array
from cv_bridge import CvBridge

def unpack_bag(bag_path, topic, msg_type=None):
    """Extracts all msgs (with time) of a topic from a rosbag"""
    # Load rosbag
    bag = rosbag.Bag(bag_path)

    t_msg = []
    msgs = []
    for topic, msg, t in bag.read_messages(topics=topic):
        if msg_type == 'imu':
            msgs.append((
                [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
                [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
            ))
        elif msg_type == 'quat':
            msgs.append((
                [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
            ))
        elif msg_type == 'car_odom':
            msgs.append((
                msg.handwheel_angle,
                msg.front_right_wheel_speed,
                msg.front_left_wheel_speed,
                msg.rear_right_wheel_speed,
                msg.rear_left_wheel_speed
            ))
        elif msg_type == 'hdl_odom':
            msgs.append(msg.pose.pose)
        elif msg_type == 'lidar':
            msgs.append(pointcloud2_to_xyz_array(msg, remove_nans=True))

        elif msg_type == 'camera':
            CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
        else:
            msgs.append(msg)

        # Time at which msg was recorded
        t_msg.append(t.to_time())
    bag.close()
    return np.asarray(msgs), t_msg


def synchronize_topics(topic1, t_topic1, topic2, t_topic2, t_thresh=0.1):
    """Synchronizes two topics (even if different rate)."""
    if len(topic1) == len(topic2):
        return topic1, topic2

    # Make sure both topics start around the same time and remove first values of
    # signal which is ahead if this is not the case
    topic1, t_topic1, topic2, t_topic2 = correct_for_time_diff(
        topic1, t_topic1, topic2, t_topic2)

    # Assume first array is bigger
    array_big = t_topic1
    array_small = t_topic2
    is_topic1_bigger = True
    # Swap if not
    if len(t_topic1) < len(t_topic2):
        array_small, array_big = array_big, array_small
        is_topic1_bigger = False
    indices = []

    # For each time of the small array, check which is
    # the closest msg in the big array
    for i, v in enumerate(array_small):
        # Offset big array with a msg from small array
        diff = np.abs(array_big - v)
        # Index where time difference is the smallest
        idx = diff.argmin()
        if diff[idx] < t_thresh:
            indices.append(idx)

    print(
        "t_start_diff: {:.3f} s, t_end_diff: {:.3f} s after synchronization".format(
            np.abs(array_small[0] - array_big[indices][0]),
            np.abs(array_big[indices][-1] - array_small[len(array_big[indices]) - 1]),
        )
    )

    if is_topic1_bigger:
        topic1_synched = topic1[indices]
        topic2_synched = topic2[: len(topic1_synched)]
        return topic1_synched, topic2_synched
    else:
        topic2_synched = topic2[indices]
        topic1_synched = topic1[: len(topic2_synched)]
        return topic1_synched, topic2_synched


def correct_for_time_diff(topic1, t_topic1, topic2, t_topic2, t_thresh=0.1):
    """Checks for a time offset of two topics at the beginning and shorts ahead signal"""
    # Assume first topic is behind and convert to numpy array
    topic_behind, topic_ahead = np.asarray(topic1), np.asarray(topic2)
    t_topic_behind, t_topic_ahead = np.asarray(t_topic1), np.asarray(t_topic2)
    is_topic1_behind = True
    # Swap if other way around
    if t_topic2[0] - t_topic1[0] > t_thresh:
        topic_ahead, topic_behind = topic_behind, topic_ahead
        t_topic_ahead, t_topic_behind = t_topic_behind, t_topic_ahead
        is_topic1_behind = False
    # No offset
    elif np.abs(t_topic1[0] - t_topic2[0]) < t_thresh:
        return tuple([np.asarray(x) for x in [topic1, t_topic1, topic2, t_topic2]])

    # Search for time of ahead topic which is closest to start of behind topic
    diff = np.abs(t_topic_ahead - t_topic_behind[0])
    # Index where time difference is the smallest
    idx = diff.argmin()
    # Remove first couple values to make both start at the same time
    topic_ahead = topic_ahead[idx:]
    t_topic_ahead = t_topic_ahead[idx:]

    if is_topic1_behind:
        print("topic1 was behind by {}".format(idx))
        return topic_behind, t_topic_behind, topic_ahead, t_topic_ahead
    else:
        print("topic2 was behind by {}".format(idx))
        return topic_ahead, t_topic_ahead, topic_behind, t_topic_behind