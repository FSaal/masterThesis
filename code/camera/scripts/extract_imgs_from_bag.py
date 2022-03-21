#!/usr/bin/python
# Extract images from a bag file.

# Start up ROS pieces.
# PKG = "my_package"
# import roslib

# roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Reading bag filename from command line or roslaunch parameter.
import os
import sys


class ImageCreator:
    def __init__(self):
        image_type = ".jpg"
        image_topic = "/zed2i/zed_node/left/image_rect_color"
        self.image_index = 0
        save_dir = "/home/user/rosbags/final/object_detection/"
        input_dir = "/home/user/rosbags/final/"
        input_file = "u_d2e.bag"
        rate = 30

        # Use a CvBridge to convert ROS images to OpenCV images so they can be saved.
        self.bridge = CvBridge()

        # Open bag file.
        with rosbag.Bag(input_dir + input_file, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=image_topic):
                # Only every 1 s
                if self.image_index % rate != 0:
                    self.image_index += 1
                    continue
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                output_file = (
                    save_dir + input_file.split(".")[0] + str(self.image_index) + image_type
                )
                cv2.imwrite(output_file, cv_image)
                self.image_index += 1


# Main function.
if __name__ == "__main__":
    # Initialize the node and name it.
    # rospy.init_node("load")
    # Go to class functions that do all the heavy lifting. Do error checking.
    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass
