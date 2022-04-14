#!/usr/bin/python
# Extract images from a bag file.

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
        input_dir = "/home/user/rosbags/final/newramps/"
        save_dir = "/home/user/rosbags/final/newramps/imgs/"
        input_file = "2021-12-10-10-23-00.bag"
        rate = 15

        # Use a CvBridge to convert ROS images to OpenCV images so they can be saved.
        self.bridge = CvBridge()

        bag_lst = os.listdir(input_dir)
        # remove directories
        bag_lst = [x for x in bag_lst if not os.path.isdir(os.path.join(input_dir, x))]

        for bag in bag_lst:
            input_file = bag
            # Open bag file.
            with rosbag.Bag(input_dir + input_file, "r") as bag:
                for topic, msg, t in bag.read_messages(topics=image_topic):
                    # Only every 1 s
                    if self.image_index % rate != 0:
                        self.image_index += 1
                        continue
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                    output_file = (
                        save_dir
                        + input_file.split(".")[0]
                        + "_"
                        + str(self.image_index)
                        + image_type
                    )
                    cv2.imwrite(output_file, cv_image)
                    self.image_index += 1
            self.image_index = 0


# Main function.
if __name__ == "__main__":
    image_creator = ImageCreator()
