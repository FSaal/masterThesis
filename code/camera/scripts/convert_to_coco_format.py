"""Convert annotations from labelme format to COCO format"""
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "/home/user/rosbags/final/final_object_detection/imgs"

# set export dir
export_dir = "/home/user/rosbags/final/final_object_detection/dataset_full.json"

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir)
