"""Convert annotations from labelme format to COCO format"""
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "/home/user/rosbags/final/object_detection"

# set export dir
export_dir = "/home/user/rosbags/final/coco/gen_coco.json"

# set train split rate
train_split_rate = 0.85

# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, export_dir)
