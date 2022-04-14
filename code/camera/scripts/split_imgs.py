"""Split images into training and test sets."""

import json
import shutil

train_json_path = "/home/user/rosbags/final/final_object_detection/train.json"
test_json_path = "/home/user/rosbags/final/final_object_detection/test.json"
train_dest_path = "/home/user/rosbags/final/final_object_detection/imgs_train/"
test_dest_path = "/home/user/rosbags/final/final_object_detection/imgs_test/"

# Load the json file
with open(train_json_path, "r") as f:
    data = json.load(f)
# Get all images and annotations used for training
train_lst = []
for img in data["images"]:
    file_path_img = img["file_name"]
    train_lst.append(file_path_img)
    # Also add the annotations
    # Change jpg to json
    file_path_json = file_path_img.replace(".jpg", ".json")
    train_lst.append(file_path_json)

with open(test_json_path, "r") as f:
    data = json.load(f)
# Get all images and annotations used for test
test_lst = []
for img in data["images"]:
    file_path_img = img["file_name"]
    test_lst.append(file_path_img)
    # Also add the annotations
    # Change jpg to json
    file_path_json = file_path_img.replace(".jpg", ".json")
    test_lst.append(file_path_json)


# Copy all training files to imgs_train
for file in train_lst:
    shutil.copy(file, train_dest_path)
# Copy all test files to imgs_test
for file in test_lst:
    shutil.copy(file, test_dest_path)


print("Copied training files to imgs_train")
