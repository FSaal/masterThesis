#!/usr/bin/env python
"""This script runs the lidar detection algorithm on all recorded bags and
calculates scores such as detection rate, standard deviation of angle, width
etc. estimations for different distance intervals and outputs them grouped by
the ramp type in a latex table format.
"""
from __future__ import division, print_function

import numpy as np
import pandas as pd
from IPython.display import display

# Load offline ramp detection algorithm
from lidar_ramp_offline import VisualDetection

# Load methods to extract data from rosbags
from unpack_rosbag import unpack_bag, synchronize_topics


class GetScore(object):
    """Run lidar ramp detection on rosbag and calculate some scores"""

    def __init__(self, bag_name, x_range, y_range, ramp_type, true_angle, true_width, true_length):
        self.bag_name = bag_name
        self.x_range = x_range
        self.y_range = y_range
        self.ramp_type = ramp_type
        self.true_angle = true_angle
        self.true_width = true_width
        self.true_length = true_length

    @staticmethod
    def extract_data(bag_path):
        """Get odometer and lidar data from rosbag"""
        # Odometer from hdl_slam
        pose_async, t_pose = unpack_bag(bag_path, ODOM_TOPIC, "hdl_odom")
        # Velodyne point cloud
        lidar_async, t_lidar = unpack_bag(bag_path, LIDAR_TOPIC)

        # Synchronize both topics (odom usually is behind lidar)
        pose, lidar = synchronize_topics(pose_async, t_pose, lidar_async, t_lidar)
        return lidar, pose

    def run_the_algorithm(self, lidar, pose):
        """Perform ramp detection on data"""
        # Create instance of class (using standard parameters):
        vd = VisualDetection(self.x_range[0])
        # Lists to fill, will contain entry for each frame
        planes = []
        ramp_stats = []
        true_dists = []
        for i, _ in enumerate(lidar):
            plane_points, data, _ = vd.spin(lidar[i], pose[i])
            # Point coordinates of points detected as ramp
            planes.append(plane_points)
            # Ramp stats [angle, width, dist, true_dist]
            ramp_stats.append(data)
            # True distance to start of ramp
            # (necessary because ramp_stats empty for frames where no ramp)
            true_dist = self.x_range[0] - pose[i].position.x
            true_dists.append(true_dist)
        return planes, ramp_stats, true_dists

    @staticmethod
    def get_ramp_frames(planes, plane_stats):
        """Removes all samples, where no ramp has been detected"""
        # Remove samples where no ramp has been detected
        ramp_arrays = [x for x in planes if not isinstance(x, list)]
        ramp_stats = [x for x in plane_stats if x]
        return ramp_arrays, ramp_stats

    @staticmethod
    def convert_to_df(ramp_arrays):
        """Convert array of as ramp detected points into a dataframe
        and add sample and point index. Each row is one point"""
        # Convert list to dictionary
        dic = []
        for i, arr in enumerate(ramp_arrays):
            for j, point in enumerate(arr):
                dic.append(
                    {
                        "sampleIdx": i,
                        "pointIdx": j,
                        "x": point[0],
                        "y": point[1],
                        "z": point[2],
                    }
                )
        # And finally to pandas data frame
        ramp_points_df = pd.DataFrame(dic)
        # Reorder columns
        ramp_points_df = ramp_points_df[["sampleIdx", "pointIdx", "x", "y", "z"]]
        return ramp_points_df

    def ground_truth_check(self, ramp_points_df, ramp_stats, only_before_ramp=True):
        """Calculate percentage of as ramp identified points, which
        do really inside the ramp region, and return them together
        with estimated angle, width, distance and true distance in a df"""
        # Check if a point lies within ramp region
        lies_inside = []
        for i, x in enumerate(ramp_points_df["x"]):
            if (
                self.x_range[0] < x < self.x_range[1]
                and self.y_range[0] < ramp_points_df["y"][i] < self.y_range[1]
            ):
                # True if x and y coordinate inside region
                lies_inside.append(True)
            else:
                lies_inside.append(False)
        # Add column (bool: if point lies in region) to data frame
        ramp_points_df["inlier"] = lies_inside

        # Calculate how many points of each sample lie in ramp region
        true_inliers = []
        samples_num = ramp_points_df.sampleIdx.max() + 1
        for i in range(samples_num):
            # Bool list of inliers and outliers of sample
            bool_lst = ramp_points_df[ramp_points_df["sampleIdx"] == i]["inlier"]
            # Percentage of inliers of sample
            true_inlier_perc = sum(bool_lst) / float(len(bool_lst))
            true_inliers.append(true_inlier_perc)

        # New dataframe with stats for each frame
        # Structure reminder of ramp_stats: [angle, width, dist, true_dist, length]
        dic = []
        for i in range(samples_num):
            dic.append(
                {
                    "sampleIdx": i,
                    "TrueInliers": true_inliers[i],
                    "Angle": ramp_stats[i][0],
                    "Width": ramp_stats[i][1],
                    "Dist": ramp_stats[i][2],
                    "TrueDist": ramp_stats[i][3],
                    "Length": ramp_stats[i][4],
                }
            )
        # Convert dictionary to dataframe
        df_stats = pd.DataFrame(dic)
        # Reorder columns
        df_stats = df_stats[
            ["sampleIdx", "TrueInliers", "Angle", "Width", "Dist", "TrueDist", "Length"]
        ]
        # Remove all estimations from when after the ramp has been entered
        if only_before_ramp:
            df_stats = df_stats[df_stats["TrueDist"] > 0]
        return df_stats

    def calc_score(self, df_stats, true_dists, min_dist, max_dist, step_dist):
        """Calculates some scores to determine performance"""
        scores = []
        self.min_rmse(df_stats)
        for _, min_d in enumerate(range(min_dist, max_dist, step_dist)):
            max_d = min_d + step_dist
            # Reduce df to range
            df_range = df_stats[(min_d < df_stats["TrueDist"]) & (df_stats["TrueDist"] < max_d)]
            # Samples recorded in the given distance range
            true_dists = np.asarray(true_dists)

            # Assuming ramp is visible in every frame --> num of frames == expected detections
            frames = len(true_dists[(min_d < true_dists) & (true_dists < max_d)])
            # Actual detections only counts if at least 50% of points lie inside ramp region
            detections_tp = len(df_range[(df_range["TrueInliers"] > 0.5)])
            # Ramp has been detected, but less than 50% of points lie in ramp region
            detections_fp = len(df_range[(df_range["TrueInliers"] < 0.5)])

            # Calculate ratio
            try:
                true_positives = float(detections_tp) / frames * 100
                false_positives = float(detections_fp) / frames * 100
            except ZeroDivisionError:
                true_positives = np.NaN
                false_positives = np.NaN

            # Calculate std_dev and stuff
            diff_dist = df_range["Dist"] - df_range["TrueDist"]
            diff_angle = df_range["Angle"] - self.true_angle
            diff_width = df_range["Width"] - self.true_width
            diff_length = df_range["Length"] - self.true_length
            # diff_length = df_range["Length"]
            diff_all = np.vstack((diff_dist, diff_angle, diff_width, diff_length))
            rmse = np.sqrt(np.mean(diff_all ** 2, axis=1))
            # print("RMSE: {}".format(rmse))
            # Print information for every bag
            # print(
            #     (
            #         "In the range {}m to {}m {} frames have been recorded with "
            #         "{:.2f}% true positives and {:.2f} false positives"
            #     ).format(min_d, max_d, frames, true_positives, false_positives)
            # )
            scores.append(
                (
                    self.ramp_type,
                    min_d,
                    max_d,
                    frames,
                    true_positives,
                    false_positives,
                    rmse[0],
                    rmse[1],
                    rmse[2],
                    rmse[3],
                )
            )
        return scores

    def min_rmse(self, df_stats):
        """Find best true value to minimize rmse"""
        df_range = df_stats[(1 < df_stats["TrueDist"]) & (df_stats["TrueDist"] < 15)]
        vals = []
        for i in np.arange(2.5, 4.5, 0.02):
            # diff_angle = df_range["Angle"] - self.true_angle
            diff_width = df_range["Width"] - i
            rmse = np.sqrt(np.mean(diff_width ** 2))
            vals.append((rmse, i))
        vals = np.asarray(vals)
        min_rmse = np.argmin(vals[:, 0])
        print("Best RMSE: {} with width {}".format(vals[min_rmse, 0], vals[min_rmse, 1]))
        vals = []
        for i in np.arange(6, 9, 0.1):
            diff_angle = df_range["Angle"] - i
            rmse = np.sqrt(np.mean(diff_angle ** 2))
            vals.append((rmse, i))
        vals = np.asarray(vals)
        min_rmse = np.argmin(vals[:, 0])
        print("Best RMSE: {} with angle {}".format(vals[min_rmse, 0], vals[min_rmse, 1]))
        vals = []
        for i in np.arange(-1, 1, 0.02):
            diff_dist = df_range["Dist"] - df_range["TrueDist"] + i
            rmse = np.sqrt(np.mean(diff_dist ** 2))
            vals.append((rmse, i))
        vals = np.asarray(vals)
        min_rmse = np.argmin(vals[:, 0])
        print("Best RMSE: {} with distance {}".format(vals[min_rmse, 0], vals[min_rmse, 1]))
        vals = []
        for i in np.arange(6, 15, 0.1):
            diff_length = df_range["Length"] - i
            rmse = np.sqrt(np.mean(diff_length ** 2))
            vals.append((rmse, i))
        vals = np.asarray(vals)
        min_rmse = np.argmin(vals[:, 0])
        print("Best RMSE: {} with length {}".format(vals[min_rmse, 0], vals[min_rmse, 1]))

    def boss_method(self, min_dist=0, max_dist=30):
        """Run all methods above"""
        # Load rosbag
        bag_path = BAG_DIRECTORY + self.bag_name
        lidar, pose = self.extract_data(bag_path)
        planes, plane_stats, true_dists = self.run_the_algorithm(lidar, pose)
        ramp_arrays, ramp_stats = self.get_ramp_frames(planes, plane_stats)
        ramp_points_df = self.convert_to_df(ramp_arrays)
        df_stats = self.ground_truth_check(ramp_points_df, ramp_stats)
        scores = self.calc_score(df_stats, true_dists, min_dist, max_dist, 5)
        return scores


def run_evaluation(bag_info):
    """Get scores from all bags"""
    scores = []
    for _, v in enumerate(bag_info):
        # Unpack dictionary
        bag_name = v["bag_name"]
        ramp_type = v["ramp_type"]
        # if ramp_type != "u_d2e":
        #     continue
        x_range, y_range = v["xy_range"]
        true_angle = v["true_angle"]
        true_width = v["true_width"]
        true_length = v["true_length"]

        print("Loading bag: {}".format(bag_name))
        # Create instance of class
        gs = GetScore(bag_name, x_range, y_range, ramp_type, true_angle, true_width, true_length)
        # Run algorithm + evaluation
        score = gs.boss_method()
        scores.append(score)
        print("\n\n")
    # Flatten scores list
    scores_flat = [item for sublist in scores for item in sublist]
    return scores_flat


def create_latex_table(scores):
    """Create latex table with correct formatting"""
    # Convert to dataframe
    df = pd.DataFrame(scores)
    # Rename columns
    df.columns = [
        "rampType",
        "min_d",
        "max_d",
        "frames",
        "truePositives",
        "falsePositives",
        "rmse_dist",
        "rmse_ang",
        "rmse_width",
        "rmse_length",
    ]
    # Group by ramp type and distance range and calculate sum of frames and avg detection rate
    df = df.groupby(["rampType", "min_d", "max_d"], as_index=False).agg(
        {
            "frames": "sum",
            "truePositives": "mean",
            "falsePositives": "mean",
            "rmse_dist": "mean",
            "rmse_ang": "mean",
            "rmse_width": "mean",
            "rmse_length": "mean",
        }
    )
    display(df)
    print("\nAnd in latex format:")
    # Print each row
    for i in range(len(df)):
        row = df.iloc[i]
        if i == 0:
            ramp_type = "\multirow{{6}}{{*}}{{{}}}".format(row["rampType"])
        else:
            ramp_type = ""
        print(
            "{} & \\SIrange{{{}}}{{{}}}{{\\metre}} & {} & {:.2f}\% & {:.2f}\% & \SI{{{:.2f}}}{{\\metre}} & \SI{{{:.2f}}}{{\\metre}} & \SI{{{:.2f}}}{{\\degree}} & \SI{{{:.2f}}}{{\\metre}}\\\\".format(
                ramp_type,
                row["min_d"],
                row["max_d"],
                row["frames"],
                row["truePositives"],
                row["falsePositives"],
                row["rmse_dist"],
                row["rmse_width"],
                row["rmse_ang"],
                row["rmse_length"],
            )
        )


# Rosbags path
BAG_DIRECTORY = "/home/user/rosbags/final/slam/"
# ROS topics
ODOM_TOPIC = "/odom"
LIDAR_TOPIC = "/velodyne_points"

# List of dictionaries with info about recording
BAG_INFO = [
    {
        "bag_name": "u_c2s_half_odom_hdl.bag",
        "ramp_type": "u_c2s",
        "xy_range": [[20.3, 33], [-0.9, 2.8]],
        "true_angle": 7.2,
        "true_width": 3.94,
        "true_length": 11.97,
    },
    {
        "bag_name": "u_c2s_half_odom_stereo_hdl.bag",
        "ramp_type": "u_c2s",
        "xy_range": [[23.8, 36], [-3.3, 0.5]],
        "true_angle": 7.2,
        "true_width": 3.94,
        "true_length": 11.97,
    },
    {
        "bag_name": "u_c2s_hdl.bag",
        "ramp_type": "u_c2s",
        "xy_range": [[45, 58], [-1.9, 1.8]],
        "true_angle": 7.2,
        "true_width": 3.94,
        "true_length": 11.97,
    },
    {
        "bag_name": "u_c2s_stop_hdl.bag",
        "ramp_type": "u_c2s",
        "xy_range": [[38, 51], [-1.5, 2.2]],
        "true_angle": 7.2,
        "true_width": 3.94,
        "true_length": 11.97,
    },
    {
        "bag_name": "u_d2e_hdl.bag",
        "ramp_type": "u_d2e",
        "xy_range": [[32.5, 44], [2, 5.5]],
        "true_angle": 7.4,
        "true_width": 3.9,
        "true_length": 11.8,
    },
    {
        "bag_name": "u_s2c_half_odom_hdl.bag",
        "ramp_type": "u_s2c",
        "xy_range": [[42, 56], [-2.2, 2]],
        "true_angle": 6.5,
        "true_width": 3.96,
        "true_length": 14.15,
    },
    {
        "bag_name": "u_s2c2d_part1_hdl.bag",
        "ramp_type": "u_s2c",
        "xy_range": [[47.3, 62], [-2.8, 1.5]],
        "true_angle": 6.5,
        "true_width": 3.96,
        "true_length": 14.15,
    },
    # {
    #     "bag_name": "u_s2c2d_part2_hdl.bag",
    #     "ramp_type": "u_c2d",
    #     "xy_range": [[47, 58.8], [36.5, 40.5]],
    #     "true_angle": 7,
    #     "true_width": 3.85,
    #     "true_length": 99,
    # },
]

# Run evaluation
SCORES = run_evaluation(BAG_INFO)
# Display results in table format
create_latex_table(SCORES)
