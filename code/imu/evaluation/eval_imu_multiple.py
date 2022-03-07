import numpy as np
import pandas as pd
from scipy import signal

# Load methods to extract data from rosbags
from unpack_rosbag import unpack_bag, synchronize_topics
from imu_ramp_offline import ImuRampDetect

# Standard plotly imports
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"

# ROS topics
ODOM_TOPIC = "/eGolf/sensors/odometry"
IMU_TOPIC = "/zed2i/zed_node/imu/data"
# IMU_TOPIC = '/imu/data'

# Rosbags path
BAG_DIRECTORY = "/home/user/rosbags/final/slam/"

BAG_NAMES = [
    "d_d2r2s_lidar_wo_odom_hdl.bag",
    "d_d2r2s_odom_hdl.bag",
    "d_e2q_hdl.bag",
    "straight_wo_ramps_odom_hdl.bag",
    "u_c2s_half_odom_hdl.bag",
    "u_c2s_half_odom_stereo_hdl.bag",
    "u_c2s_hdl.bag",
    "u_c2s_stop_hdl.bag",
    "u_d2e_hdl.bag",
    "u_s2c_half_odom_hdl.bag",
    "u_s2c2d_hdl.bag",
]


def get_data(bag_name, imu_topic=IMU_TOPIC):
    bag_path = BAG_DIRECTORY + bag_name
    # IMU data (angular velocity and linear acceleration)
    imu, t_imu = unpack_bag(bag_path, imu_topic, "imu")
    # Odom data (wheel speeds and handwheel angle)
    odom, t_odom = unpack_bag(bag_path, ODOM_TOPIC, "car_odom")

    # Ignore odom if none has been recorded
    if len(odom) == 0:
        f = 400
        odom = np.zeros((len(imu), 5))
        if imu_topic == "/imu/data":
            f = 100
    else:
        # Synchronize both topics (IMU is 400 Hz whereas odom is only 100 Hz)
        # Downsamples higher freq signal and matches start and end time
        imu, odom = synchronize_topics(imu, t_imu, odom, t_odom)
        f = 100

    # Split linear acceleration and angular velocity
    ang_vel = imu[:, 0]
    lin_acc = imu[:, 1]

    t = np.linspace(0, odom.shape[0] / f, odom.shape[0])

    return lin_acc, ang_vel, odom, t, f


def run_algo(lin_acc, ang_vel, odom, f):
    ird = ImuRampDetect(f)
    tf, gyr_bias = ird.align_imu(lin_acc, ang_vel, odom)
    angles = []
    for i, _ in enumerate(lin_acc):
        _, _, angs = ird.spin(lin_acc[i], ang_vel[i], odom[i], tf, gyr_bias)
        angles.append(angs)
    return angles


def visualize_results(angles, t, bag_name):
    # Convert from radians to degree
    angles = np.asarray(np.rad2deg(angles))
    # Convert to pandas dataframe (for plotting)
    df_angles = pd.DataFrame(angles, index=t, columns=["acc", "gyr", "acc_odom", "compl_0.01"])
    fig = px.line(df_angles)
    fig.update_layout(title=bag_name, xaxis_title="Time [s]", yaxis_title="Pitch angle [deg]")
    fig.show()


all_angles = []
for _, bag_name in enumerate(BAG_NAMES):
    print("\nLoading bag {}".format(bag_name))
    lin_acc, ang_vel, odom, t, f = get_data(bag_name)
    print("Running algorithm")
    angles = run_algo(lin_acc, ang_vel, odom, f)
    # break
    all_angles.append(angles)
    visualize_results(angles, t, bag_name)
