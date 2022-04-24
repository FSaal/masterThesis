import numpy as np
import pandas as pd
from IPython.display import display
from scipy.signal import resample
from tf.transformations import euler_from_quaternion
from sklearn.metrics import r2_score

# Load methods to extract data from rosbags
from unpack_rosbag import unpack_bag, synchronize_topics, correct_time_diff
from imu_ramp_offline import ImuRampDetect

# Standard plotly imports
import plotly.express as px
import plotly.io as pio

pio.templates.default = "plotly_dark"


def get_data(bag_name, imu_topic):
    """Get data from bag"""
    bag_path = BAG_DIRECTORY + bag_name
    # IMU data (angular velocity and linear acceleration)
    imu, t_imu = unpack_bag(bag_path, IMU_TOPIC, "imu")
    # quat, _ = unpack_bag(bag_path, IMU_TOPIC, "quat")
    # Odom data (wheel speeds and handwheel angle)
    odom, t_odom = unpack_bag(bag_path, ODOM_TOPIC, "car_odom")
    # Odom data from hdl slam
    odom_hdl = unpack_bag(bag_path, ODOMHDL_TOPIC, "hdl_odom")

    # Ignore odom if none has been recorded
    if len(odom) == 0:
        f = 400
        # Align all signals in time (closest start and end time for all signals)
        data, time = correct_time_diff((imu, t_imu), odom_hdl)
        # Unpack values
        imu, odom_hdl = data
        odom = np.zeros((len(imu), 5))
        if imu_topic == "/imu/data":
            f = 100
    else:
        # Align all signals in time (closest start and end time for all signals)
        data, time = correct_time_diff((imu, t_imu), (odom, t_odom), odom_hdl)
        # Unpack values
        imu, odom, odom_hdl = data
        t_imu, t_odom = time[:2]
        # Downsamples higher freq imu signal to match odom data
        imu, odom = synchronize_topics(imu, t_imu, odom, t_odom)
        f = 100

    # Split linear acceleration and angular velocity
    ang_vel = imu[:, 0]
    lin_acc = imu[:, 1]

    t = np.linspace(0, odom.shape[0] / f, odom.shape[0])

    return lin_acc, ang_vel, odom, odom_hdl, t, f


def run_algo(lin_acc, ang_vel, odom, f, is_odom_available):
    ird = ImuRampDetect(f, is_odom_available)
    tf, gyr_bias = ird.align_imu(lin_acc, ang_vel, odom)
    angles = []
    ramp_angles = []
    for i, _ in enumerate(lin_acc):
        _, _, angs, _ = ird.spin(lin_acc[i], ang_vel[i], odom[i], tf, gyr_bias)
        angles.append(angs)
    angles_arr = np.array(angles)
    print(angles_arr.shape)
    for i in range(angles_arr.shape[1]):
        # Get angles from method and convert to degrees
        angs = np.rad2deg(angles_arr[:, i])
        ramp_angle = ramp_angle_est(angs, 4, 2, 0.2, f)
        ramp_angles.append(ramp_angle)
    return angles, ramp_angles


def visualize_results(angles, lidar_ang, t, bag_name):
    # Convert from radians to degree
    angles = np.asarray(np.rad2deg(angles))
    # Convert to pandas dataframe (for plotting)
    df_angles = pd.DataFrame(
        angles, index=t, columns=["acc", "gyr", "acc_odom", "compl", "compl_grav"]
    )
    df_angles["lidar"] = lidar_ang
    fig = px.line(df_angles)
    fig.update_layout(title=bag_name, xaxis_title="Time [s]", yaxis_title="Pitch angle [deg]")
    fig.show()


def pitch_from_lidar(odom_hdl, size):
    euler_angles = []
    for i, ori in enumerate(odom_hdl):
        q = [ori.orientation.x, ori.orientation.y, ori.orientation.z, ori.orientation.w]
        euler_angle = euler_from_quaternion(q)
        euler_angles.append(euler_angle)
    euler_angles = -np.rad2deg(np.asarray(euler_angles))
    pitch_angle = upsample(euler_angles[:, 1], size)
    return pitch_angle


def upsample(sig, size):
    # Get the flipped signal
    flip = np.flip(sig)
    # Mirror signal at the beginning at start and end
    sig_extended = np.hstack((flip, sig, flip))
    # Resample signal (*3 due to adding two flipped parts)
    sig_ext_up = resample(sig_extended, size * 3)
    # Get original signal again
    sig_up = np.split(sig_ext_up, 3)[1]
    return sig_up


# Evaluation
def RMSE(y, y_pred):
    return np.sqrt(((y_pred - y) ** 2)).mean()


def create_latex_table(scores, ramp_angles):
    """Create latex table with correct formatting"""
    # Convert to dataframe
    df = pd.DataFrame(scores)
    # Rename columns
    df.columns = ["rampType", "method", "rmse", "r2", "maxError"]
    # df["angle"] = ramp_angles
    # Group by ramp type and distance range and calculate sum of frames and avg detection rate
    df = df.groupby(["rampType", "method"], as_index=False).agg(
        {
            "rmse": "mean",
            "r2": "mean",
            "maxError": "mean",
            # "angle": "mean",
        }
    )
    print(df)
    print("\nAnd in latex format:")
    # Print them in certain order
    order = np.hstack([np.array([0, 4, 1, 2, 3]) + i * 5 for i in range(len(df) / 5)])
    # Print each row
    for i in order:
        row = df.iloc[i]
        # print("{} & {:.2f} & {:.2f}\\\\".format(row["method"], row["rmse"], row["r2"]))
        print(
            "{} & {} & {:.3f} & {:.3f} & {:.3f}".format(
                row["rampType"],
                row["method"].ljust(10),
                row["rmse"],
                row["r2"],
                row["maxError"],
            )
        )


def calc_score(angles_pred, angle_ref, ramp_type):
    scores = []
    # Convert from radians to degree
    angles_pred = np.asarray(np.rad2deg(angles_pred))
    methods = ["acc", "gyr", "acc_odom", "compl", "compl_grav"]
    df_angles = pd.DataFrame(angles_pred, columns=methods)
    if ramp_type == "d_full_long":
        # Split into two parts
        df_angles_1 = df_angles.iloc[:3006]
        angle_ref1 = angle_ref[:3006]
        df_angles_2 = df_angles.iloc[3006:6900]
        angle_ref2 = angle_ref[3006:6900]
        for i in range(df_angles_1.shape[1]):
            ang = df_angles_1.iloc[:, i]
            rmse = RMSE(angle_ref1, ang)
            r2 = r2_score(angle_ref1, ang)
            max_error = np.max(np.abs(angle_ref1 - ang))
            scores.append(("part1", methods[i], rmse, r2, max_error))
        for i in range(df_angles_2.shape[1]):
            ang = df_angles_2.iloc[:, i]
            rmse = RMSE(angle_ref2, ang)
            r2 = r2_score(angle_ref2, ang)
            max_error = np.max(np.abs(angle_ref2 - ang))
            scores.append(("part2", methods[i], rmse, r2, max_error))
    else:
        for i in range(df_angles.shape[1]):
            ang = df_angles.iloc[:, i]
            rmse = RMSE(angle_ref, ang)
            r2 = r2_score(angle_ref, ang)
            max_error = np.max(np.abs(angle_ref - ang))
            scores.append((ramp_type, methods[i], rmse, r2, max_error))
    return scores


def run_evaluation(bag_info):
    """Get scores from all bags"""
    scores = []
    ramp_angles = []
    for _, v in enumerate(bag_info):
        # if _ < 2:
        #     continue
        # Unpack dictionary
        bag_name = v["bag_name"]
        ramp_type = v["ramp_type"]
        print("\nLoading bag {}".format(bag_name))
        # Check if odometer data has been recorded
        is_odom_available = False
        if "odom" in bag_name:
            is_odom_available = True
        lin_acc, ang_vel, odom, odom_hdl, t, f = get_data(bag_name, IMU_TOPIC)
        print("Running algorithm")
        angles, ramp_angle = run_algo(lin_acc, ang_vel, odom, f, is_odom_available)
        ramp_angles.append(ramp_angle)
        # Add angle from lidar
        lidar_ang = pitch_from_lidar(odom_hdl, len(angles))
        score = calc_score(angles, lidar_ang, ramp_type)
        scores.append(score)
        # visualize_results(angles, lidar_ang, t, bag_name)
        print("\n\n")
        # if _ == 0:
        #     break
    # Flatten scores list
    scores_flat = [item for sublist in scores for item in sublist]
    ramp_angles_flat = [item for sublist in ramp_angles for item in sublist]
    return scores_flat, ramp_angles_flat


def ramp_angle_est(angle, min_ang, win, buf, f):
    """Estimate max angle of the ramp

    Args:
        angle (list): Road grade angle estimation over time [deg]
        min_ang (float): Minimum angle to be considered a peak
        win (float): Window length, in which peak will count as global [s]
        buf (float): Window length, before and after peak to use for smoothing [s]
        f (int): Frequency in Hz

    Returns:
        float: Estimated maximum angle of the ramp [deg]
    """
    # Only run until max angle has been found
    # for i in range(len(angle)):
    # Convert window length from sec to samples
    win *= f
    buf = int(buf * f)
    ang_buffer = [0]
    counter = 0

    for i, ang in enumerate(angle):
        # Iterate through all angles (real time simulation)
        if len(ang_buffer) > 2 * win:
            ang_buffer.pop(0)
        counter += 1
        # Set new max angle and reset counter
        if np.abs(ang) > np.max(np.abs(ang_buffer)):
            max_ang = np.abs(ang)
            counter = 0
        ang_buffer.append(ang)
        # Return if no new max angle for a while and angle big enough
        if counter > win and max_ang > min_ang:
            # Get some val before and after peak
            max_ang_est = np.mean(ang_buffer[-counter - buf : -counter + buf])
            # print("The ramp has an estimated peak angle of {}".format(max_ang_est))
            # print("True peak angle: {} at {}".format(max_ang, i))
            return max_ang_est


# Rosbags path
BAG_DIRECTORY = "/home/user/rosbags/final/slam/"
# ROS topics
ODOM_TOPIC = "/eGolf/sensors/odometry"
ODOMHDL_TOPIC = "/odom"
IMU_TOPIC = "/zed2i/zed_node/imu/data"
# IMU_TOPIC = "/imu/data"

# List of dictionaries with info about recording
BAG_INFO = [
    {
        "bag_name": "d_d2r2s_lidar_wo_odom_hdl.bag",
        "ramp_type": "d_full_long",
    },
    {
        "bag_name": "d_d2r2s_odom_hdl.bag",
        "ramp_type": "d_full_long",
    },
    {
        "bag_name": "d_e2q_hdl.bag",
        "ramp_type": "d_full_short",
    },
    {
        "bag_name": "straight_wo_ramps_odom_hdl.bag",
        "ramp_type": "straight",
    },
    {
        "bag_name": "u_c2s_half_odom_hdl.bag",
        "ramp_type": "u_half_short",
    },
    {
        "bag_name": "u_c2s_half_odom_stereo_hdl.bag",
        "ramp_type": "u_half_short",
    },
    {
        "bag_name": "u_c2s_hdl.bag",
        "ramp_type": "u_full_short",
    },
    {
        "bag_name": "u_c2s_stop_hdl.bag",
        "ramp_type": "u_full_short",
    },
    {
        "bag_name": "u_d2e_hdl.bag",
        "ramp_type": "u_full_short2",
    },
    {
        "bag_name": "u_s2c_half_odom_hdl.bag",
        "ramp_type": "u_half_short3",
    },
    {
        "bag_name": "u_s2c2d_hdl.bag",
        "ramp_type": "u_full_long",
    },
]

# Run evaluation
SCORES, RAMP_ANGLES = run_evaluation(BAG_INFO)
# Display results in table format
create_latex_table(SCORES, RAMP_ANGLES)
