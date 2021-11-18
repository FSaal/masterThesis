import numpy as np

def gyr_no_drift(ang_vel, ang_vel_still=vel_rot[:500, 1]):
    """
    Calculate drift free angle over time from angular velocity
    :param ang_vel:       Angular velocity from IMU
    :return ang_no_drift: Drift free angle over time
    """
    ang_still = np.degrees(np.cumsum(ang_vel_still / f_imu))
    drift = np.linspace(ang_still[0], ang_still[-1] / (len(ang_vel_still)/len(ang_vel)), len(ang_vel))
    ang = np.degrees(np.cumsum(ang_vel) / f_imu)
    ang_no_drift = ang - drift
    return ang_no_drift

vel_rot = np.random