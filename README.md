# Multi Sensor Ramp Detection and Localization for Autonomous Valet Parking

Master thesis at TU Berlin in collaboration with Expleo Germany GmbH

[PDF (work in progress)](https://github.com/FSaal/masterThesis/blob/main/Thesis/Main_Thesis/main.pdf)

## Problem:
Autonomous Valet Parking (AVP) will make parking easier in the future, by allowing the driver to exit the car in a drop off zone in front of a parking garage, and the car will find a parking spot on its own. When the driver calls the car again, it will also autonomously pick up the driver again. For this to work, a map of the parking garage and a precise localization of the car is necessary.
A challenging part is the change of levels, because the ramps in parking garages are usually very narrow and require a very precise localization and control of the car.  Therefore an information about whether or not the car is driving onto a ramp is necessary. This allows the controller of the car to adjust for the changing road conditions, e.g. increasing or decreasing the motor output power when driving up or down respectively. Also because the maps used for the localization of the car are usually stored separately for each parking level, the loading of the new map can be initiated when the car is on a ramp.

## Task:
The goal of this thesis is the implementation of an algorithm for a car, which can detect ramps. Besides the detection, ramp properties such as the inclination angle or length should be measured. To implement this, various sensor setups will be used and compared. An Inertial Measurement Unit (IMU) will be the main sensor and will be responsible for the exact measurement of the ramp properties, in conjunction with a wheel odometer. Additionally a LiDAR sensor will be used to allow for the detection of the ramp, before entering it.  The data of the LiDAR could also be fused with the IMU data to prevent false detections from the IMU. A camera will be tested as well for the early detection of a ramp, and compared to the LiDAR.
Test drives in one specific parking garage and test car will be performed. A camera will be used as ground truth to determine if the detection was at the right time and the estimated ramp properties will be compared to manual measurements.

## Research Steps:
- Research of current methods to determine road grade angle using IMU, LiDAR or camera
- Comparison and selection of the most appropriate method for each sensor
- Implementation of a ramp detection algorithm
    - using an IMU
    - using a LiDAR sensor
-    using a camera
- Testing and optimizing of the methods
- Comparison and evaluation of the different methods used
- Documentation and presentation of the results and thesis
