# Program to control the position of platform p123 based on the height, roll, and pitch of the platform

import numpy as np
import time
import pypot.dynamixel

# Connecting to Dynamixels
ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('No ports detected!')

print('Detected ports: ', ports)

dxl_io = pypot.dynamixel.DxlIO(ports[0], baudrate=1000000)
dxl_ids = [1, 2, 3]  # Specification of the 3 Dynamixel IDs to minimize ID search time

print('Attempting to connect to Dynamixel IDs:', dxl_ids)

# Checking that the Dynamixels are connected
connected_ids = dxl_io.scan(dxl_ids)
print('Connected Dynamixel IDs: ', connected_ids)

# Verifying that all Dynamixels are detected at the specified IDs
if not all(motor_id in connected_ids for motor_id in dxl_ids):
    raise IOError('Not all Dynamixels are detected!')

# Enabling torque for the Dynamixels on the platform
dxl_io.enable_torque(dxl_ids)

# Constant parameters:

R = 140.116 / 2  # Radius of the base in mm
r = 140.116 / 2  # Radius of the moving platform in mm

# Bi is the point at the center of the i-th joint between the effector and the base
# Pi is the point at the center of the i-th joint between the effector and the platform
B1 = np.array([-R * np.cos(np.radians(0)), R * np.sin(np.radians(0)), 0])
B2 = np.array([-R * np.cos(np.radians(120)), R * np.sin(np.radians(120)), 0])
B3 = np.array([-R * np.cos(np.radians(240)), R * np.sin(np.radians(240)), 0])
P1 = np.array([-r * np.cos(np.radians(0)), r * np.sin(np.radians(0)), 0])
P2 = np.array([-r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0])
P3 = np.array([-r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0])

# Lists of the positions of the arm attachment points to the Dynamixels and the centers of the gimbal joints
B = [B1, B2, B3]
P = [P1, P2, P3]

d = 50.014  # Arm length (distance between the centers of the arm pivots) in mm
e = 81.867  # Forearm length (distance from the arm pivot to the gimbal cross) in mm

phi0 = 0
zmax = 131.881  # Height at high position (Dynamixel at 0°)
zmin = 31.881   # Height at low position (Dynamixel at 180°)

# Limit positions for MX_64 for this configuration
xi_min = 0
xi_max = 180

# Function to generate a rotation matrix around the X axis (roll)
def rotationX(theta):  # Euler angle theta related to roll = alpha in the thesis
    RotX = np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return RotX

# Function to generate a rotation matrix around the Y axis (pitch)
def rotationY(psi):  # Euler angle psi related to pitch = beta in the thesis
    RotY = np.array([[np.cos(psi), 0, np.sin(psi)],
                     [0, 1, 0],
                     [-np.sin(psi), 0, np.cos(psi)]])
    return RotY

# Function to calculate vectors corresponding to lengths BiPi in the desired position
def VecteurLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Function to calculate the absolute lengths BiPi from the VecteurLi vectors
def ValeurLi(VecteurLi):
    return np.linalg.norm(VecteurLi)

# Function to calculate the angular position of the Dynamixels based on Valeur Li
def AngleLi_MX_64(Li):
    Edeg = ((Li**2) + (d**2) - (e**2)) / (2 * d * Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = abs(-np.arccos(Edeg) + np.radians(phi0))  # Removed the 90° from the Microgoat team's calculation as our O is vertical and taking the absolute value because MX-64 is controlled in the 0° to 180° range
    xi = np.clip(np.degrees(xi), 0, 180)
    return round(xi)

# Function to calculate the inverse kinematics and move the platform based on the previous functions
def Move(roll, pitch, height):
    
    # Clamping the input parameters
    roll = np.clip(roll, -45, 45)
    pitch = np.clip(pitch, -45, 45)
    height = np.clip(height, zmin, zmax)
    
    T = np.array([0, 0, height])  # Vector corresponding to the distance between the center of the platform and the base
    
    theta = np.radians(roll)  # Roll (rotation around the X axis)
    psi = np.radians(pitch)  # Pitch (rotation around the Y axis)
    
    # Calculating the transformation matrix from the rotation matrices
    Rot_BtoP = rotationY(psi) @ rotationX(theta)
   
    # Lists to store the values for each arm and thus perform a loop
    VecteurL = []
    L = []
    xi = []

    # Loop to perform inverse kinematics calculation for each arm 
    for i in range(3):
    
        # Calculating the vector between points Pi and Bi
        VecteurL.append(VecteurLi(Rot_BtoP, P[i], B[i], T))
         
        # Calculating the norm of the vector corresponding to the distance BiPi
        L.append(ValeurLi(VecteurL[i]))
        
        # Displaying the norm of the vector corresponding to the distance BiPi
        print(f'L{i+1} :', L[i])

        # Calculating the angular position to send to Dynamixel i
        xi.append(AngleLi_MX_64(L[i]))

        # Limiting the extreme position of Dynamixel i
        if xi[i] < xi_min:
            xi[i] = xi_min
        elif xi[i] > xi_max:
            xi[i] = xi_max
    
        # Displaying the angular position of Dynamixel i
        print(f'xi{i+1} : ', xi[i])
   
    # Correspondence             x1       x2       x3
    dxl_io.set_goal_position({1: xi[0], 2: xi[1], 3: xi[2]})
    
# Parameters governing the movement of the platform
roll = 10
pitch = 10
height = 90
    
# Performing a circular trajectory with the inclined platform
# for i in range(360):
#     # Example function call
#     Move(roll * np.sin(np.radians(i)), pitch * np.cos(np.radians(i)), height)
#     time.sleep(0.0025)

# Performing an ascent and then a descent following a circle with the inclined platform
# for t in range(4):
#     for i in range(360):
#         # Example function call
#         Move(roll * np.sin(np.radians(i)), pitch * np.cos(np.radians(i)), (height - 50) + 50 * i / 360)
#         time.sleep(0.0025)
#     for i in range(360):
#         # Example function call
#         Move(roll * np.sin(np.radians(360 - i)), pitch * np.cos(np.radians(360 - i)), (height - 50) + 50 * (360 - i) / 360)
#         time.sleep(0.0025)   

# Rolling wave
# for t in range(4):
#     for i in range(360):
#         # Example function call
#         Move(roll * 0, pitch * np.cos(np.radians(i / 2)), (height - 50) + 50 * i / 360)
#         time.sleep(0.0025)
#     for i in range(360):
#         # Example function call
#         Move(roll * 0, pitch * np.cos(np.radians(180 + i / 2)), (height - 50) + 50 * (360 - i) / 360)
#         time.sleep(0.0025)

# Bounce
# for i in range(3):
#     simulate(0, -5, height)
#     time.sleep(0.250)
#     simulate(0, 20, height)
#     time.sleep(0.250)
#     simulate(0, -5, height)

# Catapult
# simulate(0, 40, height)
# time.sleep(0.50)
# simulate(0, -5, height)

# 3DoF demonstration with limit positions
Move(0, 0, 31.8)
time.sleep(2)
Move(0, 0, 131.8)
time.sleep(2)
Move(0, 0, 80)
time.sleep(2)
Move(30, 0, 80)
time.sleep(2)
Move(-30, 0, 80)
time.sleep(2)
Move(0, 30, 80)
time.sleep(2)
Move(0, -30, 80)
time.sleep(2)
Move(0, 0, 80) 

for i in range(3 * 360):
    # Example function call
    Move(10 * np.sin(np.radians(i)), 10 * np.cos(np.radians(i)), 80)
    time.sleep(0.0025)

Move(0, 0, 80)

time.sleep(1)

# Disabling the torque of the detected motors
dxl_io.disable_torque(dxl_ids)

