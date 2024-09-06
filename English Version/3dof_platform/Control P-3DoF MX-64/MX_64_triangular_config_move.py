# Program to control the position of the p456 platform based on the height, roll, and pitch of the platform

import numpy as np
import time
import pypot.dynamixel

# Connection to the Dynamixels
ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('No ports detected!')

print('Detected ports:', ports)

dxl_io = pypot.dynamixel.DxlIO(ports[0], baudrate=1000000)
dxl_ids = [4, 5, 6]  # Specify the 3 IDs of the Dynamixels to minimize the ID search time

print('Attempting to connect to Dynamixels with IDs:', dxl_ids)

# Check if the Dynamixels are connected
connected_ids = dxl_io.scan(dxl_ids)
print('Connected Dynamixel IDs:', connected_ids)

# Verify that all Dynamixels are detected with the specified IDs
if not all(motor_id in connected_ids for motor_id in dxl_ids):
    raise IOError('Not all Dynamixels are detected!')

# Enable torque for the platform's Dynamixels
dxl_io.enable_torque(dxl_ids)

# Constant parameters
R = 140.116 / 2  # Base radius in mm
r = 140.116 / 2  # Mobile platform radius in mm

# Bi is the point at the center of the i-th joint between the effector and the base
# Pi is the point at the center of the i-th joint between the effector and the platform
B1 = np.array([-R * np.cos(np.radians(0)), R * np.sin(np.radians(0)), 0])
B2 = np.array([-R * np.cos(np.radians(120)), R * np.sin(np.radians(120)), 0])
B3 = np.array([-R * np.cos(np.radians(240)), R * np.sin(np.radians(240)), 0])
P1 = np.array([-r * np.cos(np.radians(0)), r * np.sin(np.radians(0)), 0])
P2 = np.array([-r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0])
P3 = np.array([-r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0])

# Lists of the positions of the arm attachment points to the Dynamixels and the centers of the universal joint pivots
B = [B1, B2, B3]
P = [P1, P2, P3]

d = 50.014  # Arm length (distance between the pivot centers of the arm) in mm
e = 81.867  # Forearm length (distance from the pivot center of the arm to the universal joint cross) in mm

phi0 = 0  
zmax = 131.881  # Maximum height (Dynamixel at 0°)
zmin = 31.881  # Minimum height (Dynamixel at 180°)

# Position limits for the MX_64 for this configuration
xi_min = 0 
xi_max = 120

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

# Function to calculate the vectors corresponding to the lengths BiPi in the desired position
def VecteurLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Function to calculate the absolute values of the lengths BiPi from the VecteurLi vectors
def ValeurLi(VecteurLi):
    return np.linalg.norm(VecteurLi)

# Function to calculate the angular position of the servomotor based on the Valeur Li 
def AngleLi_MX_64(Li):
    Edeg = ((Li**2) + (d**2) - (e**2)) / (2 * d * Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = abs(-np.arccos(Edeg) + np.radians(phi0))  # Removed the 90° compared to the Microgoat team calculation as our O is vertical and an absolute value as MX-64s are controlled from 0° to 180°
    xi = np.clip(np.degrees(xi), 0, 180)
    return round(xi)

# Function to calculate the inverse kinematics from the previous functions and move the platform
def Move(roll, pitch, height):
    # Clamp the input parameter values
    roll = np.clip(roll, -45, 45)
    pitch = np.clip(pitch, -45, 45)
    height = np.clip(height, zmin, zmax)
    
    T = np.array([0, 0, height])  # Vector corresponding to the distance between the center of the platform and the base
    
    theta = np.radians(roll)  # Roll (rotation around the X axis)
    psi = np.radians(pitch)  # Pitch (rotation around the Y axis)
    
    # Calculate the transformation matrix from the rotation matrices
    Rot_BtoP = rotationY(psi) @ rotationX(theta)
   
    # Lists to store variable values for each arm to perform a for loop
    VecteurL = []
    L = []
    xi = []

    # For loop to calculate inverse kinematics for each arm 
    for i in range(3):
        # Calculate the vector between the points Pi and Bi
        VecteurL.append(VecteurLi(Rot_BtoP, P[i], B[i], T))
         
        # Calculate the norm of the vector corresponding to the distance BiPi
        L.append(ValeurLi(VecteurL[i]))
        
        # Display the norm of the vector corresponding to the distance BiPi
        print(f'L{i+1}:', L[i])

        # Calculate the angular position to send to Dynamixel i
        xi.append(AngleLi_MX_64(L[i]))

        # Limit the extreme position of Dynamixel i
        if xi[i] < xi_min:
            xi[i] = xi_min
        elif xi[i] > xi_max:
            xi[i] = xi_max
    
        # Display the angular position of Dynamixel i
        print(f'xi{i+1}:', xi[i])
    
    # Correspondence             xi4     xi5       xi6
    dxl_io.set_goal_position({4: xi[0], 5: xi[1], 6: xi[2]})
    
# Parameters governing the movement of the platform
roll = 10
pitch = 10
height = 90
    
# Perform a trajectory following a circle with the platform tilted
for i in range(3 * 360):
    # Example of calling the function
    Move(roll * np.sin(np.radians(i)), pitch * np.cos(np.radians(i)), height)
    time.sleep(0.0025)
    
time.sleep(1)

# Disable torque for the Dynamixels
dxl_io.disable_torque(dxl_ids)
