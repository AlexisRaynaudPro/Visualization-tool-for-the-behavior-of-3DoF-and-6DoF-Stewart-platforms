import os
import numpy as np
import matplotlib.pyplot as plt

# Constant parameters:
R = 155e-3  # Radius of the base in meters
r = 80e-3   # Radius of the moving platform in meters

# Bi is the point at the center of the i-th joint between the effector and the base
# Pi is the point at the center of the i-th joint between the effector and the platform
B1 = np.array([-R * np.cos(np.radians(0)), R * np.sin(np.radians(0)), 0])
B2 = np.array([-R * np.cos(np.radians(120)), R * np.sin(np.radians(120)), 0])
B3 = np.array([-R * np.cos(np.radians(240)), R * np.sin(np.radians(240)), 0])
P1 = np.array([-r * np.cos(np.radians(0)), r * np.sin(np.radians(0)), 0])
P2 = np.array([-r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0])
P3 = np.array([-r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0])

B = np.array([B1, B2, B3])
P0 = np.array([P1, P2, P3])

d = 250e-3  # Arm length (distance between the centers of the arm pivots) in meters
e = 415e-3  # Forearm length (distance from the arm pivot to the gimbal cross) in meters

phi0 = 0  # Initial angular position of the servos
z0 = 331e-3  # Height when the arms are at 90°
zmin = 250e-3  # Height in the low position (servo at 180°)
zmax = 665e-3  # Height in the high position (servo at 0°)
roll_min = -45
roll_max = 45
pitch_min = -45
pitch_max = 45

# Limit positions for MX_64 for this configuration
xi_min = 0
xi_max = 180

# Offset for annotations on the graph
B_offset = 2e-3
P_offset = 2e-3
L_offset = 2e-3
A_offset = 2e-3

# Function generating a rotation matrix around the X axis (roll)
def rotationX(theta):  # Euler angle theta related to roll = alpha in the thesis
    RotX = np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return RotX

# Function generating a rotation matrix around the Y axis (pitch)
def rotationY(psi):  # Euler angle psi related to pitch = beta in the thesis
    RotY = np.array([[np.cos(psi), 0, np.sin(psi)],
                     [0, 1, 0],
                     [-np.sin(psi), 0, np.cos(psi)]])
    return RotY

# Function calculating the vectors corresponding to lengths BiPi in the desired position
def VecteurLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Function calculating the absolute values of the lengths BiPi from the VecteurLi vectors
def ValeurLi(VecteurLi):
    return np.linalg.norm(VecteurLi)

# Function calculating the angle to send to the motor if not controlled in position based on Valeur Li
# This angle corresponds to the angle between the arm (yellow) and the forearm (turquoise)
def AngleLi(Li):
    Edeg = ((Li**2) + (d**2) - (e**2)) / (2 * d * Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = -np.arccos(Edeg) + np.radians(90.0) + np.radians(phi0)
    return round(np.degrees(xi))

# Function calculating the angular position of the servos based on Valeur Li
def AngleLi_MX_64(Li):
    Edeg = ((Li**2) + (d**2) - (e**2)) / (2 * d * Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = abs(-np.arccos(Edeg) + np.radians(phi0))  # Removed the 90° from the Microgoat team's calculation because our O is vertical and taking the absolute value as MX-64 is controlled in the 0° to 180° range
    xi = np.clip(np.degrees(xi), 0, 180)
    return round(xi)

def Intermediate_Point(xi):       
    # Calculate the positions
    x_int = d * np.cos(np.radians(xi))
    z_int = d * np.sin(np.radians(xi))
    return x_int, z_int

# Function to plot a circle in a 3D coordinate system
def plot_circle(ax, center, radius, normal, color='k', linestyle='--'):
    alpha = np.linspace(0, 2 * np.pi, 100)
    normal = normal / np.linalg.norm(normal)
    v = np.array([1, 0, 0], dtype=float) if abs(normal[0]) < abs(normal[1]) else np.array([0, 1, 0], dtype=float)
    v = v - v.dot(normal) * normal
    v = v.astype(float) / np.linalg.norm(v)
    w = np.cross(normal, v)
    
    circle = center[:, None] + radius * (v[:, None] * np.cos(alpha) + w[:, None] * np.sin(alpha))
    ax.plot(circle[0, :], circle[1, :], circle[2, :], color=color, linestyle=linestyle)
    
# Simulation function
def simulate(roll, pitch, height):
    
    # Initialize the figure
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    roll = np.clip(roll, roll_min, roll_max)
    pitch = np.clip(pitch, pitch_min, pitch_max)
    height = np.clip(height, zmin, zmax)
    
    T = np.array([0, 0, height])
    
    theta = np.radians(roll)  # Roll (rotation around the X axis)
    psi = np.radians(pitch)  # Pitch (rotation around the Y axis)
    
    # Calculate the transformation matrix from the rotation matrices
    Rot_BtoP = rotationY(psi) @ rotationX(theta)
       
    # Lists to store the values for each arm and needed for the next for loop
    VecteurL = []
    P = []
    L = []
    xi = []
    xz = [] 
    A = []

    for i in range(3):
             
        # Calculate the vector between points Pi and Bi
        VLi = VecteurLi(Rot_BtoP, P0[i], B[i], T)
        VecteurL.append(VLi)
        
        # Calculate the new position of point Pi
        P.append(B[i] + VLi)
         
        # Calculate the norm of the vector corresponding to the distance BiPi
        L.append(ValeurLi(VecteurL[i]))
        
        # Print the norm of the vector corresponding to the distance BiPi
        print(f'L{i+1} :', L[i])
        
        # Calculate the angular position to return to servo i
        xi.append(AngleLi_MX_64(L[i]))

        # Limit the extreme position of servo i
        if xi[i] < xi_min:
            xi[i] = xi_min
        elif xi[i] > xi_max:
            xi[i] = xi_max
    
        # Print the angular position of servo i
        print(f'xi{i+1} : ', xi[i])
        
        xz.append(Intermediate_Point(AngleLi(L[i])))

        # Position of point Ai corresponding to the joint between the arm and the forearm
        
        # Standard configuration (arm pivots contained within the 3DoF platform volume)
        ai = np.array([
            (-R + xz[i][0]) * np.cos(np.radians(120 * i)),
            (R - xz[i][0]) * np.sin(np.radians(120 * i)),
            xz[i][1]
        ])
        
        # 'Triangular' configuration where arm axes are tangent to the base, functional configuration only if r=R
        # ai = np.array([
        #     -R * np.cos(np.radians(120 * i)) + xz[i][0] * np.cos(np.radians(120 * i + 90)),
        #     R * np.sin(np.radians(120 * i)) - xz[i][0] * np.sin(np.radians(120 * i + 90)),
        #     xz[i][1]
        # ])
        
        # List of positions of points corresponding to the pivot i between the arm and the forearm
        A.append(ai)


    # Convert lists A and P to numpy arrays (vectors)
    A = np.array(A)
    P = np.array(P)
    
    # Generate the legend block
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='#386480', s=100, label='Mobile Base')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='#ffd783', s=100, label='Arm')
    ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='#72bdba', s=100, label='Forearm')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#f2887c', s=100, label='3DoF Platform')
    
    # Connection lines and annotations
    for i in range(3):
            
        # Plot segments BiPi
        ax.plot([B[i, 0], P[i, 0]], 
                [B[i, 1], P[i, 1]], 
                [B[i, 2], P[i, 2]], '#5f7fbf', lw=3, linestyle='--')
        
        # Plot segments BiAi representing arm i
        ax.plot([B[i, 0], A[i, 0]], 
                [B[i, 1], A[i, 1]], 
                [B[i, 2], A[i, 2]], '#ffd783', lw=3)
        
        # Plot segments BiAi representing forearm i
        ax.plot([A[i, 0], P[i, 0]], 
                [A[i, 1], P[i, 1]], 
                [A[i, 2], P[i, 2]], '#72bdba', lw=3)
        
        # Display point names and segments with an offset for readability
        ax.text(B[i, 0] + B_offset, B[i, 1] + B_offset, B[i, 2] + B_offset, f'B{i+1}', color='#386480', fontsize=12, weight='bold')
        ax.text(P[i, 0] + P_offset, P[i, 1] + P_offset, P[i, 2] + P_offset, f'P{i+1}', color='#f2887c', fontsize=12, weight='bold')
        ax.text(A[i, 0] + A_offset, A[i, 1] + A_offset, A[i, 2] + A_offset, f'A{i+1}', color='#72bdba', fontsize=12, weight='bold')

        ax.text(B[i, 0] + L_offset, B[i, 1] + L_offset, z0 / 2 + L_offset, f'L{i+1}', color='#5f7fbf', fontsize=12, fontweight='bold')

    # Plot circles representing the base and platform
    plot_circle(ax, np.array([0, 0, 0]), R, np.array([0, 0, 1]), color='#386480', linestyle='--')
    plot_circle(ax, T, r, Rot_BtoP @ np.array([0, 0, 1]), color='#f2887c', linestyle='--')
    
    # Visual enhancements
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio 1:1:1

    # Fix the viewing space dimensions
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0, 0.9])

    # Define axis labels with tilt and offset from the axes
    ax.set_xlabel('X Axis (mm)', fontsize=15, labelpad=18, weight='book')
    ax.set_ylabel('Y Axis (mm)', fontsize=15, labelpad=18, weight='book')
    ax.set_zlabel('Z Axis (mm)', fontsize=15, labelpad=18, weight='book')
    
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add system state information in a box
    info_text = (f'\n'
                 f' 3DoF Platform Simulation:  \n'
                 f'\n'
                 f' Roll ($\\theta$) : {roll:.2f}°\n'
                 f'\n'
                 f' Pitch ($\\psi$) : {pitch:.2f}°\n'
                 f'\n'
                 f' Height : {height:.4f} m\n'
                 f'\n'
                 f' Length L1 : {L[0]:.4f} m\n'
                 f'\n'
                 f' Length L2 : {L[1]:.4f} m\n'
                 f'\n'
                 f' Length L3 : {L[2]:.4f} m\n'
                 f'\n'
                 f' Angle xi1 : {xi[0]:.2f} °\n'
                 f'\n'
                 f' Angle xi2 : {xi[1]:.2f} °\n'
                 f'\n'
                 f' Angle xi3 : {xi[2]:.2f} °'
                 f'\n')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.15)
    ax.text2D(-0.26, 0.475, info_text, transform=ax.transAxes, fontsize=15, weight='book', fontstyle='italic', verticalalignment='center', bbox=props)   
    ax.legend(loc=(-0.265, 0.09), prop={'size': 15, 'weight': 'book', 'style': 'italic'}) # loc=(-0.265, 0.22) before adding the padding in the legend
    
    # Generate a folder, if it doesn't exist, at the same location as the program file  
    simulation_result = 'P-3DoF Servos Simulations Results'
    if not os.path.exists(simulation_result):
        os.makedirs(simulation_result)
    
    # Access path to save the GIF
    png_path = os.path.join(simulation_result, 'P-3DoF Servos.png')
    plt.savefig(png_path)    
    plt.show()
    
# Simulation parameters
roll = 0
pitch = 0
height = 59
    
# Example function call
simulate(roll, pitch, height)
