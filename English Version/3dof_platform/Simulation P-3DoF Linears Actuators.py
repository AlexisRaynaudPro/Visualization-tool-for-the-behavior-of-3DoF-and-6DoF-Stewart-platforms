import os
import numpy as np
import matplotlib.pyplot as plt

# Constant parameters:
R = 155e-3  # Radius of the base in meters
r = 80e-3   # Radius of the mobile platform in meters

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

cv = 415e-3  # Actuator body length
z0 = 665e-3  # Height of the midpoint position
zmax = 915e-3  # Height in the upper position
zmin = 415e-3  # Height in the lower position
roll_min = -45
roll_max = 45
pitch_min = -45
pitch_max = 45

# Offset for annotations on the plot
B_offset = 10e-3
P_offset = 30e-3
L_offset = 5e-3

# Function to generate a rotation matrix around the X-axis (roll)
def rotationX(theta):  # Euler angle theta related to roll = alpha in the thesis
    RotX = np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return RotX

# Function to generate a rotation matrix around the Y-axis (pitch)
def rotationY(psi): # Euler angle psi related to pitch = beta in the thesis
    RotY = np.array([[np.cos(psi), 0, np.sin(psi)],
                     [0, 1, 0],
                     [-np.sin(psi), 0, np.cos(psi)]])
    return RotY

# Function to calculate vectors corresponding to lengths BiPi in the desired position
def VectorLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Function to calculate the absolute values of lengths BiPi from the VectorLi vectors
def LengthLi(VectorLi):
    return np.linalg.norm(VectorLi)

# Function to plot a circle in 3D coordinates
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

    # Limit parameter values
    roll = np.clip(roll, -30, 30)  # Limit roll between -30 and 30 degrees
    pitch = np.clip(pitch, -30, 30)  # Limit pitch between -30 and 30 degrees
    height = np.clip(height, 0.236, 0.836)  # Limit height between 0.236 and 0.836 meters

    T = np.array([0, 0, height])

    theta = np.radians(roll)  # Roll (rotation around the X-axis)
    psi = np.radians(pitch)  # Pitch (rotation around the Y-axis)

    # Calculate the rotation matrix from the rotation matrices
    Rot_BtoP = rotationY(psi) @ rotationX(theta)
    
    # Lists to store variable values for each Arm and necessary for the following loop
    VectorL = []
    P = []
    L = []
    S = []

    for i in range(3):
             
        # Calculate the vector between points Pi and Bi
        VLi = VectorLi(Rot_BtoP, P0[i], B[i], T)
        VectorL.append(VLi)
        
        # Calculate the new position of point Pi
        P.append(B[i] + VLi)
         
        # Calculate the norm of the vector corresponding to the distance BiPi
        # which is the sum of the actuator body length and its stroke
        L.append(LengthLi(VectorL[i]))
        
        # Display the norm of the vector corresponding to the distance BiPi
        print(f'L{i+1} :', L[i])
        
        # Calculate the stroke of actuator Vi
        S.append(L[i] - cv)
            
        # Display the stroke of actuator Vi
        print(f'Stroke S{i+1} :', S[i])
        
    # Convert the list P to numpy array (vector)
    P = np.array(P)
    
    # Generate the legend block 
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='#386480', s=100, label='Mobile Base')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#72bdba', s=100, label='Linears Actuators Bodies')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#ffd783', s=100, label='Linears Actuators Strokes')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#f2887c', s=100, label='3DoF Platform')

       # Connection lines and annotations
    for i in range(3):
        
        # Plot the segments BiPi
        ax.plot([B[i, 0], P[i, 0]], 
                [B[i, 1], P[i, 1]], 
                [B[i, 2], P[i, 2]], '#5f7fbf', lw=3, linestyle='--')
        
        # Plot the actuator bodies
        ax.plot([B[i, 0], B[i, 0] + cv / L[i] * (P[i, 0] - B[i, 0])], 
                [B[i, 1], B[i, 1] + cv / L[i] * (P[i, 1] - B[i, 1])], 
                [B[i, 2], B[i, 2] + cv / L[i] * (P[i, 2] - B[i, 2])], '#72bdba', lw=4)
        
        # Plot the actuator strokes
        ax.plot([B[i, 0] + cv / L[i] * (P[i, 0] - B[i, 0]), P[i, 0]], 
                [B[i, 1] + cv / L[i] * (P[i, 1] - B[i, 1]), P[i, 1]], 
                [B[i, 2] + cv / L[i] * (P[i, 2] - B[i, 2]), P[i, 2]], '#ffd783', lw=4)
           
        # Display the names of the points and segments with an offset to make them readable
        ax.text(B[i, 0] + B_offset, B[i, 1] + B_offset, B[i, 2] + B_offset, f'B{i+1}', color='#386480', fontsize=12, weight='bold')
        ax.text(P[i, 0], P[i, 1], P[i, 2] + P_offset, f'P{i+1}', color='#f2887c', fontsize=12, weight='bold')
        ax.text(B[i, 0] + cv / L[i] * (P[i, 0] - B[i, 0]) + L_offset, 
                B[i, 1] + cv / L[i] * (P[i, 1] - B[i, 1]) + L_offset, 
                B[i, 2] + cv / L[i] * (P[i, 2] - B[i, 2]) + L_offset, 
                f'L{i+1}', color='#5f7fbf', fontsize=12, fontweight='bold')

    # Plot circles representing the base and the platform
    plot_circle(ax, np.array([0, 0, 0]), R, np.array([0, 0, 1]), color='#386480', linestyle='--')
    plot_circle(ax, T, r, Rot_BtoP @ np.array([0, 0, 1]), color='#f2887c', linestyle='--')
    
    # Visual improvements
    ax.set_box_aspect([1, 1, 1])  # Aspect ratio 1:1:1

    # Set the dimensions of the viewing space
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([0, 1])

    # Define axis labels with inclination and offset from the axes
    ax.set_xlabel('X Axis (m)', fontsize=15, labelpad=18, weight='book')
    ax.set_ylabel('Y Axis (m)', fontsize=15, labelpad=18, weight='book')
    ax.set_zlabel('Z Axis (m)', fontsize=15, labelpad=18, weight='book')
    
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Add system status information in a box
    info_text = (f'\n'
                 f' 3DoF Platform Simulation: \n'
                 f'\n'
                 f' Roll ($\\theta$): {roll:.2f}°\n'
                 f'\n'
                 f' Pitch ($\\psi$): {pitch:.2f}°\n'
                 f'\n'
                 f' Height: {height:.4f} m\n'
                 f'\n'
                 f' Length L1: {L[0]:.4f} m\n'
                 f'\n'
                 f' Length L2: {L[1]:.4f} m\n'
                 f'\n'
                 f' Length L3: {L[2]:.4f} m\n'
                 f'\n'
                 f' Stroke S1: {S[0]:.4f} m\n'
                 f'\n'
                 f' Stroke S2: {S[1]:.4f} m\n'
                 f'\n'
                 f' Stroke S3: {S[2]:.4f} m'
                 f'\n')
    props = dict(boxstyle='round', facecolor='white', alpha=0.15)
    ax.text2D(-0.26, 0.475, info_text, transform=ax.transAxes, fontsize=15, weight='book', fontstyle='italic', verticalalignment='center', bbox=props)   
    ax.legend(loc=(-0.265, 0.09), prop={'size': 15, 'weight': 'book', 'style': 'italic'})   

    # Create a directory, if it does not exist, at the same location as the program file  
    simulation_result = 'P-3DoF Linears Actuators Simulation Result'
    if not os.path.exists(simulation_result):
        os.makedirs(simulation_result)
    
    # Path for saving the PNG
    png_path = os.path.join(simulation_result, 'P-3DoF Linears Actuators.png')
    plt.savefig(png_path)    
    plt.show()

# Example of calling the function
simulate(0, 0, 600e-3)
