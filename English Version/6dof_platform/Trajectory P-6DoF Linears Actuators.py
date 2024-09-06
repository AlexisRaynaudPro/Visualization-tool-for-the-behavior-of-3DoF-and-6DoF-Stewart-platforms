import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constant parameters:
R = 155e-3  # Base radius in meters
r = 80e-3   # Mobile platform radius in meters

# Bi is the point at the center of the i-th joint between the effector and the base
# Pi is the point at the center of the i-th joint between the effector and the platform

# 6-6 configuration
# B1 = np.array([-R * np.cos(np.radians(0)), R * np.sin(np.radians(0)), 0])
# B2 = np.array([-R * np.cos(np.radians(60)), R * np.sin(np.radians(60)), 0])
# B3 = np.array([-R * np.cos(np.radians(120)), R * np.sin(np.radians(120)), 0])
# B4 = np.array([-R * np.cos(np.radians(180)), R * np.sin(np.radians(180)), 0])
# B5 = np.array([-R * np.cos(np.radians(240)), R * np.sin(np.radians(240)), 0])
# B6 = np.array([-R * np.cos(np.radians(300)), R * np.sin(np.radians(300)), 0])
# P1 = np.array([-r * np.cos(np.radians(0)), r * np.sin(np.radians(0)), 0])
# P2 = np.array([-r * np.cos(np.radians(60)), r * np.sin(np.radians(60)), 0])
# P3 = np.array([-r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0])
# P4 = np.array([-r * np.cos(np.radians(180)), r * np.sin(np.radians(180)), 0])
# P5 = np.array([-r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0])
# P6 = np.array([-r * np.cos(np.radians(300)), r * np.sin(np.radians(300)), 0])

# 3-3 configuration
omega = 10
sigma = 50

B1 = np.array([-R * np.cos(np.radians(0-omega)), R * np.sin(np.radians(0-omega)), 0])
B2 = np.array([-R * np.cos(np.radians(0+omega)), R * np.sin(np.radians(0+omega)), 0])
B3 = np.array([-R * np.cos(np.radians(120-omega)), R * np.sin(np.radians(120-omega)), 0])
B4 = np.array([-R * np.cos(np.radians(120+omega)), R * np.sin(np.radians(120+omega)), 0])
B5 = np.array([-R * np.cos(np.radians(240-omega)), R * np.sin(np.radians(240-omega)), 0])
B6 = np.array([-R * np.cos(np.radians(240+omega)), R * np.sin(np.radians(240+omega)), 0])
P1 = np.array([-r * np.cos(np.radians(0-sigma)), r * np.sin(np.radians(0-sigma)), 0])
P2 = np.array([-r * np.cos(np.radians(0+sigma)), r * np.sin(np.radians(0+sigma)), 0])
P3 = np.array([-r * np.cos(np.radians(120-sigma)), r * np.sin(np.radians(120-sigma)), 0])
P4 = np.array([-r * np.cos(np.radians(120+sigma)), r * np.sin(np.radians(120+sigma)), 0])
P5 = np.array([-r * np.cos(np.radians(240-sigma)), r * np.sin(np.radians(240-sigma)), 0])
P6 = np.array([-r * np.cos(np.radians(240+sigma)), r * np.sin(np.radians(240+sigma)), 0])


B = np.array([B1, B2, B3, B4, B5, B6]) 
P0 = np.array([P1, P2, P3, P4, P5, P6]) 

lab = 415e-3  # Linear actuator body lenght
z0 = 665e-3  # Height at the middle position
zmax = 915e-3  # Height at the upper position
zmin = 415e-3  # Height at the lower position
roll_min = -45
roll_max = 45
pitch_min = -45
pitch_max = 45

# Offset for annotations present on the graph
B_offset = 10e-3
P_offset = 30e-3
L_offset = 5e-3

# Function generating a rotation matrix around the X-axis (roll)
def rotationX(theta):  # Euler angle theta related to roll = alpha in the thesis
    RotX = np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return RotX

# Function generating a rotation matrix around the Y-axis (pitch)
def rotationY(psi):  # Euler angle psi related to pitch = beta in the thesis
    RotY = np.array([[np.cos(psi), 0, np.sin(psi)],
                     [0, 1, 0],
                     [-np.sin(psi), 0, np.cos(psi)]])
    return RotY

# Function generating a rotation matrix around the Z-axis (yaw)
def rotationZ(phi):  # Euler angle phi related to yaw = gamma in the thesis
    RotZ = np.array([[np.cos(phi), -np.sin(phi), 0],
                     [np.sin(phi), np.cos(phi), 0],
                     [0, 0, 1]])
    return RotZ

# Function calculating the vectors corresponding to the lengths BiPi in the desired position
def VecteurLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Function calculating the absolute values of the lengths BiPi from the VecteurLi vectors
def ValeurLi(VecteurLi):
    return np.linalg.norm(VecteurLi)

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

# Function generating omegaIF based on the plots representing the trajectory
def linear_actuator_trajectory(roll_amp, pitch_amp, yaw_amp, xf, yf, zf, steps):
    
    # Initialize the figure
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Refresh function for the animation
    def update(frame):
        ax.cla()  # Clear previous plots
        
        # Yaw
        # roll = 0
        # pitch = 0
        # yaw = yaw_amp * frame / steps
        # x = 0 
        # y = 0
        # z = z0       
        
        # Helix trajectory
        # roll = roll_amp  # roll = roll_amp * np.sin(np.radians(frame))
        # pitch = pitch_amp  # roll_amp * np.sin(np.radians(frame))
        # yaw = yaw_amp  # 0
        # x = xf * np.cos(np.radians(frame))  # where xf and zf are the helix radii
        # y = yf * np.sin(np.radians(frame))
        # z = zmin + zf * frame  # zf corresponds to the helix pitch
            
        # Wave trajectory
        # roll = roll_amp * np.sin(np.radians(frame))
        # pitch = pitch_amp * np.cos(np.radians(frame))
        # yaw = yaw_amp * np.sin(np.radians(frame / 2))
        # x = xf * np.sin(np.radians(frame))
        # y = yf * np.cos(np.radians(frame))
        # z = z0 + zf * np.sin(np.radians(frame * 2))  # zf is the amplitude of the wave
        
        # Parameters for the involute of a circle
        roll = 0
        pitch = 0
        yaw = 0
        t = 3 * 2 * np.pi * frame / steps
        x = 5e-3 * (np.cos(t) + t * np.sin(t))
        y = 5e-3 * (np.sin(t) - t * np.cos(t))
        z = z0
    
        T = np.array([x, y, z])  # Vector corresponding to the distance between the center of the platform and the base
    
        theta = np.radians(roll)  # Roll (rotation around the X-axis)
        psi = np.radians(pitch)  # Pitch (rotation around the Y-axis)
        phi = np.radians(yaw)
    
        # Calculation of the transformation matrix using rotation matrices
        Rot_BtoP = rotationY(psi) @ rotationX(theta) @ rotationZ(phi)
        
        # Lists to store the values of the variables for each arm, necessary for the following for-loop
        VectorL = []
        P = []
        L = []
        C = []


        for i in range(6):
                 
            # Calculation of the vector between points Pi and Bi
            VLi = VecteurLi(Rot_BtoP, P0[i], B[i], T)
            VectorL.append(VLi)
            
            # Calculation of the new position of point Pi
            P.append(B[i] + VLi)
            
            # Calculation of the norm of the vector corresponding to the distance BiPi
            # which is the sum of the length of the cylinder body i and its stroke i
            L.append(ValeurLi(VectorL[i]))
            
            # Display of the norm of the vector corresponding to the distance BiPi
            print(f'L{i+1} :', L[i])
            
            # Calculation of the stroke of cylinder Vi
            C.append(L[i] - lab)
                
            # Display of the stroke of cylinder Vi
            print(f'Stroke C{i+1} :', C[i])
            
        # Conversion of the list P to a numpy array (vector)
        P = np.array(P)
        
        # Generation of the legend block
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='#386480', s=100, label='Base')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#72bdba', s=100, label='Cylinder Bodies')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#ffd783', s=100, label='Cylinder Strokes')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#f2887c', s=100, label='3DOF Platform')
        
        # Connection lines and annotations
        for i in range(6):
            
            # Plots of segments BiPi
            ax.plot([B[i, 0], P[i, 0]], 
                    [B[i, 1], P[i, 1]], 
                    [B[i, 2], P[i, 2]], '#5f7fbf', lw=3, linestyle='--')
            
            # Plots of the cylinder bodies
            ax.plot([B[i, 0], B[i, 0] + lab/L[i]*(P[i, 0] - B[i, 0])], 
                    [B[i, 1], B[i, 1] + lab/L[i]*(P[i, 1] - B[i, 1])], 
                    [B[i, 2], B[i, 2] + lab/L[i]*(P[i, 2] - B[i, 2])], '#72bdba', lw=4,)
            
            # Plots of the cylinder strokes
            ax.plot([B[i, 0] + lab/L[i]*(P[i, 0] - B[i, 0]), P[i, 0]], 
                    [B[i, 1] + lab/L[i]*(P[i, 1] - B[i, 1]), P[i, 1]], 
                    [B[i, 2] + lab/L[i]*(P[i, 2] - B[i, 2]), P[i, 2]], '#ffd783', lw=4,)
               
            # Displaying the names of points and segments with an offset for readability
            ax.text(B[i, 0] + B_offset, B[i, 1] + B_offset, B[i, 2] + B_offset, f'B{i+1}', color='#386480', fontsize=12, weight='bold')
            ax.text(P[i, 0], P[i, 1], P[i, 2] + P_offset, f'P{i+1}', color='#f2887c', fontsize=12, weight='bold')
            ax.text(B[i, 0] + lab/L[i]*(P[i, 0] - B[i, 0]) + L_offset, B[i, 1] + lab/L[i]*(P[i, 1] - B[i, 1]) + L_offset, B[i, 2] + lab/L[i]*(P[i, 2] - B[i, 2]) + L_offset, f'L{i+1}', color='#5f7fbf', fontsize=12, fontweight='bold')
    
        # Plots of circles representing the base and the platform
        plot_circle(ax, np.array([0, 0, 0]), R, np.array([0, 0, 1]), color='#386480', linestyle='--')
        plot_circle(ax, T, r, Rot_BtoP @ np.array([0, 0, 1]), color='#f2887c', linestyle='--')
        
        # Visual improvements
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio 1:1:1
    
        # Set the dimensions of the visualization space
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([0, 1])
    
        # Set axis labels with tilt and offset from the axes
        ax.set_xlabel('X Axis (m)', fontsize=15, labelpad=18, weight='book')
        ax.set_ylabel('Y Axis (m)', fontsize=15, labelpad=18, weight='book')
        ax.set_zlabel('Z Axis (m)', fontsize=15, labelpad=18, weight='book')
        
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Add system state information in a text box
        info_text = (f'\n'
                     f' 6DOF Platform Simulation:  \n'
                     f'\n'
                     f'  Roll ($\\theta$): {roll:.2f}°\n'
                     f'\n'
                     f'  Pitch ($\\psi$): {pitch:.2f}°\n'
                     f'\n'
                     f'  Yaw ($\\phi$): {yaw:.2f}°\n'
                     f'\n'
                     f'  Height: {z:.4f} m\n'
                     f'\n'
                     f'  Stroke C1: {C[0]:.4f} m\n'
                     f'\n'
                     f'  Stroke C2: {C[1]:.4f} m\n'
                     f'\n'
                     f'  Stroke C3: {C[2]:.4f} m\n'
                     f'\n'
                     f'  Stroke C4: {C[3]:.4f} m\n'
                     f'\n'
                     f'  Stroke C5: {C[4]:.4f} m\n'
                     f'\n'
                     f'  Stroke C6: {C[5]:.4f} m'
                     f'\n')
        props = dict(boxstyle='round', facecolor='white', alpha=0.15)
        ax.text2D(-0.26, 0.475, info_text, transform=ax.transAxes, fontsize=15, weight='book', fontstyle='italic', verticalalignment='center', bbox=props)   
        ax.legend(loc=(-0.265, 0.065), prop={'size': 15, 'weight': 'book', 'style': 'italic'}) 
 
    # Generates a folder, if it does not exist, in the same location as the program file
    result_animation = 'Trajectory Result P-6DoF Actuators'
    if not os.path.exists(result_animation):
        os.makedirs(result_animation)
    
    # Path to save the generated animation GIF
    gif_path = os.path.join(result_animation, 'Trajectory P-6DoF Actuators.gif')
        
    # Create the animation
    ani = FuncAnimation(fig, update, frames=range(steps), repeat=False)
    ani.save(gif_path, writer='pillow', fps=40)
    plt.show()

# Call the function for spiral or helical trajectories
# steps correspond to the number of revolutions to perform in this case, and zf is the height step at each increment
# linear_actuator_trajectory(roll_amp=30, pitch_amp=30, yaw_amp=0, xf=25e-3, yf=25e-3, zf=1e-3, steps=360) 

# Call the function for a Wave trajectory
linear_actuator_trajectory(roll_amp=30, pitch_amp=30, yaw_amp=10, xf=0.1, yf=0.05, zf=0.02, steps=300)
