import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constant parameters:
R = 155e-3 # Base radius in m
r = 80e-3  # Mobile platform radius in m

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

cv = 415e-3 # Actuator body length
z0 = 665e-3  # Middle position height
zmax = 915e-3  # Highest position height
zmin = 415e-3  # Lowest position height 
roll_min = -45
roll_max = 45
pitch_min = -45
pitch_max = 45

# Offset for annotations on the graph
B_offset = 10e-3
P_offset = 30e-3
L_offset = 5e-3

# Function generating a rotation matrix around the X axis (roll)
def rotationX(theta):  # Euler angle theta related to roll = alpha in the thesis
    RotX = np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return RotX

# Function generating a rotation matrix around the Y axis (pitch)
def rotationY(psi): # Euler angle psi related to pitch = beta in the thesis
    RotY = np.array([[np.cos(psi), 0, np.sin(psi)],
                     [0, 1, 0],
                     [-np.sin(psi), 0, np.cos(psi)]])
    return RotY

# Function calculating the vectors corresponding to the lengths BiPi in the desired position
def VecteurLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Function calculating the absolute values of the lengths BiPi from the vectors VecteurLi
def ValeurLi(VecteurLi):
    return np.linalg.norm(VecteurLi)

# Function to plot a circle in a 3D reference frame
def plot_circle(ax, center, radius, normal, color='k', linestyle='--'):
    alpha = np.linspace(0, 2 * np.pi, 100)
    normal = normal / np.linalg.norm(normal)
    v = np.array([1, 0, 0], dtype=float) if abs(normal[0]) < abs(normal[1]) else np.array([0, 1, 0], dtype=float)
    v = v - v.dot(normal) * normal
    v = v.astype(float) / np.linalg.norm(v)
    w = np.cross(normal, v)
    
    circle = center[:, None] + radius * (v[:, None] * np.cos(alpha) + w[:, None] * np.sin(alpha))
    ax.plot(circle[0, :], circle[1, :], circle[2, :], color=color, linestyle=linestyle)

# Function generating a GIF from plots representing the trajectory between the initial and final positions
def animation(roll_start, pitch_start, height_start, roll_end, pitch_end, height_end, steps=100):
    
    # Linear interpolation
    roll_steps = np.linspace(roll_start, roll_end, steps)
    pitch_steps = np.linspace(pitch_start, pitch_end, steps)
    height_steps = np.linspace(height_start, height_end, steps)
    
    # Figure initialization
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Animation update function
    def update(frame):
        
        ax.cla()  # Clear previous plots
        
        # Clipping the input parameters
        roll = np.clip(roll_steps[frame], roll_min, roll_max)
        pitch = np.clip(pitch_steps[frame], pitch_min, pitch_max)
        height = np.clip(height_steps[frame], zmin, zmax)

        T = np.array([0, 0, height]) # Vector corresponding to the distance between the center of the platform and that of the base

        theta = np.radians(roll)  # Roll (rotation around the X axis)
        psi = np.radians(pitch)  # Pitch (rotation around the Y axis)

        # Calculating the transformation matrix from the rotation matrices
        Rot_BtoP = rotationY(psi) @ rotationX(theta)
        
        # Lists for storing variable values for each Arm, needed for the following for loop
        VecteurL = []
        P = []
        L = []
        S = []

        for i in range(3):
                 
            # Calculating the vector between points Pi and Bi
            VLi = VecteurLi(Rot_BtoP, P0[i], B[i], T)
            VecteurL.append(VLi)
            
            # Calculating the new position of point Pi
            P.append(B[i]+VLi)
             
            # Calculating the norm of the vector corresponding to the distance BiPi
            # i.e., the sum of the length of the actuator body and its stroke
            L.append(ValeurLi(VecteurL[i]))
            
            # Displaying the norm of the vector corresponding to the distance BiPi
            print(f'L{i+1} :', L[i])
            
            # Calculating the stroke of actuator Vi
            S.append(L[i]-cv)
                
            # Displaying the stroke of actuator Vi
            print(f'Stroke S{i+1} :', S[i])
            
        # Converting the list P to a numpy array (vector)
        P = np.array(P)
        
        # Generate the legend block 
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='#386480', s=100, label='Mobile Base')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#72bdba', s=100, label='Linears Actuators Bodies')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#ffd783', s=100, label='Linears Actuators Strokes')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#f2887c', s=100, label='3DoF Platform')

        
        # Connection lines and annotations
        for i in range(3):
            
            # Plotting segments BiPi
            ax.plot([B[i, 0], P[i, 0]], 
                    [B[i, 1], P[i, 1]], 
                    [B[i, 2], P[i, 2]], '#5f7fbf', lw=3, linestyle='--')
            
            # Plotting actuator bodies
            ax.plot([B[i, 0], B[i, 0]+cv/L[i]*(P[i,0]-B[i, 0])], 
                    [B[i, 1], B[i, 1]+cv/L[i]*(P[i,1]-B[i, 1])], 
                    [B[i, 2], B[i, 2]+cv/L[i]*(P[i,2]-B[i, 2])], '#72bdba', lw=4,)
            
            # Plotting actuator strokes
            ax.plot([B[i, 0]+cv/L[i]*(P[i,0]-B[i, 0]), P[i, 0]], 
                    [B[i, 1]+cv/L[i]*(P[i,1]-B[i, 1]), P[i, 1]], 
                    [B[i, 2]+cv/L[i]*(P[i,2]-B[i, 2]), P[i, 2]], '#ffd783', lw=4,)
               
                  
            # Displaying point and segment names with an offset for readability
            ax.text(B[i, 0] + B_offset, B[i, 1] + B_offset, B[i, 2] + B_offset, f'B{i+1}', color='#386480', fontsize=12, weight='bold')
            ax.text(P[i, 0], P[i, 1], P[i, 2] + P_offset, f'P{i+1}', color='#f2887c', fontsize=12, weight='bold')
            ax.text(B[i, 0]+cv/L[i]*(P[i,0]-B[i, 0]) + L_offset, B[i, 1]+cv/L[i]*(P[i,1]-B[i, 1]) + L_offset, B[i, 2]+cv/L[i]*(P[i,2]-B[i, 2]) + L_offset, f'L{i+1}', color='#5f7fbf', fontsize=12, fontweight='bold')

        # Plotting circles representing the base and the platform
        plot_circle(ax, np.array([0, 0, 0]), R, np.array([0, 0, 1]), color='#386480', linestyle='--')
        plot_circle(ax, T, r, Rot_BtoP @ np.array([0, 0, 1]), color='#f2887c', linestyle='--')
        
        # Visual enhancements
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio 1:1:1

        # Fixing the dimensions of the visualization space
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([0, 1])

        # Defining axis labels with tilt and offset from the axes
        ax.set_xlabel('X Axis (m)', fontsize=15, labelpad=18, weight='book')
        ax.set_ylabel('Y Axis (m)', fontsize=15, labelpad=18, weight='book')
        ax.set_zlabel('Z Axis (m)', fontsize=15, labelpad=18, weight='book')
        
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Adding system state information in a box
        info_text = (f'\n'
                     f' 3DoF Platform Simulation:  \n'
                     f'\n'
                     f'  Roll ($\\theta$) : {roll:.2f}°\n'
                     f'\n'
                     f'  Pitch ($\\psi$) : {pitch:.2f}°\n'
                     f'\n'
                     f'  Height : {height:.4f} m\n'
                     f'\n'
                     f'  Length L1 : {L[0]:.4f} m\n'
                     f'\n'
                     f'  Length L2 : {L[1]:.4f} m\n'
                     f'\n'
                     f'  Length L3 : {L[2]:.4f} m\n'
                     f'\n'
                     f'  Stroke S1 : {S[0]:.4f} m\n'
                     f'\n'
                     f'  Stroke S2 : {S[1]:.4f} m\n'
                     f'\n'
                     f'  Stroke S3 : {S[2]:.4f} m'
                     f'\n')
        props = dict(boxstyle='round', facecolor='white', alpha=0.15)
        ax.text2D(-0.26, 0.475, info_text, transform=ax.transAxes, fontsize=15, weight='book', fontstyle='italic', verticalalignment='center', bbox=props)   
        ax.legend(loc=(-0.265, 0.09),prop={'size': 15, 'weight': 'book', 'style': 'italic'})   
        
    # Create a folder if it doesn't exist, in the same location as the program file  
    resultat_animation = 'P-3DoF Linears Actuators Animation Result'
    if not os.path.exists(resultat_animation):
        os.makedirs(resultat_animation)
    
    # Path to save the generated GIF animation
    gif_path = os.path.join(resultat_animation, 'P-3DoF Linears Actuators.gif')
        
    # Creating the animation
    ani = FuncAnimation(fig, update, frames=range(steps), repeat=False)
    ani.save(gif_path, writer='pillow', fps=20)
    plt.show()

# Simulation parameters
roulis_start = 0
tangage_start = 0
hauteur_start = zmin  # Initial height in mm
roulis_end = 0
tangage_end = 0
hauteur_end = zmax  # Final height in mm

# Executing the simulation
animation(roulis_start, tangage_start, hauteur_start, roulis_end, tangage_end, hauteur_end, steps=100)


