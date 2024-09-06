import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constant parameters:
R = 140.116 / 2  # Radius of the base in mm
r = 140.116 / 2  # Radius of the mobile platform in mm

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

d = 50.014  # Arm length (distance between the centers of the arm pivots) in mm
e = 81.867  # Forearm length (distance from the arm-pivot center to the universal joint cross) in mm

phi0 = 0  # Initial angular position of the Dynamixels
z0 = 64.814  # Height of the position when the arms are at 90°
zmax = 131.881  # Maximum height position (Dynamixel at 0°)
zmin = 31.881  # Minimum height position (Dynamixel at 180°)
roll_min = -45
roll_max = 45
pitch_min = -45
pitch_max = 45

# Limit positions of the MX_64 for this configuration
xi_min = 0
xi_max = 180

# Offset for annotations on the graph
B_offset = 2
P_offset = 2
L_offset = 2
A_offset = 2

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

# Function calculating the vectors corresponding to the lengths BiPi in the desired position
def VectorLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Function calculating the absolute values of the lengths BiPi from the VectorLi vectors
def ValueLi(VectorLi):
    return np.linalg.norm(VectorLi)

# Function calculating the angle value to send to the motor if it is not position-controlled based on Value Li
# This angle corresponds to the angle between the arm (yellow) and the forearm (turquoise)
def AngleLi(Li):
    Edeg = ((Li**2) + (d**2) - (e**2)) / (2 * d * Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = -np.arccos(Edeg) + np.radians(90.0) + np.radians(phi0)
    return round(np.degrees(xi))

# Function calculating the angular position of the Dynamixels based on Value Li
def AngleLi_MX_64(Li):
    Edeg = ((Li**2) + (d**2) - (e**2)) / (2 * d * Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = abs(-np.arccos(Edeg) + np.radians(phi0))  # Removed 90° compared to the calculation by the Microgoat team because our O is vertical, and an absolute value because MX-64 are controlled in the range of 0° to 180°
    xi = np.clip(np.degrees(xi), 0, 180)
    return round(xi)

def Intermediate_point(xi):
    # Calculate positions
    x_int = d * np.cos(np.radians(xi))
    z_int = d * np.sin(np.radians(xi))
    return x_int, z_int

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

    # Update function for the animation
    def update(frame):
        
        ax.cla()  # Clear previous plots
        
        # Clamping input parameter values
        roll = np.clip(roll_steps[frame], roll_min, roll_max)
        pitch = np.clip(pitch_steps[frame], pitch_min, pitch_max)
        height = np.clip(height_steps[frame], zmin, zmax)

        T = np.array([0, 0, height])  # Vector corresponding to the distance between the center of the platform and the base

        theta = np.radians(roll)  # Roll (rotation around the X axis)
        psi = np.radians(pitch)  # Pitch (rotation around the Y axis)

        # Calculate the transformation matrix from the rotation matrices
        Rot_BtoP = rotationY(psi) @ rotationX(theta)
        
        # Lists to store variable values for each arm, necessary for the following loop
        VectorL = []
        P = []
        L = []
        xi = []
        xz = [] 
        A = []

        for i in range(3):
                 
            # Calculate the vector between points Pi and Bi
            VLi = VectorLi(Rot_BtoP, P0[i], B[i], T)
            VectorL.append(VLi)
            
            # Calculate the new position of point Pi
            P.append(B[i] + VLi)
             
            # Calculate the norm of the vector corresponding to the distance BiPi
            L.append(ValueLi(VectorL[i]))
            
            # Display the norm of the vector corresponding to the distance BiPi
            print(f'L{i+1} :', L[i])
            
            # Calculate the angular position to send to Dynamixel i
            xi.append(AngleLi_MX_64(L[i]))
    
            # Limit the extreme position of Dynamixel i
            if xi[i] < xi_min:
                xi[i] = xi_min
            elif xi[i] > xi_max:
                xi[i] = xi_max
        
            # Display the angular position of Dynamixels i
            print(f'xi{i+1} : ', xi[i])
            
            xz.append(Intermediate_point(AngleLi(L[i])))

            # Position of point Ai corresponding to the joint between the arm and the forearm
            
            # Standard configuration (arm pivot contained within the platform volume 3DoF)
            ai = np.array([
                (-R + xz[i][0]) * np.cos(np.radians(120 * i)),
                (R - xz[i][0]) * np.sin(np.radians(120 * i)),
                xz[i][1]
            ])
            
            # Triangular configuration (arm axes tangent to the base, functional configuration only if r=R)
            # ai = np.array([
            #     -R * np.cos(np.radians(120 * i)) + xz[i][0]* np.cos(np.radians(120 * i + 90)),
            #     R * np.sin(np.radians(120 * i)) - xz[i][0]* np.sin(np.radians(120 * i + 90)),
            #     xz[i][1]
            # ])
    
            # List of positions of the points corresponding to the pivot i between the arm and the forearm
            A.append(ai)
            
        # Convert lists A and P into numpy arrays (vectors)
        A = np.array(A)
        P = np.array(P)
        
        # Generate the legend block
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='#386480', s=100, label='Mobile base')
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='#ffd783', s=100, label='Arm')
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='#72bdba', s=100, label='Forearm')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#f2887c', s=100, label='3DoF platform')
        
        for i in range(3):
            
            # Connection lines and annotations:
            
            # Draw BiPi segments
            ax.plot([B[i, 0], P[i, 0]], 
                    [B[i, 1], P[i, 1]], 
                    [B[i, 2], P[i, 2]], '#5f7fbf', lw=3, linestyle='--')
            
            # Draw BiAi segments representing arm i
            ax.plot([B[i, 0], A[i, 0]], 
                    [B[i, 1], A[i, 1]], 
                    [B[i, 2], A[i, 2]], '#ffd783', lw=3)
            
            # Draw AiPi segments representing forearm i
            ax.plot([A[i, 0], P[i, 0]], 
                    [A[i, 1], P[i, 1]], 
                    [A[i, 2], P[i, 2]], '#72bdba', lw=3)
            
            # Display point and segment names with offset for readability
            ax.text(B[i, 0] + B_offset, B[i, 1] + B_offset, B[i, 2] + B_offset, f'B{i+1}', color='#386480', fontsize=12, weight='bold')
            ax.text(P[i, 0] + P_offset, P[i, 1]+ P_offset, P[i, 2] + P_offset, f'P{i+1}', color='#f2887c', fontsize=12, weight='bold')
            ax.text(A[i, 0] + A_offset, A[i, 1] + A_offset, A[i, 2] + A_offset, f'A{i+1}', color='#72bdba', fontsize=12, weight='bold')
        
            ax.text(B[i, 0] + L_offset, B[i, 1] + L_offset, z0 / 2 + L_offset, f'L{i+1}', color='#5f7fbf', fontsize=12, fontweight='bold')
        
        # Draw circles representing the base and the platform
        plot_circle(ax, np.array([0, 0, 0]), R, np.array([0, 0, 1]), color='#386480', linestyle='--')
        plot_circle(ax, T, r, Rot_BtoP @ np.array([0, 0, 1]), color='#f2887c', linestyle='--')
        
        # Visual enhancements
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio 1:1:1
        
        # Set dimensions of the visualization space
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([0, 250])
        
        # Set axis labels with inclination and offset from axes
        ax.set_xlabel('X-axis (mm)', fontsize=15, labelpad=18, weight='book')
        ax.set_ylabel('Y-axis (mm)', fontsize=15, labelpad=18, weight='book')
        ax.set_zlabel('Z-axis (mm)', fontsize=15, labelpad=18, weight='book')
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add system status information in a box
        info_text = (f'\n'
                     f' 3DoF Platform Simulation: \n'
                     f'\n'
                     f'  Roll ($\\theta$) : {roll:.2f}°\n'
                     f'\n'
                     f'  Pitch ($\\psi$) : {pitch:.2f}°\n'
                     f'\n'
                     f'  Height : {height:.4f} mm\n'
                     f'\n'
                     f'  Length L1 : {L[0]:.4f} mm\n'
                     f'\n'
                     f'  Length L2 : {L[1]:.4f} mm\n'
                     f'\n'
                     f'  Length L3 : {L[2]:.4f} mm\n'
                     f'\n'
                     f'  Angle xi1 : {xi[0]:.2f} °\n'
                     f'\n'
                     f'  Angle xi2 : {xi[1]:.2f} °\n'
                     f'\n'
                     f'  Angle xi3 : {xi[2]:.2f} °'
                     f'\n')
        props = dict(boxstyle='round', facecolor='white', alpha=0.15)
        ax.text2D(-0.26, 0.475, info_text, transform=ax.transAxes, fontsize=15, weight='book', fontstyle='italic', verticalalignment='center', bbox=props)   
        ax.legend(loc=(-0.265, 0.09),prop={'size': 15, 'weight': 'book', 'style': 'italic'}) # loc=(-0.265, 0.22) before adding the stroke to the legend
        
    # Generate a folder, if it does not exist, in the same location as the program file  
    animation_result = 'P-3DoF MX64 Animation Result'
    if not os.path.exists(animation_result):
        os.makedirs(animation_result)
    
    # Path to save the generated animation GIF
    gif_path = os.path.join(animation_result, 'P-3DoF MX64.gif')
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=range(steps), repeat=False)
    ani.save(gif_path, writer='pillow', fps=20)
    plt.show()
        
# Simulation parameters
roll_start = 0
pitch_start = 0
height_start = zmin  # Initial height in mm
roll_end = 0
pitch_end = 0
height_end = zmax  # Final height in mm

# Run the simulation
animation(roll_start, pitch_start, height_start, roll_end, pitch_end, height_end, steps=100)




