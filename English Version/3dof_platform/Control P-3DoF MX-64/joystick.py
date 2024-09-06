# Program to control one of the 3DoF platforms based on the movement of the second platform manually

import time
import pypot.dynamixel

# Connecting to Dynamixels
ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('No ports detected!')

print('Detected ports: ', ports)

dxl_io = pypot.dynamixel.DxlIO(ports[0], baudrate=1000000)
dxl_ids = [1, 2, 3, 4, 5, 6]  # Specification of the 6 Dynamixel IDs to minimize the ID search time

print('Attempting to connect to Dynamixels with IDs: ', dxl_ids)

# Checking that the Dynamixels are connected
connected_ids = dxl_io.scan(dxl_ids)
print('Connected Dynamixel IDs: ', connected_ids)

# Verifying that all Dynamixels are detected at the specified IDs
if not all(motor_id in connected_ids for motor_id in dxl_ids):
    raise IOError('Not all Dynamixels are detected!')

p123 = (1, 2, 3)  # List of Dynamixels assigned to the image platform
p456 = (4, 5, 6)  # List of Dynamixels assigned to the control platform

# Enabling torque for the Dynamixels on the image platform p123
dxl_io.enable_torque(p123)

# Getting the positions of the Dynamixels on platform p456 in real-time, then reading and transferring to the Dynamixels on platform p123
# Positions of p456 are retrieved instead of p123 because p456 is more constrained in its movements than p123 due to overconstraints in certain positions
# and could risk damaging itself in the inverse configuration
try:
    while True:
        
        # Variable corresponding to time t
        #   start = time.time()
        
        # Retrieve the current positions of the Dynamixels on p456
        present_pos = dxl_io.get_present_position(p456)
        
        # Transfer the retrieved positions to the Dynamixels on p123
        dxl_io.set_goal_position({1: present_pos[0], 2: present_pos[1], 3: present_pos[2]})
        
        time.sleep(0.0025)
        
        # for i in range(3):
        #     # Display the retrieved positions
        #     print(f'xi{i+1} : ', dxl_io.get_present_position(p456)[i])

        # print(time.time() - start) # Corresponds to the time taken for the while loop

# Exiting the while loop
except KeyboardInterrupt:
    print("Program interrupted by the user.")

finally:
    # Disabling torque for the Dynamixels
    dxl_io.disable_torque(p123)
    print("Dynamixel torque disabled.")
