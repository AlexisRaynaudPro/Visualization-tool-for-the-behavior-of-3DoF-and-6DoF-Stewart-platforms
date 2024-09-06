import pypot.dynamixel

# Connection to the Dynamixel
ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('No ports detected!')

print('Detected ports:', ports)

dxl_io = pypot.dynamixel.DxlIO(ports[0], baudrate=57600)  # Standard baudrate when new, however, the baudrate is 1000000 for Dynamixels on 3DoF platforms
dxl_ids = dxl_io.scan(range(253))  # Scan for possible Dynamixel IDs to find the current ID of the connected Dynamixel
print('Current Dynamixel ID:', dxl_ids)

# Variables defining the current Dynamixel ID and the desired new ID
current_id = dxl_ids[0]  # Current ID of the Dynamixel
new_id = 6               # Desired new ID

# Check if the Dynamixel with the retrieved ID is detected
if current_id not in dxl_ids:
    raise IOError(f'Dynamixel with ID {current_id} not detected!')

# Change the ID of the Dynamixel
dxl_io.change_id({current_id: new_id})
print(f'Dynamixel ID has been changed from {current_id} to {new_id}')

# Verify the ID change
dxl_ids = dxl_io.scan(range(253))
print('Dynamixel ID after modification:', dxl_ids)

# Close the connection
dxl_io.close()

