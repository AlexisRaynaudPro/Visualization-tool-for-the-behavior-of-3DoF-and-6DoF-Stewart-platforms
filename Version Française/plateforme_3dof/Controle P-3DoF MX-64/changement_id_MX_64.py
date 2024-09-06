import pypot.dynamixel

# Connexion au Dynamixel
ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('Aucun ports détectés !')

print('Ports détectés : ', ports)

dxl_io = pypot.dynamixel.DxlIO(ports[0], baudrate=57600) # baudrate standard lorsque neuf, cependant baudrate=1000000 des Dynamixels des plateformes 3DoF
dxl_ids = dxl_io.scan(range(253)) # Parcours des IDs de Dynamixel possibles pour trouver l'ID actuel du Dynamixel connecté 
print('ID actuel du Dynamixel : ', dxl_ids)

# Variables définissants l'ID actuel du Dynamixel et le nouvel ID désirée
current_id = dxl_ids[0]  # ID actuel du Dynamixel
new_id = 6        # Nouvel ID souhaité

# Vérification de la détection du Dynamixel avec l'ID récupérez
if current_id not in dxl_ids:
    raise IOError(f'Dynamixel avec ID {current_id} non détecté !')

# Changement de l'ID du Dynamixel
dxl_io.change_id({current_id: new_id})
print(f'lID du Dynamixel à été changé de {current_id} à {new_id}')

# Vérification du changement d'ID
dxl_ids = dxl_io.scan(range(253))
print('ID du Dynamixel après modification : ', dxl_ids)

# Cloture de la connexion
dxl_io.close()

