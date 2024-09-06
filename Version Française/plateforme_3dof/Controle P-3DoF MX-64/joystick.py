# Programme permettant de piloter l'une des plateforme 3DoF comme image de la seconde plateforme mise en mouvement manuellement 

import time
import pypot.dynamixel

# Connexion aux Dynamixels
ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('Aucun ports détectés !')

print('Ports détectés : ', ports)

dxl_io = pypot.dynamixel.DxlIO(ports[0], baudrate=1000000)
dxl_ids = [1, 2, 3, 4, 5, 6]  # Spécification des 6 IDs des Dynamixels pour miniser le temps de recherche des IDs 

print('Tentative de connection aux Dynamixels dIDs : ', dxl_ids)

# Vérification que les Dynamixels sont connectés
connected_ids = dxl_io.scan(dxl_ids)
print('IDs des Dynamixels connectés : ', connected_ids)

# Vérification que tous les Dynamixels sont détectés aux IDs spécifiés
if not all(motor_id in connected_ids for motor_id in dxl_ids):
    raise IOError('Les Dynamixels ne sont pas tous détectés !')
    
p123 = (1,2,3) # Liste des Dynamixels affectés à la plateforme image 
p456 = (4,5,6) # Liste des Dynamixels affectés à la plateforme de controle 

# Activation du couple des Dynamixels de la plateforme image p123
dxl_io.enable_torque(p123)  

# Récupération des positions des Dynamixels de plateforme p456 en temps réels puis lecture et tranfert aux Dynamixels de la plateforme p123
# On récupere les positions de p456 et non celle de p123, car p456 est davantage limité dans ces mouvements que p123 du fait de surcontraintes dans certaines position 
# et risquerai de forcer et de s'endommager dans la configuration inverse  
try:
    while True:
        
        # Variable correspondant à l'instant t 
        #   start=time.time()
        
        # Récupération des positions actuelles des Dynamixels de p456
        present_pos=dxl_io.get_present_position(p456)
        
        # Tranfert des positions récupérées aux Dynamixels de p123
        dxl_io.set_goal_position({1: present_pos[0], 2:present_pos[1], 3:present_pos[2]})
        
        time.sleep(0.0025)
        
        # for i in range(3):
        #     # Affichage des positions récupérées 
        #     print(f'xi{i+1} : ' , dxl_io.get_present_position(p456)[i])


        # print(time.time()-start) # Correspond au temps de parcour de la boucle while

# Sortie de la boucle while  
except KeyboardInterrupt:
    print("Programme interrompu par l'utilisateur.")

finally:
    # Désactivation du couple des Dynamixels
    dxl_io.disable_torque(p123)
    print("Couple des Dynamixels désactivé.")
    