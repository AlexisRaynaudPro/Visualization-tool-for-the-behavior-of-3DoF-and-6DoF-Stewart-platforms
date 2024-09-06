# Programme permettant de piloter en position la plateforme p123 en fonction de la hauteur, du roulis et du tangage de la plateforme 

import numpy as np
import time
import pypot.dynamixel

# Connexion aux Dynamixels
ports = pypot.dynamixel.get_available_ports()
if not ports:
    raise IOError('Aucun ports détectés !')

print('Ports détectés : ', ports)

dxl_io = pypot.dynamixel.DxlIO(ports[0], baudrate=1000000)
dxl_ids = [1, 2, 3]  # Spécification des 3 IDs des Dynamixels pour miniser le temps de recherche des IDs 

print('Attempting to connect to Dynamixel IDs:', dxl_ids)

# Vérification que les Dynamixels sont connectés
connected_ids = dxl_io.scan(dxl_ids)
print('IDs des Dynamixels connectés : ', connected_ids)

# Vérification que tous les Dynamixels sont détectés aux IDs spécifiés
if not all(motor_id in connected_ids for motor_id in dxl_ids):
    raise IOError('Les Dynamixels ne sont pas tous détectés !')
    
# Activation du couple des Dynamixels de la plateforme
dxl_io.enable_torque(dxl_ids)

# Paramètres constants:
    
R = 140.116/2  # Rayon de la base en mm
r = 140.116/2  # Rayon de la plateforme mobile en mm

# Positions des points de fixation des bras aux Dynamixels et des centres des pivots des joints de cardans
B1 = np.array([-R * np.cos(np.radians(0)), R * np.sin(np.radians(0)), 0])
B2 = np.array([-R * np.cos(np.radians(120)), R * np.sin(np.radians(120)), 0])
B3 = np.array([-R * np.cos(np.radians(240)), R * np.sin(np.radians(240)), 0])
P1 = np.array([-r * np.cos(np.radians(0)), r * np.sin(np.radians(0)), 0])
P2 = np.array([-r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0])
P3 = np.array([-r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0])

# Listes des positions des points de fixation des bras aux Dynamixels et des centres des pivots des joints de cardans
B = [B1,B2,B3]
P = [P1,P2,P3]

d = 50.014 # Longueur bras (distance entre les centres des pivots du bras) en mm
e = 81.867 # Longueur avant-bras (distance des centre des pivot bras-croix de cardan) en mm

phi0 = 0 # 
zmax = 131.881 # Hauteur ent position haute (Dynamixel à 0°)
zmin = 31.881 # Hauteur ent position basse (Dynamixel à 180°)

# Positions limites des MX_64 pour cette configuration
xi_min = 0 
xi_max = 180

# Fonction générant une matrice de rotation autour de l'axe X (roulis)
def rotationX(theta):  # Angle d'Euler theta lié au roulis = alpha dans la thèse
    RotX = np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])
    return RotX

# Fonction générant une matrice de rotation autour de l'axe Y (tangage)
def rotationY(psi): # Angle d'Euler psi lié au tangage = beta dans la thèse
    RotY = np.array([[np.cos(psi), 0, np.sin(psi)],
                     [0, 1, 0],
                     [-np.sin(psi), 0, np.cos(psi)]])
    return RotY

# Fonction calculant les vecteurs correspondant aux longueurs BiPi dans la position désirée
def VecteurLi(Rot_BtoP, P, B, T):
    return T + (Rot_BtoP @ P) - B

# Fonction calculant les valeurs absolues les longueurs BiPi à partir des vecteurs VecteurLi
def ValeurLi(VecteurLi):
    return np.linalg.norm(VecteurLi)

# Fonction calculant la position angulaire des Dynamixels en fonction de Valeur Li 
def AngleLi_MX_64(Li):
    Edeg = ((Li**2)+(d**2)-(e**2))/(2*d*Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = abs(-np.arccos(Edeg) + np.radians(phi0)) # On a supprimer le 90° par rapport au calcul de la team Microgoat car notre O est à la verticale et une valeur absolue car MX-64 sont piloter sur la plage de 0° à 180°
    xi = np.clip(np.degrees(xi), 0, 180)
    return round(xi)

# Fonction calculant la cinématique inverse à partir des fonctions précédentes et de mise en mouvement de la plateforme
def Move(roulis, tangage, hauteur):
    
    
    # Encadrements des valeurs des paramètres d'entrée de la fonction
    roulis = np.clip(roulis, -45, 45)
    tangage = np.clip(tangage, -45, 45)
    hauteur = np.clip(hauteur, zmin, zmax)
    
    T = np.array([0, 0, hauteur]) # Vecteur correspondant à la distance entre le centre de la plateforme et celui de la base
    
    theta = np.radians(roulis)  # Roulis (rotation autour de l'axe X)
    psi = np.radians(tangage)  # Tangage (rotation autour de l'axe Y)
    
    # Calcul de la matrice de passage à partir des matrices de rotations
    Rot_BtoP = rotationY(psi) @ rotationX(theta)
   
    # Listes permettant de stocker les valeurs des variables pour chaque Bras et ainsi réaliser une boucle for
    VecteurL = []
    L = []
    xi = []

    # Boucle for permettant le calcul de la cinématique inverse pour chaque Bras 
    for i in range(3):
    
        # Calcul du vecteur entre les points Pi et Bi
        VecteurL.append(VecteurLi(Rot_BtoP, P[i], B[i], T))
         
        # Calcul de la norme du vecteur correspondant a la distance BiPi
        L.append(ValeurLi(VecteurL[i]))
        
        # Affichage de la norme du vecteur correspondant à la distance BiPi
        print(f'L{i+1} :', L[i])

        # Calcul de la position angulaire à renvoyer au Dynamixel i
        xi.append(AngleLi_MX_64(L[i]))

        # Limitation de la position extrême du Dynamixel i
        if xi[i]< xi_min:
            xi[i] = xi_min
        elif xi[i] > xi_max:
            xi[i] = xi_max
    
        # Affichage de la position angulaire du Dynamixels i
        print(f'xi{i+1} : ', xi[i])
   
    # Correspondance             x1       x2       x3
    dxl_io.set_goal_position({1: xi[0], 2:xi[1], 3:xi[2]})
    
# Paramètres régissant le mouvement de la plateforme
roulis = 10
tangage = 10
hauteur = 90
    
# Effectue une trajetoire suivant un cercle avec la plateforme inclinée
# for i in range(360):
#     # Exemple d'appel de la fonction
#     Move(roulis * np.sin(np.radians(i)), tangage * np.cos(np.radians(i)),hauteur)
#     time.sleep(0.0025)


# Effectue une ascension puis une descente suivant un cercle avec la plateforme inclinée
# for t in range (4):
#     for i in range(360):
#         # Exemple d'appel de la fonction
#         Move(roulis * np.sin(np.radians(i)), tangage * np.cos(np.radians(i)),(hauteur-50)+50*i/360)
#         time.sleep(0.0025)
#     for i in range(360):
#         # Exemple d'appel de la fonction
#         Move(roulis * np.sin(np.radians(360-i)), tangage * np.cos(np.radians(360-i)),(hauteur-50)+50*(360-i)/360)
#         time.sleep(0.0025)   

# Houle
# for t in range (4):
#     for i in range(360):
#         # Exemple d'appel de la fonction
#         Move(roulis * 0, tangage * np.cos(np.radians(i/2)),(hauteur-50)+50*i/360)
#         time.sleep(0.0025)
#     for i in range(360):
#         # Exemple d'appel de la fonction
#         Move(roulis * 0, tangage * np.cos(np.radians(180+i/2)),(hauteur-50)+50*(360-i)/360)
#         time.sleep(0.0025)


# Rebond
# for i in range(3):
#     simulate(0, -5, hauteur)
#     time.sleep(0.250)
#     simulate(0, 20, hauteur)
#     time.sleep(0.250)
#     simulate(0, -5, hauteur)

# Catapulte 
# simulate(0, 40, hauteur)
# time.sleep(0.50)
# simulate(0, -5, hauteur)

# Démonstration 3DoF position limite 
Move(0,0,zmin)
time.sleep(2)
Move(0,0,zmax)
time.sleep(2)
Move(0,0,80)
time.sleep(2)
Move(30,0,80)
time.sleep(2)
Move(-30,0,80)
time.sleep(2)
Move(0,30,80)
time.sleep(2)
Move(0,-30,80)
time.sleep(2)
Move(0,0,80) 

for i in range(3*360):
    # Exemple d'appel de la fonction
    Move(10 * np.sin(np.radians(i)), 10 * np.cos(np.radians(i)),80)
    time.sleep(0.0025)

Move(0,0,80) 


time.sleep(1)

# Désactivez le couple des moteurs détectés
dxl_io.disable_torque(dxl_ids)









# Version developpée du calcul cinématique (sans for) : 
    
    #     # Calcul des nouvelles positions des moteurs
    #     VecteurLi1 = VecteurLi(Rot_BtoP, P1, B1, T)
    #     VecteurLi2 = VecteurLi(Rot_BtoP, P2, B2, T)
    #     VecteurLi3 = VecteurLi(Rot_BtoP, P3, B3, T)
         
    #     # Calcul des longueurs des moteurs
    #     Li1 = ValeurLi(VecteurLi1)
    #     Li2 = ValeurLi(VecteurLi2)
    #     Li3 = ValeurLi(VecteurLi3)
        
    #     # Affichage des résultats
    #     print("Li1 : ", Li1)
    #     print("Li2 : ", Li2)
    #     print("Li3 : ", Li3)
         
    #     # Calcul des angles à envoyer au servo moteurs
    #     xi1 = AngleLi_MX_64(Li1)
    #     xi2 = AngleLi_MX_64(Li2)
    #     xi3 = AngleLi_MX_64(Li3)
        
    #     xi = [xi1, xi2, xi3]
    
    #     # Limitation des positions extrêmes des MX_64
    #     if xi[i]< xi_min:
    #         xi[i] = xi_min
    #     elif xi[i] > xi_max:
    #         xi[i] = xi_max
    
    # # Affichage des résultats
    # print("xi1 : ", xi1)
    # print("xi2 : ", xi2)
    # print("xi3 : ", xi3)

    # # Activez le couple des moteurs détectés
    # dxl_io.enable_torque(dxl_ids)
    
    # dxl_io.set_goal_position({1: xi1, 2:xi2, 3:xi3})