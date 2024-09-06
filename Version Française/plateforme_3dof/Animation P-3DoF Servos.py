import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres constants :
R = 155e-3 # Rayon de la base en m #155e-3
r = 80e-3  # Rayon de la plateforme mobile en m

# Bi est le point, centre de la liaison i entre l’effecteur et la base
# Pi est le point, centre de la liaison i entre l’effecteur et la plateforme
B1 = np.array([-R * np.cos(np.radians(0)), R * np.sin(np.radians(0)), 0])
B2 = np.array([-R * np.cos(np.radians(120)), R * np.sin(np.radians(120)), 0])
B3 = np.array([-R * np.cos(np.radians(240)), R * np.sin(np.radians(240)), 0])
P1 = np.array([-r * np.cos(np.radians(0)), r * np.sin(np.radians(0)), 0])
P2 = np.array([-r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0])
P3 = np.array([-r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0])

B = np.array([B1, B2, B3]) 
P0 = np.array([P1, P2, P3]) 

d = 250e-3  # Longueur bras (distance entre les centres des pivots du bras) en mm
e = 415e-3  # Longueur avant-bras (distance des centre des pivot bras-croix de cardan) en mm

phi0 = 0 # Position angulaire initiale des Servos 
z0 = 331e-3  # Hauteur de la position lorsque les bras sont 90°
zmax = 665e-3  # Hauteur ent position haute (Servo à 0°)
zmin = 250e-3  # Hauteur ent position basse (Servo à 180°)
roulis_min = -45
roulis_max = 45
tangage_min = -45
tangage_max = 45


# Positions limites des MX_64 pour cette configuration
xi_min = 0 
xi_max = 180

# Décalage des annotation présente sur le graphique
B_offset = 2e-3
P_offset = 2e-3
L_offset = 2e-3
A_offset = 2e-3

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

# Fonction calculant la valeur de l'angle à envoyer au moteur si il n'est pas piloté en position en fonction de Valeur Li
# Cet Angle correspond à l'angle entre le bras (jaune) et l'avant bras (turquoise) 
def AngleLi(Li):
    Edeg = ((Li**2)+(d**2)-(e**2))/(2*d*Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = -np.arccos(Edeg) + np.radians(90.0) + np.radians(phi0)
    return round(np.degrees(xi))

# Fonction calculant la position angulaire des Servos en fonction de Valeur Li 
def AngleLi_MX_64(Li):
    Edeg = ((Li**2)+(d**2)-(e**2))/(2*d*Li)
    Edeg = np.clip(Edeg, -1, 1)
    xi = abs(-np.arccos(Edeg) + np.radians(phi0)) # On a supprimer le 90° par rapport au calcul de la team Microgoat car notre O est à la verticale et une valeur absolue car MX-64 sont piloter sur la plage de 0° à 180°
    xi = np.clip(np.degrees(xi), 0, 180)
    return round(xi)

def Point_intermédiaire(xi):       
    # Calcul des positions
    x_int = d * np.cos(np.radians(xi))
    z_int = d * np.sin(np.radians(xi))
    return x_int,z_int

# Fonction pour tracer un cercle dans un repère 3D
def plot_cercle(ax, centre, rayon, normale, color='k', linestyle='--'):
    alpha = np.linspace(0, 2 * np.pi, 100)
    normale = normale / np.linalg.norm(normale)
    v = np.array([1, 0, 0], dtype=float) if abs(normale[0]) < abs(normale[1]) else np.array([0, 1, 0], dtype=float)
    v = v - v.dot(normale) * normale
    v = v.astype(float) / np.linalg.norm(v)
    w = np.cross(normale, v)
    
    circle = centre[:, None] + rayon * (v[:, None] * np.cos(alpha) + w[:, None] * np.sin(alpha))
    ax.plot(circle[0, :], circle[1, :], circle[2, :], color=color, linestyle=linestyle)

# Fonction générant un GIF à partir des plots représentant la trajectoire entre la position initiale et la position finale
def animation(roulis_start, tangage_start, hauteur_start, roulis_end, tangage_end, hauteur_end, steps=100):
    
    # Interpolation linéaire
    roulis_steps = np.linspace(roulis_start, roulis_end, steps)
    tangage_steps = np.linspace(tangage_start, tangage_end, steps)
    hauteur_steps = np.linspace(hauteur_start, hauteur_end, steps)
    
    # Initialisation de la figure
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Fonction de rafraichissement de l'nimation update function
    def update(frame):
        
        ax.cla()  # Efface les tracés précédents
        
        # Encadrements des valeurs des paramètres d'entrée de la fonction
        roulis = np.clip(roulis_steps[frame], roulis_min, roulis_max)
        tangage = np.clip(tangage_steps[frame], tangage_min, tangage_max)
        hauteur = np.clip(hauteur_steps[frame], zmin, zmax)

        T = np.array([0, 0, hauteur]) # Vecteur correspondant à la distance entre le centre de la plateforme et celui de la base

        theta = np.radians(roulis)  # Roulis (rotation autour de l'axe X)
        psi = np.radians(tangage)  # Tangage (rotation autour de l'axe Y)

        # Calcul de la matrice de passage à partir des matrices de rotations
        Rot_BtoP = rotationY(psi) @ rotationX(theta)
        
        # Listes permettant de stocker les valeurs des variables pour chaque Bras et nécessaire pour réaliser la boucle for suivante
        VecteurL = []
        P = []
        L = []
        xi = []
        xz = [] 
        A = []

        for i in range(3):
                 
            # Calcul du vecteur entre les points Pi et Bi
            VLi = VecteurLi(Rot_BtoP, P0[i], B[i], T)
            VecteurL.append(VLi)
            
            # Calcul de la nouvelle position du point Pi
            P.append(B[i]+VLi)
             
            # Calcul de la norme du vecteur correspondant a la distance BiPi
            L.append(ValeurLi(VecteurL[i]))
            
            # Affichage de la norme du vecteur correspondant à la distance BiPi
            print(f'L{i+1} :', L[i])
            
            # Calcul de la position angulaire à renvoyer au Servo i
            xi.append(AngleLi_MX_64(L[i]))
    
            # Limitation de la position extrême du Servo i
            if xi[i]< xi_min:
                xi[i] = xi_min
            elif xi[i] > xi_max:
                xi[i] = xi_max
        
            # Affichage de la position angulaire du Servos i
            print(f'xi{i+1} : ', xi[i])
            
            xz.append(Point_intermédiaire(AngleLi(L[i])))

            # Position du point Ai correspondant à l'articulation entre le bras et l'avant bras
            
            # Configuration standard (pivot des bras contenue dans le volume de la plateforme 3DoF)
            ai = np.array([
                (-R + xz[i][0]) * np.cos(np.radians(120 * i)),
                (R - xz[i][0]) * np.sin(np.radians(120 * i)),
                xz[i][1]
            ])
            
            # Configuration 'triangulaire' axes des bras tangents à la base, configuration fonctionnelle uniquement si r=R
            # ai = np.array([
            #     -R * np.cos(np.radians(120 * i)) + xz[i][0]* np.cos(np.radians(120 * i + 90)),
            #     R * np.sin(np.radians(120 * i)) - xz[i][0]* np.sin(np.radians(120 * i + 90)),
            #     xz[i][1]
            # ])
            
            # Listes des positions des points correspondant à la pivot i entre le bras et l'avant-bras 
            A.append(ai)
    
        # Convertion des listes A et P en numpy array (vecteur)
        A = np.array(A)
        P = np.array(P)
        
        # Génération du bloc de légende 
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='#386480', s=100, label='Base mobile')
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='#ffd783', s=100, label='Bras')
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], color='#72bdba', s=100, label='Avant-bras')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#f2887c', s=100, label='Plateforme 3DoF')

        # Lignes de connexion et annotations
        for i in range(3):
            
            
            # Tracés des segments BiPi 
            ax.plot([B[i, 0], P[i, 0]], 
                    [B[i, 1], P[i, 1]], 
                    [B[i, 2], P[i, 2]], '#5f7fbf', lw=3, linestyle='--')
            
            # Tracés des segments BiAi représentant le bras i
            ax.plot([B[i, 0], A[i, 0]], 
                    [B[i, 1], A[i, 1]], 
                    [B[i, 2], A[i, 2]], '#ffd783', lw=3)
            
            # Tracés des segments BiAi représentant l'avant-bras i
            ax.plot([A[i, 0], P[i, 0]], 
                    [A[i, 1], P[i, 1]], 
                    [A[i, 2], P[i, 2]], '#72bdba', lw=3)
            
            # Affichage des noms des points et des segments avec un offset pour être lisible 
            ax.text(B[i, 0] + B_offset, B[i, 1] + B_offset, B[i, 2] + B_offset, f'B{i+1}', color='#386480', fontsize=12, weight='bold')
            ax.text(P[i, 0] + P_offset, P[i, 1]+ P_offset, P[i, 2] + P_offset, f'P{i+1}', color='#f2887c', fontsize=12, weight='bold')
            ax.text(A[i, 0] + A_offset, A[i, 1] + A_offset, A[i, 2] + A_offset, f'A{i+1}', color='#72bdba', fontsize=12, weight='bold')

            ax.text(B[i, 0] + L_offset, B[i, 1] + L_offset, z0/ 2 + L_offset, f'L{i+1}', color='#5f7fbf', fontsize=12, fontweight='bold')

        # Tracés des cercles représentant la base et la plateforme
        plot_cercle(ax, np.array([0, 0, 0]), R, np.array([0, 0, 1]), color='#386480', linestyle='--')
        plot_cercle(ax, T, r, Rot_BtoP @ np.array([0, 0, 1]), color='#f2887c', linestyle='--')
        
        # Améliorations visuelles
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio 1:1:1

        # Fixe les dimensions de l'espace de visualisation
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([0, 0.9])

        # Définition des étiquettes des axes avec l'inclinaison et le décalage par rapport aux axes
        ax.set_xlabel('Axe X (mm)', fontsize=15, labelpad=18, weight='book')
        ax.set_ylabel('Axe Y (mm)', fontsize=15, labelpad=18, weight='book')
        ax.set_zlabel('Axe Z (mm)', fontsize=15, labelpad=18, weight='book')
        
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Ajout des informations sur l'état du système dans un encadré
        info_text = (f'\n'
                     f' Simulation plateforme 3DoF :  \n'
                     f'\n'
                     f'  Roulis ($\\theta$) : {roulis:.2f}°\n'
                     f'\n'
                     f'  Tangage ($\\psi$) : {tangage:.2f}°\n'
                     f'\n'
                     f'  Hauteur : {hauteur:.4f} m\n'
                     f'\n'
                     f'  Longueur L1 : {L[0]:.4f} m\n'
                     f'\n'
                     f'  Longueur L2 : {L[1]:.4f} m\n'
                     f'\n'
                     f'  Longueur L3 : {L[2]:.4f} m\n'
                     f'\n'
                     f'  Angle xi1 : {xi[0]:.2f} °\n'
                     f'\n'
                     f'  Angle xi2 : {xi[1]:.2f} °\n'
                     f'\n'
                     f'  Angle xi3 : {xi[2]:.2f} °'
                     f'\n')
        props = dict(boxstyle='round', facecolor='white', alpha=0.15)
        ax.text2D(-0.26, 0.475, info_text, transform=ax.transAxes, fontsize=15, weight='book', fontstyle='italic', verticalalignment='center', bbox=props)   
        ax.legend(loc=(-0.265, 0.09),prop={'size': 15, 'weight': 'book', 'style': 'italic'}) # loc=(-0.265, 0.22) avant l'ajout de la course dans la légende
        
    # Génere un dossier, si il n'existe pas, au même emplacement que le fichier du programme  
    resultat_animation = 'Résultat animation P-3DoF Servos'
    if not os.path.exists(resultat_animation):
        os.makedirs(resultat_animation)
    
    # Chemin d'accés pour sauvegarder le GIF de l'animation générer
    gif_path = os.path.join(resultat_animation, 'P-3DoF Servos.gif')
        
    # Création de l'animation
    ani = FuncAnimation(fig, update, frames=range(steps), repeat=False)
    ani.save(gif_path, writer='pillow', fps=20)
    plt.show()

# Paramètres de simulation
roulis_start = 0
tangage_start = 0
hauteur_start = zmin  # Hauteur initiale en mm
roulis_end = 0
tangage_end = 0
hauteur_end = zmax  # Hauteur finale en mm

# Exécution de la simulation
animation(roulis_start, tangage_start, hauteur_start, roulis_end, tangage_end, hauteur_end, steps=100)
