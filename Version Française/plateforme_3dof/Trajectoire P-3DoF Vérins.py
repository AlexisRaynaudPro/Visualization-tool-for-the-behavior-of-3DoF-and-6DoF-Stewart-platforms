import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Paramètres constants :
R = 155e-3 # Rayon de la base en m #155e-3
r = 80e-3  # Rayon de la plateforme mobile en m

# Bi est le point, centre de la liaison i entre l’effecteur et la base
# Pi est le point, centre de la liaison i entre l’effecteur et la plateforme.
B1 = np.array([-R * np.cos(np.radians(0)), R * np.sin(np.radians(0)), 0])
B2 = np.array([-R * np.cos(np.radians(120)), R * np.sin(np.radians(120)), 0])
B3 = np.array([-R * np.cos(np.radians(240)), R * np.sin(np.radians(240)), 0])
P1 = np.array([-r * np.cos(np.radians(0)), r * np.sin(np.radians(0)), 0])
P2 = np.array([-r * np.cos(np.radians(120)), r * np.sin(np.radians(120)), 0])
P3 = np.array([-r * np.cos(np.radians(240)), r * np.sin(np.radians(240)), 0])

B = np.array([B1, B2, B3]) 
P0 = np.array([P1, P2, P3]) 

cv = 415e-3 # Longueurs des corps des vérins
z0 = 665e-3  # Hauteur de la position médiane
zmax = 915e-3  # Hauteur ent position haute
zmin = 415e-3  # Hauteur ent position basse 
roulis_min = -45
roulis_max = 45
tangage_min = -45
tangage_max = 45

# Décalage des annotation présente sur le graphique
B_offset = 10e-3
P_offset = 30e-3
L_offset = 5e-3

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

# Fonction générant un GIF à partir des plots représentant la trajectoire
def trajectoire_vérins(roulis_amp, tangage_amp, hauteur, steps=100):
    
    # Initialisation de la figure
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')

    # Fonction de rafraichissement de l'animation
    def update(frame):
        
        ax.cla()  # Efface les tracés précédents
        
        # Calcul des paramètres de roulis, tangage et de hauteur en fonction de l'itération
        roulis = roulis_amp * np.sin(np.radians(frame))
        tangage = tangage_amp * np.cos(np.radians(frame))
        hauteur_actuelle = hauteur #(hauteur-300e-3)+300e-3*frame/360

        T = np.array([0, 0, hauteur_actuelle]) # Vecteur correspondant à la distance entre le centre de la plateforme et celui de la base

        theta = np.radians(roulis)  # Roulis (rotation autour de l'axe X)
        psi = np.radians(tangage)  # Tangage (rotation autour de l'axe Y)

        # Calcul de la matrice de passage à partir des matrices de rotations
        Rot_BtoP = rotationY(psi) @ rotationX(theta)
        
        # Listes permettant de stocker les valeurs des variables pour chaque Bras et nécessaire pour réaliser la boucle for suivante
        VecteurL = []
        P = []
        L = []
        C = []

        for i in range(3):
                 
            # Calcul du vecteur entre les points Pi et Bi
            VLi = VecteurLi(Rot_BtoP, P0[i], B[i], T)
            VecteurL.append(VLi)
            
            # Calcul de la nouvelle position du point Pi
            P.append(B[i]+VLi)
             
            # Calcul de la norme du vecteur correspondant a la distance BiPi
            # soi la somme de la longueur du corps du vérin i et de sa course i
            L.append(ValeurLi(VecteurL[i]))
            
            # Affichage de la norme du vecteur correspondant à la distance BiPi
            print(f'L{i+1} :', L[i])
            
            # Calcul de la course du vérin Vi
            C.append(L[i]-cv)
                
            # Affichage de la course du vérin Vi
            print(f'Course C{i+1} :', C[i])
            
        # Convertion de la liste P en numpy array (vecteur)
        P = np.array(P)
        
        # Génération du bloc de légende 
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], color='#386480', s=100, label='Base mobile')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#72bdba', s=100, label='Corps des vérins')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#ffd783', s=100, label='Courses des vérins')
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], color='#f2887c', s=100, label='Plateforme 3DoF')
        
        # Lignes de connexion et annotations
        for i in range(3):
            
            # Tracés des segments BiPi 
            ax.plot([B[i, 0], P[i, 0]], 
                    [B[i, 1], P[i, 1]], 
                    [B[i, 2], P[i, 2]], '#5f7fbf', lw=3, linestyle='--')
            
            # Tracés des corps des vérins
            ax.plot([B[i, 0], B[i, 0]+cv/L[i]*(P[i,0]-B[i, 0])], 
                    [B[i, 1], B[i, 1]+cv/L[i]*(P[i,1]-B[i, 1])], 
                    [B[i, 2], B[i, 2]+cv/L[i]*(P[i,2]-B[i, 2])], '#72bdba', lw=4,)
            
            # Tracés des courses des vérins
            ax.plot([B[i, 0]+cv/L[i]*(P[i,0]-B[i, 0]), P[i, 0]], 
                    [B[i, 1]+cv/L[i]*(P[i,1]-B[i, 1]), P[i, 1]], 
                    [B[i, 2]+cv/L[i]*(P[i,2]-B[i, 2]), P[i, 2]], '#ffd783', lw=4,)
               
            # Affichage des noms des points et des segments avec un offset pour être lisible 
            ax.text(B[i, 0] + B_offset, B[i, 1] + B_offset, B[i, 2] + B_offset, f'B{i+1}', color='#386480', fontsize=12, weight='bold')
            ax.text(P[i, 0], P[i, 1], P[i, 2] + P_offset, f'P{i+1}', color='#f2887c', fontsize=12, weight='bold')
            ax.text(B[i, 0]+cv/L[i]*(P[i,0]-B[i, 0]) + L_offset, B[i, 1]+cv/L[i]*(P[i,1]-B[i, 1]) + L_offset, B[i, 2]+cv/L[i]*(P[i,2]-B[i, 2]) + L_offset, f'L{i+1}', color='#5f7fbf', fontsize=12, fontweight='bold')

        # Tracés des cercles représentant la base et la plateforme
        plot_cercle(ax, np.array([0, 0, 0]), R, np.array([0, 0, 1]), color='#386480', linestyle='--')
        plot_cercle(ax, T, r, Rot_BtoP @ np.array([0, 0, 1]), color='#f2887c', linestyle='--')
        
        # Améliorations visuelles
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio 1:1:1

        # Fixe les dimensions de l'espace de visualisation
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([0, 1])

        # Définition des étiquettes des axes avec l'inclinaison et le décalage par rapport aux axes
        ax.set_xlabel('Axe X (m)', fontsize=15, labelpad=18, weight='book')
        ax.set_ylabel('Axe Y (m)', fontsize=15, labelpad=18, weight='book')
        ax.set_zlabel('Axe Z (m)', fontsize=15, labelpad=18, weight='book')
        
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Ajout des informations sur l'état du système dans un encadré
        info_text = (f'\n'
                     f' Simulation plateforme 3DoF :  \n'
                     f'\n'
                     f'  Roulis ($\\theta$) : {roulis:.2f}°\n'
                     f'\n'
                     f'  Tangage ($\\psi$) : {tangage:.2f}°\n'
                     f'\n'
                     f'  Hauteur : {hauteur_actuelle:.4f} m\n'
                     f'\n'
                     f'  Longueur L1 : {L[0]:.4f} m\n'
                     f'\n'
                     f'  Longueur L2 : {L[1]:.4f} m\n'
                     f'\n'
                     f'  Longueur L3 : {L[2]:.4f} m\n'
                     f'\n'
                     f'  Course C1 : {C[0]:.4f} m\n'
                     f'\n'
                     f'  Course C2 : {C[1]:.4f} m\n'
                     f'\n'
                     f'  Course C3 : {C[2]:.4f} m'
                     f'\n')
        props = dict(boxstyle='round', facecolor='white', alpha=0.15)
        ax.text2D(-0.26, 0.475, info_text, transform=ax.transAxes, fontsize=15, weight='book', fontstyle='italic', verticalalignment='center', bbox=props)   
        ax.legend(loc=(-0.265, 0.09),prop={'size': 15, 'weight': 'book', 'style': 'italic'})   
        
    # Génere un dossier, si il n'existe pas, au même emplacement que le fichier du programme  
    resultat_animation = 'Résultat trajectoire P-3DoF Vérins'
    if not os.path.exists(resultat_animation):
        os.makedirs(resultat_animation)
    
    # Chemin d'accés pour sauvegarder le GIF de l'animation générer
    gif_path = os.path.join(resultat_animation, 'Trajectoire P-3DoF Vérins.gif')
        
    # Création de l'animation
    ani = FuncAnimation(fig, update, frames=range(steps), repeat=False)
    ani.save(gif_path, writer='pillow', fps=40)
    plt.show()

trajectoire_vérins(roulis_amp=15, tangage_amp=15, hauteur=800e-3, steps=360)