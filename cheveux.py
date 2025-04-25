import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# --------- Fonctions mathématiques ---------
def arctan(t, A, B, D, C_fixed):
    return A * np.arctan(B * (t - C_fixed)) + D


def kidder(t, t1, p1, t2, p2, t3, p3):
    t_data = np.array([t1, t2, t3])
    p_data = np.array([p1, p2, p3])
    C_fixed = t3

    # Bon point de départ basé sur une approximation analytique
    A0 = (p3 - p1) / np.pi
    B0 = 1.0 / (t2 - t1)
    D0 = p3 - A0 * np.arctan(B0 * (t3 - C_fixed))

    # Courbe avec C fixé
    def model_fixed_C(t, A, B, D):
        return arctan(t, A, B, D, C_fixed)

    # Fit rapide et plus stable
    popt, _ = curve_fit(model_fixed_C, t_data, p_data, p0=[A0, B0, D0], maxfev=3000)

    # Résultat final sur l'ensemble des t demandés
    return model_fixed_C(np.asarray(t), *popt)

def droite(t, t1, p1, t2, p2):
    if t1 == t2:
        return p1
    return p1 + (p2 - p1) * (t - t1) / (t2 - t1)

def demi_gaussienne_croissante(t, t1, p1, t2, p2, m):
    sigma = (t2 - t1) / (5 ** (1 / m))
    return p1 + (p2 - p1) * (1 - np.exp(-((t - t1) / sigma) ** m))

def gaussienne_ordre_m(t, t1, p1, sigma, m):
    alpha = sigma / (0.693 ** (1 / m))
    return p1 * np.exp(-((t - t1) / alpha) ** m)

def gaussienne_fwhm_ordre_m(t, t1, p1, fwhm, m): # fhwhm largeur à mi-hauteur
    alpha = fwhm / (2 * (np.log(2) ** (1 / m)))
    return p1 * np.exp(-((t - t1) / alpha) ** m)

# --------- Modèle principal ---------
def nucknuck(params):
    t_ini, t_fin, t_picket, p_picket, t_foot, t_plateau, p_plateau, t_kidder, p_kidder, t_debut_plateau_fin, t_plateau_fin, p_plateau_fin = params
    temps = np.linspace(0, t_fin + 0.5, 1000)
    res = []

    for t in temps:
        if t <= t_ini or t >= t_fin:
            puissance = 0
        elif t <= t_picket:
            puissance = gaussienne_fwhm_ordre_m(t, t_picket / 2, p_picket, 0.4, 2)
        elif t_plateau < t <= t_foot:
            puissance = droite(t, t_plateau, 0, t_foot, p_plateau)
        elif t_foot < t <= t_debut_plateau_fin:
            puissance = kidder(t, t_foot, p_plateau, t_kidder, p_kidder, t_debut_plateau_fin, p_plateau_fin)
        elif t_debut_plateau_fin < t <= t_plateau_fin:
            puissance = p_plateau_fin 
        elif t_plateau_fin < t :
            puissance = gaussienne_fwhm_ordre_m(t, t_plateau_fin, p_plateau_fin, 0.3, 4) 
        else:
            puissance = 0.0
        res.append(puissance)

    return np.array(res)

# --------- Paramètres initiaux ---------
t_ini = 0
t_fin = 16
t_picket = 1.25
p_picket = 80
t_plateau = 1.4
t_foot = 2.0
p_plateau = 13.0
t_kidder = 8.0
p_kidder = 60.0
t_debut_plateau_fin = 9.5
t_plateau_fin = t_fin
p_plateau_fin = 260.0

# --------- Fonction de variation ---------
def repart(X, delta, N):
# Generates N values evenly spaced around X ± delta
    Xs = np.linspace(X-delta, X+delta, N)
    return Xs

# --------- Définition des plages ---------
t_inis = repart(t_ini, 0,1)
t_fins = repart(t_fin, 0,1)
t_pickets = repart(t_picket, 0,1)
p_pickets = repart(p_picket, 0,1)
t_foots = repart(t_foot, 0,1)
t_plateaus = repart(t_plateau, 0,1)
p_plateaus = repart(p_plateau, 0,1)
t_kidders = repart(t_kidder, 0, 1)
p_kidders = repart(p_kidder, 0,1)
t_debut_plateau_fins = repart(t_debut_plateau_fin, 0,1)
t_plateau_fins = repart(t_plateau_fin, 0,1)
p_plateau_fins = repart(p_plateau_fin, 0,1)

# t_inis = repart(t_ini, 0,1)
# t_fins = repart(t_fin, 0,1)
# t_pickets = repart(t_picket, 0,1)
# p_pickets = repart(p_picket, 0,1)
# t_foots = repart(t_foot, 0,1)
# t_plateaus = repart(t_plateau, 0,1)
# p_plateaus = repart(p_plateau, 5,3)
# t_kidders = repart(t_kidder, 0.5, 3)
# p_kidders = repart(p_kidder, 20,3)
# t_debut_plateau_fins = repart(t_debut_plateau_fin, 1,5)
# t_plateau_fins = repart(t_plateau_fin, 0.,1)
# p_plateau_fins = repart(p_plateau_fin, 20,5)





# --------- Génération des combinaisons ---------
params_list = [
(t_ini, t_fin, t_picket, p_picket, t_foot, t_plateau, p_plateau,
t_kidder, p_kidder, t_debut_plateau_fin, t_plateau_fin, p_plateau_fin)
for t_ini in t_inis
for t_fin in t_fins
for t_picket in t_pickets
for p_picket in p_pickets
for t_foot in t_foots
for t_plateau in t_plateaus
for p_plateau in p_plateaus
for t_kidder in t_kidders
for p_kidder in p_kidders
for t_debut_plateau_fin in t_debut_plateau_fins
for t_plateau_fin in t_plateau_fins
for p_plateau_fin in p_plateau_fins
]

print(f"Nombre de cas : {len(params_list)}")

# --------- Traitement et sauvegarde ---------
def process(params, index):
    t_ini, t_fin, *_ = params
    temps = np.linspace(0, t_fin + 0.5, 1000)
    puissances = nucknuck(params)
    energie_totale_mj = np.trapz(puissances, temps * 1e-9) * 1e12 / 1e6

    # Plotting the result
    plt.plot(temps, puissances)
    plt.xlabel('Time (ns)')
    plt.ylabel('Power (TW)')
    #plt.title(f'Power vs Time' | Total Energy: {energie_totale_mj:.2f} MJ')
    #plt.legend()
    plt.grid(True)
    # # Save the normalized data (no header)
    directory_name = "./RES"
    os.makedirs(os.path.join(directory_name, f'rep_{index:04d}'), exist_ok=True)

    np.savetxt(
    f"{directory_name}/rep_{index:04d}/shot.dat",
    np.column_stack((temps * 1e-9, puissances)),
    fmt='%e',
    delimiter=' ',
    comments=''
    )

    # Processing all parameter sets
    idx = 0
    for p in params_list:
        process(p, idx)
    idx += 1
    print(int(idx/len(params_list)*100),"%")

    #plt.show()



# --------- Génération de courbes voisines pour le dernier cas traité ---------
# Affichage d’un effet "cheveux" aléatoire autour de la dernière courbe

n_cheveux = 50 # Nombre de courbes voisines
bruit_relatif = 0.15 # Variation relative (15%)
bruit_absolu = 2 # Variation absolue minimale (TW)
taille_fenetre_lissage = 201 # Plus grand = plus lisse
decalage_abscisse_max = 0.1 # Variation max en abscisse (ns)
scale = 0.7 # Plus de variations relatives à basse puissance

# Recalcul de la courbe principale pour affichage avec "cheveux"
derniere_params = params_list[-1]
t_ini, t_fin, *_ = derniere_params
temps = np.linspace(0, t_fin + 0.5, 1000)
puissances = nucknuck(derniere_params)

for _ in range(n_cheveux):
# --- Génère un bruit aléatoire ---
    bruit = np.random.normal(0, 1, size=puissances.shape)

# --- Applique un lissage gaussien ---
fenetre = np.exp(-np.linspace(-5, 5, taille_fenetre_lissage) ** 2)
fenetre /= np.sum(fenetre)
bruit_lisse = np.convolve(bruit, fenetre, mode='same')

# --- Modulateur de variation : + bruit à basse puissance, normal à haute ---
#attenuation = 1 / (1 + puissances / scale)
attenuation = (1 + scale*(np.max(puissances)-puissances) / (np.max(puissances)))
#print(np.max(puissances),attenuation)
variation_amplitude = np.maximum(bruit_relatif * attenuation * puissances, bruit_absolu)

# --- Renforcement local dans les zones à fort gradient ---
gradient = np.abs(np.gradient(puissances, temps))
gradient_boost = 1 + gradient / (np.max(gradient) + 1e-8)
bruit_lisse *= variation_amplitude * gradient_boost * 2

# --- Décalage temporel ---
decalage = np.random.uniform(-decalage_abscisse_max, decalage_abscisse_max)
temps_decales = np.abs(temps + decalage)

# --- Génère la courbe cheveu ---
courbe_voisine = np.abs(puissances + bruit_lisse)
plt.plot(temps_decales, courbe_voisine, color='red', alpha=0.3, linewidth=0.8)

# --- Courbe principale ---
plt.plot(temps, puissances, color='black', linewidth=2, label='Courbe principale')
plt.xlabel('Time (ns)')
plt.ylabel('Power (TW)')
plt.title('Courbe principale et variations voisines ("cheveux")')
plt.grid(True)
plt.legend()
plt.show()