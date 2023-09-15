# -*- coding: utf-8 -*-

import numpy as np
import sys
import solution_regression as sr
import gestion_donnees as gd
import matplotlib.pyplot as plt


def warning(erreur_test, erreur_apprentissage, bruit):
    """
    Fonction qui affiche un WARNING à l'ecran lorsque les erreurs obtenues en fonction du bruit
    indique une possibilite de sur- ou de sous-apprentissage

    erreur_test: erreur obtenue sur l'ensemble de test
    erreur_apprentissage: erreur obtenue sur l'ensemble d'apprentissage
    bruit: magnitude du bruit
    """
    # AJOUTER CODE ICI
    # Écrivez des conditions simples, avec des valeurs approximatives "harcodées",
    # qui vérifient si nous sommes en présence de sur- ou sous-apprentissage.

################################
# Execution en tant que script 
#
# tapper python3 regression.py 1 sin 20 20 0.3 10 0.001
#
# dans un terminal
################################


def main():
    
    if len(sys.argv) < 8:
        print("Usage: python regression.py sk modele_gen nb_train nb_test bruit M lambda\n")
        print("\t sk=0: using_sklearn=False, sk=1: using_sklearn=True")
        print("\t modele_gen=lineaire, sin ou tanh")
        print("\t nb_train: nombre de donnees d'entrainement")
        print("\t nb_test: nombre de donnees de test")
        print("\t bruit: amplitude du bruit appliqué aux données")
        print("\t M: degré du polynome de la fonction de base (recherche d'hyperparametre lorsque M<=0) ")
        print("\t lambda: lambda utilisé par le modele de Ridge\n")
        print(" exemple: python3 regression.py 1 sin 20 20 0.3 10 0.001\n")
        return
    
    skl = int(sys.argv[1]) > 0.5
    modele_gen = sys.argv[2]
    nb_train = int(sys.argv[3])
    nb_test = int(sys.argv[4])
    bruit = float(sys.argv[5])
    m = int(sys.argv[6])
    lamb = float(sys.argv[7])
    w = [0.3, 4.1]  # Parametres du modele generatif

    # Creer le gestionnaire de donnees et generer les donnees d'entraînement et de test
    gestionnaire_donnees = gd.GestionDonnees(w, modele_gen, nb_train, nb_test, bruit)
    [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()

    # Entrainement du modele de regression
    regression = sr.Regression(lamb, m)
    regression.entrainement(x_train, t_train, using_sklearn=skl)

    # Predictions sur les ensembles d'entrainement et de test
    predictions_train = np.array([regression.prediction(x) for x in x_train])
    predictions_test = np.array([regression.prediction(x) for x in x_test])

    # Calcul des erreurs
    erreurs_entrainement = np.array([regression.erreur(t_n, p_n)
                                     for t_n, p_n in zip(t_train, predictions_train)])
    erreurs_test = np.array([regression.erreur(t_n, p_n)
                             for t_n, p_n in zip(t_test, predictions_test)])

    print("Erreur d'entraînement :", "%.2f" % erreurs_entrainement.mean())
    print("Erreur de test :", "%.2f" % erreurs_test.mean())
    print("")

    warning(erreurs_test.mean(), erreurs_entrainement.mean(), bruit)

    # Affichage
    gestionnaire_donnees.afficher_donnees_et_modele(x_train, t_train, True)
    predictions_range = np.array([regression.prediction(x) for x in np.arange(0, 1, 0.01)])
    gestionnaire_donnees.afficher_donnees_et_modele(np.arange(0, 1, 0.01), predictions_range, False)

    if m >= 0:
        plt.suptitle('Resultat SANS recherche d\'hyperparametres')
    else:
        plt.suptitle('Resultat AVEC recherche d\'hyperparametres')
    plt.show()

if __name__ == "__main__":
    main()
