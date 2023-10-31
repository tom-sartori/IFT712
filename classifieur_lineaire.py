# -*- coding: utf-8 -*-

import numpy as np
import sys
import solution_classifieur_lineaire as solution
import gestion_donnees as gd

#################################################
# Execution en tant que script dans un terminal
#
# Exemple:
# python classifieur_lineaire.py 1 280 280 0.001 1 1
#
#################################################


def main():

    if len(sys.argv) < 7:
        usage = "\n Usage: python classifieur.py method nb_train nb_test lambda bruit corruption don_ab\
        \n\n\t method : 1 => Classification generative\
        \n\t method : 2 => Perceptron + SDG \n\t method : 3 => Perceptron + SDG [sklearn]\
        \n\t nb_train, nb_test : nombre de donnees d'entrainement et de test\
        \n\t lambda >=0\
        \n\t bruit : multiplicateur de la matrice de variance-covariance (entre 0.1 et 50)\
        \n\t don_ab : production ou non de données aberrantes (0 ou 1) \
        \n\n\t ex : python classifieur_lineaire.py 1 280 280 0.001 1 1"
        print(usage)
        return

    method = int(sys.argv[1])
    nb_train = int(sys.argv[2])
    nb_test = int(sys.argv[3])
    lamb = float(sys.argv[4])
    bruit = float(sys.argv[5])
    donnees_aberrantes = bool(int(sys.argv[6]))

    print("Generation des données d'entrainement...")

    gestionnaire_donnees = gd.GestionDonnees(donnees_aberrantes, nb_train, nb_test, bruit)
    [x_train, t_train, x_test, t_test] = gestionnaire_donnees.generer_donnees()

    classifieur = solution.ClassifieurLineaire(lamb, method)

    # Entraînement de la classification linéaire
    classifieur.entrainement(x_train, t_train)

    # Prédictions sur les ensembles d'entraînement et de test
    predictions_entrainement = np.array([classifieur.prediction(x) for x in x_train])
    print("Erreur d'entrainement = ", 100*np.sum(np.abs(predictions_entrainement-t_train))/len(t_train), "%")

    predictions_test = np.array([classifieur.prediction(x) for x in x_test])
    print("Erreur de test = ", 100*np.sum(np.abs(predictions_test-t_test))/len(t_test), "%")

    # Affichage
    classifieur.afficher_donnees_et_modele(x_train, t_train, x_test, t_test)

if __name__ == "__main__":
    main()
