# -*- coding: utf-8 -*-

###
# |          Nom          | Matricule  |   CIP    |
# |:---------------------:|:----------:|:--------:|
# |   Alexandre Theisse   | 23 488 180 | thea1804 |
# | Louis-Vincent Capelli | 23 211 533 | capl1101 |
# |      Tom Sartori      | 23 222 497 | sart0701 |
###

"""
Execution dans un terminal

Exemple:
   python non_lineaire_classification.py rbf 100 200 0 0
"""

import numpy as np
import sys
from map_noyau import MAPnoyau
import gestion_donnees as gd


def analyse_erreur(err_train, err_test):
    """
    Fonction qui affiche un WARNING lorsqu'il y a apparence de sur ou de sous
    apprentissage
    """
    # TODO : AJOUTER CODE ICI

    # Si err_train très petit et err_test élevé, alors sur-apprentissage.
    # Si err_train élevé et err_test élevé, alors sous-apprentissage.

    precision = 10

    if precision < err_test - err_train:
        print("Attention : sur-apprentissage. ")
    elif precision < err_train and precision < err_test:
        print("Attention : sous-apprentissage. ")


def main():
    if len(sys.argv) < 6:
        usage = "\n Usage: python non_lineaire_classification.py type_noyau nb_train nb_test lin validation\
        \n\n\t type_noyau: rbf, lineaire, polynomial, sigmoidal\
        \n\t nb_train, nb_test: nb de donnees d'entrainement et de test\
        \n\t lin : 0: donnees non lineairement separables, 1: donnees lineairement separable\
        \n\t validation: 0: pas de validation croisee,  1: validation croisee\n"
        print(usage)
        return

    type_noyau = sys.argv[1]
    nb_train = int(sys.argv[2])
    nb_test = int(sys.argv[3])
    lin_sep = int(sys.argv[4])
    vc = bool(int(sys.argv[5]))

    # On génère les données d'entraînement et de test
    generateur_donnees = gd.GestionDonnees(nb_train, nb_test, lin_sep)
    [x_train, t_train, x_test, t_test] = generateur_donnees.generer_donnees()

    # On entraine le modèle
    mp = MAPnoyau(noyau=type_noyau)

    if vc is False:
        mp.entrainement(x_train, t_train)
    else:
        mp.validation_croisee(x_train, t_train)

    # TODO :  ~= À MODIFIER =~.
    # AJOUTER CODE AFIN DE CALCULER L'ERREUR D'APPRENTISSAGE
    # ET DE VALIDATION EN % DU NOMBRE DE POINTS MAL CLASSES
    prediction_train = [mp.prediction(x) for x in x_train]  # [1, 1, 0, ....]
    nb_diff_train = (t_train != prediction_train).sum()  # Nombre d'éléments tel que : t_train[i] != prediction_train[i]
    err_train = (nb_diff_train / len(t_train)) * 100  # (nb différents / nb total) * 100

    prediction_test = [mp.prediction(x) for x in x_test]
    nb_diff_test = (t_test != prediction_test).sum()
    err_test = (nb_diff_test / len(t_test)) * 100

    print('Erreur train = ', err_train, '%')
    print('Erreur test = ', err_test, '%')
    analyse_erreur(err_train, err_test)

    # Affichage
    mp.affichage(x_test, t_test)

if __name__ == "__main__":
    main()
