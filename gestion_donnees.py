# -*- coding: utf-8 -*-

#####
# VotreNom (VotreMatricule) .~= À MODIFIER =~.
###

import numpy as np


class GestionDonnees:
    def __init__(self, donnees_aberrantes, nb_train, nb_test, bruit):
        self.donnees_aberrantes = donnees_aberrantes
        self.nb_train = nb_train
        self.nb_test = nb_test
        self.bruit = bruit

    def generer_donnees(self):
        """
        Fonction qui genere des donnees de test et d'entrainement.

        modele_gen : 'lineaire', 'sin' ou 'tanh'
        nb_train : nb de donnees d'entrainement
        nb_test : nb de donnees de test
        bruit : amplitude du bruit (superieur ou egale a zero
        """
        np.random.seed(12)  # commentez cette ligne pour tester differentes configurations
        mu1 = np.random.randn(2)
        mu2 = mu1 + np.random.randn(2)*10
        mu3 = mu2*4
        sigma1 = np.array([[4., 3.], [3., 4.]])
        sigma2 = np.array([[6., 4.], [4., 6.]])
        sigma3 = np.array([[1., 0.5], [0.5, 1.]])

        if self.donnees_aberrantes is True:
            nb_data = int(self.nb_train/3.0)
            x_1 = np.random.multivariate_normal(mu1, sigma1*self.bruit, nb_data)
            t_1 = np.ones(nb_data)
            x_2 = np.random.multivariate_normal(mu2, sigma2*self.bruit, nb_data)
            t_2 = np.zeros(nb_data)
            x_3 = np.random.multivariate_normal(mu3, sigma3*self.bruit, nb_data)
            t_3 = np.zeros(nb_data)
        else:
            nb_data = int(self.nb_train / 2.0)
            x_1 = np.random.multivariate_normal(mu1, sigma1*self.bruit, nb_data)
            t_1 = np.ones(nb_data)
            x_2 = np.random.multivariate_normal(mu2, sigma2*self.bruit, nb_data)
            t_2 = np.zeros(nb_data)

        # Fusionne toutes les données dans un seul ensemble d'entraînement
        x_train = np.vstack([x_1, x_2])
        t_train = np.hstack([t_1, t_2])
        if self.donnees_aberrantes is True:
            x_train = np.vstack([x_train, x_3])
            t_train = np.hstack([t_train, t_3])

        # Mélange dans un ordre aléatoire
        p = np.random.permutation(len(t_train))
        x_train = x_train[p, :]
        t_train = t_train[p]

        print("Generation des données de test...")
        nb_data = int(self.nb_test / 2.0)
        x_1 = np.random.multivariate_normal(mu1, sigma1*self.bruit, nb_data)
        t_1 = np.ones(nb_data)
        x_2 = np.random.multivariate_normal(mu2, sigma2*self.bruit, nb_data)
        t_2 = np.zeros(nb_data)

        # Fusionne toutes les données dans un seul ensemble de test
        x_test = np.vstack([x_1, x_2])
        t_test = np.hstack([t_1, t_2])

        return x_train, t_train, x_test, t_test

