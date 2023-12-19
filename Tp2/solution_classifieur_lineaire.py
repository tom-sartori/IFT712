# -*- coding: utf-8 -*-

###
# |          Nom          | Matricule  |   CIP    |
# |:---------------------:|:----------:|:--------:|
# |   Alexandre Theisse   | 23 488 180 | thea1804 |
# | Louis-Vincent Capelli | 23 211 533 | capl1101 |
# |      Tom Sartori      | 23 222 497 | sart0701 |
###

import numpy as np
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt


class ClassifieurLineaire:
    def __init__(self, lamb, methode):
        """
        Algorithmes de classification lineaire

        L'argument ``lamb`` est une constante pour régulariser la magnitude
        des poids w et w_0

        ``methode`` :   1 pour classification generative
                        2 pour Perceptron
                        3 pour Perceptron sklearn
        """
        self.w = np.array([1., 2.]) # paramètre aléatoire
        self.w_0 = -5.              # paramètre aléatoire
        self.lamb = lamb
        self.methode = methode

    def entrainement(self, x_train, t_train):
        """
        Entraîne deux classifieurs sur l'ensemble d'entraînement formé des
        entrées ``x_train`` (un tableau 2D Numpy) et des étiquettes de classe cibles
        ``t_train`` (un tableau 1D Numpy).

        Lorsque self.method = 1 : implémenter la classification générative de
        la section 4.2.2 du libre de Bishop. Cette méthode doit calculer les
        variables suivantes:

        - ``p`` scalaire spécifié à l'équation 4.73 du livre de Bishop.

        - ``mu_1`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.75 du livre de Bishop.

        - ``mu_2`` vecteur (tableau Numpy 1D) de taille D, tel que spécifié à
                    l'équation 4.76 du livre de Bishop.

        - ``sigma`` matrice de covariance (tableau Numpy 2D) de taille DxD,
                    telle que spécifiée à l'équation 4.78 du livre de Bishop,
                    mais à laquelle ``self.lamb`` doit être ADDITIONNÉ À LA
                    DIAGONALE (comme à l'équation 3.28).

        - ``self.w`` un vecteur (tableau Numpy 1D) de taille D tel que
                    spécifié à l'équation 4.66 du livre de Bishop.

        - ``self.w_0`` un scalaire, tel que spécifié à l'équation 4.67
                    du livre de Bishop.

        lorsque method = 2 : Implementer l'algorithme de descente de gradient
                        stochastique du perceptron avec 1000 iterations

        lorsque method = 3 : utiliser la librairie sklearn pour effectuer une
                        classification binaire à l'aide du perceptron

        """
        if self.methode == 1:  # Classification generative
            print('Classification generative')
            # TODO : AJOUTER CODE ICI
            # Calcul de N1 et N2
            N1 = 0
            N2 = 0
            for i in range(len(t_train)):
                if t_train[i] == 1:
                    # classe 1
                    N1 += 1
                else:
                    # classe 2
                    N2 += 1
            
            # Calcul de p
            p = N1 / (N1 + N2)

            # Calcul de mu1 et mu2
            mu1 = np.zeros(len(x_train[0]))
            mu2 = np.zeros(len(x_train[0]))
            for i in range(len(t_train)):
                if t_train[i] == 1:
                    # classe 1
                    mu1 += x_train[i]
                else:
                    # classe 2
                    mu2 += x_train[i]
            mu1 /= N1
            mu2 /= N2

            # Calcul de sigma
            sigma = np.zeros((len(x_train[0]), len(x_train[0])))
            S1 = np.zeros((len(x_train[0]), len(x_train[0])))
            S2 = np.zeros((len(x_train[0]), len(x_train[0])))
            for i in range(len(t_train)):
                if t_train[i] == 1:
                    # classe 1
                    S1 += np.dot(np.transpose([x_train[i] - mu1]), [x_train[i] - mu1])
                else:
                    # classe 2
                    S2 += np.dot(np.transpose([x_train[i] - mu2]), [x_train[i] - mu2])
            sigma = N1 / (N1 + N2) * S1 + N2 / (N1 + N2) * S2
            sigma += self.lamb * np.identity(len(x_train[0]))

            # Calcul de w et w_0
            self.w = np.dot(np.linalg.inv(sigma), mu1 - mu2)
            self.w_0 = -1 / 2 * np.dot(np.dot(np.transpose(mu1), np.linalg.inv(sigma)), mu1) + 1 / 2 * np.dot(np.dot(np.transpose(mu2), np.linalg.inv(sigma)), mu2) + np.log(p / (1 - p))

            print('w = ', self.w, 'w_0 = ', self.w_0, '\n')


        elif self.methode == 2:  # Perceptron + SGD, learning rate = 0.001, nb_iterations_max = 1000
            print('Perceptron')

            # TODO : AJOUTER CODE ICI
            learning_rate = 0.001
            nb_iterations_max = 1000

            # Remplace 0 par -1 et laisse les 1.
            t_train_bis = np.where(t_train == 0, -1, t_train)

            for i in range(nb_iterations_max):
                mauvaise_classe = False
                for j in range(len(t_train)):
                    if self.erreur(t=t_train[j], prediction=self.prediction(x_train[j])):
                        # Si erreur alors mal classé et on met à jour les poids.
                        mauvaise_classe = True
                        self.w = self.w + learning_rate * (x_train[j] * t_train_bis[j] + self.lamb * self.w)
                        self.w_0 = self.w_0 + learning_rate * (t_train_bis[j] + self.lamb * self.w_0)

                if not mauvaise_classe:
                    break

        else:  # Perceptron + SGD [sklearn] + learning rate = 0.001 + penalty 'l2' voir http://scikit-learn.org/
            print('Perceptron [sklearn]')

            # TODO : AJOUTER CODE ICI
            learning_rate = 0.001
            penalty = 'l2'

            clf = Perceptron(eta0=learning_rate, penalty=penalty, alpha=self.lamb)
            clf.fit(X=x_train, y=t_train)
            self.w = clf.coef_[0]
            self.w_0 = clf.intercept_[0]

        print('w = ', self.w, 'w_0 = ', self.w_0, '\n')

    def prediction(self, x):
        """
        Retourne la prédiction du classifieur lineaire.  Retourne 1 si x est
        devant la frontière de décision et 0 sinon.

        ``x`` est un tableau 1D Numpy

        Cette méthode suppose que la méthode ``entrainement()``
        a préalablement été appelée. Elle doit utiliser les champs ``self.w``
        et ``self.w_0`` afin de faire cette classification.
        """

        # TODO : AJOUTER CODE ICI
        # Frontière: y = (a * x) + b <=> y = (self.w * x) + self.w_0 = 0
        y = np.dot(self.w, x) + self.w_0
        return 1 if y >= 0 else 0

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de classification, i.e.
        1. si la cible ``t`` et la prédiction ``prediction``
        sont différentes, 0. sinon.
        """

        # TODO : AJOUTER CODE ICI
        return 0 if t == prediction else 1

    def afficher_donnees_et_modele(self, x_train, t_train, x_test, t_test):
        """
        afficher les donnees et le modele

        x_train, t_train : donnees d'entrainement
        x_test, t_test : donnees de test
        """
        plt.figure(0)
        plt.scatter(x_train[:, 0], x_train[:, 1], s=t_train * 100 + 20, c=t_train)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Training data')

        plt.figure(1)
        plt.scatter(x_test[:, 0], x_test[:, 1], s=t_test * 100 + 20, c=t_test)

        pente = -self.w[0] / self.w[1]
        xx = np.linspace(np.min(x_test[:, 0]) - 2, np.max(x_test[:, 0]) + 2)
        yy = pente * xx - self.w_0 / self.w[1]
        plt.plot(xx, yy)
        plt.title('Testing data')

        plt.show()

    def parametres(self):
        """
        Retourne les paramètres du modèle
        """
        return self.w_0, self.w
