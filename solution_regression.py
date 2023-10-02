# -*- coding: utf-8 -*-

###
# |          Nom          | Matricule  |   CIP    |
# |:---------------------:|:----------:|:--------:|
# |   Alexandre Theisse   | 23 488 180 | thea1804 |
# | Louis-Vincent Capelli | 23 211 533 | capl1101 |
# |      Tom Sartori      | 23 222 497 | sart0701 |
###

import random

import numpy as np
from sklearn import linear_model


class Regression:
    def __init__(self, lamb, m=1):
        self.lamb = lamb
        self.w = None
        self.M = m

    def fonction_base_polynomiale(self, x):
        """
        Fonction de base qui projette la donnee x vers un espace polynomial tel que mentionne au chapitre 3.
        Si x est un scalaire, alors phi_x sera un vecteur à self.M dimensions : (x^1,x^2,...,x^self.M)
        Si x est un vecteur de N scalaires, alors phi_x sera un tableau 2D de taille NxM

        NOTE : En mettant phi_x = x, on a une fonction de base lineaire qui fonctionne pour une regression lineaire
        """
        # AJOUTER CODE ICI

        if np.isscalar(x):
            # x is scalar.
            phi_x = np.ones(self.M + 1)
            for i in range(0, self.M + 1):
                phi_x[i] = x ** i
        elif x.ndim == 1:
            # x is a vector.
            phi_x = np.ones((x.shape[0], self.M + 1))
            for i in range(0, self.M + 1):
                phi_x[:, i] = x ** i
        else:
            raise ValueError("x is not a scalar or a vector.")

        return phi_x

    def recherche_hyperparametre(self, X, t):
        """
        Trouver la meilleure valeur pour l'hyper-parametre self.M (pour un lambda fixe donné en entrée).

        Option 1
        Validation croisée de type "k-fold" avec k=10. La méthode array_split de numpy peut être utlisée 
        pour diviser les données en "k" parties. Si le nombre de données en entrée N est plus petit que "k", 
        k devient égal à N. Il est important de mélanger les données ("shuffle") avant de les sous-diviser
        en "k" parties.

        Option 2
        Sous-échantillonage aléatoire avec ratio 80:20 pour Dtrain et Dvalid, avec un nombre de répétition k=10.

        Note: 

        Le resultat est mis dans la variable self.M

        X: vecteur de donnees
        t: vecteur de cibles
        """
        # AJOUTER CODE ICI

        # K-Fold
        k = 10
        tested_M = np.arange(1, k + 1)
        errors_valid = np.zeros(k)

        if len(X) < k:
            k = len(X)

        X_copy = X.copy()
        t_copy = t.copy()

        # Shuffle
        c = list(zip(X_copy, t_copy))
        random.shuffle(c)
        X_copy, t_copy = zip(*c)

        X_copy = np.array(X_copy)
        t_copy = np.array(t_copy)

        # Split
        X_split = np.array_split(X_copy, k)
        t_split = np.array_split(t_copy, k)

        # Hyperparameter optimization
        for i in range(0, k):
            X_train = np.concatenate(X_split[:i] + X_split[i + 1:])
            t_train = np.concatenate(t_split[:i] + t_split[i + 1:])
            X_valid = X_split[i]
            t_valid = t_split[i]
            self.M = tested_M[i]

            self.entrainement(X_train, t_train)

            predictions_valid = np.array([self.prediction(x) for x in X_valid])
            errors_valid[i] = np.array([self.erreur(t_n, p_n) for t_n, p_n in zip(t_valid, predictions_valid)]).mean()

        self.M = tested_M[np.argmin(errors_valid)]

    def entrainement(self, X, t, using_sklearn=False):
        """
        Entraîne la regression lineaire sur l'ensemble d'entraînement forme des
        entrees ``X`` (un tableau 2D Numpy, ou la n-ieme rangee correspond à l'entree
        x_n) et des cibles ``t`` (un tableau 1D Numpy ou le
        n-ieme element correspond à la cible t_n). L'entraînement doit
        utiliser le poids de regularisation specifie par ``self.lamb``.

        Cette methode doit assigner le champs ``self.w`` au vecteur
        (tableau Numpy 1D) de taille D+1, tel que specifie à la section 3.1.4
        du livre de Bishop.
        
        Lorsque using_sklearn=True, vous devez utiliser la classe "Ridge" de 
        la librairie sklearn (voir http://scikit-learn.org/stable/modules/linear_model.html)
        
        Lorsque using_sklearn=Fasle, vous devez implementer l'equation 3.28 du
        livre de Bishop. Il est suggere que le calcul de ``self.w`` n'utilise
        pas d'inversion de matrice, mais utilise plutôt une procedure
        de resolution de systeme d'equations lineaires (voir np.linalg.solve).

        Aussi, la variable membre self.M sert à projeter les variables X vers un espace polynomiale de degre M
        (voir fonction self.fonction_base_polynomiale())

        NOTE IMPORTANTE : lorsque self.M <= 0, il faut trouver la bonne valeur de self.M

        """
        # AJOUTER CODE ICI

        if self.M <= 0:
            self.recherche_hyperparametre(X, t)

        phi_x = self.fonction_base_polynomiale(X)

        if using_sklearn:
            clf = linear_model.Ridge(alpha=self.lamb)
            clf.fit(X=phi_x, y=t)
            self.w = clf.coef_
            self.w[0] = clf.intercept_
        else:
            self.w = np.linalg.solve(
                self.lamb * np.eye(self.M + 1) + np.dot(np.transpose(phi_x), phi_x),
                np.dot(np.transpose(phi_x), t)
            )

    def prediction(self, x):
        """
        Retourne la prediction de la regression lineaire
        pour une entree, representee par un tableau 1D Numpy ``x``.

        Cette methode suppose que la methode ``entrainement()``
        a prealablement ete appelee. Elle doit utiliser le champs ``self.w``
        afin de calculer la prediction y(x,w) (equation 3.1 et 3.3).
        """
        # AJOUTER CODE ICI

        return np.dot(np.transpose(self.w), self.fonction_base_polynomiale(x))

    @staticmethod
    def erreur(t, prediction):
        """
        Retourne l'erreur de la difference au carre entre
        la cible ``t`` et la prediction ``prediction``.
        """
        # AJOUTER CODE ICI

        return (t - prediction) ** 2
