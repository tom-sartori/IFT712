# -*- coding: utf-8 -*-

###
# |          Nom          | Matricule  |   CIP    |
# |:---------------------:|:----------:|:--------:|
# |   Alexandre Theisse   | 23 488 180 | thea1804 |
# | Louis-Vincent Capelli | 23 211 533 | capl1101 |
# |      Tom Sartori      | 23 222 497 | sart0701 |
###

import numpy as np
import matplotlib.pyplot as plt


class MAPnoyau:
    def __init__(self, lamb=0.2, sigma_square=1.06, b=1.0, c=0.1, d=1.0, M=2, noyau='rbf'):
        """
        Classe effectuant de la segmentation de données 2D 2 classes à l'aide de la méthode à noyau.

        lamb: coefficiant de régularisation L2
        sigma_square: paramètre du noyau rbf
        b, d: paramètres du noyau sigmoidal
        M,c: paramètres du noyau polynomial
        noyau: rbf, lineaire, olynomial ou sigmoidal
        """
        self.lamb = lamb
        self.a = None
        self.sigma_square = sigma_square
        self.M = M
        self.c = c
        self.b = b
        self.d = d
        self.noyau = noyau
        self.x_train = None

    def entrainement(self, x_train, t_train):
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable x_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.noyau'') et dont les parametres
        sont contenus dans les variables self.sigma_square, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'equation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.x_train``
        """
        # TODO : AJOUTER CODE ICI

        # x_train: un tableau 2D Numpy, où la n-ième rangée correspond à l'entrée x_n
        # t_train: un tableau 1D Numpy où le n-ième élément correspond à la cible t_n
        # self.noyau: 'rbf' | 'lineaire' | 'sigmoidal' | 'polynomial'
        # self.sigma_square

        # Params :
        # self.c
        # self.b
        # self.d
        # self.M

        # Poids de régularisation :
        # self.lamb

        # Assigner :
        # self.a = ( K + (\lamb * I_N) )^-1 * t         (6.8)
        # self.x_train

        self.x_train = x_train

        K = self.k(x=self.x_train, x_prime=self.x_train)
        I_N = np.identity(len(t_train))
        t = np.transpose(t_train)

        a = K + (self.lamb * I_N)
        a = np.linalg.inv(a)
        a = np.matmul(a, t)

        self.a = a  # ( K + (\lamb * I_N) )^-1 * t      (6.8)

    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        """
        # TODO : AJOUTER CODE ICI

        # k(x, x') = phi(x)^T * phi(x')     (6.1)
        # y(x)  = a^T * Phi * phi(x)        (6.9)
        #       = a^T * k
        # Return y(x) > 0.5 ? 1 : 0

        k = self.k(x=self.x_train, x_prime=x)
        y = np.matmul(np.transpose(self.a), k)

        return 1 if y > 0.5 else 0

    def erreur(self, t, prediction):
        """
        Retourne la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        """
        # TODO : AJOUTER CODE ICI

        # Return (t - prediction)^2

        return (t - prediction) ** 2

    def pourcentage_erreur(self, x_tab, t_tab) -> float:
        predictions = [self.prediction(x) for x in x_tab]  # [1, 1, 0, ....]
        nb_diff = (t_tab != predictions).sum()  # Nombre d'éléments tel que : t_train[i] != prediction_train[i]
        return (nb_diff / len(t_tab)) * 100  # (nb différents / nb total) * 100

    def validation_croisee(self, x_tab, t_tab):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_square``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` avec une validation croisée de type "k-fold" où k=10 avec les
        données contenues dans x_tab et t_tab.  Une fois les meilleurs hyperparamètres
        trouvés, le modèle est entraîné une dernière fois.

        SUGGESTION: Les valeurs de ``self.sigma_square`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        """
        # TODO : AJOUTER CODE ICI

        print("Début validation croisée. Peut être long. Possible de changer la variable \"num\", à la ligne 165 pour "
              "aller plus vite. ")

        data_shuffled = np.c_[x_tab, t_tab]
        np.random.shuffle(data_shuffled)  # Mélange l'ordre des données.
        x_tab = data_shuffled[:, :-1]
        t_tab = data_shuffled[:, -1]

        k = 10
        best_err_test = float("inf")  # On veut minimiser l'erreur, donc on initialise au max float.
        best_lamb = self.lamb
        best_sigma_square = self.sigma_square
        best_c = self.c
        best_M = self.M
        best_b = self.b
        best_d = self.d

        # self.lamb: [0.000000001, 2]               Global.
        # self.sigma_square: [0.000000001, 2]       Rbf.
        # self.c [0, 5]                             Polynomial.
        # self.M: [2, 6]                            Polynomial.
        # self.b: [0.00001, 0.01]                   Sigmoidal.
        # self.d: [0.00001, 0.01]                   Sigmoidal.

        num = 15  # Nombre de valeurs à tester pour chaque paramètre.
        for lamb in np.linspace(start=0.000000001, stop=2, num=num):
            self.lamb = lamb

            if self.noyau == 'rbf':
                for sigma_square in np.linspace(start=0.000000001, stop=2, num=num):
                    self.sigma_square = sigma_square
                    err_mean = self.get_validation_croisee_erreur_moyenne(x_tab, t_tab, k)
                    if err_mean < best_err_test:  # Si on a trouvé une meilleure moyenne d'erreurs.
                        best_err_test = err_mean
                        best_lamb = lamb
                        best_sigma_square = sigma_square

            elif self.noyau == 'lineaire':
                err_mean = self.get_validation_croisee_erreur_moyenne(x_tab, t_tab, k)
                if err_mean < best_err_test:  # Si on a trouvé une meilleure moyenne d'erreurs.
                    best_err_test = err_mean
                    best_lamb = lamb

            elif self.noyau == 'polynomial':
                for c in np.linspace(start=0, stop=5, num=num):
                    self.c = c
                    for M in np.linspace(start=2, stop=6, num=num):
                        self.M = M
                        err_mean = self.get_validation_croisee_erreur_moyenne(x_tab, t_tab, k)
                        if err_mean < best_err_test:  # Si on a trouvé une meilleure moyenne d'erreurs.
                            best_err_test = err_mean
                            best_lamb = lamb
                            best_c = c
                            best_M = M

            elif self.noyau == 'sigmoidal':
                for b in np.linspace(start=0.00001, stop=0.01, num=num):
                    self.b = b
                    for d in np.linspace(start=0.00001, stop=0.01, num=num):
                        self.d = d
                        err_mean = self.get_validation_croisee_erreur_moyenne(x_tab, t_tab, k)
                        if err_mean < best_err_test:
                            best_err_test = err_mean
                            best_lamb = lamb
                            best_b = b
                            best_d = d
            else:
                print("Noyau inconnu. ")
                return

        self.lamb = best_lamb
        self.sigma_square = best_sigma_square
        self.c = best_c
        self.M = best_M
        self.b = best_b
        self.d = best_d

        self.entrainement(x_tab, t_tab)

    def affichage(self, x_tab, t_tab):

        # Affichage
        ix = np.arange(x_tab[:, 0].min(), x_tab[:, 0].max(), 0.1)
        iy = np.arange(x_tab[:, 1].min(), x_tab[:, 1].max(), 0.1)
        iX, iY = np.meshgrid(ix, iy)
        x_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
        contour_out = np.array([self.prediction(x) for x in x_vis])
        contour_out = contour_out.reshape(iX.shape)

        plt.contourf(iX, iY, contour_out > 0.5)
        plt.scatter(x_tab[:, 0], x_tab[:, 1], s=(t_tab + 0.5) * 100, c=t_tab, edgecolors='y')
        plt.show()

    def k(self, x, x_prime):
        if self.noyau == 'rbf':
            # k(x, x') = exp{ - (||x - x'||^2) / (2 * sigma^2) }
            k = np.linalg.norm(x - x_prime, axis=1) ** 2
            k = k / (2 * self.sigma_square)
            k = - k
            k = np.exp(k)

        elif self.noyau == 'lineaire':
            # k(x, x') = x^T * x'
            k = np.matmul(x_prime, np.transpose(x))

        elif self.noyau == 'polynomial':
            # k(x, x') = (x^T * x' + c)^M
            k = np.matmul(x_prime, np.transpose(x))
            k = k + self.c
            k = k ** self.M

        elif self.noyau == 'sigmoidal':
            # k(x, x') = tanh(b * x^T * x' + d)
            k = self.b * np.transpose(x)
            k = np.matmul(x_prime, k)
            k = k + self.d
            k = np.tanh(k)

        else:
            print("Noyau inconnu. ")
            return

        return k

    def get_validation_croisee_erreur_moyenne(self, x_tab, t_tab, k) -> float:
        n = len(t_tab)
        fold_size = n // k
        err_mean = 0

        for i in range(k):
            inf_test = i * fold_size
            sup_test = inf_test + fold_size

            x_train = np.delete(x_tab, np.s_[inf_test:sup_test], axis=0)  # Enleve la partie de test.
            t_train = np.delete(t_tab, np.s_[inf_test:sup_test])  # Enleve la partie de test.

            x_test = x_tab[inf_test:sup_test, :]  # Données de test contenues entre les bornes.
            t_test = t_tab[inf_test:sup_test]  # Données de test contenues entre les bornes.

            self.entrainement(x_train=x_train, t_train=t_train)
            err_mean += self.pourcentage_erreur(x_test, t_test)  # Ajoute erreur de test à la moyenne.

        return err_mean / k
