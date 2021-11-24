import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial.distance import pdist, squareform, cdist


class SVM():

    def __init__(self, X, y, C, gamma, kernel):

        self.X = X
        self.y = y
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self._generate_intermediate_variables()

    @staticmethod
    def _kernel_fun(x1, x2, kernel, gamma):
        """
        Kernel function
        """
        if kernel == 'polynomial':
            return (1 + np.dot(x1, x2.T)) ** gamma
        elif kernel == 'rbf':
            minus_matrix = []
            for i in range(len(x2)):
                minus_matrix.append(x1 - x2[i])
            minus_matrix = np.array(minus_matrix)
            return np.exp(-(np.linalg.norm(minus_matrix, ord=2, axis=2) * gamma) ** 2)
        else:
            raise Exception('Kernel function does not exist')

    def _generate_intermediate_variables(self):
        """
        Since we will be using cvxopt we have to generate an optimization problem of the following shape:

        min 0.5x^Px + q^Tx
        s.t. Gx <= h
             Ax = b

        where in our case the optimization will be of the shape:

        max \sum \alpha_i - 0.5\sum_{ij}\alpha_i\alpha_jy_iy_j<x_i, x_j>
        s.t \alpha_i >= 0
            \alpha_i <= C
            \sum \alpha_i y_i = 0

        We will have to adjust our notation accordingly

        """

        obs = len(self.X)
        self.K = self._kernel_fun(self.X, self.X, self.kernel, self.gamma)

        self.P = matrix(np.outer(self.y, self.y) * self.K)
        self.q = matrix(np.ones(obs) * -1)
        self.G = matrix(np.vstack((np.diag(np.ones(obs) * -1), np.identity(obs))))
        self.h = matrix(np.hstack((np.zeros(obs), np.ones(obs) * self.C)))
        self.A = matrix(self.y, (1, obs), 'd')
        self.b = matrix(np.zeros(1))

    def fit(self, tol=1e-4, fix_intercept=False):
        """
        This method performs the optimization over the alpha values using cvxopt

        """

        solvers.options['show_progress'] = False
        self.fit_sol = solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b)

        self.alpha = np.ravel(self.fit_sol['x'])

        self.w, self.bias = self.compute_params(alphas=self.alpha, tol=tol, fix_intercept=fix_intercept)

    def compute_params(self, alphas, tol, fix_intercept=False):
        """
        This method returns a set of parameters estimated based on the previous fit
        """

        self.sv_idx = (alphas > tol).reshape(len(self.X),)

        self.w = (self.y[self.sv_idx] * alphas[self.sv_idx]).T @ self.X[self.sv_idx]

        self.bias = 0
        if not fix_intercept:
            for i in range(sum(self.sv_idx)):
                self.bias += self.y[self.sv_idx][i] - np.sum(
                    (self.y[self.sv_idx] * alphas[self.sv_idx]).T @ self._kernel_fun(self.X[self.sv_idx],
                                                                           self.X[self.sv_idx][i],
                                                                           self.kernel,
                                                                           self.gamma).T)

            self.bias = self.bias / sum(self.sv_idx)

        return self.w, self.bias

    def pred(self, X, y=None):
        """
        Perform prediction and if y is available return prediction metrics

        """

        pred = np.sign(np.dot(X, self.w) + self.bias)

        if isinstance(y, np.ndarray):
            return sum(pred == y)/len(y)
        else:
            return pred
