from os import supports_effective_ids
import numpy as np
from cvxopt import matrix, solvers


def rbf(x1, x2, gamma):
    """
    Compute the RBF function

    :param x1: observations
    :param x2: centers of the RBF function
    :param gamma: strictly positive

    :return rbf(x1)
    """
    minus_matrix = x1 - np.expand_dims(x2, axis=1)
    return np.exp(-gamma * (np.linalg.norm(minus_matrix, ord=2, axis=2)**2)).T


def polynomial(x1, x2, gamma):
    """
    Compute the polynomial function

    :param x1: observations
    :param x2: support vectors
    :param gamma: strictly positive

    :return polynomial(x1)
    """
    return (1 + np.dot(x1, x2.T)) ** gamma


class SVM():
    # class attribute
    kernel_functions = {'poly':polynomial, # convenient alias which matches the Sklearn API
                        'polynomial':polynomial,
                        'rbf':rbf}

    def __init__(self, X, y, C, gamma, kernel):

        self.X = X
        self.y = y
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        # dynamically set the kernel function 
        try:
            self._kernel_fun = self.kernel_functions[self.kernel]
        except:
            raise NotImplementedError
        self._generate_intermediate_variables()

    @property
    def state(self):
        """
        Get the state of the SVM object
        """
        state_dict = {'kernel':self.kernel,
                      'gamma':self.gamma,
                      'C':self.C}
        return state_dict

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
        self.K = self._kernel_fun(self.X, self.X, self.gamma)

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
        self.idx = np.arange(len(alphas))[self.sv_idx]

        self.w = (self.y[self.sv_idx] * alphas[self.sv_idx]).T @ self.X[self.sv_idx]

        self.bias = 0
        if not fix_intercept:
            # for i in range(np.sum(self.sv_idx)):
            #     self.bias += self.y[self.sv_idx][i] - np.sum(
            #         self.y[self.sv_idx] * alphas[self.sv_idx] * self.K[self.idx[i], self.sv_idx])

            # self.bias = self.bias / np.sum(self.sv_idx)
            self.bias = np.sum(self.y[self.sv_idx] - np.sum(
                self.y[self.sv_idx] * alphas[self.sv_idx] * self.K[np.ix_(self.idx, self.sv_idx)], axis=1))
            self.bias /= np.sum(self.sv_idx)

        return self.w, self.bias

    # TODO: Improve this part a lot...
    def pred(self, X):
        """
        Perform prediction and if y is available return prediction metrics

        """
        K_test = self._kernel_fun(X, self.X[self.sv_idx], self.gamma) # self.X[self.sv_idx] are the support vectors 
        prediction = np.sum(self.y[self.sv_idx] * self.alpha[self.sv_idx] * K_test,
                            axis=1) + self.bias
        y_pred = np.sign(prediction)
        return y_pred

    def eval(self, X, y):
        """
        Return prediction metrics
        """
        y_pred = self.pred(X)
        return np.sum(y_pred == y) / len(y)


class MultiSVM(SVM):
    """
    We will chose the one-vs-one since we have very few classes and in this manner we avoid the unbalance of one-vs-all
    Make a nice implemenation for this part, but no sure how to do it really...The code that runs is present in the notebook
    """

