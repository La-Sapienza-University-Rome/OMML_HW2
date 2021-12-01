import numpy as np
from cvxopt import matrix, solvers
import itertools
from  collections import Counter


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


class SVM:
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





class SVMDecomposition(SVM):

    def __init__(self, X, y, C, gamma, kernel):
        super().__init__(X, y, C, gamma, kernel)


    # def _select_working_set(self, q, iteration):
    #     """
    #     """
    #     # if iteration == 0:
    #     #     return 1, 0, np.random.choice(np.arange(len(self.y)), q)
        
    #     R = np.logical_or(
    #         np.logical_and(self.alpha < self.C, self.y == 1),
    #         np.logical_and(self.alpha > 0, self.y == -1)
    #     )
    #     S = np.logical_or(
    #         np.logical_and(self.alpha < self.C, self.y == -1),
    #         np.logical_and(self.alpha > 0, self.y == 1)
    #     )
    #     I = np.argpartition(-self.y[R] * self.gradients[R], -q//2)[-q//2:][::-1] # equivalent to argmax
    #     J = np.argpartition(-self.y[S] * self.gradients[S], q//2)[:q//2] # equivalent to argmin
    #     m_a = -self.y[I[0]] * self.gradients[I[0]]; M_a = -self.y[J[0]] * self.gradients[J[0]]
    #     working_set = np.concatenate((I,J))
    #     return m_a, M_a, working_set


    def _select_working_set(self, q, iteration):
        """
        """
        L = self.alpha == 0
        U = self.alpha == self.C
        L_plus = L & (self.y > 0)
        L_minus = L & (self.y < 0)
        U_plus = U & (self.y > 0)
        U_minus = U & (self.y < 0)
        R = L_plus | U_minus | ( (self.alpha > 0) & (self.alpha < self.C) )
        S = L_minus | U_plus | ( (self.alpha > 0) & (self.alpha < self.C) )
        I = np.argsort(- (-self.y[R] * self.gradients[R]) )[:q//2] # argmax
        J = np.argsort(  (-self.y[S] * self.gradients[S]) )[:q//2] # argmin
        working_set = np.concatenate( (I, J) )
        # m_a = -self.y[I[0]] * self.gradients[I[0]] # max i
        # M_a = -self.y[J[0]] * self.gradients[J[0]] # min j
        return working_set


    # TODO: implement efficient cache
    def _hessian_handler(self, working_set, working_set_size):
        """
        """
        # Q = np.empty((working_set_size, working_set_size))
        # I = working_set[:working_set_size//2]; J = working_set[working_set_size//2:]
        # for idx_i, i in enumerate(I):
        #     for idx_j, j in enumerate(J): 
        #         if (i,j) not in self._kernel_cache.keys():
        #             self._kernel_cache[(i,j)] = self._kernel_fun(self.X[i, np.newaxis], self.X[j, np.newaxis], self.gamma)
        #         Q[idx_i:idx_i+working_set_size//2, idx_j:idx_j+working_set_size//2] = self._kernel_cache[(i,j)]
       
        return self._hessian_cache[working_set[:,np.newaxis], working_set[np.newaxis,:]]


    def _solve_subproblem(self, working_set, working_set_size):
        """
        """
        P = self._hessian_handler(working_set, working_set_size)
        q = matrix(np.ones(working_set_size) * -1)
        G = matrix(np.vstack((np.diag(np.ones(working_set_size) * -1), np.identity(working_set_size))))
        h = matrix(np.hstack((np.zeros(working_set_size), np.ones(working_set_size) * self.C)))
        A = matrix(self.y[working_set], (1, working_set_size), 'd')
        b = matrix(np.zeros(1))
        # print(P.shape)
        return P, solvers.qp(matrix(P), q, G, h, A, b)


    """
    self.q = matrix(np.ones(obs) * -1)
        self.G = matrix(np.vstack((np.diag(np.ones(obs) * -1), np.identity(obs))))
        self.h = matrix(np.hstack((np.zeros(obs), np.ones(obs) * self.C)))
        self.A = matrix(self.y, (1, obs), 'd')
        self.b = matrix(np.zeros(1))
    """


    def fit(self, working_set_size, max_iters=500, tol=1e-4, fix_intercept=False):
        """

        """
        solvers.options['show_progress'] = False
        yi_yj = np.outer(self.y, self.y)
        self._hessian_cache = yi_yj * self._kernel_fun(self.X, self.X, self.gamma) # change into an efficient cache
        self.alpha = np.zeros(self.y.shape)
        self.gradients = np.full(self.y.shape, fill_value=-1)
        i = 0; m_a = 1; M_a = 0
        while  (i < max_iters) and (m_a - M_a > 0):
            working_set = self._select_working_set(working_set_size, iteration=i)
            P, fit_sol = self._solve_subproblem(working_set, working_set_size)
            self.gradients = self.gradients + np.squeeze(self._hessian_cache[:,working_set] @ ((np.ravel(fit_sol['x']) - self.alpha[working_set])[:,np.newaxis]))
            self.alpha[working_set] = np.ravel(fit_sol['x'])
            i += 1
            m_a = -self.y[working_set[0]] * self.gradients[working_set[0]] # max i
            M_a = -self.y[working_set[0]] * self.gradients[working_set[0]] # min j
        self.w, self.bias = self.compute_params(alphas=self.alpha, tol=tol, fix_intercept=fix_intercept)




        



def encode(y, letters):
    """
    Encode the labels y in {-1, 1}.

    :param y: labels to encode
    :param letters: list or tuple of length 2
    
    :return {-1, 1} encoding
    """
    return np.where(y == letters[0], -1, 1)


def decode(y_pred, letters):
    """
    Decode the target values y_pred.

    :param y_pred: target values in {-1, 1}
    :param letters: class labels

    :return decoded y
    """
    mapping = {n:l for l,n in zip(letters, (-1,1))}
    return np.where(y_pred == -1, mapping[-1], mapping[1])


def split_X_y(df):
    """
    Split the DataFrame into X and y.

    :param df: DataFrame

    :return X, y
    """
    X = df.drop(['letter'], axis=1).to_numpy(copy=True)
    y = df['letter'].to_numpy(copy=True)
    return X, y


def process_df(df, letters):
    """
    Filter the rows belonging to the classes in letters and return processed X, y.

    :param df: DataFrame
    :param letters: class labels

    :return X, y
    """
    mask = df.letter.isin(letters)
    X, y = split_X_y(df[mask])
    y = encode(y, letters)
    return X, y



class MultiSVM():

    def __init__(self, df,  C, gamma, kernel):
        """
        This class performs multi-class classification, providing the same API of the SVM class.

        :param df: raw pandas.DataFrame containing features and labels
        :param C: same as SVM()
        :param gamma: same as SVM()
        :param kernel: same as SVM()
        """
        self.classes = df['letter'].unique().tolist()
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.df = df


    def fit(self, tol=1e-4, fix_intercept=False):
        """
        Generate as many SVM classifiers as the combinations of the classes (one-vs-one approach) 
        and fit them

        :param tol: same as SVM()
        :param fix_intercept: same as SVM()
        """
        self._classifiers = {}
        for letters_pair in itertools.combinations(self.classes, r=2):
            X, y = process_df(self.df, letters_pair)
            self._classifiers[letters_pair] = SVM(X, y, self.C, self.gamma, self.kernel)
            self._classifiers[letters_pair].fit(tol, fix_intercept)

    
    def pred(self, X):
        """
        Return the predictions given the observations X

        :param X: features without labels; numpy.ndarray

        :return predictions; np.ndarray
        """
        pred_list = []
        for letters_pair in self._classifiers.keys():
            y_pred_pair = self._classifiers[letters_pair].pred(X)
            pred_list.append(decode(y_pred_pair, letters_pair))
        votes = zip(*pred_list)
        y_pred = np.fromiter(map(lambda x: Counter(x).most_common(1)[0][0], votes), dtype='<U1')
        return y_pred


    def eval(self, X, y):
        """
        Predict the class of the observations X and compare with the ground-truth y.
        Return the accuracy score.

        :param X: observations; numpy.ndarray
        :param y: ground-truth labels

        :return accuracy score
        """
        y_pred = self.pred(X)
        return np.sum(y_pred == y) / len(y)
