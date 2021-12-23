import numpy as np
from cvxopt import matrix, solvers
import itertools
from  collections import Counter
from functools import lru_cache
import seaborn as sns
import matplotlib.pyplot as plt


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


def KKT_violations(alpha, y, X, w, bias, C):
    """

    :param alpha: Vector
    :param y: Vector
    :param X: Matrix
    :param w: Vector
    :param bias: Float
    :param C: Integer
    :return: The number of KKT violations that this solutions incurs in
    """

    # y(w*x_i+bias)
    res = y * (np.dot(w, X.T) + bias)
    # alpha = 0 --> res >= 1
    idx1 = np.isclose(alpha, 0, atol=1e-2)
    # alpha = C --> res <= 1
    idx2 = np.isclose(alpha, C, atol=1e-2)
    # 0 < alpha < C --> res = 1
    idx3 = ~(idx1 + idx2)
    num = sum(res[idx1] < 1)
    num += sum(res[idx2] > 1)
    num += sum(~np.isclose(res[idx3], 1, atol=1e-2))

    return num


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

    @property
    def state(self):
        """
        Get the state of the SVM object
        """
        state_dict = {'kernel':self.kernel,
                      'gamma':self.gamma,
                      'C':self.C}
        return state_dict

    def __generate_intermediate_variables(self):
        """
        Since we will be using cvxopt we have to generate an optimization problem of the following shape:

        min 0.5x^TPx + q^Tx
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
        self.__generate_intermediate_variables()
        self.fit_sol = solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b)

        self.alpha = np.ravel(self.fit_sol['x'])

        self.w, self.bias = self._compute_params(alphas=self.alpha, tol=tol, fix_intercept=fix_intercept)

    def _compute_params(self, alphas, tol, fix_intercept=False):
        """
        This method returns a set of parameters estimated based on the previous fit
        """

        self.sv_idx = (alphas > tol).reshape(len(self.X),)
        self.idx = np.arange(len(alphas))[self.sv_idx]

        self.w = (self.y[self.sv_idx] * alphas[self.sv_idx]).T @ self.X[self.sv_idx]

        self.bias = 0
        if not fix_intercept:
        #     for i in range(np.sum(self.sv_idx)):
        #         self.bias += self.y[self.sv_idx][i] - np.sum(
        #             self.y[self.sv_idx] * alphas[self.sv_idx] * self.K[self.idx[i], self.sv_idx])

        #     self.bias = self.bias / np.sum(self.sv_idx)
            self.bias = np.sum(self.y[self.sv_idx] - np.sum(
                self.y[self.sv_idx] * alphas[self.sv_idx] * self.K[np.ix_(self.sv_idx, self.sv_idx)], axis=1))
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
        """
        Class which implements the decomposition method for q>=2.
        It can be subclassed or a new definition of the function _solve_subproblem() can be provided
        in order to solve the case q=2 analitically.
        """
        super().__init__(X, y, C, gamma, kernel)


    def _compute_params(self, alphas, tol, fix_intercept=False):
        """
        This method returns a set of parameters estimated based on the previous fit
        """

        self.sv_idx = (alphas > tol).reshape(len(self.X),)
        self.idx = np.arange(len(alphas))[self.sv_idx]

        self.w = (self.y[self.sv_idx] * alphas[self.sv_idx]).T @ self.X[self.sv_idx]

        self.bias = 0
        if not fix_intercept:
            self.bias = np.sum(self.y[self.sv_idx] - np.sum(
                self.y[self.sv_idx] * alphas[self.sv_idx] * self._kernel_fun(self.X[self.sv_idx], self.X[self.sv_idx], self.gamma), axis=1))
            self.bias /= np.sum(self.sv_idx)

        return self.w, self.bias


    def _select_working_set(self, q, atol=1e-4):
        """
        Select the working set according to the Most Violating Pair (MVP) strategy.

        :param q: cardinality of the working set
        :param atol: absolute tolerance for the inequalities

        :return m_a: max {-y[I] * ∇f[I]}
                M_a: min {-y[J] * ∇f[J]}
                working_set: set of indices; numpy.ndarray
        """
        R = (((self.alpha + atol < self.C) & (self.y ==  1)) | ((self.alpha - atol > 0) & (self.y == -1)))
        S = (((self.alpha + atol < self.C) & (self.y == -1)) | ((self.alpha - atol > 0) & (self.y ==  1)))
        int_idx = np.argsort( -self.y * self._gradients ) # sort the array
        I = int_idx[R[int_idx]][-q//2:][::-1] # argmax
        J = int_idx[S[int_idx]][:q//2] # argmin
        m_a = -self.y[I[0]] * self._gradients[I[0]] # max i
        M_a = -self.y[J[0]] * self._gradients[J[0]] # min j
        working_set = np.concatenate( (I, J) )
        return m_a, M_a, working_set


    @lru_cache()
    def _hessian_cache(self, index):
        """
        Implement the cache of the Hessian matrix columns exploiting the efficient 
        Python decorator @lru_cache.

        :param index: index of the working set for which compute (or return) 
                      the corresponding column of the Hessian
        
        :return column Q_index
        """
        K_index = self._kernel_fun(self.X, self.X[index,np.newaxis], self.gamma)
        Q_index = (self.y * self.y[index])[:,np.newaxis] * K_index
        return Q_index


    def _hessian_handler(self, working_set):
        """
        Manage the Hessian matrix by exploiting an efficient cache.

        :param working_set: indices of the working set; numpy.ndarray

        :return columns of the Hessian matrix Q (cfr. theory); numpy.ndarray
        """
        Q_ws = []
        for i in working_set:
            Q_ws.append(self._hessian_cache(i))
        Q_ws = np.hstack(Q_ws)
        return Q_ws


    def _solve_subproblem(self, working_set, working_set_size):
        """
        Solve numerically the subproblem w.r.t. the alphas in the working set 
        by means of a quadratic programming solver.

        :param working_set: indices of the working set; numpy.ndarray

        :return solution object of the subproblem
        """
        solvers.options['show_progress'] = False
        ws_mask = np.full(self.y.shape, fill_value=True)
        ws_mask[working_set] = False # get the mask for the complement set of the working set
        Q = self._hessian_handler(working_set)
        P = matrix(Q[working_set,:]) # select the ij rows of the ij columns of the hessian
        q = matrix(np.squeeze(self.alpha[ws_mask][np.newaxis,:] @ Q[ws_mask,:]) -1, (working_set_size, 1), 'd') 
        G = matrix(np.vstack( ( np.diag(np.full(working_set_size, fill_value=-1)), np.identity(working_set_size) ) ))
        h = matrix(np.hstack( ( np.zeros(working_set_size), np.full(working_set_size, fill_value=self.C) ) ))
        A = matrix(self.y[working_set], (1, working_set_size), 'd')
        b = matrix(-self.y[ws_mask][np.newaxis,:] @ self.alpha[ws_mask][:, np.newaxis])
        return solvers.qp(P, q, G, h, A, b)


    def _get_di(self, di, dj, alphai, alphaj, c):
        """
        Retrieve the value for beta_var depending on the rules defined.

        :param di: direction for i; Integer
        :param dj: direction for j; Integer
        :param alphai: alpha value for i; Float
        :param alphaj: alpha value for j; Float
        :param c: Penalty parameter of the error term; Float
        
        :return value of beta_bar
        """   
        if (di>0 and dj>0):
            beta_bar = min(c - alphai, c - alphaj)
        elif (di<0 and dj<0):
            beta_bar = min(alphai, alphaj)
        elif (di>0 and dj<0):
            beta_bar = min(c - alphai, alphaj)
        elif (di<0 and dj>0):
            beta_bar = min(alphai, c - alphaj)
        return beta_bar
        
    
    def _solve_subproblem_analytical(self, working_set, working_set_size = 2):
        """
        Solve analitycally the subproblem w.r.t. the alphas in the working set.

        :param working_set: indices of the working set; numpy.ndarray
        :param working_set_size: even number equal to 2; called q in the homework/literature

        :return solution object of the subproblem
        """
        
        # Get indexes from the working data set
        I = working_set[0]
        J = working_set[1]
        
        # Get information with the indexes of the working data set
        X_ws = self.X[[I,J],]
        y_ws = self.y[[I,J]]
        alpha_ws = self.alpha[[I,J]]
        gradients_ws = self._gradients[[I,J]]
        
        # Generate Q (semidefinite positive) matrix 
        Q_sub = np.dot((np.dot(np.diag(y_ws), rbf(X_ws, X_ws, self.gamma))), np.diag(y_ws))
        
        # Initialize direction and alpha_star result
        di = np.zeros(working_set_size)
        alpha_star = np.zeros(working_set_size)
        
        # Define directions
        di[0] =   y_ws[0]
        di[1] = - y_ws[1]
        
        # Define beta bar by the conditions given
        beta_bar = self._get_di(di[0], di[1], self.alpha[I], self.alpha[J], self.C)
        
        # Evaluate 
        if np.dot(gradients_ws.T, di) == 0: 
            beta_star = 0
        else:
            if np.dot(gradients_ws.T, di) < 0:
                d_star = di
            else:
                d_star = -di
        
        # Calculate the maximum allowable step beta_bar along d_star:
        
        # Define the value for beta_star
        if beta_bar == 0:
            beta_star = 0
        elif np.dot(np.dot(d_star.T, Q_sub), d_star) == 0:
            beta_star = beta_bar
        else:
            if np.dot(np.dot(d_star.T, Q_sub), d_star) > 0:
                beta_nv = -np.dot(gradients_ws.T, d_star) / np.dot(np.dot(d_star.T, Q_sub), d_star)
                beta_star = min(beta_bar, beta_nv)
        
        
        # Calculate the solution
        alpha_star[0] = alpha_ws[0] + (beta_star * d_star[0])
        alpha_star[1] = alpha_ws[1] + (beta_star * d_star[1])
        
        return alpha_star


    def fit(self, working_set_size, max_iters=500, stop_thr=1e-5, tol=1e-4, fix_intercept=False):
        """
        Perform the optimization over the alpha values using the decomposition algorithm.

        :param working_set_size: even number greater or equal than 2; called q in the homework/literature
        :param max_iters: stop the optimization loop after a certain number of iterations
        :param stop_thr: tolerance threshold for the stopping criterion m(a) - M(a) > stop_thr
        :param M: 
        :param tol: tolerance for computing the comparison between floating-point numbers
        :param fix_intercept: whether fix the intercept to 0 or not

        :rerturn (no return value)  
        """
        # Setup initial values
        self.alpha = np.zeros(self.y.shape)
        self._gradients = np.full(self.y.shape, fill_value=-1)
        self.i = 0; m_a = 1; M_a = 0
        
        fit_sol = {'x': []}
        
        # Decomposition
        while  (self.i < max_iters) and (m_a - M_a > stop_thr):
            m_a, M_a, working_set = self._select_working_set(working_set_size, atol=tol)
            
            if working_set_size == 2:
                # Solve the problem by means of most violating pair (MVP)
                fit_sol['x'] = self._solve_subproblem_analytical(working_set)
                
            else:
                fit_sol = self._solve_subproblem(working_set, working_set_size)
                
            step = (np.ravel(fit_sol['x']) - self.alpha[working_set])[:,np.newaxis]
            Q_ws = self._hessian_handler(working_set)
            self._gradients = self._gradients + np.squeeze(Q_ws @ step)
            self.alpha[working_set] = np.ravel(fit_sol['x'])
            self.i += 1
        
        self.w, self.bias = self._compute_params(alphas=self.alpha, tol=tol, fix_intercept=fix_intercept)

        


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


def confusion_matrix(y_test, y_fit):
    
    mat = confusion_matrix(y_test, y_fit)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=[class_A, class_B],
                yticklabels=['class_A', 'class_B'])
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()
