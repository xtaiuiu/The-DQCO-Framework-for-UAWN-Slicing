# Use the method of "A novel projected gradient-like method for optimization problems with simple constraints"
# to optimize the problem:
# min f(x)
# s.t. x >= 0
import numpy as np
import scipy.linalg as lg


# the objective function
def f(x):
    pass


# the gradient of f
def g(x):
    pass




# Algorithm 1 in the paper
def pg_alg(f, g, x0, eps=1e-8):
    """
    :param f: callable, the objective function
    :param g: callable, the gradient of the function
    :param x0: initial feasible
    :param eps: precision
    :return: f_opt, x_opt
    """
    beta, sigma, gamma = 0.5, 0.5, 2
    alpha = gamma
    n, n_max = len(x0), 100
    x = x0
    P = np.minimum(x, g(x))  # P(x^k, \Nabla f(x^k))

    # helper function that can evaluate m_k that satisfies Armijo rule
    def Armijo_stepsize():
        m_max = 100
        for m in range(m_max):
            x_beta_gamma = np.maximum(np.zeros(n), x - beta**m * gamma * P)  # x^k(beta^m gamma)
            lhs = f(x) - f(x_beta_gamma)  # LHS of Equ. (20)
            rhs = sigma * np.dot(P, x - x_beta_gamma)  # RHS of Equ. (20)
            if lhs >= rhs:
                break
        return m

    while (lg.norm(P) > eps) and (n < n_max):
        print(f"n = {n}, precision = {lg.norm(P)}, f = {f(x)}")
        m = Armijo_stepsize()
        alpha = beta**m*gamma
        x = np.maximum(np.zeros(0), x - alpha*P)
        P = np.minimum(x, g(x))  # update P
        n += 1
    return f(x), x


