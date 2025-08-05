
import cvxpy as cp
import numpy as np

from algorithms.main_algorithms.rounding.rounding_algorithms import rounding_by_opt_condition


def proj_simplex(y, r):
    """
    project y onto the simplex x>=0, sum(x) <= r by quadratic programming using cvxpy
    :param y: numpy array
    :param r: float
    :return: numpy array
    """
    if sum(y) <= r and np.all(y >= 0):
        return y
    else:
        n = len(y)
        x = cp.Variable(n)
        objective = cp.Minimize(cp.norm(x - y))

        # 定义约束
        constraints = [
            x >= 0,  # x的每个元素大于等于0
            cp.sum(x) <= r  # x的元素之和小于等于r
        ]

        # 形成并解决问题
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return x.value

def proj_generalized_simplex_QP(a, y, r):
    """
    project y onto the simplex x>=0, <a, x> = r by quadratic programming using cvxpy
    :param y: numpy array
    :param r: float
    :return: numpy array
    """
    if sum(y) == r and np.all(y >= 0):
        return y
    else:
        n = len(y)
        x = cp.Variable(n)
        objective = cp.Minimize(cp.norm(x - y))

        # 定义约束
        constraints = [
            x >= 0,  # x的每个元素大于等于0
            a@x == r  # x的元素之和小于等于r
        ]

        # 形成并解决问题
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.ECOS)
        return np.array(x.value)

def proj_simplex_kkt(y, r):
    """
    project y onto the simplex sum(x) = r, x >= 0 by using kkt condition
    :param y: numpy array
    :param r: float
    :return: numpy array
    """
    u = np.array(sorted(y, reverse=True))
    rho = 0
    for j in range(len(u)):
        if u[j]+(r-np.sum(u[:j]))/(j+1) > 0:
            rho = j
        else:
            break
    lam = (r-np.sum(u[:rho+1]))/(rho+1)
    return np.array([max(0, y[i]+lam) for i in range(len(y))])

def proj_generalized_simplex(a, y, r):
    """
    project y onto the generalized simplex <a, x> = r, x >= 0 by using kkt condition
    :param a: numpy array
    :param y: numpy array
    :param r: float
    :return: numpy array
    """
    u = -np.sort(-y/a)
    sorted_idx = np.argsort(-y/a)
    s_a = a[sorted_idx]
    s_a_squared = s_a**2
    z = s_a_squared * u

    rho = 0
    for j in range(len(u)):
        if u[j] + (r - np.sum(z[:j+1])) / (np.sum(s_a_squared[:j+1])) > 0:
            rho = j
        else:
            break
    lam = (r - np.sum(s_a_squared[:rho+1]*u[:rho+1])) / (np.sum(s_a_squared[:rho+1]))
    return np.array([max(0, y[i]+lam*a[i]) for i in range(len(y))])


def proj_generalized_simplex_lb(a, y, r, l):
    """
    project y onto the generalized simplex <a, x> = r, x >= l by using kkt condition
    :param a: numpy array
    :param y: numpy array
    :param r: float
    :param l: a numpy array represents the lower bound
    :return: numpy array
    """
    z = proj_generalized_simplex(a, y-l, r - a@l)
    return z + l


def proj_conv_kkt(y, r):
    """
    project y onto the simplex x>=0, sum(x) <= r
    :param y: numpy array
    :param r: float
    :return: numpy array
    """
    y[y < 0] = 0
    if np.sum(y) <= r:
        return y
    else:
        return proj_simplex_kkt(y, r)


def rounding_by_proj_QP(a, y, r):
    """
    Project y onto the constraints defined by <a, x> <= r, x >=0, x integer using cvxpy
    :param a: numpy array of the coefficient
    :param y: numpy array represents the point that will be projected
    :param r: scalar
    :return: integer numpy array
    """
    z = np.floor(y)
    y_bar = y - z
    u = cp.Variable(len(y), integer=True)
    objective = cp.Minimize(cp.sum_squares(u - y_bar))
    constraints = [0 <= u, u <= 1, a@u <= r - a@z]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    print("Status: ", prob.status)
    #print("The optimal value is", prob.value)
    print("A solution x is")
    print(u.value + z)
    return u.value + z


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    c = np.array([2, 4, 2])

    y = np.array([1.2, 2.5, 3.7])
    f_opt = min(c*y)
    r = 17
    res_QP = rounding_by_proj_QP(a, y, r)
    res_cond = rounding_by_opt_condition(a, c, y, f_opt, r)
    print(f"QP rounding = {res_QP}, cond_rounding = {res_cond}")
    #np.random.seed(0)

    # for i in range(1000):
    # # 随机生成一个正整数n作为数组的长度
    #     n = np.random.randint(1, 100)  # 假设数组长度在1到99之间
    #
    #     # 随机生成两个长度为n的numpy数组a和b
    #     # 使用numpy.random.uniform来生成[1, 10)之间的随机数，确保a中的元素大于0
    #     a = np.random.uniform(low=1, high=10, size=n)
    #
    #     # 生成数组b，这里没有对b的元素做出大于0的限制
    #     y = np.random.uniform(low=0, high=10, size=n)
    #
    #     r = 10
    #     z1 = proj_generalized_simplex(a, y, r)
    #     z2 = (proj_generalized_simplex_QP(a, y, r))
    #     try:
    #         assert np.linalg.norm(z1-z2, ord=2) < 1e-2
    #     except Exception as e:
    #         print(f"assert error: {np.linalg.norm(z1-z2, ord=2): .12f}")