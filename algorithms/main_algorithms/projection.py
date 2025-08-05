import cvxpy as cp
import numpy as np
from numpy import diag


# def proj_b(b):
#     """
#     Calculate the projection of vec b on len(b)-dimensional probability simplex.
#     :param b: a numpy array
#     :return: a numpy array represents the projection of vector b.
#     """
#     u = np.sort(b, kind='mergesort')
#     u = np.insert(u, 0, 0)
#     rho = -1
#     for j in range(1, len(u)):
#         if u[j] + (1 - sum(u[1:j+1]))/j > 0:
#             rho = max(rho, j)
#     lamb = (1 - sum(u[1:rho+1]))/rho
#     x = [max(0, b[i] + lamb) for i in range(len(b))]
#     return np.array(x)

def proj_b(b):
    """
    Calculate the projection of vec b on len(b)-dimensional probability simplex.
    :param b: np array
    :return: the projection of vector b
    """
    b = np.array(b)
    if (np.all(b >= 0)) and np.sum(b) <= 1:
        return b
    else:
        n = len(b)
        P = np.eye(n)
        A = np.ones(n)

        # Define and solve the CVXPY problem.
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)),
                          [x >= -b,
                           A @ x == 1 - np.sum(b)])
        prob.solve()
        res = np.array([abs(xi) for xi in x.value + b])
        return res

def SimplexProj(Y):
    N, D = Y.shape  # 获取 Y 的维度
    X = np.sort(Y, axis=1)[::-1, :]  # 降序排序，然后反转以获得降序排列
    Xtmp = (np.cumsum(X, axis=1) - 1) * diag(1.0 / np.arange(1, D+1))
    Xtmp = Xtmp.astype(float)  # 确保 Xtmp 是浮点数类型
    X = np.maximum(Y - Xtmp[np.newaxis, :, :], 0)  # 使用广播来实现矩阵运算

    return X




if __name__ == '__main__':
    # p_old = [11.85  0.08  0.08], p = [12.12  0.    0.  ], grad = [-0.02 -0.   -0.  ], min_f = -6.6157, f_val = -6.6157, alpha_k =  17.2437
    # p_old = np.array([11.85, 0.08, 0.08])
    # grad = np.array([-0.02, 0, 0])
    # alpha_k = 17.2437
    # p = p_old - grad * alpha_k
    # print(p/12)
    # c = proj_b(p/12)
    # print(c)
    # print(sum(c))
    print(proj_b([0.5, -1, 0]))