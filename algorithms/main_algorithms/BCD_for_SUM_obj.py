# this file solve the following problem by bcd
# \max_{x, y} \sum\limits_{i} a_i*x_i ln(1 + b_i*y_i)
# s.t. x >= c > 0, d^T x = e
# y >= q > 0, 1^T y = p
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import linprog


# \max\limits_{x, y} \sum\limits_{i=1}^n a_i * x_i *log(1 + b_i * y_i), s.t. x \ge c, d^T * x = e, y \ge q, 1^T * y =p
class OptimizationProblem:
    def __init__(self, a, b, c, d, e, p, q):
        """
        初始化优化问题。

        :param a: 系数 a_i 的数组。
        :param b: 系数 b_i 的数组。
        :param c: x 的下界数组。
        :param d: 约束 d^T * x = e 中的系数数组。
        :param e: 约束 d^T * x = e 中的常数。
        :param p: 约束 1^T * y = p 中的常数。
        :param q: y 的下界数组。
        """
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.d = np.array(d)
        self.e = e
        self.p = p
        self.q = np.array(q)

    def objective_function(self, x, y):
        """
        计算目标函数的值。

        :param x: x 的值数组。
        :param y: y 的值数组。
        :return: 目标函数的值。
        """
        return np.sum([self.a[i]*x[i]*np.log(1 + self.b[i]*y[i]) for i in range(len(x))])


def create_optimization_instance(n=10):
    """
    创建优化问题的实例。

    :param a: 系数 a_i 的数组。
    :param b: 系数 b_i 的数组。
    :param c: x 的下界数组。
    :param d: 约束 d^T * x = e 中的系数数组。
    :param e: 约束 d^T * x = e 中的常数。
    :param p: 约束 1^T * y = p 中的常数。
    :param q: y 的下界数组。
    :return: OptimizationProblem 类的实例。
    """
    x = np.random.randint(low=1, high=20, size=n)
    y = 1 + np.random.random(size=n) * 10
    a = 1 + np.random.random(size=n) * 10
    b = 1 + np.random.random(size=n) * 10
    d = 1 + np.random.random(size=n) * 10
    e = np.dot(x, d)
    p = np.sum(y)
    x_min, y_min = np.min(x), np.min(y)
    c = x_min - np.random.random(size=n) * 0.2 * x_min
    q = y_min - np.random.random(size=n) * 0.2 * y_min

    return OptimizationProblem(a, b, c, d, e, p, q)


# maximize \sum\limits_{i} a_i ln(1 + b_i*y_i), s.t., y >= q, 1^T y = p
def optimize_y(a, b, q, p):
    """
    a, b, q are numpy arrays
    :return: opt_val => double, opt_val => numpy array
    """
    def objective(x):
        return -np.sum(a * np.log(1 + b * x))

    def constraint_sum(x):
        return np.sum(x) - p

    constr = ({'type': 'eq', 'fun': constraint_sum})
    bounds = tuple((q[i], None) for i in range(len(a)))
    result = minimize(objective, q, method='SLSQP', bounds=bounds, constraints=constr)
    print("Optimization result:", result)
    print("Optimal value:", -result.fun)  # 由于我们最小化了负目标函数，所以需要取负值
    print("Optimal x:", result.x)
    return -result.fun, result.x


# maximize \sum\limits_{i} a_i*x_i, s.t. x >= b, c^T * x = d
def optimize_x(a, b, c, d):
    """
    a, b, c are numpy array, d is double
    :return: opt_val => double, opt_var => numpy array
    """
    result = linprog(-a, A_eq=c.reshape(-1, c.size), b_eq=d, bounds=[(b[i], None) for i in range(len(a))])

    # 输出结果
    print("Optimization result:", result)
    print("Optimal value:", -result.fun)  # 由于我们最小化了目标函数的负值，所以需要取负值
    print("Optimal x:", result.x)
    return -result.fun, result.x


def genetic(problem):
    from geneticalgorithm import geneticalgorithm as ga

    def f(X):
        n = int(len(X)/2)
        x, y = X[:n], X[n:]
        OF = -problem.objective_function(x, y)
        pen = 0
        if np.dot(x, problem.d) > problem.e:
            pen += 500000 + 1000000 * (np.dot(x, problem.d) - problem.e)
        if np.sum(y) > problem.p:
            pen += 500000 + 1000000 * (np.sum(y) - problem.p)
        return OF + pen
    bounds = []
    for i in range(n):
        bounds.append([problem.c[i], problem.e])
    for i in range(n):
        bounds.append([problem.q[i], problem.p])
    varbound = np.array(bounds)

    algorithm_param = {'max_num_iteration': 10000, \
                       'population_size': 200, \
                       'mutation_probability': 0.1, \
                       'elit_ratio': 0.01, \
                       'crossover_probability': 0.5, \
                       'parents_portion': 0.3, \
                       'crossover_type': 'uniform', \
                       'max_iteration_without_improv': None}

    model = ga(function=f, dimension=2*(len(problem.a)), variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

    model.run()
    solution = model.output_dict
    print(solution)
    return -solution['function'], solution['variable']


if __name__ == '__main__':
    n_iteration = 2
    np.random.seed(0)
    n = 4
    P = create_optimization_instance(n)
    genetic(P)
    y = P.q
    x = P.c
    opt_vals = []
    for i in range(10):

        val_x, x = optimize_x(np.array([P.a[i] * np.log(1 + P.b[i] * y[i]) for i in range(n)]), P.c, P.d, P.e)
        opt_vals.append(val_x)

        val_y, y = optimize_y(np.array([P.a[i] * x[i] for i in range(n)]), P.b, P.q, P.p)
        opt_vals.append(val_y)
    plt.plot(np.arange(10*2), opt_vals)
    plt.show()
