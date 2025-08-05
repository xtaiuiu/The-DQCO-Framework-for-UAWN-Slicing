import math
import time

import numpy as np
import cvxpy as cp

from scenarios.scenario_creators import create_scenario


def optimal_power(B_array, nu, x_l, p_l, B_tot, P_max):
    # Input parameters: α and β are constants from R_i equation
    n = len(B_array)

    p = cp.Variable(shape=n, pos=True)
    x = cp.Variable(shape=n, pos=True)
    z = cp.Variable(shape=n, pos=True)
    t = cp.Variable(shape=1, pos=True)


    # This function will be used as the objective so must be DCP;
    # i.e. elementwise multiplication must occur inside kl_div,
    # not outside otherwise the solver does not know if it is DCP...


    objective = cp.Maximize(t)
    constraints = [p>=p_l,
                   x>=x_l,
                   cp.sum(p)-P_max==0.0,
                   cp.sum(cp.multiply(B_array, x))-B_tot==0.0]
    for i in range(n):
        constraints.append(t*cp.inv_pos(x[i])/B[i] - cp.log1p(nu[i]*p[i]) <= 0.0)

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True)

    return prob.status, prob.value, t.value, p.value, x.value


if __name__ == '__main__':
    # 无法运行，因为问题不符合DCP规范，这个问题无法用内点法求解。
    np.set_printoptions(precision=3)
    np.random.seed(2000)
    sc = create_scenario(5, 10)
    h = max((max(0, np.sqrt(sc.uav.h_bar) - sc.uav.c * sc.pn.t_s)) ** 2, sc.pn.h_min ** 2)
    sc.set_h(h)
    sc.set_theta(h)
    pn = sc.pn
    theta_h = max(pn.theta_min ** 2, (math.atan(pn.radius / (math.sqrt(sc.uav.h)))) ** 2)
    B_tot, P_max = pn.b_tot, pn.p_max
    nu, B, l_x, l_p = [], [], [], []  # which are the coefficients, B_array and the lower bound of the problem, respectively
    for i in range(len(sc.slices)):
        s = sc.slices[i]
        for k in range(len(sc.slices[i].UEs)):
            u = sc.slices[i].UEs[k]
            nu.append(pn.g_0 * u.tilde_g / (
                    pn.sigma * (theta_h ** 2) * (u.loc_x ** 2 + u.loc_y ** 2 + sc.uav.h) ** (pn.alpha / 2)))
            B.append(s.b_width)
            l_x.append(u.tilde_r)
            l_p.append((np.exp(s.r_sla/s.b_width) - 1))
    nu, B, l_x, l_p = np.array(nu), np.array(B), np.array(l_x), np.array(l_p)
    l_p = l_p/nu
    status, utility, t, power, bandwidth = optimal_power(B, nu, l_x, l_p, B_tot, P_max)

    print('Status: {}'.format(status))
    print('Optimal utility value = {:.4g}'.format(utility))
    print('Optimal power level:\n{}'.format(power))
    print('Optimal bandwidth:\n{}'.format(bandwidth))
