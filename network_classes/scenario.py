import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from network_classes.Variables import Variables


class Scenario:

    def __init__(self, pn, uav, slices):
        self.pn = pn
        self.uav = uav
        self.slices = slices

    def scenario_variables(self):
        #  return the variable list ordered as: x, p, h
        x, p, h = [], [], 0
        for s in self.slices:
            x_row, p_row = [], []
            for k in range(len(s.UEs)):
                x_row.append(s.UEs[k].x)
                p_row.append(s.UEs[k].p)
            x.append(x_row)
            p.append(p_row)
        h = self.uav.h
        return Variables(np.array(x), np.array(p), h)

    def scenario_zero_variables(self):
        x, p, h = [], [], 0
        for s in self.slices:
            x_row, p_row = [], []
            for k in range(len(s.UEs)):
                x_row.append(0)
                p_row.append(0)
            x.append(x_row)
            p.append(p_row)
        return Variables(x, p, h)

    def set_vars(self, z):
        """
        Set all variables for the scenario
        :param z: ndarray, the variables
        :return: None
        """
        n = 0
        for s in self.slices:
            for u in s.UEs:
                n += 1
        self.set_x(z[:n])
        self.set_p(z[n:2*n])
        self.set_h(z[-1])


    def set_p(self, p):
        """
        Set the power of each UE with p
        :param p: a numpy array or a list
        :return: no return
        """
        idx = 0
        for s in self.slices:
            for u in s.UEs:
                u.p = p[idx]
                idx += 1


    def set_vars_no_h(self, z):
        """
        Set all variables for the scenario
        :param z: ndarray, the variables
        :return: None
        """
        n = int(len(z)/2)
        self.set_x(z[:n])
        self.set_p(z[n:2*n])

    def reset_scenario(self):
        """
        Set all variables to 0, except h
        :return: None
        """
        n = len(self.scenario_variables().x)
        self.set_vars_no_h(np.zeros(2*n))



    def set_x(self, x):
        """
        Set the No. channels for each UE with x
        :param x: a numpy array or a list
        :return: no return
        """
        idx = 0
        for s in self.slices:
            for u in s.UEs:
                u.x = x[idx]
                idx += 1

    def set_h(self, h):
        """
        Set the height of the UAV
        :param h: the new hight
        :return: None
        """
        self.uav.h = h

    def set_theta(self, h):
        theta = max(self.pn.theta_min ** 2, (math.atan(self.pn.radius / (math.sqrt(h)))) ** 2)
        self.uav.theta = theta

    def get_UE_rate(self):
        """
        Get the rate of each UE under the current system parameters
        :return: a numpy array represents the rate of each UE
        """
        rates, obj = [], []
        pn, uav = self.pn, self.uav
        for s in self.slices:
            for u in s.UEs:
                rate = u.x * s.b_width * np.log(1 + pn.g_0 * u.tilde_g * u.p / (
                            pn.sigma * (uav.theta**2) * (u.loc_x ** 2 + u.loc_y ** 2 + uav.h) ** (pn.alpha / 2)))
                rates.append(rate)
                obj.append(rate/u.tilde_r)
        return np.array(rates), np.array(obj)


    def get_xp_lb(self):
        """
        Get the lower bounds of x and p
        :return: two numpy array, i.e., lb_x and lb_y
        """
        pn, uav = self.pn, self.uav
        theta_h = max(pn.theta_min ** 2, (math.atan(pn.radius / (math.sqrt(uav.h)))) ** 2)
        lb_x, lb_p = [], []

        for i in range(len(self.slices)):
            slice = self.slices[i]
            for k in range(len(slice.UEs)):
                ue = slice.UEs[k]
                lb_x.append(ue.tilde_r)
                nu = pn.g_0 * ue.tilde_g / (
                        pn.sigma * (theta_h ** 2) * (ue.loc_x ** 2 + ue.loc_y ** 2 + uav.h) ** (pn.alpha / 2))
                lb_p.append((np.exp(slice.r_sla / slice.b_width) - 1) / nu)
        return np.array(lb_x), np.array(lb_p)

    def plot_scenario(self):
        network_radius = self.pn.radius

        circle = plt.Circle((0, 0), self.pn.radius, fill=False)
        fig, ax = plt.subplots()

        # plot the circle representing the UAWN
        ax.add_patch(circle)
        plt.xlim(-network_radius - 1, network_radius + 1)
        plt.ylim(-network_radius - 1, network_radius + 1)

        # plot UEs for each slice
        cmap = get_cmap('tab20')
        for i in range(len(self.slices)):
            plt.scatter([p.loc_x for p in self.slices[i].UEs], [p.loc_y for p in self.slices[i].UEs],
                        color=cmap(i % 20), label=f'UE of slice {i}')
        ax.legend()
        plt.axis('equal')
        plt.xlabel('x (meter)')
        plt.ylabel('y (meter)')
        plt.title('User distribution')

        plt.show()
        return fig
