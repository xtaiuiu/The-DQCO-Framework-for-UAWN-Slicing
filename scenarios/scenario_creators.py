import pickle

from network_classes.physical_network import PhysicalNetwork
from network_classes.scenario import Scenario
from network_classes.uav import Uav
from scenarios.UE_creators import plot_UE
from scenarios.slice_creators import create_slice_set
import numpy as np


def create_scenario(n_slices, network_radius):
    # a scenario consists of: an UAV, a physical network (mainly physical parameters), a set of slices

    #(self, uav, p_max, b_tot, radius, h_min, h_max, theta_min, theta_max, t_s, g_0, alpha):
    #(self, height, height_prev, theta, speed):
    # uav parameters
    uav_height = 10
    uav_height_prev = 10
    uav_theta = np.pi/6
    uav_speed = 5
    uav = Uav(uav_height, uav_height_prev, uav_theta, uav_speed)

    # network parameters
    p_max = 120
    b_tot = 200
    h_min = 5
    h_max = 20
    theta_min = np.pi/10
    theta_max = np.pi/2.5
    t_s = 100
    g_0 = 1
    alpha = 2
    sigma = 1e-5
    pn = PhysicalNetwork(uav, p_max, b_tot, network_radius, h_min, h_max, theta_min, theta_max, t_s, g_0, alpha, sigma)
    slices = create_slice_set(n_slices, network_radius)
    scenario = Scenario(pn, uav, slices)
    return scenario

def load_scenario(filename):
    with open(filename, 'rb') as file:
        sc = pickle.load(file)
    return sc


def save_scenario(sc, filename):
    with open(filename, 'wb') as file:
        pickle.dump(sc, file)


if __name__ == '__main__':
    scenario = create_scenario(1, 1)
    save_scenario(scenario, 'scenario_2_UEs.pickle')
    zero_var = scenario.scenario_zero_variables()
    var = scenario.scenario_variables()
    print(var.variable_dist(var))
