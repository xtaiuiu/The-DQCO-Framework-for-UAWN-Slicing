class PhysicalNetwork:
    # Create a physical network with an UAV, without any UEs
    def __init__(self, uav, p_max, b_tot, radius, h_min, h_max, theta_min, theta_max, t_s, g_0, alpha, sigma):
        self.uav = uav  # The UAV in the network
        self.p_max = p_max
        self.b_tot = b_tot
        self.radius = radius
        self.h_min = h_min
        self.h_max = h_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.t_s = t_s
        self.g_0 = g_0
        self.alpha = alpha
        self.sigma = sigma
        