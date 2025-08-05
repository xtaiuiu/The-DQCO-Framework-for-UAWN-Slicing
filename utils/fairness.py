import numpy as np

def Jain_fairness(u):
    """
    Compute the Jain's fairness index
    :param u: numpy array represent the utility of each person
    :return: the fairness index
    """
    return (np.sum(u))**2 / (len(u)* np.linalg.norm(u) ** 2)
