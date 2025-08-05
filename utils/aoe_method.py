import numpy as np
from mealpy import FloatVar, SHIO

def objective_function(solution):
    return np.sum(solution**2)

problem_dict = {
    "bounds": FloatVar(lb=(-10.,) * 30, ub=(10.,) * 30, name="delta"),
    "minmax": "min",
    "obj_func": objective_function
}

model = SHIO.OriginalSHIO(epoch=1000, pop_size=50)
g_best = model.solve(problem_dict)
print(f"Solution: {g_best.solution}, Fitness: {g_best.target.fitness}")
print(f"Solution: {model.g_best.solution}, Fitness: {model.g_best.target.fitness}")