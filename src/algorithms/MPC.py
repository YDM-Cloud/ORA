import numpy as np


class MPC:
    def __init__(self, pop_size, dim, lb, ub, max_iter, horizon=6, samples=500):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.max_iter = max_iter
        self.horizon = horizon
        self.samples = samples

    def optimize(self, func):
        solution = np.zeros(self.dim)
        curve = []
        current_best = np.inf

        for t in range(self.dim):
            start = t
            end = min(t + self.horizon, self.dim)
            best_window = None
            best_cost = np.inf

            for _ in range(self.samples):
                candidate = np.random.uniform(self.lb[start:end], self.ub[start:end])
                full_solution = solution.copy()
                full_solution[start:end] = candidate
                cost = func(full_solution)

                if cost < best_cost:
                    best_cost = cost
                    best_window = candidate.copy()

            solution[start:end] = best_window

            if best_cost < current_best:
                current_best = best_cost

            curve.append(current_best)

        return solution, current_best, np.array(curve)
