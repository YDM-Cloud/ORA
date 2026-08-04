import numpy as np


class GTO:
    def __init__(self, pop_size, dim, lb, ub, max_iter):
        self.pop_size, self.dim = pop_size, dim
        self.lb, self.ub = np.array(lb), np.array(ub)
        self.max_iter = max_iter

    def optimize(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_pos, best_fit = pop[best_idx].copy(), fitness[best_idx]
        curve = []

        for t in range(self.max_iter):
            a = 3 * (1 - t / self.max_iter)

            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    pop[i] = (self.ub - self.lb) * np.random.rand(self.dim) + self.lb
                else:
                    pop[i] = pop[i] + a * (pop[i] - pop[np.random.randint(self.pop_size)])

                pop[i] = pop[i] + a * (best_pos - pop[i]) * np.random.randn(self.dim)
                pop[i] = np.clip(pop[i], self.lb, self.ub)

            fitness = np.array([func(ind) for ind in pop])
            current_best_idx = np.argmin(fitness)

            if fitness[current_best_idx] < best_fit:
                best_fit = fitness[current_best_idx]
                best_pos = pop[current_best_idx].copy()

            curve.append(best_fit)

        return best_pos, best_fit, np.array(curve).flatten()
