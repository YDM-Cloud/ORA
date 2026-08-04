import numpy as np


class PSO:
    def __init__(self, pop_size, dim, lb, ub, max_iter, w=0.7, c1=1.5, c2=1.5):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocity = np.zeros_like(pop)
        fitness = np.array([func(ind) for ind in pop])
        pbest = pop.copy()
        pbest_fit = fitness.copy()
        gbest_idx = np.argmin(fitness)
        gbest = pop[gbest_idx].copy()
        gbest_fit = fitness[gbest_idx]
        curve = []

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocity[i] = self.w * velocity[i] + \
                              self.c1 * r1 * (pbest[i] - pop[i]) + \
                              self.c2 * r2 * (gbest - pop[i])
                pop[i] += velocity[i]
                pop[i] = np.clip(pop[i], self.lb, self.ub)
            fitness = np.array([func(ind) for ind in pop])
            improve = fitness < pbest_fit
            pbest[improve] = pop[improve]
            pbest_fit[improve] = fitness[improve]
            idx = np.argmin(fitness)

            if fitness[idx] < gbest_fit:
                gbest_fit = fitness[idx]
                gbest = pop[idx].copy()
            curve.append(gbest_fit)

        return gbest, gbest_fit, np.array(curve).flatten()
