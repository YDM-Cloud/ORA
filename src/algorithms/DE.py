import numpy as np


class DE:
    def __init__(self, pop_size, dim, lb, ub, max_iter, F=0.5, CR=0.9):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.max_iter = max_iter
        self.F = F
        self.CR = CR

    def optimize(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_pos = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        curve = []

        for t in range(self.max_iter):
            for i in range(self.pop_size):
                candidates = list(range(self.pop_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                trial = pop[i].copy()
                mask = np.random.rand(self.dim) < self.CR
                jrand = np.random.randint(self.dim)
                mask[jrand] = True
                trial[mask] = mutant[mask]
                trial_fit = func(trial)

                if trial_fit < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit
                    if trial_fit < best_fit:
                        best_fit = trial_fit
                        best_pos = trial.copy()

            curve.append(best_fit)

        return best_pos, best_fit, np.array(curve).flatten()
