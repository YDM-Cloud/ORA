import numpy as np


def evaluate_population(func, pop):
    fitness = np.array([func(ind) for ind in pop])
    if fitness.ndim > 1:
        return np.sum(fitness, axis=1)
    return fitness


class ORA:
    def __init__(self, pop_size, dim, lb, ub, max_iter,
                 archive_size=5, archive_weight=0.7, archive_probability=0.7, resonance_strength=0.4):
        self.pop_size = pop_size
        self.dim = dim
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.max_iter = max_iter
        self.archive_size = archive_size
        self.archive_weight = archive_weight
        self.archive_probability = archive_probability
        self.resonance_strength = resonance_strength

    def optimize(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = evaluate_population(func, pop)
        idx = np.argsort(fitness)[:self.archive_size]
        archive = pop[idx].copy()
        archive_fitness = fitness[idx].copy()
        curve = []

        for t in range(self.max_iter):
            progress = t / self.max_iter

            # ==========================
            # Archive guidance
            # ==========================

            r1 = np.random.randint(0, self.archive_size, self.pop_size)
            r2 = np.random.randint(0, self.archive_size, self.pop_size)
            dist1 = np.linalg.norm(pop - archive[r1], axis=1)
            dist2 = np.linalg.norm(pop - archive[r2], axis=1)
            guide_idx = np.where(dist1 > dist2, r1, r2)
            guide = archive[guide_idx]
            use_archive = np.random.rand(self.pop_size) < self.archive_probability
            guide[~use_archive] = pop[~use_archive]
            archive_step = self.archive_weight * (guide - pop)

            # ==========================
            # Exploration
            # ==========================

            random_pop = pop[np.random.randint(0, self.pop_size, self.pop_size)]
            exploration_step = random_pop - pop
            ranks = np.argsort(np.argsort(fitness)) / self.pop_size
            noise_scale = np.where(ranks < 0.2, 0.1 * (1 - progress), 2.0)
            step_scale = 0.5 * (1 - progress) + 0.1
            random_noise = np.random.randn(self.pop_size, self.dim) * noise_scale[:, None]
            exploration_step *= step_scale * random_noise
            pop_new = pop + archive_step + exploration_step

            # ==========================
            # Resonance mechanism
            # ==========================

            resonance = archive[0] - pop
            pop_new += self.resonance_strength * (1 - progress) * resonance
            pop_new = np.clip(pop_new, self.lb, self.ub)
            new_fitness = evaluate_population(func, pop_new)
            improved = new_fitness < fitness
            pop[improved] = pop_new[improved]
            fitness[improved] = new_fitness[improved]

            # ==========================
            # Archive update
            # ==========================

            combined_pop = np.vstack((archive, pop))
            combined_fit = np.concatenate((archive_fitness, fitness))
            idx = np.argsort(combined_fit)[:self.archive_size]
            archive = combined_pop[idx]
            archive_fitness = combined_fit[idx]
            curve.append(archive_fitness[0])

        return archive[0], archive_fitness[0], np.array(curve)


# =====================================================
# Ablation:
# Remove Archive
# =====================================================


class ORA_NoArchive(ORA):
    def optimize(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = evaluate_population(func, pop)
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()
        best_fit = fitness[best_idx]
        curve = []

        for t in range(self.max_iter):
            progress = t / self.max_iter
            random_step = np.random.randn(self.pop_size, self.dim)
            step_scale = 0.5 * (1 - progress) + 0.1
            pop_new = pop + step_scale * random_step
            pop_new += 0.2 * (best - pop)
            pop_new = np.clip(pop_new, self.lb, self.ub)
            new_fit = evaluate_population(func, pop_new)
            improved = new_fit < fitness
            pop[improved] = pop_new[improved]
            fitness[improved] = new_fit[improved]
            idx = np.argmin(fitness)

            if fitness[idx] < best_fit:
                best_fit = fitness[idx]
                best = pop[idx].copy()
            curve.append(best_fit)

        return best, best_fit, np.array(curve)


# =====================================================
# Ablation:
# Remove Resonance
# =====================================================


class ORA_NoResonance(ORA):
    def optimize(self, func):
        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = evaluate_population(func, pop)
        idx = np.argsort(fitness)[:self.archive_size]
        archive = pop[idx].copy()
        archive_fitness = fitness[idx].copy()
        curve = []

        for t in range(self.max_iter):
            progress = t / self.max_iter
            r = np.random.randint(0, self.archive_size, self.pop_size)
            guide = archive[r]
            random_pop = pop[np.random.randint(0, self.pop_size, self.pop_size)]
            step = 0.5 * (1 - progress) + 0.1
            pop_new = pop + step * (guide + random_pop - 2 * pop) * np.random.randn(self.pop_size, self.dim)
            pop_new = np.clip(pop_new, self.lb, self.ub)
            new_fit = evaluate_population(func, pop_new)
            improved = new_fit < fitness
            pop[improved] = pop_new[improved]
            fitness[improved] = new_fit[improved]
            combined_pop = np.vstack((archive, pop))
            combined_fit = np.concatenate((archive_fitness, fitness))
            idx = np.argsort(combined_fit)[:self.archive_size]
            archive = combined_pop[idx]
            archive_fitness = combined_fit[idx]
            curve.append(archive_fitness[0])

        return archive[0], archive_fitness[0], np.array(curve)
