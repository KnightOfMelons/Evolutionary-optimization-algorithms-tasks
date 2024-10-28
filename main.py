import numpy as np
from scipy.optimize import minimize, differential_evolution


# Определение функции Растригина
def rastrigin(x):
    a = 10
    return a * len(x) + sum([(xi ** 2 - a * np.cos(2 * np.pi * xi)) for xi in x])


# Определение градиента (якобиана) функции Растригина
def rastrigin_jac(x):
    a = 10
    return np.array([2 * xi + 2 * a * np.pi * np.sin(2 * np.pi * xi) for xi in x])


# Параметры для теста
dim = 2  # размерность пространства
bounds = [(-5.12, 5.12) for _ in range(dim)]  # границы для каждого параметра
num_runs = 100  # количество прогонов

newton_results = []
for _ in range(num_runs):
    x0 = np.random.uniform(-5.12, 5.12, dim)  # случайная начальная точка
    res = minimize(rastrigin, x0, method='Newton-CG', jac=rastrigin_jac)  # метод Ньютона с якобианом
    newton_results.append(res.fun)

de_params = {'strategy': 'best1bin', 'mutation': 0.5, 'recombination': 0.7, 'popsize': 15}
de_results = []
for _ in range(num_runs):
    res = differential_evolution(rastrigin, bounds, **de_params)
    de_results.append(res.fun)

newton_mean = np.mean(newton_results)
newton_variance = np.var(newton_results)

de_mean = np.mean(de_results)
de_variance = np.var(de_results)

# Результат вычислений функции Растригина
print(f"Метод Ньютона:\n"
      f"Среднее финальное значение функции: {newton_mean}\n"
      f"Дисперсия финальных значений функции: {newton_variance}\n")

print(f"\nДифференциальная эволюция:\n"
      f"Среднее финальное значение функции: {de_mean}\n"
      f"Дисперсия финальных значений функции: {de_variance}")
