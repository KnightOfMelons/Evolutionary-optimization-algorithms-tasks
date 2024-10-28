import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


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

# Шаг 1: Метод Ньютона
newton_results = []
for _ in range(num_runs):
    x0 = np.random.uniform(-5.12, 5.12, dim)  # случайная начальная точка
    res = minimize(rastrigin, x0, method='Newton-CG', jac=rastrigin_jac)  # метод Ньютона с якобианом
    newton_results.append(res.fun)

# Шаг 2: Дифференциальная эволюция
de_params = {'strategy': 'best1bin', 'mutation': 0.5, 'recombination': 0.7, 'popsize': 15}
de_results = []
for _ in range(num_runs):
    res = differential_evolution(rastrigin, bounds, **de_params)
    de_results.append(res.fun)

# Шаг 3: Расчет статистики
newton_mean = np.mean(newton_results)
newton_variance = np.var(newton_results)

de_mean = np.mean(de_results)
de_variance = np.var(de_results)

# Часть 1: Результат вычислений функции Растригина
print(f"Часть 1.\n"
      f"Метод Ньютона:\n"
      f"Среднее финальное значение функции: {newton_mean}\n"
      f"Дисперсия финальных значений функции: {newton_variance}\n")

print(f"Дифференциальная эволюция:\n"
      f"Среднее финальное значение функции: {de_mean}\n"
      f"Дисперсия финальных значений функции: {de_variance}")

# Часть 2: Оценка времени для достижения глобального экстремума
target_value = 1.0  # более "достижимое" значение для метода Ньютона
newton_times = []
for _ in range(num_runs):
    x0 = np.random.uniform(-5.12, 5.12, dim)
    start_time = time.time()
    res = minimize(rastrigin, x0, method='Newton-CG', jac=rastrigin_jac)
    elapsed_time = time.time() - start_time
    if res.fun <= target_value:
        newton_times.append(elapsed_time)
    else:
        newton_times.append(None)

de_times = []
for _ in range(num_runs):
    start_time = time.time()
    res = differential_evolution(rastrigin, bounds, **de_params)
    elapsed_time = time.time() - start_time
    if res.fun <= target_value:
        de_times.append(elapsed_time)
    else:
        de_times.append(None)

newton_times = [t for t in newton_times if t is not None]
de_times = [t for t in de_times if t is not None]

newton_mean_time = np.mean(newton_times) if newton_times else None
de_mean_time = np.mean(de_times) if de_times else None

# Часть 2: Результаты времени для достижения целевого значения
print(f"\nЧасть 2.\n"
      f"Метод Ньютона:\n"
      f"Среднее время для достижения целевого значения: {newton_mean_time} секунд\n")

print(f"Дифференциальная эволюция:\n"
      f"Среднее время для достижения целевого значения: {de_mean_time} секунд")
