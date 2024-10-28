import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import matplotlib.pyplot as plt
import pandas as pd


def rastrigin(input_vector):
    a = 10
    return a * len(input_vector) + sum([(xi ** 2 - a * np.cos(2 * np.pi * xi)) for xi in input_vector])


def rastrigin_jac(input_vector):
    a = 10
    return np.array([2 * xi + 2 * a * np.pi * np.sin(2 * np.pi * xi) for xi in input_vector])


dim = 2
bounds = [(-5.12, 5.12) for _ in range(dim)]
num_runs = 100
max_generations = 100

newton_results = []
for _ in range(num_runs):
    x0 = np.random.uniform(-5.12, 5.12, dim)
    res = minimize(rastrigin, x0, method='Newton-CG', jac=rastrigin_jac)
    newton_results.append(res.fun)

de_params = {'strategy': 'best1bin', 'mutation': 0.5, 'recombination': 0.7, 'popsize': 15, 'maxiter': max_generations}
de_results = []
de_times = []
de_fitness_history = []
for _ in range(num_runs):
    start_time = time.time()
    fitness_history = []


    def callback(xk, _=None):
        fitness_history.append(rastrigin(xk))


    res = differential_evolution(rastrigin, bounds, **de_params, callback=callback)
    elapsed_time = time.time() - start_time
    de_results.append(res.fun)
    de_times.append(elapsed_time)
    de_fitness_history.append(fitness_history)

newton_mean = np.mean(newton_results)
newton_variance = np.var(newton_results)

de_mean = np.mean(de_results)
de_variance = np.var(de_results)

print(f"====== Часть 1. ======\n"
      f"Метод Ньютона:\n"
      f"Среднее финальное значение функции: {newton_mean}\n"
      f"Дисперсия финальных значений функции: {newton_variance}\n")

print(f"Дифференциальная эволюция:\n"
      f"Среднее финальное значение функции: {de_mean}\n"
      f"Дисперсия финальных значений функции: {de_variance}")

target_value = 1.0
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

print(f"\n====== Часть 2. ======\n"
      f"Метод Ньютона:\n"
      f"Среднее время для достижения целевого значения: {newton_mean_time} секунд\n")

print(f"Дифференциальная эволюция:\n"
      f"Среднее время для достижения целевого значения: {de_mean_time} секунд")

de_mean_final_time = np.mean(de_times)
de_variance_final_time = np.var(de_times)

de_mean_final_result = np.mean(de_results)
de_variance_final_result = np.var(de_results)

print(f"\n====== Часть 3. ======\n"
      f"Дифференциальная эволюция:\n"
      f"Среднее время нахождения последнего локального экстремума: {de_mean_final_time} секунд\n"
      f"Дисперсия времени нахождения последнего локального экстремума: {de_variance_final_time}\n"
      f"Среднее значение последнего локального экстремума: {de_mean_final_result}\n"
      f"Дисперсия последнего локального экстремума: {de_variance_final_result}")

results_table = pd.DataFrame({
    'Метод': ['Метод Ньютона', 'Дифференциальная эволюция'],
    'Среднее значение функции': [newton_mean, de_mean],
    'Дисперсия значения функции': [newton_variance, de_variance],
    'Среднее время (сек.)': [newton_mean_time, de_mean_time],
    'Дисперсия времени (сек.)': [np.var(newton_times), de_variance_final_time]
})

print(f"\n====== Часть 4. ======\n"
      f"Сейчас должна произойти 3D-визуализация, либо ищите скриншоты с результатами в директории OTHER\n")

x = np.linspace(-5.12, 5.12, 400)
y = np.linspace(-5.12, 5.12, 400)
X, Y = np.meshgrid(x, y)
Z = rastrigin([X, Y])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_title('Часть 5. Визуализация для оптимизируемой функции')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

plt.figure(figsize=(12, 6))
for fitness in de_fitness_history:
    plt.plot(fitness, label='Пробежка')

plt.title('Визуализация для значений функции соответствия (целевой функции) в зависимости от числа поколений')
plt.xlabel('Поколение')
plt.ylabel('Значение функции')
plt.grid()
plt.show()

print("====== Часть 5. Таблица результатов ======\n")
print(results_table)
