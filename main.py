import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm

import matplotlib
matplotlib.use('TkAgg')  # или 'Qt5Agg'

# Данные
data = {
    'День': list(range(1, 13)),
    'Доллар': [28.12, 28.18, 28.13, 28.08, 28.06, 28.03, 28.02, 28.00, 27.99, 27.93, 27.95, 27.97],
    'Евро': [36.13, 35.97, 35.97, 36.00, 36.13, 36.28, 36.34, 36.47, 36.54, 36.50, 36.52, 36.54],
    'Иена': [26.97, 26.80, 26.77, 26.63, 26.53, 26.70, 26.67, 26.63, 26.60, 26.50, 26.55, 26.52],
    'Фунт': [52.63, 52.32, 52.26, 52.28, 52.43, 52.58, 52.90, 52.99, 52.81, 52.89, 52.62, 52.67]
}

df = pd.DataFrame(data)

# Функция для теста Фаррара – Глоубера
def farra_glob_test(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

# Проверка мультиколлинеарности между евро, иеной и фунтом
variables = ['Евро', 'Иена', 'Фунт']
results = {}

for var in variables:
    X = df[variables].drop(var, axis=1)  # Убираем одну переменную
    y = df[var]
    model = farra_glob_test(X, y)
    results[var] = model.summary()

# Вывод результатов
for var, result in results.items():
    print(f"Результаты для {var}:\n{result}\n")

# Линейная регрессия
X = df[['Евро', 'Иена', 'Фунт']]
y = df['Доллар']
X = sm.add_constant(X)  # добавляем константу

model = sm.OLS(y, X).fit()
print(model.summary())

# Влияние факторов
coefficients = model.params[1:]  # исключаем константу
print("Коэффициенты влияния факторов:")
print(coefficients)


# Остатки модели
residuals = model.resid
dw_stat = durbin_watson(residuals)
print(f"Статистика Дурбина-Уотсона: {dw_stat}")

# Функция для построения и вывода результатов регрессионной модели
def run_regression(dependent_var, independent_vars):
    X = df[independent_vars]
    y = df[dependent_var]
    X = sm.add_constant(X)  # добавляем константу
    model = sm.OLS(y, X).fit()
    print(f"Результаты регрессии для {dependent_var} от {', '.join(independent_vars)}:\n")
    print(model.summary())
    print("\n" + "="*80 + "\n")
    # Графики
    # остатков
    residuals = model.resid

    # Гистограмма остатков
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=10, edgecolor='k', alpha=0.7)
    plt.title('Гистограмма остатков')
    plt.xlabel('Остатки')
    plt.ylabel('Частота')

    # Q-Q график
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q график остатков')

    plt.tight_layout()
    plt.show()

    return model

# 1. Регрессия: Доллар от Евро
model_euro = run_regression('Доллар', ['Евро'])

# 2. Регрессия: Доллар от Иены
model_yen = run_regression('Доллар', ['Иена'])

# 3. Регрессия: Доллар от Иены и Фунта
model_yen_pound = run_regression('Доллар', ['Иена', 'Фунт'])

# Прогнозирование курса доллара на следующие два дня
# Предположим, что курсы евро, иены и фунта на следующие два дня будут следующими:
future_data = pd.DataFrame({
    'Евро': [36.55, 36.60],  # Прогнозируемые значения
    'Иена': [26.55, 26.50],
    'Фунт': [52.70, 52.75]
})

# Прогноз для модели Доллар от Евро
future_data_with_const = sm.add_constant(future_data[['Евро']])
predictions_euro = model_euro.predict(future_data_with_const)

# Прогноз для модели Доллар от Иены
predictions_yen = model_yen.predict(sm.add_constant(future_data[['Иена']]))

# Прогноз для модели Доллар от Иены и Фунта
predictions_yen_pound = model_yen_pound.predict(sm.add_constant(future_data[['Иена', 'Фунт']]))

# Вывод прогнозов
print("Прогноз курса доллара на следующие два дня (от Евро):", predictions_euro)
print("Прогноз курса доллара на следующие два дня (от Иены):", predictions_yen)
print("Прогноз курса доллара на следующие два дня (от Иены и Фунта):", predictions_yen_pound)
