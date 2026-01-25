import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

df = pd.read_excel('Семинар 4.xlsx')

df.replace(0, np.nan, inplace=True)

df.interpolate(inplace=True)

def detect_outliers_zscore(df_column, threshold=3):
    z_scores = zscore(df_column)
    return np.abs(z_scores) > threshold

outliers_y = detect_outliers_zscore(df[1])
df_cleaned = df[~outliers_y]

outliers_x = detect_outliers_zscore(df_cleaned[0])
mean_x = df_cleaned[0].mean()
df_cleaned.loc[outliers_x, 0] = mean_x  # Замена выбросов средним значением

z_scores_x = zscore(df_cleaned[0])

z_scores_y = zscore(df_cleaned[1])

df_cleaned = df_cleaned[np.abs(z_scores_y) <= 3]

# Замена выбросов в столбце X на среднее значение
mean_x = df_cleaned[0].mean()
df_cleaned.loc[np.abs(z_scores_x) > 3, 0] = mean_x

z_scores_x = zscore(df_cleaned[0])
print("Максимальный Z-score для столбца X:", np.abs(z_scores_x).max())  # Если >3, значит выбросы есть

# Проверка выбросов в столбце Y
z_scores_y = zscore(df_cleaned[1])
print("Максимальный Z-score для столбца Y:", np.abs(z_scores_y).max())  # Если >3, значит выбросы есть

# Удаление строк, где Z-оценка для столбца Y больше 3
df_cleaned = df_cleaned[np.abs(z_scores_y) <= 3]
z_scores_y = zscore(df_cleaned[1])
print("Максимальный Z-score для столбца Y:", np.abs(z_scores_y).max())  # Если >3, значит выбросы есть

plt.figure(figsize=(8, 6))
plt.hist2d(df_cleaned[0], df_cleaned[1], bins=50, cmap='coolwarm')
plt.colorbar(label='Частота')
plt.title('Тепловая диаграмма рассеивания')
plt.xlabel('Значения X')
plt.ylabel('Значения Y')
plt.show()

df_cleaned[[0, 1]].hist(figsize=(10, 5), bins=10)
plt.suptitle('Гистограммы столбцов X и Y')
plt.show()

df_sample = df_cleaned.sample(8)

print(df_cleaned[df_cleaned.isnull().any(axis=1)])  # Интерполяция пропущенных значений (нулей)
print(f'До очистки: {len(df)}, после очистки: {len(df_cleaned)}') # Удаление строк с выбросами в столбце Y

plt.figure(figsize=(6, 6))
plt.scatter(df_sample[0], df_sample[1], color='blue', marker='o')
plt.title('Диаграмма рассеяния для случайной выборки из 8 значений')
plt.xlabel('Значения X')
plt.ylabel('Значения Y')
plt.grid(True)
plt.show()

print(df_cleaned)
