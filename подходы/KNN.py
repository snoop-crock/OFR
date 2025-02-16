# примено тоже самое что и кот но быстрее

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Загрузка данных
# Убедитесь, что путь правильный
df = pd.read_excel('Dataset.xlsx', decimal=',')

# Заполнение пропусков средними значениями (если есть)
df.fillna(df.mean(), inplace=True)

# Разделение данных на признаки (X) и целевую переменную (y)
X = df.drop(columns=['OFR'])  # Все столбцы кроме целевой переменной
y = df['OFR']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Оптимизация гиперпараметров с помощью GridSearchCV
param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Манхэттенское расстояние (p=1) или Евклидово (p=2)
}

knn = KNeighborsRegressor()

# Обучение с кросс-валидацией
grid_search = GridSearchCV(knn, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Лучшие параметры
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Обучение модели с лучшими параметрами
best_knn = grid_search.best_estimator_

# Оценка модели
y_pred_train = best_knn.predict(X_train_scaled)
y_pred_test = best_knn.predict(X_test_scaled)

# Метрики модели
mse = mean_squared_error(y_test, y_pred_test)
rmse = mean_squared_error(y_test, y_pred_test, squared=False)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Final Model MSE: {mse}")
print(f"Final Model RMSE: {rmse}")
print(f"Final Model MAE: {mae}")
print(f"Final Model R²: {r2}")

# Сохранение модели в файл
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn, f)
