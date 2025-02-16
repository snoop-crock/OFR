# че к чему сам не понимаю

import joblib
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import warnings

# Загрузка данных
df = pd.read_excel('Dataset.xlsx')  # Замените на ваш файл

# Предположим, что ваш набор данных имеет целевую переменную 'OFR', а остальные - признаки
X = df.drop(columns=['OFR'])
y = df['OFR']

# Разделим на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

# Функция для тренировки модели


def objective(trial):
    # Определяем гиперпараметры для поиска
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 10, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        # исправлено здесь
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }

    # Создание модели с подобранными гиперпараметрами
    model = RandomForestRegressor(**param)

    # Кросс-валидация для оценки
    scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='neg_mean_squared_error')
    return scores.mean()


# Создание объекта для исследования с помощью Optuna
study = optuna.create_study(direction='minimize')  # Минимизируем MSE
study.optimize(objective, n_trials=100)

# Печать лучших параметров
print('Best parameters:', study.best_params)

# Обучаем модель с лучшими параметрами
best_params = study.best_params
best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = best_model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = sum(abs(y_test - y_pred)) / len(y_test)
r2 = best_model.score(X_test, y_test)

print(f"Final Model MSE: {mse}")
print(f"Final Model RMSE: {rmse}")
print(f"Final Model MAE: {mae}")
print(f"Final Model R²: {r2}")

# Сохранение модели
joblib.dump(best_model, 'random_forest_optuna_model.pkl')
