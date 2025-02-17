# === БЛОК 1: Импорты библиотек ===
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.pruners import MedianPruner
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import gc

# === БЛОК 2: Функции для Optuna ===


def objective(trial, X_train, X_val, y_train, y_val, cat_features):
    try:
        params = {
            # Уменьшен максимальный предел
            'iterations': trial.suggest_int('iterations', 500, 2000),
            # Увеличен минимальный порог
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
            # Уменьшена максимальная глубина
            'depth': trial.suggest_int('depth', 4, 10),
            # Упрощен диапазон
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10),
            # Удален Lossguide
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
            'early_stopping_rounds': 200  # Добавлен параметр для ранней остановки
        }

        model = CatBoostRegressor(
            **params,
            cat_features=cat_features,
            verbose=0,  # Отключение вывода
            task_type="GPU",
            devices='0:0',  # Явное указание устройства
            allow_writing_files=False  # Отключение записи файлов
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=0,
            use_best_model=True
        )

        y_pred = model.predict(X_val)
        return mean_squared_error(y_val, y_pred)

    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        return float('inf')

    finally:
        del model
        gc.collect()

# === БЛОК 3: Основная функция обучения ===


def process_and_train_model(file_path, test_size=0.1, model_save_path="catboost_model.cbm", n_trials=100):
    # Загрузка и предобработка данных
    data = pd.read_excel(file_path, engine="openpyxl")
    data = data.dropna().reset_index(drop=True)

    if 'OFR' not in data.columns:
        raise KeyError("Файл не содержит столбца 'OFR'.")

    X = data.drop(columns=['OFR'])
    y = data['OFR']

    # Оптимизация использования памяти
    for col in X.select_dtypes(include=['object']):
        X[col] = X[col].astype('category')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Оптимизация гиперпараметров
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=MedianPruner()  # Добавлен прунер
    )

    study.optimize(
        lambda trial: objective(trial, X_train, X_val,
                                y_train, y_val, cat_features=[]),
        n_trials=n_trials,
        n_jobs=1,  # Принудительное использование одного job
        gc_after_trial=True  # Сборка мусора после каждого trial
    )

    # Обучение финальной модели с лучшими параметрами
    best_params = study.best_params
    best_params.update({
        'task_type': 'GPU',
        'devices': '0:0',
        'verbose': 100,
        'early_stopping_rounds': 200
    })

    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Сохранение результатов
    final_model.save_model(model_save_path)
    joblib.dump(study, 'optuna_study.pkl')

    return final_model, study


# === БЛОК 4: Запуск приложения ===
if __name__ == "__main__":
    file_path = "Dataset.xlsx"
    model, study = process_and_train_model(
        file_path=file_path,
        n_trials=500  # Пример увеличенного количества испытаний
    )
