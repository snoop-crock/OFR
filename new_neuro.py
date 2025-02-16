# === БЛОК 1: Импорты библиотек ===
import optuna  # NEW
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour  # NEW
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # NEW

# === БЛОК 2: Функции для Optuna ===  # NEW


def objective(trial, X_train, X_val, y_train, y_val, cat_features):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 15000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 16),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 1e-3, 10),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
    }

    model = CatBoostRegressor(
        **params,
        cat_features=cat_features,
        verbose=500,
        early_stopping_rounds=1000,
        task_type="GPU",
        devices='0'
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)

# === БЛОК 3: Модифицированная функция обучения ===


def process_and_train_model(file_path, test_size=0.1, threshold_percentage=1,
                            model_save_path="catboost_model.cbm", n_trials=1000):  # NEW

    # ... (прежний код загрузки и подготовки данных без изменений) ...
    # Вплоть до раздела "Определение категориальных признаков"
    # --- Часть 1: Загрузка данных ---
    data = pd.read_excel(file_path, engine="openpyxl")

    # Проверка наличия целевого столбца 'Optimized Flow Rate'
    if 'OFR' in data.columns:
        X = data.drop(columns=['OFR'])
        y = data['OFR']
    else:
        raise KeyError("Файл не содержит столбца 'Optimized Flow Rate'.")

    # Вывод общей информации о датасете
    print("Информация о датасете:")
    print(data.info())

    # Проверка и обработка пропущенных значений
    missing_values = X.isnull().sum()
    print(f"Пропущенные значения:\n{missing_values}")
    X = X.dropna()  # Удаляем строки с пропущенными значениями
    y = y[X.index]  # Синхронизация индексов

    # --- Часть 2: Разделение данных на обучающую и тестовую выборки ---
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42)
    print(
        f"Данные разделены: {100 - test_size * 100}% на обучение, {test_size * 100}% на тест.")

    # Определение категориальных признаков
    cat_features = [i for i, col in enumerate(
        X.columns) if X[col].dtype == 'object']
    print(f"Категориальные признаки: {cat_features}")

    # === NEW: Оптимизация гиперпараметров ===
    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val, cat_features),
                   n_trials=n_trials,
                   n_jobs=-1 if n_trials > 10 else 1,
                   show_progress_bar=True)  # NEW: Показывает прогресс оптимизации

    # Сохранение исследования
    joblib.dump(study, 'optuna_study.pkl')
    print("="*50)
    print("Лучшие параметры:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # === NEW: Визуализация оптимизации ===
    fig = plot_optimization_history(study)
    fig.write_html("optimization_history.html")

    fig = plot_param_importances(study)
    fig.write_html("param_importances.html")

    fig = plot_contour(study)
    fig.write_html("contour_plot.html")

    # === Обучение финальной модели с лучшими параметрами ===
    best_params = study.best_params.copy()
    best_params.update({
        'cat_features': cat_features,
        'verbose': 500,
        'early_stopping_rounds': 1000,
        'task_type': "GPU",
        'devices': '0'
    })

    final_model = CatBoostRegressor(**best_params)
    final_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

    # ... (прежний код оценки модели и вывода метрик) ...
# --- Часть 4: Оценка модели и вывод метрик ---

    # Сохранение обученной модели
    final_model.save_model(model_save_path)
    print(f"Модель сохранена в файл: {model_save_path}")

    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

 # Добавление расчёта MAPE
    def calculate_mape(actual, predicted):
        actual, predicted = np.array(actual), np.array(predicted)
        return np.mean(np.abs((actual - predicted) / actual)) * 100

    mape = calculate_mape(y_val, y_pred)

    print(f"Среднеквадратичная ошибка (MSE): {mse:.5f}")
    print(f"Средняя абсолютная ошибка (MAE): {mae:.5f}")
    print(f"Коэффициент детерминации (R²): {r2:.5f}")
    print(f"Средняя абсолютная процентная ошибка (MAPE): {mape:.5}%")

    # --- Часть 5: Расчёт точности предсказаний и отклонений ---
    threshold = (threshold_percentage / 100) * y_val
    accuracy = np.mean(np.abs(y_val - y_pred) <= threshold) * 100

    absolute_errors = np.abs(y_val - y_pred)
    std_deviation = np.std(absolute_errors)

    print(
        f"Точность предсказаний: {accuracy:.2f}% (с учётом порога {threshold_percentage}%)")
    print(f"Среднее абсолютное отклонение: {absolute_errors.mean():.2f}")
    print(f"Стандартное отклонение абсолютных ошибок: {std_deviation:.2f}")

    # Добавление метрик в матрицу ошибок
    error_matrix = pd.DataFrame({
        'Истинное значение': y_val.values,
        'Предсказанное значение': y_pred,
        'Абсолютная ошибка': absolute_errors,
        # MAPE для каждого значения
        'MAPE': np.abs((y_val - y_pred) / y_val) * 100
    })

    print("\nМатрица ошибок (первые 10 строк):")
    print(error_matrix.head(10))

    # === NEW: Сохранение полного отчета ===
    full_report = pd.DataFrame({
        'trial': [t.number for t in study.trials],
        'value': [t.value for t in study.trials],
        'params': [t.params for t in study.trials],
        'datetime_start': [t.datetime_start for t in study.trials],
        'datetime_complete': [t.datetime_complete for t in study.trials]
    })

    full_report.to_csv('full_optimization_report.csv', index=False)
    print("\nПолный отчет по оптимизации сохранен в full_optimization_report.csv")

    return final_model, X_train, X_val, y_train, y_val, y_pred, cat_features, error_matrix, study  # NEW

# === БЛОК 4: Модифицированная функция визуализации ===


def visualize_results(model, X_train, y_train, y_val, y_pred, cat_features, error_matrix, study):  # NEW
    # --- Визуализация значимости признаков ---
    feature_importances = model.get_feature_importance(
        Pool(X_train, label=y_train, cat_features=cat_features))
    feature_names = X_train.columns
    importance_df = pd.DataFrame(
        {'Признак': feature_names, 'Значимость': feature_importances})
    importance_df = importance_df.sort_values(by='Значимость', ascending=False)
    print(importance_df)
    importance_df.to_excel("feature_importances.xlsx", index=False)
    print("Файл 'feature_importances.xlsx' успешно создан с данными о значимости признаков.")

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Значимость', y='Признак',
                data=importance_df, palette='viridis')
    plt.xlabel("Значимость признаков")
    plt.ylabel("Признаки")
    plt.title("Значимость признаков в прогнозировании оптимизированного дебита")
    plt.show()

    # --- Визуализация распределения ошибок ---
    absolute_errors = error_matrix['Абсолютная ошибка']

    plt.figure(figsize=(10, 6))
    plt.hist(absolute_errors, bins=30, color='skyblue', edgecolor='black')
    plt.axvline(absolute_errors.mean(), color='green',
                linestyle='--', label='Среднее отклонение')
    plt.axvline(np.std(absolute_errors), color='red',
                linestyle='--', label='Стандартное отклонение')
    plt.xlabel('Абсолютная ошибка')
    plt.ylabel('Частота')
    plt.title('Распределение абсолютных ошибок')
    plt.legend()
    plt.show()

    # --- Визуализация распределения MAPE ---
    mape_values = error_matrix['MAPE']

    plt.figure(figsize=(10, 6))
    plt.hist(mape_values, bins=30, color='orange', edgecolor='black')
    plt.axvline(mape_values.mean(), color='green',
                linestyle='--', label='Среднее MAPE')
    plt.axvline(np.median(mape_values), color='blue',
                linestyle='--', label='Медиана MAPE')
    plt.xlabel('MAPE (%)')
    plt.ylabel('Частота')
    plt.title('Распределение MAPE')
    plt.legend()
    plt.show()

    # --- Боксплот для анализа MAPE ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=mape_values, color='orange')
    plt.title('Боксплот для MAPE')
    plt.ylabel('MAPE (%)')
    plt.show()

    # --- Визуализация матрицы корреляции ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(error_matrix.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Корреляция между ошибками и значениями")
    plt.show()

    # === NEW: Визуализация результатов Optuna ===
    plt.figure(figsize=(12, 6))
    plot_optimization_history(study)
    plt.title("История оптимизации")
    plt.show()

    plt.figure(figsize=(12, 6))
    plot_param_importances(study)
    plt.title("Важность параметров")
    plt.show()

    plt.figure(figsize=(12, 6))
    plot_contour(study)
    plt.title("Контурные графики параметров")
    plt.tight_layout()
    plt.show()


# === БЛОК 5: Запуск ===
file_path = "Dataset.xlsx"
test_size = 0.1
threshold_percentage = 10
n_trials = 10  # NEW: Увеличено для более долгой работы

model, X_train, X_val, y_train, y_val, y_pred, cat_features, error_matrix, study = process_and_train_model(  # NEW
    file_path=file_path,
    test_size=test_size,
    threshold_percentage=threshold_percentage,
    n_trials=n_trials  # NEW
)

visualize_results(model, X_train, y_train, y_val, y_pred,
                  cat_features, error_matrix, study)  # NEW
