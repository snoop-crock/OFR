# === БЛОК 1: Импорты библиотек ===
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# === БЛОК 2: Функция для Optuna ===


def objective(trial, X_train, X_val, y_train, y_val, cat_features):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 8000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 10),
        'random_strength': trial.suggest_float('random_strength', 0.1, 1.5),
        'grow_policy': trial.suggest_categorical('grow_policy', ['Lossguide', 'Depthwise']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bernoulli', 'Bayesian']),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 20),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 10),
        'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE', 'Quantile', 'Huber']),
        'used_ram_limit': '4gb',
        'task_type': 'GPU',
        'devices': '0:0'
    }

    if params['bootstrap_type'] == 'Bayesian':
        params['bagging_temperature'] = trial.suggest_float(
            'bagging_temperature', 0.1, 10)

    if params['bootstrap_type'] == 'Bernoulli':
        params['subsample'] = trial.suggest_float('subsample', 0.6, 0.95)

    model = CatBoostRegressor(
        **params,
        cat_features=cat_features,
        verbose=0,
        early_stopping_rounds=100
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              verbose=False, use_best_model=True)
    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)

# === БЛОК 3: Функция обучения модели ===


def process_and_train_model(file_path, test_size=0.1, model_save_path="catboost_model.cbm", n_trials=100):

    # Загрузка данных
    data = pd.read_excel(file_path).dropna().reset_index(drop=True)
    target_col = 'OFR'

    # Разделение признаков и целевой переменной
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Определение категориальных признаков
    cat_features = [i for i, col in enumerate(
        X.columns) if X[col].dtype == 'object']
    if cat_features:
        X[cat_features] = X[cat_features].astype('category')
        cat_features = list(map(int, cat_features))  # Преобразуем в int

    # Конвертация числовых данных в float32
    X = X.astype('float32')

    # Разделение данных
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True)

    # Оптимизация гиперпараметров
    study = optuna.create_study(
        direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val, cat_features),
                   n_trials=n_trials, gc_after_trial=True, show_progress_bar=True)

    # Обучение финальной модели с лучшими параметрами
    best_params = study.best_params
    best_params.update({'used_ram_limit': '4gb', 'task_type': 'GPU',
                       'devices': '0:0', 'allow_writing_files': False})

    final_model = CatBoostRegressor(
        **best_params, cat_features=cat_features, verbose=100, early_stopping_rounds=100)
    final_model.fit(X_train, y_train, eval_set=(
        X_val, y_val), metric_period=50)

    # Сохранение модели
    final_model.save_model(model_save_path)

    # Предсказание
    y_pred = final_model.predict(X_val)

    # Метрики
    mse, mae, r2 = mean_squared_error(y_val, y_pred), mean_absolute_error(
        y_val, y_pred), r2_score(y_val, y_pred)
    mape = np.mean(np.abs((y_val - y_pred) / y_val) * 100)

    print(f"\nMSE: {mse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}\nMAPE: {mape:.3f}%")

    # Матрица ошибок
    error_matrix = pd.DataFrame({
        'Истинное значение': y_val,
        'Предсказанное': y_pred,
        'Абс. ошибка': np.abs(y_val - y_pred),
        'MAPE': np.abs((y_val - y_pred) / y_val) * 100
    })

    # Оптимизация Optuna
    optuna_report = pd.DataFrame(study.trials)

    return final_model, X_train, y_val, y_pred, cat_features, error_matrix, optuna_report, study

# === БЛОК 4: Функция визуализации ===


def visualize_results(model, X_train, y_val, y_pred, cat_features, error_matrix, optuna_report, study):

    # Важность признаков
    feat_imp = model.get_feature_importance(
        Pool(X_train, cat_features=cat_features))
    importance_df = pd.DataFrame({'Признак': X_train.columns, 'Важность': feat_imp}).sort_values(
        'Важность', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Важность', y='Признак', data=importance_df)
    plt.title('Важность признаков')
    plt.show()

    # Визуализация параметров Optuna
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
    plot_contour(study).show()

    # Гистограмма ошибок
    plt.figure(figsize=(10, 6))
    sns.histplot(error_matrix['Абс. ошибка'], kde=True)
    plt.title('Распределение абсолютных ошибок')
    plt.show()

    # === Сохранение результатов в Excel ===
    with pd.ExcelWriter('results.xlsx') as writer:
        importance_df.to_excel(
            writer, sheet_name="Feature_Importance", index=False)
        error_matrix.to_excel(writer, sheet_name="Error_Matrix", index=False)
        optuna_report.to_excel(writer, sheet_name="Optuna_Report", index=False)

    print("Все результаты сохранены в results.xlsx!")


# === БЛОК 5: Запуск ===
if __name__ == "__main__":
    model, X_train, y_val, y_pred, cat_features, error_matrix, optuna_report, study = process_and_train_model(
        file_path="Dataset.xlsx",
        test_size=0.1,
        n_trials=10
    )
    visualize_results(model, X_train, y_val, y_pred,
                      cat_features, error_matrix, optuna_report, study)
