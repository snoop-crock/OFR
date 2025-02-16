# === БЛОК 1: Импорты библиотек === дерьма кусок блять -==
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# === БЛОК 2: Функция для Optuna ===


def objective(trial, X_train, X_val, y_train, y_val, cat_features):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 1.0),
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }

    # Обработка категориальных признаков
    if cat_features:
        params['categorical_feature'] = [f'column_{i}' for i in cat_features]

    train_data = lgb.Dataset(X_train, label=y_train,
                             categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=50)
        ]
    )

    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)

# === БЛОК 3: Функция обучения модели ===


def process_and_train_model(file_path, test_size=0.1, model_save_path="lgbm_model.txt", n_trials=100):
    # Загрузка данных
    data = pd.read_excel(file_path).dropna().reset_index(drop=True)
    target_col = 'OFR'

    # Разделение признаков и целевой переменной
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Проверим, есть ли пропущенные значения
    if X.isnull().any().any():
        print("Обнаружены пропущенные значения. Заполняем их средними значениями.")
        X = X.fillna(X.mean())  # Заполнение пропущенных значений средним

    # Определение категориальных признаков
    cat_features = []
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].nunique() < 100:
            X[col] = X[col].astype('category')
            cat_features.append(col)

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
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'device': 'gpu'
    })

    # Преобразуем параметры в формат для lgb.train
    train_data = lgb.Dataset(X_train, label=y_train,
                             categorical_feature=cat_features)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Обучаем модель с использованием callback для ранней остановки
    callbacks = [lgb.early_stopping(
        stopping_rounds=100), lgb.log_evaluation(period=50)]

    final_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=10000,
        callbacks=callbacks
    )

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

    return final_model, X_train, y_val, y_pred, error_matrix, optuna_report, study


# === БЛОК 4: Функция визуализации ===


def visualize_results(model, X_train, y_val, y_pred, error_matrix, optuna_report, study):
    # Важность признаков
    importance_df = pd.DataFrame({
        'Признак': X_train.columns,
        # Используем feature_importance() вместо feature_importances_
        'Важность': model.feature_importance()
    }).sort_values('Важность', ascending=False)

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

    # Сохранение результатов
    with pd.ExcelWriter('results.xlsx') as writer:
        importance_df.to_excel(
            writer, sheet_name="Feature_Importance", index=False)
        error_matrix.to_excel(writer, sheet_name="Error_Matrix", index=False)
        optuna_report.to_excel(writer, sheet_name="Optuna_Report", index=False)

    print("Все результаты сохранены в results.xlsx!")


# === БЛОК 5: Запуск ===
if __name__ == "__main__":
    model, X_train, y_val, y_pred, error_matrix, optuna_report, study = process_and_train_model(
        file_path="Dataset.xlsx",
        test_size=0.1,
        n_trials=50
    )
    visualize_results(model, X_train, y_val, y_pred,
                      error_matrix, optuna_report, study)
