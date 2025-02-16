# я заебался его настраивать


import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    "n_trials": 100,
    "test_size": 0.1,
    "early_stopping_rounds": 100,
    "seed": 42,
    "metrics": ["rmse", "mae", "r2", "mape"],
    "plot_params": {
        "figsize": (10, 6),
        "importance_color": "skyblue",
        "hist_bins": 30,
    },
}

# Параметры для поиска Optuna
OPTUNA_PARAMS = {
    "learning_rate": (1e-4, 0.1, "log"),
    "max_depth": (4, 10),
    "lambda": (1e-2, 10.0),
    "alpha": (1e-2, 10.0),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "gamma": (0.1, 1.0),
    "scale_pos_weight": (1, 10),
}


@dataclass
class TrainingResults:
    model: xgb.XGBRegressor
    X_train: pd.DataFrame
    y_val: pd.Series
    y_pred: np.ndarray
    error_matrix: pd.DataFrame
    optuna_report: pd.DataFrame
    study: optuna.Study
    metrics: Dict[str, float]


def objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    fixed_params: Dict[str, Any],
) -> float:
    """
    Целевая функция для оптимизации гиперпараметров с помощью Optuna.

    Args:
        trial: Объект испытания Optuna
        X_train: Обучающие данные
        X_val: Валидационные данные
        y_train: Целевая переменная обучающих данных
        y_val: Целевая переменная валидационных данных
        fixed_params: Фиксированные параметры модели

    Returns:
        Значение метрики MSE для оптимизации
    """
    params = {}
    for param_name, (low, high, *rest) in OPTUNA_PARAMS.items():
        if rest and rest[0] == "log":
            params[param_name] = trial.suggest_float(
                param_name, low, high, log=True
            )
        else:
            params[param_name] = trial.suggest_float(param_name, low, high)

    params.update(fixed_params)
    model = xgb.XGBRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=fixed_params["early_stopping_rounds"],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    return mean_squared_error(y_val, y_pred)


def calculate_metrics(
    y_true: pd.Series, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Вычисление метрик качества модели.

    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения

    Returns:
        Словарь с вычисленными метриками
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    try:
        metrics["mape"] = (
            np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        )
    except ZeroDivisionError:
        logger.warning(
            "Обнаружены нулевые значения в y_true. MAPE не рассчитан.")
        metrics["mape"] = np.nan

    return metrics


def create_error_matrix(
    y_true: pd.Series, y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Создание матрицы ошибок.

    Args:
        y_true: Истинные значения
        y_pred: Предсказанные значения

    Returns:
        DataFrame с различными видами ошибок
    """
    return pd.DataFrame(
        {
            "Истинное значение": y_true,
            "Предсказанное": y_pred,
            "Абс. ошибка": np.abs(y_true - y_pred),
            "Отн. ошибка": np.abs((y_true - y_pred) / y_true),
        }
    )


def process_and_train_model(
    file_path: str,
    target_col: str = "OFR",
    test_size: float = DEFAULT_CONFIG["test_size"],
    model_save_path: Optional[str] = "xgboost_model.model",
    n_trials: int = DEFAULT_CONFIG["n_trials"],
) -> TrainingResults:
    """
    Основной пайплайн обработки данных и обучения модели.

    Args:
        file_path: Путь к файлу с данными
        target_col: Название целевой переменной
        test_size: Доля тестовой выборки
        model_save_path: Путь для сохранения модели
        n_trials: Количество испытаний Optuna

    Returns:
        Объект TrainingResults с результатами обучения
    """
    try:
        data = pd.read_excel(file_path).dropna().reset_index(drop=True)
    except FileNotFoundError:
        logger.error(f"Файл {file_path} не найден")
        raise

    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=DEFAULT_CONFIG["seed"],
        shuffle=True
    )

    fixed_params = {
        "n_estimators": 10000,
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "eval_metric": "rmse",
        "early_stopping_rounds": DEFAULT_CONFIG["early_stopping_rounds"],
        "random_state": DEFAULT_CONFIG["seed"],
    }

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=DEFAULT_CONFIG["seed"]),
    )
    study.optimize(
        lambda trial: objective(trial, X_train, X_val,
                                y_train, y_val, fixed_params),
        n_trials=n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    best_params = {**study.best_params, **fixed_params}
    final_model = xgb.XGBRegressor(**best_params)

    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=fixed_params["early_stopping_rounds"],
        verbose=False,
    )

    if model_save_path:
        final_model.save_model(model_save_path)
        logger.info(f"Модель сохранена в {model_save_path}")

    y_pred = final_model.predict(X_val)
    metrics = calculate_metrics(y_val, y_pred)

    logger.info(
        "\nМетрики модели:\n"
        f"MSE: {metrics['mse']:.3f}\n"
        f"MAE: {metrics['mae']:.3f}\n"
        f"R²: {metrics['r2']:.3f}\n"
        f"MAPE: {metrics.get('mape', 'N/A'):.3f}%"
    )

    return TrainingResults(
        model=final_model,
        X_train=X_train,
        y_val=y_val,
        y_pred=y_pred,
        error_matrix=create_error_matrix(y_val, y_pred),
        optuna_report=study.trials_dataframe(),
        study=study,
        metrics=metrics,
    )


def visualize_results(results: TrainingResults) -> None:
    """
    Визуализация результатов обучения.

    Args:
        results: Объект с результатами обучения
    """
    # Важность признаков
    feat_imp = results.model.feature_importances_
    importance_df = pd.DataFrame({
        "Признак": results.X_train.columns,
        "Важность": feat_imp
    }).sort_values("Важность", ascending=False)

    plt.figure(figsize=DEFAULT_CONFIG["plot_params"]["figsize"])
    sns.barplot(
        x="Важность",
        y="Признак",
        data=importance_df,
        color=DEFAULT_CONFIG["plot_params"]["importance_color"],
    )
    plt.title("Важность признаков")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    # Визуализация Optuna
    for plot_func in [plot_optimization_history, plot_param_importances, plot_contour]:
        fig = plot_func(results.study)
        fig.write_image(f"{plot_func.__name__}.png")

    # Гистограмма ошибок
    plt.figure(figsize=DEFAULT_CONFIG["plot_params"]["figsize"])
    sns.histplot(
        results.error_matrix["Абс. ошибка"],
        kde=True,
        bins=DEFAULT_CONFIG["plot_params"]["hist_bins"],
    )
    plt.title("Распределение абсолютных ошибок")
    plt.tight_layout()
    plt.savefig("error_distribution.png")
    plt.close()

    # Сохранение результатов
    with pd.ExcelWriter("results.xlsx") as writer:
        importance_df.to_excel(
            writer, sheet_name="Важность_признаков", index=False)
        results.error_matrix.to_excel(writer, sheet_name="Ошибки", index=False)
        results.optuna_report.to_excel(
            writer, sheet_name="Optuna", index=False)

    logger.info("Все результаты сохранены в файлы *.png и results.xlsx")


if __name__ == "__main__":
    try:
        results = process_and_train_model(
            file_path="Dataset.xlsx",
            test_size=0.1,
            n_trials=5
        )
        visualize_results(results)
    except Exception as e:
        logger.error(f"Ошибка в выполнении: {str(e)}")
        raise
