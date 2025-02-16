"""
CatBoost Model Optimization Pipeline with Hyperparameter Tuning and Reporting

This script implements a complete machine learning pipeline for regression tasks
using CatBoost and Optuna, including data processing, model optimization,
visualization, and PDF report generation.
"""

import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import tempfile
import os
import sys
from tkinter import filedialog, messagebox
from catboost.utils import get_gpu_device_count
from typing import Tuple, Dict, Optional, Any, List
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import tkinter as tk

# region Constants and Configuration


class Config:
    """Central configuration class for model parameters and settings"""

    # File paths
    FONT_PATH = 'DejaVuSans.ttf'
    DEFAULT_MODEL_NAME = 'optimized_model.cbm'
    STUDY_NAME = 'optimization_study.pkl'

    # Visualization
    PLOT_SIZE = (190, 100)
    CORRELATION_TOP_FEATURES = 3
    CORRELATION_BOTTOM_FEATURES = 3

    # Optimization
    OPTUNA_TRIALS = 50
    EARLY_STOPPING_ROUNDS = 500
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    HYPERPARAM_SPACE = {
        'iterations': (1000, 5000),
        'learning_rate': (1e-5, 0.5),
        'depth': (4, 12),
        'l2_leaf_reg': (1e-3, 50),
        'bagging_temperature': (0.0, 15.0),
        'random_strength': (1e-3, 20)
    }

    # Reporting
    METRIC_NAMES = {
        'MSE': 'Mean Squared Error',
        'MAE': 'Mean Absolute Error',
        'R2': 'R-squared',
        'MAPE': 'Mean Absolute Percentage Error'
    }
    REPORT_SECTIONS = [
        "Ключевые метрики",
        "Визуализации данных",
        "Анализ важности признаков",
        "Параметры модели",
        "Анализ ошибок"
    ]


class Style:
    """Visual style configuration for reports and plots"""

    # Colors
    PRIMARY_COLOR = (33, 150, 243)
    SECONDARY_COLOR = (245, 245, 245)

    # Font sizes
    TITLE_FONT_SIZE = 24
    SECTION_FONT_SIZE = 16
    TABLE_HEADER_FONT_SIZE = 11
    TABLE_BODY_FONT_SIZE = 10

    # Layout
    MARGIN = 10
    ROW_HEIGHT = 8
    PLOT_DPI = 300


# region Exceptions
class PipelineError(Exception):
    """Base class for all pipeline exceptions"""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


class DataValidationError(PipelineError):
    """Data validation or processing error"""


class ModelTrainingError(PipelineError):
    """Model training or optimization error"""


class ReportGenerationError(PipelineError):
    """Report generation error"""
# endregion


# region Core Components
class DataHandler:
    """Handles data loading, validation, and preprocessing"""

    @classmethod
    def load_and_validate(cls, file_path: str) -> pd.DataFrame:
        """Load and validate input data file"""
        try:
            data = pd.read_excel(file_path)
            cls._validate_data_structure(data)
            cls._validate_data_content(data)
            return data
        except (FileNotFoundError, pd.errors.ParserError) as e:
            raise DataValidationError(
                f"Data loading failed: {str(e)}", e) from e

    @staticmethod
    def _validate_data_structure(data: pd.DataFrame) -> None:
        """Validate basic data structure"""
        if data.empty:
            raise DataValidationError("Input file is empty")
        if 'OFR' not in data.columns:
            raise DataValidationError("Target column 'OFR' not found")

    @staticmethod
    def _validate_data_content(data: pd.DataFrame) -> None:
        """Validate data content quality"""
        if data['OFR'].isnull().any():
            raise DataValidationError("Missing values in target column 'OFR'")
        if (data['OFR'] <= 0).all():
            raise DataValidationError("All target values are non-positive")

    @classmethod
    def preprocess_data(cls, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data for modeling"""
        filtered_data = data[data['OFR'] > 0].copy()
        if filtered_data.empty:
            raise DataValidationError("No valid data after filtering OFR > 0")

        y = np.log1p(filtered_data['OFR'])
        X = filtered_data.drop(columns=['OFR'])
        return X, y


class ModelOptimizer:
    """Handles model training and hyperparameter optimization"""

    def __init__(self):
        self._init_hardware()
        self.best_model = None

    def _init_hardware(self) -> None:
        """Initialize hardware settings for training"""
        self.use_gpu = get_gpu_device_count() > 0
        self.task_type = "GPU" if self.use_gpu else "CPU"
        self.devices = '0:0' if self.use_gpu else '0'
        logging.info(f"Using {self.task_type} for training")

    def optimize(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Full optimization workflow"""
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        study = self._run_optimization(X_train, X_val, y_train, y_val)
        self.best_model = self._train_final_model(
            study, X_train, y_train, X_val, y_val)
        return self._evaluate_model(self.best_model, X_val, y_val)

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Split data into training and validation sets"""
        return train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )

    def _run_optimization(self, X_train, X_val, y_train, y_val) -> optuna.Study:
        """Run Optuna hyperparameter optimization"""
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE)
        study.optimize(
            lambda trial: self._objective(
                trial, X_train, X_val, y_train, y_val),
            n_trials=Config.OPTUNA_TRIALS
        )
        return study

    def _objective(self, trial, X_train, X_val, y_train, y_val) -> float:
        """Objective function for Optuna optimization"""
        params=self._get_trial_params(trial)
        model=self._create_model(params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose=False
        )
        return mean_squared_error(y_val, model.predict(X_val))

    def _get_trial_params(self, trial: optuna.Trial) -> Dict:
        """Generate parameters for Optuna trial"""
        return {
            'iterations': trial.suggest_int('iterations', *Config.HYPERPARAM_SPACE['iterations']),
            'learning_rate': trial.suggest_float('learning_rate', *Config.HYPERPARAM_SPACE['learning_rate'], log=True),
            'depth': trial.suggest_int('depth', *Config.HYPERPARAM_SPACE['depth']),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', *Config.HYPERPARAM_SPACE['l2_leaf_reg'], log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', *Config.HYPERPARAM_SPACE['bagging_temperature']),
            'random_strength': trial.suggest_float('random_strength', *Config.HYPERPARAM_SPACE['random_strength']),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
            'task_type': self.task_type,
            'devices': self.devices,
            'verbose': False
        }

    def _create_model(self, params: Dict) -> CatBoostRegressor:
        """Create CatBoost model instance"""
        return CatBoostRegressor(**params, loss_function='RMSE')

    def _train_final_model(self, study, X_train, y_train, X_val, y_val) -> CatBoostRegressor:
        """Train final model with best parameters"""
        try:
            params=study.best_params.copy()
            params.update({
                'task_type': self.task_type,
                'devices': self.devices,
                'od_type': 'Iter',
                'od_wait': 500,
                'verbose': 100
            })
            model=CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            return model
        except Exception as e:
            raise ModelTrainingError("Final model training failed", e) from e

    @ staticmethod
    def _evaluate_model(model, X_val, y_val) -> Dict:
        """Calculate evaluation metrics"""
        y_pred=model.predict(X_val)
        return {
            'MSE': mean_squared_error(y_val, y_pred),
            'MAE': mean_absolute_error(y_val, y_pred),
            'R2': r2_score(y_val, y_pred),
            'MAPE': np.mean(np.abs((y_val - y_pred) / y_val)) * 100
        }


class Visualizer:
    """Handles data visualization and plot generation"""

    @ staticmethod
    def create_plots(X: pd.DataFrame, y: pd.Series, model: CatBoostRegressor,
                    y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, str]:
        """Generate all diagnostic plots"""
        plots={}
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                plots['correlation']=Visualizer._plot_correlation_matrix(
                    X, tmpdir)
                plots['dependencies']=Visualizer._plot_feature_dependencies(
                    X, y, tmpdir)
                plots['importance']=Visualizer._plot_feature_importance(
                    model, X, tmpdir)
                plots['predictions']=Visualizer._plot_prediction_analysis(
                    y_true, y_pred, tmpdir)
            except Exception as e:
                raise ReportGenerationError("Plot generation failed", e) from e
        return plots

    @ staticmethod
    def _plot_correlation_matrix(X: pd.DataFrame, tmpdir: str) -> str:
        """Plot feature correlation matrix"""
        plt.figure(figsize=(12, 10))
        numeric_cols=X.select_dtypes(include=np.number).columns
        corr_matrix=X[numeric_cols].corr()
        mask=np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False)
        plt.title('Feature Correlation Matrix')
        return Visualizer._save_plot(tmpdir, 'correlation.png')

    @ staticmethod
    def _plot_feature_dependencies(X: pd.DataFrame, y: pd.Series, tmpdir: str) -> str:
        """Plot top feature dependencies"""
        plt.figure(figsize=(15, 10))
        numeric_cols=X.select_dtypes(include=np.number).columns
        corr=X[numeric_cols].apply(
            lambda x: x.corr(y)).sort_values(ascending=False)

        for idx, feature in enumerate(corr.head(Config.CORRELATION_TOP_FEATURES).index):
            plt.subplot(2, 3, idx+1)
            sns.regplot(x=X[feature], y=y, scatter_kws={'alpha': 0.3})
            plt.title(f'{feature}\nCorrelation: {corr[feature]:.2f}')

        for idx, feature in enumerate(corr.tail(Config.CORRELATION_BOTTOM_FEATURES).index):
            plt.subplot(2, 3, idx+4)
            sns.regplot(x=X[feature], y=y, scatter_kws={'alpha': 0.3})
            plt.title(f'{feature}\nCorrelation: {corr[feature]:.2f}')

        plt.tight_layout()
        return Visualizer._save_plot(tmpdir, 'dependencies.png')

    @ staticmethod
    def _plot_feature_importance(model: CatBoostRegressor, X: pd.DataFrame, tmpdir: str) -> str:
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))
        importance=pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.get_feature_importance()
        }).sort_values('Importance', ascending=False).head(10)

        sns.barplot(x='Importance', y='Feature',
                    data=importance, palette='viridis')
        plt.title('Top 10 Feature Importance')
        return Visualizer._save_plot(tmpdir, 'importance.png')

    @ staticmethod
    def _plot_prediction_analysis(y_true: np.ndarray, y_pred: np.ndarray, tmpdir: str) -> str:
        """Plot prediction diagnostics"""
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [
                 y_true.min(), y_true.max()], '--r')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        return Visualizer._save_plot(tmpdir, 'predictions.png')

    @ staticmethod
    def _save_plot(tmpdir: str, filename: str) -> str:
        """Save plot to temporary directory"""
        path=os.path.join(tmpdir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=Style.PLOT_DPI)
        plt.close()
        return path


class ReportGenerator:
    """Generates PDF reports with model analysis"""

    def __init__(self):
        self.pdf=FPDF()
        self._configure_pdf()
        self.section_num=1

    def _configure_pdf(self) -> None:
        """Initialize PDF document settings"""
        self.pdf.add_page()
        self.pdf.set_auto_page_break(True, margin=15)
        self._register_fonts()

    def _register_fonts(self) -> None:
        """Register custom fonts with fallback"""
        try:
            self.pdf.add_font('DejaVu', '', Config.FONT_PATH)
            self.pdf.set_font('DejaVu', '', 12)
        except RuntimeError:
            self.pdf.set_font('Helvetica', '', 12)

    def generate(self, metrics: Dict, model: CatBoostRegressor,
                plots: Dict, output_path: str) -> None:
        """Generate complete PDF report"""
        try:
            self._add_title_page()
            self._add_metrics_section(metrics)
            self._add_visualizations(plots)
            self._add_feature_analysis(model)
            self._add_model_parameters(model)
            self.pdf.output(output_path)
        except Exception as e:
            raise ReportGenerationError("Failed to generate report", e) from e

    def _add_title_page(self) -> None:
        """Add title page to report"""
        self.pdf.add_page()
        self.pdf.set_font_size(Style.TITLE_FONT_SIZE)
        self.pdf.cell(0, 20, "Model Analysis Report", align='C',
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.pdf.ln(20)

    def _add_metrics_section(self, metrics: Dict) -> None:
        """Add metrics table section"""
        self._start_section("Key Metrics")
        self.pdf.set_fill_color(*Style.SECONDARY_COLOR)

        # Table header
        self.pdf.set_font(style='B', size=Style.TABLE_HEADER_FONT_SIZE)
        self.pdf.cell(40, Style.ROW_HEIGHT, "Metric", border=1, fill=True)
        self.pdf.cell(40, Style.ROW_HEIGHT, "Value",
                      border=1, fill=True, ln=True)

        # Table rows
        self.pdf.set_font(style='', size=Style.TABLE_BODY_FONT_SIZE)
        for metric, value in metrics.items():
            self.pdf.cell(40, Style.ROW_HEIGHT,
                          Config.METRIC_NAMES.get(metric, metric), border=1)
            self.pdf.cell(40, Style.ROW_HEIGHT, self._format_value(
                metric, value), border=1, ln=True)

        self.pdf.ln(10)

    def _add_visualizations(self, plots: Dict) -> None:
        """Add visualization section"""
        self._start_section("Data Visualizations")
        for plot_name, path in plots.items():
            if os.path.exists(path):
                self.pdf.image(path, x=Style.MARGIN, w=190)
                self.pdf.ln(5)

    def _start_section(self, title: str) -> None:
        """Start new report section with proper numbering and spacing"""
        # Add section header
        self.pdf.set_font(style='B', size=Style.SECTION_FONT_SIZE)
        self.pdf.set_text_color(*Style.PRIMARY_COLOR)
        self.pdf.cell(0, 10, f"{self.section_num}. {title}", ln=1)

        # Add section spacing
        self.pdf.ln(5)

        # Update section counter
        self.section_num += 1

        # Reset text color for content
        self.pdf.set_text_color(0, 0, 0)  # Black color
        self.pdf.set_font(style='', size=Style.TABLE_BODY_FONT_SIZE)

            def _add_feature_analysis(self, model: CatBoostRegressor) -> None:
        """Add feature importance analysis section"""
        self._start_section("Feature Importance Analysis")

        # Get feature importance data
        importance=model.get_feature_importance()
        features=model.feature_names_
        df_importance=pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(10)

        # Create importance table
        self.pdf.set_font(style='B', size=Style.TABLE_HEADER_FONT_SIZE)
        self.pdf.cell(80, Style.ROW_HEIGHT, "Feature", border=1, fill=True)
        self.pdf.cell(30, Style.ROW_HEIGHT, "Importance",
                      border=1, fill=True, ln=True)

        self.pdf.set_font(style='', size=Style.TABLE_BODY_FONT_SIZE)
        for _, row in df_importance.iterrows():
            self.pdf.cell(80, Style.ROW_HEIGHT, row['Feature'], border=1)
            self.pdf.cell(30, Style.ROW_HEIGHT,
                          f"{row['Importance']:.4f}", border=1, ln=True)

        self.pdf.ln(10)

    def _add_model_parameters(self, model: CatBoostRegressor) -> None:
        """Add model parameters section"""
        self._start_section("Model Configuration")

        params=model.get_params()
        excluded_params=['task_type', 'devices', 'verbose']

        self.pdf.set_font(style='B', size=Style.TABLE_HEADER_FONT_SIZE)
        self.pdf.cell(60, Style.ROW_HEIGHT, "Parameter", border=1, fill=True)
        self.pdf.cell(60, Style.ROW_HEIGHT, "Value",
                      border=1, fill=True, ln=True)

        self.pdf.set_font(style='', size=Style.TABLE_BODY_FONT_SIZE)
        for param, value in params.items():
            if param not in excluded_params and not param.startswith('_'):
                self.pdf.cell(60, Style.ROW_HEIGHT, param, border=1)
                self.pdf.cell(60, Style.ROW_HEIGHT,
                              str(value), border=1, ln=True)

        self.pdf.ln(10)

    def _format_value(self, metric: str, value: float) -> str:
        """Format metric value based on type"""
        if metric == 'MAPE':
            return f"{value:.2f}%"
        return f"{value:.4f}"


class PipelineController:
    """Orchestrates the complete ML pipeline"""

    def __init__(self):
        self.data_handler=DataHandler()
        self.optimizer=ModelOptimizer()
        self.visualizer=Visualizer()
        self.report_generator=ReportGenerator()

    def execute_pipeline(self, input_path: str) -> None:
        """Execute complete ML pipeline"""
        try:
            # Data processing stage
            raw_data=self.data_handler.load_and_validate(input_path)
            X, y=self.data_handler.preprocess_data(raw_data)

            # Model training stage
            metrics=self.optimizer.optimize(X, y)

            # Visualization stage
            plots=self.visualizer.create_plots(
                X, y, self.optimizer.best_model,
                self.optimizer.best_model.eval_set[0][1],
                self.optimizer.best_model.predict(X)
            )

            # Reporting stage
            output_path=self._get_save_path()
            if output_path:
                self.report_generator.generate(
                    metrics, self.optimizer.best_model, plots, output_path
                )
                self._show_success_message(output_path)

        except PipelineError as e:
            self._handle_error(e)
        except Exception as e:
            self._handle_error(PipelineError("Unexpected error", e))

    def _get_save_path(self) -> Optional[str]:
        """Get report save path from user"""
        return filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf")],
            title="Save Analysis Report As"
        )

    def _show_success_message(self, path: str) -> None:
        """Show final success message"""
        messagebox.showinfo(
            "Pipeline Complete",
            f"Successfully generated report:\n{path}"
        )

    def _handle_error(self, error: PipelineError) -> None:
        """Handle pipeline errors"""
        logging.error(f"Pipeline failed: {str(error)}")
        messagebox.showerror(
            "Pipeline Error",
            f"{error.__class__.__name__}:\n{str(error)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    # Configure logging and run pipeline
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()]
    )

    root=tk.Tk()
    root.withdraw()

    try:
        controller=PipelineController()
        input_file=filedialog.askopenfilename(
            title="Select Input Data File",
            filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
        )
        if input_file:
            controller.execute_pipeline(input_file)
        else:
            logging.info("Operation cancelled by user")
    except Exception as e:
        logging.critical(f"Critical failure: {str(e)}")
        messagebox.showerror(
            "Fatal Error",
            f"Application terminated unexpectedly:\n{str(e)}"
        )
    finally:
        plt.close('all')
        root.destroy()
