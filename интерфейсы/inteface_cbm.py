import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from catboost import CatBoostRegressor
import os
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class MLApp:
    def __init__(self, root):
        self.root = root
        self.model = None
        self.file_path = None
        self.model_path = None

        self.configure_ui()
        self.create_widgets()

    def configure_ui(self):
        self.root.title("Оптимизация дебитов скважин")
        self.root.geometry("800x500")
        self.root.resizable(False, False)
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=10)
        style.configure('TLabel', font=('Arial', 10))

    def create_widgets(self):
        # Заголовок
        header = ttk.Label(
            self.root,
            text="Программа оптимизации дебитов нефтяных скважин",
            font=('Arial', 12, 'bold')
        )
        header.pack(pady=20)

        # Фрейм для кнопок
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)

        # Кнопки управления
        self.btn_model = ttk.Button(
            button_frame,
            text="Загрузить модель (.cbm)",
            command=self.load_model
        )
        self.btn_model.grid(row=0, column=0, padx=10)

        self.btn_file = ttk.Button(
            button_frame,
            text="Выбрать данные (.xlsx)",
            command=self.load_data
        )
        self.btn_file.grid(row=0, column=1, padx=10)

        self.btn_predict = ttk.Button(
            button_frame,
            text="Выполнить прогноз",
            command=self.run_prediction,
            state=tk.DISABLED
        )
        self.btn_predict.grid(row=0, column=2, padx=10)

        # Панель статуса
        self.status_bar = ttk.Label(
            self.root,
            text="Готов к работе",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Таблица для предпросмотра данных
        self.preview_frame = ttk.Frame(self.root)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.tree = ttk.Treeview(self.preview_frame)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def update_status(self, message):
        self.status_bar.config(text=message)
        self.root.update_idletasks()

    def load_model(self):
        try:
            self.model_path = filedialog.askopenfilename(
                filetypes=[("CatBoost Model", "*.cbm")]
            )
            if self.model_path:
                self.model = CatBoostRegressor()
                self.model.load_model(self.model_path)
                self.btn_predict.config(state=tk.NORMAL)
                messagebox.showinfo(
                    "Успех",
                    f"Модель успешно загружена:\n{os.path.basename(self.model_path)}"
                )
                logging.info(f"Модель загружена: {self.model_path}")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {str(e)}")
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить модель: {str(e)}")

    def load_data(self):
        try:
            self.file_path = filedialog.askopenfilename(
                filetypes=[("Excel Files", "*.xlsx")]
            )
            if self.file_path:
                self.df = pd.read_excel(self.file_path)
                self.show_data_preview()
                logging.info(f"Данные загружены: {self.file_path}")
        except Exception as e:
            logging.error(f"Ошибка загрузки данных: {str(e)}")
            messagebox.showerror(
                "Ошибка", f"Не удалось загрузить данные: {str(e)}")

    def show_data_preview(self):
        # Очистка предыдущих данных
        for i in self.tree.get_children():
            self.tree.delete(i)

        # Настройка колонок
        self.tree["columns"] = list(self.df.columns)
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)

        # Добавление данных
        for _, row in self.df.head(10).iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def validate_data(self):
        required_columns = ['W_ID']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(
                f"Отсутствуют обязательные колонки: {', '.join(missing)}")

        non_numeric = self.df.select_dtypes(exclude='number').columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Некорректные типы данных в колонках: {', '.join(non_numeric)}")

        if self.df.isnull().values.any():
            raise ValueError("Обнаружены пропущенные значения в данных")

    def run_prediction(self):
        try:
            self.update_status("Проверка данных...")
            self.validate_data()

            self.update_status("Выполнение прогноза...")
            X = self.df.drop(columns=['W_ID'], errors='ignore')
            predictions = self.model.predict(X)

            self.update_status("Обработка результатов...")
            result_df = self.process_predictions(predictions)

            self.update_status("Сохранение результатов...")
            self.save_results(result_df)

            messagebox.showinfo("Успех", "Прогноз успешно выполнен!")
            logging.info("Прогноз успешно завершен")

        except Exception as e:
            logging.error(f"Ошибка выполнения прогноза: {str(e)}")
            messagebox.showerror("Ошибка", str(e))
        finally:
            self.update_status("Готов к работе")

    def process_predictions(self, predictions):
        result_df = self.df[['W_ID']].copy()
        result_df['Прогнозный дебит'] = predictions

        # Дополнительные расчеты
        result_df['Нормированный дебит'] = (
            result_df['Прогнозный дебит'] / result_df['Прогнозный дебит'].sum()
        )
        result_df['Рекомендуемый приоритет'] = (
            result_df['Нормированный дебит'].rank(ascending=False).astype(int)
        )
        return result_df

    def save_results(self, df):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            title="Сохранить результаты"
        )
        if save_path:
            df.to_excel(save_path, index=False)
            logging.info(f"Результаты сохранены: {save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()
