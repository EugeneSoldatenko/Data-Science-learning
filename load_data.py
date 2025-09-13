import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self):
        self.data = None
        self.visualizations = {}

    def load(self):
        dataset = load_digits(as_frame=True)
        df = dataset.frame
        # В digits нет "target" в df, поэтому добавим вручную
        df["target"] = dataset.target
        self.data = df
        return self.data

    # ==== Методы для визуализаций ====
    def add_histogram(self, column):
        self.visualizations[column] = ('hist', plt.hist)
        plt.hist(self.data[column], bins=20)
        plt.title(f"Histogram of {column}")
        plt.show()

    def add_line_plot(self, column):
        self.visualizations[column] = ('line', plt.plot)
        plt.plot(self.data[column])
        plt.title(f"Line plot of {column}")
        plt.show()

    def add_scatter(self, col_x, col_y):
        self.visualizations[(col_x, col_y)] = ('scatter', plt.scatter)
        plt.scatter(self.data[col_x], self.data[col_y])
        plt.title(f"Scatter plot: {col_x} vs {col_y}")
        plt.show()

    def remove_visualization(self, key):
        if key in self.visualizations:
            del self.visualizations[key]

    # ==== Методы работы с пропусками ====
    def count_missing(self):
        return self.data.isnull().sum()

    def missing_report(self):
        missing = self.count_missing()
        print("Отчёт о пропущенных значениях:")
        print(missing[missing > 0])

    def fill_missing(self, strategy="mean"):
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if strategy == "mean":
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif strategy == "median":
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif strategy == "mode":
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        return self.data
