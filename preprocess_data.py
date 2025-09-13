import pandas as pd
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, data):
        self.data = data.copy()
        self.visualizations = {}
        self.scaler = None

    def scale(self):
        self.scaler = StandardScaler()
        features = self.data.drop("target", axis=1)
        scaled = self.scaler.fit_transform(features)
        self.data.loc[:, features.columns] = scaled
        return self.data

    # ==== Методы для визуализаций ====
    def add_histogram(self, column):
        self.data[column].hist(bins=20)
        plt.title(f"Histogram of {column} (preprocessed)")
        plt.show()

    def add_line_plot(self, column):
        plt.plot(self.data[column])
        plt.title(f"Line plot of {column} (preprocessed)")
        plt.show()

    def add_scatter(self, col_x, col_y):
        plt.scatter(self.data[col_x], self.data[col_y])
        plt.title(f"Scatter plot: {col_x} vs {col_y} (preprocessed)")
        plt.show()

    # ==== Методы работы с пропусками ====
    def count_missing(self):
        return self.data.isnull().sum()

    def missing_report(self):
        missing = self.count_missing()
        print("Отчёт о пропущенных значениях (preprocessed):")
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
