import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


class MetricsVisualizer:
    def __init__(self, csv_path, title_suffix=''):
        self.csv_path = csv_path
        self.title_suffix = title_suffix

    def load_metrics(self):
        if not os.path.exists(self.csv_path):
            print(f"Plik {self.csv_path} nie istnieje.")
            return None
        return pd.read_csv(self.csv_path)

    def validate_columns(self, metrics_df):
        required_columns = {'epoch', 'train_loss_epoch', 'val_loss', 'val_dice', 'val_iou'}
        if not required_columns.issubset(metrics_df.columns):
            print(f"Brak wymaganych kolumn w {self.csv_path}. Dostępne kolumny: {metrics_df.columns}")
            return False
        return True

    def plot_loss(self, metrics_df):
        epochs = metrics_df['epoch']
        plt.plot(epochs, metrics_df['train_loss_epoch'], marker='o', linestyle='-', label='Strata dane treningowe',
                 color='salmon')
        plt.plot(epochs, metrics_df['val_loss'], marker='o', linestyle='-', label='Strata dane walidacyjne', color='teal')
        plt.title(f'Strata podczas trenowania i walidacji {self.title_suffix}', fontsize=16)
        plt.xlabel('Epoki', fontsize=12)
        plt.ylabel('Strata', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.7)

    def plot_metrics(self, metrics_df):
        epochs = metrics_df['epoch']
        plt.plot(epochs, metrics_df['val_dice'], marker='o', linestyle='-', label='Dice (Walidacja)', color='green')
        plt.plot(epochs, metrics_df['val_iou'], marker='s', linestyle='-', label='IoU (Walidacja)', color='blue')
        plt.title(f'Metryki (Dice, IoU) na zbiorze walidacyjnym {self.title_suffix}', fontsize=16)
        plt.xlabel('Epoki', fontsize=12)
        plt.ylabel('Metryki', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.7)

    def visualize(self):
        metrics_df = self.load_metrics()
        if metrics_df is None or not self.validate_columns(metrics_df):
            return

        plt.figure(figsize=(14, 10))
        plt.subplot(2, 1, 1)
        self.plot_loss(metrics_df)
        plt.subplot(2, 1, 2)
        self.plot_metrics(metrics_df)
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Wizualizacja metryk dla modelu segmentacji wieloklasowej")
    parser.add_argument('--csv_path', type=str, required=True, help='Plik CSV z metrykami')
    parser.add_argument('--title_suffix', type=str, default='', help='Tytuł do wykresów')
    args = parser.parse_args()

    visualizer = MetricsVisualizer(csv_path=args.csv_path, title_suffix=args.title_suffix)
    visualizer.visualize()


if __name__ == "__main__":
    main()

