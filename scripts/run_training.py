import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from segmentation.train import train_model

if __name__ == "__main__":
    train_model()
