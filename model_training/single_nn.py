import numpy as np
import pandas as pd
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm

DATASET_DIR = "../data/"
KFOLD = 5

def main(dataset_name):
    X = pd.read_csv(DATASET_DIR + dataset_name + "_X_train.csv").to_numpy()
    y = pd.read_csv(DATASET_DIR + dataset_name + "_y_train.csv").to_numpy().T[0]

    model = MLPClassifier(hidden_layer_sizes=[100, 200, 200, 100], max_iter=500, verbose=True)
    
