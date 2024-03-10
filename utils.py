import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Load your data (replace 'your_dataset.csv' with the actual file path)
    data = pd.read_csv('ASSN1_Q1\Cleaned_Diabetes.csv')

    # Assuming 'Outcome' is the column name for the target variable
    X = data.drop('Outcome', axis=1).values
    #y = data['Outcome'].values
    y = np.where(data['Outcome'].values == 0, -1, 1)

    # Split the data into training and testing sets
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = get_data()
print("X_train:", X_train.shape[0])
print("y_train:", y_train.shape[0])
print("X_test:", X_test.shape[0])
print("y_test:", y_test.shape[0])
