from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

pd.read_csv("student_sleep_patterns.csv", delimiter=",")