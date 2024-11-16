from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("student_sleep_patterns.csv", delimiter=",")
df = df.dropna()
scaler = StandardScaler()

gender_mapping = {'Male': 1, 'Female': 2, 'Other': 3}
df['Gender_numeric'] = df['Gender'].map(gender_mapping)

year_mapping = {'1st Year': 1, '2nd Year': 2, '3rd Year': 3, '4th Year': 4}
df['Year_numeric'] = df['University_Year'].map(year_mapping)

coeffients = ["Study_Hours", "Gender_numeric", "Year_numeric", "Caffeine_Intake", "Physical_Activity"]
Y = "Sleep_Quality"

sleep_X = df[coeffients]
sleep_Y = df[Y]

X_train, X_test, Y_train, Y_test = train_test_split(sleep_X, sleep_Y, test_size=0.2)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

degree = 2
poly_regression_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
poly_regression_model.fit(X_train_scaled, Y_train)

y_pred = poly_regression_model.predict(X_test_scaled)

print("MSE: %.2f" % mean_squared_error(Y_test, y_pred))
print("R^2: %.2f" % r2_score(Y_test, y_pred))

sns.histplot(df['Sleep_Quality'], kde=True)
plt.title("Distribution of Sleep Quality")
plt.show()
