from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def runRegressionModel(trainX, testX, trainY, testY):

    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(trainX, trainY)

    # Make predictions using the testing set
    wine_y_pred = regr.predict(testX)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("MSE: %.2f" % mean_squared_error(testY, wine_y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("R^2: %.2f" % r2_score(testY, wine_y_pred))


    return 0

df = pd.read_csv("student_sleep_patterns.csv", delimiter=",")
scaler = StandardScaler()

gender_mapping = {'Male': 1, 'Female': 2, 'Other': 3}
df['Gender_numeric'] = df['Gender'].map(gender_mapping)

year_mapping = {'1st Year': 1, '2nd Year': 2, '3rd Year': 3, '4th Year': 4}
df['Year_numeric'] = df['University_Year'].map(year_mapping)

coeffients = ["Study_Hours", "Gender_numeric", "Year_numeric", "Caffeine_Intake", "Physical_Activity"]
Y = "Sleep_Quality"

sleep_X = df[coeffients]
sleep_Y = df[Y]

sleep_X = scaler.fit_transform(sleep_X)

X_train, X_test, Y_train, Y_test = train_test_split(sleep_X, sleep_Y, test_size=0.2)

runRegressionModel(X_train, X_test, Y_train, Y_test)