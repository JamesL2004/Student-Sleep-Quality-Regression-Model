from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

gender_mapping = {'Male': 1, 'Female': 2, 'Other': 3}
df['Gender_numeric'] = df['Gender'].map(gender_mapping)

coeffients = ["Gender_numeric"]
Y = "Sleep_Quality"

sleep_X = df[coeffients]
sleep_Y = df[Y]

