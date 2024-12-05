import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.api import OLS, add_constant


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

scaler = StandardScaler()

df2 = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", delimiter=",")
print(df2.columns)

gender = {'Male': 1, 'Female': 2}
df2['gender_numeric'] = df2['Gender'].map(gender)
occupation_mapping = {'Nurse': 1, 'Sales Representative': 2, 'Salesperson': 3, 'Scientist': 4, 'Software Engineer': 5, 'Teacher': 6, 'Accountant': 7, 'Doctor': 8, 'Engineer': 9, 'Lawyer': 10, 'Manager': 11}
df2['occupation_numeric'] = df2['Occupation'].map(occupation_mapping)
disorder_mapping = {'Insomnia': 1, 'None': 2, 'Sleep Apnea': 3}
df2['disorder_numeric'] = df2["Sleep Disorder"].map(disorder_mapping)
bmi_mapping = {'Normal': 1, 'Normal Weight': 2, 'Obese': 3, 'Overweight': 4}
df2['bmi_numeric'] = df2['BMI Category'].map(bmi_mapping)

df2[['systolic', 'diastolic']] = df2['Blood Pressure'].str.split('/', expand=True)
df2['systolic'] = df2['systolic'].astype(int)
df2['diastolic'] = df2['diastolic'].astype(int)

coeffients = ["gender_numeric", "occupation_numeric","bmi_numeric", "Age", "Sleep Duration", "Stress Level", "systolic", "diastolic"]
Y = "Quality of Sleep"

df2 = df2.dropna()

#df[coeffients] = (df[coeffients] - df[coeffients].mean()) / df[coeffients].std()

sleep_X = df2[coeffients]
sleep_Y = df2[Y]

X_with_constant = add_constant(sleep_X)

model = OLS(sleep_Y, X_with_constant).fit()
print("P-Values for each feature:")
print(model.pvalues)
print(model.summary())

X_train, X_test, Y_train, Y_test = train_test_split(sleep_X, sleep_Y, test_size=0.3)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifierKNN = KNeighborsClassifier(n_neighbors=8)
classifierKNN.fit(X_train_scaled, Y_train)
otherClassifierTestPred = classifierKNN.predict(X_test_scaled)
npYtest = np.array(Y_test)
print("K-Nearest Neighbour " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierRndForest = RandomForestClassifier(verbose=True)
classifierRndForest.fit(X_train_scaled, Y_train)
otherClassifierTestPred = classifierRndForest.predict(X_test_scaled)
npYtest = np.array(Y_test)
print("Random Forest " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierNB = GaussianNB()
classifierNB.fit(X_train, Y_train)
otherClassifierTestPred = classifierNB.predict(X_test)
npYtest = np.array(Y_test)
print("Gaussian NB" + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train_scaled, Y_train)
Y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

runRegressionModel(X_train_scaled, X_test_scaled, Y_train, Y_test)