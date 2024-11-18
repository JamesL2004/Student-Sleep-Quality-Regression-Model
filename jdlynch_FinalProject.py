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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def TotalPercentOfStudying(sleepDuration, studyHours):

    percOfStudy = 0

    awakeHours = 24 - sleepDuration
    percOfStudy = studyHours/awakeHours

    return percOfStudy

df = pd.read_csv("student_sleep_patterns.csv", delimiter=",")
df = df.dropna()
scaler = StandardScaler()

df2 = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", delimiter=",")
print(df2.columns)

gender = {'Male': 1, 'Female': 2}
df2['gender_numeric'] = df2['Gender'].map(gender)
occupation_mapping = {'Nurse': 1, 'Sales Representative': 2, 'Salesperson': 3, 'Scientist': 4, 'Software Engineer': 5, 'Teacher': 6, 'Accountant': 7, 'Doctor': 8, 'Engineer': 9, 'Lawyer': 10, 'Manager': 11}
disorder_mapping = {'Insomnia': 1, 'None': 2, 'Sleep Apnea': 3}
df2['disorder_numeric'] = df2["Occupation"].apply(occupation_mapping)
bmi_mapping = {'Normal': 1, 'Normal Weight': 2, 'Obese': 3, 'Overweight': 4}

gender_mapping = {'Male': 1, 'Female': 2, 'Other': 3}
df['Gender_numeric'] = df['Gender'].map(gender_mapping)

year_mapping = {'1st Year': 1, '2nd Year': 2, '3rd Year': 3, '4th Year': 4}
df['Year_numeric'] = df['University_Year'].map(year_mapping)

df['Sleep_Quality_Category'] = pd.cut(df['Sleep_Quality'], bins=[0, 2, 4, 6, 8, 10], labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
df['percOfStudy'] = df.apply(lambda row: TotalPercentOfStudying(row['Sleep_Duration'], row['Study_Hours']), axis=1)

coeffients = ["Sleep Duration", "Heart Rate", "Daily Steps", "gender_numeric", "occupation_mapping"]
Y = "Quality of Sleep"

#df[coeffients] = (df[coeffients] - df[coeffients].mean()) / df[coeffients].std()

sleep_X = df2[coeffients]
sleep_Y = df2[Y]

X_train, X_test, Y_train, Y_test = train_test_split(sleep_X, sleep_Y, test_size=0.2)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

classifierKNN = KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train, Y_train)
otherClassifierTestPred = classifierKNN.predict(X_test)
npYtest = np.array(Y_test)
print("K-Nearest Neighbour " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierRndForest = RandomForestClassifier(verbose=True)
classifierRndForest.fit(X_train_scaled, Y_train)
otherClassifierTestPred = classifierRndForest.predict(X_test_scaled)
npYtest = np.array(Y_test)
print("Random Forest " + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

classifierNB = GaussianNB()
classifierNB.fit(X_train_scaled, Y_train)
otherClassifierTestPred = classifierNB.predict(X_test_scaled)
npYtest = np.array(Y_test)
print("Gaussian NB" + " Test set score: {:.2f}".format(np.mean(otherClassifierTestPred == npYtest)))

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train_scaled, Y_train)
Y_pred = clf.predict(X_test_scaled)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

#degree = 2
#poly_regression_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
#poly_regression_model.fit(X_train_scaled, Y_train)

#y_pred = poly_regression_model.predict(X_test_scaled)

#print("MSE: %.2f" % mean_squared_error(Y_test, y_pred))
#print("R^2: %.2f" % r2_score(Y_test, y_pred))

