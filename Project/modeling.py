# Rigoberto Valadez Mena
# Agrotech Project
# Importing Libraries 
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate

#Loading the dataset into the program
df = pd.read_csv('processedData.csv')
# Separating the labels from the rest of the features
y = df[['headWeight', 'polarDiameter', 'radialDiameter']]
x = df.drop(['headWeight', 'polarDiameter', 'radialDiameter'], axis=1)

# Linear Regression
linReg = LinearRegression()
# Loading the model into a Multi-Output Regressor
model = MultiOutputRegressor(linReg)
# Performing 10-fold Cross-Validation
scores = cross_validate(model, x, y, cv=10, scoring=('r2', 'neg_root_mean_squared_error'))
# Printing scores
print("Linear Regression")
print("%0.2f root mean squared error with a standard deviation of %0.2f" % (scores['test_neg_root_mean_squared_error'].mean(), scores['test_neg_root_mean_squared_error'].std()))
print("%0.2f r2 with a standard deviation of %0.2f" % (scores['test_r2'].mean(), scores['test_r2'].std()))

# Decision Tree Regressor
decTree = DecisionTreeRegressor()
# Loading the model into a Multi-Output Regressor
model = MultiOutputRegressor(decTree)
# Performing 10-fold Cross-Validation
scores = cross_validate(model, x, y, cv=10, scoring=('r2', 'neg_root_mean_squared_error'))
# Printing Scores
print("Decision Tree")
print("%0.2f root mean squared error with a standard deviation of %0.2f" % (scores['test_neg_root_mean_squared_error'].mean(), scores['test_neg_root_mean_squared_error'].std()))
print("%0.2f r2 with a standard deviation of %0.2f" % (scores['test_r2'].mean(), scores['test_r2'].std()))

# Random Forest Regressor
myRandomForest = RandomForestRegressor()
# Loading the model into a Multi-Output Regressor
model = MultiOutputRegressor(myRandomForest)
# Performing 10-fold Cross-Validation
scores = cross_validate(model, x, y, cv=10, scoring=('r2', 'neg_root_mean_squared_error'))
# Printing Scores
print("Random Forest Regressor")
print("%0.2f root mean squared error with a standard deviation of %0.2f" % (scores['test_neg_root_mean_squared_error'].mean(), scores['test_neg_root_mean_squared_error'].std()))
print("%0.2f r2 with a standard deviation of %0.2f" % (scores['test_r2'].mean(), scores['test_r2'].std()))

# Gradient Booster Regressor
myGradientBooster = GradientBoostingRegressor()
# Loading the model into a Multi-Output Regressor
model = MultiOutputRegressor(myGradientBooster)
# Performing 10-fold Cross-Validation
scores = cross_validate(model, x, y, cv=10, scoring=('r2', 'neg_root_mean_squared_error'))
# Printing Scores
print("Gradient Booester Regressor")
print("%0.2f root mean squared error with a standard deviation of %0.2f" % (scores['test_neg_root_mean_squared_error'].mean(), scores['test_neg_root_mean_squared_error'].std()))
print("%0.2f r2 with a standard deviation of %0.2f" % (scores['test_r2'].mean(), scores['test_r2'].std()))