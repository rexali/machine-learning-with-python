# Let's proceed with seeing how we can invoke some 
# machine learning functionalities in scikit-learn
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# Data Preprocessing routines
x = np.asmatrix([[1,2],[2,4]])
# instantiate poynomial feature
poly = PolynomialFeatures(degree = 2)
# buid model
poly.fit(x)
x_poly = poly.transform(x)

print ("Original x variable shape",x.shape)
print (x)
print ('\n##############################\n')
print ("Transformed x variables",x_poly.shape)
print (x_poly)

#alternatively 
x_poly = poly.fit_transform(x)
print ('##################alternatively')
print(x_poly)



from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
#Load data: Let's use the iris dataset to see how the tree algorithm can be used

data = load_iris()
# We will load the iris dataset in the x and y variables.
x = data['data']
y = data['target']
# We will proceed to build the model by invoking the fit 
# function and passing our x predictor and y response variable.
# This will build the tree model

#We will then instantiate DecisonTreeClassifier
estimator = DecisionTreeClassifier()
#buid the model
estimator.fit(x,y)
# Now, we are ready with our model to do some predictions.
# We will use the predict function in order to predict 
# the class labels for the given input.
predicted_y = estimator.predict(x)
# predict_proba function, which gives the probability of the prediction
predicted_y_prob = estimator.predict_proba(x)
# and predict_log_proba function, which provides the logarithm of the prediction probability.
# predicted_y_lprob = estimator.predict_log_proba(x)
print (predicted_y_prob)



# Various machine learning methods can be chained together using pipe lining:
from sklearn.pipeline import Pipeline
#We will then instantiate PolynomialFeatures
poly = PolynomialFeatures(3)
# We will then instantiate DecisonTreeClassifier
tree_estimator = DecisionTreeClassifier()
#We will define a list of tuples to indicate the order of our chaining.
steps = [('poly',poly),('tree',tree_estimator)]
# chain the methods together
# We can now instantiate our Pipeline object with the list declared 
# using the steps variable.
estimator = Pipeline(steps=steps)
# build model
estimator.fit(x,y)
# use the model to predict
predicted_y = estimator.predict(x)
print ('##################22222222')
print (predicted_y)

# We can invoke the named_steps attribute in order to inspect the models in 
# the various stages of our pipeline:
print('#########use named_steps')
print(estimator.named_steps)

