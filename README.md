# Batch_gradient_descent
Batch_gradient_descent_On the linear regression (diabetes dataset)


This code demonstrates the implementation of a simple linear regression model using gradient descent from scratch, and then compares its performance to sklearn's **LinearRegression** model on the diabetes dataset.
# Importing Libraries:
<pre>
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
</pre>

This code imports necessary libraries, including **numpy** for numerical computations, load_diabetes from **sklearn** to load the diabetes dataset, LinearRegression to use sklearn's linear regression model, **train_test_split** to split the data into training and testing sets, and **r2_score** to calculate the coefficient of determination (R-squared) score.
# Loading Data:
<pre>
data = load_diabetes()
x = data.data
y = data.target
</pre>
The diabetes dataset is loaded, where **x** contains the feature data and **y** contains the target variable (diabetes progression).
# Printing Data Shapes:
<pre>
print(x.shape)
print(y.shape)
</pre>
This code snippet prints the shapes of the feature matrix **x** and the target array **y**.
# Data Splitting:
<pre>
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
</pre>
The data is split into training and testing sets using a test size of 20% and a random seed of 2.
# Using sklearn's LinearRegression:
<pre>
reg = LinearRegression()
reg.fit(x_train, y_train)
print(reg.coef_)
print(reg.intercept_)
y_pred = reg.predict(x_test)
r2_score(y_test, y_pred)
</pre>
The code above uses sklearn's **LinearRegression** model to fit the training data. It then prints the learned coefficients (**coef_**) and intercept (**intercept_**), predicts the target values for the test data, and calculates the R-squared score to evaluate the model's performance.
# Custom Gradient Descent Linear Regression:
<pre>
class GDRegression:
    # ...
</pre>
This code defines a custom class named **GDRegression** to implement linear regression using gradient descent.
# Class Initialization:
<pre>
def __init__(self, learning_rate=0.1, epochs=100):
    # ...
</pre>
The class constructor initializes instance variables such as the learning rate, number of epochs, coefficients (**coef_**), intercept (**intercept_**), and more.
# Fitting the Model:
<pre>
def fit(self, x_train, y_train):
    # ...
</pre>
The **fit** method implements gradient descent. It initializes coefficients and **intercept**, and then iteratively updates them using gradient descent equations for linear regression.
# Predicting:
<pre>
def predict(self, x_test):
    return np.dot(x_test, self.coef_) + self.intercept_
</pre>
The **predict** method takes test data and predicts target values using the learned coefficients and intercept.
# Using Custom Model:
<pre>
gdr = GDRegression(epochs=1000, learning_rate=0.7)
gdr.fit(x_train, y_train)
y_pred = gdr.predict(x_test)
r2_score(y_test, y_pred)
</pre>
Here, an instance of the custom **GDRegression** class is created with specified hyperparameters (epochs and learning rate). The model is fitted to the training data using gradient descent, predictions are made on the test data, and the R-squared score is calculated to evaluate the custom model's performance.

Overall, this code provides a comparison between sklearn's built-in linear regression model and a custom implementation using gradient descent.
