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

