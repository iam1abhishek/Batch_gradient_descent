# Batch_gradient_descent
Batch_gradient_descent_On the linear regression (diabetes dataset)


This code demonstrates the implementation of a simple linear regression model using gradient descent from scratch, and then compares its performance to sklearn's LinearRegression model on the diabetes dataset.
# Importing Libraries:
<pre>
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
</pre>

This code imports necessary libraries, including **numpy** for numerical computations, load_diabetes from **sklearn** to load the diabetes dataset, LinearRegression to use sklearn's linear regression model, train_test_split to split the data into training and testing sets, and r2_score to calculate the coefficient of determination (R-squared) score.

