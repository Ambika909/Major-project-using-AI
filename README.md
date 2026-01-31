#Introduction to Machine Learning (Predictive Analytics).
Imagine we have a small online shop that sells a product. We advertise this product through various channels – say TV commercials, radio ads, and newspaper ads – and we have data on how much we spent on each type of advertising and what our sales were in each scenario (for example, each row of data could be a different market or a different month). We want to answer a simple question: if we invest a certain amount in advertising, how many units can we expect to sell? This is a classic predictive analytics question, and it's perfect for linear regression.
### Train-Test Split: Why We Need It

In machine learning, our goal is to build models that can make accurate predictions on data they haven't seen before. If we train a model on all of our data and then evaluate it on the same data, we might get a misleadingly optimistic picture of its performance. This is because the model could have simply memorized the training data rather than learning the underlying patterns.

To address this, we use a technique called **train-test split**. We split our dataset into two parts:

1.  **Training Set:** This portion of the data is used to train the model. The model learns the relationships between features and the target variable from this data.
2.  **Testing Set:** This portion of the data is kept separate and is not used during training. After the model is trained, we use the testing set to evaluate how well the model generalizes to new, unseen data.

By evaluating the model on the testing set, we get a more realistic assessment of its performance and can identify if the model is overfitting (performing well on the training data but poorly on the testing data).

Let's apply this to our advertising example. We'll split our advertising data into a training set and a testing set.
As you can see from the output, the original data (10 rows, 4 columns) has been split into:

*   **Training set:** 8 rows for training the model (80% of the data).
*   **Testing set:** 2 rows for evaluating the model (20% of the data).

Now we can train our linear regression model on the `X_train` and `y_train` data and then evaluate its performance on the `X_test` and `y_test` data.

train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True)
Let's check if there are any missing values in our dataset using the `.isnull().sum()` method. This is an important step in data cleaning to ensure our model doesn't encounter errors due to missing data.
The mathematical formula for Mean Absolute Error (MAE) is:

$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

Where:
- $n$ is the number of data points.
- $y_i$ is the actual value for the $i$-th data point.
- $\hat{y}_i$ is the predicted value for the $i$-th data point.
- $|y_i - \hat{y}_i|$ is the absolute difference between the actual and predicted values for the $i$-th data point.
