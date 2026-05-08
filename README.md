# Regularization and Stability in Linear Regression

This project studies the relationship between regularization strength, model stability, model complexity, and test performance. The main focus is Ridge regression, with OLS and Lasso included as comparison models. The experiments are designed to connect regularization with algorithmic stability using empirical leave-one-out stability measurements.

## Datasets

Two regression datasets are used in this project.

### Diabetes Dataset

The first dataset is the Diabetes dataset from `sklearn.datasets`. It contains 442 observations and 10 numerical input features:

- age
- sex
- bmi
- bp
- tc
- ldl
- hdl
- tch
- ltg
- glu

The target variable is a quantitative measure of diabetes disease progression after one year. Since this is a regression task, the goal is to predict a continuous target value rather than a class label.

### Synthetic Regression Dataset

The second dataset is a synthetic linear regression dataset generated inside the project. It is used because the true data-generating process is controlled. The features are generated from a Gaussian distribution, and the target is generated from a linear model with added Gaussian noise.

This dataset is useful for studying stability because the relationship between the features and the target is known, and the sample size can be changed in a controlled way.

### Train/Test Split

Both datasets are split into training and test sets using an 80/20 split. In other words, 80% of the observations are used for training and 20% are held out for testing. A fixed random seed is used so that the results are reproducible.

For the dataset-size experiment, the test set is kept fixed while different numbers of training observations are sampled. This allows the effect of training sample size on stability and test error to be studied directly.

## Models

The core regression models are implemented from scratch using NumPy. The project does not rely on ready-made machine learning model classes for OLS, Ridge, or Lasso. This makes the implementation more transparent and helps connect the code directly to the mathematical definitions.

### Ordinary Least Squares

Ordinary Least Squares, or OLS, fits a linear model by minimizing the residual sum of squares:

$$
\hat{w}_{OLS}
=
\arg\min_w
\|y - Xw\|_2^2
$$

OLS does not use any regularization penalty. Because of this, it can achieve very low training error, but it may be sensitive to changes in the training data, especially when the feature matrix is ill-conditioned or when features are correlated.

### Ridge Regression

Ridge regression adds an L2 penalty to the OLS objective:

$$
\hat{w}_{Ridge}
=
\arg\min_w
\|y - Xw\|_2^2
+
\lambda \|w\|_2^2
$$

The parameter `lambda` controls the strength of regularization. When `lambda` is very small, Ridge behaves similarly to OLS. As `lambda` increases, the coefficients are shrunk toward zero. This usually reduces model complexity and can improve stability, but if `lambda` is too large, the model may underfit.

### Lasso Regression

Lasso regression uses an L1 penalty instead of an L2 penalty:

$$
\hat{w}_{Lasso}
=
\arg\min_w
\|y - Xw\|_2^2
+
\lambda \|w\|_1
$$

The L1 penalty can shrink some coefficients exactly to zero, which makes Lasso useful for feature selection and sparse models. In this project, Lasso is included as an additional regularized comparison to Ridge.

## Evaluation

The project evaluates each model using several metrics.

### Prediction Performance

The main prediction metrics are:

- Training MSE
- Test MSE
- Training R2
- Test R2

Test MSE is used as the main measure of generalization performance.

### Model Complexity

Model complexity is measured using the coefficient norm. For Ridge, the coefficient norm decreases gradually as `lambda` increases. For Lasso, the coefficient norm often decreases more sharply because the L1 penalty can shrink coefficients to zero.

### Empirical Stability

Stability is measured using a leave-one-out procedure. For a fixed value of `lambda`, the model is first trained on the full training set. Then, one training observation is removed, the model is retrained, and predictions are computed again on the same test set.

Two stability measures are computed:

#### Prediction-based Stability

$$
\frac{1}{n}
\sum_{i=1}^{n}
\frac{1}{|T|}
\sum_{x_t \in T}
|f(x_t; S) - f(x_t; S \setminus i)|
$$

This measures how much the predictions change when one training point is removed.

#### Loss-based Stability

Loss-based stability compares the squared loss of the full model with the squared loss of the leave-one-out model.

Lower values mean better stability.

## How to Run the Project

First, install the required Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Then run the project from the main project folder:

```bash
python main.py
```

The script will train OLS, Ridge, and Lasso models, evaluate them on the Diabetes and synthetic datasets, estimate leave-one-out stability, and save the resulting figures.

## Output

The project produces figures such as:

- Training and test error curves
- Prediction-stability curves
- Loss-stability curves
- Stability vs. test error plots
- Model-complexity plots
- Dataset-size stability plots
- Dataset-size test error plots

The figures are saved in the `figures/` folder.

## Main Findings

The experiments show that regularization affects both stability and prediction performance. Ridge and Lasso generally improve empirical stability compared with OLS when moving from very small to moderate values of `lambda`.

However, very large values of `lambda` can shrink the coefficients too strongly and cause underfitting.

The results also show that the most stable model is not always the model with the lowest test error. Good generalization requires a balance between stability, model flexibility, and predictive accuracy.

The dataset-size experiment supports the theoretical idea that stability improves as the number of training observations increases. With more training data, each individual observation has less influence on the fitted model.

The script will train OLS, Ridge, and Lasso models, evaluate them on the Diabetes and synthetic datasets, estimate leave-one-out stability, and save the resulting figures.

## Output

The project produces figures such as:

- Training and test error curves
- Prediction-stability curves
- Loss-stability curves
- Stability vs. test error plots
- Model-complexity plots
- Dataset-size stability plots
- Dataset-size test error plots

The figures are saved in the `figures/` folder.

## Main Findings

The experiments show that regularization affects both stability and prediction performance. Ridge and Lasso generally improve empirical stability compared with OLS when moving from very small to moderate values of `lambda`.

However, very large values of `lambda` can shrink the coefficients too strongly and cause underfitting.

The results also show that the most stable model is not always the model with the lowest test error. Good generalization requires a balance between stability, model flexibility, and predictive accuracy.

The dataset-size experiment supports the theoretical idea that stability improves as the number of training observations increases. With more training data, each individual observation has less influence on the fitted model.