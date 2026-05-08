RANDOM_STATE = 42

# Lambda values used to study the effect of regularization.
# lambda=0 is handled separately as the OLS baseline.
LAMBDA_VALUES = [
    1e-15,1e-10,1e-8,1e-3,1e-2,1,5,30,50,100
]

FIGURE_FOLDER = "figures"
SAVE_FIGURES = True
SHOW_FIGURES = False

# Extension required by the assignment: compare L1 vs L2 regularization.
RUN_L1_L2_EXTENSION = True
COMPUTE_LASSO_STABILITY = True
LASSO_MAX_ITER = 5000
LASSO_TOL = 1e-6
