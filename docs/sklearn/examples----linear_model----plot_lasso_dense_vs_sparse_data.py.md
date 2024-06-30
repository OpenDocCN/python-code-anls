# `D:\src\scipysrc\scikit-learn\examples\linear_model\plot_lasso_dense_vs_sparse_data.py`

```
"""
==============================
Lasso on dense and sparse data
==============================

We show that linear_model.Lasso provides the same results for dense and sparse
data and that in the case of sparse data the speed is improved.

"""

from time import time

from scipy import linalg, sparse

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

# %%
# Comparing the two Lasso implementations on Dense data
# -----------------------------------------------------
#
# We create a linear regression problem that is suitable for the Lasso,
# that is to say, with more features than samples. We then store the data
# matrix in both dense (the usual) and sparse format, and train a Lasso on
# each. We compute the runtime of both and check that they learned the
# same model by computing the Euclidean norm of the difference between the
# coefficients they learned. Because the data is dense, we expect better
# runtime with a dense data format.

# Generate synthetic dense data for regression with 200 samples and 5000 features
X, y = make_regression(n_samples=200, n_features=5000, random_state=0)
# Create a copy of X in sparse COO format
X_sp = sparse.coo_matrix(X)

alpha = 1
# Initialize Lasso models for sparse and dense data
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)

t0 = time()
sparse_lasso.fit(X_sp, y)
print(f"Sparse Lasso done in {(time() - t0):.3f}s")

t0 = time()
dense_lasso.fit(X, y)
print(f"Dense Lasso done in {(time() - t0):.3f}s")

# compare the regression coefficients
coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
print(f"Distance between coefficients : {coeff_diff:.2e}")

#
# %%
# Comparing the two Lasso implementations on Sparse data
# ------------------------------------------------------
#
# We make the previous problem sparse by replacing all small values with 0
# and run the same comparisons as above. Because the data is now sparse, we
# expect the implementation that uses the sparse data format to be faster.

# Create a sparse version of X by setting values less than 2.5 to 0
Xs = X.copy()
Xs[Xs < 2.5] = 0.0
# Create a sparse COO matrix from Xs and then convert it to CSC format
Xs_sp = sparse.coo_matrix(Xs)
Xs_sp = Xs_sp.tocsc()

# compute the proportion of non-zero coefficient in the data matrix
print(f"Matrix density : {(Xs_sp.nnz / float(X.size) * 100):.3f}%")

alpha = 0.1
# Initialize Lasso models for sparse and dense data
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

t0 = time()
sparse_lasso.fit(Xs_sp, y)
print(f"Sparse Lasso done in {(time() - t0):.3f}s")

t0 = time()
dense_lasso.fit(Xs, y)
print(f"Dense Lasso done in  {(time() - t0):.3f}s")

# compare the regression coefficients
coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
print(f"Distance between coefficients : {coeff_diff:.2e}")

# %%
```