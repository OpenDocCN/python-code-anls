# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_gradient_boosting_early_stopping.py`

```
# %%
# Model Training and Comparison
# -----------------------------
# Two :class:`~sklearn.ensemble.GradientBoostingRegressor` models are trained:
# one with and another without early stopping. The purpose is to compare their
# performance. It also calculates the training time and the `n_estimators_`
# used by both models.

params = dict(n_estimators=1000, max_depth=5, learning_rate=0.1, random_state=42)

# 创建一个GradientBoostingRegressor对象，用于完全训练模型
gbm_full = GradientBoostingRegressor(**params)

# 创建另一个GradientBoostingRegressor对象，用于实现提前停止的训练
gbm_early_stopping = GradientBoostingRegressor(
    **params,
    validation_fraction=0.1,  # 验证集比例
    n_iter_no_change=10,      # 连续多少次迭代性能没有改善就停止训练
)

start_time = time.time()  # 记录开始训练的时间
gbm_full.fit(X_train, y_train)  # 使用完整数据集训练模型
training_time_full = time.time() - start_time  # 计算完整训练所需时间
n_estimators_full = gbm_full.n_estimators_  # 获取完整训练模型的估计器数量

start_time = time.time()  # 记录开始训练的时间
gbm_early_stopping.fit(X_train, y_train)  # 使用提前停止训练模型
training_time_early_stopping = time.time() - start_time
estimators_early_stopping = gbm_early_stopping.n_estimators_

# %%
# Error Calculation
# -----------------
# The code calculates the :func:`~sklearn.metrics.mean_squared_error` for both
# training and validation datasets for the models trained in the previous
# section. It computes the errors for each boosting iteration. The purpose is
# to assess the performance and convergence of the models.

train_errors_without = []
val_errors_without = []

train_errors_with = []
val_errors_with = []

# Calculate errors for gbm_full model
for i, (train_pred, val_pred) in enumerate(
    zip(
        gbm_full.staged_predict(X_train),
        gbm_full.staged_predict(X_val),
    )
):
    train_errors_without.append(mean_squared_error(y_train, train_pred))
    val_errors_without.append(mean_squared_error(y_val, val_pred))

# Calculate errors for gbm_early_stopping model
for i, (train_pred, val_pred) in enumerate(
    zip(
        gbm_early_stopping.staged_predict(X_train),
        gbm_early_stopping.staged_predict(X_val),
    )
):
    train_errors_with.append(mean_squared_error(y_train, train_pred))
    val_errors_with.append(mean_squared_error(y_val, val_pred))

# %%
# Visualize Comparison
# --------------------
# It includes three subplots:
#
# 1. Plotting training errors of both models over boosting iterations.
# 2. Plotting validation errors of both models over boosting iterations.
# 3. Creating a bar chart to compare the training times and the estimator used
#    of the models with and without early stopping.
#

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

# Plot training errors over boosting iterations
axes[0].plot(train_errors_without, label="gbm_full")
axes[0].plot(train_errors_with, label="gbm_early_stopping")
axes[0].set_xlabel("Boosting Iterations")
axes[0].set_ylabel("MSE (Training)")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].set_title("Training Error")

# Plot validation errors over boosting iterations
axes[1].plot(val_errors_without, label="gbm_full")
axes[1].plot(val_errors_with, label="gbm_early_stopping")
axes[1].set_xlabel("Boosting Iterations")
axes[1].set_ylabel("MSE (Validation)")
axes[1].set_yscale("log")
axes[1].legend()
axes[1].set_title("Validation Error")

# Create a bar chart comparing training times and estimators
training_times = [training_time_full, training_time_early_stopping]
labels = ["gbm_full", "gbm_early_stopping"]
bars = axes[2].bar(labels, training_times)
axes[2].set_ylabel("Training Time (s)")

# Annotate bars with number of estimators used
for bar, n_estimators in zip(bars, [n_estimators_full, estimators_early_stopping]):
    height = bar.get_height()
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.001,
        f"Estimators: {n_estimators}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.show()

# %%
# The difference in training error between the `gbm_full` and the
# `gbm_early_stopping` stems from the fact that `gbm_early_stopping` sets
# aside `validation_fraction` of the training data as internal validation set.
# Early stopping is decided based on this internal validation score.

# %%
# Summary
# -------
# 在我们的示例中，使用 :class:`~sklearn.ensemble.GradientBoostingRegressor`
# 模型和加利福尼亚房价数据集，展示了早期停止的实际好处：
#
# - **防止过拟合：** 我们展示了验证误差在某一点后稳定或开始增加，表明模型对未见过的数据泛化能力更强。
#   这通过在过拟合发生之前停止训练过程来实现。
# - **提升训练效率：** 我们比较了使用和不使用早期停止的模型的训练时间。早期停止的模型在达到相当精度的同时，
#   需要的估计器数量显著减少，从而加快了训练速度。
```