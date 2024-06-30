# `D:\src\scipysrc\scikit-learn\examples\ensemble\plot_stack_predictors.py`

```
"""
=================================
Combine predictors using stacking
=================================

.. currentmodule:: sklearn

Stacking refers to a method to blend estimators. In this strategy, some
estimators are individually fitted on some training data while a final
estimator is trained using the stacked predictions of these base estimators.

In this example, we illustrate the use case in which different regressors are
stacked together and a final linear penalized regressor is used to output the
prediction. We compare the performance of each individual regressor with the
stacking strategy. Stacking slightly improves the overall performance.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Download the dataset
######################
#
# We will use the `Ames Housing`_ dataset which was first compiled by Dean De Cock
# and became better known after it was used in Kaggle challenge. It is a set
# of 1460 residential homes in Ames, Iowa, each described by 80 features. We
# will use it to predict the final logarithmic price of the houses. In this
# example we will use only 20 most interesting features chosen using
# GradientBoostingRegressor() and limit number of entries (here we won't go
# into the details on how to select the most interesting features).
#
# The Ames housing dataset is not shipped with scikit-learn and therefore we
# will fetch it from `OpenML`_.
#
# .. _`Ames Housing`: http://jse.amstat.org/v19n3/decock.pdf
# .. _`OpenML`: https://www.openml.org/d/42165

import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle


def load_ames_housing():
    # Fetch the Ames Housing dataset from OpenML and store it as a pandas DataFrame
    df = fetch_openml(name="house_prices", as_frame=True)
    # Separate features (X) and target variable (y) from the dataset
    X = df.data
    y = df.target

    # Define a list of 20 selected features from the dataset
    features = [
        "YrSold",
        "HeatingQC",
        "Street",
        "YearRemodAdd",
        "Heating",
        "MasVnrType",
        "BsmtUnfSF",
        "Foundation",
        "MasVnrArea",
        "MSSubClass",
        "ExterQual",
        "Condition2",
        "GarageCars",
        "GarageType",
        "OverallQual",
        "TotalBsmtSF",
        "BsmtFinSF1",
        "HouseStyle",
        "MiscFeature",
        "MoSold",
    ]

    # Select only the specified features from the dataset
    X = X.loc[:, features]
    # Shuffle the dataset rows with a fixed random state
    X, y = shuffle(X, y, random_state=0)

    # Limit dataset entries to 600 rows for demonstration purposes
    X = X.iloc[:600]
    y = y.iloc[:600]
    return X, np.log(y)


X, y = load_ames_housing()

# %%
# Make pipeline to preprocess the data
######################################
#
# Before we can use Ames dataset we still need to do some preprocessing.
# First, we will select the categorical and numerical columns of the dataset to
# construct the first step of the pipeline.

from sklearn.compose import make_column_selector

# Select columns with object (categorical) dtype using make_column_selector
cat_selector = make_column_selector(dtype_include=object)
cat_selector(X)

# %%
# Select columns with numerical (np.number) dtype using make_column_selector
num_selector = make_column_selector(dtype_include=np.number)
num_selector(X)

# %%
# Then, we will need to design preprocessing pipelines which depends on the
# ending regressor. If the ending regressor is a linear model, one needs to
# one-hot encode the categories. If the ending regressor is a tree-based model
# an ordinal encoder will be sufficient. Besides, numerical values need to be
# standardized for a linear model while the raw numerical data can be treated
# as is by a tree-based model. However, both models need an imputer to
# handle missing values.
#
# We will first design the pipeline required for the tree-based models.

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

# 创建一个OrdinalEncoder对象用于处理分类特征，将未知的分类值设定为指定值，缺失值编码为另一个指定值
cat_tree_processor = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=-2,
)

# 创建一个SimpleImputer对象用于处理数值特征，使用均值填充缺失值，并添加指示器特征
num_tree_processor = SimpleImputer(strategy="mean", add_indicator=True)

# 创建一个列转换器对象，将数值特征使用num_tree_processor处理，分类特征使用cat_tree_processor处理
tree_preprocessor = make_column_transformer(
    (num_tree_processor, num_selector), (cat_tree_processor, cat_selector)
)
tree_preprocessor

# %%
# Then, we will now define the preprocessor used when the ending regressor
# is a linear model.

from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 创建一个OneHotEncoder对象用于处理分类特征，忽略未知的分类值
cat_linear_processor = OneHotEncoder(handle_unknown="ignore")

# 创建一个包含StandardScaler和SimpleImputer对象的Pipeline，用于处理数值特征
num_linear_processor = make_pipeline(
    StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
)

# 创建一个列转换器对象，将数值特征使用num_linear_processor处理，分类特征使用cat_linear_processor处理
linear_preprocessor = make_column_transformer(
    (num_linear_processor, num_selector), (cat_linear_processor, cat_selector)
)
linear_preprocessor

# %%
# Stack of predictors on a single data set
##########################################
#
# It is sometimes tedious to find the model which will best perform on a given
# dataset. Stacking provide an alternative by combining the outputs of several
# learners, without the need to choose a model specifically. The performance of
# stacking is usually close to the best model and sometimes it can outperform
# the prediction performance of each individual model.
#
# Here, we combine 3 learners (linear and non-linear) and use a ridge regressor
# to combine their outputs together.
#
# .. note::
#    Although we will make new pipelines with the processors which we wrote in
#    the previous section for the 3 learners, the final estimator
#    :class:`~sklearn.linear_model.RidgeCV()` does not need preprocessing of
#    the data as it will be fed with the already preprocessed output from the 3
#    learners.

from sklearn.linear_model import LassoCV

# 创建一个Pipeline，将linear_preprocessor和LassoCV模型串联起来
lasso_pipeline = make_pipeline(linear_preprocessor, LassoCV())
lasso_pipeline

# %%
from sklearn.ensemble import RandomForestRegressor

# 创建一个Pipeline，将tree_preprocessor和RandomForestRegressor模型串联起来
rf_pipeline = make_pipeline(tree_preprocessor, RandomForestRegressor(random_state=42))
rf_pipeline

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

# 创建一个Pipeline，将tree_preprocessor和HistGradientBoostingRegressor模型串联起来
gbdt_pipeline = make_pipeline(
    tree_preprocessor, HistGradientBoostingRegressor(random_state=0)
)
gbdt_pipeline

# %%
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

estimators = [
    ("Random Forest", rf_pipeline),
    ("Lasso", lasso_pipeline),
    ("Gradient Boosting", gbdt_pipeline),


# 创建一个包含机器学习模型名称和对应管道的元组列表
("Random Forest", rf_pipeline),
("Lasso", lasso_pipeline),
("Gradient Boosting", gbdt_pipeline),


这段代码是一个包含三个元组的列表，每个元组包含一个机器学习模型的名称和相应的管道（pipeline）。
]

# 创建一个堆叠回归器，使用给定的基础估计器和最终估计器（RidgeCV）
stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())
stacking_regressor

# %%
# 测量并绘制结果
##############################
#
# 现在我们可以使用 Ames 房屋数据集来进行预测。我们检查每个单独预测器的性能，以及回归器堆叠的性能。

import time  # 导入时间模块，用于测量执行时间

import matplotlib.pyplot as plt  # 导入 matplotlib 用于绘图

from sklearn.metrics import PredictionErrorDisplay  # 导入 PredictionErrorDisplay 用于展示预测误差
from sklearn.model_selection import cross_val_predict, cross_validate  # 导入交叉验证相关函数

fig, axs = plt.subplots(2, 2, figsize=(9, 7))  # 创建一个 2x2 的子图布局
axs = np.ravel(axs)  # 将子图数组展平，以便迭代

# 遍历每个子图和其对应的估计器（包括堆叠回归器）
for ax, (name, est) in zip(
    axs, estimators + [("Stacking Regressor", stacking_regressor)]
):
    scorers = {"R2": "r2", "MAE": "neg_mean_absolute_error"}  # 定义评分指标

    start_time = time.time()  # 记录开始时间
    scores = cross_validate(
        est, X, y, scoring=list(scorers.values()), n_jobs=-1, verbose=0
    )  # 进行交叉验证并计算得分
    elapsed_time = time.time() - start_time  # 计算交叉验证所需时间

    y_pred = cross_val_predict(est, X, y, n_jobs=-1, verbose=0)  # 进行交叉验证预测

    # 计算并格式化得分结果
    scores = {
        key: (
            f"{np.abs(np.mean(scores[f'test_{value}'])):.2f} +- "
            f"{np.std(scores[f'test_{value}']):.2f}"
        )
        for key, value in scorers.items()
    }

    # 创建 PredictionErrorDisplay 对象，展示实际值与预测值的关系
    display = PredictionErrorDisplay.from_predictions(
        y_true=y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        ax=ax,
        scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
        line_kwargs={"color": "tab:red"},
    )
    ax.set_title(f"{name}\nEvaluation in {elapsed_time:.2f} seconds")  # 设置子图标题，包含估计器名称和评估时间

    # 添加得分信息到子图
    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")  # 设置图例位置

plt.suptitle("Single predictors versus stacked predictors")  # 设置总标题
plt.tight_layout()  # 调整布局使子图适应画布
plt.subplots_adjust(top=0.9)  # 调整子图之间的间距
plt.show()

# %%
# 堆叠回归器将结合不同回归器的优势。然而，我们也注意到训练堆叠回归器要比单独回归器显著更加昂贵。
```