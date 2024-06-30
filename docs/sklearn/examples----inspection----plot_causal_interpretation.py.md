# `D:\src\scipysrc\scikit-learn\examples\inspection\plot_causal_interpretation.py`

```
"""
===================================================
Failure of Machine Learning to infer causal effects
===================================================

Machine Learning models are great for measuring statistical associations.
Unfortunately, unless we're willing to make strong assumptions about the data,
those models are unable to infer causal effects.

To illustrate this, we will simulate a situation in which we try to answer one
of the most important questions in economics of education: **what is the causal
effect of earning a college degree on hourly wages?** Although the answer to
this question is crucial to policy makers, `Omitted-Variable Biases
<https://en.wikipedia.org/wiki/Omitted-variable_bias>`_ (OVB) prevent us from
identifying that causal effect.
"""

# %%
# The dataset: simulated hourly wages
# -----------------------------------
#
# The data generating process is laid out in the code below. Work experience in
# years and a measure of ability are drawn from Normal distributions; the
# hourly wage of one of the parents is drawn from Beta distribution. We then
# create an indicator of college degree which is positively impacted by ability
# and parental hourly wage. Finally, we model hourly wages as a linear function
# of all the previous variables and a random component. Note that all variables
# have a positive effect on hourly wages.
import numpy as np
import pandas as pd

# Number of samples for simulation
n_samples = 10_000
# Random number generator with a fixed seed for reproducibility
rng = np.random.RandomState(32)

# Simulate work experience in years from a Normal distribution
experiences = rng.normal(20, 10, size=n_samples).astype(int)
experiences[experiences < 0] = 0  # Set negative values to zero

# Simulate ability from a Normal distribution
abilities = rng.normal(0, 0.15, size=n_samples)

# Simulate hourly wage of one parent from a Beta distribution
parent_hourly_wages = 50 * rng.beta(2, 8, size=n_samples)
parent_hourly_wages[parent_hourly_wages < 0] = 0  # Set negative values to zero

# Simulate indicator of college degree, positively influenced by ability and parent hourly wage
college_degrees = (
    9 * abilities + 0.02 * parent_hourly_wages + rng.randn(n_samples) > 0.7
).astype(int)

# True coefficients for the linear model relating variables to hourly wages
true_coef = pd.Series(
    {
        "college degree": 2.0,
        "ability": 5.0,
        "experience": 0.2,
        "parent hourly wage": 1.0,
    }
)

# Calculate simulated hourly wages as a linear combination of variables plus random noise
hourly_wages = (
    true_coef["experience"] * experiences
    + true_coef["parent hourly wage"] * parent_hourly_wages
    + true_coef["college degree"] * college_degrees
    + true_coef["ability"] * abilities
    + rng.normal(0, 1, size=n_samples)
)

hourly_wages[hourly_wages < 0] = 0  # Set negative wages to zero

# %%
# Description of the simulated data
# ---------------------------------
#
# Create a pandas DataFrame to store simulated data including college degrees,
# ability, hourly wages, work experience, and parent hourly wages. This DataFrame
# will be used for generating pair plots and other visualizations.
import seaborn as sns

df = pd.DataFrame(
    {
        "college degree": college_degrees,
        "ability": abilities,
        "hourly wage": hourly_wages,
        "experience": experiences,
        "parent hourly wage": parent_hourly_wages,
    }
)

# Generate pair plots using seaborn to visualize distributions and relationships
grid = sns.pairplot(df, diag_kind="kde", corner=True)

# %%
# In the next section, we train predictive models and we therefore split the
# dataset into training and testing sets. This step is crucial to evaluate
# the performance of machine learning models in predicting hourly wages based
# on simulated data.
# 导入 train_test_split 函数用于分割数据集为训练集和测试集
from sklearn.model_selection import train_test_split

# 指定目标列名
target_name = "hourly wage"

# 从数据框 df 中分离出特征 X 和目标变量 y
X, y = df.drop(columns=target_name), df[target_name]

# 使用 train_test_split 函数将数据集按照指定比例分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Income prediction with fully observed variables
# -----------------------------------------------
#
# 首先，我们训练一个预测模型，即 :class:`~sklearn.linear_model.LinearRegression` 模型。
# 在这个实验中，我们假设真实生成模型所用的所有变量都是可用的。
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 指定用于预测的特征名称列表
features_names = ["experience", "parent hourly wage", "college degree", "ability"]

# 创建具有能力特征的线性回归模型对象
regressor_with_ability = LinearRegression()
# 使用训练集拟合模型
regressor_with_ability.fit(X_train[features_names], y_train)
# 对测试集进行预测
y_pred_with_ability = regressor_with_ability.predict(X_test[features_names])
# 计算 R2 分数
R2_with_ability = r2_score(y_test, y_pred_with_ability)

# 打印带有能力特征的 R2 分数
print(f"R2 score with ability: {R2_with_ability:.3f}")

# %%
# This model predicts well the hourly wages as shown by the high R2 score. We
# plot the model coefficients to show that we exactly recover the values of
# the true generative model.
import matplotlib.pyplot as plt

# 创建 Series 对象，包含模型的系数值和相应的特征名称
model_coef = pd.Series(regressor_with_ability.coef_, index=features_names)

# 将真实生成模型的系数和模型的系数合并到一个 DataFrame 中
coef = pd.concat(
    [true_coef[features_names], model_coef],
    keys=["Coefficients of true generative model", "Model coefficients"],
    axis=1,
)

# 创建水平条形图显示模型系数
ax = coef.plot.barh()
ax.set_xlabel("Coefficient values")
ax.set_title("Coefficients of the linear regression including the ability features")
_ = plt.tight_layout()

# %%
# Income prediction with partial observations
# -------------------------------------------
#
# 在实际应用中，智力能力往往无法观察到，或者只能通过间接测量教育水平（例如 IQ 测试）来估计。
# 但是，从线性模型中排除 "ability" 特征会通过正的观察偏误（OVB）来夸大估计。
features_names = ["experience", "parent hourly wage", "college degree"]

# 创建没有能力特征的线性回归模型对象
regressor_without_ability = LinearRegression()
# 使用训练集拟合模型
regressor_without_ability.fit(X_train[features_names], y_train)
# 对测试集进行预测
y_pred_without_ability = regressor_without_ability.predict(X_test[features_names])
# 计算不带能力特征的 R2 分数
R2_without_ability = r2_score(y_test, y_pred_without_ability)

# 打印不带能力特征的 R2 分数
print(f"R2 score without ability: {R2_without_ability:.3f}")

# %%
# The predictive power of our model is similar when we omit the ability feature
# in terms of R2 score. We now check if the coefficient of the model are
# different from the true generative model.
#
# 模型的预测能力在省略能力特征时的 R2 分数表现类似。现在我们检查模型的系数是否与真实生成模型不同。
model_coef = pd.Series(regressor_without_ability.coef_, index=features_names)

# 将真实生成模型的系数和模型的系数合并到一个 DataFrame 中
coef = pd.concat(
    [true_coef[features_names], model_coef],
    keys=["Coefficients of true generative model", "Model coefficients"],
    axis=1,
)

# 创建水平条形图显示模型系数
ax = coef.plot.barh()
ax.set_xlabel("Coefficient values")
# 设置图表的标题为"Coefficients of the linear regression excluding the ability feature"
_ = ax.set_title("Coefficients of the linear regression excluding the ability feature")

# 调整图表布局使得图表更紧凑
plt.tight_layout()

# 显示图表
plt.show()

# %%
# 由于省略了一个变量，模型会夸大学位特征的系数。因此，将这个系数值解释为真实生成模型的因果效应是不正确的。
#
# 学到的教训
# ---------------
#
# 机器学习模型并不是为了估计因果效应而设计的。尽管我们是用线性模型展示了这一点，但遗漏变量偏误可能影响任何类型的模型。
#
# 每当解释一个系数或者由于特征变化引起的预测变化时，重要的是要记住可能存在未观察到的变量，这些变量可能与待分析特征和目标变量都有相关性。这些变量被称为“混杂变量 <https://en.wikipedia.org/wiki/Confounding>`_。
# 
# 为了在混杂变量存在的情况下仍然估计因果效应，研究人员通常进行实验，随机化处理变量（例如学位）。当实验代价高昂或不道德时，研究人员有时可以使用其他因果推断技术，比如`工具变量 <https://en.wikipedia.org/wiki/Instrumental_variables_estimation>`_（IV）估计。
```