# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\utils.py`

```
# 导入 NumPy 库，并将其命名为 np
import numpy as np

# 从 sklearn 库中导入 balanced_accuracy_score 和 r2_score 两个函数
from sklearn.metrics import balanced_accuracy_score, r2_score


# 定义函数 neg_mean_inertia，计算负平均惯性
def neg_mean_inertia(X, labels, centers):
    # 计算每个样本点到其所属聚类中心的欧式距离平方和，取负数后求平均
    return -(np.asarray(X - centers[labels]) ** 2).sum(axis=1).mean()


# 定义函数 make_gen_classif_scorers，生成分类模型评分器
def make_gen_classif_scorers(caller):
    # 设置训练评分器为 balanced_accuracy_score 函数
    caller.train_scorer = balanced_accuracy_score
    # 设置测试评分器为 balanced_accuracy_score 函数
    caller.test_scorer = balanced_accuracy_score


# 定义函数 make_gen_reg_scorers，生成回归模型评分器
def make_gen_reg_scorers(caller):
    # 设置测试评分器为 r2_score 函数
    caller.test_scorer = r2_score
    # 设置训练评分器为 r2_score 函数
    caller.train_scorer = r2_score


# 定义函数 neg_mean_data_error，计算负均方数据误差
def neg_mean_data_error(X, U, V):
    # 计算预测数据与原始数据的均方误差，取负数
    return -np.sqrt(((X - U.dot(V)) ** 2).mean())


# 定义函数 make_dict_learning_scorers，生成字典学习模型评分器
def make_dict_learning_scorers(caller):
    # 设置训练评分器为 lambda 函数，计算训练集的负均方数据误差
    caller.train_scorer = lambda _, __: (
        neg_mean_data_error(
            caller.X, caller.estimator.transform(caller.X), caller.estimator.components_
        )
    )
    # 设置测试评分器为 lambda 函数，计算验证集的负均方数据误差
    caller.test_scorer = lambda _, __: (
        neg_mean_data_error(
            caller.X_val,
            caller.estimator.transform(caller.X_val),
            caller.estimator.components_,
        )
    )


# 定义函数 explained_variance_ratio，计算解释方差比
def explained_variance_ratio(Xt, X):
    # 计算转换后数据的方差之和与原始数据方差之和的比值
    return np.var(Xt, axis=0).sum() / np.var(X, axis=0).sum()


# 定义函数 make_pca_scorers，生成主成分分析模型评分器
def make_pca_scorers(caller):
    # 设置训练评分器为 lambda 函数，计算训练集数据的解释方差比之和
    caller.train_scorer = lambda _, __: caller.estimator.explained_variance_ratio_.sum()
    # 设置测试评分器为 lambda 函数，计算验证集数据的解释方差比
    caller.test_scorer = lambda _, __: (
        explained_variance_ratio(caller.estimator.transform(caller.X_val), caller.X_val)
    )
```