# `D:\src\scipysrc\scikit-learn\benchmarks\bench_hist_gradient_boosting_adult.py`

```
# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块
from time import time  # 从时间模块中导入时间函数

import numpy as np  # 导入NumPy库
import pandas as pd  # 导入Pandas库

from sklearn.compose import make_column_selector, make_column_transformer  # 导入Sklearn库中的列选择器和列转换器
from sklearn.datasets import fetch_openml  # 导入Sklearn库中的fetch_openml函数，用于获取数据集
from sklearn.ensemble import HistGradientBoostingClassifier  # 导入Sklearn库中的HistGradientBoostingClassifier类，用于梯度增强分类
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator  # 导入Sklearn库中的获取等效估计器函数
from sklearn.metrics import accuracy_score, roc_auc_score  # 导入Sklearn库中的评估指标函数
from sklearn.model_selection import train_test_split  # 导入Sklearn库中的数据集划分函数
from sklearn.preprocessing import OrdinalEncoder  # 导入Sklearn库中的OrdinalEncoder类，用于序数编码

# 创建参数解析器对象
parser = argparse.ArgumentParser()
# 添加命令行参数
parser.add_argument("--n-leaf-nodes", type=int, default=31)  # 叶节点数目
parser.add_argument("--n-trees", type=int, default=100)  # 树的数量
parser.add_argument("--lightgbm", action="store_true", default=False)  # 是否使用LightGBM
parser.add_argument("--learning-rate", type=float, default=0.1)  # 学习率
parser.add_argument("--max-bins", type=int, default=255)  # 最大箱数
parser.add_argument("--no-predict", action="store_true", default=False)  # 是否禁用预测
parser.add_argument("--verbose", action="store_true", default=False)  # 是否详细输出
args = parser.parse_args()  # 解析命令行参数并存储在args变量中

# 从参数中获取所需的值
n_leaf_nodes = args.n_leaf_nodes  # 叶节点数目
n_trees = args.n_trees  # 树的数量
lr = args.learning_rate  # 学习率
max_bins = args.max_bins  # 最大箱数
verbose = args.verbose  # 是否详细输出

# 定义训练模型的函数
def fit(est, data_train, target_train, libname, **fit_params):
    print(f"Fitting a {libname} model...")  # 打印正在拟合的模型名称
    tic = time()  # 记录开始时间
    est.fit(data_train, target_train, **fit_params)  # 使用传入的数据进行模型拟合
    toc = time()  # 记录结束时间
    print(f"fitted in {toc - tic:.3f}s")  # 打印拟合耗时

# 定义预测函数
def predict(est, data_test, target_test):
    if args.no_predict:  # 如果禁用预测，则直接返回
        return
    tic = time()  # 记录开始时间
    predicted_test = est.predict(data_test)  # 对测试数据进行预测
    predicted_proba_test = est.predict_proba(data_test)  # 对测试数据计算预测概率
    toc = time()  # 记录结束时间
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])  # 计算ROC AUC
    acc = accuracy_score(target_test, predicted_test)  # 计算准确率
    print(f"predicted in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc:.4f}")  # 打印预测耗时、ROC AUC和准确率

# 从OpenML获取数据集，并将其作为DataFrame格式存储到data变量中（成年人数据集）
data = fetch_openml(data_id=179, as_frame=True)
X, y = data.data, data.target  # 将数据和标签分别存储到X和y中

# 使用列选择器选择数据集中的分类列，并创建列转换器对象
cat_columns = make_column_selector(dtype_include="category")(X)
preprocessing = make_column_transformer(
    (OrdinalEncoder(), cat_columns),  # 使用OrdinalEncoder对分类列进行编码
    remainder="passthrough",  # 其余列保持不变
    verbose_feature_names_out=False,  # 禁用详细特征名称输出
)

# 对数据集进行转换，并将转换后的DataFrame赋值给X
X = pd.DataFrame(
    preprocessing.fit_transform(X),  # 对X进行拟合和转换
    columns=preprocessing.get_feature_names_out(),  # 获取转换后的特征名称
)

# 计算数据集中类别的数量、特征的数量以及分类特征的数量
n_classes = len(np.unique(y))  # 类别数量
n_features = X.shape[1]  # 特征数量
n_categorical_features = len(cat_columns)  # 分类特征数量
n_numerical_features = n_features - n_categorical_features  # 数值特征数量
print(f"Number of features: {n_features}")  # 打印特征数量
print(f"Number of categorical features: {n_categorical_features}")  # 打印分类特征数量
print(f"Number of numerical features: {n_numerical_features}")  # 打印数值特征数量

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建一个列表，标记每个特征是否为分类特征
is_categorical = [True] * n_categorical_features + [False] * n_numerical_features

# 创建HistGradientBoostingClassifier模型对象，配置相关参数
est = HistGradientBoostingClassifier(
    loss="log_loss",  # 损失函数类型
    learning_rate=lr,  # 学习率
    max_iter=n_trees,  # 最大迭代次数（树的数量）
    max_bins=max_bins,  # 最大箱数
    max_leaf_nodes=n_leaf_nodes,  # 最大叶节点数目
    categorical_features=is_categorical,  # 指定哪些特征是分类特征
    verbose=verbose,  # 是否详细输出
)
    # 早停策略是否启用的标志，此处设置为 False 表示不启用
    early_stopping=False,
    # 随机数生成器的种子，设置为 0 可以复现随机过程
    random_state=0,
    # 控制输出详细程度的参数，verbose 变量控制是否输出详细信息
    verbose=verbose,
)

# 调用定义的fit函数，用estimator（est）对训练集（X_train, y_train）进行拟合
fit(est, X_train, y_train, "sklearn")

# 调用定义的predict函数，用estimator（est）对测试集（X_test, y_test）进行预测
predict(est, X_test, y_test)

# 如果命令行参数中指定了要使用lightgbm
if args.lightgbm:
    # 调用get_equivalent_estimator函数，获取等效于给定estimator（est）的lightgbm版本，并设置参数max_cat_to_onehot为1（不使用独热编码）
    est = get_equivalent_estimator(est, lib="lightgbm", n_classes=n_classes)
    # 根据is_categorical中的布尔值筛选出所有类别特征的索引，存储在categorical_features列表中
    categorical_features = [
        f_idx for (f_idx, is_cat) in enumerate(is_categorical) if is_cat
    ]
    # 用新的lightgbm版本的estimator（est）对训练集（X_train, y_train）进行拟合，指定类别特征为categorical_features
    fit(est, X_train, y_train, "lightgbm", categorical_feature=categorical_features)
    # 用estimator（est）对测试集（X_test, y_test）进行预测
    predict(est, X_test, y_test)
```