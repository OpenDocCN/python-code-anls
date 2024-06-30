# `D:\src\scipysrc\scikit-learn\benchmarks\bench_hist_gradient_boosting_categorical_only.py`

```
# 导入必要的模块和函数
import argparse  # 用于解析命令行参数的模块
from time import time  # 从时间模块中导入时间函数

from sklearn.datasets import make_classification  # 用于生成分类数据集的函数
from sklearn.ensemble import HistGradientBoostingClassifier  # 导入直方图梯度提升分类器
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator  # 导入获取等效估计器的函数
from sklearn.preprocessing import KBinsDiscretizer  # 用于特征分箱的类

# 创建命令行解析器
parser = argparse.ArgumentParser()
# 添加命令行参数
parser.add_argument("--n-leaf-nodes", type=int, default=31)  # 叶节点数的命令行参数，默认为31
parser.add_argument("--n-trees", type=int, default=100)  # 树的数量的命令行参数，默认为100
parser.add_argument("--n-features", type=int, default=20)  # 特征数量的命令行参数，默认为20
parser.add_argument("--n-cats", type=int, default=20)  # 类别特征数量的命令行参数，默认为20
parser.add_argument("--n-samples", type=int, default=10_000)  # 样本数量的命令行参数，默认为10000
parser.add_argument("--lightgbm", action="store_true", default=False)  # 是否使用LightGBM的命令行标志，默认为False
parser.add_argument("--learning-rate", type=float, default=0.1)  # 学习率的命令行参数，默认为0.1
parser.add_argument("--max-bins", type=int, default=255)  # 最大箱数的命令行参数，默认为255
parser.add_argument("--no-predict", action="store_true", default=False)  # 是否禁止预测的命令行标志，默认为False
parser.add_argument("--verbose", action="store_true", default=False)  # 是否显示详细信息的命令行标志，默认为False
args = parser.parse_args()

# 将命令行参数存储到变量中以便后续使用
n_leaf_nodes = args.n_leaf_nodes
n_features = args.n_features
n_categories = args.n_cats  # 命令行参数的名称应为--n-cats，这里应为--n-cats的变量名错误
n_samples = args.n_samples
n_trees = args.n_trees
lr = args.learning_rate
max_bins = args.max_bins
verbose = args.verbose

# 定义模型拟合函数
def fit(est, data_train, target_train, libname, **fit_params):
    # 打印正在拟合的模型名称
    print(f"Fitting a {libname} model...")
    tic = time()  # 记录开始时间
    est.fit(data_train, target_train, **fit_params)  # 调用模型的拟合方法
    toc = time()  # 记录结束时间
    print(f"fitted in {toc - tic:.3f}s")  # 打印拟合耗时

# 定义模型预测函数
def predict(est, data_test):
    # 如果禁止预测，则直接返回
    if args.no_predict:
        return
    tic = time()  # 记录开始时间
    est.predict(data_test)  # 调用模型的预测方法
    toc = time()  # 记录结束时间
    print(f"predicted in {toc - tic:.3f}s")  # 打印预测耗时

# 生成分类数据集
X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=0)

# 将特征数据进行分箱处理
X = KBinsDiscretizer(n_bins=n_categories, encode="ordinal").fit_transform(X)

# 打印特征数量和样本数量信息
print(f"Number of features: {n_features}")
print(f"Number of samples: {n_samples}")

# 创建特征是否为分类特征的列表
is_categorical = [True] * n_features

# 创建HistGradientBoostingClassifier分类器对象
est = HistGradientBoostingClassifier(
    loss="log_loss",  # 损失函数设为对数损失
    learning_rate=lr,  # 学习率设为命令行参数的值
    max_iter=n_trees,  # 迭代次数设为命令行参数的值
    max_bins=max_bins,  # 最大箱数设为命令行参数的值
    max_leaf_nodes=n_leaf_nodes,  # 最大叶节点数设为命令行参数的值
    categorical_features=is_categorical,  # 分类特征列表
    early_stopping=False,  # 关闭早停
    random_state=0,  # 随机种子设为0
    verbose=verbose,  # 是否显示详细信息设为命令行参数的值
)

# 拟合模型并输出拟合耗时
fit(est, X, y, "sklearn")

# 预测模型并输出预测耗时
predict(est, X)

# 如果选择使用LightGBM
if args.lightgbm:
    # 获取等效的LightGBM估计器对象
    est = get_equivalent_estimator(est, lib="lightgbm", n_classes=2)
    est.set_params(max_cat_to_onehot=1)  # 设置最大类别转为独热编码的参数为1，即不使用独热编码
    categorical_features = list(range(n_features))  # 创建分类特征索引列表
    fit(est, X, y, "lightgbm", categorical_feature=categorical_features)  # 拟合LightGBM模型
    predict(est, X)  # 预测LightGBM模型
```