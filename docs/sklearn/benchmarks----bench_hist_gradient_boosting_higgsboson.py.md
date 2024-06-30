# `D:\src\scipysrc\scikit-learn\benchmarks\bench_hist_gradient_boosting_higgsboson.py`

```
# 导入必要的库
import argparse  # 解析命令行参数的库
import os  # 提供与操作系统交互的功能
from gzip import GzipFile  # Gzip 文件解压缩库
from time import time  # 计时功能
from urllib.request import urlretrieve  # 下载文件的库

import numpy as np  # 数组操作库
import pandas as pd  # 数据分析库
from joblib import Memory  # 缓存结果的库

from sklearn.ensemble import HistGradientBoostingClassifier  # 高性能梯度提升树分类器
from sklearn.ensemble._hist_gradient_boosting.utils import get_equivalent_estimator  # 获取等效的估算器
from sklearn.metrics import accuracy_score, roc_auc_score  # 准确率和ROC AUC评分的库
from sklearn.model_selection import train_test_split  # 数据集分割库

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--n-leaf-nodes", type=int, default=31)  # 叶节点数目的命令行参数，默认为31
parser.add_argument("--n-trees", type=int, default=10)  # 树的数目的命令行参数，默认为10
parser.add_argument("--lightgbm", action="store_true", default=False)  # 是否使用LightGBM的命令行参数，默认为False
parser.add_argument("--xgboost", action="store_true", default=False)  # 是否使用XGBoost的命令行参数，默认为False
parser.add_argument("--catboost", action="store_true", default=False)  # 是否使用CatBoost的命令行参数，默认为False
parser.add_argument("--learning-rate", type=float, default=1.0)  # 学习率的命令行参数，默认为1.0
parser.add_argument("--subsample", type=int, default=None)  # 子样本大小的命令行参数，默认为None
parser.add_argument("--max-bins", type=int, default=255)  # 最大箱数的命令行参数，默认为255
parser.add_argument("--no-predict", action="store_true", default=False)  # 是否禁用预测的命令行参数，默认为False
parser.add_argument("--cache-loc", type=str, default="/tmp")  # 缓存位置的命令行参数，默认为/tmp
parser.add_argument("--no-interactions", type=bool, default=False)  # 是否禁用交互的命令行参数，默认为False
parser.add_argument("--max-features", type=float, default=1.0)  # 最大特征数的命令行参数，默认为1.0
args = parser.parse_args()

HERE = os.path.dirname(__file__)  # 获取当前脚本文件所在的目录路径
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"  # 数据集下载链接
m = Memory(location=args.cache_loc, mmap_mode="r")  # 创建一个Memory对象，用于缓存

# 从命令行参数获取模型参数
n_leaf_nodes = args.n_leaf_nodes  # 叶节点数
n_trees = args.n_trees  # 树的数目
subsample = args.subsample  # 子样本大小
lr = args.learning_rate  # 学习率
max_bins = args.max_bins  # 最大箱数
max_features = args.max_features  # 最大特征数

@m.cache  # 使用Memory对象进行缓存
def load_data():
    filename = os.path.join(HERE, URL.rsplit("/", 1)[-1])  # 构建本地保存的文件名
    if not os.path.exists(filename):  # 如果文件不存在则下载
        print(f"Downloading {URL} to {filename} (2.6 GB)...")  # 输出下载提示
        urlretrieve(URL, filename)  # 下载文件
        print("done.")  # 下载完成提示

    print(f"Parsing {filename}...")  # 输出解析提示
    tic = time()  # 记录开始时间
    with GzipFile(filename) as f:  # 使用GzipFile打开压缩文件
        df = pd.read_csv(f, header=None, dtype=np.float32)  # 读取CSV文件到DataFrame
    toc = time()  # 记录结束时间
    print(f"Loaded {df.values.nbytes / 1e9:0.3f} GB in {toc - tic:0.3f}s")  # 输出加载数据的信息
    return df  # 返回读取的DataFrame对象


def fit(est, data_train, target_train, libname):
    print(f"Fitting a {libname} model...")  # 输出拟合模型的信息
    tic = time()  # 记录开始时间
    est.fit(data_train, target_train)  # 使用给定的估算器拟合数据
    toc = time()  # 记录结束时间
    print(f"fitted in {toc - tic:.3f}s")  # 输出拟合所用时间的信息


def predict(est, data_test, target_test):
    if args.no_predict:  # 如果禁用预测
        return  # 直接返回
    tic = time()  # 记录开始时间
    predicted_test = est.predict(data_test)  # 预测测试数据
    predicted_proba_test = est.predict_proba(data_test)  # 计算测试数据的预测概率
    toc = time()  # 记录结束时间
    roc_auc = roc_auc_score(target_test, predicted_proba_test[:, 1])  # 计算ROC AUC评分
    acc = accuracy_score(target_test, predicted_test)  # 计算准确率
    print(f"predicted in {toc - tic:.3f}s, ROC AUC: {roc_auc:.4f}, ACC: {acc :.4f}")  # 输出预测信息


df = load_data()  # 加载数据集到DataFrame
target = df.values[:, 0]  # 提取目标列
data = np.ascontiguousarray(df.values[:, 1:])  # 提取特征数据并转换为连续的数组
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=0
)  # 划分训练集和测试集

n_classes = len(np.unique(target))  # 计算目标类别数

if subsample is not None:
    # 从训练数据中分片取出指定数量的样本和对应的目标标签，用于训练模型
    data_train, target_train = data_train[:subsample], target_train[:subsample]
# 获取训练数据集的样本数和特征数
n_samples, n_features = data_train.shape

# 打印训练集的样本数和特征数信息
print(f"Training set with {n_samples} records with {n_features} features.")

# 根据命令行参数判断是否需要生成特征交互项的常数列表
if args.no_interactions:
    # 如果不需要交互项，则创建一个包含每个特征索引的单元素列表
    interaction_cst = [[i] for i in range(n_features)]
else:
    # 否则，将交互项常数列表设为 None
    interaction_cst = None

# 创建 HistGradientBoostingClassifier 分类器对象
est = HistGradientBoostingClassifier(
    loss="log_loss",              # 损失函数设为对数损失
    learning_rate=lr,             # 学习率设为 lr
    max_iter=n_trees,             # 最大迭代次数设为 n_trees
    max_bins=max_bins,            # 最大箱数设为 max_bins
    max_leaf_nodes=n_leaf_nodes,  # 最大叶节点数设为 n_leaf_nodes
    early_stopping=False,         # 禁用早停
    random_state=0,               # 随机种子设为 0
    verbose=1,                    # 输出详细信息
    interaction_cst=interaction_cst,  # 交互项常数列表
    max_features=max_features     # 最大特征数设为 max_features
)

# 使用训练函数对 est 进行训练，使用 sklearn 框架
fit(est, data_train, target_train, "sklearn")

# 使用训练好的 est 进行预测，使用 sklearn 框架
predict(est, data_test, target_test)

# 如果需要使用 LightGBM 进行兼容性转换
if args.lightgbm:
    # 将 est 转换为与 LightGBM 兼容的估计器
    est = get_equivalent_estimator(est, lib="lightgbm", n_classes=n_classes)
    # 使用转换后的 est 进行训练，使用 LightGBM 框架
    fit(est, data_train, target_train, "lightgbm")
    # 使用转换后的 est 进行预测，使用 LightGBM 框架
    predict(est, data_test, target_test)

# 如果需要使用 XGBoost 进行兼容性转换
if args.xgboost:
    # 将 est 转换为与 XGBoost 兼容的估计器
    est = get_equivalent_estimator(est, lib="xgboost", n_classes=n_classes)
    # 使用转换后的 est 进行训练，使用 XGBoost 框架
    fit(est, data_train, target_train, "xgboost")
    # 使用转换后的 est 进行预测，使用 XGBoost 框架
    predict(est, data_test, target_test)

# 如果需要使用 CatBoost 进行兼容性转换
if args.catboost:
    # 将 est 转换为与 CatBoost 兼容的估计器
    est = get_equivalent_estimator(est, lib="catboost", n_classes=n_classes)
    # 使用转换后的 est 进行训练，使用 CatBoost 框架
    fit(est, data_train, target_train, "catboost")
    # 使用转换后的 est 进行预测，使用 CatBoost 框架
    predict(est, data_test, target_test)
```