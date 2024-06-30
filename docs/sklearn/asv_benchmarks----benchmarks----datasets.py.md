# `D:\src\scipysrc\scikit-learn\asv_benchmarks\benchmarks\datasets.py`

```
# 导入必要的模块和函数
from pathlib import Path  # 导入Path类，用于处理文件和目录路径
import numpy as np  # 导入NumPy库，用于科学计算
import scipy.sparse as sp  # 导入SciPy稀疏矩阵模块
from joblib import Memory  # 导入Memory类，用于缓存数据

# 导入数据集获取函数
from sklearn.datasets import (
    fetch_20newsgroups,        # 获取20个新闻组数据集
    fetch_olivetti_faces,      # 获取奥利维蒂脸部数据集
    fetch_openml,              # 从OpenML获取数据集
    load_digits,               # 加载手写数字数据集
    make_blobs,                # 创建多类别数据集
    make_classification,       # 创建分类数据集
    make_regression,           # 创建回归数据集
)
from sklearn.decomposition import TruncatedSVD  # 导入截断SVD类，用于降维
from sklearn.feature_extraction.text import TfidfVectorizer  # 导入TF-IDF向量化类，用于文本特征提取
from sklearn.model_selection import train_test_split  # 导入数据集划分函数，用于划分训练集和测试集
from sklearn.preprocessing import MaxAbsScaler, StandardScaler  # 导入数据预处理类，用于特征缩放

# 内存缓存数据集的位置
M = Memory(location=str(Path(__file__).resolve().parent / "cache"))


@M.cache
def _blobs_dataset(n_samples=500000, n_features=3, n_clusters=100, dtype=np.float32):
    # 创建多类别数据集，用于聚类任务
    X, _ = make_blobs(
        n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=0
    )
    X = X.astype(dtype, copy=False)

    # 划分数据集为训练集和验证集
    X, X_val = train_test_split(X, test_size=0.1, random_state=0)
    return X, X_val, None, None


@M.cache
def _20newsgroups_highdim_dataset(n_samples=None, ngrams=(1, 1), dtype=np.float32):
    # 获取高维文本数据集，用于文本分类任务
    newsgroups = fetch_20newsgroups(random_state=0)
    vectorizer = TfidfVectorizer(ngram_range=ngrams, dtype=dtype)
    X = vectorizer.fit_transform(newsgroups.data[:n_samples])
    y = newsgroups.target[:n_samples]

    # 划分数据集为训练集和验证集
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


@M.cache
def _20newsgroups_lowdim_dataset(n_components=100, ngrams=(1, 1), dtype=np.float32):
    # 获取低维文本数据集，用于文本分类任务
    newsgroups = fetch_20newsgroups()
    vectorizer = TfidfVectorizer(ngram_range=ngrams)
    X = vectorizer.fit_transform(newsgroups.data)
    X = X.astype(dtype, copy=False)
    svd = TruncatedSVD(n_components=n_components)
    X = svd.fit_transform(X)
    y = newsgroups.target

    # 划分数据集为训练集和验证集
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


@M.cache
def _mnist_dataset(dtype=np.float32):
    # 获取手写数字数据集（MNIST）
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)

    # 划分数据集为训练集和验证集
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


@M.cache
def _digits_dataset(n_samples=None, dtype=np.float32):
    # 获取手写数字数据集
    X, y = load_digits(return_X_y=True)
    X = X.astype(dtype, copy=False)
    X = MaxAbsScaler().fit_transform(X)
    X = X[:n_samples]
    y = y[:n_samples]

    # 划分数据集为训练集和验证集
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


@M.cache
def _synth_regression_dataset(n_samples=100000, n_features=100, dtype=np.float32):
    # 创建合成回归数据集
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 10,
        noise=50,
        random_state=0,
    )
    X = X.astype(dtype, copy=False)
    X = StandardScaler().fit_transform(X)

    # 划分数据集为训练集和验证集
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val
# 生成稀疏回归数据集
def _synth_regression_sparse_dataset(
    n_samples=10000, n_features=10000, density=0.01, dtype=np.float32
):
    # 生成稀疏矩阵 X，格式为 CSR，密度为 density
    X = sp.random(
        m=n_samples, n=n_features, density=density, format="csr", random_state=0
    )
    # 用随机数填充 X 的非零元素
    X.data = np.random.RandomState(0).randn(X.getnnz())
    # 将 X 转换为指定的数据类型 dtype，不进行复制
    X = X.astype(dtype, copy=False)
    
    # 生成稀疏系数向量 coefs
    coefs = sp.random(m=n_features, n=1, density=0.5, random_state=0)
    coefs.data = np.random.RandomState(0).randn(coefs.getnnz())
    
    # 计算目标变量 y，通过 X 与 coefs 的矩阵乘法得到，然后重塑为一维数组
    y = X.dot(coefs.toarray()).reshape(-1)
    # 添加噪声到 y 中，使得标准差变化为原来的 20%
    y += 0.2 * y.std() * np.random.randn(n_samples)

    # 使用 train_test_split 函数分割数据集 X 和 y，测试集占比 10%，随机种子为 0
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


# 缓存装饰器，用于分类数据集生成函数
@M.cache
def _synth_classification_dataset(
    n_samples=1000, n_features=10000, n_classes=2, dtype=np.float32
):
    # 生成分类数据集 X 和目标变量 y
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        random_state=0,
        n_informative=n_features,
        n_redundant=0,
    )
    # 将 X 转换为指定的数据类型 dtype，不进行复制
    X = X.astype(dtype, copy=False)
    # 对 X 进行标准化处理
    X = StandardScaler().fit_transform(X)

    # 使用 train_test_split 函数分割数据集 X 和 y，测试集占比 10%，随机种子为 0
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    return X, X_val, y, y_val


# 缓存装饰器，用于 Olivetti 人脸数据集生成函数
@M.cache
def _olivetti_faces_dataset():
    # 获取 Olivetti 人脸数据集，打乱顺序，随机种子为 42
    dataset = fetch_olivetti_faces(shuffle=True, random_state=42)
    faces = dataset.data
    n_samples, n_features = faces.shape
    # 中心化人脸数据
    faces_centered = faces - faces.mean(axis=0)
    # 局部中心化
    faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)
    X = faces_centered

    # 使用 train_test_split 函数分割数据集 X，测试集占比 10%，随机种子为 0
    X, X_val = train_test_split(X, test_size=0.1, random_state=0)
    return X, X_val, None, None


# 缓存装饰器，用于生成随机数据集函数
@M.cache
def _random_dataset(
    n_samples=1000, n_features=1000, representation="dense", dtype=np.float32
):
    if representation == "dense":
        # 生成密集表示的随机数据集 X，形状为 (n_samples, n_features)
        X = np.random.RandomState(0).random_sample((n_samples, n_features))
        # 将 X 转换为指定的数据类型 dtype，不进行复制
        X = X.astype(dtype, copy=False)
    else:
        # 生成稀疏表示的随机数据集 X，密度为 0.05，格式为 CSR
        X = sp.random(
            n_samples,
            n_features,
            density=0.05,
            format="csr",
            dtype=dtype,
            random_state=0,
        )

    # 使用 train_test_split 函数分割数据集 X，测试集占比 10%，随机种子为 0
    X, X_val = train_test_split(X, test_size=0.1, random_state=0)
    return X, X_val, None, None
```