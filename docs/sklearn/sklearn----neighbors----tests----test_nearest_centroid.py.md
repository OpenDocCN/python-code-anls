# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_nearest_centroid.py`

```
"""
Testing for the nearest centroid module.
"""

# 导入必要的库
import numpy as np
import pytest
from numpy.testing import assert_array_equal

# 导入数据集和最近质心分类器
from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn.utils.fixes import CSR_CONTAINERS

# toy sample 数据
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]  # 样本特征
y = [-1, -1, -1, 1, 1, 1]  # 样本标签
T = [[-1, -1], [2, 2], [3, 2]]  # 待预测的样本特征
true_result = [-1, 1, 1]  # 预期的预测结果

# also load the iris dataset
# and randomly permute it
# 加载鸢尾花数据集并进行随机排列
iris = datasets.load_iris()
rng = np.random.RandomState(1)
perm = rng.permutation(iris.target.size)
iris.data = iris.data[perm]
iris.target = iris.target[perm]

# 使用 pytest 的 parametrize 装饰器对 CSR_CONTAINERS 中的每个对象运行测试
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_classification_toy(csr_container):
    # Check classification on a toy dataset, including sparse versions.
    # 在一个玩具数据集上进行分类测试，包括稀疏版本。
    
    X_csr = csr_container(X)  # 将 X 转换为 csr 格式
    T_csr = csr_container(T)  # 将 T 转换为 csr 格式

    clf = NearestCentroid()
    clf.fit(X, y)  # 使用训练数据拟合分类器
    assert_array_equal(clf.predict(T), true_result)  # 断言预测结果与预期结果一致

    # Same test, but with a sparse matrix to fit and test.
    # 使用稀疏矩阵进行相同的测试
    clf = NearestCentroid()
    clf.fit(X_csr, y)  # 使用稀疏格式的训练数据拟合分类器
    assert_array_equal(clf.predict(T_csr), true_result)  # 断言预测结果与预期结果一致

    # Fit with sparse, test with non-sparse
    # 使用稀疏数据进行拟合，使用非稀疏数据进行测试
    clf = NearestCentroid()
    clf.fit(X_csr, y)  # 使用稀疏格式的训练数据拟合分类器
    assert_array_equal(clf.predict(T), true_result)  # 断言预测结果与预期结果一致

    # Fit with non-sparse, test with sparse
    # 使用非稀疏数据进行拟合，使用稀疏数据进行测试
    clf = NearestCentroid()
    clf.fit(X, y)  # 使用原始格式的训练数据拟合分类器
    assert_array_equal(clf.predict(T_csr), true_result)  # 断言预测结果与预期结果一致

    # Fit and predict with non-CSR sparse matrices
    # 使用非 CSR 格式的稀疏矩阵进行拟合和预测
    clf = NearestCentroid()
    clf.fit(X_csr.tocoo(), y)  # 将 X_csr 转换为 coo 格式后拟合分类器
    assert_array_equal(clf.predict(T_csr.tolil()), true_result)  # 将 T_csr 转换为 lil 格式后断言预测结果与预期结果一致


def test_iris():
    # Check consistency on dataset iris.
    # 在鸢尾花数据集上检查分类器的一致性
    for metric in ("euclidean", "manhattan"):
        clf = NearestCentroid(metric=metric).fit(iris.data, iris.target)
        score = np.mean(clf.predict(iris.data) == iris.target)
        assert score > 0.9, "Failed with score = " + str(score)


def test_iris_shrinkage():
    # Check consistency on dataset iris, when using shrinkage.
    # 在鸢尾花数据集上使用收缩参数时检查分类器的一致性
    for metric in ("euclidean", "manhattan"):
        for shrink_threshold in [None, 0.1, 0.5]:
            clf = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)
            clf = clf.fit(iris.data, iris.target)
            score = np.mean(clf.predict(iris.data) == iris.target)
            assert score > 0.8, "Failed with score = " + str(score)


def test_pickle():
    import pickle

    # classification
    # 分类器的序列化和反序列化测试
    obj = NearestCentroid()
    obj.fit(iris.data, iris.target)
    score = obj.score(iris.data, iris.target)
    s = pickle.dumps(obj)

    obj2 = pickle.loads(s)
    assert type(obj2) == obj.__class__
    score2 = obj2.score(iris.data, iris.target)
    assert_array_equal(
        score,
        score2,
        "Failed to generate same score after pickling (classification).",
    )


def test_shrinkage_correct():
    # Ensure that the shrinking is correct.
    # The expected result is calculated by R (pamr),
    # which is implemented by the author of the original paper.
    # 确保收缩参数的计算正确性，期望的结果由 R (pamr) 计算得出
    # 创建一个包含样本特征的 NumPy 数组 X，每行代表一个样本，每列代表一个特征
    X = np.array([[0, 1], [1, 0], [1, 1], [2, 0], [6, 8]])
    
    # 创建一个 NumPy 数组 y，其中包含与 X 中每个样本对应的目标值
    y = np.array([1, 1, 2, 2, 2])
    
    # 使用 NearestCentroid 类创建一个分类器 clf，设置缩小阈值为 0.1
    clf = NearestCentroid(shrink_threshold=0.1)
    
    # 使用 fit 方法拟合分类器 clf 到数据 X 和目标 y
    clf.fit(X, y)
    
    # 预期的结果值，是一个 NumPy 数组，包含两个中心点的坐标
    expected_result = np.array([[0.7787310, 0.8545292], [2.814179, 2.763647]])
    
    # 使用 np.testing.assert_array_almost_equal 函数检查 clf.centroids_ 是否与 expected_result 几乎相等
    np.testing.assert_array_almost_equal(clf.centroids_, expected_result)
# 测试收缩阈值对解码后的 y 的影响
def test_shrinkage_threshold_decoded_y():
    # 创建最近质心分类器，设置收缩阈值为 0.01
    clf = NearestCentroid(shrink_threshold=0.01)
    # 将 y 转换为 NumPy 数组
    y_ind = np.asarray(y)
    # 将 y 中所有值为 -1 的元素替换为 0
    y_ind[y_ind == -1] = 0
    # 使用 X 和处理后的 y 训练分类器
    clf.fit(X, y_ind)
    # 获取分类器的质心编码
    centroid_encoded = clf.centroids_
    # 重新用 X 和原始的 y 训练分类器
    clf.fit(X, y)
    # 断言收缩编码后的质心与重新训练后的质心相等
    assert_array_equal(centroid_encoded, clf.centroids_)


# 测试在翻译数据上预测的一致性
def test_predict_translated_data():
    # 创建随机数生成器
    rng = np.random.RandomState(0)
    # 生成一个 50x50 的随机矩阵 X
    X = rng.rand(50, 50)
    # 生成一个长度为 50 的随机整数数组 y，取值范围为 0 到 2
    y = rng.randint(0, 3, 50)
    # 生成一个长度为 50 的随机噪声数组
    noise = rng.rand(50)
    # 创建最近质心分类器，设置收缩阈值为 0.1
    clf = NearestCentroid(shrink_threshold=0.1)
    # 在原始数据上训练分类器
    clf.fit(X, y)
    # 获取初始数据的预测结果
    y_init = clf.predict(X)
    # 重新创建一个最近质心分类器，设置收缩阈值为 0.1
    clf = NearestCentroid(shrink_threshold=0.1)
    # 将 X 加上噪声得到 X_noise
    X_noise = X + noise
    # 在加噪声后的数据上训练分类器
    clf.fit(X_noise, y)
    # 获取加噪声后数据的预测结果
    y_translate = clf.predict(X_noise)
    # 断言初始数据和翻译数据的预测结果一致
    assert_array_equal(y_init, y_translate)


# 使用 pytest 的参数化装饰器，测试曼哈顿距离度量
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_manhattan_metric(csr_container):
    # 测试曼哈顿距离度量
    # 将 X 转换为指定的稀疏矩阵容器类型
    X_csr = csr_container(X)

    # 创建最近质心分类器，设置距离度量方式为曼哈顿距离
    clf = NearestCentroid(metric="manhattan")
    # 在原始数据上训练分类器
    clf.fit(X, y)
    # 获取稠密格式的质心
    dense_centroid = clf.centroids_
    # 在稀疏格式的数据 X_csr 上重新训练分类器
    clf.fit(X_csr, y)
    # 断言稠密格式质心与稀疏格式质心相等
    assert_array_equal(clf.centroids_, dense_centroid)
    # 断言稠密质心与期望值相等
    assert_array_equal(dense_centroid, [[-1, -1], [1, 1]])


# 测试具有零方差特征时是否引发错误
def test_features_zero_var():
    # 测试具有零方差特征时是否引发 ValueError

    # 创建一个形状为 (10, 2) 的空数组 X
    X = np.empty((10, 2))
    # 将 X 的第一列填充为 -0.13725701
    X[:, 0] = -0.13725701
    # 将 X 的第二列填充为 -0.9853293
    X[:, 1] = -0.9853293
    # 创建一个长度为 10 的零数组 y，并将第一个元素设为 1
    y = np.zeros((10))
    y[0] = 1

    # 创建最近质心分类器，设置收缩阈值为 0.1
    clf = NearestCentroid(shrink_threshold=0.1)
    # 使用 X 和 y 训练分类器，预期会引发 ValueError
    with pytest.raises(ValueError):
        clf.fit(X, y)
```