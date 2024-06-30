# `D:\src\scipysrc\scikit-learn\sklearn\neighbors\tests\test_graph.py`

```
# 导入所需的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from sklearn.metrics import euclidean_distances  # 导入欧氏距离计算函数
from sklearn.neighbors import KNeighborsTransformer, RadiusNeighborsTransformer  # 导入K近邻和半径邻居转换器类
from sklearn.neighbors._base import _is_sorted_by_data  # 导入用于检查数据是否按顺序排列的函数
from sklearn.utils._testing import assert_array_equal  # 导入用于比较数组是否相等的测试函数


def test_transformer_result():
    # 测试返回的邻居数目
    n_neighbors = 5  # 邻居数目设为5
    n_samples_fit = 20  # 用于训练的样本数目设为20
    n_queries = 18  # 查询样本数目设为18
    n_features = 10  # 特征数设为10

    rng = np.random.RandomState(42)  # 创建随机数生成器对象
    X = rng.randn(n_samples_fit, n_features)  # 生成符合正态分布的训练数据
    X2 = rng.randn(n_queries, n_features)  # 生成符合正态分布的查询数据
    radius = np.percentile(euclidean_distances(X), 10)  # 计算训练数据的欧氏距离第10百分位数作为半径

    # 使用邻居数目进行测试
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"  # 如果模式是距离模式，则增加一个
        nnt = KNeighborsTransformer(n_neighbors=n_neighbors, mode=mode)  # 创建K近邻转换器对象
        Xt = nnt.fit_transform(X)  # 对训练数据进行转换
        assert Xt.shape == (n_samples_fit, n_samples_fit)  # 检查转换后的形状是否正确
        assert Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)  # 检查转换后数据的形状是否正确
        assert Xt.format == "csr"  # 检查转换后的数据格式是否为CSR格式
        assert _is_sorted_by_data(Xt)  # 检查转换后的数据是否按数据排序

        X2t = nnt.transform(X2)  # 对查询数据进行转换
        assert X2t.shape == (n_queries, n_samples_fit)  # 检查查询数据转换后的形状是否正确
        assert X2t.data.shape == (n_queries * (n_neighbors + add_one),)  # 检查查询数据转换后数据的形状是否正确
        assert X2t.format == "csr"  # 检查查询数据转换后的数据格式是否为CSR格式
        assert _is_sorted_by_data(X2t)  # 检查查询数据转换后的数据是否按数据排序

    # 使用半径进行测试
    for mode in ["distance", "connectivity"]:
        add_one = mode == "distance"  # 如果模式是距离模式，则增加一个
        nnt = RadiusNeighborsTransformer(radius=radius, mode=mode)  # 创建半径邻居转换器对象
        Xt = nnt.fit_transform(X)  # 对训练数据进行转换
        assert Xt.shape == (n_samples_fit, n_samples_fit)  # 检查转换后的形状是否正确
        assert not Xt.data.shape == (n_samples_fit * (n_neighbors + add_one),)  # 检查转换后数据的形状是否正确
        assert Xt.format == "csr"  # 检查转换后的数据格式是否为CSR格式
        assert _is_sorted_by_data(Xt)  # 检查转换后的数据是否按数据排序

        X2t = nnt.transform(X2)  # 对查询数据进行转换
        assert X2t.shape == (n_queries, n_samples_fit)  # 检查查询数据转换后的形状是否正确
        assert not X2t.data.shape == (n_queries * (n_neighbors + add_one),)  # 检查查询数据转换后数据的形状是否正确
        assert X2t.format == "csr"  # 检查查询数据转换后的数据格式是否为CSR格式
        assert _is_sorted_by_data(X2t)  # 检查查询数据转换后的数据是否按数据排序


def _has_explicit_diagonal(X):
    """Return True if the diagonal is explicitly stored"""
    X = X.tocoo()  # 将稀疏矩阵转换为COO格式
    explicit = X.row[X.row == X.col]  # 找出显式存储的对角线元素
    return len(explicit) == X.shape[0]  # 返回是否所有对角线元素都显式存储


def test_explicit_diagonal():
    # 测试稀疏图中是否显式存储对角线元素
    n_neighbors = 5  # 邻居数目设为5
    n_samples_fit, n_samples_transform, n_features = 20, 18, 10  # 训练样本数、转换样本数和特征数

    rng = np.random.RandomState(42)  # 创建随机数生成器对象
    X = rng.randn(n_samples_fit, n_features)  # 生成符合正态分布的训练数据
    X2 = rng.randn(n_samples_transform, n_features)  # 生成符合正态分布的查询数据

    nnt = KNeighborsTransformer(n_neighbors=n_neighbors)  # 创建K近邻转换器对象
    Xt = nnt.fit_transform(X)  # 对训练数据进行转换
    assert _has_explicit_diagonal(Xt)  # 检查转换后的稀疏图是否显式存储对角线元素
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)  # 检查对角线元素是否为0

    Xt = nnt.transform(X)  # 对训练数据进行再次转换
    assert _has_explicit_diagonal(Xt)  # 检查转换后的稀疏图是否显式存储对角线元素
    assert np.all(Xt.data.reshape(n_samples_fit, n_neighbors + 1)[:, 0] == 0)  # 检查对角线元素是否为0

    # 对新数据使用transform不应总是有零对角线
    X2t = nnt.transform(X2)  # 对查询数据进行转换
    assert not _has_explicit_diagonal(X2t)  # 检查转换后的稀疏图是否未显式存储所有对角线元素


@pytest.mark.parametrize("Klass", [KNeighborsTransformer, RadiusNeighborsTransformer])
def test_graph_feature_names_out(Klass):
    """Check `get_feature_names_out` for transformers defined in `_graph.py`."""

    # 定义用于测试的样本数和特征数
    n_samples_fit = 20
    n_features = 10
    # 使用随机种子创建随机数生成器
    rng = np.random.RandomState(42)
    # 生成符合正态分布的随机数据作为输入特征矩阵 X
    X = rng.randn(n_samples_fit, n_features)

    # 创建指定类别 Klass 的实例，并对输入数据 X 进行拟合
    est = Klass().fit(X)
    # 调用拟合后的对象的方法获取输出特征名列表
    names_out = est.get_feature_names_out()

    # 获取类名的小写形式作为前缀
    class_name_lower = Klass.__name__.lower()
    # 生成预期的输出特征名列表，形式为类名小写加索引号的字符串数组
    expected_names_out = np.array(
        [f"{class_name_lower}{i}" for i in range(est.n_samples_fit_)],
        dtype=object,
    )
    # 断言实际输出特征名列表与预期输出特征名列表相等
    assert_array_equal(names_out, expected_names_out)
```