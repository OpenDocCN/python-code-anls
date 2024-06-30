# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\tests\test_feature_hasher.py`

```
# 导入必要的库
import numpy as np
import pytest
from numpy.testing import assert_array_equal
# 导入特征哈希相关模块
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction._hashing_fast import transform as _hashing_transform


# 测试特征哈希器对字典类型数据的处理
def test_feature_hasher_dicts():
    # 创建特征哈希器对象，设定特征数量为16
    feature_hasher = FeatureHasher(n_features=16)
    # 断言输入类型为字典
    assert "dict" == feature_hasher.input_type

    # 定义原始数据集
    raw_X = [{"foo": "bar", "dada": 42, "tzara": 37}, {"foo": "baz", "gaga": "string1"}]
    # 使用特征哈希器转换数据集并生成稀疏矩阵X1
    X1 = FeatureHasher(n_features=16).transform(raw_X)
    # 生成一个字典迭代器
    gen = (iter(d.items()) for d in raw_X)
    # 使用特征哈希器以"pair"输入类型转换生成稀疏矩阵X2
    X2 = FeatureHasher(n_features=16, input_type="pair").transform(gen)
    # 断言两个稀疏矩阵相等
    assert_array_equal(X1.toarray(), X2.toarray())


# 测试特征哈希器对字符串类型数据的处理
def test_feature_hasher_strings():
    # 定义包含混合字节和Unicode字符串的原始数据集
    raw_X = [
        ["foo", "bar", "baz", "foo".encode("ascii")],
        ["bar".encode("ascii"), "baz", "quux"],
    ]

    # 遍历不同的lg_n_features值
    for lg_n_features in (7, 9, 11, 16, 22):
        n_features = 2**lg_n_features

        # 创建数据集迭代器it
        it = (x for x in raw_X)  # iterable

        # 创建特征哈希器对象，设定特征数量和输入类型
        feature_hasher = FeatureHasher(
            n_features=n_features, input_type="string", alternate_sign=False
        )
        # 使用特征哈希器转换数据集it并生成稀疏矩阵X
        X = feature_hasher.transform(it)

        # 断言生成稀疏矩阵的形状符合预期
        assert X.shape[0] == len(raw_X)
        assert X.shape[1] == n_features

        # 断言每行稀疏矩阵的非零元素个数符合预期
        assert X[0].sum() == 4
        assert X[1].sum() == 3

        # 断言稀疏矩阵的总非零元素个数符合预期
        assert X.nnz == 6


# 使用参数化测试，测试特征哈希器处理单个字符串时是否会引发错误
@pytest.mark.parametrize(
    "raw_X",
    [
        ["my_string", "another_string"],
        (x for x in ["my_string", "another_string"]),
    ],
    ids=["list", "generator"],
)
def test_feature_hasher_single_string(raw_X):
    """FeatureHasher raises error when a sample is a single string.

    Non-regression test for gh-13199.
    """
    msg = "Samples can not be a single string"

    # 创建特征哈希器对象，设定特征数量和输入类型为字符串
    feature_hasher = FeatureHasher(n_features=10, input_type="string")
    # 断言调用特征哈希器转换raw_X会引发特定错误
    with pytest.raises(ValueError, match=msg):
        feature_hasher.transform(raw_X)


# 测试_hashing_transform函数的种子影响
def test_hashing_transform_seed():
    # 定义包含混合字节和Unicode字符串的原始数据集
    raw_X = [
        ["foo", "bar", "baz", "foo".encode("ascii")],
        ["bar".encode("ascii"), "baz", "quux"],
    ]

    # 创建生成器raw_X_
    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    # 调用_hashing_transform函数计算哈希值，不设置种子
    indices, indptr, _ = _hashing_transform(raw_X_, 2**7, str, False)

    # 重新创建生成器raw_X_
    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    # 调用_hashing_transform函数计算哈希值，设置种子为0
    indices_0, indptr_0, _ = _hashing_transform(raw_X_, 2**7, str, False, seed=0)
    # 断言不设置种子和种子为0时计算出的哈希值相等
    assert_array_equal(indices, indices_0)
    assert_array_equal(indptr, indptr_0)

    # 重新创建生成器raw_X_
    raw_X_ = (((f, 1) for f in x) for x in raw_X)
    # 调用_hashing_transform函数计算哈希值，设置种子为1
    indices_1, _, _ = _hashing_transform(raw_X_, 2**7, str, False, seed=1)
    # 断言设置种子为1时计算出的哈希值与未设置种子时不相等
    with pytest.raises(AssertionError):
        assert_array_equal(indices, indices_1)


# 测试特征哈希器对(pair)类型数据的处理
def test_feature_hasher_pairs():
    # 定义包含字典生成器的原始数据集raw_X
    raw_X = (
        iter(d.items())
        for d in [{"foo": 1, "bar": 2}, {"baz": 3, "quux": 4, "foo": -1}]
    )
    # 创建特征哈希器对象，设定特征数量和输入类型为(pair)
    feature_hasher = FeatureHasher(n_features=16, input_type="pair")
    # 使用特征哈希器转换raw_X并生成稀疏矩阵，分别赋值给x1和x2
    x1, x2 = feature_hasher.transform(raw_X).toarray()
    # 对数组 x1 中非零元素取绝对值并排序，赋值给 x1_nz
    x1_nz = sorted(np.abs(x1[x1 != 0]))
    # 对数组 x2 中非零元素取绝对值并排序，赋值给 x2_nz
    x2_nz = sorted(np.abs(x2[x2 != 0]))
    # 断言 x1_nz 应该等于 [1, 2]
    assert [1, 2] == x1_nz
    # 断言 x2_nz 应该等于 [1, 3, 4]
    assert [1, 3, 4] == x2_nz
def test_feature_hasher_pairs_with_string_values():
    # 创建一个生成器，逐个迭代字典中的键值对列表
    raw_X = (
        iter(d.items())
        for d in [{"foo": 1, "bar": "a"}, {"baz": "abc", "quux": 4, "foo": -1}]
    )
    # 初始化特征哈希器，指定特征数量和输入类型为"pair"
    feature_hasher = FeatureHasher(n_features=16, input_type="pair")
    # 对 raw_X 应用特征哈希器进行转换，并转换为稀疏矩阵的数组表示
    x1, x2 = feature_hasher.transform(raw_X).toarray()
    # 提取非零元素的绝对值并排序
    x1_nz = sorted(np.abs(x1[x1 != 0]))
    x2_nz = sorted(np.abs(x2[x2 != 0]))
    # 断言检查结果
    assert [1, 1] == x1_nz
    assert [1, 1, 4] == x2_nz

    # 创建另一个生成器，逐个迭代字典中的键值对列表
    raw_X = (iter(d.items()) for d in [{"bax": "abc"}, {"bax": "abc"}])
    # 再次对 raw_X 应用特征哈希器进行转换，并转换为稀疏矩阵的数组表示
    x1, x2 = feature_hasher.transform(raw_X).toarray()
    # 提取非零元素的绝对值
    x1_nz = np.abs(x1[x1 != 0])
    x2_nz = np.abs(x2[x2 != 0])
    # 断言检查结果
    assert [1] == x1_nz
    assert [1] == x2_nz
    # 断言检查两个数组是否相等
    assert_array_equal(x1, x2)


def test_hash_empty_input():
    # 定义特征数量
    n_features = 16
    # 创建一个空列表、空元组和空迭代器的列表
    raw_X = [[], (), iter(range(0))]

    # 初始化特征哈希器，指定特征数量和输入类型为"string"
    feature_hasher = FeatureHasher(n_features=n_features, input_type="string")
    # 对 raw_X 应用特征哈希器进行转换
    X = feature_hasher.transform(raw_X)

    # 断言检查转换后的稀疏矩阵是否全为零
    assert_array_equal(X.toarray(), np.zeros((len(raw_X), n_features)))


def test_hasher_zeros():
    # 断言检查输出的数据形状是否为空
    X = FeatureHasher().transform([{"foo": 0}])
    assert X.data.shape == (0,)


def test_hasher_alternate_sign():
    # 创建包含单词列表的列表
    X = [list("Thequickbrownfoxjumped")]

    # 初始化特征哈希器，指定使用交替符号和输入类型为"string"，并进行拟合和转换
    Xt = FeatureHasher(alternate_sign=True, input_type="string").fit_transform(X)
    # 断言检查转换后的数据中是否存在负数和正数
    assert Xt.data.min() < 0 and Xt.data.max() > 0

    # 初始化特征哈希器，指定不使用交替符号和输入类型为"string"，并进行拟合和转换
    Xt = FeatureHasher(alternate_sign=False, input_type="string").fit_transform(X)
    # 断言检查转换后的数据是否全为正数
    assert Xt.data.min() > 0


def test_hash_collisions():
    # 创建包含单词列表的列表
    X = [list("Thequickbrownfoxjumped")]

    # 初始化特征哈希器，指定使用交替符号、特征数量为1和输入类型为"string"，并进行拟合和转换
    Xt = FeatureHasher(
        alternate_sign=True, n_features=1, input_type="string"
    ).fit_transform(X)
    # 断言检查转换后的数据中哈希碰撞是否存在
    assert abs(Xt.data[0]) < len(X[0])

    # 初始化特征哈希器，指定不使用交替符号、特征数量为1和输入类型为"string"，并进行拟合和转换
    Xt = FeatureHasher(
        alternate_sign=False, n_features=1, input_type="string"
    ).fit_transform(X)
    # 断言检查转换后的数据中哈希碰撞是否完全相等
    assert Xt.data[0] == len(X[0])
```