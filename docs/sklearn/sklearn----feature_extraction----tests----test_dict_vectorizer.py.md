# `D:\src\scipysrc\scikit-learn\sklearn\feature_extraction\tests\test_dict_vectorizer.py`

```
# 从 random 模块导入 Random 类
from random import Random

# 导入 numpy 库并使用别名 np
import numpy as np

# 导入 pytest 模块
import pytest

# 导入 scipy.sparse 模块并使用别名 sp
import scipy.sparse as sp

# 从 numpy.testing 模块中导入两个函数 assert_allclose 和 assert_array_equal
from numpy.testing import assert_allclose, assert_array_equal

# 导入 sklearn 库中的异常类 NotFittedError
from sklearn.exceptions import NotFittedError

# 导入 sklearn.feature_extraction 模块中的 DictVectorizer 类
from sklearn.feature_extraction import DictVectorizer

# 导入 sklearn.feature_selection 模块中的 SelectKBest 和 chi2 函数
from sklearn.feature_selection import SelectKBest, chi2


# 使用 pytest 的 parametrize 装饰器为 test_dictvectorizer 函数参数化
@pytest.mark.parametrize("sparse", (True, False))
@pytest.mark.parametrize("dtype", (int, np.float32, np.int16))
@pytest.mark.parametrize("sort", (True, False))
@pytest.mark.parametrize("iterable", (True, False))
def test_dictvectorizer(sparse, dtype, sort, iterable):
    # 创建一个包含三个字典的列表 D
    D = [{"foo": 1, "bar": 3}, {"bar": 4, "baz": 2}, {"bar": 1, "quux": 1, "quuux": 2}]

    # 创建一个 DictVectorizer 对象 v，根据参数设置 sparse, dtype, sort
    v = DictVectorizer(sparse=sparse, dtype=dtype, sort=sort)
    
    # 对列表 D 进行拟合转换为特征矩阵 X
    X = v.fit_transform(iter(D) if iterable else D)

    # 断言特征矩阵 X 是否为稀疏矩阵，根据参数 sparse 判断
    assert sp.issparse(X) == sparse

    # 断言特征矩阵 X 的形状是否为 (3, 5)
    assert X.shape == (3, 5)

    # 断言特征矩阵 X 的所有元素之和是否为 14
    assert X.sum() == 14

    # 断言通过逆变换得到的结果是否与原始字典列表 D 相等
    assert v.inverse_transform(X) == D

    # 如果 sparse 为 True，则进行下面的断言
    if sparse:
        # CSR 矩阵不能直接比较相等，需要转换为数组再比较
        assert_array_equal(
            X.toarray(), v.transform(iter(D) if iterable else D).toarray()
        )
    else:
        # 直接比较特征矩阵 X 和转换后的结果
        assert_array_equal(X, v.transform(iter(D) if iterable else D))

    # 如果 sort 为 True，则断言特征名字列表是否按字母顺序排序
    if sort:
        assert v.feature_names_ == sorted(v.feature_names_)


# 定义一个测试函数 test_feature_selection
def test_feature_selection():
    # 创建两个特征字典 d1 和 d2，包含一些有用和无用的特征
    d1 = dict([("useless%d" % i, 10) for i in range(20)], useful1=1, useful2=20)
    d2 = dict([("useless%d" % i, 10) for i in range(20)], useful1=20, useful2=1)

    # 对特征字典列表 [d1, d2] 进行拟合转换为特征矩阵 X
    for indices in (True, False):
        v = DictVectorizer().fit([d1, d2])
        X = v.transform([d1, d2])
        
        # 使用 chi2 函数进行特征选择，选择 k=2 个特征
        sel = SelectKBest(chi2, k=2).fit(X, [0, 1])

        # 根据选择的特征支持进行限制 DictVectorizer 对象 v 的特征
        v.restrict(sel.get_support(indices=indices), indices=indices)

        # 断言限制后的特征名字列表是否为 ["useful1", "useful2"]
        assert_array_equal(v.get_feature_names_out(), ["useful1", "useful2"])


# 定义一个测试函数 test_one_of_k
def test_one_of_k():
    # 创建输入特征字典列表 D_in
    D_in = [
        {"version": "1", "ham": 2},
        {"version": "2", "spam": 0.3},
        {"version=3": True, "spam": -1},
    ]

    # 创建一个 DictVectorizer 对象 v
    v = DictVectorizer()

    # 对特征字典列表 D_in 进行拟合转换为特征矩阵 X
    X = v.fit_transform(D_in)

    # 断言特征矩阵 X 的形状是否为 (3, 5)
    assert X.shape == (3, 5)

    # 对特征矩阵 X 进行逆变换，得到输出特征字典列表 D_out
    D_out = v.inverse_transform(X)

    # 断言逆变换后的第一个字典是否与原始的第一个字典相等
    assert D_out[0] == {"version=1": 1, "ham": 2}

    # 获取 DictVectorizer 对象 v 的输出特征名字列表
    names = v.get_feature_names_out()

    # 断言输出特征名字列表中是否包含 "version=2"，且不包含 "version"
    assert "version=2" in names
    assert "version" not in names


# 定义一个测试函数 test_iterable_value
def test_iterable_value():
    # 创建输入特征名字列表 D_names 和预期的特征矩阵 X_expected
    D_names = ["ham", "spam", "version=1", "version=2", "version=3"]
    X_expected = [
        [2.0, 0.0, 2.0, 1.0, 0.0],
        [0.0, 0.3, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0, 1.0],
    ]

    # 创建输入特征字典列表 D_in
    D_in = [
        {"version": ["1", "2", "1"], "ham": 2},
        {"version": "2", "spam": 0.3},
        {"version=3": True, "spam": -1},
    ]

    # 创建一个 DictVectorizer 对象 v
    v = DictVectorizer()

    # 对特征字典列表 D_in 进行拟合转换为特征矩阵 X，并转换为数组形式
    X = v.fit_transform(D_in)
    X = X.toarray()

    # 断言转换后的特征矩阵 X 是否与预期的 X_expected 相等
    assert_array_equal(X, X_expected)

    # 对特征矩阵 X 进行逆变换，得到输出特征字典列表 D_out
    D_out = v.inverse_transform(X)

    # 断言逆变换后的第一个字典是否与原始的第一个字典相等
    assert D_out[0] == {"version=1": 2, "version=2": 1, "ham": 2}
    # 调用对象 v 的 get_feature_names_out 方法，获取特征名称列表
    names = v.get_feature_names_out()
    
    # 使用断言检查 names 列表是否与 D_names 相等，如果不相等会引发 AssertionError
    assert_array_equal(names, D_names)
def test_iterable_not_string_error():
    # 定义预期的错误信息
    error_value = (
        "Unsupported type <class 'int'> in iterable value. "
        "Only iterables of string are supported."
    )
    # 创建包含不同类型数据的字典列表
    D2 = [{"foo": "1", "bar": "2"}, {"foo": "3", "baz": "1"}, {"foo": [1, "three"]}]
    # 创建DictVectorizer对象，禁用稀疏矩阵模式
    v = DictVectorizer(sparse=False)
    # 使用 pytest 检查是否引发了预期的 TypeError 异常
    with pytest.raises(TypeError) as error:
        v.fit(D2)
    # 断言错误信息与预期相符
    assert str(error.value) == error_value


def test_mapping_error():
    # 定义预期的错误信息
    error_value = (
        "Unsupported value type <class 'dict'> "
        "for foo: {'one': 1, 'three': 3}.\n"
        "Mapping objects are not supported."
    )
    # 创建包含不同类型数据的字典列表
    D2 = [
        {"foo": "1", "bar": "2"},
        {"foo": "3", "baz": "1"},
        {"foo": {"one": 1, "three": 3}},
    ]
    # 创建DictVectorizer对象，禁用稀疏矩阵模式
    v = DictVectorizer(sparse=False)
    # 使用 pytest 检查是否引发了预期的 TypeError 异常
    with pytest.raises(TypeError) as error:
        v.fit(D2)
    # 断言错误信息与预期相符
    assert str(error.value) == error_value


def test_unseen_or_no_features():
    # 创建包含特征字典的列表
    D = [{"camelot": 0, "spamalot": 1}]
    # 遍历稀疏和非稀疏模式的DictVectorizer对象
    for sparse in [True, False]:
        v = DictVectorizer(sparse=sparse).fit(D)

        # 转换新特征字典并检查结果是否为全零数组
        X = v.transform({"push the pram a lot": 2})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))

        # 转换空字典并检查结果是否为全零数组
        X = v.transform({})
        if sparse:
            X = X.toarray()
        assert_array_equal(X, np.zeros((1, 2)))

        # 使用 pytest 检查是否引发了预期的 ValueError 异常
        with pytest.raises(ValueError, match="empty"):
            v.transform([])


def test_deterministic_vocabulary(global_random_seed):
    # 生成具有不同内存布局但内容相同的字典
    items = [("%03d" % i, i) for i in range(1000)]
    rng = Random(global_random_seed)
    d_sorted = dict(items)
    rng.shuffle(items)
    d_shuffled = dict(items)

    # 创建两个DictVectorizer对象并拟合相同的输入字典
    v_1 = DictVectorizer().fit([d_sorted])
    v_2 = DictVectorizer().fit([d_shuffled])

    # 断言两个DictVectorizer对象的词汇表相同
    assert v_1.vocabulary_ == v_2.vocabulary_


def test_n_features_in():
    # 对于向量化器，n_features_in_属性无意义且不存在
    dv = DictVectorizer()
    assert not hasattr(dv, "n_features_in_")
    # 创建包含特征字典的列表，并拟合DictVectorizer对象
    d = [{"foo": 1, "bar": 2}, {"foo": 3, "baz": 1}]
    dv.fit(d)
    # 再次确认n_features_in_属性不存在
    assert not hasattr(dv, "n_features_in_")


def test_dictvectorizer_dense_sparse_equivalence():
    """Check the equivalence between between sparse and dense DictVectorizer.
    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/19978
    """
    # 创建包含不同特征和未见特征的字典列表
    movie_entry_fit = [
        {"category": ["thriller", "drama"], "year": 2003},
        {"category": ["animation", "family"], "year": 2011},
        {"year": 1974},
    ]
    movie_entry_transform = [{"category": ["thriller"], "unseen_feature": "3"}]
    # 创建稠密和稀疏模式的DictVectorizer对象
    dense_vectorizer = DictVectorizer(sparse=False)
    sparse_vectorizer = DictVectorizer(sparse=True)

    # 拟合和转换输入字典列表，并检查结果的稀疏性
    dense_vector_fit = dense_vectorizer.fit_transform(movie_entry_fit)
    sparse_vector_fit = sparse_vectorizer.fit_transform(movie_entry_fit)

    # 断言稠密模式输出不是稀疏矩阵
    assert not sp.issparse(dense_vector_fit)
    # 断言稀疏模式输出是稀疏矩阵
    assert sp.issparse(sparse_vector_fit)
    # 使用 NumPy 测试函数 assert_allclose 检查 dense_vector_fit 和 sparse_vector_fit.toarray() 的近似相等性
    assert_allclose(dense_vector_fit, sparse_vector_fit.toarray())
    
    # 使用 dense_vectorizer 对 movie_entry_transform 进行向量化转换，得到稠密表示
    dense_vector_transform = dense_vectorizer.transform(movie_entry_transform)
    
    # 使用 sparse_vectorizer 对 movie_entry_transform 进行向量化转换，得到稀疏表示
    sparse_vector_transform = sparse_vectorizer.transform(movie_entry_transform)
    
    # 检查 dense_vector_transform 是否为稠密矩阵
    assert not sp.issparse(dense_vector_transform)
    
    # 检查 sparse_vector_transform 是否为稀疏矩阵
    assert sp.issparse(sparse_vector_transform)
    
    # 使用 NumPy 测试函数 assert_allclose 检查 dense_vector_transform 和 sparse_vector_transform.toarray() 的近似相等性
    assert_allclose(dense_vector_transform, sparse_vector_transform.toarray())
    
    # 使用 dense_vectorizer 对 dense_vector_transform 进行逆向转换，得到原始条目的表示
    dense_inverse_transform = dense_vectorizer.inverse_transform(dense_vector_transform)
    
    # 使用 sparse_vectorizer 对 sparse_vector_transform 进行逆向转换，得到原始条目的表示
    sparse_inverse_transform = sparse_vectorizer.inverse_transform(sparse_vector_transform)
    
    # 预期的逆向转换结果
    expected_inverse = [{"category=thriller": 1.0}]
    
    # 检查 dense_inverse_transform 是否与预期的逆向转换结果相等
    assert dense_inverse_transform == expected_inverse
    
    # 检查 sparse_inverse_transform 是否与预期的逆向转换结果相等
    assert sparse_inverse_transform == expected_inverse
def test_dict_vectorizer_unsupported_value_type():
    """检查当与特征关联的值不受支持时是否引发错误。

    针对非回归测试：
    https://github.com/scikit-learn/scikit-learn/issues/19489
    """

    class A:
        pass

    # 创建稀疏矩阵的字典向量化器对象
    vectorizer = DictVectorizer(sparse=True)
    # 设置输入数据 X，其中包含一个特征 "foo" 对应的值为类 A 的实例
    X = [{"foo": A()}]
    # 定义期望的错误消息
    err_msg = "Unsupported value Type"
    # 使用 pytest 来验证是否会引发 TypeError，并匹配错误消息
    with pytest.raises(TypeError, match=err_msg):
        vectorizer.fit_transform(X)


def test_dict_vectorizer_get_feature_names_out():
    """检查整数特征名称在 feature_names_out 中被转换为字符串的情况。"""

    # 定义输入数据 X，包含两个字典，每个字典表示一个样本
    X = [{1: 2, 3: 4}, {2: 4}]
    # 创建非稀疏矩阵的字典向量化器对象，并进行拟合
    dv = DictVectorizer(sparse=False).fit(X)

    # 获取转换后的特征名称列表
    feature_names = dv.get_feature_names_out()
    # 断言 feature_names 是一个 numpy 数组
    assert isinstance(feature_names, np.ndarray)
    # 断言 feature_names 的数据类型是 object
    assert feature_names.dtype == object
    # 断言 feature_names 与期望的列表 ["1", "2", "3"] 相等
    assert_array_equal(feature_names, ["1", "2", "3"])


@pytest.mark.parametrize(
    "method, input",
    [
        ("transform", [{1: 2, 3: 4}, {2: 4}]),
        ("inverse_transform", [{1: 2, 3: 4}, {2: 4}]),
        ("restrict", [True, False, True]),
    ],
)
def test_dict_vectorizer_not_fitted_error(method, input):
    """检查未拟合的 DictVectorizer 实例是否会引发 NotFittedError。

    这应该是常见测试的一部分，但目前它们测试接受文本输入的估算器。
    """
    # 创建非稀疏矩阵的字典向量化器对象
    dv = DictVectorizer(sparse=False)

    # 使用 pytest 来验证调用指定方法时是否会引发 NotFittedError
    with pytest.raises(NotFittedError):
        getattr(dv, method)(input)
```