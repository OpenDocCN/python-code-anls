# `D:\src\scipysrc\scikit-learn\sklearn\preprocessing\tests\test_function_transformer.py`

```
# 引入警告模块，用于显示警告信息
import warnings

# 引入 NumPy 库，并使用 np 别名
import numpy as np

# 引入 pytest 测试框架
import pytest

# 引入 sklearn.pipeline 模块中的 make_pipeline 函数
from sklearn.pipeline import make_pipeline

# 引入 sklearn.preprocessing 模块中的 FunctionTransformer 和 StandardScaler 类
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# 引入 sklearn.utils._testing 模块中的若干函数，包括 _convert_container, assert_allclose_dense_sparse, assert_array_equal
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose_dense_sparse,
    assert_array_equal,
)

# 引入 sklearn.utils.fixes 模块中的 CSC_CONTAINERS 和 CSR_CONTAINERS 常量
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS


def _make_func(args_store, kwargs_store, func=lambda X, *a, **k: X):
    # 定义一个内部函数 _func，用于执行具体的转换操作
    def _func(X, *args, **kwargs):
        # 将 X 添加到 args_store 中
        args_store.append(X)
        # 将额外的位置参数添加到 args_store 中
        args_store.extend(args)
        # 更新 kwargs_store 字典
        kwargs_store.update(kwargs)
        # 调用传入的 func 函数并返回结果
        return func(X)

    return _func


def test_delegate_to_func():
    # args_store 将保存传递给 FunctionTransformer 内部函数的位置参数
    args_store = []
    # kwargs_store 将保存传递给 FunctionTransformer 内部函数的关键字参数
    kwargs_store = {}
    # 创建一个 5x2 的 NumPy 数组 X
    X = np.arange(10).reshape((5, 2))
    # 调用 FunctionTransformer 对象的 transform 方法，传入自定义函数 _make_func 的结果
    assert_array_equal(
        FunctionTransformer(_make_func(args_store, kwargs_store)).transform(X),
        X,
        "transform should have returned X unchanged",
    )

    # 函数应该只收到了 X 这一个参数
    assert args_store == [
        X
    ], "Incorrect positional arguments passed to func: {args}".format(args=args_store)

    # 不应该有额外的关键字参数传递给函数
    assert (
        not kwargs_store
    ), "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)

    # 重置参数存储
    args_store[:] = []
    kwargs_store.clear()
    # 再次调用 FunctionTransformer 的 transform 方法
    transformed = FunctionTransformer(
        _make_func(args_store, kwargs_store),
    ).transform(X)

    # 确认 transform 方法返回的结果与 X 相同
    assert_array_equal(
        transformed, X, err_msg="transform should have returned X unchanged"
    )

    # 函数应该只收到了 X 这一个参数
    assert args_store == [
        X
    ], "Incorrect positional arguments passed to func: {args}".format(args=args_store)

    # 不应该有额外的关键字参数传递给函数
    assert (
        not kwargs_store
    ), "Unexpected keyword arguments passed to func: {args}".format(args=kwargs_store)


def test_np_log():
    # 创建一个 5x2 的 NumPy 数组 X
    X = np.arange(10).reshape((5, 2))

    # 测试 numpy.log1p 函数是否按预期工作
    assert_array_equal(
        FunctionTransformer(np.log1p).transform(X),
        np.log1p(X),
    )


def test_kw_arg():
    # 创建一个 5x2 的 NumPy 数组 X，其元素从 0 到 1 均匀分布
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    # 创建一个 FunctionTransformer 对象 F，调用 numpy.around 函数并传入关键字参数 decimals=3
    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    # 测试四舍五入的正确性
    assert_array_equal(F.transform(X), np.around(X, decimals=3))


def test_kw_arg_update():
    # 创建一个 5x2 的 NumPy 数组 X，其元素从 0 到 1 均匀分布
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    # 创建一个 FunctionTransformer 对象 F，调用 numpy.around 函数并传入关键字参数 decimals=3
    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    # 更新 FunctionTransformer 对象的关键字参数 decimals=1
    F.kw_args["decimals"] = 1

    # 测试四舍五入的正确性
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_kw_arg_reset():
    # 创建一个 5x2 的 NumPy 数组 X，其元素从 0 到 1 均匀分布
    X = np.linspace(0, 1, num=10).reshape((5, 2))

    # 创建一个 FunctionTransformer 对象 F，调用 numpy.around 函数并传入关键字参数 decimals=3
    F = FunctionTransformer(np.around, kw_args=dict(decimals=3))

    # 重置 FunctionTransformer 对象的关键字参数为 decimals=1
    F.kw_args = dict(decimals=1)

    # 测试四舍五入的正确性
    assert_array_equal(F.transform(X), np.around(X, decimals=1))


def test_inverse_transform():
    # 创建一个 2x2 的 NumPy 数组 X
    X = np.array([1, 4, 9, 16]).reshape((2, 2))
    # 测试 inverse_transform 方法是否正确工作
    F = FunctionTransformer(
        func=np.sqrt,                     # 设置转换函数为平方根函数
        inverse_func=np.around,           # 设置逆转换函数为四舍五入函数
        inv_kw_args=dict(decimals=3),     # 设置逆转换函数的关键字参数为保留小数点后三位
    )
    
    # 断言 inverse_transform 方法应用于 transform 方法的结果应该与 np.around(np.sqrt(X), decimals=3) 相等
    assert_array_equal(
        F.inverse_transform(F.transform(X)),  # 对 X 应用 transform 方法后再应用 inverse_transform 方法
        np.around(np.sqrt(X), decimals=3),    # 直接使用 numpy 的四舍五入函数对 sqrt(X) 进行处理
    )
# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，测试不同的稀疏容器类型以及 None 情况
@pytest.mark.parametrize("sparse_container", [None] + CSC_CONTAINERS + CSR_CONTAINERS)
def test_check_inverse(sparse_container):
    # 创建一个 2x2 的浮点数数组 X
    X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))
    # 如果 sparse_container 不为 None，则将 X 转换为稀疏格式
    if sparse_container is not None:
        X = sparse_container(X)

    # 创建 FunctionTransformer 对象 trans，配置其功能函数为 np.sqrt，逆函数为 np.around
    # 设置接受稀疏矩阵（如果 sparse_container 不为 None），并且检查逆函数是否有效
    trans = FunctionTransformer(
        func=np.sqrt,
        inverse_func=np.around,
        accept_sparse=sparse_container is not None,
        check_inverse=True,
        validate=True,
    )

    # 设置警告消息，用于检查提供的函数是否严格是彼此的逆函数
    warning_message = (
        "The provided functions are not strictly"
        " inverse of each other. If you are sure you"
        " want to proceed regardless, set"
        " 'check_inverse=False'."
    )

    # 使用 pytest 的 warns 上下文管理器，检查是否会引发 UserWarning，并且匹配警告消息
    with pytest.warns(UserWarning, match=warning_message):
        # 对数据 X 进行拟合
        trans.fit(X)

    # 创建另一个 FunctionTransformer 对象 trans，配置其功能函数为 np.expm1，逆函数为 np.log1p
    # 设置接受稀疏矩阵（如果 sparse_container 不为 None），并且检查逆函数是否有效
    trans = FunctionTransformer(
        func=np.expm1,
        inverse_func=np.log1p,
        accept_sparse=sparse_container is not None,
        check_inverse=True,
        validate=True,
    )

    # 使用 warnings 的 catch_warnings 上下文管理器，捕获 UserWarning 并转换为错误
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 对数据 X 进行拟合并转换
        Xt = trans.fit_transform(X)

    # 使用 assert_allclose_dense_sparse 检查原始数据 X 和转换后数据的逆转换结果是否接近
    assert_allclose_dense_sparse(X, trans.inverse_transform(Xt))


# 定义函数测试逆函数未提供或函数未提供时是否会引发错误
def test_check_inverse_func_or_inverse_not_provided():
    # 创建一个 2x2 的浮点数数组 X
    X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))

    # 创建 FunctionTransformer 对象 trans，配置其功能函数为 np.expm1，逆函数未提供，且需要检查逆函数是否有效
    trans = FunctionTransformer(
        func=np.expm1, inverse_func=None, check_inverse=True, validate=True
    )

    # 使用 warnings 的 catch_warnings 上下文管理器，捕获 UserWarning 并转换为错误
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 对数据 X 进行拟合
        trans.fit(X)

    # 创建另一个 FunctionTransformer 对象 trans，配置其功能函数未提供，逆函数为 np.expm1，且需要检查逆函数是否有效
    trans = FunctionTransformer(
        func=None, inverse_func=np.expm1, check_inverse=True, validate=True
    )

    # 使用 warnings 的 catch_warnings 上下文管理器，捕获 UserWarning 并转换为错误
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # 对数据 X 进行拟合
        trans.fit(X)


# 定义函数测试在 DataFrame 上使用 FunctionTransformer
def test_function_transformer_frame():
    # 导入 pandas，如果导入失败则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个包含随机数的 100x10 的 DataFrame X_df
    X_df = pd.DataFrame(np.random.randn(100, 10))
    # 创建 FunctionTransformer 对象 transformer
    transformer = FunctionTransformer()
    # 对 DataFrame X_df 进行拟合和转换
    X_df_trans = transformer.fit_transform(X_df)
    # 断言转换后的对象 X_df_trans 具有 loc 属性
    assert hasattr(X_df_trans, "loc")


# 使用 pytest.mark.parametrize 装饰器定义参数化测试函数，测试不同数据类型（array 和 series）
@pytest.mark.parametrize("X_type", ["array", "series"])
def test_function_transformer_raise_error_with_mixed_dtype(X_type):
    """Check that `FunctionTransformer.check_inverse` raises error on mixed dtype."""
    # 定义映射和逆映射字典
    mapping = {"one": 1, "two": 2, "three": 3, 5: "five", 6: "six"}
    inverse_mapping = {value: key for key, value in mapping.items()}
    # 数据类型设置为 object
    dtype = "object"

    # 创建包含混合数据类型的列表 data，并将其转换为指定的数据类型 X_type
    data = ["one", "two", "three", "one", "one", 5, 6]
    data = _convert_container(data, X_type, columns_name=["value"], dtype=dtype)

    # 定义功能函数 func 和逆函数 inverse_func
    def func(X):
        return np.array([mapping[X[i]] for i in range(X.size)], dtype=object)

    def inverse_func(X):
        return _convert_container(
            [inverse_mapping[x] for x in X],
            X_type,
            columns_name=["value"],
            dtype=dtype,
        )

    # 创建 FunctionTransformer 对象 transformer，配置其功能函数为 func 和 inverse_func
    # 设置不验证数据，但需要检查逆函数是否有效
    transformer = FunctionTransformer(
        func=func, inverse_func=inverse_func, validate=False, check_inverse=True
    )
    # 定义错误消息，指出在数据集 `X` 的所有元素都为数值类型时才支持 'check_inverse'。
    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    
    # 使用 pytest 来断言抛出 ValueError 异常，并匹配特定的错误消息 `msg`。
    with pytest.raises(ValueError, match=msg):
        # 对转换器进行拟合操作，预期会引发 ValueError 异常
        transformer.fit(data)
# 定义一个测试函数，检查 FunctionTransformer 是否支持仅包含数值数据的 DataFrame，并验证反转函数。
def test_function_transformer_support_all_nummerical_dataframes_check_inverse_True():
    """Check support for dataframes with only numerical values."""
    # 导入并检查是否有 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个包含数值数据的 DataFrame
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    
    # 创建 FunctionTransformer 对象，指定转换函数和反转换函数，并启用反转函数检查
    transformer = FunctionTransformer(
        func=lambda x: x + 2, inverse_func=lambda x: x - 2, check_inverse=True
    )

    # 使用 FunctionTransformer 对 DataFrame 进行拟合转换，不应该引发错误
    df_out = transformer.fit_transform(df)
    # 验证转换后的结果与预期的 DataFrame + 2 相等
    assert_allclose_dense_sparse(df_out, df + 2)


# 定义一个测试函数，检查在 check_inverse=True 时是否会引发错误。
# 这是 gh-25261 的非回归测试。
def test_function_transformer_with_dataframe_and_check_inverse_True():
    """Check error is raised when check_inverse=True.

    Non-regresion test for gh-25261.
    """
    # 导入并检查是否有 pandas 库，如果不存在则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建 FunctionTransformer 对象，指定转换函数和反转换函数，并启用反转函数检查
    transformer = FunctionTransformer(
        func=lambda x: x, inverse_func=lambda x: x, check_inverse=True
    )

    # 创建一个包含混合数据（数值和非数值）的 DataFrame
    df_mixed = pd.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    
    # 预期引发的错误消息，当输入数据中包含非数值数据时，'check_inverse' 只支持数值数据
    msg = "'check_inverse' is only supported when all the elements in `X` is numerical."
    
    # 使用 pytest 来检查是否引发预期的 ValueError 错误，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        transformer.fit(df_mixed)


@pytest.mark.parametrize(
    "X, feature_names_out, input_features, expected",
    [
        (
            # NumPy inputs, default behavior: generate names
            np.random.rand(100, 3),  # 生成一个 100 行 3 列的随机数数组作为输入
            "one-to-one",            # 数据变换的模式为一对一
            None,                    # 没有额外的参数传递
            ("x0", "x1", "x2"),      # 输入特征的名称为 x0, x1, x2
        ),
        (
            # Pandas input, default behavior: use input feature names
            {"a": np.random.rand(100), "b": np.random.rand(100)},  # 生成两列长度为 100 的随机数作为输入，列名为 a 和 b
            "one-to-one",            # 数据变换的模式为一对一
            None,                    # 没有额外的参数传递
            ("a", "b"),              # 输入特征的名称为 a, b
        ),
        (
            # NumPy input, feature_names_out=callable
            np.random.rand(100, 3),   # 生成一个 100 行 3 列的随机数数组作为输入
            lambda transformer, input_features: ("a", "b"),  # 自定义函数返回输出特征的名称为 a, b
            None,                     # 没有额外的参数传递
            ("a", "b"),               # 输入特征的名称为 a, b
        ),
        (
            # Pandas input, feature_names_out=callable
            {"a": np.random.rand(100), "b": np.random.rand(100)},  # 生成两列长度为 100 的随机数作为输入，列名为 a 和 b
            lambda transformer, input_features: ("c", "d", "e"),   # 自定义函数返回输出特征的名称为 c, d, e
            None,                     # 没有额外的参数传递
            ("c", "d", "e"),           # 输入特征的名称为 c, d, e
        ),
        (
            # NumPy input, feature_names_out=callable – default input_features
            np.random.rand(100, 3),   # 生成一个 100 行 3 列的随机数数组作为输入
            lambda transformer, input_features: tuple(input_features) + ("a",),  # 自定义函数返回输出特征名称为 input_features 加上 "a"
            None,                     # 没有额外的参数传递
            ("x0", "x1", "x2", "a"),   # 输入特征的名称为 x0, x1, x2, a
        ),
        (
            # Pandas input, feature_names_out=callable – default input_features
            {"a": np.random.rand(100), "b": np.random.rand(100)},  # 生成两列长度为 100 的随机数作为输入，列名为 a 和 b
            lambda transformer, input_features: tuple(input_features) + ("c",),  # 自定义函数返回输出特征名称为 input_features 加上 "c"
            None,                     # 没有额外的参数传递
            ("a", "b", "c"),           # 输入特征的名称为 a, b, c
        ),
        (
            # NumPy input, input_features=list of names
            np.random.rand(100, 3),   # 生成一个 100 行 3 列的随机数数组作为输入
            "one-to-one",             # 数据变换的模式为一对一
            ("a", "b", "c"),          # 输入特征的名称为 a, b, c
            ("a", "b", "c"),          # 输出特征的名称为 a, b, c
        ),
        (
            # Pandas input, input_features=list of names
            {"a": np.random.rand(100), "b": np.random.rand(100)},  # 生成两列长度为 100 的随机数作为输入，列名为 a 和 b
            "one-to-one",             # 数据变换的模式为一对一
            ("a", "b"),               # 输入特征的名称为 a, b，必须和 feature_names_in_ 匹配
            ("a", "b"),               # 输出特征的名称为 a, b
        ),
        (
            # NumPy input, feature_names_out=callable, input_features=list
            np.random.rand(100, 3),   # 生成一个 100 行 3 列的随机数数组作为输入
            lambda transformer, input_features: tuple(input_features) + ("d",),  # 自定义函数返回输出特征名称为 input_features 加上 "d"
            ("a", "b", "c"),          # 输入特征的名称为 a, b, c
            ("a", "b", "c", "d"),     # 输出特征的名称为 a, b, c, d
        ),
        (
            # Pandas input, feature_names_out=callable, input_features=list
            {"a": np.random.rand(100), "b": np.random.rand(100)},  # 生成两列长度为 100 的随机数作为输入，列名为 a 和 b
            lambda transformer, input_features: tuple(input_features) + ("c",),  # 自定义函数返回输出特征名称为 input_features 加上 "c"
            ("a", "b"),               # 输入特征的名称为 a, b，必须和 feature_names_in_ 匹配
            ("a", "b", "c"),          # 输出特征的名称为 a, b, c
        ),
    ],
@pytest.mark.parametrize(
    "feature_names_out, expected",
    # 参数化测试，定义参数 feature_names_out 和 expected，用于后续的测试用例
)
def test_function_transformer_get_feature_names_out(
    X, feature_names_out, input_features, expected, validate
):
    # 如果输入 X 是字典类型，则导入并使用 pandas 库将其转换为 DataFrame
    if isinstance(X, dict):
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)

    # 创建 FunctionTransformer 对象，并传入参数 feature_names_out 和 validate
    transformer = FunctionTransformer(
        feature_names_out=feature_names_out, validate=validate
    )
    # 对 transformer 对象进行拟合
    transformer.fit(X)
    # 调用 get_feature_names_out 方法获取特征名列表
    names = transformer.get_feature_names_out(input_features)
    # 断言返回的 names 类型为 np.ndarray
    assert isinstance(names, np.ndarray)
    # 断言 names 的数据类型为 object
    assert names.dtype == object
    # 使用 assert_array_equal 方法断言 names 和期望的结果 expected 相等
    assert_array_equal(names, expected)


def test_function_transformer_get_feature_names_out_without_validation():
    # 创建不带验证的 FunctionTransformer 对象
    transformer = FunctionTransformer(feature_names_out="one-to-one", validate=False)
    # 生成一个随机的 100x2 的数组作为输入 X
    X = np.random.rand(100, 2)
    # 对 transformer 对象进行拟合转换
    transformer.fit_transform(X)

    # 调用 get_feature_names_out 方法获取特征名列表
    names = transformer.get_feature_names_out(("a", "b"))
    # 断言返回的 names 类型为 np.ndarray
    assert isinstance(names, np.ndarray)
    # 断言 names 的数据类型为 object
    assert names.dtype == object
    # 使用 assert_array_equal 方法断言 names 和 ("a", "b") 相等
    assert_array_equal(names, ("a", "b"))


def test_function_transformer_feature_names_out_is_None():
    # 创建一个默认的 FunctionTransformer 对象
    transformer = FunctionTransformer()
    # 生成一个随机的 100x2 的数组作为输入 X
    X = np.random.rand(100, 2)
    # 对 transformer 对象进行拟合转换
    transformer.fit_transform(X)

    # 准备一个异常消息
    msg = "This 'FunctionTransformer' has no attribute 'get_feature_names_out'"
    # 使用 pytest.raises 检查是否抛出预期的 AttributeError 异常
    with pytest.raises(AttributeError, match=msg):
        transformer.get_feature_names_out()


def test_function_transformer_feature_names_out_uses_estimator():
    # 定义一个函数，用于向输入 X 中添加 n 个随机特征
    def add_n_random_features(X, n):
        return np.concatenate([X, np.random.rand(len(X), n)], axis=1)

    # 定义一个函数，用于生成特征名列表，输入参数包括 transformer 和 input_features
    def feature_names_out(transformer, input_features):
        n = transformer.kw_args["n"]
        return list(input_features) + [f"rnd{i}" for i in range(n)]

    # 创建 FunctionTransformer 对象，传入自定义的函数、特征名生成函数及其他参数
    transformer = FunctionTransformer(
        func=add_n_random_features,
        feature_names_out=feature_names_out,
        kw_args=dict(n=3),
        validate=True,
    )
    # 导入 pandas 库，创建一个包含随机数据的 DataFrame
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
    # 对 transformer 对象进行拟合转换
    transformer.fit_transform(df)
    # 调用 get_feature_names_out 方法获取特征名列表
    names = transformer.get_feature_names_out()

    # 断言返回的 names 类型为 np.ndarray
    assert isinstance(names, np.ndarray)
    # 断言 names 的数据类型为 object
    assert names.dtype == object
    # 使用 assert_array_equal 方法断言 names 和 ("a", "b", "rnd0", "rnd1", "rnd2") 相等
    assert_array_equal(names, ("a", "b", "rnd0", "rnd1", "rnd2"))


def test_function_transformer_validate_inverse():
    """Test that function transformer does not reset estimator in
    `inverse_transform`."""

    # 定义一个函数，向输入 X 中添加常数特征列
    def add_constant_feature(X):
        X_one = np.ones((X.shape[0], 1))
        return np.concatenate((X, X_one), axis=1)

    # 定义一个逆变换函数，用于从增加常数特征后的 X 中移除新增的特征列
    def inverse_add_constant(X):
        return X[:, :-1]

    # 创建输入数据 X
    X = np.array([[1, 2], [3, 4], [3, 4]])
    # 创建 FunctionTransformer 对象，传入自定义的函数及其他参数
    trans = FunctionTransformer(
        func=add_constant_feature,
        inverse_func=inverse_add_constant,
        validate=True,
    )
    # 对 trans 对象进行拟合转换
    X_trans = trans.fit_transform(X)
    # 断言 trans.n_features_in_ 等于 X 的列数
    assert trans.n_features_in_ == X.shape[1]

    # 调用 inverse_transform 方法对 X_trans 进行逆转换
    trans.inverse_transform(X_trans)
    # 再次断言 trans.n_features_in_ 等于 X 的列数
    assert trans.n_features_in_ == X.shape[1]
    [
        # 元组中的第一个元素表示 "one-to-one"，其对应的属性列表为 ["pet", "color"]
        ("one-to-one", ["pet", "color"]),
        # 元组中的第二个元素是一个列表，其中包含一个 lambda 函数和其所需的参数列表
        # lambda 函数接受两个参数 est 和 names，返回一个列表，将 names 中的每个元素添加后缀 "_out"
        [lambda est, names: [f"{n}_out" for n in names], ["pet_out", "color_out"]],
    ],
# 使用 pytest 的装饰器标记这个函数作为一个测试函数，使用参数化来测试不同的情况
@pytest.mark.parametrize("in_pipeline", [True, False])
def test_get_feature_names_out_dataframe_with_string_data(
    feature_names_out, expected, in_pipeline
):
    """Check that get_feature_names_out works with DataFrames with string data."""
    # 导入 pandas 库，如果失败则跳过测试
    pd = pytest.importorskip("pandas")
    
    # 创建一个包含两列的 DataFrame，每列有两个字符串值
    X = pd.DataFrame({"pet": ["dog", "cat"], "color": ["red", "green"]})

    # 定义一个函数 func，根据 feature_names_out 的不同返回不同的 DataFrame
    def func(X):
        if feature_names_out == "one-to-one":
            return X
        else:
            # 根据 feature_names_out 函数生成新的列名，并返回重命名后的 DataFrame
            name = feature_names_out(None, X.columns)
            return X.rename(columns=dict(zip(X.columns, name)))

    # 创建一个 FunctionTransformer 对象，传入 func 函数和 feature_names_out 参数
    transformer = FunctionTransformer(func=func, feature_names_out=feature_names_out)
    
    # 如果 in_pipeline 为 True，则将 transformer 放入管道中
    if in_pipeline:
        transformer = make_pipeline(transformer)

    # 对 X 应用 transformer 进行转换
    X_trans = transformer.fit_transform(X)
    
    # 断言转换后的结果类型为 pandas 的 DataFrame
    assert isinstance(X_trans, pd.DataFrame)

    # 获取 transformer 的输出列名
    names = transformer.get_feature_names_out()
    
    # 断言获取的列名类型为 numpy 数组，并且数据类型为 object
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    
    # 断言获取的列名与期望的列名数组相等
    assert_array_equal(names, expected)


# 定义一个测试函数，用于测试 set_output 方法的行为
def test_set_output_func():
    """Check behavior of set_output with different settings."""
    # 导入 pandas 库，如果失败则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个包含两列的 DataFrame，每列有三个整数值
    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})

    # 创建一个 FunctionTransformer 对象，使用 np.log 函数进行转换，并指定 feature_names_out 为 "one-to-one"
    ft = FunctionTransformer(np.log, feature_names_out="one-to-one")

    # 在 feature_names_out 定义时不会触发警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ft.set_output(transform="pandas")

    # 对 X 应用 ft 进行转换
    X_trans = ft.fit_transform(X)
    
    # 断言转换后的结果类型为 pandas 的 DataFrame
    assert isinstance(X_trans, pd.DataFrame)
    
    # 断言转换后的列名与原始列名相等
    assert_array_equal(X_trans.columns, ["a", "b"])

    # 创建一个 FunctionTransformer 对象，使用 lambda 函数进行转换
    ft = FunctionTransformer(lambda x: 2 * x)
    ft.set_output(transform="pandas")

    # 在 func 返回 pandas DataFrame 时不会触发警告
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        X_trans = ft.fit_transform(X)
    
    # 断言转换后的结果类型为 pandas 的 DataFrame
    assert isinstance(X_trans, pd.DataFrame)
    
    # 断言转换后的列名与原始列名相等
    assert_array_equal(X_trans.columns, ["a", "b"])

    # 创建一个 FunctionTransformer 对象，使用 lambda 函数返回 ndarray
    ft_np = FunctionTransformer(lambda x: np.asarray(x))

    # 分别设置 transform 参数为 "pandas" 和 "polars" 时，会触发警告
    for transform in ("pandas", "polars"):
        ft_np.set_output(transform=transform)
        msg = (
            f"When `set_output` is configured to be '{transform}'.*{transform} "
            "DataFrame.*"
        )
        with pytest.warns(UserWarning, match=msg):
            ft_np.fit_transform(X)

    # 当 transform 参数设置为 "default" 时不会触发警告
    ft_np.set_output(transform="default")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ft_np.fit_transform(X)


# 定义一个测试函数，用于测试 feature names 在 `FunctionTransformer` 和 pipeline 中的一致性
def test_consistence_column_name_between_steps():
    """Check that we have a consistence between the feature names out of
    `FunctionTransformer` and the feature names in of the next step in the pipeline.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/27695
    """
    # 导入 pandas 库，如果失败则跳过测试
    pd = pytest.importorskip("pandas")

    # 定义一个函数，对输入的列名列表添加后缀 "__log"，并返回新的列名列表
    def with_suffix(_, names):
        return [name + "__log" for name in names]
    # 创建一个数据处理流水线，包括对特征进行对数变换和标准化处理
    pipeline = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out=with_suffix), StandardScaler()
    )

    # 创建一个包含指定数据的 pandas DataFrame，列名为 "a" 和 "b"
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["a", "b"])
    
    # 使用创建的流水线对 DataFrame 进行拟合和转换
    X_trans = pipeline.fit_transform(df)
    
    # 断言流水线的输出特征名列表应该与预期的 ["a__log", "b__log"] 相匹配
    assert pipeline.get_feature_names_out().tolist() == ["a__log", "b__log"]
    
    # 断言 X_trans 的类型应为 numpy 数组
    # 因为 StandardScaler 将数据转换为 numpy 数组
    assert isinstance(X_trans, np.ndarray)
@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
@pytest.mark.parametrize("transform_output", ["default", "pandas", "polars"])
def test_function_transformer_overwrite_column_names(dataframe_lib, transform_output):
    """Check that we overwrite the column names when we should."""
    # 导入所需的数据框库，如果库不存在则跳过测试
    lib = pytest.importorskip(dataframe_lib)
    if transform_output != "numpy":
        # 如果输出转换选项不是 numpy，则需要导入对应的转换库，否则跳过测试
        pytest.importorskip(transform_output)

    # 创建一个数据框，包含两列 'a' 和 'b'
    df = lib.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})

    # 定义一个函数，用于在列名后面添加后缀 '__log'
    def with_suffix(_, names):
        return [name + "__log" for name in names]

    # 创建一个函数转换器对象，设置输出特征名称处理函数为 with_suffix
    transformer = FunctionTransformer(feature_names_out=with_suffix).set_output(
        transform=transform_output
    )
    # 对数据框进行拟合转换
    X_trans = transformer.fit_transform(df)
    # 断言转换后的数组与原始数据框的数组表示相等
    assert_array_equal(np.asarray(X_trans), np.asarray(df))

    # 获取转换后的特征名称列表
    feature_names = transformer.get_feature_names_out()
    # 断言转换后的数据框列名与应用 with_suffix 后的列名列表相等
    assert list(X_trans.columns) == with_suffix(None, df.columns)
    # 断言特征名称列表与应用 with_suffix 后的列名列表相等
    assert feature_names.tolist() == with_suffix(None, df.columns)


@pytest.mark.parametrize(
    "feature_names_out",
    ["one-to-one", lambda _, names: [f"{name}_log" for name in names]],
)
def test_function_transformer_overwrite_column_names_numerical(feature_names_out):
    """Check the same as `test_function_transformer_overwrite_column_names`
    but for the specific case of pandas where column names can be numerical."""
    # 导入 pandas 库，如果库不存在则跳过测试
    pd = pytest.importorskip("pandas")

    # 创建一个包含数字列名的 pandas 数据框
    df = pd.DataFrame({0: [1, 2, 3], 1: [10, 20, 100]})

    # 创建一个函数转换器对象，设置输出特征名称处理函数为 feature_names_out 参数
    transformer = FunctionTransformer(feature_names_out=feature_names_out)
    # 对数据框进行拟合转换
    X_trans = transformer.fit_transform(df)
    # 断言转换后的数组与原始数据框的数组表示相等
    assert_array_equal(np.asarray(X_trans), np.asarray(df))

    # 获取转换后的特征名称列表
    feature_names = transformer.get_feature_names_out()
    # 断言转换后的数据框列名与特征名称列表相等
    assert list(X_trans.columns) == list(feature_names)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
@pytest.mark.parametrize(
    "feature_names_out",
    ["one-to-one", lambda _, names: [f"{name}_log" for name in names]],
)
def test_function_transformer_error_column_inconsistent(
    dataframe_lib, feature_names_out
):
    """Check that we raise an error when `func` returns a dataframe with new
    column names that become inconsistent with `get_feature_names_out`."""
    # 导入所需的数据框库，如果库不存在则跳过测试
    lib = pytest.importorskip(dataframe_lib)

    # 创建一个数据框，包含两列 'a' 和 'b'
    df = lib.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})

    # 定义一个函数 func，根据数据框库类型重命名 'a' 列为 'c'
    def func(df):
        if dataframe_lib == "pandas":
            return df.rename(columns={"a": "c"})
        else:
            return df.rename({"a": "c"})

    # 创建一个函数转换器对象，设置输出特征名称处理函数为 feature_names_out 参数和 func 函数
    transformer = FunctionTransformer(func=func, feature_names_out=feature_names_out)
    # 准备一个错误消息，用于断言抛出 ValueError 异常
    err_msg = "The output generated by `func` have different column names"
    # 断言调用 fit_transform 方法时抛出 ValueError 异常，并且异常消息与 err_msg 匹配
    with pytest.raises(ValueError, match=err_msg):
        transformer.fit_transform(df).columns
```