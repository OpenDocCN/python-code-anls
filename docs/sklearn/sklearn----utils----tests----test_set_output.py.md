# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_set_output.py`

```
import importlib  # 导入 importlib 模块，用于动态导入其他模块
from collections import namedtuple  # 导入 namedtuple 类型，用于创建具名元组

import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 库，用于单元测试

from numpy.testing import assert_array_equal  # 导入 assert_array_equal 函数，用于比较 NumPy 数组

from sklearn._config import config_context, get_config  # 导入 sklearn 库的配置相关模块
from sklearn.preprocessing import StandardScaler  # 导入 sklearn 库的数据预处理模块
from sklearn.utils._set_output import (  # 导入 sklearn 库的输出设置相关函数和类
    ADAPTERS_MANAGER,  # Adapter 管理器，用于管理不同数据容器的适配器
    ContainerAdapterProtocol,  # 容器适配器协议
    _get_adapter_from_container,  # 从容器获取适配器的函数
    _get_output_config,  # 获取输出配置的函数
    _safe_set_output,  # 安全设置输出的函数
    _SetOutputMixin,  # 设置输出的 Mixin 类
    _wrap_data_with_container,  # 将数据封装在容器中的函数
    check_library_installed,  # 检查库是否安装的函数
)
from sklearn.utils.fixes import CSR_CONTAINERS  # 导入 sklearn 库的修复模块，处理 CSR 容器

def test_pandas_adapter():
    """Check pandas adapter has expected behavior."""
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果不存在则跳过测试

    # 创建 NumPy 数组 X_np 和对应的列名 columns、行索引 index
    X_np = np.asarray([[1, 0, 3], [0, 0, 1]])
    columns = np.asarray(["f0", "f1", "f2"], dtype=object)
    index = np.asarray([0, 1])
    # 创建原始的 Pandas DataFrame X_df_orig
    X_df_orig = pd.DataFrame([[1, 2], [1, 3]], index=index)

    # 获取 pandas 适配器
    adapter = ADAPTERS_MANAGER.adapters["pandas"]
    # 使用适配器创建 X_np 的 Pandas DataFrame 容器 X_container
    X_container = adapter.create_container(X_np, X_df_orig, columns=lambda: columns)
    assert isinstance(X_container, pd.DataFrame)  # 断言 X_container 是 Pandas DataFrame 类型
    assert_array_equal(X_container.columns, columns)  # 断言 X_container 的列名与 columns 相等
    assert_array_equal(X_container.index, index)  # 断言 X_container 的行索引与 index 相等

    # 输入的 DataFrame X_df 的索引不会改变
    new_columns = np.asarray(["f0", "f1"], dtype=object)
    X_df = pd.DataFrame([[1, 2], [1, 3]], index=[10, 12])
    new_df = adapter.create_container(X_df, X_df_orig, columns=new_columns)
    assert_array_equal(new_df.columns, new_columns)  # 断言 new_df 的列名与 new_columns 相等
    assert_array_equal(new_df.index, X_df.index)  # 断言 new_df 的行索引与 X_df 相等

    assert adapter.is_supported_container(X_df)  # 断言适配器支持 X_df
    assert not adapter.is_supported_container(X_np)  # 断言适配器不支持 X_np

    # adapter.update_columns 更新列名
    new_columns = np.array(["a", "c"], dtype=object)
    new_df = adapter.rename_columns(X_df, new_columns)
    assert_array_equal(new_df.columns, new_columns)  # 断言 new_df 的列名已更新

    # adapter.hstack 水平堆叠两个 DataFrame
    X_df_1 = pd.DataFrame([[1, 2, 5], [3, 4, 6]], columns=["a", "b", "e"])
    X_df_2 = pd.DataFrame([[4], [5]], columns=["c"])
    X_stacked = adapter.hstack([X_df_1, X_df_2])

    # 创建预期的 DataFrame
    expected_df = pd.DataFrame(
        [[1, 2, 5, 4], [3, 4, 6, 5]], columns=["a", "b", "e", "c"]
    )
    pd.testing.assert_frame_equal(X_stacked, expected_df)  # 断言 X_stacked 与 expected_df 相等

    # 检查在重命名列名时处理重复列名的情况
    X_df = pd.DataFrame([[1, 2], [1, 3]], columns=["a", "a"])
    new_columns = np.array(["x__a", "y__a"], dtype=object)
    new_df = adapter.rename_columns(X_df, new_columns)
    assert_array_equal(new_df.columns, new_columns)  # 断言 new_df 的列名已更新

    # 检查 `create_container` 函数中 inplace 参数的行为，应触发复制操作
    X_df = pd.DataFrame([[1, 2], [1, 3]], index=index)
    X_output = adapter.create_container(X_df, X_df, columns=["a", "b"], inplace=False)
    assert X_output is not X_df  # 断言 X_output 不是 X_df 的引用
    assert list(X_df.columns) == [0, 1]  # 断言 X_df 的列名不变
    assert list(X_output.columns) == ["a", "b"]  # 断言 X_output 的列名已更新
    # 创建一个包含指定数据和索引的 Pandas DataFrame 对象
    X_df = pd.DataFrame([[1, 2], [1, 3]], index=index)
    # 调用适配器对象的方法来创建一个容器对象，并将结果赋给 X_output
    X_output = adapter.create_container(X_df, X_df, columns=["a", "b"], inplace=True)
    # 断言 X_output 和 X_df 是同一个对象
    assert X_output is X_df
    # 断言 X_df 的列名为 ["a", "b"]
    assert list(X_df.columns) == ["a", "b"]
    # 断言 X_output 的列名也为 ["a", "b"]，验证 inplace 操作的影响
    assert list(X_output.columns) == ["a", "b"]
# 定义测试函数，验证 Polars 适配器的预期行为
def test_polars_adapter():
    # 导入 pytest，如果不存在则跳过测试
    pl = pytest.importorskip("polars")
    
    # 创建一个 NumPy 数组作为输入数据
    X_np = np.array([[1, 0, 3], [0, 0, 1]])
    
    # 指定数据框的列名
    columns = ["f1", "f2", "f3"]
    
    # 使用 Polars 创建原始数据框 X_df_orig
    X_df_orig = pl.DataFrame(X_np, schema=columns, orient="row")
    
    # 获取 Polars 适配器对象
    adapter = ADAPTERS_MANAGER.adapters["polars"]
    
    # 使用适配器创建容器 X_container
    X_container = adapter.create_container(X_np, X_df_orig, columns=lambda: columns)
    
    # 断言 X_container 是 Polars 的 DataFrame 类型
    assert isinstance(X_container, pl.DataFrame)
    
    # 断言 X_container 的列名与预期的 columns 相同
    assert_array_equal(X_container.columns, columns)
    
    # 使用 create_container 更新列名
    new_columns = np.asarray(["a", "b", "c"], dtype=object)
    new_df = adapter.create_container(X_df_orig, X_df_orig, columns=new_columns)
    
    # 断言新数据框的列名与更新后的 new_columns 相同
    assert_array_equal(new_df.columns, new_columns)
    
    # 断言 adapter 能够识别 X_df_orig 作为支持的容器
    assert adapter.is_supported_container(X_df_orig)
    
    # 断言 adapter 不能识别 X_np 作为支持的容器
    assert not adapter.is_supported_container(X_np)
    
    # 使用 rename_columns 方法更新列名
    new_columns = np.array(["a", "c", "g"], dtype=object)
    new_df = adapter.rename_columns(X_df_orig, new_columns)
    
    # 断言新数据框的列名与更新后的 new_columns 相同
    assert_array_equal(new_df.columns, new_columns)
    
    # 使用 adapter.hstack 方法水平堆叠数据框
    X_df_1 = pl.DataFrame([[1, 2, 5], [3, 4, 6]], schema=["a", "b", "e"], orient="row")
    X_df_2 = pl.DataFrame([[4], [5]], schema=["c"], orient="row")
    X_stacked = adapter.hstack([X_df_1, X_df_2])
    
    # 断言堆叠后的数据框与预期的 expected_df 相同
    expected_df = pl.DataFrame(
        [[1, 2, 5, 4], [3, 4, 6, 5]], schema=["a", "b", "e", "c"], orient="row"
    )
    from polars.testing import assert_frame_equal
    assert_frame_equal(X_stacked, expected_df)
    
    # 检查 create_container 中 inplace 参数的行为，预期应该触发复制
    X_df = pl.DataFrame([[1, 2], [1, 3]], schema=["a", "b"], orient="row")
    X_output = adapter.create_container(X_df, X_df, columns=["c", "d"], inplace=False)
    
    # 断言 X_output 不等于 X_df，即已复制
    assert X_output is not X_df
    assert list(X_df.columns) == ["a", "b"]  # X_df 的列名未改变
    assert list(X_output.columns) == ["c", "d"]  # X_output 的列名已更新
    
    # inplace 参数为 True 时的行为
    X_df = pl.DataFrame([[1, 2], [1, 3]], schema=["a", "b"], orient="row")
    X_output = adapter.create_container(X_df, X_df, columns=["c", "d"], inplace=True)
    
    # 断言 X_output 等于 X_df，即操作是原地进行的
    assert X_output is X_df
    assert list(X_df.columns) == ["c", "d"]  # X_df 的列名已更新
    assert list(X_output.columns) == ["c", "d"]  # X_output 的列名也已更新


# 使用 pytest 的参数化功能来定义测试函数，测试 CSR_CONTAINERS 中的容器
@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test__container_error_validation(csr_container):
    # 创建一个示例输入数据 X
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    
    # 使用 csr_container 创建稀疏矩阵 X_csr
    X_csr = csr_container(X)
    
    # 定义匹配错误消息
    match = "The transformer outputs a scipy sparse matrix."
    
    # 在配置上下文中，设置 transform_output 为 "pandas"，并断言引发 ValueError 异常
    with config_context(transform_output="pandas"):
        with pytest.raises(ValueError, match=match):
            _wrap_data_with_container("transform", X_csr, X, StandardScaler())


# 定义一个没有 set_output 方法和 transform 方法的估计器类
class EstimatorWithoutSetOutputAndWithoutTransform:
    pass


# 定义一个没有 set_output 方法但有 transform 方法的估计器类
class EstimatorNoSetOutputWithTransform:
    def transform(self, X, y=None):
        return X  # pragma: no cover


# 定义一个带有 set_output 方法的估计器类，继承自 _SetOutputMixin
class EstimatorWithSetOutput(_SetOutputMixin):
    pass
    # 定义模型适配方法，用于拟合数据
    def fit(self, X, y=None):
        # 设置对象属性，记录输入数据的特征数目
        self.n_features_in_ = X.shape[1]
        # 返回对象本身，通常用于方法链式调用或者兼容 scikit-learn 接口
        return self

    # 定义数据转换方法，直接返回输入的特征数据 X
    def transform(self, X, y=None):
        return X

    # 定义获取输出特征名称方法
    def get_feature_names_out(self, input_features=None):
        # 创建一个包含特征名称的 numpy 数组，命名规则为 "X{i}"，其中 i 从 0 到 self.n_features_in_-1
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)
def test__safe_set_output():
    """Check _safe_set_output works as expected."""

    # 创建一个没有 transform 方法的估算器实例，设置输出为 "pandas" 不会引发异常
    est = EstimatorWithoutSetOutputAndWithoutTransform()
    _safe_set_output(est, transform="pandas")

    # 创建一个带有 transform 方法但没有 set_output 方法的估算器实例，设置输出为 "pandas" 会引发 ValueError 异常
    est = EstimatorNoSetOutputWithTransform()
    with pytest.raises(ValueError, match="Unable to configure output"):
        _safe_set_output(est, transform="pandas")

    # 创建一个带有 set_output 方法的估算器实例，并进行拟合
    est = EstimatorWithSetOutput().fit(np.asarray([[1, 2, 3]]))
    # 设置输出为 "pandas" 后，获取 transform 的输出配置信息，并断言其为 "pandas"
    _safe_set_output(est, transform="pandas")
    config = _get_output_config("transform", est)
    assert config["dense"] == "pandas"

    # 再次设置输出为 "default"，并验证配置为 "default"
    _safe_set_output(est, transform="default")
    config = _get_output_config("transform", est)
    assert config["dense"] == "default"

    # 设置 transform 为 None，这是一个无操作，配置应保持为 "default"
    _safe_set_output(est, transform=None)
    config = _get_output_config("transform", est)
    assert config["dense"] == "default"


class EstimatorNoSetOutputWithTransformNoFeatureNamesOut(_SetOutputMixin):
    def transform(self, X, y=None):
        return X  # pragma: no cover


def test_set_output_mixin():
    """Estimator without get_feature_names_out does not define `set_output`."""
    # 创建一个没有 get_feature_names_out 方法的估算器实例，并验证其没有定义 set_output 方法
    est = EstimatorNoSetOutputWithTransformNoFeatureNamesOut()
    assert not hasattr(est, "set_output")


def test__safe_set_output_error():
    """Check transform with invalid config."""
    X = np.asarray([[1, 0, 3], [0, 0, 1]])

    # 创建一个带有 set_output 方法的估算器实例
    est = EstimatorWithSetOutput()
    # 使用无效的 transform 参数调用 _safe_set_output，预期引发 ValueError 异常
    _safe_set_output(est, transform="bad")

    # 使用 pytest 的断言检查是否引发了包含特定消息的 ValueError 异常
    msg = "output config must be in"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_set_output_method(dataframe_lib):
    """Check that the output is a dataframe."""
    lib = pytest.importorskip(dataframe_lib)

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)

    # 设置 transform=None，这是一个无操作
    est2 = est.set_output(transform=None)
    assert est2 is est
    X_trans_np = est2.transform(X)
    assert isinstance(X_trans_np, np.ndarray)

    # 设置 transform 为 dataframe_lib，验证输出是否为对应的 DataFrame 类型
    est.set_output(transform=dataframe_lib)
    X_trans_pd = est.transform(X)
    assert isinstance(X_trans_pd, lib.DataFrame)


def test_set_output_method_error():
    """Check transform fails with invalid transform."""

    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    est = EstimatorWithSetOutput().fit(X)

    # 使用无效的 transform 参数调用 set_output 方法，预期引发 ValueError 异常
    est.set_output(transform="bad")
    msg = "output config must be in"
    with pytest.raises(ValueError, match=msg):
        est.transform(X)


@pytest.mark.parametrize("transform_output", ["pandas", "polars"])
def test__get_output_config(transform_output):
    """Check _get_output_config works as expected."""

    # 没有设置特定的配置时，应使用全局配置
    global_config = get_config()["transform_output"]
    config = _get_output_config("transform")
    # 断言检查当前配置中 "dense" 键对应的值是否与全局配置中的相同
    assert config["dense"] == global_config

    # 进入上下文管理器，设置 transform_output 参数为当前上下文中的值
    with config_context(transform_output=transform_output):
        # 获取 "transform" 转换后的输出配置
        config = _get_output_config("transform")
        # 断言检查当前配置中 "dense" 键对应的值是否与 transform_output 相同
        assert config["dense"] == transform_output

        # 创建一个没有设置输出的 EstimatorNoSetOutputWithTransform 实例
        est = EstimatorNoSetOutputWithTransform()
        # 获取 "transform" 转换后的输出配置，使用特定的估算器 est
        config = _get_output_config("transform", est)
        # 断言检查当前配置中 "dense" 键对应的值是否与 transform_output 相同
        assert config["dense"] == transform_output

        # 创建一个设置了输出的 EstimatorWithSetOutput 实例
        est = EstimatorWithSetOutput()
        # 如果估算器具有配置，则使用本地配置，获取 "transform" 转换后的输出配置
        config = _get_output_config("transform", est)
        # 断言检查当前配置中 "dense" 键对应的值是否为 "default"
        assert config["dense"] == "default"

        # 设置 EstimatorWithSetOutput 实例的输出转换为 transform_output
        est.set_output(transform="default")
        # 获取 "transform" 转换后的输出配置，使用特定的估算器 est
        config = _get_output_config("transform", est)
        # 断言检查当前配置中 "dense" 键对应的值是否与 transform_output 相同
        assert config["dense"] == transform_output

    # 设置 EstimatorWithSetOutput 实例的输出转换为 transform_output
    est.set_output(transform=transform_output)
    # 获取 "transform" 转换后的输出配置，使用特定的估算器 est
    config = _get_output_config("transform", est)
    # 断言检查当前配置中 "dense" 键对应的值是否与 transform_output 相同
    assert config["dense"] == transform_output
# 定义一个继承自 _SetOutputMixin 的类 EstimatorWithSetOutputNoAutoWrap，未指定 auto_wrap_output_keys 参数
class EstimatorWithSetOutputNoAutoWrap(_SetOutputMixin, auto_wrap_output_keys=None):
    # 实现 transform 方法，直接返回输入 X，不做任何处理
    def transform(self, X, y=None):
        return X


# 定义测试函数 test_get_output_auto_wrap_false
def test_get_output_auto_wrap_false():
    """Check that auto_wrap_output_keys=None does not wrap."""
    # 创建 EstimatorWithSetOutputNoAutoWrap 类的实例 est
    est = EstimatorWithSetOutputNoAutoWrap()
    # 断言 est 没有属性 "set_output"
    assert not hasattr(est, "set_output")

    # 创建一个 NumPy 数组 X
    X = np.asarray([[1, 0, 3], [0, 0, 1]])
    # 断言 X 和 est.transform(X) 引用相同的对象
    assert X is est.transform(X)


# 定义测试函数 test_auto_wrap_output_keys_errors_with_incorrect_input
def test_auto_wrap_output_keys_errors_with_incorrect_input():
    # 错误消息
    msg = "auto_wrap_output_keys must be None or a tuple of keys."
    # 使用 pytest 的 raises 断言捕获 ValueError 异常，并验证错误消息是否匹配
    with pytest.raises(ValueError, match=msg):
        # 定义一个错误的 Estimator 类 BadEstimator，auto_wrap_output_keys 参数应为 None 或键的元组
        class BadEstimator(_SetOutputMixin, auto_wrap_output_keys="bad_parameter"):
            pass


# 定义类 AnotherMixin
class AnotherMixin:
    # 定义 __init_subclass__ 方法，传入 custom_parameter 参数
    def __init_subclass__(cls, custom_parameter, **kwargs):
        super().__init_subclass__(**kwargs)
        # 设置类属性 custom_parameter
        cls.custom_parameter = custom_parameter


# 定义测试函数 test_set_output_mixin_custom_mixin
def test_set_output_mixin_custom_mixin():
    """Check that multiple init_subclasses passes parameters up."""
    # 定义一个混合使用 _SetOutputMixin 和 AnotherMixin 的 Estimator 类 BothMixinEstimator，设置 custom_parameter=123
    class BothMixinEstimator(_SetOutputMixin, AnotherMixin, custom_parameter=123):
        # 实现 transform 方法，直接返回输入 X，不做任何处理
        def transform(self, X, y=None):
            return X

        # 实现 get_feature_names_out 方法，返回输入特征的名称列表
        def get_feature_names_out(self, input_features=None):
            return input_features

    # 创建 BothMixinEstimator 的实例 est
    est = BothMixinEstimator()
    # 断言 est.custom_parameter 的值为 123
    assert est.custom_parameter == 123
    # 断言 est 有属性 "set_output"
    assert hasattr(est, "set_output")


# 定义测试函数 test_set_output_mro
def test_set_output_mro():
    """Check that multi-inheritance resolves to the correct class method.

    Non-regression test gh-25293.
    """
    # 定义基类 Base，继承自 _SetOutputMixin，实现 transform 方法返回字符串 "Base"
    class Base(_SetOutputMixin):
        def transform(self, X):
            return "Base"  # noqa

    # 定义类 A，继承自 Base，未定义 transform 方法，继承 Base 的方法
    class A(Base):
        pass

    # 定义类 B，继承自 Base，重写 transform 方法返回字符串 "B"
    class B(Base):
        def transform(self, X):
            return "B"

    # 定义类 C，继承自 A 和 B，按照 C3 线性化顺序解析得到的 transform 方法应为 B 类的实现
    class C(A, B):
        pass

    # 断言调用 C 的 transform 方法返回 "B"
    assert C().transform(None) == "B"


# 定义类 EstimatorWithSetOutputIndex，继承自 _SetOutputMixin
class EstimatorWithSetOutputIndex(_SetOutputMixin):
    # 实现 fit 方法，设置输入 X 的特征数为 n_features_in_
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    # 实现 transform 方法，将输入 X 转换为 Pandas DataFrame，并设置新的索引
    def transform(self, X, y=None):
        import pandas as pd
        return pd.DataFrame(X.to_numpy(), index=[f"s{i}" for i in range(X.shape[0])])

    # 实现 get_feature_names_out 方法，返回特征名称的数组
    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


# 定义测试函数 test_set_output_pandas_keep_index
def test_set_output_pandas_keep_index():
    """Check that set_output does not override index.

    Non-regression test for gh-25730.
    """
    # 导入 pytest，如果导入失败则跳过该测试
    pd = pytest.importorskip("pandas")

    # 创建一个带有索引的 Pandas DataFrame X
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], index=[0, 1])
    # 创建 EstimatorWithSetOutputIndex 的实例 est，并调用 set_output 方法设置输出为 "pandas"
    est = EstimatorWithSetOutputIndex().set_output(transform="pandas")
    # 对 est 进行拟合
    est.fit(X)

    # 对 X 进行转换得到 X_trans
    X_trans = est.transform(X)
    # 断言 X_trans 的索引数组与预期的 ["s0", "s1"] 相等
    assert_array_equal(X_trans.index, ["s0", "s1"])


# 定义类 EstimatorReturnTuple，继承自 _SetOutputMixin
class EstimatorReturnTuple(_SetOutputMixin):
    # 初始化方法，接受一个名为 OutputTuple 的参数
    def __init__(self, OutputTuple):
        self.OutputTuple = OutputTuple

    # 实现 transform 方法，返回一个元组 OutputTuple(X, 2*X)
    def transform(self, X, y=None):
        return self.OutputTuple(X, 2 * X)


# 定义测试函数 test_set_output_named_tuple_out
def test_set_output_named_tuple_out():
    """Check that namedtuples are kept by default."""
    # 导入 namedtuple 方法
    Output = namedtuple("Output", "X, Y")
    # 创建一个包含单个样本向量的 NumPy 数组 X
    X = np.asarray([[1, 2, 3]])
    
    # 使用 EstimatorReturnTuple 类的实例化对象 est，并传入 OutputTuple=Output 作为参数
    est = EstimatorReturnTuple(OutputTuple=Output)
    
    # 调用 est 对象的 transform 方法，将输入 X 进行转换得到 X_trans
    X_trans = est.transform(X)
    
    # 断言 X_trans 是 Output 类型的实例
    assert isinstance(X_trans, Output)
    
    # 断言 X_trans 的 X 属性与输入 X 的内容相等
    assert_array_equal(X_trans.X, X)
    
    # 断言 X_trans 的 Y 属性与输入 X 的两倍相等
    assert_array_equal(X_trans.Y, 2 * X)
class EstimatorWithListInput(_SetOutputMixin):
    # 定义一个估计器类，支持列表类型的输入数据
    def fit(self, X, y=None):
        # 确保输入数据 X 是列表类型
        assert isinstance(X, list)
        # 记录输入数据 X 中的特征数量
        self.n_features_in_ = len(X[0])
        # 返回自身，用于方法链
        return self

    def transform(self, X, y=None):
        # 无需进行任何转换，直接返回输入数据 X
        return X

    def get_feature_names_out(self, input_features=None):
        # 返回一个包含输出特征名称的 NumPy 数组，以 "X{i}" 命名，其中 i 是特征的索引
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def test_set_output_list_input(dataframe_lib):
    """Check set_output for list input.

    Non-regression test for #27037.
    """
    # 导入指定的数据框架库，如果库不可用则跳过测试
    lib = pytest.importorskip(dataframe_lib)

    # 创建一个示例输入数据 X
    X = [[0, 1, 2, 3], [4, 5, 6, 7]]
    # 创建一个 EstimatorWithListInput 类的实例
    est = EstimatorWithListInput()
    # 设置输出的数据容器为指定的数据框架库
    est.set_output(transform=dataframe_lib)

    # 对输入数据 X 进行拟合并进行转换
    X_out = est.fit(X).transform(X)
    # 断言转换后的输出 X_out 是指定数据框架库的 DataFrame 类型
    assert isinstance(X_out, lib.DataFrame)
    # 断言转换后的 DataFrame 的列名符合预期
    assert_array_equal(X_out.columns, ["X0", "X1", "X2", "X3"])


@pytest.mark.parametrize("name", sorted(ADAPTERS_MANAGER.adapters))
def test_adapter_class_has_interface(name):
    """Check adapters have the correct interface."""
    # 断言特定适配器具有正确的接口
    assert isinstance(ADAPTERS_MANAGER.adapters[name], ContainerAdapterProtocol)


def test_check_library_installed(monkeypatch):
    """Check import error changed."""
    # 备份原始的 import_module 函数
    orig_import_module = importlib.import_module

    # 定义一个修补后的 import_module 函数，当导入 pandas 时抛出 ImportError
    def patched_import_module(name):
        if name == "pandas":
            raise ImportError()
        orig_import_module(name, package=None)

    # 使用 monkeypatch 修改 importlib 模块的 import_module 函数
    monkeypatch.setattr(importlib, "import_module", patched_import_module)

    # 执行测试，验证导入 pandas 库时抛出 ImportError
    msg = "Setting output container to 'pandas' requires"
    with pytest.raises(ImportError, match=msg):
        check_library_installed("pandas")


def test_get_adapter_from_container():
    """Check the behavior fo `_get_adapter_from_container`."""
    # 导入 pandas 库，如果 pandas 不可用则跳过测试
    pd = pytest.importorskip("pandas")
    # 创建一个 DataFrame X 作为输入
    X = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 100]})
    # 获取 X 的适配器
    adapter = _get_adapter_from_container(X)
    # 断言适配器的容器库是 pandas
    assert adapter.container_lib == "pandas"
    # 准备一个错误信息，用于断言当输入是 NumPy 数组时会抛出 ValueError
    err_msg = "The container does not have a registered adapter in scikit-learn."
    # 断言当输入是 NumPy 数组时会抛出 ValueError
    with pytest.raises(ValueError, match=err_msg):
        _get_adapter_from_container(X.to_numpy())
```