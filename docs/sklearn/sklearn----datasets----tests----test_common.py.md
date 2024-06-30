# `D:\src\scipysrc\scikit-learn\sklearn\datasets\tests\test_common.py`

```
"""Test loaders for common functionality."""

# 导入必要的模块
import inspect  # 用于检查对象的属性和方法
import os  # 提供了与操作系统相关的功能

import numpy as np  # 数值计算库
import pytest  # 测试框架

import sklearn.datasets  # sklearn 的数据集模块


def is_pillow_installed():
    try:
        import PIL  # 检查是否安装了 PIL 库

        return True
    except ImportError:
        return False


# 定义了一些在使用 fetch 函数时的标记和条件
FETCH_PYTEST_MARKERS = {
    "return_X_y": {
        "fetch_20newsgroups": pytest.mark.xfail(
            reason="X is a list and does not have a shape argument"
        ),
        "fetch_openml": pytest.mark.xfail(
            reason="fetch_opeml requires a dataset name or id"
        ),
        "fetch_lfw_people": pytest.mark.skipif(
            not is_pillow_installed(), reason="pillow is not installed"
        ),
    },
    "as_frame": {
        "fetch_openml": pytest.mark.xfail(
            reason="fetch_opeml requires a dataset name or id"
        ),
    },
}


def check_pandas_dependency_message(fetch_func):
    try:
        import pandas  # 尝试导入 pandas 库

        pytest.skip("This test requires pandas to not be installed")
    except ImportError:
        # 如果导入失败，则验证预期的错误消息
        name = fetch_func.__name__
        expected_msg = f"{name} with as_frame=True requires pandas"
        with pytest.raises(ImportError, match=expected_msg):
            fetch_func(as_frame=True)


def check_return_X_y(bunch, dataset_func):
    # 检查返回的数据格式是否正确
    X_y_tuple = dataset_func(return_X_y=True)
    assert isinstance(X_y_tuple, tuple)
    assert X_y_tuple[0].shape == bunch.data.shape
    assert X_y_tuple[1].shape == bunch.target.shape


def check_as_frame(
    bunch, dataset_func, expected_data_dtype=None, expected_target_dtype=None
):
    pd = pytest.importorskip("pandas")  # 导入 pandas 库，如果不存在则跳过测试
    frame_bunch = dataset_func(as_frame=True)
    # 验证返回的对象是否是 DataFrame
    assert hasattr(frame_bunch, "frame")
    assert isinstance(frame_bunch.frame, pd.DataFrame)
    assert isinstance(frame_bunch.data, pd.DataFrame)
    assert frame_bunch.data.shape == bunch.data.shape
    if frame_bunch.target.ndim > 1:
        assert isinstance(frame_bunch.target, pd.DataFrame)
    else:
        assert isinstance(frame_bunch.target, pd.Series)
    assert frame_bunch.target.shape[0] == bunch.target.shape[0]
    if expected_data_dtype is not None:
        assert np.all(frame_bunch.data.dtypes == expected_data_dtype)
    if expected_target_dtype is not None:
        assert np.all(frame_bunch.target.dtypes == expected_target_dtype)

    # 测试返回值为 DataFrame 时的另一种情况
    frame_X, frame_y = dataset_func(as_frame=True, return_X_y=True)
    assert isinstance(frame_X, pd.DataFrame)
    if frame_y.ndim > 1:
        assert isinstance(frame_X, pd.DataFrame)
    else:
        assert isinstance(frame_y, pd.Series)


def _skip_network_tests():
    return os.environ.get("SKLEARN_SKIP_NETWORK_TESTS", "1") == "1"


def _generate_func_supporting_param(param, dataset_type=("load", "fetch")):
    markers_fetch = FETCH_PYTEST_MARKERS.get(param, {})
    # 使用 inspect 模块获取 sklearn.datasets 中的所有成员名和对象
    for name, obj in inspect.getmembers(sklearn.datasets):
        # 如果对象不是函数，则继续下一个成员
        if not inspect.isfunction(obj):
            continue
        
        # 检查函数名是否以 dataset_type 中任一类型开头
        is_dataset_type = any([name.startswith(t) for t in dataset_type])
        
        # 检查函数的参数签名中是否包含指定的参数 param
        is_support_param = param in inspect.signature(obj).parameters
        
        # 如果函数是所需的数据集类型且支持指定的参数
        if is_dataset_type and is_support_param:
            # 如果函数名以 "fetch" 开头且 _skip_network_tests() 返回 True，
            # 则使用 pytest.mark.skipif 标记为跳过执行，
            # 原因是 fetch 类函数需要网络支持
            marks = [
                pytest.mark.skipif(
                    condition=name.startswith("fetch") and _skip_network_tests(),
                    reason="Skip because fetcher requires internet network",
                )
            ]
            
            # 如果函数名在 markers_fetch 字典中，则添加相应的标记
            if name in markers_fetch:
                marks.append(markers_fetch[name])
            
            # 使用 pytest.param 创建一个参数化的测试项，并通过 yield 返回
            yield pytest.param(name, obj, marks=marks)
# 使用 pytest 的 parametrize 装饰器，为测试函数提供参数化的支持，参数包括函数名和数据集函数
@pytest.mark.parametrize(
    "name, dataset_func", _generate_func_supporting_param("return_X_y")
)
# 定义测试函数，用于测试数据集函数返回值是否符合 check_return_X_y 函数的要求
def test_common_check_return_X_y(name, dataset_func):
    # 调用数据集函数，获取数据集对象 bunch
    bunch = dataset_func()
    # 调用 check_return_X_y 函数，检查数据集对象 bunch 的返回值是否符合预期
    check_return_X_y(bunch, dataset_func)


# 使用 pytest 的 parametrize 装饰器，为测试函数提供参数化的支持，参数包括函数名和数据集函数
@pytest.mark.parametrize(
    "name, dataset_func", _generate_func_supporting_param("as_frame")
)
# 定义测试函数，用于测试数据集函数返回值是否符合 check_as_frame 函数的要求
def test_common_check_as_frame(name, dataset_func):
    # 调用数据集函数，获取数据集对象 bunch
    bunch = dataset_func()
    # 调用 check_as_frame 函数，检查数据集对象 bunch 的返回值是否符合预期
    check_as_frame(bunch, dataset_func)


# 使用 pytest 的 parametrize 装饰器，为测试函数提供参数化的支持，参数包括函数名和数据集函数
@pytest.mark.parametrize(
    "name, dataset_func", _generate_func_supporting_param("as_frame")
)
# 定义测试函数，用于测试数据集函数是否正确处理 pandas 依赖消息
def test_common_check_pandas_dependency(name, dataset_func):
    # 调用 check_pandas_dependency_message 函数，验证数据集函数是否正确处理 pandas 依赖消息
    check_pandas_dependency_message(dataset_func)
```