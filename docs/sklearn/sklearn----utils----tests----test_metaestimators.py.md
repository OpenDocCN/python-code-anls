# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_metaestimators.py`

```
import pickle  # 导入pickle模块，用于对象的序列化和反序列化操作

import pytest  # 导入pytest模块，用于编写和运行测试用例

from sklearn.utils.metaestimators import available_if  # 从sklearn.utils.metaestimators模块导入available_if装饰器


class AvailableParameterEstimator:
    """This estimator's `available` parameter toggles the presence of a method"""

    def __init__(self, available=True, return_value=1):
        self.available = available  # 初始化可用性标志，默认为True
        self.return_value = return_value  # 初始化返回值，默认为1

    @available_if(lambda est: est.available)
    def available_func(self):
        """This is a mock available_if function"""
        return self.return_value  # 如果self.available为True，则返回self.return_value


def test_available_if_docstring():
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.__dict__["available_func"].__doc__
    )  # 断言验证类字典中的available_func函数文档字符串包含指定内容
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator.available_func.__doc__
    )  # 断言验证类实例中的available_func函数文档字符串包含指定内容
    assert "This is a mock available_if function" in str(
        AvailableParameterEstimator().available_func.__doc__
    )  # 断言验证类实例化对象中的available_func函数文档字符串包含指定内容


def test_available_if():
    assert hasattr(AvailableParameterEstimator(), "available_func")  # 断言验证类实例中存在available_func方法
    assert not hasattr(AvailableParameterEstimator(available=False), "available_func")  # 断言验证类实例中不存在available_func方法


def test_available_if_unbound_method():
    # This is a non regression test for:
    # https://github.com/scikit-learn/scikit-learn/issues/20614
    # to make sure that decorated functions can be used as an unbound method,
    # for instance when monkeypatching.
    est = AvailableParameterEstimator()
    AvailableParameterEstimator.available_func(est)  # 调用未绑定的available_func方法

    est = AvailableParameterEstimator(available=False)
    with pytest.raises(
        AttributeError,
        match="This 'AvailableParameterEstimator' has no attribute 'available_func'",
    ):
        AvailableParameterEstimator.available_func(est)  # 验证在不可用状态下调用available_func方法抛出AttributeError异常


def test_available_if_methods_can_be_pickled():
    """Check that available_if methods can be pickled.

    Non-regression test for #21344.
    """
    return_value = 10  # 设置返回值为10
    est = AvailableParameterEstimator(available=True, return_value=return_value)  # 实例化AvailableParameterEstimator对象
    pickled_bytes = pickle.dumps(est.available_func)  # 序列化available_func方法
    unpickled_func = pickle.loads(pickled_bytes)  # 反序列化获取可调用函数
    assert unpickled_func() == return_value  # 断言验证反序列化后调用返回值与预期相等
```