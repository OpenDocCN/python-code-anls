# `D:\src\scipysrc\scikit-learn\sklearn\utils\tests\test_deprecation.py`

```
# 导入 pickle 模块，用于对象序列化和反序列化
import pickle

# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 scikit-learn 库中导入 _is_deprecated 和 deprecated 函数
from sklearn.utils.deprecation import _is_deprecated, deprecated


# 使用 @deprecated 装饰器标记 MockClass1 类为已弃用，提供警告信息 "qwerty"
@deprecated("qwerty")
class MockClass1:
    pass


class MockClass2:
    # 使用 @deprecated 装饰器标记 method 方法为已弃用，提供警告信息 "mockclass2_method"
    @deprecated("mockclass2_method")
    def method(self):
        pass

    # 使用 @deprecated 装饰器标记 n_features_ 属性为已弃用，提供警告信息 "n_features_ is deprecated"
    @deprecated("n_features_ is deprecated")  # type: ignore
    @property
    def n_features_(self):
        """Number of input features."""
        return 10


class MockClass3:
    # 使用 @deprecated 装饰器标记 __init__ 方法为已弃用，没有提供特定的警告信息
    @deprecated()
    def __init__(self):
        pass


class MockClass4:
    pass


# MockClass5 继承自 MockClass1，但未调用其父类的 __init__ 方法
class MockClass5(MockClass1):
    """Inherit from deprecated class but does not call super().__init__."""

    def __init__(self, a):
        self.a = a


# 使用 @deprecated 装饰器标记 MockClass6 类为已弃用，提供警告信息 "a message"
@deprecated("a message")
class MockClass6:
    """A deprecated class that overrides __new__."""

    # 覆盖 __new__ 方法，确保至少接收一个参数
    def __new__(cls, *args, **kwargs):
        assert len(args) > 0
        return super().__new__(cls)


# 使用 @deprecated 装饰器标记 mock_function 函数为已弃用，没有提供特定的警告信息
@deprecated()
def mock_function():
    return 10


# 定义测试函数 test_deprecated，用于测试已弃用的类和函数的行为
def test_deprecated():
    # 检查是否在创建 MockClass1 实例时收到 FutureWarning，警告信息包含 "qwerty"
    with pytest.warns(FutureWarning, match="qwerty"):
        MockClass1()
    
    # 检查是否在调用 MockClass2 的 method 方法时收到 FutureWarning，警告信息包含 "mockclass2_method"
    with pytest.warns(FutureWarning, match="mockclass2_method"):
        MockClass2().method()
    
    # 检查是否在创建 MockClass3 实例时收到 FutureWarning，警告信息包含 "deprecated"
    with pytest.warns(FutureWarning, match="deprecated"):
        MockClass3()
    
    # 检查是否在创建 MockClass5 实例时收到 FutureWarning，警告信息包含 "qwerty"
    with pytest.warns(FutureWarning, match="qwerty"):
        MockClass5(42)
    
    # 检查是否在创建 MockClass6 实例时收到 FutureWarning，警告信息包含 "a message"
    with pytest.warns(FutureWarning, match="a message"):
        MockClass6(42)
    
    # 检查是否在调用 mock_function 函数时收到 FutureWarning，警告信息包含 "deprecated"
    with pytest.warns(FutureWarning, match="deprecated"):
        val = mock_function()
    
    # 断言 mock_function 返回值为 10
    assert val == 10


# 定义测试函数 test_is_deprecated，用于测试 _is_deprecated 函数的行为
def test_is_deprecated():
    # 测试 _is_deprecated 辅助函数是否能够识别通过 @deprecated 装饰器标记为已弃用的方法和函数
    assert _is_deprecated(MockClass1.__new__)
    assert _is_deprecated(MockClass2().method)
    assert _is_deprecated(MockClass3.__init__)
    assert not _is_deprecated(MockClass4.__init__)  # MockClass4 未被标记为已弃用
    assert _is_deprecated(MockClass5.__new__)
    assert _is_deprecated(mock_function)


# 定义测试函数 test_pickle，用于测试 pickle 序列化和反序列化 mock_function 函数的行为
def test_pickle():
    pickle.loads(pickle.dumps(mock_function))
```