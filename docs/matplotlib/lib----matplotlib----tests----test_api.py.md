# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_api.py`

```py
# 引入 Python 未来的特性，使得注解支持类型的自引用
from __future__ import annotations

# 导入正则表达式模块和类型提示相关模块
import re
import typing
from typing import Any, Callable, TypeVar

# 导入 NumPy 库并重命名为 np
import numpy as np
# 导入 pytest 库
import pytest

# 导入 matplotlib 库并部分导入 _api 模块
import matplotlib as mpl
from matplotlib import _api

# 如果支持类型检查，则导入 Self 类型
if typing.TYPE_CHECKING:
    from typing_extensions import Self

# 定义类型变量 T
T = TypeVar('T')

# 使用 pytest 的 parametrize 装饰器定义参数化测试
@pytest.mark.parametrize('target,shape_repr,test_shape',
                         [((None, ), "(N,)", (1, 3)),
                          ((None, 3), "(N, 3)", (1,)),
                          ((None, 3), "(N, 3)", (1, 2)),
                          ((1, 5), "(1, 5)", (1, 9)),
                          ((None, 2, None), "(M, 2, N)", (1, 3, 1))
                          ])
# 定义测试函数 test_check_shape，参数为 target、shape_repr 和 test_shape，返回 None
def test_check_shape(target: tuple[int | None, ...],
                     shape_repr: str,
                     test_shape: tuple[int, ...]) -> None:
    # 构造错误模式正则表达式，用于匹配 ValueError 异常信息
    error_pattern = "^" + re.escape(
        f"'aardvark' must be {len(target)}D with shape {shape_repr}, but your input "
        f"has shape {test_shape}")
    # 创建指定形状的全零 NumPy 数组作为测试数据
    data = np.zeros(test_shape)
    # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配错误模式
    with pytest.raises(ValueError, match=error_pattern):
        _api.check_shape(target, aardvark=data)

# 定义测试函数 test_classproperty_deprecation，验证类属性装饰器的废弃警告
def test_classproperty_deprecation() -> None:
    # 定义类 A
    class A:
        # 废弃警告装饰器，Matplotlib 版本 0.0.0 起废弃
        @_api.deprecated("0.0.0")
        # 类属性装饰器
        @_api.classproperty
        # 类方法 f，接收类本身 cls 作为参数，无返回值
        def f(cls: Self) -> None:
            pass
    # 使用 pytest 的 warns 断言捕获 MatplotlibDeprecationWarning 警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        A.f
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 创建类 A 的实例 a，并访问其属性 f
        a = A()
        a.f

# 定义测试函数 test_deprecate_privatize_attribute，验证私有属性的废弃警告
def test_deprecate_privatize_attribute() -> None:
    # 定义类 C
    class C:
        # 构造方法，初始化私有属性 _attr 为 1
        def __init__(self) -> None: self._attr = 1
        # 私有方法 _meth，接收泛型参数 arg，返回 arg
        def _meth(self, arg: T) -> T: return arg
        # 属性 attr，使用废弃警告装饰器，Matplotlib 版本 0.0 起废弃
        attr: int = _api.deprecate_privatize_attribute("0.0")
        # 方法 meth，使用废弃警告装饰器，Matplotlib 版本 0.0 起废弃
        meth: Callable = _api.deprecate_privatize_attribute("0.0")

    # 创建类 C 的实例 c
    c = C()
    # 使用 pytest 的 warns 断言捕获 MatplotlibDeprecationWarning 警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert c.attr == 1
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        c.attr = 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert c.attr == 2
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        assert c.meth(42) == 42

# 定义测试函数 test_delete_parameter，验证删除函数参数的废弃警告
def test_delete_parameter() -> None:
    # 使用废弃警告装饰器，Matplotlib 版本 3.0 起废弃函数参数 foo
    @_api.delete_parameter("3.0", "foo")
    def func1(foo: Any = None) -> None:
        pass

    # 使用废弃警告装饰器，Matplotlib 版本 3.0 起废弃函数参数 foo
    @_api.delete_parameter("3.0", "foo")
    def func2(**kwargs: Any) -> None:
        pass

    # 遍历函数列表 [func1, func2]，执行函数并捕获 MatplotlibDeprecationWarning 警告
    for func in [func1, func2]:  # type: ignore[list-item]
        func()  # 没有警告
        with pytest.warns(mpl.MatplotlibDeprecationWarning):
            func(foo="bar")

    # 定义 pyplot_wrapper 函数，参数为 foo，默认为废弃参数
    def pyplot_wrapper(foo: Any = _api.deprecation._deprecated_parameter) -> None:
        func1(foo)

    pyplot_wrapper()  # 没有警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        func(foo="bar")

# 定义测试函数 test_make_keyword_only，验证函数参数强制成关键字参数的废弃警告
def test_make_keyword_only() -> None:
    # 使用废弃警告装饰器，Matplotlib 版本 3.0 起将参数 arg 强制成关键字参数
    @_api.make_keyword_only("3.0", "arg")
    def func(pre: Any, arg: Any, post: Any = None) -> None:
        pass

    # 调用函数 func，传入 pre 和 arg 参数，检查是否没有警告
    func(1, arg=2)  # Check that no warning is emitted.

    # 使用 pytest 的 warns 断言捕获 MatplotlibDeprecationWarning 警告
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        func(1, 2)
    # 使用 pytest 模块来检测是否发出特定类型的警告信息
    with pytest.warns(mpl.MatplotlibDeprecationWarning):
        # 调用函数 func，并传入参数 1, 2, 3
        func(1, 2, 3)
# 定义一个测试函数，用于测试API的废弃功能提示和替代方式
def test_deprecation_alternative() -> None:
    # 定义一个备选方案的字符串，包含可能的替代方式
    alternative = "`.f1`, `f2`, `f3(x) <.f3>` or `f4(x)<f4>`"
    
    # 使用装饰器标记函数为废弃，并提供替代方式的说明
    @_api.deprecated("1", alternative=alternative)
    def f() -> None:
        pass
    
    # 如果函数没有文档字符串，则跳过测试，因为文档功能已禁用
    if f.__doc__ is None:
        pytest.skip('Documentation is disabled')
    
    # 断言替代方式的描述信息在函数的文档字符串中出现
    assert alternative in f.__doc__


# 定义一个测试函数，用于测试在列表操作中的空参数检查
def test_empty_check_in_list() -> None:
    # 使用pytest的上下文管理器检查是否会抛出TypeError，并匹配特定的错误消息
    with pytest.raises(TypeError, match="No argument to check!"):
        # 调用API的函数，传入一个空列表作为参数，预期会抛出TypeError异常
        _api.check_in_list(["a"])
```