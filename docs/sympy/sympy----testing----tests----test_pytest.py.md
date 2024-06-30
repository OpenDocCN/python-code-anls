# `D:\src\scipysrc\sympy\sympy\testing\tests\test_pytest.py`

```
# 引入警告模块
import warnings

# 从 sympy.testing.pytest 模块中导入 raises, warns, ignore_warnings, warns_deprecated_sympy, Failed 类
from sympy.testing.pytest import (raises, warns, ignore_warnings,
                                    warns_deprecated_sympy, Failed)
# 从 sympy.utilities.exceptions 模块中导入 sympy_deprecation_warning 异常

# 测试可调用对象

# 测试预期异常不会引发任何错误的可调用函数
def test_expected_exception_is_silent_callable():
    # 定义一个函数 f，抛出 ValueError 异常
    def f():
        raise ValueError()
    # 使用 raises 断言 ValueError 异常被引发
    raises(ValueError, f)


# 在 pytest 中，如果没有引发异常，raises 会引发 Failed 而不是 AssertionError
def test_lack_of_exception_triggers_AssertionError_callable():
    try:
        # 使用 lambda 表达式测试 lambda: 1 + 1 是否引发 Exception 异常
        raises(Exception, lambda: 1 + 1)
        # 如果没有引发异常，则断言失败
        assert False
    except Failed as e:
        # 检查错误消息中是否包含 "DID NOT RAISE"
        assert "DID NOT RAISE" in str(e)


# 测试意外异常会被传递的可调用函数
def test_unexpected_exception_is_passed_through_callable():
    # 定义一个函数 f，抛出带有特定错误消息的 ValueError 异常
    def f():
        raise ValueError("some error message")
    try:
        # 使用 raises 断言应该引发 TypeError 异常
        raises(TypeError, f)
        # 如果没有引发异常，则断言失败
        assert False
    except ValueError as e:
        # 检查异常的错误消息是否与预期的一致
        assert str(e) == "some error message"

# 测试 with 语句

# 测试预期异常不会引发任何错误的 with 语句
def test_expected_exception_is_silent_with():
    # 使用 raises 上下文管理器断言 ValueError 异常被引发
    with raises(ValueError):
        raise ValueError()


# 测试在 with 语句中缺少异常会触发 AssertionError
def test_lack_of_exception_triggers_AssertionError_with():
    try:
        # 使用 raises 上下文管理器断言应该引发 Exception 异常
        with raises(Exception):
            1 + 1
        # 如果没有引发异常，则断言失败
        assert False
    except Failed as e:
        # 检查错误消息中是否包含 "DID NOT RAISE"
        assert "DID NOT RAISE" in str(e)


# 测试意外异常会被传递的 with 语句
def test_unexpected_exception_is_passed_through_with():
    try:
        # 使用 raises 上下文管理器断言应该引发 TypeError 异常
        with raises(TypeError):
            raise ValueError("some error message")
        # 如果没有引发异常，则断言失败
        assert False
    except ValueError as e:
        # 检查异常的错误消息是否与预期的一致
        assert str(e) == "some error message"

# 现在可以使用 raises() 而不是 try/catch 来测试特定的异常类是否被引发

# 测试第二个参数应该是可调用对象或字符串
def test_second_argument_should_be_callable_or_string():
    # 使用 lambda 表达式测试 raises 函数是否接受 "irrelevant", 42 的调用
    raises(TypeError, lambda: raises("irrelevant", 42))


# 测试 warns 能捕获警告
def test_warns_catches_warning():
    # 使用 warnings 上下文管理器捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        # 使用 warns 上下文管理器断言应该引发 UserWarning 警告
        with warns(UserWarning):
            # 发出一个警告消息
            warnings.warn('this is the warning message')
        # 检查捕获的警告列表长度是否为 0
        assert len(w) == 0


# 测试 warns 在没有警告时会引发 Failed 异常
def test_warns_raises_without_warning():
    # 使用 raises 上下文管理器断言应该引发 Failed 异常
    with raises(Failed):
        # 使用 warns 上下文管理器断言应该引发 UserWarning 警告
        with warns(UserWarning):
            pass


# 测试 warns 能屏蔽其他警告
def test_warns_hides_other_warnings():
    # 使用 raises 上下文管理器断言应该引发 RuntimeWarning 警告
    with raises(RuntimeWarning):
        # 使用 warns 上下文管理器断言应该引发 UserWarning 警告
        with warns(UserWarning):
            # 发出第一个警告消息
            warnings.warn('this is the warning message', UserWarning)
            # 发出第二个警告消息
            warnings.warn('this is the other message', RuntimeWarning)


# 测试 warns 在警告后继续执行
def test_warns_continues_after_warning():
    # 使用 warnings 上下文管理器捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        finished = False
        # 使用 warns 上下文管理器断言应该引发 UserWarning 警告
        with warns(UserWarning):
            # 发出一个警告消息
            warnings.warn('this is the warning message')
            finished = True
        # 断言代码块中的代码执行完成
        assert finished
        # 检查捕获的警告列表长度是否为 0
        assert len(w) == 0


# 测试 warns 多个警告
def test_warns_many_warnings():
    # 使用 warns 上下文管理器断言应该引发 UserWarning 警告
    with warns(UserWarning):
        # 发出第一个警告消息
        warnings.warn('this is the warning message', UserWarning)
        # 发出第二个警告消息
        warnings.warn('this is the other warning message', UserWarning)


# 测试 warns 匹配特定的警告消息
def test_warns_match_matching():
    # 使用 warnings 上下文管理器捕获所有警告
    with warnings.catch_warnings(record=True) as w:
        # 使用 warns 上下文管理器断言应该引发 UserWarning 警告，并且匹配特定的警告消息
        with warns(UserWarning, match='this is the warning message'):
            # 发出一个警告消息
            warnings.warn('this is the warning message', UserWarning)
        # 检查捕获的警告列表长度是否为 0
        assert len(w) == 0
# 测试函数：验证 warns 和 raises(Failed) 能捕获预期的 UserWarning
def test_warns_match_non_matching():
    # 捕获所有 warnings
    with warnings.catch_warnings(record=True) as w:
        # 预期引发 Failed 异常
        with raises(Failed):
            # 预期引发 UserWarning 并且匹配指定的警告消息
            with warns(UserWarning, match='this is the warning message'):
                # 发出一个不符合预期的 UserWarning
                warnings.warn('this is not the expected warning message', UserWarning)
        # 断言捕获的 warnings 数量为 0
        assert len(w) == 0

# 内部函数：用于生成 Sympy 弃用警告
def _warn_sympy_deprecation(stacklevel=3):
    # 触发 Sympy 弃用警告
    sympy_deprecation_warning(
        "feature",
        active_deprecations_target="active-deprecations",
        deprecated_since_version="0.0.0",
        stacklevel=stacklevel,
    )

# 测试函数：验证 warns_deprecated_sympy 能捕获预期的 Sympy 弃用警告
def test_warns_deprecated_sympy_catches_warning():
    # 捕获所有 warnings
    with warnings.catch_warnings(record=True) as w:
        # 预期引发 Sympy 弃用警告
        with warns_deprecated_sympy():
            # 触发 Sympy 弃用警告
            _warn_sympy_deprecation()
        # 断言捕获的 warnings 数量为 0
        assert len(w) == 0

# 测试函数：验证 warns_deprecated_sympy 能检测到未发出警告时引发 Failed 异常
def test_warns_deprecated_sympy_raises_without_warning():
    # 预期引发 Failed 异常
    with raises(Failed):
        # 预期引发 Sympy 弃用警告
        with warns_deprecated_sympy():
            # 不触发任何警告
            pass

# 测试函数：验证 warns_deprecated_sympy 在错误的 stacklevel 下引发 Failed 异常
def test_warns_deprecated_sympy_wrong_stacklevel():
    # 预期引发 Failed 异常
    with raises(Failed):
        # 预期引发 Sympy 弃用警告
        with warns_deprecated_sympy():
            # 触发 Sympy 弃用警告，但使用错误的 stacklevel
            _warn_sympy_deprecation(stacklevel=1)

# 测试函数：验证 warns_deprecated_sympy 不会隐藏其他警告
def test_warns_deprecated_sympy_doesnt_hide_other_warnings():
    # 预期引发 RuntimeWarning
    with raises(RuntimeWarning):
        # 预期引发 Sympy 弃用警告
        with warns_deprecated_sympy():
            # 触发 Sympy 弃用警告
            _warn_sympy_deprecation()
            # 发出一个其他的 RuntimeWarning
            warnings.warn('this is the other message', RuntimeWarning)

# 测试函数：验证 warns_deprecated_sympy 能在发出警告后继续执行代码
def test_warns_deprecated_sympy_continues_after_warning():
    finished = False
    # 捕获所有 warnings
    with warnings.catch_warnings(record=True) as w:
        # 预期引发 Sympy 弃用警告
        with warns_deprecated_sympy():
            # 触发 Sympy 弃用警告
            _warn_sympy_deprecation()
            finished = True
        # 断言代码执行完成
        assert finished
        # 断言捕获的 warnings 数量为 0
        assert len(w) == 0

# 测试函数：验证 ignore_warnings 能忽略指定的 UserWarning
def test_ignore_ignores_warning():
    # 捕获所有 warnings
    with warnings.catch_warnings(record=True) as w:
        # 使用 ignore_warnings 忽略 UserWarning
        with ignore_warnings(UserWarning):
            # 发出一个被忽略的 UserWarning
            warnings.warn('this is the warning message')
        # 断言捕获的 warnings 数量为 0
        assert len(w) == 0

# 测试函数：验证 ignore_warnings 在没有警告时不引发异常
def test_ignore_does_not_raise_without_warning():
    # 捕获所有 warnings
    with warnings.catch_warnings(record=True) as w:
        # 使用 ignore_warnings，但不发出任何警告
        with ignore_warnings(UserWarning):
            pass
        # 断言捕获的 warnings 数量为 0
        assert len(w) == 0

# 测试函数：验证 ignore_warnings 能允许其他警告继续被捕获
def test_ignore_allows_other_warnings():
    # 捕获所有 warnings
    with warnings.catch_warnings(record=True) as w:
        # 确保所有 warnings 都被记录
        warnings.simplefilter("always")
        # 使用 ignore_warnings 忽略 UserWarning
        with ignore_warnings(UserWarning):
            # 发出一个被忽略的 UserWarning 和一个其他的 RuntimeWarning
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
        # 断言捕获的 warnings 数量为 1
        assert len(w) == 1
        # 断言捕获的消息类型为 RuntimeWarning
        assert isinstance(w[0].message, RuntimeWarning)
        # 断言捕获的消息内容为 'this is the other message'
        assert str(w[0].message) == 'this is the other message'

# 测试函数：验证 ignore_warnings 能在发出警告后继续执行代码
def test_ignore_continues_after_warning():
    finished = False
    # 捕获所有 warnings
    with warnings.catch_warnings(record=True) as w:
        # 使用 ignore_warnings 忽略 UserWarning
        with ignore_warnings(UserWarning):
            # 发出一个被忽略的 UserWarning
            warnings.warn('this is the warning message')
            finished = True
        # 断言代码执行完成
        assert finished
        # 断言捕获的 warnings 数量为 0
        assert len(w) == 0
def test_ignore_many_warnings():
    # 使用 warnings.catch_warnings() 上下文管理器捕获警告信息
    with warnings.catch_warnings(record=True) as w:
        # 设置警告过滤器，始终显示警告
        warnings.simplefilter("always")
        
        # 使用 ignore_warnings 上下文管理器来忽略特定类型的警告
        with ignore_warnings(UserWarning):
            # 发出用户警告消息
            warnings.warn('this is the warning message', UserWarning)
            # 发出运行时警告消息
            warnings.warn('this is the other message', RuntimeWarning)
            warnings.warn('this is the warning message', UserWarning)
            warnings.warn('this is the other message', RuntimeWarning)
            warnings.warn('this is the other message', RuntimeWarning)
        
        # 断言捕获的警告数量为3
        assert len(w) == 3
        # 遍历每个捕获的警告对象
        for wi in w:
            # 断言警告消息类型为 RuntimeWarning
            assert isinstance(wi.message, RuntimeWarning)
            # 断言警告消息内容为指定的文本
            assert str(wi.message) == 'this is the other message'
```