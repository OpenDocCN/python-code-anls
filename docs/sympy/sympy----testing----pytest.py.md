# `D:\src\scipysrc\sympy\sympy\testing\pytest.py`

```
# 导入必要的标准库和第三方库
import sys  # 导入系统模块
import re  # 导入正则表达式模块
import functools  # 导入函数工具模块
import os  # 导入操作系统模块
import contextlib  # 导入上下文管理模块
import warnings  # 导入警告模块
import inspect  # 导入检查模块
import pathlib  # 导入路径操作模块
from typing import Any, Callable  # 导入类型提示模块

from sympy.utilities.exceptions import SymPyDeprecationWarning  # 导入SymPy的警告异常处理模块
# 以下导入主要用于向后兼容性。注意：不要在库代码中从这里导入（在库代码中导入 sympy.pytest 会破坏 pytest 的集成）。
from sympy.utilities.exceptions import ignore_warnings  # noqa:F401

# 检查是否在持续集成环境中
ON_CI = os.getenv('CI', None) == "true"

try:
    import pytest  # 尝试导入 pytest 测试框架
    USE_PYTEST = getattr(sys, '_running_pytest', False)
except ImportError:
    USE_PYTEST = False  # 如果导入失败，则标记为不使用 pytest 测试框架


raises: Callable[[Any, Any], Any]  # 声明 raises 类型提示为接受两个任意类型参数并返回任意类型的可调用对象
XFAIL: Callable[[Any], Any]  # 声明 XFAIL 类型提示为接受任意类型参数并返回任意类型的可调用对象
skip: Callable[[Any], Any]  # 声明 skip 类型提示为接受任意类型参数并返回任意类型的可调用对象
SKIP: Callable[[Any], Any]  # 声明 SKIP 类型提示为接受任意类型参数并返回任意类型的可调用对象
slow: Callable[[Any], Any]  # 声明 slow 类型提示为接受任意类型参数并返回任意类型的可调用对象
tooslow: Callable[[Any], Any]  # 声明 tooslow 类型提示为接受任意类型参数并返回任意类型的可调用对象
nocache_fail: Callable[[Any], Any]  # 声明 nocache_fail 类型提示为接受任意类型参数并返回任意类型的可调用对象

# 如果正在使用 pytest 测试框架，则定义 raises、skip、XFAIL、SKIP、slow、tooslow 和 nocache_fail
if USE_PYTEST:
    raises = pytest.raises  # 设置 raises 为 pytest.raises 函数的引用
    skip = pytest.skip  # 设置 skip 为 pytest.skip 函数的引用
    XFAIL = pytest.mark.xfail  # 设置 XFAIL 为 pytest.mark.xfail 函数的引用
    SKIP = pytest.mark.skip  # 设置 SKIP 为 pytest.mark.skip 函数的引用
    slow = pytest.mark.slow  # 设置 slow 为 pytest.mark.slow 函数的引用
    tooslow = pytest.mark.tooslow  # 设置 tooslow 为 pytest.mark.tooslow 函数的引用
    nocache_fail = pytest.mark.nocache_fail  # 设置 nocache_fail 为 pytest.mark.nocache_fail 函数的引用
    from _pytest.outcomes import Failed  # 导入 pytest 的 Failed 类

else:
    # 如果未使用 pytest 测试框架，则定义模拟的异常信息类 ExceptionInfo
    # _pytest._code.code.ExceptionInfo
    class ExceptionInfo:
        def __init__(self, value):
            self.value = value

        def __repr__(self):
            return "<ExceptionInfo {!r}>".format(self.value)
    def raises(expectedException, code=None):
        """
        Tests that ``code`` raises the exception ``expectedException``.

        ``code`` may be a callable, such as a lambda expression or function
        name.

        If ``code`` is not given or None, ``raises`` will return a context
        manager for use in ``with`` statements; the code to execute then
        comes from the scope of the ``with``.

        ``raises()`` does nothing if the callable raises the expected exception,
        otherwise it raises an AssertionError.

        Examples
        ========

        >>> from sympy.testing.pytest import raises

        >>> raises(ZeroDivisionError, lambda: 1/0)
        <ExceptionInfo ZeroDivisionError(...)>
        >>> raises(ZeroDivisionError, lambda: 1/2)
        Traceback (most recent call last):
        ...
        Failed: DID NOT RAISE

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0
        >>> with raises(ZeroDivisionError):
        ...     n = 1/2
        Traceback (most recent call last):
        ...
        Failed: DID NOT RAISE

        Note that you cannot test multiple statements via
        ``with raises``:

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0    # will execute and raise, aborting the ``with``
        ...     n = 9999/0 # never executed

        This is just what ``with`` is supposed to do: abort the
        contained statement sequence at the first exception and let
        the context manager deal with the exception.

        To test multiple statements, you'll need a separate ``with``
        for each:

        >>> with raises(ZeroDivisionError):
        ...     n = 1/0    # will execute and raise
        >>> with raises(ZeroDivisionError):
        ...     n = 9999/0 # will also execute and raise

        """
        # 如果 code 参数为 None，则返回一个 RaisesContext 对象，用于处理 with 语句
        if code is None:
            return RaisesContext(expectedException)
        # 如果 code 参数是一个可调用对象（函数或 lambda 表达式），则尝试执行它
        elif callable(code):
            try:
                code()
            # 捕获预期的异常，并返回 ExceptionInfo 对象
            except expectedException as e:
                return ExceptionInfo(e)
            # 如果没有抛出预期的异常，则抛出 Failed 异常
            raise Failed("DID NOT RAISE")
        # 如果 code 参数是字符串，则抛出 TypeError，提醒使用正确的用法
        elif isinstance(code, str):
            raise TypeError(
                '\'raises(xxx, "code")\' has been phased out; '
                'change \'raises(xxx, "expression")\' '
                'to \'raises(xxx, lambda: expression)\', '
                '\'raises(xxx, "statement")\' '
                'to \'with raises(xxx): statement\'')
        # 如果 code 参数不是合法类型，则抛出 TypeError
        else:
            raise TypeError(
                'raises() expects a callable for the 2nd argument.')

    class RaisesContext:
        def __init__(self, expectedException):
            self.expectedException = expectedException

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            # 如果
    # 定义一个自定义异常类 XPass，用于表示测试通过
    class XPass(Exception):
        pass

    # 定义一个自定义异常类 Skipped，用于表示测试被跳过
    class Skipped(Exception):
        pass

    # 定义一个自定义异常类 Failed，并标注类型为忽略（type: ignore）
    class Failed(Exception):  # type: ignore
        pass

    # 定义装饰器 XFAIL，用于处理预期失败的测试
    def XFAIL(func):
        def wrapper():
            try:
                # 调用被装饰的函数
                func()
            except Exception as e:
                message = str(e)
                # 检查异常消息是否为 "Timeout"，如果不是则抛出 XFail 异常
                if message != "Timeout":
                    raise XFail(func.__name__)
                else:
                    # 如果异常消息为 "Timeout"，则抛出 Skipped 异常
                    raise Skipped("Timeout")
            # 如果没有抛出异常，表示测试通过，抛出 XPass 异常
            raise XPass(func.__name__)

        # 更新 wrapper 函数的元数据，使其与 func 保持一致
        wrapper = functools.update_wrapper(wrapper, func)
        return wrapper

    # 定义函数 skip，抛出 Skipped 异常并附带给定的字符串作为原因
    def skip(str):
        raise Skipped(str)

    # 定义装饰器 SKIP，用于标记跳过测试的装饰器版本
    def SKIP(reason):
        """Similar to ``skip()``, but this is a decorator. """
        def wrapper(func):
            # 定义一个包装函数 func_wrapper，用于抛出 Skipped 异常并传递 reason
            def func_wrapper():
                raise Skipped(reason)

            # 更新 func_wrapper 函数的元数据，与 func 保持一致
            func_wrapper = functools.update_wrapper(func_wrapper, func)
            return func_wrapper

        return wrapper

    # 定义装饰器 slow，用于标记测试函数为慢速执行
    def slow(func):
        # 设置 func 的 _slow 属性为 True
        func._slow = True

        # 定义 func_wrapper 函数，其行为与原始 func 函数一致
        def func_wrapper():
            func()

        # 更新 func_wrapper 函数的元数据，与 func 保持一致
        func_wrapper = functools.update_wrapper(func_wrapper, func)
        # 将原始 func 函数的引用存储在 func_wrapper 函数的 __wrapped__ 属性中
        func_wrapper.__wrapped__ = func
        return func_wrapper

    # 定义装饰器 tooslow，用于标记测试函数为过于缓慢而被跳过
    def tooslow(func):
        # 设置 func 的 _slow 和 _tooslow 属性为 True
        func._slow = True
        func._tooslow = True

        # 定义 func_wrapper 函数，用于跳过测试并抛出 Skipped 异常，原因是 "Too slow"
        def func_wrapper():
            skip("Too slow")

        # 更新 func_wrapper 函数的元数据，与 func 保持一致
        func_wrapper = functools.update_wrapper(func_wrapper, func)
        # 将原始 func 函数的引用存储在 func_wrapper 函数的 __wrapped__ 属性中
        func_wrapper.__wrapped__ = func
        return func_wrapper

    # 定义装饰器 nocache_fail，用于标记测试函数在缓存禁用时会失败
    def nocache_fail(func):
        "Dummy decorator for marking tests that fail when cache is disabled"
        return func
@contextlib.contextmanager
# 定义一个上下文管理器 warns，用于测试是否发出特定类型的警告
def warns(warningcls, *, match='', test_stacklevel=True):
    '''
    Like raises but tests that warnings are emitted.

    >>> from sympy.testing.pytest import warns
    >>> import warnings

    >>> with warns(UserWarning):
    ...     warnings.warn('deprecated', UserWarning, stacklevel=2)

    >>> with warns(UserWarning):
    ...     pass
    Traceback (most recent call last):
    ...
    Failed: DID NOT WARN. No warnings of type UserWarning\
    was emitted. The list of emitted warnings is: [].

    ``test_stacklevel`` makes it check that the ``stacklevel`` parameter to
    ``warn()`` is set so that the warning shows the user line of code (the
    code under the warns() context manager). Set this to False if this is
    ambiguous or if the context manager does not test the direct user code
    that emits the warning.

    If the warning is a ``SymPyDeprecationWarning``, this additionally tests
    that the ``active_deprecations_target`` is a real target in the
    ``active-deprecations.md`` file.

    '''
    # Absorbs all warnings in warnrec
    # 使用 catch_warnings 捕获所有警告信息并记录在 warnrec 中
    with warnings.catch_warnings(record=True) as warnrec:
        # Any warning other than the one we are looking for is an error
        # 设置简单过滤器，将非指定警告类型的警告设为错误
        warnings.simplefilter("error")
        # 设定警告类型为 warningcls 的警告总是被记录
        warnings.filterwarnings("always", category=warningcls)
        # 现在运行测试代码
        yield warnrec

    # Raise if expected warning not found
    # 如果没有发出预期的警告，则引发异常
    if not any(issubclass(w.category, warningcls) for w in warnrec):
        msg = ('Failed: DID NOT WARN.'
               ' No warnings of type %s was emitted.'
               ' The list of emitted warnings is: %s.'
               ) % (warningcls, [w.message for w in warnrec])
        raise Failed(msg)

    # We don't include the match in the filter above because it would then
    # fall to the error filter, so we instead manually check that it matches
    # here
    # 检查是否每个警告信息的消息与 match 参数指定的正则表达式匹配
    for w in warnrec:
        # Should always be true due to the filters above
        # 由于上述过滤器的设置，这里的警告类型应始终是 warningcls 的子类
        assert issubclass(w.category, warningcls)
        # 检查警告消息是否与给定的正则表达式匹配
        if not re.compile(match, re.I).match(str(w.message)):
            raise Failed(f"Failed: WRONG MESSAGE. A warning with of the correct category ({warningcls.__name__}) was issued, but it did not match the given match regex ({match!r})")

    # 如果 test_stacklevel 为 True，则检查警告的 stacklevel 是否正确
    if test_stacklevel:
        # 遍历堆栈帧信息，查找发出警告的文件名
        for f in inspect.stack():
            thisfile = f.filename
            file = os.path.split(thisfile)[1]
            # 如果文件名以 'test_' 开头，则找到了测试文件
            if file.startswith('test_'):
                break
            # 如果文件名为 'doctest.py'，则跳过堆栈级别的测试
            elif file == 'doctest.py':
                return
        else:
            # 如果找不到发出警告的文件名，则引发运行时错误
            raise RuntimeError("Could not find the file for the given warning to test the stacklevel")
        
        # 再次遍历 warnrec，确保警告的堆栈级别正确
        for w in warnrec:
            if w.filename != thisfile:
                msg = f'''\
Failed: Warning has the wrong stacklevel. The warning stacklevel needs to be
set so that the line of code shown in the warning message is user code that
@contextlib.contextmanager
def warns_deprecated_sympy():
    '''
    上下文管理器函数，用于捕获 SymPyDeprecationWarning 警告

    这是测试 SymPy 中废弃特性是否正常触发 SymPyDeprecationWarning 的推荐方法。
    若要测试其他警告，请使用 `warns`。要在不断言警告被触发的情况下抑制警告，请使用 `ignore_warnings`。

    .. 注意::

       `warns_deprecated_sympy()` 仅用于 SymPy 测试套件中，在那里测试废弃警告是否正确触发。
       SymPy 代码库中的所有其他代码，包括文档示例，不应使用废弃的行为。

       如果您是 SymPy 的用户，并希望禁用 SymPyDeprecationWarnings，请使用警告过滤器（参见：`silencing-sympy-deprecation-warnings`）。
    '''
    try:
        yield
    except SymPyDeprecationWarning as warnrec:
        '''
        捕获 SymPyDeprecationWarning 异常对象

        检查是否在 `active-deprecations.md` 文件中正确指定了警告目标。
        如果指定的目标在文件中找不到，将抛出 `Failed` 异常。
        '''
        this_file = pathlib.Path(__file__)
        active_deprecations_file = (this_file.parent.parent.parent / 'doc' /
                                    'src' / 'explanation' /
                                    'active-deprecations.md')
        if not active_deprecations_file.exists():
            '''
            如果 `active-deprecations.md` 文件不存在，则无法测试 `active_deprecations_target` 是否有效。
            只有在 git 仓库中才能进行这个测试。
            '''
            return
        targets = []
        for w in warnrec:
            targets.append(w.message.active_deprecations_target)
        with open(active_deprecations_file, encoding="utf-8") as f:
            text = f.read()
        for target in targets:
            '''
            检查警告目标是否出现在 `active-deprecations.md` 文件中

            如果目标未在文件中找到，则抛出 `Failed` 异常，指示指定的废弃目标无效。
            '''
            if f'({target})=' not in text:
                raise Failed(f"The active deprecations target {target!r} does not appear to be a valid target in the active-deprecations.md file ({active_deprecations_file}).")
    '''
    # 使用 `warns` 上下文管理器捕获 SymPyDeprecationWarning 警告
    with warns(SymPyDeprecationWarning):
        # 在此上下文中产生的 SymPyDeprecationWarning 将被捕获
        yield
    '''
# 检查当前是否在 Pyodide 环境下运行的函数
def _running_under_pyodide():
    """Test if running under pyodide."""
    try:
        import pyodide_js  # type: ignore  # noqa
    except ImportError:
        # 如果导入 pyodide_js 失败，则说明不在 Pyodide 环境下
        return False
    else:
        # 成功导入 pyodide_js，说明在 Pyodide 环境下
        return True


# 用于在 Pyodide 环境下跳过测试的装饰器
def skip_under_pyodide(message):
    """Decorator to skip a test if running under pyodide."""
    def decorator(test_func):
        @functools.wraps(test_func)
        def test_wrapper():
            # 如果当前运行在 Pyodide 环境下，则跳过测试并输出消息
            if _running_under_pyodide():
                skip(message)
            # 否则，继续执行测试函数
            return test_func()
        return test_wrapper
    return decorator
```