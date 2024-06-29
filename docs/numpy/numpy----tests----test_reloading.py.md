# `.\numpy\numpy\tests\test_reloading.py`

```
# 导入系统相关模块
import sys
# 导入子进程管理模块
import subprocess
# 导入文本包装模块
import textwrap
# 导入模块重新加载函数
from importlib import reload
# 导入 pickle 序列化模块
import pickle

# 导入 pytest 测试框架
import pytest

# 导入 numpy 异常模块
import numpy.exceptions as ex
# 从 numpy 测试模块中导入多个断言函数
from numpy.testing import (
    assert_raises,
    assert_warns,
    assert_,
    assert_equal,
    IS_WASM,
)


# 定义测试函数：测试 NumPy 的重新加载
def test_numpy_reloading():
    # gh-7844. Also check that relevant globals retain their identity.
    # 导入 NumPy 主模块和全局变量模块
    import numpy as np
    import numpy._globals

    # 从 NumPy 模块中获取 _NoValue 对象
    _NoValue = np._NoValue
    # 导入 NumPy 异常模块中的警告类
    VisibleDeprecationWarning = ex.VisibleDeprecationWarning
    ModuleDeprecationWarning = ex.ModuleDeprecationWarning

    # 使用 assert_warns 断言捕获 UserWarning 异常
    with assert_warns(UserWarning):
        # 重新加载 NumPy 模块
        reload(np)
    # 使用 assert_ 断言 _NoValue 对象保持一致
    assert_(_NoValue is np._NoValue)
    # 使用 assert_ 断言 ModuleDeprecationWarning 对象保持一致
    assert_(ModuleDeprecationWarning is ex.ModuleDeprecationWarning)
    # 使用 assert_ 断言 VisibleDeprecationWarning 对象保持一致
    assert_(VisibleDeprecationWarning is ex.VisibleDeprecationWarning)

    # 使用 assert_raises 断言捕获 RuntimeError 异常
    assert_raises(RuntimeError, reload, numpy._globals)
    # 再次使用 assert_warns 断言捕获 UserWarning 异常
    with assert_warns(UserWarning):
        # 再次重新加载 NumPy 模块
        reload(np)
    # 使用 assert_ 断言 _NoValue 对象保持一致
    assert_(_NoValue is np._NoValue)
    # 使用 assert_ 断言 ModuleDeprecationWarning 对象保持一致
    assert_(ModuleDeprecationWarning is ex.ModuleDeprecationWarning)
    # 使用 assert_ 断言 VisibleDeprecationWarning 对象保持一致
    assert_(VisibleDeprecationWarning is ex.VisibleDeprecationWarning)


# 定义测试函数：测试 _NoValue 对象的序列化和反序列化
def test_novalue():
    # 导入 NumPy 主模块
    import numpy as np
    # 遍历序列化协议的范围
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        # 使用 assert_equal 断言 _NoValue 对象的字符串表示
        assert_equal(repr(np._NoValue), '<no value>')
        # 使用 assert_ 断言反序列化后的对象仍然是 _NoValue
        assert_(pickle.loads(pickle.dumps(np._NoValue,
                                          protocol=proto)) is np._NoValue)


# 使用 pytest 的装饰器标记此测试用例为条件跳过，若 IS_WASM 为真则跳过
@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
# 定义测试函数：全面重新导入 NumPy 测试
def test_full_reimport():
    """At the time of writing this, it is *not* truly supported, but
    apparently enough users rely on it, for it to be an annoying change
    when it started failing previously.
    """
    # 在撰写时，此功能不完全受支持，但显然有足够的用户依赖它，
    # 如果此前失败，这将是一个令人讨厌的变化。

    # 在一个新进程中进行测试，以确保在测试运行期间不会影响全局状态
    # （可能导致难以理解的测试失败）。这通常是不安全的，特别是因为我们还重新加载了 C 模块。
    # 使用 textwrap.dedent 方法去除代码段的缩进
    code = textwrap.dedent(r"""
        import sys
        from pytest import warns
        import numpy as np

        # 清理所有包含 "numpy" 的模块，以便重新导入
        for k in list(sys.modules.keys()):
            if "numpy" in k:
                del sys.modules[k]

        # 使用 warns 捕获 UserWarning 警告
        with warns(UserWarning):
            # 重新导入 NumPy 模块
            import numpy as np
        """)
    # 使用 subprocess.run 在新进程中执行 Python 代码，捕获输出
    p = subprocess.run([sys.executable, '-c', code], capture_output=True)
    # 如果返回码不为零，抛出 AssertionError 异常，输出详细信息
    if p.returncode:
        raise AssertionError(
            f"Non-zero return code: {p.returncode!r}\n\n{p.stderr.decode()}"
        )
```