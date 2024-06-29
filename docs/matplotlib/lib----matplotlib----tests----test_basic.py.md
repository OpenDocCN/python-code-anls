# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_basic.py`

```py
import builtins
import os
import sys
import textwrap

from matplotlib.testing import subprocess_run_for_testing

# 定义一个简单的测试函数，用于验证 1 + 1 是否等于 2
def test_simple():
    assert 1 + 1 == 2

# 测试覆盖内置变量
def test_override_builtins():
    import pylab  # type: ignore

    # 允许覆盖的内置变量集合
    ok_to_override = {
        '__name__',
        '__doc__',
        '__package__',
        '__loader__',
        '__spec__',
        'any',
        'all',
        'sum',
        'divmod'
    }

    # 查找并收集被覆盖的内置变量
    overridden = {key for key in {*dir(pylab)} & {*dir(builtins)}
                  if getattr(pylab, key) != getattr(builtins, key)}

    # 断言被覆盖的变量在允许列表内
    assert overridden <= ok_to_override

# 测试延迟导入
def test_lazy_imports():
    # 定义测试用的源码字符串
    source = textwrap.dedent("""
    import sys

    import matplotlib.figure
    import matplotlib.backend_bases
    import matplotlib.pyplot

    assert 'matplotlib._tri' not in sys.modules
    assert 'matplotlib._qhull' not in sys.modules
    assert 'matplotlib._contour' not in sys.modules
    assert 'urllib.request' not in sys.modules
    """)

    # 运行子进程进行测试
    subprocess_run_for_testing(
        [sys.executable, '-c', source],
        env={**os.environ, "MPLBACKEND": "", "MATPLOTLIBRC": os.devnull},
        check=True)
```