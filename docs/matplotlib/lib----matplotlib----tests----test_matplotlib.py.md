# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_matplotlib.py`

```py
# 导入必要的标准库和第三方库
import os  # 导入操作系统接口模块
import subprocess  # 导入执行外部命令的模块
import sys  # 导入系统相关的功能模块

import pytest  # 导入 pytest 测试框架

import matplotlib  # 导入 matplotlib 绘图库
from matplotlib.testing import subprocess_run_for_testing  # 导入测试辅助函数

@pytest.mark.parametrize('version_str, version_tuple', [
    ('3.5.0', (3, 5, 0, 'final', 0)),
    ('3.5.0rc2', (3, 5, 0, 'candidate', 2)),
    ('3.5.0.dev820+g6768ef8c4c', (3, 5, 0, 'alpha', 820)),
    ('3.5.0.post820+g6768ef8c4c', (3, 5, 1, 'alpha', 820)),
])
def test_parse_to_version_info(version_str, version_tuple):
    assert matplotlib._parse_to_version_info(version_str) == version_tuple


@pytest.mark.skipif(sys.platform == "win32",
                    reason="chmod() doesn't work as is on Windows")
@pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0,
                    reason="chmod() doesn't work as root")
def test_tmpconfigdir_warning(tmp_path):
    """
    Test that a warning is emitted if a temporary configdir must be used.
    测试如果必须使用临时配置目录，则发出警告。
    """
    mode = os.stat(tmp_path).st_mode  # 获取临时路径的权限模式
    try:
        os.chmod(tmp_path, 0)  # 修改临时路径的权限模式为 0
        proc = subprocess_run_for_testing(
            [sys.executable, "-c", "import matplotlib"],
            env={**os.environ, "MPLCONFIGDIR": str(tmp_path)},
            stderr=subprocess.PIPE, text=True, check=True)
        assert "set the MPLCONFIGDIR" in proc.stderr  # 断言在标准错误输出中包含特定字符串
    finally:
        os.chmod(tmp_path, mode)  # 恢复临时路径的权限模式为原始值


def test_importable_with_no_home(tmp_path):
    """
    Test that matplotlib.pyplot can be imported even if pathlib.Path.home raises an exception.
    测试即使 pathlib.Path.home 抛出异常，也可以导入 matplotlib.pyplot。
    """
    subprocess_run_for_testing(
        [sys.executable, "-c",
         "import pathlib; pathlib.Path.home = lambda *args: 1/0; "
         "import matplotlib.pyplot"],
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path)}, check=True)


def test_use_doc_standard_backends():
    """
    Test that the standard backends mentioned in the docstring of
    matplotlib.use() are the same as in matplotlib.rcsetup.
    测试 matplotlib.use() 文档字符串中提到的标准后端与 matplotlib.rcsetup 中的相同。
    """
    def parse(key):
        backends = []
        for line in matplotlib.use.__doc__.split(key)[1].split('\n'):
            if not line.strip():
                break
            backends += [e.strip().lower() for e in line.split(',') if e]
        return backends

    from matplotlib.backends import BackendFilter, backend_registry

    assert (set(parse('- interactive backends:\n')) ==
            set(backend_registry.list_builtin(BackendFilter.INTERACTIVE)))
    assert (set(parse('- non-interactive backends:\n')) ==
            set(backend_registry.list_builtin(BackendFilter.NON_INTERACTIVE)))


def test_importable_with__OO():
    """
    When using -OO or export PYTHONOPTIMIZE=2, docstrings are discarded,
    this simple test may prevent something like issue #17970.
    测试在使用 -OO 或导出 PYTHONOPTIMIZE=2 时，文档字符串被丢弃，此简单测试可能可以避免类似 #17970 问题。
    """
    program = (
        "import matplotlib as mpl; "
        "import matplotlib.pyplot as plt; "
        "import matplotlib.cbook as cbook; "
        "import matplotlib.patches as mpatches"
    )
    subprocess_run_for_testing(
        [sys.executable, "-OO", "-c", program],
        env={**os.environ, "MPLBACKEND": ""}, check=True
        )
```