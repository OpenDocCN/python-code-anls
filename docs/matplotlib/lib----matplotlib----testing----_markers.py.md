# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\_markers.py`

```py
"""
pytest markers for the internal Matplotlib test suite.
"""

# 导入日志和文件操作模块
import logging
import shutil

# 导入 pytest 测试框架
import pytest

# 导入 Matplotlib 的测试相关模块和函数
import matplotlib.testing
import matplotlib.testing.compare
from matplotlib import _get_executable_info, ExecutableNotFoundError

# 获取当前模块的日志对象
_log = logging.getLogger(__name__)

# 检查是否存在 tex 可执行文件，若不存在则警告并返回 False
def _checkdep_usetex() -> bool:
    if not shutil.which("tex"):
        _log.warning("usetex mode requires TeX.")
        return False
    try:
        _get_executable_info("dvipng")
    except ExecutableNotFoundError:
        _log.warning("usetex mode requires dvipng.")
        return False
    try:
        _get_executable_info("gs")
    except ExecutableNotFoundError:
        _log.warning("usetex mode requires ghostscript.")
        return False
    return True

# 创建一个 pytest marker，条件为如果 'eps' 不在 matplotlib.testing.compare.converter 中，则跳过
needs_ghostscript = pytest.mark.skipif(
    "eps" not in matplotlib.testing.compare.converter,
    reason="This test needs a ghostscript installation")

# 创建一个 pytest marker，条件为若不满足 matplotlib.testing._check_for_pgf('lualatex')，则跳过
needs_pgf_lualatex = pytest.mark.skipif(
    not matplotlib.testing._check_for_pgf('lualatex'),
    reason='lualatex + pgf is required')

# 创建一个 pytest marker，条件为若不满足 matplotlib.testing._check_for_pgf('pdflatex')，则跳过
needs_pgf_pdflatex = pytest.mark.skipif(
    not matplotlib.testing._check_for_pgf('pdflatex'),
    reason='pdflatex + pgf is required')

# 创建一个 pytest marker，条件为若不满足 matplotlib.testing._check_for_pgf('xelatex')，则跳过
needs_pgf_xelatex = pytest.mark.skipif(
    not matplotlib.testing._check_for_pgf('xelatex'),
    reason='xelatex + pgf is required')

# 创建一个 pytest marker，条件为若不满足 _checkdep_usetex() 函数，即没有正确的 TeX 安装，则跳过
needs_usetex = pytest.mark.skipif(
    not _checkdep_usetex(),
    reason="This test needs a TeX installation")
```