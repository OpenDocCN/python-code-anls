# `D:\src\scipysrc\matplotlib\lib\matplotlib\tests\test_backend_webagg.py`

```
import os  # 导入操作系统模块
import sys  # 导入系统相关的模块
import pytest  # 导入 pytest 测试框架

import matplotlib.backends.backend_webagg_core  # 导入 Matplotlib 的 WebAgg 核心后端
from matplotlib.testing import subprocess_run_for_testing  # 导入 Matplotlib 测试模块中的子进程运行函数


@pytest.mark.parametrize("backend", ["webagg", "nbagg"])
def test_webagg_fallback(backend):
    pytest.importorskip("tornado")  # 如果没有安装 tornado 模块则跳过测试
    if backend == "nbagg":
        pytest.importorskip("IPython")  # 如果 backend 是 'nbagg' 并且没有安装 IPython 则跳过测试

    env = dict(os.environ)  # 复制当前环境变量到字典 env 中
    if sys.platform != "win32":
        env["DISPLAY"] = ""  # 如果不是 Windows 系统，设置 DISPLAY 环境变量为空字符串

    env["MPLBACKEND"] = backend  # 设置 MPLBACKEND 环境变量为指定的 backend

    # 准备测试代码字符串，检查设置的 MPLBACKEND 是否正确，并验证 matplotlib 的后端
    test_code = (
        "import os;"
        + f"assert os.environ['MPLBACKEND'] == '{backend}';"
        + "import matplotlib.pyplot as plt; "
        + "print(plt.get_backend());"
        f"assert '{backend}' == plt.get_backend().lower();"
    )
    subprocess_run_for_testing([sys.executable, "-c", test_code], env=env, check=True)  # 在测试环境下运行测试代码


def test_webagg_core_no_toolbar():
    fm = matplotlib.backends.backend_webagg_core.FigureManagerWebAgg  # 获取 WebAgg 核心后端的图形管理器类
    assert fm._toolbar2_class is None  # 断言图形管理器的工具栏为 None，即不存在第二工具栏
```