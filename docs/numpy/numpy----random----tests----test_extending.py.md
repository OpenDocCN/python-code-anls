# `.\numpy\numpy\random\tests\test_extending.py`

```
# 导入需要的模块和库
from importlib.util import spec_from_file_location, module_from_spec  # 导入模块动态加载相关函数
import os  # 导入操作系统相关功能
import pathlib  # 导入路径操作库
import pytest  # 导入 pytest 测试框架
import shutil  # 导入文件和目录操作库
import subprocess  # 导入子进程管理模块
import sys  # 导入系统相关功能
import sysconfig  # 导入 Python 配置信息模块
import textwrap  # 导入文本包装模块
import warnings  # 导入警告控制模块

import numpy as np  # 导入 NumPy 数学计算库
from numpy.testing import IS_WASM, IS_EDITABLE  # 导入 NumPy 测试相关标志

try:
    import cffi  # 尝试导入 cffi 库
except ImportError:
    cffi = None  # 如果导入失败，则设为 None

if sys.flags.optimize > 1:
    # 当 Python 优化标志大于 1 时，没有文档字符串可以检查
    # 所以 cffi 无法成功
    cffi = None

try:
    with warnings.catch_warnings(record=True) as w:
        # 解决 numba 问题 gh-4733
        warnings.filterwarnings('always', '', DeprecationWarning)
        import numba  # 尝试导入 numba 库
except (ImportError, SystemError):
    # 某些 numpy/numba 版本由于 numba 的错误而触发 SystemError
    numba = None  # 如果导入失败，则设为 None

try:
    import cython  # 尝试导入 cython
    from Cython.Compiler.Version import version as cython_version  # 导入 Cython 版本信息
except ImportError:
    cython = None  # 如果导入失败，则设为 None
else:
    from numpy._utils import _pep440  # 导入 NumPy 版本管理工具
    # 注意：与 pyproject.toml 中的版本信息保持同步
    required_version = '3.0.6'
    if _pep440.parse(cython_version) < _pep440.Version(required_version):
        # 如果 Cython 版本过旧或者错误，则跳过测试
        cython = None

@pytest.mark.skipif(
    IS_EDITABLE,
    reason='Editable install cannot find .pxd headers'
)
@pytest.mark.skipif(
        sys.platform == "win32" and sys.maxsize < 2**32,
        reason="Failing in 32-bit Windows wheel build job, skip for now"
)
@pytest.mark.skipif(IS_WASM, reason="Can't start subprocess")
@pytest.mark.skipif(cython is None, reason="requires cython")
@pytest.mark.slow
def test_cython(tmp_path):
    import glob  # 导入文件名匹配库
    # 在临时目录中构建示例
    srcdir = os.path.join(os.path.dirname(__file__), '..')
    shutil.copytree(srcdir, tmp_path / 'random')  # 复制目录树到临时路径下的 'random' 文件夹
    build_dir = tmp_path / 'random' / '_examples' / 'cython'
    target_dir = build_dir / "build"
    os.makedirs(target_dir, exist_ok=True)  # 创建目标目录，如果不存在则创建

    if sys.platform == "win32":
        # 在 Windows 平台下执行的构建步骤
        subprocess.check_call(["meson", "setup",
                               "--buildtype=release",
                               "--vsenv", str(build_dir)],
                              cwd=target_dir,
                              )
    else:
        # 在其他平台下执行的构建步骤
        subprocess.check_call(["meson", "setup", str(build_dir)],
                              cwd=target_dir
                              )
    subprocess.check_call(["meson", "compile", "-vv"], cwd=target_dir)

    # 确保在 Cython 中使用了 numpy 的 __init__.pxd
    # 虽然不是这个测试的一部分，但这是一个检查的便利位置

    g = glob.glob(str(target_dir / "*" / "extending.pyx.c"))
    with open(g[0]) as fid:
        txt_to_find = 'NumPy API declarations from "numpy/__init__'
        for i, line in enumerate(fid):
            if txt_to_find in line:
                break
        else:
            assert False, ("Could not find '{}' in C file, "
                           "wrong pxd used".format(txt_to_find))
    # 获取当前系统上 Python 模块的动态链接库文件的后缀名，通常用于确定共享库的文件名后缀
    suffix = sysconfig.get_config_var('EXT_SUFFIX')

    # 定义一个函数 load，用于动态加载指定模块名的共享库文件
    def load(modname):
        # 根据模块名拼接共享库文件的完整路径，带上动态链接库文件后缀名
        so = (target_dir / modname).with_suffix(suffix)
        # 根据共享库文件路径创建模块规范对象
        spec = spec_from_file_location(modname, so)
        # 根据模块规范对象创建模块对象
        mod = module_from_spec(spec)
        # 执行模块对象，将其载入内存
        spec.loader.exec_module(mod)
        # 返回载入的模块对象
        return mod

    # 测试能否导入指定模块名为 "extending" 的共享库
    load("extending")
    # 测试能否导入指定模块名为 "extending_cpp" 的共享库
    load("extending_cpp")

    # 实际测试 Cython 编写的 C 扩展模块 "extending_distributions"
    extending_distributions = load("extending_distributions")
    # 从 numpy.random 中导入 PCG64 生成器
    from numpy.random import PCG64
    # 使用扩展模块中的 uniforms_ex 函数生成随机数数组
    values = extending_distributions.uniforms_ex(PCG64(0), 10, 'd')
    # 断言生成的数组形状为 (10,)
    assert values.shape == (10,)
    # 断言生成的数组数据类型为 np.float64
    assert values.dtype == np.float64
# 如果没有安装 numba 或 cffi 模块，则跳过该测试
@pytest.mark.skipif(numba is None or cffi is None,
                    reason="requires numba and cffi")
# 定义一个测试函数，用于测试 numba 功能
def test_numba():
    # 导入 numba 扩展示例模块，这里使用 noqa: F401 禁止未使用的导入警告
    from numpy.random._examples.numba import extending  # noqa: F401

# 如果没有安装 cffi 模块，则跳过该测试
@pytest.mark.skipif(cffi is None, reason="requires cffi")
# 定义一个测试函数，用于测试 cffi 功能
def test_cffi():
    # 导入 cffi 扩展示例模块，这里使用 noqa: F401 禁止未使用的导入警告
    from numpy.random._examples.cffi import extending  # noqa: F401
```