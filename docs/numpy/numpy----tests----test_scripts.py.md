# `.\numpy\numpy\tests\test_scripts.py`

```py
""" Test scripts

Test that we can run executable scripts that have been installed with numpy.
"""
# 导入必要的模块
import sys  # 系统相关操作
import os   # 操作系统相关功能
import pytest  # 测试框架
from os.path import join as pathjoin, isfile, dirname  # 文件路径操作
import subprocess  # 子进程管理

import numpy as np  # 导入 numpy 库
from numpy.testing import assert_equal, IS_WASM  # 导入断言函数和 WASM 标志

# 检查是否在 inplace 安装环境下
is_inplace = isfile(pathjoin(dirname(np.__file__), '..', 'setup.py'))


def find_f2py_commands():
    # 根据操作系统选择不同的 f2py 脚本路径
    if sys.platform == 'win32':
        exe_dir = dirname(sys.executable)
        if exe_dir.endswith('Scripts'): # 如果是 virtualenv 下
            return [os.path.join(exe_dir, 'f2py')]
        else:
            return [os.path.join(exe_dir, "Scripts", 'f2py')]
    else:
        # Unix-like 系统中有三个可能的 f2py 脚本名
        version = sys.version_info
        major = str(version.major)
        minor = str(version.minor)
        return ['f2py', 'f2py' + major, 'f2py' + major + '.' + minor]


@pytest.mark.skipif(is_inplace, reason="Cannot test f2py command inplace")
@pytest.mark.xfail(reason="Test is unreliable")
@pytest.mark.parametrize('f2py_cmd', find_f2py_commands())
def test_f2py(f2py_cmd):
    # 测试能否运行 f2py 脚本
    stdout = subprocess.check_output([f2py_cmd, '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
def test_pep338():
    # 测试 PEP 338 兼容性
    stdout = subprocess.check_output([sys.executable, '-mnumpy.f2py', '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))
```