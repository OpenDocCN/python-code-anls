# `.\numpy\numpy\tests\test_configtool.py`

```
# 导入所需的标准库和第三方库
import os
import subprocess
import sysconfig

# 导入 pytest 和 numpy 相关模块
import pytest
import numpy as np

# 通过检查 numpy 的路径确定是否为可编辑安装
is_editable = not bool(np.__path__)
# 检查 numpy 是否安装在系统库中
numpy_in_sitepackages = sysconfig.get_path('platlib') in np.__file__

# 如果 numpy 不在系统库中且不是可编辑安装，则跳过测试
if not (numpy_in_sitepackages or is_editable):
    pytest.skip("`numpy-config` not expected to be installed",
                allow_module_level=True)

# 定义一个函数，用于运行 numpy-config 命令并返回标准输出结果
def check_numpyconfig(arg):
    # 执行 numpy-config 命令，并捕获标准输出
    p = subprocess.run(['numpy-config', arg], capture_output=True, text=True)
    # 检查命令的返回码，如果不是成功的返回码，会抛出异常
    p.check_returncode()
    # 返回去除首尾空白字符后的标准输出结果
    return p.stdout.strip()

# 使用 pytest.mark.skipif 装饰器标记测试用例，如果运行环境是 WASM 则跳过测试
@pytest.mark.skipif(IS_WASM, reason="wasm interpreter cannot start subprocess")
def test_configtool_version():
    # 运行 numpy-config 命令获取版本信息
    stdout = check_numpyconfig('--version')
    # 断言获取的版本信息与 numpy 库的版本相符
    assert stdout == np.__version__

# 使用 pytest.mark.skipif 装饰器标记测试用例，如果运行环境是 WASM 则跳过测试
@pytest.mark.skipif(IS_WASM, reason="wasm interpreter cannot start subprocess")
def test_configtool_cflags():
    # 运行 numpy-config 命令获取编译标志信息
    stdout = check_numpyconfig('--cflags')
    # 断言编译标志信息以指定路径结尾
    assert stdout.endswith(os.path.join('numpy', '_core', 'include'))

# 使用 pytest.mark.skipif 装饰器标记测试用例，如果运行环境是 WASM 则跳过测试
@pytest.mark.skipif(IS_WASM, reason="wasm interpreter cannot start subprocess")
def test_configtool_pkgconfigdir():
    # 运行 numpy-config 命令获取 pkgconfig 目录路径
    stdout = check_numpyconfig('--pkgconfigdir')
    # 断言 pkgconfig 目录路径以指定路径结尾
    assert stdout.endswith(os.path.join('numpy', '_core', 'lib', 'pkgconfig'))

    # 如果不是可编辑安装，则进一步检查 .pc 文件是否存在
    if not is_editable:
        assert os.path.exists(os.path.join(stdout, 'numpy.pc'))
```