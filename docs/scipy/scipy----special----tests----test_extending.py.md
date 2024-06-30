# `D:\src\scipysrc\scipy\scipy\special\tests\test_extending.py`

```
# 导入必要的模块
import os
import platform

# 导入 pytest 模块，用于测试
import pytest

# 导入 scipy 库中的测试工具函数和变量
from scipy._lib._testutils import IS_EDITABLE, _test_cython_extension, cython

# 导入 scipy 库中的特殊函数 beta 和 gamma
from scipy.special import beta, gamma


# 使用 pytest 的装饰器标记这是一个慢速失败的测试用例
@pytest.mark.fail_slow(40)
# 如果当前安装是可编辑的，则跳过测试，因为无法找到 .pxd 头文件
@pytest.mark.skipif(IS_EDITABLE,
                    reason='Editable install cannot find .pxd headers.')
# 如果运行平台是 wasm32 或 wasm64，则跳过测试，因为无法启动子进程
@pytest.mark.skipif(platform.machine() in ["wasm32", "wasm64"],
                    reason="Can't start subprocess")
# 如果没有安装 cython，则跳过测试
@pytest.mark.skipif(cython is None, reason="requires cython")
# 定义测试函数 test_cython，接受一个临时路径作为参数
def test_cython(tmp_path):
    # 获取当前文件所在目录的父目录
    srcdir = os.path.dirname(os.path.dirname(__file__))
    # 调用 _test_cython_extension 函数进行测试，返回两个对象
    extensions, extensions_cpp = _test_cython_extension(tmp_path, srcdir)

    # 断言：测试 Cython 扩展中的 beta 函数是否与 scipy.special 中的 beta 函数返回相同结果
    assert extensions.cy_beta(0.5, 0.1) == beta(0.5, 0.1)
    # 断言：测试 Cython 扩展中的 gamma 函数是否与 scipy.special 中的 gamma 函数返回相同结果
    assert extensions.cy_gamma(0.5 + 1.0j) == gamma(0.5 + 1.0j)

    # 断言：测试 C++ 编写的 Cython 扩展中的 beta 函数是否与 scipy.special 中的 beta 函数返回相同结果
    assert extensions_cpp.cy_beta(0.5, 0.1) == beta(0.5, 0.1)
    # 断言：测试 C++ 编写的 Cython 扩展中的 gamma 函数是否与 scipy.special 中的 gamma 函数返回相同结果
    assert extensions_cpp.cy_gamma(0.5 + 1.0j) == gamma(0.5 + 1.0j)
```