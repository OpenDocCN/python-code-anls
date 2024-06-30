# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_extending.py`

```
# 导入标准库和第三方库
import os
import platform

# 导入 pytest 库，用于测试框架
import pytest

# 从 scipy._lib._testutils 中导入需要的变量和函数
from scipy._lib._testutils import IS_EDITABLE, _test_cython_extension, cython

# 使用 pytest 的标记，标记此测试函数为 "fail_slow"，对应的参数为 40
@pytest.mark.fail_slow(40)

# 如果是可编辑安装，则跳过测试，并给出原因
@pytest.mark.skipif(IS_EDITABLE,
                    reason='Editable install cannot find .pxd headers.')

# 如果运行环境是 wasm32 或 wasm64，则跳过测试，并给出原因
@pytest.mark.skipif(platform.machine() in ["wasm32", "wasm64"],
                    reason="Can't start subprocess")

# 如果 cython 模块不可用，则跳过测试，并给出原因
@pytest.mark.skipif(cython is None, reason="requires cython")
def test_cython(tmp_path):
    # 获取当前文件所在目录的上级目录作为源目录
    srcdir = os.path.dirname(os.path.dirname(__file__))
    
    # 调用 _test_cython_extension 函数，返回两个元组：extensions 和 extensions_cpp
    extensions, extensions_cpp = _test_cython_extension(tmp_path, srcdir)
    
    # 断言 extensions 对象的 brentq_example 方法返回值为预期值
    x = extensions.brentq_example()
    assert x == 0.6999942848231314
    
    # 断言 extensions_cpp 对象的 brentq_example 方法返回值为预期值
    x = extensions_cpp.brentq_example()
    assert x == 0.6999942848231314
```