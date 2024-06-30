# `D:\src\scipysrc\scipy\scipy\io\matlab\tests\test_byteordercodes.py`

```
''' Tests for byteorder module '''

# 导入 sys 模块，用于获取系统的字节序信息
import sys

# 从 numpy.testing 中导入 assert_ 函数，用于断言测试结果
from numpy.testing import assert_

# 从 pytest 中导入 raises 函数并命名为 assert_raises，用于捕获异常断言
from pytest import raises as assert_raises

# 导入 scipy.io.matlab._byteordercodes 模块，命名为 sibc，用于测试
import scipy.io.matlab._byteordercodes as sibc


# 定义测试函数 test_native
def test_native():
    # 检查系统的字节序是否为 little-endian，返回布尔值
    native_is_le = sys.byteorder == 'little'
    # 断言 sibc 模块中的 sys_is_le 属性与本地系统字节序一致
    assert_(sibc.sys_is_le == native_is_le)


# 定义测试函数 test_to_numpy
def test_to_numpy():
    # 如果系统字节序为 little-endian
    if sys.byteorder == 'little':
        # 断言转换为 numpy 格式的本地表示为 '<'
        assert_(sibc.to_numpy_code('native') == '<')
        # 断言转换为 numpy 格式的交换表示为 '>'
        assert_(sibc.to_numpy_code('swapped') == '>')
    else:
        # 断言转换为 numpy 格式的本地表示为 '>'
        assert_(sibc.to_numpy_code('native') == '>')
        # 断言转换为 numpy 格式的交换表示为 '<'
        assert_(sibc.to_numpy_code('swapped') == '<')
    
    # 断言转换为 numpy 格式的本地表示与 '=' 表示相同
    assert_(sibc.to_numpy_code('native') == sibc.to_numpy_code('='))
    # 断言转换为 numpy 格式的 big-endian 表示为 '>'
    assert_(sibc.to_numpy_code('big') == '>')
    
    # 对于以下多种代码表示方式，断言转换为 numpy 格式的 little-endian 表示均为 '<'
    for code in ('little', '<', 'l', 'L', 'le'):
        assert_(sibc.to_numpy_code(code) == '<')
    
    # 对于以下多种代码表示方式，断言转换为 numpy 格式的 big-endian 表示均为 '>'
    for code in ('big', '>', 'b', 'B', 'be'):
        assert_(sibc.to_numpy_code(code) == '>')
    
    # 断言对于 'silly string' 的输入，调用 to_numpy_code 函数会引发 ValueError 异常
    assert_raises(ValueError, sibc.to_numpy_code, 'silly string')
```