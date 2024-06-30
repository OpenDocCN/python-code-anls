# `D:\src\scipysrc\sympy\bin\test_py2_import.py`

```
#!/usr/bin/env python
#
# Tests that a useful message is give in the ImportError when trying to import
# sympy from Python 2. This ensures that we don't get a Py2 SyntaxError from
# sympy/__init__.py

# 导入 sys 模块，用于获取 Python 解释器相关信息
import sys
# 断言当前 Python 版本是 2.7
assert sys.version_info[:2] == (2, 7), "This test is for Python 2.7 only"

# 导入 os 模块
import os
# 获取当前脚本文件所在目录路径
thisdir = os.path.dirname(__file__)
# 获取当前脚本文件所在目录的父目录路径
parentdir = os.path.normpath(os.path.join(thisdir, '..'))

# 将 SymPy 根目录添加到系统路径中
sys.path.append(parentdir)

# 尝试导入 sympy 模块
try:
    import sympy
except ImportError as exc:
    # 将 ImportError 异常的消息转换成字符串
    message = str(exc)
    # 断言错误消息以 "Python version" 开头
    assert message.startswith("Python version")
    # 断言错误消息以 " or above is required for SymPy." 结尾
    assert message.endswith(" or above is required for SymPy.")
else:
    # 如果成功导入 sympy 模块，则抛出 AssertionError
    raise AssertionError("import sympy should give ImportError on Python 2.7")
```