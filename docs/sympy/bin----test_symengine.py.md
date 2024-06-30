# `D:\src\scipysrc\sympy\bin\test_symengine.py`

```
#!/usr/bin/env python
"""
Run tests involving SymEngine

These are separate from the other optional dependency tests because they need
to be run with the `USE_SYMENGINE=1` environment variable set.

Run this as:

    $ USE_SYMENGINE=1 bin/test_symengine.py

"""

# 定义测试列表，包含需要测试的模块路径
TEST_LIST = [
    'sympy/physics/mechanics',
    'sympy/liealgebras',
]

# 主程序入口，检查是否在直接运行此脚本
if __name__ == "__main__":

    import os  # 导入操作系统相关功能
    import sys  # 导入系统相关功能
    os.environ["USE_SYMENGINE"] = "1"  # 设置环境变量 USE_SYMENGINE 为 "1"，启用 SymEngine

    # 导入路径修正函数，用于添加本地 SymPy 到 sys.path（CI 环境需要）
    from get_sympy import path_hack
    path_hack()

    import sympy  # 导入 SymPy 库

    # 注意：这里并不测试 doctests，因为使用 symengine 运行它们会有很多失败。
    args = TEST_LIST  # 将测试列表作为参数传递给测试函数
    exit_code = sympy.test(*args, verbose=True)  # 运行 SymPy 的测试，记录详细信息
    sys.exit(exit_code)  # 退出程序，返回测试结果的退出码
```