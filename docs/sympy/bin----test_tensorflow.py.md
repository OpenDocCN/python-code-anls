# `D:\src\scipysrc\sympy\bin\test_tensorflow.py`

```
#!/usr/bin/env python
"""
Run tests involving tensorflow

These are separate from the other optional dependency tests because tensorflow
pins the numpy version.
"""

# 定义测试列表，包括需要进行 doctest 的文件路径
TEST_LIST = DOCTEST_LIST = [
    'sympy/printing/tensorflow.py',
    'sympy/printing/tests/test_tensorflow.py',
    'sympy/stats/sampling',
    'sympy/utilities/lambdify.py',
    'sympy/utilities/tests/test_lambdify.py',
]

# 如果当前脚本被直接执行
if __name__ == "__main__":

    import sys

    # 导入本地的 SymPy 库并添加到 sys.path 中（在 CI 中需要）
    from get_sympy import path_hack
    path_hack()
    import sympy

    # 注意：这里不会测试 doctest，因为在使用 symengine 运行时会有很多失败。
    # 设置测试参数为 TEST_LIST
    args = TEST_LIST
    # 运行 SymPy 的测试，verbose=True 表示详细输出
    test_exit_code = sympy.test(*args, verbose=True)
    # 如果测试退出码非零，直接退出并返回退出码
    if test_exit_code != 0:
        sys.exit(test_exit_code)
    
    # 运行 SymPy 的 doctest，对 DOCTEST_LIST 中的路径进行测试
    doctest_exit_code = sympy.doctest(*DOCTEST_LIST)
    # 如果 doctest 退出码是 True，则将最终退出码设为 0，否则设为 1
    exit_code = 0 if doctest_exit_code is True else 1
    # 使用 sys.exit() 退出程序并返回退出码
    sys.exit(exit_code)
```