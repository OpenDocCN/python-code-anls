# `D:\src\scipysrc\sympy\sympy\external\tests\test_importtools.py`

```
# 从 sympy.external 模块导入 import_module 函数
from sympy.external import import_module
# 从 sympy.testing.pytest 模块导入 warns 函数
from sympy.testing.pytest import warns

# 修复了解决 issue 6533 时出现的问题
def test_no_stdlib_collections():
    '''
    确保在不是较大列表的一部分时获取正确的 collections
    '''
    # 导入标准库的 collections 模块
    import collections
    # 尝试导入 matplotlib 模块，设置导入参数和最小模块版本，捕获 RuntimeError 异常
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['cm', 'collections']},
        min_module_version='1.1.0', catch=(RuntimeError,))
    # 如果成功导入了 matplotlib 模块
    if matplotlib:
        # 断言标准库的 collections 与 matplotlib 中的 collections 不相等
        assert collections != matplotlib.collections

# 另一个测试函数，与 test_no_stdlib_collections 类似，仅导入参数不同
def test_no_stdlib_collections2():
    '''
    确保在不是较大列表的一部分时获取正确的 collections
    '''
    # 导入标准库的 collections 模块
    import collections
    # 尝试导入 matplotlib 模块，设置导入参数，不包括 'cm'，捕获 RuntimeError 异常
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['collections']},
        min_module_version='1.1.0', catch=(RuntimeError,))
    # 如果成功导入了 matplotlib 模块
    if matplotlib:
        # 断言标准库的 collections 与 matplotlib 中的 collections 不相等
        assert collections != matplotlib.collections

# 另一个测试函数，与 test_no_stdlib_collections 类似，但不捕获异常
def test_no_stdlib_collections3():
    '''确保在没有捕获异常时获取正确的 collections'''
    # 导入标准库的 collections 模块
    import collections
    # 尝试导入 matplotlib 模块，设置导入参数和最小模块版本，不捕获任何异常
    matplotlib = import_module('matplotlib',
        import_kwargs={'fromlist': ['cm', 'collections']},
        min_module_version='1.1.0')
    # 如果成功导入了 matplotlib 模块
    if matplotlib:
        # 断言标准库的 collections 与 matplotlib 中的 collections 不相等
        assert collections != matplotlib.collections

# 测试函数，测试在 Python 3 中最小模块版本的 basestring 错误是否会引发 UserWarning
def test_min_module_version_python3_basestring_error():
    # 在引发 UserWarning 时进行警告
    with warns(UserWarning):
        # 尝试导入 mpmath 模块，设置最小模块版本为 '1000.0.1'
        import_module('mpmath', min_module_version='1000.0.1')
```