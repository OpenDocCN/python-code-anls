# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_exceptions.py`

```
# 从 sympy.testing.pytest 模块中导入 raises 函数，用于测试异常情况
# 从 sympy.utilities.exceptions 模块中导入 sympy_deprecation_warning 函数，用于处理 SymPy 废弃警告

# 定义测试函数 test_sympy_deprecation_warning，用于测试 SymPy 废弃警告的异常情况
def test_sympy_deprecation_warning():
    # 测试 TypeError 异常，验证 sympy_deprecation_warning 函数的行为
    raises(TypeError, lambda: sympy_deprecation_warning('test',
                                                        deprecated_since_version=1.10,
                                                        active_deprecations_target='active-deprecations'))

    # 测试 ValueError 异常，验证 sympy_deprecation_warning 函数的行为
    raises(ValueError, lambda: sympy_deprecation_warning('test',
                                                          deprecated_since_version="1.10",
                                                          active_deprecations_target='(active-deprecations)='))
```