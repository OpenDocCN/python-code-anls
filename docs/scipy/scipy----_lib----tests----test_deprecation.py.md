# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_deprecation.py`

```
# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 定义测试函数 test_cython_api_deprecation，用于测试某个 API 的废弃情况
def test_cython_api_deprecation():
    # 定义匹配字符串，用于检测废弃警告消息是否符合预期
    match = ("`scipy._lib._test_deprecation_def.foo_deprecated` "
             "is deprecated, use `foo` instead!\n"
             "Deprecated in Scipy 42.0.0")
    
    # 使用 pytest 的 warns 方法捕获特定类型的警告并验证其内容是否匹配预期
    with pytest.warns(DeprecationWarning, match=match):
        # 导入需要测试的模块 _test_deprecation_call，并执行测试
        from .. import _test_deprecation_call
    
    # 断言调用 _test_deprecation_call 模块的 call 方法返回值是否为 (1, 1)
    assert _test_deprecation_call.call() == (1, 1)
```