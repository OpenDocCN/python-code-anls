# `D:\src\scipysrc\scipy\scipy\_lib\_test_deprecation_call.pyx`

```
# 导入当前包中的 _test_deprecation_def 模块，并从中导入 foo 和 foo_deprecated 函数
from ._test_deprecation_def cimport foo, foo_deprecated

# 定义一个函数 call
def call():
    # 调用当前模块中导入的 foo 和 foo_deprecated 函数，并返回它们的结果
    return foo(), foo_deprecated()
```