# `.\pytorch\test\package\package_a\test_all_leaf_modules_tracer.py`

```
# Owner(s): ["oncall: package/deploy"]
# 导入 torch.fx 中的 Tracer 类
from torch.fx import Tracer

# 定义一个新的类 TestAllLeafModulesTracer，继承自 Tracer 类
class TestAllLeafModulesTracer(Tracer):
    # 定义方法 is_leaf_module，用于判断是否为叶子模块
    def is_leaf_module(self, m, qualname):
        # 总是返回 True，表示所有模块都被认为是叶子模块
        return True
```