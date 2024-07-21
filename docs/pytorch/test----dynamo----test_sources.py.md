# `.\pytorch\test\dynamo\test_sources.py`

```py
# Owner(s): ["module: dynamo"]

# 导入PyTorch库
import torch
# 导入PyTorch内部模块 "_dynamo"
import torch._dynamo
# 导入PyTorch内部测试用例模块 "_dynamo.test_case"
import torch._dynamo.test_case
# 导入神经网络模块
import torch.nn as nn
# 从 "_dynamo.source" 模块导入相关源码类和函数
from torch._dynamo.source import (
    AttrSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
)

# 定义一个包含常量值的类
class CausalLMOutputWithPast:
    value = 5

# 定义测试类 "SourceTests"，继承自 "_dynamo.test_case.TestCase"
class SourceTests(torch._dynamo.test_case.TestCase):

    # 测试函数，验证是否来自本地源
    def test_is_local(self):
        # 创建本地源对象 "x_src"
        x_src = LocalSource("x")
        # 创建全局源对象 "y_src"
        y_src = GlobalSource("y")

        # 使用 "x_src" 创建属性源对象 "attr_x_a"
        attr_x_a = AttrSource(x_src, "a")
        # 使用 "y_src" 创建属性源对象 "attr_y_b"
        attr_y_b = AttrSource(y_src, "b")

        # 断言 "attr_x_a" 是否来自本地源
        self.assertTrue(is_from_local_source(attr_x_a))
        # 断言 "attr_y_b" 是否不来自本地源
        self.assertEqual(is_from_local_source(attr_y_b), False)

    # 测试函数，验证属性闭包
    def test_property_closure(self):
        # 外部函数，返回闭包函数
        def external_property():
            closed_value = 7

            def internal_function(self):
                return closed_value

            return internal_function

        # 定义一个类 "Elements"
        class Elements:
            # 创建属性 "myprop"，其值为外部函数返回的闭包函数
            myprop = property(external_property())

        # 定义函数 "func"，接受一个 "Elements" 类的实例作为参数
        def func(elements):
            # 如果 "elements.myprop" 为假，则返回一个张量
            if not elements.myprop:
                return torch.tensor([1, 2, 3])
            else:
                return torch.tensor([4, 5, 6])

        # 创建 "Elements" 类的实例 "e"
        e = Elements()
        # 调用函数 "func"，传入实例 "e"，并将结果赋给 "a"
        a = func(e)
        # 使用 "torch.compile" 函数，将 "func" 编译为后端为 "eager"、全图模式为真的函数，并传入实例 "e"，将结果赋给 "b"
        b = torch.compile(func, backend="eager", fullgraph=True)(e)
        # 断言 "a" 等于 "b"
        self.assertEqual(a, b)

    # 测试函数，验证支持的节点
    def test_supported_nodes(self):
        # 定义一个继承自 "nn.Module" 的模型类 "Model"
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.randn(10, 10)

            # 前向传播函数
            def forward(self):
                # 如果 "CausalLMOutputWithPast" 在支持节点字典中的类型为整数
                if (
                    torch.utils._pytree.SUPPORTED_NODES[CausalLMOutputWithPast].type
                    == int
                ):
                    x = torch.sin(self.x)
                else:
                    x = torch.cos(self.x)
                return x

        # 注册 "CausalLMOutputWithPast" 类为PyTree节点
        torch.utils._pytree.register_pytree_node(
            CausalLMOutputWithPast,
            lambda x: ((), None),
            lambda x, _: CausalLMOutputWithPast(),
        )

        # 导出模型 "Model"，调用 "torch.export.export" 函数
        torch.export.export(Model(), ())

# 如果当前脚本作为主程序运行，则执行 "_dynamo.test_case.run_tests()" 函数
if __name__ == "__main__":
    torch._dynamo.test_case.run_tests()
```