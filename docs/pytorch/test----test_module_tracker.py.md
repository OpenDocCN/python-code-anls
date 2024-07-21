# `.\pytorch\test\test_module_tracker.py`

```
# Owner(s): ["module: unknown"]

from copy import copy  # 导入copy模块，用于对象的浅拷贝操作

import torch  # 导入PyTorch库
from torch.testing._internal.common_utils import run_tests, TestCase, xfailIfTorchDynamo  # 导入测试相关的函数和类
from torch.utils.module_tracker import ModuleTracker  # 导入ModuleTracker模块


class TestModuleTracker(TestCase):
    # "https://github.com/pytorch/pytorch/issues/127112
    @xfailIfTorchDynamo  # 在Torch Dynamo环境下标记测试为失败
    def test_module_hierarchy(self):
        seen_fw = []  # 存储前向传播观察到的结果
        seen_bw = []  # 存储反向传播观察到的结果

        class Foo(torch.nn.Module):
            def forward(self, x):
                x = x["a"].relu_()  # 对输入字典中键为"a"的数据进行ReLU激活函数操作
                seen_fw.append((copy(tracker.parents), tracker.is_bw))  # 将当前ModuleTracker的父模块和is_bw状态拷贝并存储到seen_fw中
                x.register_hook(
                    lambda grad: seen_bw.append((copy(tracker.parents), tracker.is_bw))
                )  # 注册一个钩子函数，将当前ModuleTracker的父模块和is_bw状态拷贝并存储到seen_bw中
                return {"a": torch.mm(x, x)}  # 返回经过矩阵乘法后的结果字典

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = Foo()  # 创建一个Foo模块实例
                self.b = torch.nn.ModuleDict({"nest": Foo()})  # 创建一个包含Foo模块实例的ModuleDict
                self.c = torch.nn.ModuleList([Foo()])  # 创建一个包含Foo模块实例的ModuleList

            def forward(self, x):
                x = self.c[0](x)  # 将输入x传递给ModuleList中的第一个模块进行前向传播
                return self.b["nest"](self.a(x))  # 将经过self.a和self.b中的"nest"模块的结果进行前向传播

        mod = Mod()  # 创建Mod类的实例

        with ModuleTracker() as tracker:  # 使用ModuleTracker进行上下文管理
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()  # 对输入数据进行前向传播和反向传播，并记录ModuleTracker的状态
            mod({"a": torch.randn(10, 10, requires_grad=True).clone()})[
                "a"
            ].sum().backward()  # 再次对输入数据进行前向传播和反向传播，并记录ModuleTracker的状态

        self.assertEqual(
            seen_fw,
            [
                ({"Global", "Mod", "Mod.c.0"}, False),  # 预期的前向传播观察结果
                ({"Global", "Mod", "Mod.a"}, False),  # 预期的前向传播观察结果
                ({"Global", "Mod", "Mod.b.nest"}, False),  # 预期的前向传播观察结果
                ({"Global", "Mod", "Mod.c.0"}, False),  # 预期的前向传播观察结果
                ({"Global", "Mod", "Mod.a"}, False),  # 预期的前向传播观察结果
                ({"Global", "Mod", "Mod.b.nest"}, False),  # 预期的前向传播观察结果
            ],
        )

        self.assertEqual(
            seen_bw,
            [
                ({"Global", "Mod", "Mod.b.nest"}, True),  # 预期的反向传播观察结果
                ({"Global", "Mod", "Mod.a"}, True),  # 预期的反向传播观察结果
                ({"Global", "Mod", "Mod.c.0"}, True),  # 预期的反向传播观察结果
                ({"Global", "Mod", "Mod.b.nest"}, True),  # 预期的反向传播观察结果
                ({"Global", "Mod", "Mod.a"}, True),  # 预期的反向传播观察结果
                ({"Global", "Mod", "Mod.c.0"}, True),  # 预期的反向传播观察结果
            ],
        )

    def test_bw_detection(self):
        mod = torch.nn.Linear(2, 2)  # 创建一个线性层模块

        with ModuleTracker() as tracker:  # 使用ModuleTracker进行上下文管理
            mod(torch.rand(2, requires_grad=True)).sum().backward()  # 对输入数据进行前向传播和反向传播，并记录ModuleTracker的状态
            self.assertFalse(tracker.is_bw)  # 断言ModuleTracker的反向传播状态为False
            self.assertEqual(tracker.parents, {"Global"})  # 断言ModuleTracker的父模块为全局

if __name__ == "__main__":
    run_tests()  # 运行所有的测试用例
```