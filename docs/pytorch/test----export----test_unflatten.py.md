# `.\pytorch\test\export\test_unflatten.py`

```py
# Owner(s): ["oncall: export"]
# flake8: noqa

# 导入必要的模块和类
import copy  # 导入 copy 模块，用于复制对象
import dataclasses  # 导入 dataclasses 模块，用于定义数据类
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from contextlib import contextmanager  # 导入 contextmanager，用于创建上下文管理器
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于定义数据类

from re import escape  # 导入 escape 函数，用于对字符串进行正则表达式转义
from typing import Any, List  # 导入 Any 和 List 类型提示

import torch  # 导入 PyTorch 模块
import torch._dynamo as torchdynamo  # 导入 PyTorch 内部的 torchdynamo 模块

# 从 functorch.experimental.control_flow 模块导入 cond 和 map 函数
from functorch.experimental.control_flow import cond, map

from torch import Tensor  # 导入 Tensor 类型
# 从 torch._export.utils 模块导入一些实用函数和变量
from torch._export.utils import (
    get_buffer,
    get_param,
    is_buffer,
    is_param,
    register_dataclass_as_pytree_node,
)
# 从 torch._higher_order_ops.torchbind 导入 enable_torchbind_tracing 函数
from torch._higher_order_ops.torchbind import enable_torchbind_tracing

# 从 torch.export 模块导入 Constraint, Dim, dynamic_dim, export, FlatArgsAdapter 和 unflatten 等
from torch.export import (
    Constraint,
    Dim,
    dynamic_dim,
    export,
    FlatArgsAdapter,
    unflatten,
)

# 从 torch.export._trace 模块导入 DEFAULT_EXPORT_DYNAMO_CONFIG 变量
from torch.export._trace import DEFAULT_EXPORT_DYNAMO_CONFIG

# 从 torch.fx.experimental.proxy_tensor 模块导入 make_fx 函数
from torch.fx.experimental.proxy_tensor import make_fx

# 从 torch.testing 模块导入 FileCheck 类
from torch.testing import FileCheck

# 从 torch.testing._internal.common_utils 模块导入一系列实用函数和变量
from torch.testing._internal.common_utils import (
    find_library_location,
    IS_FBCODE,
    IS_MACOS,
    IS_SANDCASTLE,
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)

# 从 torch.testing._internal.torchbind_impls 模块导入 init_torchbind_implementations 函数
from torch.testing._internal.torchbind_impls import init_torchbind_implementations

# 从 torch.utils._pytree 模块导入 LeafSpec, tree_flatten, tree_unflatten, TreeSpec, treespec_dumps 和 treespec_loads
from torch.utils._pytree import (
    LeafSpec,
    tree_flatten,
    tree_unflatten,
    TreeSpec,
    treespec_dumps,
    treespec_loads,
)


# 定义一个测试类 TestUnflatten，继承自 TestCase
@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestUnflatten(TestCase):

    # 定义一个方法 compare_outputs，用于比较两个函数的输出
    def compare_outputs(self, eager, unflattened, args):
        # 调用 eager 函数计算原始输出
        orig_output = eager(*args)
        # 调用 unflattened 函数计算解压后的输出
        unflattened_output = unflattened(*args)
        # 断言两个输出在数值上相近
        self.assertTrue(torch.allclose(orig_output, unflattened_output))
    def test_unflatten_nested(self):
        # 定义一个嵌套的子模块，继承自 torch.nn.Module
        class NestedChild(torch.nn.Module):
            # 定义 forward 方法，实现对输入 x 的操作
            def forward(self, x):
                # 返回 x 除以自身的结果，用于示例，实际中应避免除以零操作
                return x / x

        # 定义一个子模块 Child1，继承自 torch.nn.Module
        class Child1(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 NestedChild 的实例作为子模块
                self.nested = NestedChild()
                # 注册一个参数 child1param，值为 torch.ones(2, 3)
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            # 前向传播方法
            def forward(self, x):
                # 对输入 x 执行嵌套模块操作
                x = self.nested(x)
                # 返回 x 加上子模块参数 child1param 的结果
                return x + self.child1param

        # 定义一个子模块 Child2，继承自 torch.nn.Module
        class Child2(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 注册一个缓冲区 child2buffer，值为 torch.ones(2, 3)
                self.register_buffer("child2buffer", torch.ones(2, 3))

            # 前向传播方法
            def forward(self, x):
                # 返回 x 减去子模块缓冲区 child2buffer 的结果
                return x - self.child2buffer

        # 定义一个主模块 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 Child1 的实例作为子模块 foo
                self.foo = Child1()
                # 创建一个 Child2 的实例作为子模块 bar
                self.bar = Child2()
                # 注册一个参数 rootparam，值为 torch.ones(2, 3)
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            # 前向传播方法
            def forward(self, x):
                # 对输入 x 执行与参数 rootparam 的乘法操作
                x = x * self.rootparam
                # 对输入 x 执行子模块 foo 的前向传播操作
                x = self.foo(x)
                # 对输入 x 执行子模块 bar 的前向传播操作
                x = self.bar(x)
                # 返回最终结果 x
                return x

        # 创建 MyModule 的实例 orig_eager
        orig_eager = MyModule()
        # 调用 export 函数，导出 orig_eager 模块
        export_module = export(orig_eager, (torch.rand(2, 3),), {})
        # 调用 unflatten 函数，将导出的模块 unflatten
        unflattened = unflatten(export_module)

        # 准备输入数据
        inputs = (torch.rand(2, 3),)

        # 比较原始模块及其所有子模块的输出
        self.compare_outputs(orig_eager, unflattened, inputs)
        self.compare_outputs(orig_eager.foo, unflattened.foo, inputs)
        self.compare_outputs(orig_eager.bar, unflattened.bar, inputs)
        self.compare_outputs(orig_eager.foo.nested, unflattened.foo.nested, inputs)

        # 检查状态字典是否相等
        orig_state_dict = orig_eager.state_dict()
        exported_state_dict = unflattened.state_dict()
        for name, value in orig_state_dict.items():
            # 使用 assertTrue 进行断言，验证两个状态字典的对应值是否在误差范围内相等
            self.assertTrue(torch.allclose(value, exported_state_dict[name]))
    def test_unflatten_buffer_mutation(self):
        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为"child2buffer"的缓冲区，初始值为全1的2x3张量
                self.register_buffer("child2buffer", torch.ones(2, 3))

            def forward(self, x):
                # 将输入张量x加到child2buffer上
                self.child2buffer.add_(x)
                # 返回x减去child2buffer的结果
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建Child类的实例，并将其存储在foo属性中
                self.foo = Child()
                # 注册一个名为"rootparam"的参数，初始值为全1的2x3张量
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # 调用Child实例的forward方法，并传入输入张量x
                x = self.foo(x)
                # 返回x乘以rootparam的结果
                return x * self.rootparam

        eager_module = MyModule()
        # 对eager_module进行导出，以便序列化或其他形式的持久化
        export_module = export(eager_module, (torch.rand(2, 3),), {})
        # 对导出的模块进行反序列化，使其结构与原始模块eager_module一致
        unflattened_module = unflatten(export_module)

        # 断言：在一个运行之前和之后，缓冲区应该保持不变
        eager_buffer = eager_module.foo.child2buffer
        unflattened_buffer = unflattened_module.foo.child2buffer
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))

        inputs = (torch.rand(2, 3),)
        # 调用eager_module和unflattened_module的forward方法，传入输入inputs
        eager_module(*inputs)
        unflattened_module(*inputs)
        # 再次断言：缓冲区应该保持不变
        self.assertTrue(torch.allclose(eager_buffer, unflattened_buffer))

    def test_unflatten_nested_access(self):
        class Child(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个名为"child2buffer"的缓冲区，初始值为全1的2x3张量
                self.register_buffer("child2buffer", torch.ones(2, 3))

            def forward(self, x):
                # 返回x减去child2buffer的结果
                return x - self.child2buffer

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建Child类的实例，并将其存储在foo属性中
                self.foo = Child()
                # 注册一个名为"rootparam"的参数，初始值为全1的2x3张量
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            def forward(self, x):
                # 将x加上foo.child2buffer的值
                x = x + self.foo.child2buffer
                # 调用Child实例的forward方法，并传入修改后的x
                x = self.foo(x)
                # 返回修改后的x
                return x

        eager_module = MyModule()
        # 对eager_module进行导出，以便序列化或其他形式的持久化
        export_module = export(eager_module, (torch.rand(2, 3),), {})
        # 对导出的模块进行反序列化，使其结构与原始模块eager_module一致
        unflattened_module = unflatten(export_module)

        inputs = (torch.rand(2, 3),)
        # 调用compare_outputs方法比较eager_module和unflattened_module的输出
        self.compare_outputs(eager_module, unflattened_module, inputs)
    # 定义一个测试方法，用于测试模型的反序列化和重新构造功能
    def test_unflatten_shared_submodule(self):
        # 定义一个名为Shared的内部类，继承自torch.nn.Module
        class Shared(torch.nn.Module):
            # 构造方法，初始化神经网络层
            def __init__(self):
                super().__init__()
                # 创建一个具有10个特征的LayerNorm层
                layernorm = torch.nn.LayerNorm(10)
                # 创建一个序列模块，包含两个LayerNorm层和两个ReLU层
                self.sub_net = torch.nn.Sequential(
                    layernorm,         # 第一个LayerNorm层
                    torch.nn.ReLU(),   # 第一个ReLU激活层
                    layernorm,         # 第二个LayerNorm层，与第一个相同的实例
                    torch.nn.ReLU(),   # 第二个ReLU激活层
                )

            # 前向传播方法，将输入x传递给sub_net模块
            def forward(self, x):
                return self.sub_net(x)

        # 创建Shared类的实例eager_module
        eager_module = Shared()
        # 创建一个包含一个随机张量的元组作为输入
        inps = (torch.rand(10),)
        # 调用export函数，将eager_module导出为一个可序列化的对象export_module
        export_module = export(eager_module, inps, {})
        # 使用unflatten函数重新构建export_module，得到unflattened_module
        unflattened_module = unflatten(export_module)
        # 比较eager_module和unflattened_module的输出，确保它们相同
        self.compare_outputs(eager_module, unflattened_module, inps)
        # 断言unflattened_module具有名为"sub_net"的属性
        self.assertTrue(hasattr(unflattened_module, "sub_net"))
        # 遍历eager_module的sub_net序列，断言unflattened_module的sub_net也具有相应的属性
        for i in range(len(eager_module.sub_net)):
            self.assertTrue(hasattr(unflattened_module.sub_net, str(i)))
        # 断言unflattened_module的sub_net中第一个和第三个属性引用相同的对象
        self.assertEqual(
            id(getattr(unflattened_module.sub_net, "0")),
            id(getattr(unflattened_module.sub_net, "2")),
        )

    # 如果在Windows平台上运行，则跳过此测试
    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    # 如果在Torch Dynamo非严格模式下运行，则跳过此测试
    @skipIfTorchDynamo("Non strict mode is not meant to run with dynamo")
    # 定义一个测试类方法 `test_unflatten_preserve_signature`
    def test_unflatten_preserve_signature(self):
        # 定义嵌套子类 `NestedChild`，继承自 `torch.nn.Module`
        class NestedChild(torch.nn.Module):
            # 定义前向传播方法 `forward`
            def forward(self, zx, y):
                # 返回一个字典，包含键 "x" 和 "w"，分别是两个张量运算的结果
                return {"x": y["key"] + zx[1], "w": y["key"] * zx[1]}

        # 定义子类 `Child1`，继承自 `torch.nn.Module`
        class Child1(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 `NestedChild` 类的实例
                self.nested = NestedChild()

            # 定义前向传播方法 `forward`
            def forward(self, x, y):
                # 创建一个与 `x` 形状相同的全 1 张量 `z`
                z = torch.ones_like(x)
                # 调用嵌套模块 `NestedChild` 的前向传播方法，并返回结果字典中 "w" 和 "x" 的差值
                xw = self.nested((z, x), y={"key": y})
                return xw["w"] + z - xw["x"]

        # 定义子类 `Child2`，继承自 `torch.nn.Module`
        class Child2(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()

            # 定义前向传播方法 `forward`
            def forward(self, x):
                # 返回输入张量 `x` 减去 1 的结果
                return x - 1

        # 定义主模块 `MyModule`，继承自 `torch.nn.Module`
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建 `Child1` 和 `Child2` 的实例，并分别赋值给 `foo` 和 `bar` 属性
                self.foo = Child1()
                self.bar = Child2()

            # 定义前向传播方法 `forward`
            def forward(self, x, y):
                # 调用 `Child1` 实例 `foo` 的前向传播方法，并将结果作为输入传递给 `Child2` 实例 `bar` 的前向传播方法
                x = self.foo(x, y)
                x = self.bar(x)
                return x

        # 创建 `MyModule` 类的实例 `orig_eager`
        orig_eager = MyModule()
        # 创建两个形状为 (2, 3) 的随机张量，并赋值给 `inps` 变量
        inps = torch.rand(2, 3), torch.rand(2, 3)
        # 遍历 `strict` 值为 True 和 False 的列表
        for strict in [True, False]:
            # 调用 `export` 函数，导出 `orig_eager` 模块，并赋值给 `export_module`
            export_module = export(
                orig_eager,
                inps,
                {},
                preserve_module_call_signature=("foo.nested",),
                strict=strict,
            )
            # 调用 `unflatten` 函数，将 `export_module` 展平化，并赋值给 `unflattened`
            unflattened = unflatten(export_module)
            # 调用 `self.compare_outputs` 方法，比较 `export_module.module()` 和 `unflattened` 的输出，传入 `inps` 参数
            self.compare_outputs(export_module.module(), unflattened, inps)
            # 将 `unflattened.foo.nested` 替换为 `NestedChild` 类的实例
            unflattened.foo.nested = NestedChild()
            # 再次比较 `export_module.module()` 和 `unflattened` 的输出，传入 `inps` 参数
            self.compare_outputs(export_module.module(), unflattened, inps)

            # 测试树结构规范不匹配的输入
            # 调用 `export_module.module()` 的前向传播方法，传入 `inps` 和一个额外的随机张量，将其赋值给 `orig_outs`
            orig_outs = export_module.module()(*inps)
            # 创建一个新的输入列表 `new_inps`，包含 `inps` 和一个新的形状为 (2, 3) 的随机张量
            new_inps = *inps, torch.rand(2, 3)
            # 使用 `self.assertRaisesRegex` 检查是否抛出 `TypeError` 异常，并检查异常信息是否匹配指定字符串
            with self.assertRaisesRegex(
                TypeError,
                "There is no flat args adapter sepcified. Are you sure you are calling this with the right arguments?",
            ):
                # 调用 `unflattened` 的前向传播方法，传入 `new_inps` 参数
                unflattened(new_inps)

            # 使用 `FlatArgsAdapter` 类创建一个自定义的扁平参数适配器 `KeepTwoFlatArgsAdapter`
            class KeepTwoFlatArgsAdapter(FlatArgsAdapter):
                # 定义 `adapt` 方法，用于自定义参数适配逻辑
                def adapt(
                    self,
                    target_spec: TreeSpec,
                    input_spec: TreeSpec,
                    input_args: List[Any],
                ) -> List[Any]:
                    # 当输入参数的长度大于 2 时，移除最后一个元素，直到长度等于 2
                    while len(input_args) > 2:
                        input_args.pop(-1)
                    # 返回处理后的输入参数列表
                    return input_args

            # 使用 `KeepTwoFlatArgsAdapter` 适配器重新调用 `unflatten` 函数，将结果赋值给 `unflattened`
            unflattened = unflatten(export_module, KeepTwoFlatArgsAdapter())
            # 调用 `unflattened` 的前向传播方法，传入 `new_inps` 参数，并将结果赋值给 `new_outs`
            new_outs = unflattened(*new_inps)
            # 使用 `self.assertTrue` 检查 `orig_outs` 和 `new_outs` 的所有元素是否接近
            self.assertTrue(torch.allclose(orig_outs, new_outs))
    # 定义一个测试方法，用于测试参数解构与重构的功能
    def test_unflatten_param_list_dict(self):
        # 定义一个继承自 torch.nn.Module 的类 Mod
        class Mod(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个参数列表对象
                self.param_list = torch.nn.ParameterList()
                # 创建一个参数字典对象
                self.param_dict = torch.nn.ParameterDict()
                # 循环两次，向参数列表和参数字典中添加随机生成的参数
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f"key_{i}"] = torch.nn.Parameter(
                        torch.randn((2, 3))
                    )

            # 前向传播方法
            def forward(self, x):
                # 循环两次，分别对输入的 x 加上参数列表中的参数和参数字典中的参数
                for i in range(2):
                    x = x + self.param_list[i]
                    x = x + self.param_dict[f"key_{i}"]
                return x

        # 将 Mod 类实例化并导出为 TorchScript，输入参数为随机生成的张量元组
        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),))
        # 使用 unflatten 函数对导出的 TorchScript 模块进行解构
        unflattened = unflatten(export_module)

        # 比较原始模块和解构后的模块的输出是否一致
        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    # 在 Windows 系统上跳过此测试
    @unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
    # 定义一个测试方法，测试在保留模块调用签名时的解构功能
    def test_unflatten_preserve_with_unused_input(self):
        # 定义一个继承自 torch.nn.Module 的类 M1
        class M1(torch.nn.Module):
            # 前向传播方法，接受三个输入参数，并返回前两个参数的和以及第三个参数
            def forward(self, x, a, b):
                return x + a, b

        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 M1 类的实例作为类属性
                self.m1 = M1()

            # 前向传播方法，接受两个输入参数 x 和 y
            def forward(self, x, y):
                # 使用 torch.topk 函数对输入的 y 进行计算，返回前两个最大值和对应的索引
                a, b = torch.topk(y, 2)
                # 调用 self.m1 的前向传播方法，传入 x, a, b，仅返回第一个返回值
                return self.m1(x, a, b)[0]

        # 导出 M 类的 TorchScript，输入为两个随机生成的张量
        ep = torch.export.export(
            M(),
            (torch.randn(2), torch.randn(5)),
            # 在解构过程中保留对 m1 模块调用的签名
            preserve_module_call_signature=("m1",),
            # 允许松散的模式，即未使用的输入参数
            strict=False,
        )
        # 对导出的 TorchScript 图进行死代码消除优化
        ep.graph.eliminate_dead_code()
        # 使用 unflatten 函数对导出的 TorchScript 进行解构
        unflattened = unflatten(ep)
        # 比较原始模块和解构后的模块的输出是否一致
        self.compare_outputs(ep.module(), unflattened, (torch.randn(2), torch.randn(5)))
    # 定义一个测试用例，测试 unflatten 函数处理错误输入的情况
    def test_unflatten_wrong_input(self):
        # 定义一个继承自 torch.nn.Module 的子类 Mod
        class Mod(torch.nn.Module):
            # 构造函数，初始化参数列表和参数字典
            def __init__(self):
                super().__init__()
                self.param_list = torch.nn.ParameterList()
                self.param_dict = torch.nn.ParameterDict()
                # 循环两次，向 param_list 添加随机参数，并向 param_dict 添加带有键的随机参数
                for i in range(2):
                    self.param_list.append(torch.nn.Parameter(torch.randn((2, 3))))
                    self.param_dict[f"key_{i}"] = torch.nn.Parameter(
                        torch.randn((2, 3))
                    )

            # 前向传播函数
            def forward(self, x):
                # 计算输入张量 x 的总和
                a = x.sum()
                # 循环两次，分别对 param_list 和 param_dict 中的参数求和
                for i in range(2):
                    a = a + self.param_list[i].sum()
                    a = a + self.param_dict[f"key_{i}"].sum()
                return a

        # 使用 torch.export.export 导出 Mod 的模块，并传入一个随机张量作为输入
        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),))
        # 使用 assertRaisesRegex 检测 RuntimeError 异常，并验证异常信息中是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be equal to 2, but got 6"),
        ):
            # 调用导出模块的 forward 方法，传入一个形状为 (6, 6) 的随机张量
            export_module.module()(torch.randn(6, 6))

        # 调用 unflatten 函数对导出模块进行反平铺
        unflattened = unflatten(export_module)
        # 再次使用 assertRaisesRegex 检测 RuntimeError 异常，并验证异常信息中是否包含特定字符串
        with self.assertRaisesRegex(
            RuntimeError,
            escape("Expected input at *args[0].shape[0] to be equal to 2, but got 6"),
        ):
            # 调用反平铺后的模块的 forward 方法，传入一个形状为 (6, 6) 的随机张量
            unflattened(torch.randn(6, 6))

    # 定义一个测试用例，测试 unflatten 函数在进行就地编译时的行为
    def test_unflatten_with_inplace_compile(self):
        # 定义一个嵌套的子类 NestedChild，其 forward 方法返回输入张量 x 的逐元素除法结果
        class NestedChild(torch.nn.Module):
            def forward(self, x):
                return x / x

        # 定义 Child1 类，继承自 torch.nn.Module
        class Child1(torch.nn.Module):
            # 构造函数，初始化一个嵌套模块和一个参数 child1param
            def __init__(self):
                super().__init__()
                self.nested = NestedChild()
                self.register_parameter(
                    "child1param", torch.nn.Parameter(torch.ones(2, 3))
                )

            # 前向传播函数
            def forward(self, x):
                # 将输入 x 传递给嵌套模块，然后加上 child1param 参数
                x = self.nested(x)
                return x + self.child1param

        # 定义 Child2 类，继承自 torch.nn.Module
        class Child2(torch.nn.Module):
            # 构造函数，注册一个缓冲区 child2buffer
            def __init__(self):
                super().__init__()
                self.register_buffer("child2buffer", torch.ones(2, 3))

            # 前向传播函数，从输入 x 中减去 child2buffer 缓冲区
            def forward(self, x):
                return x - self.child2buffer

        # 定义 MyModule 类，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            # 构造函数，初始化两个子模块和一个参数 rootparam
            def __init__(self):
                super().__init__()
                self.foo = Child1()
                self.bar = Child2()
                self.register_parameter(
                    "rootparam", torch.nn.Parameter(torch.ones(2, 3))
                )

            # 前向传播函数
            def forward(self, x):
                # 输入张量 x 与 rootparam 参数相乘
                x = x * self.rootparam
                # 将 x 传递给 foo 子模块，再将结果传递给 bar 子模块
                x = self.foo(x)
                x = self.bar(x)
                return x

        # 创建原始的 MyModule 实例
        orig_eager = MyModule()
        # 使用 torch.export.export 导出 orig_eager 模块，并传入一个随机张量作为输入
        export_module = torch.export.export(orig_eager, (torch.rand(2, 3),), {})
        # 调用 unflatten 函数对导出模块进行反平铺
        unflattened = unflatten(export_module)

        # 对 unflattened 模块的 foo 子模块进行就地编译，确保整个图形不会中断
        unflattened.foo.compile(fullgraph=True)

        # 定义输入张量
        inputs = (torch.rand(2, 3),)
        # 使用 compare_outputs 方法比较 orig_eager 和 unflattened 在给定输入下的输出
        self.compare_outputs(orig_eager, unflattened, inputs)
   `
    # 定义测试函数 test_fx_trace
    def test_fx_trace(self):
        # 定义一个自定义模块 MyModule，继承自 torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义前向传播方法
            def forward(self, x, y):
                x = x[0] + x[1]  # 将输入 x 的第一个元素与第二个元素相加
                x = x + y["foo"]  # 将输入 y 中键为 "foo" 的值与 x 相加
                return x  # 返回计算结果

        # 初始化 MyModule 模块的一个实例
        orig_eager = MyModule()
        # 定义输入数据，包括一个元组和一个字典
        inputs = ((torch.rand(2, 3), torch.rand(2, 3)), {"foo": torch.rand(2, 3)})
        # 使用 export 函数导出原始模块，传入输入数据和空字典
        export_module = export(orig_eager, inputs, {})

        # 将导出的模块解包
        unflattened = unflatten(export_module)
        # 使用 torch.fx.symbolic_trace 对解包后的模块进行符号跟踪，传入具体参数
        torch.fx.symbolic_trace(
            unflattened, concrete_args=(torch.fx.PH, torch.fx.PH, torch.fx.PH)
        )

    # 定义测试函数 test_double_nested_submodule
    def test_double_nested_submodule(self):
        # 定义一个子模块 SubSubMod，继承自 torch.nn.Module
        class SubSubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 定义前向传播方法，返回输入的平方
            def forward(self, x):
                return x * x

        # 定义一个子模块 SubMod，继承自 torch.nn.Module，包含 SubSubMod 子模块
        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.subsubmod = SubSubMod()

            # 定义前向传播方法，返回输入减去自身
            def forward(self, x):
                return x - x

        # 定义主模块 MyModule，继承自 torch.nn.Module，包含 SubMod 子模块
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = SubMod()

            # 定义前向传播方法，返回输入加上子模块的子模块计算结果
            def forward(self, x):
                return x + self.submod.subsubmod(x)

        # 初始化 MyModule 模块的一个实例
        orig_eager = MyModule()
        # 使用 torch.export.export 导出原始模块，传入一个随机输入张量和空字典
        export_module = torch.export.export(orig_eager, (torch.rand(2, 3),), {})
        # 将导出的模块解包
        unflattened = unflatten(export_module)

        # 定义输入数据，包括一个随机张量元组
        inputs = (torch.rand(2, 3),)
        # 比较原始模块和解包模块的输出，传入输入数据
        self.compare_outputs(orig_eager, unflattened, inputs)

    # 定义测试函数 test_unflatten_container_type
    def test_unflatten_container_type(self):
        # 定义一个叶子模块 Leaf，继承自 torch.nn.Module，包含一个线性层
        class Leaf(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            # 定义前向传播方法，返回线性层的输出
            def forward(self, x):
                return self.linear(x)

        # 定义一个模块 Bar，继承自 torch.nn.Module，包含一个 Leaf 子模块和一个缓冲区
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()
                self.register_buffer("buffer", torch.randn(4, 4))

            # 定义前向传播方法，返回缓冲区的和、叶子模块的输出和输入 z 的所有元素的和
            def forward(self, x, z):
                return self.buffer.sum() + self.leaf(x).sum() + z[0].sum() + z[1].sum()

        # 定义一个主模块 Foo，继承自 torch.nn.Module，包含一个 Bar 子模块
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bar = Bar()

            # 定义前向传播方法，返回子模块的缓冲区和输入 x、z 的所有元素的和
            def forward(self, x, z):
                y = self.bar.buffer + x + z[0] + z[1]
                return self.bar(x, z) + y.sum()

        # 定义输入数据，包括一个随机张量和一个包含两个随机张量的列表
        inp = (torch.randn(4, 4), [torch.randn(4, 4), torch.randn(4, 4)])
        # 初始化 Foo 模块的一个实例
        mod = Foo()
        # 使用 torch.export.export 导出 Foo 模块，传入输入数据
        ep_strict = torch.export.export(mod, inp)
        # 使用 torch.export.export 导出 Foo 模块，传入输入数据，设置严格模式为 False
        ep_non_strict = torch.export.export(mod, inp, strict=False)

        # 将非严格模式下的导出模块解包
        gm_unflat_non_strict = unflatten(ep_non_strict)
        # 再次导出解包后的模块，传入输入数据和设置严格模式为 False
        ep = torch.export.export(gm_unflat_non_strict, inp, strict=False)
        # 验证导出的模块与原始模块的输出是否相同，比较结果是否接近
        self.assertTrue(torch.allclose(ep.module()(*inp), mod(*inp)))
    # 定义一个测试方法，用于验证未展开的模块节点是否具有特定的元数据值
    def test_unflattened_module_nodes_has_meta_val(self):
        # 定义一个简单的子模块类，继承自torch.nn.Module
        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x, x * x

        # 定义主模块类，继承自torch.nn.Module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = SubMod()  # 创建一个SubMod实例作为子模块

            def forward(self, x):
                return x + sum(self.submod(x))  # 返回输入加上子模块输出的和

        orig_eager = MyModule()  # 创建原始的MyModule实例
        export_module = torch.export.export(orig_eager, (torch.rand(2, 3),), {})  # 导出模块
        unflattened = unflatten(export_module)  # 对导出的模块进行展开

        inputs = (torch.rand(2, 3),)
        self.compare_outputs(orig_eager, unflattened, inputs)  # 比较展开前后的模块输出

        # 定义一个函数，用于检查图中节点的元数据是否包含"val"属性
        def check_meta(gm):
            for n in gm.graph.nodes:
                if n.op == "output":  # 跳过输出节点
                    continue
                self.assertTrue(n.meta.get("val") is not None)  # 断言节点的元数据中包含"val"

        # 遍历展开后的模块中的所有子模块，并检查它们的元数据
        for m in unflattened.modules():
            check_meta(m)

    # 定义一个测试方法，用于验证展开后的模块中call_module节点的输入顺序
    def test_placeholder_and_get_attr_ordering_after_unflattened(self):
        # 定义一个简单的转置模块类，继承自torch.nn.Module
        class TransposeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)  # 创建一个卷积层

            def forward(self, x):
                x = self.conv(x)  # 对输入进行卷积操作
                return x.transpose(0, 1)  # 返回转置后的张量

        x = torch.randn(32, 3, 64, 64)
        exported_program = export(TransposeModule(), args=(x,))  # 导出转置模块
        unflattened_module = unflatten(exported_program)  # 对导出的模块进行展开

        # 检查创建的call_module节点的输入顺序是否正确
        call_module_input_order = []
        for node in unflattened_module.graph.nodes:
            if node.op == "call_module":
                transpose_module = unflattened_module.get_submodule(node.target)
                for sub_node in transpose_module.graph.nodes:
                    if sub_node.op == "placeholder" or sub_node.op == "get_attr":
                        call_module_input_order.append(sub_node.op)
        self.assertEqual(
            call_module_input_order, ["placeholder", "get_attr", "get_attr"]
        )  # 断言call_module节点的输入顺序符合预期

    # 定义一个测试方法，用于验证展开常量张量的模块
    def test_unflatten_constant_tensor(self):
        # 定义一个简单的子模块类，继承自torch.nn.Module
        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.initializer = 0.1  # 初始化一个常量值

            def forward(self, x):
                return x + torch.tensor(self.initializer)  # 返回输入加上常量张量

        # 定义主模块类，继承自torch.nn.Module
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.submod = SubMod()  # 创建一个SubMod实例作为子模块

            def forward(self, x):
                return x + self.submod(x)  # 返回输入加上子模块输出

        export_module = torch.export.export(Mod(), (torch.randn((2, 3)),))  # 导出模块
        unflattened = unflatten(export_module)  # 对导出的模块进行展开

        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )  # 比较展开前后的模块输出
    def test_unflatten_constant_obj(self):
        # 初始化 torchbind 实现
        init_torchbind_implementations()

        # 注册一个虚拟类 FakeFoo 到 TorchScript 测试类 "_TorchScriptTesting::_Foo"
        @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            def __init__(self, x: int, y: int):
                # 初始化 FakeFoo 实例的属性 x 和 y
                self.x = x
                self.y = y

            @classmethod
            def __obj_unflatten__(cls, flat_ctx):
                # 通过 flat_ctx 字典反序列化并返回一个类实例
                return cls(**dict(flat_ctx))

            def add_tensor(self, z):
                # 返回一个计算结果，使用 self.x、self.y 和 z
                return (self.x + self.y) * z

        # 定义一个继承自 torch.nn.Module 的子模块 SubMod
        class SubMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 SubMod 实例的属性 attr 为一个 _TorchScriptTesting::_Foo 实例
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # 返回计算结果，使用输入 x 和 self.attr.add_tensor(x)
                return x + self.attr.add_tensor(x)

        # 定义一个继承自 torch.nn.Module 的模块 Mod
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 Mod 实例的属性 submod 为 SubMod 实例
                self.submod = SubMod()

            def forward(self, x):
                # 返回计算结果，使用输入 x 和 self.submod(x)
                return x + self.submod(x)

        # 使用 enable_torchbind_tracing 上下文管理器启用 torchbind 追踪
        with enable_torchbind_tracing():
            # 导出 Mod 模块，输入参数为 torch.randn((2, 3))，不强制要求严格性
            export_module = torch.export.export(
                Mod(), (torch.randn((2, 3)),), strict=False
            )
        
        # 对导出的模块进行反序列化
        unflattened = unflatten(export_module)

        # 比较导出模块和反序列化后的结果
        self.compare_outputs(
            export_module.module(), unflattened, (torch.randn((2, 3)),)
        )

    def test_nested_leaf_non_strict(self):
        # 定义 Leaf 类，继承自 torch.nn.Module
        class Leaf(torch.nn.Module):
            def forward(self, x):
                # 返回 x + 1 的计算结果
                return x + 1

        # 定义 Nested 类，继承自 torch.nn.Module
        class Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 Nested 实例的属性 leaf 为 Leaf 实例
                self.leaf = Leaf()

            def forward(self, x):
                # 返回计算结果，使用 self.leaf(x) + 2
                return self.leaf(x) + 2

        # 定义 TopLevel 类，继承自 torch.nn.Module
        class TopLevel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化 TopLevel 实例的属性 nested 为 Nested 实例
                self.nested = Nested()

            def forward(self, x):
                # 返回计算结果，使用 self.nested(x) + 3
                return self.nested(x) + 3

        # 导出 TopLevel 模块，输入参数为 torch.randn(3)，不强制要求严格性
        ep = torch.export.export(
            TopLevel(),
            (torch.randn(3),),
            strict=False,
            preserve_module_call_signature=("nested",),
        )

        # 对导出的模块进行反序列化
        torch.export.unflatten(ep)
    def test_duplicate_placeholder(self):
        N, C, H, W = 1, 2, 2, 3

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个LayerNorm层对象
                layer = torch.nn.LayerNorm([C, H, W])
                # 使用ModuleList将多个相同的LayerNorm层对象放入列表中
                self.norms = torch.nn.ModuleList(
                    [
                        layer,  # 重复使用layer变量中的LayerNorm对象
                        layer,
                        layer,
                    ]
                )

            def forward(self, input_):
                # 遍历ModuleList中的每个LayerNorm层对象，对输入数据进行处理
                for i in range(len(self.norms)):
                    output = self.norms[i](input_)
                    input_ = output
                # 返回最后一个LayerNorm层处理后的输出
                return output

        # 创建MyModule实例
        mod = MyModule()
        # 创建输入数据
        input_ = torch.randn(N, C, H, W)

        # 导出模型，并在严格模式下进行反序列化
        ep_strict = export(copy.deepcopy(mod), (input_,), strict=True)
        umod = unflatten(ep_strict)
        # 断言反序列化后的模型输出与原模型在给定输入上的输出结果接近
        self.assertTrue(torch.allclose(umod(input_), mod(input_)))

        # 导出模型，并在非严格模式下进行反序列化
        ep_non_strict = export(copy.deepcopy(mod), (input_,), strict=False)
        umod = unflatten(ep_non_strict)
        # 断言反序列化后的模型输出与原模型在给定输入上的输出结果接近
        self.assertTrue(torch.allclose(umod(input_), mod(input_)))
    def test_simple_alias(self):
        # handle weight sharing, check tensor ids after unflattening
        
        # 定义一个简单的神经网络模型
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个参数化的随机偏置项
                self.bias = torch.nn.Parameter(torch.randn(4))
                # 创建一个线性层模块
                self.m = torch.nn.Linear(4, 4)
                # 将线性层的偏置项设为之前创建的偏置项
                self.m.bias = self.bias

            def forward(self, x):
                # 模型的前向传播，返回线性层加上偏置项的结果
                return self.m(x) + self.bias

        # 实例化模型
        m = Foo()
        # 创建输入数据
        inps = (torch.randn(4, 4),)
        # 导出模型的状态
        ep = export(m, inps)
        # 将导出的状态解压缩成模型
        unep = unflatten(ep)
        # 断言两个偏置项的内存地址相同
        self.assertTrue(id(unep.m.bias) == id(unep.bias))

        # 处理一个偏置项未被使用的别名情况
        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个参数化的随机偏置项
                self.bias = torch.nn.Parameter(torch.randn(4))
                # 创建一个线性层模块
                self.m = torch.nn.Linear(4, 4)
                # 将线性层的偏置项设为之前创建的偏置项
                self.m.bias = (
                    self.bias
                )  # self.bias 未被使用，但其别名关系应当被处理

            def forward(self, x):
                # 模型的前向传播，返回线性层的结果
                return self.m(x)

        # 实例化模型
        m = Foo()
        # 创建输入数据
        inps = (torch.randn(4, 4),)
        # 导出模型的状态
        ep = export(m, inps)
        # 将导出的状态解压缩成模型
        unep = unflatten(ep)
        # 断言解压缩后的模型与原始模型在给定输入下的输出结果相似
        self.assertTrue(torch.allclose(unep(*inps), m(*inps)))

    def test_attr_as_submod_input(self):
        # 定义一个简单的层类，用于模型中的操作
        class layer(torch.nn.Module):
            def forward(self, x, const) -> torch.Tensor:
                # 返回输入数据和常量的和
                return x + const

        # 定义一个包含层模块的模型类
        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 注册一个常量缓存
                self.register_buffer("const", torch.ones(4, 8))
                # 创建包含两个层模块的模型列表
                self.layers = torch.nn.ModuleList([layer() for _ in range(2)])

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 对每个层模块进行循环遍历，并对输入数据进行操作
                for layer in self.layers:
                    x = layer(x, self.const)
                return x

        # 实例化模型
        mod = M()
        # 创建输入数据
        x = torch.randn(4, 8)
        # 导出模型的状态
        ep = export(mod, (x,))
        # 将导出的状态解压缩成模型
        unflattened = unflatten(ep)
        # 使用测试工具函数，断言解压缩后的模型在给定输入下的输出结果与原始模型相似
        torch.testing.assert_close(unflattened(x), mod(x))
    def test_dedup_sym_size(self):
        # 在这里，sym_size 和 floordiv 变量在三个子图（顶层、m1、m2）中被使用，
        # 但是在初始导出图中只创建了一份 sym_size 的副本。
        # 对于 m1，sym_size 和 floordiv 应被复制以重新计算，因为我们保留了调用签名，
        # 但是对于 m2，floordiv 应该作为占位符传入。
        # 测试确保这一点被保留，并且未展开的模块可以正确运行。
        
        class M1(torch.nn.Module):
            def forward(self, x, y):
                d = x.size(0) // 2
                return y[:d]

        class M2(torch.nn.Module):
            def forward(self, x, y):
                d = x.size(0) // 2
                return y[:d]

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m1 = M1()
                self.m2 = M2()

            def forward(self, x, y):
                d = x.size(0) // 2
                m1_res = self.m1(x, y)
                m2_res = self.m2(x, y)
                return y[d:] + m1_res + m2_res
        
        # 输入数据
        inputs = (torch.ones(10), torch.ones(10))
        # 定义一个维度对象
        d_ = torch.export.Dim("foo", max=2048)
        # 计算实际维度值
        d = 2 * d_
        # 导出模型
        ep = torch.export.export(
            M(),
            inputs,
            dynamic_shapes=((d,), (d,)),
            strict=False,
            preserve_module_call_signature=("m1",),
        )
        # 对导出的图进行展开
        unflat = unflatten(ep)
        # 在未展开的模块上调用输入数据
        unflat(*inputs)

        # 定义一个函数，用于统计图中 torch.ops.aten.sym_size.int 函数调用的次数
        fn_count_sym_size = lambda graph: [node.target for node in graph.nodes].count(
            torch.ops.aten.sym_size.int
        )
        # 断言：导出的整体图中 sym_size 函数调用次数为 1
        self.assertEqual(fn_count_sym_size(unflat.graph), 1)
        # 断言：在 m1 模块的图中 sym_size 函数调用次数为 1
        self.assertEqual(fn_count_sym_size(unflat.m1.graph), 1)
        # 断言：在 m2 模块的图中 sym_size 函数调用次数为 0
        self.assertEqual(fn_count_sym_size(unflat.m2.graph), 0)
# 如果这个脚本被直接执行（而不是被导入作为模块），则运行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```