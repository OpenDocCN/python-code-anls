# `.\pytorch\test\higher_order_ops\test_with_effects.py`

```
# 导入单元测试模块
import unittest
# 导入双向队列数据结构
from collections import deque
# 导入偏函数功能
from functools import partial
# 导入列表类型提示
from typing import List

# 导入PyTorch核心库
import torch
# 导入私有的动态运行时模块
import torch._dynamo
# 导入私有的Functorch模块
import torch._functorch
# 导入私有的感应器模块
import torch._inductor
# 导入私有的感应器分解模块
import torch._inductor.decomposition
# 从torch._functorch.aot_autograd模块导入aot_export_module函数
from torch._functorch.aot_autograd import aot_export_module
# 从torch._higher_order_ops.effects模块导入with_effects函数
from torch._higher_order_ops.effects import with_effects
# 从torch._higher_order_ops.torchbind模块导入enable_torchbind_tracing函数
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
# 从torch.fx.experimental.proxy_tensor模块导入make_fx函数
from torch.fx.experimental.proxy_tensor import make_fx
# 从torch.testing模块导入FileCheck类
from torch.testing import FileCheck
# 从torch.testing._internal.common_cuda模块导入_cuda版本检查函数和SM80OrLater类
from torch.testing._internal.common_cuda import _get_torch_cuda_version, SM80OrLater
# 从torch.testing._internal.common_quantization模块导入skipIfNoDynamoSupport函数
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
# 从torch.testing._internal.common_utils模块导入一系列常用测试参数和函数
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    TEST_CUDA,
    TEST_WITH_ROCM,
    TestCase,
)
# 从torch.testing._internal.torchbind_impls模块导入初始化torchbind实现的函数
from torch.testing._internal.torchbind_impls import init_torchbind_implementations
# 从torch.utils.hooks模块导入可移除句柄类RemovableHandle

# 禁止pylint检查TCH001错误（不推荐使用torchbind）
from torch.utils.hooks import RemovableHandle  # noqa: TCH001


# 使用unittest框架的装饰器跳过测试，如果动态运行时不支持
@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't support")
class TestWithEffects(TestCase):
    # 测试初始化方法
    def setUp(self):
        # 初始化torchbind实现
        init_torchbind_implementations()

    # 测试打印功能
    def test_print(self):
        # 定义一个简单的Module子类M
        class M(torch.nn.Module):
            # 定义前向传播方法
            def forward(self, x):
                # 在前向传播过程中调用_aten命名空间下的_print函数打印字符串"moo"
                torch.ops.aten._print("moo")
                # 对输入张量x执行加法操作
                res = x + x
                # 再次调用_aten命名空间下的_print函数打印字符串"moo"
                torch.ops.aten._print("moo")
                # 返回结果作为一个元组
                return (res,)

        # 创建一个输入张量的元组
        inputs = (torch.randn(3),)

        # 使用make_fx函数将Module子类M转换为GraphModule对象gm
        gm = make_fx(M())(*inputs)
        # 使用FileCheck对象检查生成的图中_aten._print.default出现的次数是否为2
        FileCheck().check_count("torch.ops.aten._print.default", 2, exactly=True).run(
            gm.code
        )

        # 使用aot_export_module函数将Module子类M和输入元组导出为GraphModule对象gm和GraphStructured对象gs
        gm, gs = aot_export_module(M(), inputs, trace_joint=False)
        # 断言生成的gm.code与预期的前向传播函数字符串匹配
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops.aten._print.default, 'moo');  arg0_1 = None
    getitem = with_effects[0];  with_effects = None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    return (getitem_2, add)""",
        )
        # 断言生成的gs.input_tokens和gs.output_tokens长度为1
        self.assertEqual(len(gs.input_tokens), 1)
        self.assertEqual(len(gs.output_tokens), 1)

        # 使用torch._functorch.config.patch方法配置解除unlift_effect_tokens标志为True
        with torch._functorch.config.patch(unlift_effect_tokens=True):
            # 再次使用aot_export_module函数将Module子类M和输入元组导出为GraphModule对象gm和GraphStructured对象gs
            gm, gs = aot_export_module(M(), inputs, trace_joint=False)
            # 断言生成的gm.code与预期的前向传播函数字符串匹配
            self.assertExpectedInline(
                str(gm.code).strip(),
                """\
def forward(self, arg1_1):
    _make_token_default = torch.ops.prims._make_token.default()
    with_effects = torch._higher_order_ops.effects.with_effects(_make_token_default, torch.ops.aten._print.default, 'moo');  _make_token_default = None""",
            )
    # 获取 with_effects 列表的第一个元素，并将 with_effects 设为 None
    getitem = with_effects[0];  with_effects = None
    # 使用 torch.ops.aten.add.Tensor 方法进行张量相加操作，并将 arg1_1 设为 None
    add = torch.ops.aten.add.Tensor(arg1_1, arg1_1);  arg1_1 = None
    # 使用 torch._higher_order_ops.effects.with_effects 方法调用 getitem、torch.ops.aten._print.default 和 'moo' 参数，将结果赋给 with_effects_1，并将 getitem 设为 None
    with_effects_1 = torch._higher_order_ops.effects.with_effects(getitem, torch.ops.aten._print.default, 'moo');  getitem = None
    # 获取 with_effects_1 列表的第一个元素，并将 with_effects_1 设为 None
    getitem_2 = with_effects_1[0];  with_effects_1 = None
    # 使用 torch.ops.prims._sink_tokens.default 方法调用 (getitem_2,) 元组作为参数，并将 getitem_2 设为 None
    _sink_tokens_default = torch.ops.prims._sink_tokens.default((getitem_2,));  getitem_2 = None
    # 返回包含 add 结果的元组
    return (add,)""",  # noqa: B950
            )

    def test_torchbind_custom_op(self):
        # 定义一个继承自 torch.nn.Module 的类 M
        class M(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 创建一个 torch.classes._TorchScriptTesting._Foo 类的实例，并将其赋给 self.attr
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            # 前向传播方法
            def forward(self, x):
                # 返回一个元组，包含 x 和 torch.ops._TorchScriptTesting.takes_foo(self.attr, x) 的和
                return (x + torch.ops._TorchScriptTesting.takes_foo(self.attr, x),)

        # 启用 TorchBind 跟踪
        with enable_torchbind_tracing():
            # 使用 aot_export_module 方法导出类 M 的 Ahead-Of-Time 编译模块，trace_joint 参数设为 False
            gm, gs = aot_export_module(M(), (torch.ones(2, 3),), trace_joint=False)

        # 断言生成的模块代码符合预期的内联字符串
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, arg0_1, arg1_1):
    # 将 self._torchbind_obj0 赋值给 _torchbind_obj0
    _torchbind_obj0 = self._torchbind_obj0
    # 调用 torch._higher_order_ops.effects.with_effects 函数，将参数传递给它，并赋值给 with_effects
    with_effects = torch._higher_order_ops.effects.with_effects(arg0_1, torch.ops._TorchScriptTesting.takes_foo.default, _torchbind_obj0, arg1_1);  arg0_1 = _torchbind_obj0 = None
    # 获取 with_effects 中的第一个元素，并赋值给 getitem
    getitem = with_effects[0]
    # 获取 with_effects 中的第二个元素，并赋值给 getitem_1；清除 with_effects 引用
    getitem_1 = with_effects[1];  with_effects = None
    # 调用 torch.ops.aten.add.Tensor 函数，将 arg1_1 和 getitem_1 作为参数，结果赋值给 add；清除 arg1_1 和 getitem_1 引用
    add = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
    # 返回包含 getitem 和 add 的元组
    return (getitem, add)



def test_print_with_buffer_mutations(self):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 在 Module 中注册名为 "buf" 的缓冲区，并初始化为包含三个元素的全一张量
            self.register_buffer("buf", torch.ones(3))

        def forward(self, x):
            # 打印输出 "moo"
            torch.ops.aten._print("moo")
            # 计算输入 x 与自身的和，结果赋值给 res
            res = x + x
            # 将 res 加到 self.buf 上，并更新 self.buf
            self.buf.add_(res)
            # 再次计算 self.buf 与 x 的和，结果赋值给 res
            res = self.buf + x
            # 再次打印输出 "moo"
            torch.ops.aten._print("moo")
            # 返回包含 res 的元组
            return (res,)

    inputs = (torch.randn(3),)

    # 使用 aot_export_module 函数导出 Module M 的 TorchScript 表示，并进行断言比较
    gm, gs = aot_export_module(M(), inputs, trace_joint=False)
    # 断言输入 tokens 的数量为 1
    self.assertEqual(len(gs.input_tokens), 1)
    # 断言输出 tokens 的数量为 1
    self.assertEqual(len(gs.output_tokens), 1)
    # 断言要变异的缓冲区数量为 1
    self.assertEqual(len(gs.buffers_to_mutate), 1)



def test_print_with_input_mutations(self):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # 打印输出 "moo"
            torch.ops.aten._print("moo")
            # 计算输入 x 与自身的和，结果赋值给 res
            res = x + x
            # 将 res 加到 x 上，并更新 x
            x.add_(res)
            # 再次计算 x 与自身的和，结果赋值给 res
            res = x + x
            # 再次打印输出 "moo"
            torch.ops.aten._print("moo")
            # 返回包含 res 的元组
            return (res,)

    inputs = (torch.randn(3),)

    # 使用 aot_export_module 函数导出 Module M 的 TorchScript 表示，并进行断言比较
    gm, gs = aot_export_module(M(), inputs, trace_joint=False)
    # 断言输入 tokens 的数量为 1
    self.assertEqual(len(gs.input_tokens), 1)
    # 断言输出 tokens 的数量为 1
    self.assertEqual(len(gs.output_tokens), 1)
    # 断言要变异的用户输入数量为 1
    self.assertEqual(len(gs.user_inputs_to_mutate), 1)
    def test_alias_op(self):
        # 定义一个内部函数 f，接受 token 和 x 作为参数
        def f(token, x):
            # 调用 with_effects 函数，使用 torch.ops.aten.absolute_.default 操作处理 x
            token, out = with_effects(token, torch.ops.aten.absolute_.default, x)
            # 返回处理后的 token 和 out
            return token, out

        # 使用 assertRaisesRegex 上下文管理器检查 AssertionError 异常，确保其包含指定的错误信息
        with self.assertRaisesRegex(
            AssertionError, r"Ops with aliasing is not supported"
        ):
            # 调用 make_fx 函数，将 f 函数编译为一个效果函数，并传入两个张量作为输入
            make_fx(f)(torch.tensor([]), torch.tensor(4))

    def test_compile_aot_eager(self):
        # 定义一个函数 f，接受 x 作为参数
        def f(x):
            # 打印 "moo" 字符串
            torch.ops.aten._print("moo")
            # 计算 x + x，并将结果赋给 res
            res = x + x
            # 再次打印 "moo" 字符串
            torch.ops.aten._print("moo")
            # 返回计算结果 res
            return res

        # 定义输入数据 inputs，包含一个形状为 (2, 3) 的随机张量
        inputs = (torch.randn(2, 3),)

        # 使用 torch.compile 函数编译 f 函数，选择 "aot_eager" 后端，并传入 inputs 作为参数
        res = torch.compile(f, backend="aot_eager")(*inputs)
        # 断言编译结果 res 与直接调用 f 函数得到的结果在数值上相等
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "triton")
    def test_compile_inductor(self):
        # 定义一个函数 f，接受 x 作为参数
        def f(x):
            # 打印 "moo" 字符串
            torch.ops.aten._print("moo")
            # 计算 x + x，并将结果赋给 res
            res = x + x
            # 再次打印 "moo" 字符串
            torch.ops.aten._print("moo")
            # 返回计算结果 res
            return res

        # 定义输入数据 inputs，包含一个形状为 (2, 3) 的随机张量
        inputs = (torch.randn(2, 3),)

        # 使用 torch.compile 函数编译 f 函数，选择 "inductor" 后端，并传入 inputs 作为参数
        res = torch.compile(f, backend="inductor")(*inputs)
        # 断言编译结果 res 与直接调用 f 函数得到的结果在数值上相等
        self.assertTrue(torch.allclose(res, f(*inputs)))

    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @skipIfNoDynamoSupport
    def test_compile_inductor_external_op_return_none(self):
        # 进入 torch.library._scoped_library 上下文，定义名为 "mylib" 的库，类型为 "FRAGMENT"
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            # 定义 "mylib::inplace_add" 操作的签名和文档，使用 lib 库
            torch.library.define(
                "mylib::inplace_add",
                "(Tensor input, Tensor(a!) output) -> ()",
                lib=lib,
            )

            # 定义 inplace_add 函数，接受 torch.Tensor 类型的 input 和 output 参数，返回 None
            def inplace_add(input: torch.Tensor, output: torch.Tensor) -> None:
                # 断言 input 和 output 张量的设备相同
                assert input.device == output.device
                # 在 output 上执行 in-place 加法操作
                output.add_(input)

            # 将 inplace_add 函数实现为 "inplace_add" 操作，使用 "CompositeExplicitAutograd" 实现方式
            lib.impl("inplace_add", inplace_add, "CompositeExplicitAutograd")

            # 定义一个函数 f，接受 x 作为参数
            def f(x):
                # 创建一个形状为 (3,) 的空张量 out
                out = torch.empty(3)
                # 将 out 张量的值填充为 0
                out = torch.zeros_like(out)
                # 调用 torch.ops.mylib.inplace_add 函数，将 x 添加到 out 中
                torch.ops.mylib.inplace_add(x, out)
                # 返回 out 张量
                return out

            # 定义输入数据 inputs，包含一个形状为 (3,) 的随机张量
            inputs = (torch.randn(3),)

            # 使用 torch.compile 函数编译 f 函数，选择 "inductor" 后端，并传入 inputs 作为参数
            res = torch.compile(f, backend="inductor")(*inputs)
            # 断言编译结果 res 与直接调用 f 函数得到的结果在数值上相等
            self.assertTrue(torch.allclose(res, f(*inputs)))

    def test_compile_aot_eager_requires_grad(self):
        # 定义一个函数 f，接受 x 作为参数
        def f(x):
            # 打印 "moo" 字符串
            torch.ops.aten._print("moo")
            # 计算 x + x，并将结果赋给 res
            res = x + x
            # 再次打印 "moo" 字符串
            torch.ops.aten._print("moo")
            # 返回计算结果 res
            return res

        # 定义输入数据 inputs，包含一个形状为 (2, 3)，并要求梯度的随机张量
        inputs = (torch.randn(2, 3, requires_grad=True),)

        # 使用 torch.compile 函数编译 f 函数，选择 "aot_eager" 后端，并传入 inputs 作为参数
        res = torch.compile(f, backend="aot_eager")(*inputs)
        # 断言编译结果 res 与直接调用 f 函数得到的结果在数值上相等
        self.assertTrue(torch.allclose(res, f(*inputs)))

        # 对编译结果 res 求和，并执行反向传播
        res.sum().backward()

    @unittest.skipIf(IS_WINDOWS, "triton")
    @unittest.skipIf(TEST_WITH_ROCM, "triton")
    @unittest.skipIf(not SM80OrLater, "triton")
    @unittest.skipIf(_get_torch_cuda_version() >= (11, 7), "triton")
    @unittest.skipIf(not TEST_CUDA, "triton")
    @skipIfNoDynamoSupport
# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则执行以下代码块
if __name__ == "__main__":
    # 调用运行测试函数
    run_tests()
```