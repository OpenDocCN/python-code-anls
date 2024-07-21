# `.\pytorch\test\dynamo\test_activation_checkpointing.py`

```
# Owner(s): ["module: dynamo"]
# 导入标准库模块和第三方库模块
import copy  # 导入深拷贝函数
import functools  # 导入函数工具库，提供了创建偏函数的功能
import math  # 导入数学函数库
import unittest  # 导入单元测试框架，此处禁止 F811 错误提示
from importlib import import_module  # 导入动态导入模块的函数

import torch  # 导入 PyTorch 库
import torch._dynamo.config  # 导入 PyTorch 内部配置模块

import torch._dynamo.test_case  # 导入 PyTorch 内部测试用例模块
import torch._functorch.config  # 导入 PyTorch Functorch 配置模块
import torch.distributed as dist  # 导入分布式训练模块
import torch.nn as nn  # 导入神经网络模块
import torch.utils.checkpoint  # 导入 PyTorch 检查点工具

from functorch.compile import min_cut_rematerialization_partition  # 从 Functorch 编译模块导入函数
from torch._dynamo.backends.common import aot_autograd  # 从 PyTorch Dynamo 后端公共模块导入函数
from torch._dynamo.testing import CompileCounterWithBackend  # 从 PyTorch Dynamo 测试模块导入类
from torch._higher_order_ops.wrap import tag_activation_checkpoint  # 从 PyTorch 高阶运算封装模块导入函数
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfRocm  # 从 PyTorch 内部测试工具模块导入常量和装饰器
from torch.testing._internal.inductor_utils import HAS_CUDA  # 从 PyTorch 内部归纳工具模块导入常量
from torch.testing._internal.two_tensor import TwoTensor  # 从 PyTorch 内部测试工具模块导入类
from torch.utils.checkpoint import (
    checkpoint,  # 导入 PyTorch 检查点函数
    CheckpointPolicy,  # 导入 PyTorch 检查点策略类
    create_selective_checkpoint_contexts,  # 导入创建选择性检查点上下文的函数
)

requires_cuda = unittest.skipUnless(HAS_CUDA, "requires cuda")  # 如果没有 CUDA 支持则跳过测试装饰器
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)  # 分布式环境下才运行的测试装饰器部分应用函数


def checkpoint_wrapper(fn):
    def inner(*args):
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)
    return inner
    # 定义一个检查点装饰器函数，使用 PyTorch 的检查点工具进行函数包装


def count_ops(
    gm, args, freq=None, freq_ge=None, op=None, freqs=None, freqs_ge=None, ops=None
):
    def match_rng_op(node, op):
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            if node.name == "run_and_save_rng_state":
                return node.args[0] == op
            elif node.name == "run_with_rng_state":
                return node.args[1] == op
        return False

    # assert ((freq or freq_ge) and op) or ((freqs or freqs_ge) and ops)
    if op is not None:
        assert not isinstance(op, list)
        ops = [op]
    if freq is not None:
        freqs = [freq]
    if freq_ge is not None:
        freqs_ge = [freq_ge]
    if freqs:
        for op, freq in zip(ops, freqs):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            err_msg = f"In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
            assert actual_count == freq, err_msg
    else:
        assert freqs_ge is not None
        for op, freq_ge in zip(ops, freqs_ge):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            assert (
                actual_count >= freq_ge
            ), f"In graph {gm}, expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}."
    return gm
    # 统计图模型中操作节点出现次数的函数


class _InvalidContext:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    # 定义一个无效的上下文管理器类，实现 __enter__ 和 __exit__ 方法


def _invalid_context_gen():
    # 返回两个 _InvalidContext() 实例的元组
    return _InvalidContext(), _InvalidContext()
# 在给定的计算图管理器 gm 中查找第一个满足特定条件 func 的节点
def find_first_node(gm, func):
    # 遍历图管理器 gm 中的所有节点
    for node in gm.graph.nodes:
        # 如果节点的目标（target）属性等于 func，则返回该节点
        if node.target is func:
            return node
    # 如果未找到匹配的节点，则返回 None
    return None


# 计算给定计算图管理器 gm 中所有操作为 "call" 的节点数量
def op_count(gm):
    # 初始化结果变量为 0
    result = 0
    # 遍历图管理器 gm 中的所有节点
    for node in gm.graph.nodes:
        # 如果节点的操作（op）包含字符串 "call"，则结果加一
        if "call" in node.op:
            result += 1
    # 返回操作为 "call" 的节点数量
    return result


# 返回一个自定义策略函数，根据给定的 no_recompute_list 和 must_recompute_list 来确定检查点策略
def _get_custom_policy(no_recompute_list=None, must_recompute_list=None):
    def _custom_policy(ctx, func, *args, **kwargs):
        # 如果给定的 func 存在于 no_recompute_list 中，则强制保存检查点
        if no_recompute_list is not None and func in no_recompute_list:
            return CheckpointPolicy.MUST_SAVE
        # 如果给定的 func 存在于 must_recompute_list 中，则强制重新计算
        if must_recompute_list is not None and func in must_recompute_list:
            return CheckpointPolicy.MUST_RECOMPUTE
        # 否则，默认为优先重新计算
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    # 返回内部定义的自定义策略函数
    return _custom_policy


# 测试类，用于测试激活标签检查点功能
class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):
    # 验证函数，用于验证通过 torch.compile 编译的函数与原始函数在给定参数下的结果一致性
    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        # 克隆参数以确保不改变原始参数
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))

        # 设置随机种子以确保每次结果一致
        torch.manual_seed(0)
        # 调用原始函数并计算其期望值
        expected = fn(*args)
        expected.sum().backward()

        # 再次设置随机种子
        torch.manual_seed(0)
        # 使用 torch.compile 编译函数并计算结果
        result = torch.compile(fn, fullgraph=fullgraph, backend=backend)(*cloned_args)
        result.sum().backward()

        # 如果不跳过检查，则比较编译函数和原始函数的输出结果
        if not skip_check:
            self.assertEqual(
                result,
                expected,
                msg="Output mismatch between torch.compile and eager versions",
            )
            # 检查每个参数的梯度是否一致
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(
                    arg.grad,
                    cloned_arg.grad,
                    msg="Gradient mismatch between torch.compile and eager versions",
                )

    # 比较原始函数和检查点函数在给定参数下的结果一致性
    def _compare_orig_and_checkpointed_fns(
        self, orig_fn, checkpointed_fn, *args, fullgraph=True
    ):
        # 定义一个测试函数，用于验证原始版本和检查点版本的函数在 torch.compile 下产生相同的输出和梯度。

        # 运行原始版本
        cloned_args_orig_fn = []
        for arg in args:
            # 克隆参数并设置 requires_grad，以保留梯度信息
            cloned_args_orig_fn.append(
                arg.clone().detach().requires_grad_(arg.requires_grad)
            )
        torch.manual_seed(0)
        # 使用 torch.compile 编译原始版本的函数
        compiled_orig_fn = torch.compile(
            orig_fn, fullgraph=fullgraph, backend="inductor"
        )
        # 执行编译后的原始版本函数，并计算结果的梯度
        result_orig_fn = compiled_orig_fn(*cloned_args_orig_fn)
        result_orig_fn.sum().backward()

        # 运行检查点版本
        cloned_args_checkpointed_fn = []
        for arg in args:
            # 克隆参数并设置 requires_grad，以保留梯度信息
            cloned_args_checkpointed_fn.append(
                arg.clone().detach().requires_grad_(arg.requires_grad)
            )
        torch.manual_seed(0)
        # 使用 torch.compile 编译检查点版本的函数
        compiled_checkpointed_fn = torch.compile(
            checkpointed_fn, fullgraph=fullgraph, backend="inductor"
        )
        # 执行编译后的检查点版本函数，并计算结果的梯度
        result_checkpointed_fn = compiled_checkpointed_fn(*cloned_args_checkpointed_fn)
        result_checkpointed_fn.sum().backward()

        # 检查输出和梯度是否相等
        self.assertEqual(
            result_orig_fn,
            result_checkpointed_fn,
            msg="原始版本和检查点版本的函数输出不匹配",
        )
        # 检查每个参数的梯度是否相等
        for cloned_arg_orig_fn, cloned_arg_checkpointed_fn in zip(
            cloned_args_orig_fn, cloned_args_checkpointed_fn
        ):
            self.assertEqual(
                cloned_arg_orig_fn.grad,
                cloned_arg_checkpointed_fn.grad,
                msg="原始版本和检查点版本的函数梯度不匹配",
            )

    @requires_cuda
    def test_tags_function(self):
        # 定义一个测试 CUDA 函数
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # 使用 torch.utils.checkpoint.checkpoint 来创建一个函数 fn
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        # 创建 CUDA 设备上的随机输入张量，并设置 requires_grad 为 True
        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        # 配置前向计算和反向计算的编译器
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # 在反向计算中重新计算 mm
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 调用 _validate 函数来验证 fn 函数的输出和梯度
        self._validate(fn, backend, x, y)

    @requires_cuda
    def test_tags_function_via_global_checkpoint(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))
        # 定义一个简单的神经网络层次计算函数gn，返回输入张量经过sigmoid和矩阵乘法后的结果

        def fn(x, y):
            # This goes through VariableBuilder
            # 使用checkpoint函数对gn函数进行优化计算，保留中间计算结果以减少内存消耗
            return checkpoint(gn, torch.sin(x), y, use_reentrant=True)
        
        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        # 前向计算编译器，用于计算操作频率和操作类型
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        # 反向计算编译器，用于计算操作频率和操作类型，其中mm操作在反向传播中重新计算
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 使用aot_autograd后端进行自动微分计算
        self._validate(fn, backend, x, y)

    @requires_cuda
    def test_tags_function_with_kwargs(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))
        # 定义一个简单的神经网络层次计算函数gn，返回输入张量经过sigmoid和矩阵乘法后的结果

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )
        # 使用checkpoint函数对gn函数进行优化计算，保留中间计算结果，但不保留随机数生成器状态

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        y = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        # 前向计算编译器，用于计算操作频率和操作类型
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        # 反向计算编译器，用于计算操作频率和操作类型，其中mm操作在反向传播中重新计算
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 使用aot_autograd后端进行自动微分计算
        self._validate(fn, backend, x, y)

    @requires_cuda
    def test_tags_sequential_layers(self):
        def gn(x):
            x = x.cos()
            for _ in range(3):
                x = torch.mm(x, x)
            x = x.cos()
            return x
        # 定义一个简单的神经网络层次计算函数gn，进行多次余弦和矩阵乘法操作后返回结果

        def fn(x):
            x = torch.utils.checkpoint.checkpoint(gn, x)
            x = torch.utils.checkpoint.checkpoint(gn, x)
            return x
        # 使用checkpoint函数对gn函数进行优化计算，保留中间计算结果

        x = torch.randn(4, 4, device="cuda", requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=6, op=torch.ops.aten.mm.default)
        # 前向计算编译器，用于计算操作频率和操作类型
        bw_compiler = functools.partial(
            count_ops,
            freqs=[2, 18],
            ops=[torch.ops.aten.cos.default, torch.ops.aten.mm.default],
        )  # mm recomputed in the bwd
        # 反向计算编译器，用于计算不同操作的频率和类型，其中包括余弦和mm操作
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 使用aot_autograd后端进行自动微分计算
        self._validate(fn, backend, x)
    @requires_cuda
    @torch._inductor.config.patch(fallback_random=True)
    # 标记为需要 CUDA 支持的测试，并设置随机补丁以提供回退支持

    def test_tags_multiple_checkpoints(self):
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))
        # 定义一个简单的函数 gn，执行输入张量的矩阵乘法和 sigmoid 操作

        def fn(x, y):
            x = torch.sin(x)
            # 对输入张量 x 应用正弦函数
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            # 使用检查点技术执行函数 gn，传入 x, y，并开启可重入模式
            x = torch.sin(z)
            # 对结果 z 再次应用正弦函数
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            # 使用检查点技术再次执行函数 gn，传入 x, y，并开启可重入模式
            return z
            # 返回最终结果 z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)
        # 创建一个随机的 CUDA 张量 x，用于测试，需要梯度计算
        y = torch.randn(4, 4, device="cuda", requires_grad=True)
        # 创建一个随机的 CUDA 张量 y，用于测试，需要梯度计算

        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # 创建一个部分函数应用，用于前向计算，统计 mm 操作的频率
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )  # mm 在反向传播中重新计算
        # 创建一个部分函数应用，用于反向计算，统计 mm 操作的频率，这里频率设置更高，因为在反向传播中会重新计算 mm
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 使用自动编译后向传播的 AOT 自动微分，传入前向和反向编译器
        self._validate(fn, backend, x, y)
        # 调用测试辅助函数 _validate，验证函数 fn 在给定 backend、x 和 y 下的正确性

    @requires_cuda
    # 标记为需要 CUDA 支持的测试

    def test_tags_module(self):
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                # 创建一个线性层，输入和输出维度都是 10

            def forward(self, x):
                return torch.sigmoid(self.linear(x))
                # 前向传播函数，对输入 x 应用线性层和 sigmoid 操作

        mod = MockModule().cuda()
        # 创建 MockModule 的实例并移至 CUDA

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )
            # 使用检查点技术，传入 mod、sin(x) 和开启可重入模式

        x = torch.randn(10, 10, device="cuda", requires_grad=True)
        # 创建一个随机的 CUDA 张量 x，用于测试，需要梯度计算

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        # 创建一个部分函数应用，用于前向计算，统计 sigmoid 操作的频率
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        # 创建一个部分函数应用，用于反向计算，统计 sigmoid 操作的频率
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # 使用自动编译后向传播的 AOT 自动微分，传入前向和反向编译器
        self._validate(fn, backend, x)
        # 调用测试辅助函数 _validate，验证函数 fn 在给定 backend 和 x 下的正确性

    @requires_cuda
    # 标记为需要 CUDA 支持的测试

    def test_tags_decomps(self):
        # Ensures that tags are passed on through decompositions as well
        # 确保标签也通过分解传递

        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                # 创建一个线性层，输入和输出维度都是 10

            def forward(self, x):
                return torch.nn.functional.gelu(self.linear(x))
                # 前向传播函数，对输入 x 应用线性层和 gelu 操作

        mod = MockModule().cuda()
        # 创建 MockModule 的实例并移至 CUDA

        def fn(x):
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )
            # 使用检查点技术，传入 mod、sin(x) 和开启可重入模式

        x = torch.randn(10, 10, device="cuda", requires_grad=True)
        # 创建一个随机的 CUDA 张量 x，用于测试，需要梯度计算

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        # 创建一个部分函数应用，用于前向计算，统计 erf 操作的频率
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        # 创建一个部分函数应用，用于反向计算，统计 erf 操作的频率
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
        )
        # 使用自动编译后向传播的 AOT 自动微分，传入前向和反向编译器以及分解表
        self._validate(fn, backend, x)
        # 调用测试辅助函数 _validate，验证函数 fn 在给定 backend 和 x 下的正确性
    def test_tags_recomputed_rand(self):
        def gn(x, y):
            return torch.sigmoid(torch.rand_like(x) * y) * x
        def fn(x, y):
            x = torch.sin(x)  # 对输入张量 x 求正弦函数
            x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)  # 使用 checkpoint 进行梯度检查点，减少内存占用
            x = torch.sin(x)  # 再次对张量 x 求正弦函数
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)  # 使用 checkpoint 进行梯度检查点
            return z

        x = torch.randn(4, 4, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 x，需要计算梯度
        y = torch.randn(4, 4, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 y，需要计算梯度

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # fw_compiler 是一个部分应用了 count_ops 函数的对象，用于前向传播的编译器，统计操作频率为 2，操作为 torch.mm 的默认操作
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # 在反向传播中重新计算 mm 操作
        # bw_compiler 是一个部分应用了 count_ops 函数的对象，用于反向传播的编译器，统计操作频率为 6，操作为 torch.mm 的默认操作
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend 是一个指定的后端计算模式，这里是通过 aot_autograd 函数确定的，使用前向编译器和反向编译器
        backend = "inductor"  # 指定后端为 "inductor"
        self._validate(fn, backend, x, y)  # 调用 _validate 方法验证 fn 函数在指定后端上的运行结果

    @requires_cuda
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self):
        def gn(x, y):
            x = torch.mm(x, y)  # 执行矩阵乘法操作
            x = torch.mm(x, y)  # 再次执行矩阵乘法操作
            return x

        def fn(x, y):
            x = torch.sin(x)  # 对输入张量 x 求正弦函数
            x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)  # 使用 checkpoint 进行梯度检查点
            x = torch.sin(x)  # 再次对张量 x 求正弦函数
            # x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return x

        x = torch.randn(4, 4, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 x，需要计算梯度
        y = torch.randn(4, 4, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 y，需要计算梯度

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend = "aot_eager"
        backend = "inductor"  # 指定后端为 "inductor"
        self._validate(fn, backend, x, y)  # 调用 _validate 方法验证 fn 函数在指定后端上的运行结果

    @requires_cuda
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self):
        # Figure out a way to test the number of inductor_random calls
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)  # 创建一个线性层，输入和输出维度都是 10
                self.dropout = torch.nn.Dropout(0.2)  # 创建一个 dropout 层，丢弃概率为 0.2

            def forward(self, x):
                return self.dropout(self.linear(x))  # 在 forward 方法中对输入 x 进行线性变换后再进行 dropout

        mod = MockModule().cuda()  # 将 MockModule 实例移到 CUDA 设备上

        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)  # 使用 checkpoint 运行模型 mod

        x = torch.randn(10, 10, device="cuda", requires_grad=True)  # 在 CUDA 设备上生成随机张量 x，需要计算梯度
        backend = "inductor"  # 指定后端为 "inductor"
        # rand decomps do not have have numerical results as eager
        # 随机分解没有像 eager 模式那样的数值结果
        self._validate(fn, backend, x, skip_check=True)  # 调用 _validate 方法验证 fn 函数在指定后端上的运行结果，跳过检查
    def test_fallback(self):
        # 定义内部函数 gn，用于计算神经网络层次中的 sigmoid 和 cos 函数
        def gn(x, y):
            # 手动触发动态图的断点，用于图优化
            torch._dynamo.graph_break()
            # 计算 x 和 y 的矩阵乘积后经过 sigmoid 函数的结果
            a = torch.sigmoid(torch.matmul(x, y))
            # 再次触发动态图的断点
            torch._dynamo.graph_break()
            # 返回 a 的余弦值
            return torch.cos(a)

        # 定义内部函数 fn，调用 checkpoint 函数来执行 gn 函数
        def fn(x, y):
            # 使用 checkpoint 函数执行 gn 函数，禁止重入
            return torch.cos(checkpoint(gn, torch.sin(x), y, use_reentrant=False))

        # 创建两个 4x4 的随机张量，并标记为需要梯度
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        # 使用 "aot_eager" 后端进行编译计数
        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        # 计算预期结果
        expected = fn(*args)
        # 使用 cnt 后端编译 fn 函数，并传入参数
        result = torch.compile(fn, backend=cnt)(*args)

        # 断言编译结果与预期结果相等
        self.assertEqual(result, expected)

        # 断言动态图帧数为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言操作数为 2
        self.assertEqual(cnt.op_count, 2)
        # 断言编译后的图的数量为 2
        self.assertEqual(len(cnt.graphs), 2)

    @requires_cuda
    def test_kwargs(self):
        # 定义内部函数 gn，根据是否有 z 参数来执行不同的矩阵乘法
        def gn(x, y, z=None):
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

        # 定义内部函数 fn，使用 checkpoint 函数执行 gn 函数，并传入参数 x, y, z
        def fn(x, y, z):
            return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))

        # 创建三个 4x4 的随机张量，并标记为需要梯度
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4, 4, requires_grad=True)
        args = (x, y, z)

        # 使用 "aot_eager" 后端进行编译计数
        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        # 计算预期结果
        expected = fn(*args)
        # 使用 cnt 后端编译 fn 函数，并传入参数
        result = torch.compile(fn, backend=cnt)(*args)

        # 断言编译结果与预期结果相等
        self.assertEqual(result, expected)

        # 断言动态图帧数为 1
        self.assertEqual(cnt.frame_count, 1)
        # 断言编译后的图的数量为 1
        self.assertEqual(len(cnt.graphs), 1)

        # 在第一个编译的图中查找第一个节点，其标记为 activation_checkpoint
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        # 断言该节点的参数数量为 4，包括 checkpoint 和 x, y, z 三个参数
        self.assertEqual(len(wrap_node.args), 4)

        # 获取编译图中 wrap_node.args[0] 所引用的函数，并断言其操作数为 2
        body_function = getattr(cnt.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)

    @requires_cuda
    def test_symints_location(self):
        # 定义内部函数 gn，计算 x 和使用 dropout 后的 y 的矩阵乘积
        def gn(x, y):
            return torch.matmul(x, torch.nn.functional.dropout(y, 0.5))

        # 定义内部函数 fn，使用 checkpoint 函数执行 gn 函数，并传入参数 x, y
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)

        # 使用 "aot_eager" 后端进行编译计数
        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)
        opt_fn = torch.compile(fn, backend=cnt)

        # 创建两个 4x4 的随机张量，并标记为需要梯度
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)

        # 计算预期结果
        expected = fn(*args)
        # 使用 cnt 后端编译 fn 函数，并传入参数
        result = opt_fn(*args)

        # 创建两个 5x5 的随机张量，并标记为需要梯度
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        args = (x, y)

        # 计算预期结果
        expected = fn(*args)
        # 使用 cnt 后端编译 fn 函数，并传入参数
        result = opt_fn(*args)

        # 断言编译结果的形状与预期结果相等
        self.assertEqual(result.shape, expected.shape)
        # 断言动态图帧数为 2
        self.assertEqual(cnt.frame_count, 2)
        # 断言编译后的图的数量为 2
        self.assertEqual(len(cnt.graphs), 2)

        # 在第一个编译的图中查找第一个节点，其标记为 activation_checkpoint
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        # 断言该节点的参数数量为 3，包括 checkpoint 和 x, y 两个参数
        self.assertEqual(len(wrap_node.args), 3)
    # 如果运行环境是 Windows，则跳过测试，因为 torch.compile 在 Windows 上不可用
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    def test_compile_selective_checkpoint_must_recompute(self):
        # 定义一个函数，返回必须重新计算的操作列表上下文
        def context_fn_must_recompute_mm():
            must_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(
                    must_recompute_list=must_recompute_list,
                ),
            )
    
        # 定义一个函数，返回不需要重新计算的操作列表上下文
        def context_fn_no_recompute_mm():
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(
                    no_recompute_list=no_recompute_list,
                ),
            )
    
        # 定义一个测试函数，接受上下文函数和反向编译器作为参数
        def _test(context_fn, bw_compiler):
            # 定义一个简单的计算图函数 gn(x)，其中包含 torch.matmul 和 torch.sigmoid 操作
            def gn(x):
                return torch.sigmoid(torch.matmul(x, x))
    
            # 定义一个检查点函数 fn(x)，使用了 torch.utils.checkpoint.checkpoint 进行检查点操作
            def fn(x):
                return torch.utils.checkpoint.checkpoint(
                    gn,
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                )
    
            # 创建一个形状为 (4, 4) 的随机张量 x，并设置 requires_grad=True
            x = torch.randn(4, 4, requires_grad=True)
    
            # 定义一个前向编译器，部分应用了 count_ops 函数，并指定操作为 torch.ops.aten.mm.default
            fw_compiler = functools.partial(
                count_ops,
                freq=1,
                op=torch.ops.aten.mm.default,
            )
    
            # 创建一个后向编译器，部分应用了 count_ops 函数，并指定操作为 torch.ops.aten.mm.default
            backend = aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=min_cut_rematerialization_partition,
            )
            
            # 运行自定义的验证函数，验证 fn(x) 的结果
            self._validate(fn, backend, x)
    
        # 第一次测试调用，使用必须重新计算的 mm 操作列表和对应的反向编译器设置
        _test(
            context_fn=context_fn_must_recompute_mm,
            bw_compiler=functools.partial(
                count_ops,
                freq=3,  # 1 matmul recompute and 2 bwd mm ops per fwd matmul, so 1 + 2 * 1 = 3)
                op=torch.ops.aten.mm.default,
            ),
        )
        
        # 第二次测试调用，使用不需要重新计算的 mm 操作列表和对应的反向编译器设置
        _test(
            context_fn=context_fn_no_recompute_mm,
            bw_compiler=functools.partial(
                count_ops,
                freq=2,  # 2 bwd mm ops per fwd matmul
                op=torch.ops.aten.mm.default,
            ),
        )
    # 定义一个测试函数，用于验证选择性检查点功能是否正常，确保不会重新计算 GEMM 操作
    def test_compile_selective_checkpoint_must_not_recompute_gemm(self):
        # 定义一个内部函数，返回一个选择性检查点上下文对象列表
        def selective_checkpointing_context_fn():
            # 定义一个不需要重新计算的操作列表，这里只包含 torch 的 mm 默认操作
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            # 调用函数创建选择性检查点上下文对象，使用自定义策略
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        # 定义一个函数 gn，执行矩阵运算和 sigmoid 操作
        def gn(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        # 定义一个函数 fn，使用 Torch 的 checkpoint 函数执行选择性检查点
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        # 创建随机数据张量 x 和 y，要求梯度计算，使用 CUDA 设备
        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        # 创建一个偏函数 fw_compiler，用于计算操作次数，设置操作频率和操作类型
        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        
        # 创建一个偏函数 bw_compiler，用于计算反向操作次数，设置操作频率和操作类型
        bw_compiler = functools.partial(
            count_ops,
            # 我们本来期望这里是 6
            # （2 个矩阵乘法重新计算，每个前向矩阵乘法有 2 个 mm 操作，所以 2 + 2 * 2 = 6）
            # 如果没有启用选择性检查点，我们会得到 6。
            freq=4,
            op=torch.ops.aten.mm.default,
        )

        # 使用自动编译 Torch 的 AOT 自动求导函数
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        # 验证前向函数 fn 和后端是否正确
        self._validate(fn, backend, x, y)

        # 比较原始函数 gn 和检查点函数 fn 的结果
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    # 要求使用 CUDA；如果是在 Windows 系统上，则跳过测试，因为 torch.compile 不支持 Windows
    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    # 定义一个测试方法，用于编译选择性检查点的张量子类
    def test_compile_selective_checkpoint_tensor_subclass(self):
        # 定义一个上下文函数，用于创建选择性检查点上下文
        def selective_checkpointing_context_fn():
            # 不需重新计算的操作列表，例如 torch.mm 的默认实现
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            # 返回使用自定义策略创建的选择性检查点上下文
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        # 定义一个函数 gn，执行张量操作，并返回结果
        def gn(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        # 定义一个函数 fn，使用检查点技术执行张量操作
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        # 生成一个在 CUDA 设备上的随机张量
        rand_tensor = torch.randn(4, 4, requires_grad=True, device="cuda")

        # 创建包含两个相同随机张量的 TwoTensor 实例作为输入
        x = TwoTensor(rand_tensor, rand_tensor.clone())
        y = TwoTensor(rand_tensor.clone(), rand_tensor.clone())

        # 创建一个 functools.partial 对象，用于计算前向传播操作
        fw_compiler = functools.partial(
            count_ops,
            freq=4,
            op=torch.ops.aten.mm.default,
        )

        # 创建一个 functools.partial 对象，用于计算反向传播操作
        bw_compiler = functools.partial(
            count_ops,
            # 在未启用选择性检查点的情况下，我们预期应该是 12
            # (4 个矩阵乘重算，每个前向乘法操作 4 个 mm 操作，因此 4 + 2 * 4 = 12)
            freq=8,
            op=torch.ops.aten.mm.default,
        )

        # 创建一个后端对象，使用 ahead-of-time 自动求导和重编译
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        # 调用 self._validate 方法，验证 fn 函数的正确性
        self._validate(fn, backend, x, y)

        # 调用 self._compare_orig_and_checkpointed_fns 方法，比较原始函数和检查点函数的输出
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    # 标记需要 CUDA 支持，并在 Windows 平台上跳过测试
    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    def test_compile_selective_checkpoint_custom_rule(self):
        # 定义一个测试函数，用于验证自定义规则的选择性检查点功能

        def _get_custom_policy(meta):
            # 定义一个函数，返回一个自定义的选择性检查点策略
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]

            def _custom_policy(mode, func, *args, **kwargs):
                # 定义一个内部函数，实现自定义的选择性检查点策略
                mm_count_key = f"{mode}_mm_count"
                if mm_count_key not in meta:
                    meta[mm_count_key] = 0
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # 保存所有计算操作的输出，除了第二个 mm 操作
                # （即我们提示分区器在反向传播时重新计算第二个 mm 操作）
                return func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] == 2
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            # 定义一个函数，返回使用自定义策略创建的选择性检查点上下文
            meta = {}
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        def gn(x, y):
            # 定义一个函数，实现一个复杂的计算图
            return torch.sigmoid(
                torch.sigmoid(torch.matmul(torch.matmul(x, y) * y, y) * y)
            )

        def fn(x, y):
            # 定义一个函数，使用检查点技术执行计算图计算
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # Q: How do we come to this number 4?
            # A: We have 2 matmuls in the forward pass, each matmul contributes 2 `mm` ops in the backward pass,
            # so we have at least 4 `mm` ops in backward pass. It's "at least" because whether second matmul in
            # the forward pass is recomputed in the backward pass is up to the partitioner to decide.
            freq_ge=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)
    # 定义测试函数 test_compile_selective_checkpoint_partial_ctx_fn，用于测试有选择性的检查点功能
    def test_compile_selective_checkpoint_partial_ctx_fn(self):
        # 定义选择性检查点上下文函数 selective_checkpointing_context_fn，接收不需重新计算列表并返回检查点上下文
        def selective_checkpointing_context_fn(no_recompute_list):
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        # 定义函数 gn，执行 torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y 的计算并返回结果
        def gn(x, y):
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        # 定义函数 fn，使用 checkpoint 来执行函数 gn 的计算
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=functools.partial(
                    selective_checkpointing_context_fn, [torch.ops.aten.mm.default]
                ),
            )

        # 创建随机张量 x 和 y，分别在 CUDA 设备上，用于测试
        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        # 定义 fw_compiler 函数，部分应用 count_ops 函数来计算操作频率
        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )

        # 定义 bw_compiler 函数，部分应用 count_ops 函数来计算反向传播操作频率
        bw_compiler = functools.partial(
            count_ops,
            # 期望的操作次数是 6，
            # （2 次矩阵乘重计算和每次前向矩阵乘法的 2 次 mm 操作，因此 2 + 2 * 2 = 6）
            # 如果没有启用选择性检查点，则会有 6 次操作
            freq=4,
            op=torch.ops.aten.mm.default,
        )

        # 创建后端对象 backend，应用 aot_autograd 函数来设置前向和反向编译器以及分区函数
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        # 调用 self._validate 方法来验证函数 fn 的行为是否符合预期
        self._validate(fn, backend, x, y)

        # 调用 self._compare_orig_and_checkpointed_fns 方法比较原始函数 gn 和检查点函数 fn 的行为是否一致
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    # 标记测试函数需要 CUDA，且在 Windows 下会跳过执行 torch.compile
    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    # 定义一个测试方法，用于测试选择性检查点生成的不原地操作
    def test_compile_selective_checkpoint_outplace_op(self):
        # 定义一个内部方法，用于创建选择性检查点上下文
        def selective_checkpointing_context_fn():
            # 指定不需要重新计算的操作列表
            no_recompute_list = [
                torch.ops.aten.mm.default,  # 矩阵乘法操作
                torch.ops.aten.sigmoid.default,  # sigmoid 操作
            ]
            # 创建选择性检查点的上下文
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list),
            )

        # 定义一个函数 gn，对输入进行一系列非原地操作
        def gn(x, y):
            return torch.sigmoid(torch.selu(torch.matmul(torch.matmul(x, y), y))).relu()

        # 定义一个函数 fn，使用检查点技术对函数 gn 进行优化
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,  # 禁用重入以优化内存使用
                context_fn=selective_checkpointing_context_fn,  # 使用选择性检查点的上下文
            )

        # 在 CUDA 设备上生成随机输入张量 x 和 y，要求梯度计算
        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        # 定义前向计算的编译器，部分计算使用 mm 和 sigmoid 操作
        fw_compiler = functools.partial(
            count_ops,
            freqs=[2, 1],  # 操作频率列表
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],  # 使用的操作列表
        )
        # 定义反向计算的编译器，完全使用 mm 和 sigmoid 操作
        bw_compiler = functools.partial(
            count_ops,
            freqs=[4, 0],  # 操作频率列表
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],  # 使用的操作列表
        )
        # 使用 ahead-of-time (AOT) 自动微分，编译前向和反向计算
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,  # 分割函数
        )
        # 验证优化后的函数 fn 在给定输入 x, y 下的正确性
        self._validate(fn, backend, x, y)
        # 比较原始函数 gn 和优化后的函数 fn 在输入 x, y 下的效果
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    # 要求 CUDA 环境，且在 Windows 系统下跳过测试
    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skip(
        "In-place op support in selective checkpointing + torch.compile "
        "requires TorchDispatchMode + torch.compile work to complete"
    )
    # 定义测试方法：验证选择性检查点内联操作的编译
    def test_compile_selective_checkpoint_inplace_op(self):
        # 定义选择性检查点上下文函数
        def selective_checkpointing_context_fn():
            # 指定不需要重新计算的操作列表
            no_recompute_list = [
                torch.ops.aten.mm.default,  # 矩阵乘法操作
                torch.ops.aten.sigmoid.default,  # sigmoid 操作
            ]
            # 创建选择性检查点上下文
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        # 定义函数 gn，执行一系列操作并返回结果
        def gn(x, y):
            # 计算矩阵乘法，然后应用 SELU 激活函数和 sigmoid 函数
            return torch.sigmoid(
                torch.selu_(torch.matmul(torch.matmul(x, y), y))
            ).relu_()

        # 定义函数 fn，使用检查点技术执行函数 gn
        def fn(x, y):
            # 使用检查点技术运行函数 gn，以减少内存消耗
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        # 生成随机张量 x 和 y，需要梯度，使用 CUDA 加速
        x = torch.randn(4, 4, requires_grad=True, device="cuda")
        y = torch.randn(4, 4, requires_grad=True, device="cuda")

        # 设置前向编译器，指定操作频率和操作列表
        fw_compiler = functools.partial(
            count_ops,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )

        # 设置反向编译器，指定操作频率和操作列表
        bw_compiler = functools.partial(
            count_ops,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )

        # 设置后端，使用 AOT 自动微分，指定前向和反向编译器以及分区函数
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,
        )

        # 验证函数 fn 的结果与期望输出的一致性
        self._validate(fn, backend, x, y)

        # 比较原始函数 gn 和检查点优化后的函数 fn 的输出结果
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    # 使用 CUDA 时才执行此测试，如果是 Windows 系统则跳过测试（因为 torch.compile 在 Windows 上不工作）
    @requires_cuda
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    # 定义测试方法：测试有选择性的检查点和随机操作
    def test_compile_selective_checkpoint_random_op(self):
        # 针对保留或不保留随机数生成器状态进行迭代测试
        for preserve_rng_state in [True, False]:

            # 定义选择性检查点上下文函数
            def selective_checkpointing_context_fn():
                # 定义不需要重新计算的操作列表
                no_recompute_list = [
                    torch.ops.aten.sigmoid.default,
                ]
                # 创建选择性检查点上下文
                return create_selective_checkpoint_contexts(
                    _get_custom_policy(no_recompute_list=no_recompute_list)
                )

            # 定义函数 gn，对输入进行多层操作
            def gn(x):
                # 对输入进行 sigmoid 操作并加入 dropout
                return torch.sigmoid(torch.dropout(torch.sigmoid(x), p=0.5, train=True))

            # 定义函数 fn，使用检查点对 gn 函数进行处理
            def fn(x):
                return torch.utils.checkpoint.checkpoint(
                    gn,
                    x,
                    use_reentrant=False,
                    # 无论 `preserve_rng_state` 是 True 还是 False，
                    # 在使用 `torch.compile` 时都会保留 RNG 状态。
                    preserve_rng_state=preserve_rng_state,
                    context_fn=selective_checkpointing_context_fn,
                )

            # 创建测试用的输入张量 x，要求梯度计算，位于 CUDA 设备上
            x = torch.randn(4, 4, requires_grad=True, device="cuda")

            # 定义前向编译器 fw_compiler，部分函数频率和操作
            fw_compiler = functools.partial(
                count_ops,
                freqs=[2, 1],
                ops=[
                    torch.ops.aten.sigmoid.default,
                    torch.ops.aten.native_dropout.default,
                ],
            )
            # 定义反向编译器 bw_compiler，部分函数频率和操作
            bw_compiler = functools.partial(
                count_ops,
                # 注意：此单元测试期望 `dropout` 重新计算（`native_dropout` 计数为 1）。
                freqs=[0, 1],
                ops=[
                    torch.ops.aten.sigmoid.default,
                    torch.ops.aten.native_dropout.default,
                ],
            )
            # 使用自动微分编译器 backend，前向和反向编译器，以及分区函数
            backend = aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=min_cut_rematerialization_partition,
            )

            # 当 `preserve_rng_state` 为 False 时，torch.compile 与 eager 模式之间的梯度将不匹配，
            # 因为 eager 模式不保留 RNG 状态，而 torch.compile 仍然保留。
            # 因此当 `preserve_rng_state` 为 False 时，我们跳过 torch.compile 和 eager 之间的输出和梯度比较。
            self._validate(fn, backend, x, skip_check=not preserve_rng_state)
            # 比较原始函数和检查点函数的输出结果
            self._compare_orig_and_checkpointed_fns(gn, fn, x)

    # 如果在 Windows 环境下，跳过此单元测试，因为 torch.compile 在 Windows 上不工作
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    def test_compile_selective_checkpoint_invalid_context(self):
        # 定义一个简单的函数 gn，用于计算 torch.sigmoid(torch.matmul(x, y)) * y
        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y)) * y
        
        # 定义一个函数 fn，使用 checkpoint 来执行 gn 函数，设置一些参数
        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=_invalid_context_gen,  # 使用了一个无效的上下文生成函数 _invalid_context_gen
            )

        # 创建两个大小为 (4, 4) 的随机张量 x 和 y，并标记需要梯度信息
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        # 使用 functools.partial 创建一个前向编译器 fw_compiler，用于统计操作频率
        fw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=torch.ops.aten.mm.default,  # 设置操作为 torch.mm
        )
        # 使用 functools.partial 创建一个反向编译器 bw_compiler，用于统计操作频率
        bw_compiler = functools.partial(
            count_ops,
            freq_ge=2,
            op=torch.ops.aten.mm.default,  # 设置操作为 torch.mm
        )
        # 创建一个后端对象 backend，使用 aot_autograd 函数，设置编译器和分区函数
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=min_cut_rematerialization_partition,  # 设置分区函数为 min_cut_rematerialization_partition
        )
        # 使用 self.assertRaisesRegex 断言捕获异常，确保函数 _validate 抛出特定的异常信息
        with self.assertRaisesRegex(
            Exception, "must generate a tuple of two `TorchDispatchMode`s"
        ):
            self._validate(fn, backend, x, y)

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    @requires_cuda
    @skipIfRocm
    def test_autocast_flash_attention(self):
        # 定义一个函数 fn，使用 torch.ops.aten._scaled_dot_product_efficient_attention.default 来执行计算
        def fn(primals_1, primals_2, primals_3):
            return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                primals_1, primals_2, primals_3, None, True, scale=0.17677669529663687
            )[0]

        # 定义一个函数 gn，使用 checkpoint 来执行 fn 函数，设置 use_reentrant=True
        def gn(*args):
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

        # 使用 torch.cuda.amp.autocast() 上下文管理器，自动执行混合精度计算
        with torch.cuda.amp.autocast():
            # 创建三个大小为 (4, 2, 16, 32) 的 CUDA 张量 x, y, z，并标记需要梯度信息
            x = torch.randn(4, 2, 16, 32, device="cuda", requires_grad=True)
            y = torch.randn(4, 2, 16, 32, device="cuda", requires_grad=True)
            z = torch.randn(4, 2, 16, 32, device="cuda", requires_grad=True)
            args = (x, y, z)

            # 设置随机种子，并通过 gn 函数生成参考结果 ref
            torch.manual_seed(0)
            ref = gn(*args)

            # 使用 torch.compile 函数编译 gn 函数，生成优化后的函数 opt_gn
            opt_gn = torch.compile(gn)
            torch.manual_seed(0)
            # 使用优化后的函数 opt_gn 执行计算，生成结果 res
            res = opt_gn(*args)
            # 使用 self.assertEqual 断言优化前后的计算结果一致
            self.assertEqual(ref, res)

    @requires_cuda
    def test_error_msg(self):
        # 定义一个 MockModule 类，继承自 torch.nn.Module，实现了一个简单的前向传播函数 forward
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # 在前向传播过程中调用 torch._dynamo.graph_break 函数，断开计算图
                x = torch.sin(x)
                torch._dynamo.graph_break()
                x = torch.cos(x)
                return x

        # 创建 MockModule 类的实例 mod，并将其移至 CUDA 设备上
        mod = MockModule().cuda()

        # 定义一个函数 fn，使用 checkpoint 来执行 mod 的前向传播，设置 use_reentrant=True
        def fn(x):
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

        # 创建一个大小为 (4, 4) 的 CUDA 张量 x，并使用 torch.compile 函数编译 fn 函数，fullgraph=True
        x = torch.randn(4, 4).cuda()
        opt_fn = torch.compile(fn, fullgraph=True)
        # 使用 self.assertRaisesRegex 断言捕获异常，确保优化后的函数 opt_fn 执行时抛出特定的异常信息
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "skip function graph_break in file"
        ):
            opt_fn(x)

    @requires_cuda
    # 定义测试方法，用于测试输入列表情况
    def test_list_inputs(self):
        # 定义一个模拟的 PyTorch 模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # 模块的前向传播方法，接受输入 x 和 ys
            def forward(self, x, ys):
                # 计算 x 的正弦值
                a = torch.sin(x)
                # 计算 ys 列表中第一个张量的余弦值
                b = torch.cos(ys[0])
                # 计算 ys 列表中第二个张量的余弦值
                c = torch.cos(ys[1])
                # 返回 x 和包含 b、c 的列表作为结果
                return (x, [b, c])

        # 创建 MockModule 的实例并将其移到 GPU 上
        mod = MockModule().cuda()

        # 定义一个函数 fn，使用 PyTorch 的 checkpoint 方法对模块进行检查点检查
        def fn(x, ys):
            return torch.utils.checkpoint.checkpoint(mod, x, ys, use_reentrant=True)

        # 生成随机张量 x、y、z，并将它们移到 GPU 上
        x = torch.randn(4, 4).cuda()
        y = torch.randn(4, 4).cuda()
        z = torch.randn(4, 4).cuda()
        
        # 使用 fn 函数计算参考结果 ref
        ref = fn(x, [y, z])
        
        # 使用 eager 模式和完整图形编译优化 fn 函数
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        
        # 使用优化后的函数 res 计算结果
        res = opt_fn(x, [y, z])
        
        # 使用单元测试断言检查 ref 和 res 是否相等
        self.assertEqual(ref, res)

    # 标记该测试需要 CUDA 支持
    @requires_cuda
    def test_pattern_matcher(self):
        # 检查在反向图中是否重新计算了 sdpa 操作
        # 测试 percolate_tags

        @checkpoint_wrapper
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # 计算点积注意力机制
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        def fn(query, key, value):
            # 检查在反向图中是否重新计算了 sin 操作
            return dot_prod_attention(query.sin(), key, value)

        tensor_shape = (4, 2, 16, 32)
        dtype = torch.float16
        args1 = [
            torch.randn(tensor_shape, device="cuda", dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device="cuda", dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device="cuda", dtype=dtype, requires_grad=True),
        ]

        # 保存 AOT 图
        aot_graphs = []
        from torch._inductor import compile_fx

        def debug_compile_fx_inner(graph, example_inputs, *args, **kwargs):
            aot_graphs.append(graph)
            return compile_fx.compile_fx_inner(graph, example_inputs, *args, **kwargs)

        # 定义后端编译函数
        backend = functools.partial(
            compile_fx.compile_fx, inner_compile=debug_compile_fx_inner
        )

        # 编译优化后的函数
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        opt_fn(*args1).sum().backward()

        # 前向图
        fwd_graph = aot_graphs[0]
        self.assertTrue(
            count_ops(
                fwd_graph,
                [],
                freq=1,
                op=torch.ops.aten._scaled_dot_product_flash_attention.default,
            )
        )

        # 后向图
        bwd_graph = aot_graphs[1]
        # 检查在反向图中是否重新计算了 sin 操作 - 检查 percolate tags
        self.assertTrue(count_ops(bwd_graph, [], freq=0, op=torch.ops.aten.sin.default))
        # 检查在反向图中是否重新计算了 sdpa 操作
        self.assertTrue(
            count_ops(
                bwd_graph,
                [],
                freq=1,
                op=torch.ops.aten._scaled_dot_product_flash_attention.default,
            )
        )
    # 定义一个测试方法，用于测试分布式工具的检查点包装器
    def test_distributed_utils_checkpoint_wrapper(self):
        # 导入分布式检查点包装器
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as dist_checkpoint_wrapper,
        )

        # 定义一个模拟的神经网络模块
        class MockModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)  # 创建一个线性层
                self.c = 2  # 定义一个常量参数

            def forward(self, x):
                x = torch.sin(x)  # 对输入进行正弦函数处理
                x = self.linear(x)  # 对输入进行线性变换
                x = torch.cos(x)  # 对结果进行余弦函数处理
                return x * self.c  # 返回结果乘以常量参数

        # 使用分布式检查点包装器对模拟模块进行包装
        mod = dist_checkpoint_wrapper(MockModule())
        x = torch.randn(4, 4)  # 生成一个随机输入
        ref = mod(x)  # 使用包装后的模块进行前向传播
        # 使用 torch.compile 对模块进行优化编译，使用 eager 后端并保留完整图形结构
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        res = opt_mod(x)  # 使用优化后的模块进行前向传播
        self.assertEqual(ref, res)  # 断言优化前后的结果应该一致

    # 使用装饰器声明测试需要 CUDA 环境，并且需要分布式支持
    @requires_cuda
    @requires_distributed()
    # 使用 torch._dynamo.config.patch 进行配置，以内联内置的神经网络模块
    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=True)
    # 测试确保 Dynamo 不追踪 getattr 作为顶级帧
    def test_dynamo_does_not_trace_getattr_as_top_frame(self):
        # inline_inbuilt_nn_modules 是一个代理，模拟 FSDP 测试的行为
        # 导入检查点包装器类
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        # 创建一个计数器对象，用于后端编译
        cnt = CompileCounterWithBackend("eager")

        # 创建一个线性层
        lin = torch.nn.Linear(1, 1)
        # 创建一个序列模块包含两个相同的线性层
        mod = torch.nn.Sequential(lin, lin)
        # 使用 CheckpointWrapper 对模块进行包装
        mod = CheckpointWrapper(mod)
        # 给包装后的模块设置额外的属性 a
        mod._checkpoint_wrapped_module.a = torch.ones(1, 1)

        # 定义一个函数 fn，对输入使用包装后的模块进行前向传播并乘以属性 a
        def fn(x):
            return mod(x) * mod.a

        # 使用 torch.compile 对函数 fn 进行优化编译，指定后端为 cnt，并保留完整图形结构
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.randn(1, 1)  # 生成一个随机输入

        # 断言优化前后的函数 fn 的结果应该一致
        self.assertEqual(opt_fn(x), fn(x))
# 如果当前模块被直接执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests
    
    # 调用导入的 run_tests 函数，用于运行测试用例
    run_tests()
```