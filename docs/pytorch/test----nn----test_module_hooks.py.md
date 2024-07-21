# `.\pytorch\test\nn\test_module_hooks.py`

```py
# Owner(s): ["module: nn"]
# 引入必要的库和模块
import gc  # 垃圾回收模块
import math  # 数学函数库
import pickle  # Python 对象序列化库
import unittest  # 单元测试框架
import warnings  # 警告控制模块
import weakref  # 弱引用支持模块
from collections import namedtuple, OrderedDict  # 命名元组和有序字典
from copy import deepcopy  # 深拷贝函数

from functools import partial  # 函数工具模块中的 partial 函数
from tempfile import NamedTemporaryFile  # 临时文件模块中的 NamedTemporaryFile 类
from typing import Any, Dict, List, Tuple  # 类型提示模块中的类型引入

import torch  # PyTorch 深度学习库
import torch.nn as nn  # PyTorch 中的神经网络模块
from torch.testing._internal.common_nn import _create_basic_net, NNTestCase  # PyTorch 内部测试相关
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 实例化参数化测试
    IS_WINDOWS,  # 是否为 Windows 系统
    parametrize as parametrize_test,  # 参数化测试的别名
    run_tests,  # 运行测试
    skipIfTorchDynamo,  # 如果使用 TorchDynamo 就跳过
    swap,  # 交换函数
    TestCase,  # 测试用例基类
)

# 定义一个简单的神经网络模型
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 创建两个包含两个线性层的序列容器
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq2(self.seq1(x))


# 定义一个包含两个 Net 实例的模型
class ToyModel(nn.Module):
    def __init__(self, with_named_tuple=False) -> None:
        super().__init__()
        # 创建两个 Net 实例
        self.net1 = Net()
        self.net2 = Net()
        self.with_named_tuple = with_named_tuple  # 是否使用命名元组作为输出的标志

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向传播过程，通过 net1 和 net2 依次处理输入 x
        res = self.net2(self.net1(x))
        if self.with_named_tuple:
            return ToyNamedTuple(res)  # 如果使用命名元组，则将结果包装成命名元组返回
        else:
            return (res,)  # 否则返回一个包含结果的元组


# 定义一个前向钩子函数
def forward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: Tuple[torch.Tensor],
    out: torch.Tensor,
) -> None:
    # 记录钩子被触发的 ID
    fired_hooks.append(hook_id)
    # 断言当前模块的 ID 与预期模块的 ID 相同
    self.assertEqual(id(module), id(expected_module))
    # 断言输入的张量个数为1
    self.assertEqual(len(inp), 1)


# 定义一个前向预钩子函数
def forward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: Tuple[torch.Tensor],
) -> None:
    # 记录预钩子被触发的 ID
    fired_hooks.append(hook_id)
    # 断言当前模块的 ID 与预期模块的 ID 相同
    self.assertEqual(id(module), id(expected_module))
    # 断言输入的张量个数为1
    self.assertEqual(len(inp), 1)


# 定义一个完全反向钩子函数
def full_backward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: Tuple[torch.Tensor],
    grad_output: Tuple[torch.Tensor],
) -> None:
    # 记录反向钩子被触发的 ID
    fired_hooks.append(hook_id)
    # 断言当前模块的 ID 与预期模块的 ID 相同
    self.assertEqual(id(module), id(expected_module))
    # 断言梯度输入张量的个数为1
    self.assertEqual(len(grad_input), 1)
    # 断言梯度输出张量的个数为1
    self.assertEqual(len(grad_output), 1)


# 定义一个完全反向预钩子函数
def full_backward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: Tuple[torch.Tensor],
) -> None:
    # 记录反向预钩子被触发的 ID
    fired_hooks.append(hook_id)
    # 断言当前模块的 ID 与预期模块的 ID 相同
    self.assertEqual(id(module), id(expected_module))
    # 断言梯度输入张量的个数为1
    self.assertEqual(len(grad_input), 1)


# 定义一个带有两个 Net 实例的模型
class KwargModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 创建两个 Net 实例
        self.net1 = Net()
        self.net2 = Net()
    # 定义一个方法 `forward`，接收一个张量 `x` 和一个可选的张量 `bias`，返回处理后的张量
    def forward(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        # 如果传入了 `bias`，则将 `bias` 加到 `x` 上
        if bias is not None:
            x = x + bias
        # 返回处理后的张量 `x`
        return x

    # 定义一个方法 `internal_forward_hook`，用作模型中某个模块的前向钩子
    def internal_forward_hook(
        self,
        module: nn.Module,
        args: Tuple[torch.Tensor],  # 接收张量参数的元组
        kwargs: Dict[str, Any],     # 接收其他关键字参数的字典
        out: torch.Tensor,          # 接收模块的输出张量
    ):
        # 返回输出张量 `out` 加上关键字参数 `kwargs` 中的 `bias` 张量
        return out + kwargs["bias"]
class FailsInForwardModel(nn.Module):
    # 定义一个继承自 nn.Module 的模型类 FailsInForwardModel
    def __init__(self) -> None:
        super().__init__()
        # 初始化一个名为 net1 的子模块，类型为 Net
        self.net1 = Net()

    def forward(self, x: torch.Tensor, fail: bool = True) -> torch.Tensor:
        # 如果 fail 参数为 True，则抛出运行时错误
        if fail:
            raise RuntimeError("failing in forward")
        # 否则，调用 net1 的 forward 方法，传入参数 x，返回输出
        return self.net1(x)


def kwarg_forward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    args: Tuple[torch.Tensor],
    kwargs: Dict[str, Any],
) -> Tuple[Any, Any]:
    # 将 hook_id 添加到 fired_hooks 列表中
    fired_hooks.append(hook_id)
    # 断言 module 的 id 和 expected_module 的 id 相同
    self.assertEqual(id(module), id(expected_module))
    # 断言 args 的长度为 1
    self.assertEqual(len(args), 1)
    # 将 kwargs 中的 "bias" 值乘以 2
    kwargs["bias"] = 2 * kwargs["bias"]
    # 返回修改后的 args 和 kwargs
    return args, kwargs


def kwarg_forward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    args: Tuple[torch.Tensor],
    kwargs: Dict[str, Any],
    out: torch.Tensor,
) -> Any:
    # 将 hook_id 添加到 fired_hooks 列表中
    fired_hooks.append(hook_id)
    # 断言 module 的 id 和 expected_module 的 id 相同
    self.assertEqual(id(module), id(expected_module))
    # 断言 args 的长度为 1
    self.assertEqual(len(args), 1)

    # 将 out 与 kwargs 中的 "bias" 相加
    out = out + kwargs["bias"]
    # 返回处理后的 out
    return out


class DummyContextManager:
    # 定义一个虚拟的上下文管理器 DummyContextManager
    def __init__(self, inp):
        # 初始化输入 inp
        self.input = inp

    def __enter__(self, *args, **kwargs):
        # 当进入上下文时，在 self.input 中追加值 2
        self.input.append(2)

    def __exit__(self, *args, **kwargs):
        # 当退出上下文时，在 self.input 中追加值 -1
        self.input.append(-1)


class TestModuleHooks(TestCase):
    # 测试模块钩子的功能
    @parametrize_test("named_tuple", (True, False))
    def test_forward_hooks(self, named_tuple):
        # 初始化 fired_hooks 列表，用于记录钩子被触发的顺序
        fired_hooks: List[int] = []
        # 创建一个 ToyModel 实例，传入 named_tuple 参数
        model = ToyModel(named_tuple)
        # 生成一个大小为 (10, 10) 的随机张量 x
        x = torch.randn(10, 10)
        # 创建一个部分函数应用，用于绑定测试函数中的参数和 model.net1.seq2 的 forward_hook
        hook = partial(forward_hook, self, fired_hooks, model.net1.seq2)
        # 向 model.net1.seq2 注册多个 forward_hook
        model.net1.seq2.register_forward_hook(partial(hook, 0))
        model.net1.seq2.register_forward_hook(partial(hook, 1), prepend=True)
        model.net1.seq2.register_forward_hook(partial(hook, 2))
        model.net1.seq2.register_forward_hook(partial(hook, 3))
        model.net1.seq2.register_forward_hook(partial(hook, 4), prepend=True)
        # 预期的钩子触发顺序
        expected = [4, 1, 0, 2, 3]

        # 断言 fired_hooks 初始为空列表
        self.assertEqual(fired_hooks, [])
        # 执行模型的 forward 方法，获取输出 out
        out = model(x)
        # 断言 fired_hooks 中的值与 expected 相同
        self.assertEqual(fired_hooks, expected)
        # 断言 out 的类型为 ToyNamedTuple 或 tuple（取决于 named_tuple 参数）
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        # 对 out 中的第一个元素求和，并进行反向传播
        out[0].sum().backward()
        # 再次断言 fired_hooks 中的值与 expected 相同
        self.assertEqual(fired_hooks, expected)
        # 再次执行模型的 forward 方法，对 out 中的第一个元素求和，并进行反向传播
        model(x)[0].sum().backward()
        # 最终断言 fired_hooks 中的值为 expected 的两倍
        self.assertEqual(fired_hooks, expected + expected)

    @parametrize_test("named_tuple", (True, False))
    # 测试前向钩子的功能，接收一个命名元组作为参数
    def test_forward_pre_hooks(self, named_tuple):
        # 用于记录触发的钩子的编号列表
        fired_hooks: List[int] = []
        # 创建一个ToyModel对象，传入命名元组作为参数
        model = ToyModel(named_tuple)
        # 创建一个大小为10x10的随机张量x
        x = torch.randn(10, 10)
        # 定义一个部分应用了forward_pre_hook函数的钩子
        hook = partial(forward_pre_hook, self, fired_hooks, model.net2.seq1)
        # 将钩子注册到model.net2.seq1的前向预处理钩子列表中，优先级最高
        model.net2.seq1.register_forward_pre_hook(partial(hook, 0), prepend=True)
        # 将钩子注册到model.net2.seq1的前向预处理钩子列表中
        model.net2.seq1.register_forward_pre_hook(partial(hook, 1))
        # 将钩子注册到model.net2.seq1的前向预处理钩子列表中
        model.net2.seq1.register_forward_pre_hook(partial(hook, 2))
        # 将钩子注册到model.net2.seq1的前向预处理钩子列表中
        model.net2.seq1.register_forward_pre_hook(partial(hook, 3))
        # 将钩子注册到model.net2.seq1的前向预处理钩子列表中，优先级最高
        model.net2.seq1.register_forward_pre_hook(partial(hook, 4), prepend=True)
        # 预期的钩子触发顺序
        expected = [4, 0, 1, 2, 3]

        # 断言fired_hooks列表应为空
        self.assertEqual(fired_hooks, [])
        # 执行模型的前向传播
        out = model(x)
        # 断言fired_hooks列表应为预期的触发顺序
        self.assertEqual(fired_hooks, expected)
        # 断言out的类型为ToyNamedTuple（如果named_tuple为True），否则为tuple
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        # 对out中的第一个元素求和并进行反向传播
        out[0].sum().backward()
        # 断言fired_hooks列表应为预期的触发顺序
        self.assertEqual(fired_hooks, expected)
        # 再次执行模型的前向传播，对out中的第一个元素求和并进行反向传播
        model(x)[0].sum().backward()
        # 断言fired_hooks列表应为预期的触发顺序的两倍
        self.assertEqual(fired_hooks, expected + expected)

    # 使用参数化测试named_tuple来测试全向后向钩子的功能
    @parametrize_test("named_tuple", (True, False))
    def test_full_backward_hooks(self, named_tuple):
        # 用于记录触发的钩子的编号列表
        fired_hooks: List[int] = []
        # 创建一个ToyModel对象，传入命名元组作为参数
        model = ToyModel(named_tuple)
        # 创建一个大小为10x10的随机张量x
        x = torch.randn(10, 10)
        # 定义一个部分应用了full_backward_hook函数的钩子
        hook = partial(full_backward_hook, self, fired_hooks, model.net1)
        # 将钩子注册到model.net1的全向后向钩子列表中
        model.net1.register_full_backward_hook(partial(hook, 0))
        # 将钩子注册到model.net1的全向后向钩子列表中
        model.net1.register_full_backward_hook(partial(hook, 1))
        # 将钩子注册到model.net1的全向后向钩子列表中
        model.net1.register_full_backward_hook(partial(hook, 2))
        # 将钩子注册到model.net1的全向后向钩子列表中，优先级最高
        model.net1.register_full_backward_hook(partial(hook, 3), prepend=True)
        # 将钩子注册到model.net1的全向后向钩子列表中，优先级最高
        model.net1.register_full_backward_hook(partial(hook, 4), prepend=True)
        # 预期的钩子触发顺序
        expected = [4, 3, 0, 1, 2]

        # 断言fired_hooks列表应为空
        self.assertEqual(fired_hooks, [])
        # 执行模型的前向传播
        out = model(x)
        # 断言fired_hooks列表应为空
        self.assertEqual(fired_hooks, [])
        # 断言out的类型为ToyNamedTuple（如果named_tuple为True），否则为tuple
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        # 对out中的第一个元素求和并进行反向传播
        out[0].sum().backward()
        # 断言fired_hooks列表应为预期的触发顺序
        self.assertEqual(fired_hooks, expected)
        # 再次执行模型的前向传播，对out中的第一个元素求和并进行反向传播
        model(x)[0].sum().backward()
        # 断言fired_hooks列表应为预期的触发顺序的两倍
        self.assertEqual(fired_hooks, expected + expected)
    # 定义测试方法，用于测试模型的全向后预处理钩子
    def test_full_backward_pre_hooks(self, named_tuple):
        # 用于记录已触发的钩子列表，初始化为空列表
        fired_hooks: List[int] = []
        # 创建一个 ToyModel 实例
        model = ToyModel(named_tuple)
        # 生成一个 10x10 的随机张量
        x = torch.randn(10, 10)
        
        # 创建一个偏函数 hook，用于注册到 net1 的全向后预处理钩子中
        hook = partial(full_backward_pre_hook, self, fired_hooks, model.net1)
        model.net1.register_full_backward_pre_hook(partial(hook, 0), prepend=True)
        model.net1.register_full_backward_pre_hook(partial(hook, 1), prepend=True)
        model.net1.register_full_backward_pre_hook(partial(hook, 2))
        model.net1.register_full_backward_pre_hook(partial(hook, 3))
        model.net1.register_full_backward_pre_hook(partial(hook, 4))
        
        # 预期的触发顺序
        expected = [1, 0, 2, 3, 4]

        # 断言 fired_hooks 列表为空
        self.assertEqual(fired_hooks, [])
        # 对模型进行前向传播
        out = model(x)
        # 再次断言 fired_hooks 列表为空
        self.assertEqual(fired_hooks, [])
        # 断言输出类型为 ToyNamedTuple 或者 tuple，根据 named_tuple 参数决定
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        # 对输出的第一个元素求和并反向传播
        out[0].sum().backward()
        # 断言 fired_hooks 包含了预期的顺序
        self.assertEqual(fired_hooks, expected)
        # 再次对模型进行前向传播、求和并反向传播
        model(x)[0].sum().backward()
        # 断言 fired_hooks 包含了两次预期顺序的拼接
        self.assertEqual(fired_hooks, expected + expected)

        # 后向预处理钩子可以影响后续的梯度计算
        for rg in [True, False]:
            # 创建一个张量 a，是否需要梯度根据 rg 参数决定
            a = torch.ones(2, requires_grad=rg)
            # 创建一个简单的线性模型
            model = nn.Linear(2, 2)

            # 定义一个自定义的全向后预处理钩子函数 fn
            def fn(_unused_module, grad_output):
                return (grad_output[0] * 0,)

            # 将 fn 注册为模型的全向后预处理钩子
            model.register_full_backward_pre_hook(fn)

            # 对模型进行前向传播
            out = model(a)
            # 对输出求和并反向传播
            out.sum().backward()
            # 断言模型权重的梯度为全零张量
            self.assertEqual(model.weight.grad, torch.zeros(2, 2))
            # 如果 a 需要梯度，则断言 a 的梯度为全零张量；否则断言其梯度为 None
            if rg:
                self.assertEqual(a.grad, torch.zeros_like(a))
            else:
                self.assertIsNone(a.grad)

    # 使用 parametrize_test 装饰器，为测试方法 test_mixed_hooks 提供不同的 named_tuple 参数值
    @parametrize_test("named_tuple", (True, False))
    # 定义测试方法，用于测试混合类型的钩子
    def test_mixed_hooks(self, named_tuple):
        # 用于记录已触发的钩子列表，初始化为空列表
        fired_hooks: List[int] = []
        # 创建一个 ToyModel 实例
        model = ToyModel(named_tuple)
        # 生成一个 10x10 的随机张量
        x = torch.randn(10, 10)
        
        # 将不同类型的钩子注册到模型中
        model.register_forward_pre_hook(
            partial(forward_pre_hook, self, fired_hooks, model, 0)
        )
        model.register_forward_hook(partial(forward_hook, self, fired_hooks, model, 1))
        model.register_full_backward_pre_hook(
            partial(full_backward_pre_hook, self, fired_hooks, model, 2)
        )
        model.register_full_backward_hook(
            partial(full_backward_hook, self, fired_hooks, model, 3)
        )

        # 断言 fired_hooks 列表为空
        self.assertEqual(fired_hooks, [])
        # 对模型进行前向传播
        out = model(x)
        # 断言 fired_hooks 列表包含了预期的前向钩子触发顺序
        self.assertEqual(fired_hooks, [0, 1])
        # 断言输出类型为 ToyNamedTuple 或者 tuple，根据 named_tuple 参数决定
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        # 对输出的第一个元素求和并反向传播
        out[0].sum().backward()
        # 断言 fired_hooks 列表包含了预期的后向预处理钩子触发顺序
        self.assertEqual(fired_hooks, [0, 1, 2, 3])
        # 再次对模型进行前向传播、求和并反向传播
        model(x)[0].sum().backward()
        # 断言 fired_hooks 列表包含了两次预期顺序的拼接
        self.assertEqual(fired_hooks, [0, 1, 2, 3, 0, 1, 2, 3])
    def test_kwarg_hooks(self):
        # 1. test forward pre hook
        # 创建一个空列表，用于记录触发的钩子编号
        fired_hooks: List[int] = []
        # 创建一个 10x10 的全为1的张量 x 和 bias
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        # 创建 KwargModel 的实例
        model = KwargModel()
        # 注册 forward pre hook，设置参数 with_kwargs=True
        model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre 钩子功能：bias' = bias * 2
        # 因此，out = x + bias * 2
        self.assertEqual(fired_hooks, [])  # 断言 fired_hooks 列表为空
        out = model(x, bias=bias)  # 调用模型的 forward 方法
        self.assertEqual(fired_hooks, [0])  # 断言 fired_hooks 中记录了编号 0
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)  # 断言计算结果与预期相符

        # 2. test forward pre and forward hooks
        # 重新创建一个空列表，用于记录触发的钩子编号
        fired_hooks: List[int] = []
        # 再次创建 x 和 bias 的张量
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        # 创建一个新的 KwargModel 实例
        model = KwargModel()
        # 注册 forward hook，设置参数 with_kwargs=True
        model.register_forward_hook(
            partial(kwarg_forward_hook, self, fired_hooks, model, 1),
            with_kwargs=True,
        )
        # 注册 forward pre hook，设置参数 with_kwargs=True
        model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre 钩子功能：bias' = bias * 2
        # forward 钩子功能：out = x + bias'
        # forward-post 钩子功能：out = out + bias'
        # 因此，out = x + bias * 4
        self.assertEqual(fired_hooks, [])  # 断言 fired_hooks 列表为空
        out = model(x, bias=bias)  # 调用模型的 forward 方法
        self.assertEqual(fired_hooks, [0, 1])  # 断言 fired_hooks 中记录了编号 0 和 1
        self.assertEqual(out, x + 4 * bias, rtol=0, atol=1e-5)  # 断言计算结果与预期相符

        # 3. test nn.Module member method as forward-post hook
        # 再次创建 x 和 bias 的张量
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        # 创建一个新的 KwargModel 实例
        model = KwargModel()
        # 注册 forward hook，使用 nn.Module 内部的 forward-post 钩子方法，设置参数 with_kwargs=True
        model.register_forward_hook(model.internal_forward_hook, with_kwargs=True)

        # forward 钩子功能：out = x + bias
        # forward-post 钩子功能：out = out + bias
        # 因此，out = x + bias * 2
        out = model(x, bias=bias)  # 调用模型的 forward 方法
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)  # 断言计算结果与预期相符
    def test_remove_kwarg_hooks(self):
        # 测试移除关键字参数钩子的功能

        # 初始化一个空列表，用于记录触发的钩子编号
        fired_hooks: List[int] = []

        # 创建一个 10x10 的全一张量 x 和 bias
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)

        # 创建一个 KwargModel 实例
        model = KwargModel()

        # 注册 forward 钩子，传入部分参数和关键字参数
        forward_hook_handle = model.register_forward_hook(
            partial(kwarg_forward_hook, self, fired_hooks, model, 1),
            with_kwargs=True,
        )

        # 注册 forward-pre 钩子，传入部分参数和关键字参数
        forward_pre_hook_handle = model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre 阶段：bias' = bias * 2
        # forward 阶段：out = x + bias'
        # forward-post 阶段：out = out + bias'
        # 因此，out = x + bias * 4
        self.assertEqual(fired_hooks, [])
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1])
        self.assertEqual(out, x + 4 * bias, rtol=0, atol=1e-5)

        # forward-pre 阶段：bias' = bias * 2
        # forward 阶段：out = x + bias'
        # 因此，out = x + bias * 2
        forward_hook_handle.remove()
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1, 0])
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)
        self.assertFalse(forward_hook_handle.id in model._forward_hooks_with_kwargs)

        # forward 阶段：out = x + bias
        # 因此，out = x + bias
        forward_pre_hook_handle.remove()
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1, 0])
        self.assertEqual(out, x + bias, rtol=0, atol=1e-5)
        self.assertFalse(
            forward_pre_hook_handle.id in model._forward_pre_hooks_with_kwargs
        )
    def test_bw_hook_warning_for_non_tensor_or_tuple(self):
        # 定义一个测试方法，用于验证反向传播钩子在结果不是张量或张量元组时是否会引发警告
        counter = {"forward": 0, "backward": 0}  # 初始化计数器，用于记录前向和后向传播调用次数

        def fw_pre_hook(module: nn.Module, _inputs):
            counter["forward"] += 1  # 前向传播预钩子，计数器加一

        def fw_hook(module: nn.Module, _inputs, _outputs):
            counter["forward"] += 1  # 前向传播钩子，计数器加一

        def bw_hook(module: nn.Module, _inputs, _outputs):
            counter["backward"] += 1  # 后向传播钩子，计数器加一

        class TestModule(nn.Module):
            def forward(self, dict):
                inp = dict["x"]  # 获取输入张量
                x = torch.nn.functional.softmax(inp, dim=0)  # 对输入张量进行 softmax 操作
                return {"x": x}  # 返回经过 softmax 后的结果字典

        x = torch.ones(2, requires_grad=True)  # 创建一个需要梯度的张量
        model = TestModule()  # 实例化测试模块
        model.register_forward_pre_hook(fw_pre_hook)  # 注册前向传播预钩子
        model.register_forward_hook(fw_hook)  # 注册前向传播钩子
        model.register_full_backward_pre_hook(bw_hook)  # 注册完全后向传播预钩子
        model.register_full_backward_hook(bw_hook)  # 注册完全后向传播钩子

        with warnings.catch_warnings(record=True) as w:  # 捕获警告信息
            y = model({"x": x})["x"]  # 对模型进行前向传播
            loss = y.sum()  # 计算损失
            loss.backward()  # 反向传播计算梯度

        self.assertEqual(counter["forward"], 2)  # 断言前向传播调用次数为2
        self.assertEqual(counter["backward"], 0)  # 断言后向传播调用次数为0
        self.assertEqual(len(w), 1)  # 断言警告数量为1
        self.assertTrue("should be a Tensor or a tuple of Tensors" in str(w[0].message))  # 断言警告信息包含特定文本
# 定义一个空的函数，用作后续的状态字典加载前钩子
def _hook_to_pickle(*args, **kwargs):
    pass


# 定义一个测试类 TestStateDictHooks，继承自 TestCase
class TestStateDictHooks(TestCase):

    # 使用修饰器 swap 对 test_load_state_dict_pre_hook 方法进行参数交换（True 和 False）
    @swap([True, False])
    # 定义测试方法 test_load_state_dict_pre_hook
    def test_load_state_dict_pre_hook(self):
        # 创建一个 nn.Linear 模型 m，输入和输出维度都为 10
        m = nn.Linear(10, 10)
        # 获取模型 m 的状态字典
        m_state_dict = m.state_dict()

        # 创建一个新的 nn.Linear 模型 m_load，输入和输出维度都为 10
        m_load = nn.Linear(10, 10)

        # 定义一个计数钩子调用次数的变量
        hook_called = 0

        # 定义一个没有模块参数的状态字典加载前钩子函数 hook_without_module
        def hook_without_module(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            # 断言传入的状态字典与 m_state_dict 相等
            self.assertEqual(m_state_dict, state_dict)
            nonlocal hook_called
            hook_called += 1

        # 定义一个带模块参数的状态字典加载前钩子函数 hook_with_module
        def hook_with_module(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            # 断言传入的状态字典与 m_state_dict 相等
            self.assertEqual(m_state_dict, state_dict)
            # 断言传入的模块 module 是 m_load
            self.assertTrue(m_load is module)
            nonlocal hook_called
            hook_called += 1

        # 初始化 hook_called 为 0
        hook_called = 0
        # 将 hook_without_module 注册为 m_load 的状态字典加载前钩子函数
        m_load._register_load_state_dict_pre_hook(hook_without_module)
        # 加载 m_state_dict 到 m_load
        m_load.load_state_dict(m_state_dict)
        # 断言 hook_called 等于 1
        self.assertEqual(1, hook_called)

        # 初始化 hook_called 为 0
        hook_called = 0
        # 将 hook_with_module 注册为 m_load 的状态字典加载前钩子函数，传入模块参数为 True
        m_load._register_load_state_dict_pre_hook(hook_with_module, True)
        # 再次加载 m_state_dict 到 m_load
        m_load.load_state_dict(m_state_dict)
        # 断言 hook_called 等于 2
        self.assertEqual(2, hook_called)

    # 定义测试方法 test_no_extra_ref_to_module
    def test_no_extra_ref_to_module(self):
        try:
            # 禁用垃圾回收机制
            gc.disable()
            # 创建一个 nn.Linear 模型 m，输入和输出维度都为 10
            m = nn.Linear(10, 10)

            # 将 _hook_to_pickle 注册为 m 的状态字典加载前钩子函数，传入模块参数为 True
            m._register_load_state_dict_pre_hook(_hook_to_pickle, True)
            # 创建 m 的弱引用 weak_m
            weak_m = weakref.ref(m)
            # 删除 m 对象
            del m

            # 断言 weak_m() 返回 None，即 m 对象已经被销毁
            self.assertEqual(weak_m(), None)
        finally:
            # 最终始终启用垃圾回收机制
            gc.enable()

    # 定义测试方法 test_pickled_hook
    def test_pickled_hook(self):
        # 创建一个 nn.Linear 模型 m，输入和输出维度都为 10
        m = nn.Linear(10, 10)
        # 将 _hook_to_pickle 注册为 m 的状态字典加载前钩子函数，传入模块参数为 True
        m._register_load_state_dict_pre_hook(_hook_to_pickle, True)
        # 将模型 m 进行 pickle 序列化和反序列化操作
        pickle.loads(pickle.dumps(m))

    # 使用修饰器 swap 对 test_no_extra_ref_to_module 方法进行参数交换（True 和 False）
    @swap([True, False])
    # 定义一个测试函数，用于测试加载状态字典前钩子的功能
    def test_load_state_dict_module_pre_hook(self):
        # 初始化钩子调用次数为零
        hook_called = 0

        # 定义一个继承自 nn.Module 的测试模块 MyModule
        class MyModule(nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            # 自定义的加载前钩子方法，处理模块实例的状态字典加载前的逻辑
            def my_pre_load_hook(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                # 断言条件，验证错误消息、意外的键、缺失的键都为空列表
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                # 断言严格模式为真
                assert strict
                # 使用 nonlocal 关键字更新钩子调用次数
                nonlocal hook_called
                hook_called += 1

            # 另一个自定义的加载前钩子方法，带有模块作为参数
            def my_pre_load_hook_with_module(
                self,
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                # 断言条件，验证错误消息、意外的键、缺失的键都为空列表
                assert [] == error_msgs
                assert [] == unexpected_keys
                assert [] == missing_keys
                # 断言严格模式为真
                assert strict
                # 断言当前模块是预期的模块
                assert self is module
                # 使用 nonlocal 关键字更新钩子调用次数
                nonlocal hook_called
                hook_called += 1

        # 定义一个包含模块的容器类 MyModuleContainer，继承自 nn.Module
        class MyModuleContainer(nn.Module):
            # 初始化方法
            def __init__(self, mod):
                super().__init__()
                self.mod = mod

        # 对两种类的构造方式进行遍历测试
        for ctor in [MyModuleContainer, lambda x: x]:
            # 根据当前构造函数创建模块 m
            m = ctor(MyModule())
            # 获取当前模块的状态字典
            state_dict = m.state_dict()
            # 确定模块对象 mod，根据是否是 MyModuleContainer 类型来决定
            if isinstance(m, MyModuleContainer):
                mod = m.mod
            else:
                mod = m

            # 重置钩子调用次数
            hook_called = 0
            # 注册加载状态字典前钩子 my_pre_load_hook
            mod._register_load_state_dict_pre_hook(mod.my_pre_load_hook)
            # 加载状态字典
            m.load_state_dict(state_dict)
            # 断言钩子调用次数为 1
            self.assertEqual(1, hook_called)

            # 重置钩子调用次数
            hook_called = 0
            # 注册加载状态字典前钩子 my_pre_load_hook_with_module，并指定使用模块作为参数
            mod._register_load_state_dict_pre_hook(
                mod.my_pre_load_hook_with_module, True
            )
            # 再次加载状态字典
            m.load_state_dict(state_dict)
            # 断言钩子调用次数为 2
            self.assertEqual(2, hook_called)
    # 定义测试方法：验证 load_state_dict 后钩子函数的调用情况
    def test_load_state_dict_post_hook(self):
        # 计数钩子函数被调用的次数
        hook_called = 0

        # 定义一个自定义模块 MyModule
        class MyModule(nn.Module):
            # 模块初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个名为 foo 的参数
                self.foo = torch.nn.Parameter(torch.rand(10))

            # 自定义的加载后钩子函数
            def my_post_load_hook(self, module, incompatible_keys):
                # 断言加载后的模块对象是当前的 self
                assert module is self
                # 使用 nonlocal 声明 hook_called 是外部函数的局部变量
                nonlocal hook_called
                # 向不兼容键列表中添加一个缺失的键 "foo"
                incompatible_keys.missing_keys.append("foo")
                # 向不兼容键列表中添加一个意外的键 "bar"
                incompatible_keys.unexpected_keys.append("bar")
                # 钩子函数被调用次数加一
                hook_called += 1

        # 创建一个嵌套的 MyModule 实例
        nested = MyModule()
        # 使用 nn.ModuleList 将嵌套的模块包装起来
        wrapped = nn.ModuleList([nested])
        # 注册加载状态字典后的钩子函数
        handle = nested.register_load_state_dict_post_hook(
            nested.my_post_load_hook,
        )
        # 即使模块被包装，钩子函数也必须被调用
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        # 断言钩子函数被调用一次
        self.assertEqual(hook_called, 1)
        # 确保钩子函数修改了缺失键和意外键
        missing = ret.missing_keys
        unexpected = ret.unexpected_keys
        self.assertEqual(missing, ["foo"])
        self.assertEqual(unexpected, ["bar"])
        # 当 strict=True 时，引发的错误应提及钩子函数添加的缺失和意外键
        with self.assertRaisesRegex(RuntimeError, "foo.*\n.*bar"):
            wrapped.load_state_dict(wrapped.state_dict(), strict=True)
        # 钩子函数被调用次数再次加一
        self.assertEqual(hook_called, 2)
        # 移除通过 handle.remove() 注册的钩子函数，应该不再触发它
        # 钩子函数未运行，因此不应添加任何键
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(ret.missing_keys, [])
        self.assertEqual(ret.unexpected_keys, [])
        # hook_called 不应再增加
        self.assertEqual(hook_called, 2)

        # 定义一个清空不兼容键的加载钩子函数
        def load_hook_clear_incompatible(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        # 注册清空不兼容键的加载状态字典后钩子函数
        nested.register_load_state_dict_post_hook(load_hook_clear_incompatible)
        # 获取模块的状态字典
        state_dict = wrapped.state_dict()
        state_dict["extra"] = torch.ones(1)
        # 使用 strict=True 加载状态字典不应引发异常
        ret = wrapped.load_state_dict(state_dict, strict=True)
        # 明确确保后钩子函数清空了不兼容键
        self.assertEqual([], ret.missing_keys)
        self.assertEqual([], ret.unexpected_keys)
    # 定义一个测试方法，用于验证加载状态字典后钩子的向后兼容性
    def test_load_state_dict_post_hook_backward_compatibility(self):
        # 定义一个自定义的加载后钩子函数
        def my_post_load_hook(mod, _):
            nonlocal called  # 使用nonlocal声明变量，在闭包中修改外部变量
            called = True

        # 遍历包含三个不同操作的模块列表
        for m in [nn.Softmin(10), nn.Softmax(10), nn.LogSoftmax(10)]:
            called = False  # 初始化钩子调用状态为False
            sd = deepcopy(m.state_dict())  # 深拷贝模块的状态字典
            self.assertTrue(hasattr(m, "_load_state_dict_post_hooks"))  # 断言模块具有"_load_state_dict_post_hooks"属性

            # 模拟一个旧模型，删除"_load_state_dict_post_hooks"属性
            delattr(m, "_load_state_dict_post_hooks")

            # 保存并加载模型，确保load_state_dict正常工作（缺乏兼容性会导致错误）
            with NamedTemporaryFile() as f:
                # 注意，torch.save / torch.load 不推荐用于保存/加载模块
                torch.save(m, f.name)  # 保存模块到临时文件
                m = torch.load(f.name)  # 加载模块
                m.load_state_dict(sd)  # 加载模型状态字典
                self.assertFalse(called)  # 断言钩子未被调用

            # 确保可以注册和调用钩子
            m.register_load_state_dict_post_hook(my_post_load_hook)  # 注册加载状态字典后钩子
            m.load_state_dict(sd)  # 再次加载模型状态字典
            self.assertTrue(called)  # 断言钩子被调用

    # 定义一个测试方法，用于测试注册状态字典预处理钩子
    def _test_register_state_dict_pre_hook(self, model, submodule):
        _state_dict_prefix = "foo."  # 定义状态字典前缀
        state_dict_pre_hook_count = 0  # 初始化状态字典预处理钩子计数
        keep_var_setting = False  # 初始化保持变量设置为False

        # 定义自定义的状态字典预处理钩子函数
        def my_state_dict_pre_hook(module, prefix, keep_vars):
            self.assertEqual(keep_vars, keep_var_setting)  # 断言保持变量设置是否与预期一致
            nonlocal state_dict_pre_hook_count  # 使用nonlocal声明变量，在闭包中修改外部变量
            state_dict_pre_hook_count += 1  # 钩子计数加一
            self.assertTrue(prefix.startswith(_state_dict_prefix))  # 断言前缀以指定字符串开头

        model.register_state_dict_pre_hook(my_state_dict_pre_hook)  # 注册模型的状态字典预处理钩子
        submodule.register_state_dict_pre_hook(my_state_dict_pre_hook)  # 注册子模块的状态字典预处理钩子

        # 定义检查结果的函数
        def check_results(model):
            nonlocal state_dict_pre_hook_count, keep_var_setting  # 使用nonlocal声明变量，在闭包中修改外部变量
            for keep_var_setting in [True, False]:  # 遍历保持变量设置的不同取值
                _ = model.state_dict(
                    prefix=_state_dict_prefix, keep_vars=keep_var_setting
                )  # 调用模型的状态字典方法，使用指定的前缀和保持变量设置
                self.assertEqual(2, state_dict_pre_hook_count)  # 断言钩子被调用次数为2
                state_dict_pre_hook_count = 0  # 重置钩子计数为0

        # 测试模型构建后状态字典的预期工作
        check_results(model)
        # 测试前向传播后状态字典的预期工作
        model(torch.ones(10, 3))  # 执行模型的前向传播
        check_results(model)  # 检查状态字典的预处理钩子调用情况
    # 定义一个测试方法，用于测试状态字典预钩子的注册和功能
    def test_register_state_dict_pre_hook(self):
        # 定义一个继承自 torch.nn.Module 的类 MyModule
        class MyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加一个包含三个线性层的序列
                self.a = nn.Sequential(
                    nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3)
                )

            # 前向传播方法
            def forward(self, x):
                return self.a(x)

        # 创建 MyModule 的实例 mod
        mod = MyModule()
        # 调用 _test_register_state_dict_pre_hook 方法，测试 mod 和 mod.a
        self._test_register_state_dict_pre_hook(mod, mod.a)

    # 定义一个测试方法，用于测试惰性模块上状态字典预钩子的注册和功能
    def test_register_state_dict_pre_hook_lazy_module(self):
        # 定义一个继承自 torch.nn.Module 的类 MyLazyModule
        class MyLazyModule(torch.nn.Module):
            # 初始化方法
            def __init__(self):
                super().__init__()
                # 添加两个惰性线性层
                self.layer1 = nn.LazyLinear(8)
                self.layer2 = nn.LazyLinear(5)

            # 前向传播方法
            def forward(self, x):
                return self.layer2(self.layer1(x))

        # 创建 MyLazyModule 的实例 mod
        mod = MyLazyModule()
        # 调用 _test_register_state_dict_pre_hook 方法，测试 mod 和 mod.layer1
        self._test_register_state_dict_pre_hook(mod, mod.layer1)

    # 根据系统是否为 Windows 跳过测试，由于 Windows 上的临时文件权限问题
    @unittest.skipIf(IS_WINDOWS, "Tempfile permission issue on windows")
    def test_register_state_dict_pre_hook_backward_compat(self):
        called = False

        # 定义一个状态字典预钩子函数
        def my_state_dict_pre_hook(*args, **kwargs):
            nonlocal called
            called = True

        # 创建一个 nn.Linear 模块 m
        m = nn.Linear(1, 1)
        # 断言 m 具有 "_state_dict_pre_hooks" 属性
        self.assertTrue(hasattr(m, "_state_dict_pre_hooks"))
        # 删除 m 的 "_state_dict_pre_hooks" 属性
        delattr(m, "_state_dict_pre_hooks")
        # 保存并加载模型，确保能够调用 state_dict 而不会遇到问题
        with NamedTemporaryFile() as f:
            # 注意，torch.save / torch.load 不推荐用于保存 / 加载模块
            torch.save(m, f.name)
            m = torch.load(f.name)

        # 确保能够运行 state_dict 而不会遇到问题
        _ = m.state_dict()
        # 断言 called 为 False
        self.assertFalse(called)
        # 注册状态字典预钩子函数
        m.register_state_dict_pre_hook(my_state_dict_pre_hook)
        _ = m.state_dict()
        # 断言 called 为 True
        self.assertTrue(called)
# 定义一个测试类 TestModuleGlobalHooks，继承自 TestCase，用于测试模块级全局钩子
class TestModuleGlobalHooks(TestCase):

    # 在每个测试方法执行后执行的清理方法
    def tearDown(self):
        # 清空模块全局的反向传播钩子字典
        nn.modules.module._global_backward_hooks = OrderedDict()
        # 清空模块全局的前向传播钩子字典
        nn.modules.module._global_forward_hooks = OrderedDict()
        # 清空模块全局的前向预处理钩子字典
        nn.modules.module._global_forward_pre_hooks = OrderedDict()

    # 装饰器，如果 TorchDynamo 存在问题与钩子兼容，则跳过该测试方法
    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    # 测试模块全局钩子的无效输出情况
    def test_module_global_hook_invalid_outputs(self):
        # 创建一个 nn.Sigmoid 模块实例
        module = nn.Sigmoid()
        # 创建一个需要梯度的随机输入张量
        input = torch.randn(5, 5, requires_grad=True)

        # 定义一个反向传播钩子，用于测试情况1
        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        # 定义一个反向传播钩子，用于测试情况2
        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        # 注册第一个反向传播钩子并测试异常情况1
        with nn.modules.module.register_module_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, "got 0, but expected 1"):
                # 对模块进行前向传播、求和、反向传播
                module(input).sum().backward()

        # 注册第二个反向传播钩子并测试异常情况2
        with nn.modules.module.register_module_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, "got 2, but expected 1"):
                # 对模块进行前向传播、求和、反向传播
                module(input).sum().backward()

    # 测试模块的全局反向传播钩子是否可写
    def test_module_backward_global_hook_writeable(self):
        # 创建一个 nn.Sigmoid 模块实例
        module = nn.Sigmoid()
        # 创建一个需要梯度的随机输入张量
        input = torch.randn(5, 5, requires_grad=True)
        # 计算输入张量的 sigmoid 函数值
        sig_x = torch.sigmoid(input)

        # 定义一个反向传播钩子函数
        def bw_hook(module, grad_input, grad_output):
            # 检查每个梯度是否是 torch.Tensor 类型
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            # 返回每个输入梯度乘以2的结果
            return tuple(gi * 2 for gi in grad_input)

        # 注册模块的全局反向传播钩子
        nn.modules.module.register_module_backward_hook(bw_hook)
        # 对模块进行前向传播、反向传播
        module(input).backward(torch.ones(5, 5))
        # 计算预期的梯度值
        expected_grad = sig_x * (1 - sig_x) * 2
        # 断言输入张量的梯度值与预期的梯度值相等
        self.assertEqual(input.grad, expected_grad)

    # 装饰器，如果 TorchDynamo 存在问题与钩子兼容，则跳过该测试方法
    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    # 测试模块的全局前向预处理钩子是否可写
    def test_module_global_forward_preforward_hook_writeable(self):
        # 创建一个 nn.Sigmoid 模块实例
        module = nn.Sigmoid()
        # 创建一个需要梯度的随机输入张量
        input = torch.randn(5, 5, requires_grad=True)
        # 计算输入张量的 sigmoid 函数值
        sig_x = torch.sigmoid(input)

        # 定义一个前向预处理钩子函数
        def forward_pre_hook(m, input):
            # 对输入张量的第一个元素应用 relu 函数
            return torch.nn.functional.relu(input[0])

        # 定义一个前向传播钩子函数
        def forward_hook(m, input, output):
            # 对输出应用负号操作
            return -output

        # 注册模块的全局前向预处理钩子
        nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
        # 注册模块的全局前向传播钩子
        nn.modules.module.register_module_forward_hook(forward_hook)
        # 对模块进行前向传播
        output = module(input)
        # 计算预期的输出结果
        expected_res = -torch.sigmoid(torch.nn.functional.relu(input))
        # 断言模块的输出与预期结果相等
        self.assertEqual(output, expected_res)
        # 对模块的输出进行反向传播
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        # 创建一个掩码，用于标识输入张量中大于0的元素
        mask = input > 0
        # 计算预期的梯度值
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        # 断言输入张量的梯度值与预期的梯度值相等
        self.assertEqual(input.grad, expected_grad)
    def test_module_forward_preforward_hook_removable(self):
        """
        This test is to test when multiple pre-forward hook functions can be
        registered successfully and used correctly, if the handle can be removable
        during the pre-forward hook function call.
        """
        # 创建一个 sigmoid 模块作为测试对象
        module = nn.Sigmoid()

        def removable_hook(m, input):
            nonlocal handle
            # 在 hook 函数内移除当前 hook 句柄
            handle.remove()
            return input

        def removable_hook_2(m, input):
            nonlocal handle_2
            # 在 hook 函数内移除当前 hook 句柄
            handle_2.remove()
            return input

        # 注册两个 pre-forward hook，并获取其句柄
        handle = module.register_forward_pre_hook(removable_hook)
        handle_2 = module.register_forward_pre_hook(removable_hook_2)

        # 确保 hook 注册成功
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        # 创建输入张量
        input = torch.randn(2, 2)
        # 对模块进行 forward 操作
        output = module(input)
        # 确保模块的输出与预期的 sigmoid 函数输出一致
        self.assertEqual(torch.sigmoid(input), output)

        # 确保 hook 移除成功
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    def test_module_forward_forward_hook_removable(self):
        """
        This test is to test when multiple forward hook functions can be registered
        successfully and used correctly, if the handle can be removable during the
        forward hook function call.
        """
        # 创建一个 sigmoid 模块作为测试对象
        module = nn.Sigmoid()

        def removable_hook(m, input, output):
            nonlocal handle
            # 在 hook 函数内移除当前 hook 句柄
            handle.remove()
            return output

        def removable_hook_2(m, input, output):
            nonlocal handle_2
            # 在 hook 函数内移除当前 hook 句柄
            handle_2.remove()
            return output

        # 注册两个 forward hook，并获取其句柄
        handle = module.register_forward_hook(removable_hook)
        handle_2 = module.register_forward_hook(removable_hook_2)

        # 确保 hook 注册成功
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        # 创建输入张量
        input = torch.randn(2, 2)
        # 对模块进行 forward 操作
        output = module(input)
        # 确保模块的输出与预期的 sigmoid 函数输出一致
        self.assertEqual(torch.sigmoid(input), output)

        # 确保 hook 移除成功
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_global_and_local_hooks_order(self):
        # 创建一个 Sigmoid 模块实例
        module = nn.Sigmoid()

        # 初始化全局前向预处理钩子状态标志
        global_forward_pre_called = False
        # 初始化局部前向预处理钩子状态标志
        local_forward_pre_called = False
        # 初始化全局前向钩子状态标志
        global_forward_called = False
        # 初始化局部前向钩子状态标志
        local_forward_called = False
        # 初始化全局反向钩子状态标志
        global_backward_called = False
        # 初始化局部反向钩子状态标志
        local_backward_called = False

        # 定义全局前向预处理钩子函数
        def global_forward_pre_hook(m, input):
            nonlocal global_forward_pre_called
            # 断言局部前向预处理钩子尚未被调用
            self.assertTrue(not local_forward_pre_called)
            global_forward_pre_called = True
            return input

        # 定义局部前向预处理钩子函数
        def local_forward_pre_hook(m, input):
            nonlocal local_forward_pre_called
            # 断言全局前向预处理钩子已被调用
            self.assertTrue(global_forward_pre_called)
            local_forward_pre_called = True
            return input

        # 定义全局前向钩子函数
        def global_forward_hook(m, input, output):
            nonlocal global_forward_called
            # 断言局部前向钩子尚未被调用
            self.assertTrue(not local_forward_called)
            global_forward_called = True
            return output

        # 定义局部前向钩子函数
        def local_forward_hook(m, input, output):
            nonlocal local_forward_called
            # 断言全局前向钩子已被调用
            self.assertTrue(global_forward_called)
            local_forward_called = True
            return output

        # 定义全局反向钩子函数
        def global_backward_hook(m, input, output):
            nonlocal global_backward_called
            # 断言局部反向钩子尚未被调用
            self.assertTrue(not local_backward_called)
            global_backward_called = True
            return input

        # 定义局部反向钩子函数
        def local_backward_hook(m, input, output):
            nonlocal local_backward_called
            # 断言全局反向钩子已被调用
            self.assertTrue(global_backward_called)
            local_backward_called = True
            return input

        # 创建输入张量
        input = torch.randn(5, 5, requires_grad=True)
        # 注册全局前向预处理钩子到模块
        nn.modules.module.register_module_forward_pre_hook(global_forward_pre_hook)
        # 注册局部前向预处理钩子到模块
        module.register_forward_pre_hook(local_forward_pre_hook)
        # 注册全局前向钩子到模块
        nn.modules.module.register_module_forward_hook(global_forward_hook)
        # 注册局部前向钩子到模块
        module.register_forward_hook(local_forward_hook)
        # 注册全局反向钩子到模块
        nn.modules.module.register_module_backward_hook(global_backward_hook)
        # 注册局部反向钩子到模块
        module.register_backward_hook(local_backward_hook)

        # 运行模块前向传播
        output = module(input)
        # 断言各钩子函数已按正确顺序调用
        self.assertTrue(
            local_forward_called
            and local_forward_pre_called
            and global_forward_called
            and global_forward_pre_called
        )

        # 运行模块反向传播
        output.backward(torch.ones(5, 5), retain_graph=True)
        # 断言各反向钩子函数已按正确顺序调用
        self.assertTrue(local_backward_called and global_backward_called)
# 定义一个测试类 TestModuleHookNN，继承自 NNTestCase
class TestModuleHookNN(NNTestCase):
    # 开启 CUDA 内存泄漏检查标志
    _do_cuda_memory_leak_check = True
    # 开启使用非默认 CUDA 流标志
    _do_cuda_non_default_stream = True

    # 定义测试方法 test_hooks
    def test_hooks(self):
        # 调用内部方法 _test_hooks，传入参数 "register_backward_hook"，"register_full_backward_hook" 和 "register_full_backward_pre_hook"
        self._test_hooks("register_backward_hook")
        self._test_hooks("register_full_backward_hook")
        self._test_hooks("register_full_backward_pre_hook")

    # 定义测试方法 test_hook_cpp
    def test_hook_cpp(self):
        # 创建 nn.BatchNorm1d 类的实例 bn
        bn = nn.BatchNorm1d(5)

        # 定义一个钩子函数 hook
        def hook(module, grad_inputs, grad_outputs):
            # 断言 grad_inputs 的长度为 1
            self.assertEqual(len(grad_inputs), 1)
            # 断言 grad_outputs 的长度为 1
            self.assertEqual(len(grad_outputs), 1)
            # 断言 module 是 bn
            self.assertEqual(module, bn)

        # 注册 full_backward_hook 钩子函数 hook 到 bn
        bn.register_full_backward_hook(hook)
        # 对输入为随机 5x5 的张量进行前向传播
        output = bn(torch.randn(5, 5, requires_grad=True))
        # 对输出进行求和并进行反向传播
        output.sum().backward()

    # 定义测试方法 test_backward_hooks_interaction
    def test_backward_hooks_interaction(self):
        # 创建一个 nn.Sigmoid 的实例 module
        module = torch.nn.Sigmoid()

        # 定义一个计数器 cnt，用于记录 backward_cnt 的次数
        cnt = {"backward_cnt": 0}

        # 定义 full_backward_pre_hook 钩子函数 bw_pre_hook
        def bw_pre_hook(m, grad_output):
            # 每次调用增加计数器 backward_cnt 的值
            cnt["backward_cnt"] += 1
            # 返回一个元组，包含 grad_output[0] 的一半
            return (grad_output[0] * 0.5,)

        # 定义 full_backward_hook 钩子函数 bw_hook
        def bw_hook(m, grad_in, grad_output):
            # 断言 grad_output[0] 的值与 torch.full_like(grad_output[0], 0.5) 相等
            self.assertEqual(torch.full_like(grad_output[0], 0.5), grad_output[0])
            # 每次调用增加计数器 backward_cnt 的值
            cnt["backward_cnt"] += 1
            # 返回 grad_output
            return grad_output

        # 注册 full_backward_pre_hook 钩子函数 bw_pre_hook 到 module
        module.register_full_backward_pre_hook(bw_pre_hook)
        # 注册 full_backward_hook 钩子函数 bw_hook 到 module
        module.register_full_backward_hook(bw_hook)

        # 创建一个全为 1 的张量 t，并标记为需要梯度计算
        t = torch.ones(1, 2, requires_grad=True)
        # 对 module(t) 进行前向传播，求和并进行反向传播
        module(t).sum().backward()
        # 断言 backward_cnt 的值为 2
        self.assertEqual(cnt["backward_cnt"], 2)

    # 定义测试方法 test_hook_invalid_outputs
    def test_hook_invalid_outputs(self):
        # 创建 nn.Sigmoid 的实例 module
        module = nn.Sigmoid()
        # 创建一个随机张量 input，形状为 5x5，并标记为需要梯度计算
        input = torch.randn(5, 5, requires_grad=True)

        # 定义一个返回非法 grad_input 的 backward_hook 函数 bw_fail1
        def bw_fail1(self, grad_input, grad_output):
            return grad_input[:-1]

        # 定义一个返回非法 grad_input 的 backward_hook 函数 bw_fail2
        def bw_fail2(self, grad_input, grad_output):
            return grad_input + (torch.randn(2, 2),)

        # 注册 bw_fail1 到 module 的 backward_hook，并使用断言检测 RuntimeError 异常信息
        with module.register_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, "got 0, but expected 1"):
                # 对 module(input) 进行前向传播，求和并进行反向传播
                module(input).sum().backward()

        # 注册 bw_fail2 到 module 的 backward_hook，并使用断言检测 RuntimeError 异常信息
        with module.register_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, "got 2, but expected 1"):
                # 对 module(input) 进行前向传播，求和并进行反向传播
                module(input).sum().backward()

        # 定义一个返回非法 grad_output 的 full_backward_pre_hook 函数 bw_pre_fail1
        def bw_pre_fail1(self, grad_output):
            return ()

        # 定义一个返回非法 grad_output 的 full_backward_pre_hook 函数 bw_pre_fail2
        def bw_pre_fail2(self, grad_output):
            return grad_output + (torch.randn(2, 2),)

        # 注册 bw_pre_fail1 到 module 的 full_backward_pre_hook，并使用断言检测 RuntimeError 异常信息
        with module.register_full_backward_pre_hook(bw_pre_fail1):
            with self.assertRaisesRegex(RuntimeError, "got 0, but expected 1"):
                # 对 module(input) 进行前向传播，求和并进行反向传播
                module(input).sum().backward()

        # 注册 bw_pre_fail2 到 module 的 full_backward_pre_hook，并使用断言检测 RuntimeError 异常信息
        with module.register_full_backward_pre_hook(bw_pre_fail2):
            with self.assertRaisesRegex(RuntimeError, "got 2, but expected 1"):
                # 对 module(input) 进行前向传播，求和并进行反向传播
                module(input).sum().backward()
    def test_hook_requires_grad(self):
        # 复制测试对象的引用
        test_self = self

        # 定义一个继承自 nn.Module 的自定义模块
        class MyModule(nn.Module):
            # 定义前向传播函数
            def forward(self, arg1, arg2, arg3):
                # 断言第一个参数需要梯度
                test_self.assertTrue(arg1.requires_grad)
                # 断言第二个参数不需要梯度
                test_self.assertFalse(arg2.requires_grad)
                # 断言第三个参数需要梯度
                test_self.assertTrue(arg3.requires_grad)
                # 返回三个参数求和的结果
                return arg1.sum() + arg2.sum() + arg3.sum()

        # 创建一个需要梯度的随机输入张量
        inp = torch.rand(2, requires_grad=True)
        # 实例化自定义模块
        mod = MyModule()

        # 调用模块的前向传播两次，触发钩子
        mod(inp, inp.detach(), inp)
        # 注册一个完全反向传播钩子，确保梯度需求正确传播
        mod.register_full_backward_hook(lambda mod, gI, gO: None)
        mod(inp, inp.detach(), inp)

    def test_hook_no_requires_grad(self):
        # 创建一个线性层模块
        mod = nn.Linear(2, 3)

        # 创建一个随机输入张量
        inp = torch.rand(1, 2)

        # 设置返回值变量和钩子被调用次数计数器
        return_val = "None"
        hook_called = [0]

        # 定义一个钩子函数
        def hook(mod, grad_input, grad_output):
            # 增加钩子被调用次数
            hook_called[0] += 1
            # 断言梯度输入列表中的所有项都为空
            for gI in grad_input:
                self.assertIsNone(gI)
            # 断言梯度输出列表中的所有项的大小为 (1, 3)
            for gO in grad_output:
                self.assertEqual(gO.size(), (1, 3))

            # 根据返回值变量的不同取值，返回相应的值
            if return_val == "grad_input":
                return grad_input
            elif return_val == "invalid":
                # 如果输入张量需要梯度，这将是一个有效的返回值
                return inp
            elif return_val == "None":
                return None
            else:
                raise RuntimeError("Invalid return_val string")

        # 注册完全反向传播钩子
        mod.register_full_backward_hook(hook)

        # 运行模块的前向传播、求和并反向传播，触发钩子
        mod(inp).sum().backward()
        # 断言钩子被调用一次
        self.assertEqual(hook_called[0], 1)

        # 修改返回值变量为 "grad_input"
        return_val = "grad_input"

        # 再次运行前向传播、求和并反向传播，触发钩子
        mod(inp).sum().backward()
        # 断言钩子被调用两次
        self.assertEqual(hook_called[0], 2)

        # 修改返回值变量为 "invalid"
        return_val = "invalid"
        # 使用断言确保当输入需要梯度时抛出运行时错误
        with self.assertRaisesRegex(RuntimeError, "where no input requires gradient"):
            mod(inp).sum().backward()

    def test_hook_last_arg_requires_grad(self):
        # 创建一个 L1 损失函数模块
        mod = nn.L1Loss()
        # 创建一个需要梯度的随机输入张量
        inp = torch.rand(1, requires_grad=True)
        # 注册完全反向传播钩子
        mod.register_full_backward_hook(lambda m, gI, gO: None)

        try:
            # 运行模块的前向传播、反向传播
            mod(inp.detach(), inp)
        except Exception as ex:
            # 如果捕获到异常，测试失败
            self.fail(f"Unexpected exception: {ex}")

    def test_hook_extra_input(self):
        # 定义一个继承自 nn.Module 的自定义模块
        class MyModule(nn.Module):
            # 定义前向传播函数，接受一个非张量和一个需要梯度的张量作为输入
            def forward(self, non_tensor, tensor):
                # 返回张量的克隆和非张量输入
                return tensor.clone(), non_tensor

        # 创建一个需要梯度的随机输入张量
        inp = torch.rand(2, requires_grad=True)
        # 实例化自定义模块
        mod = MyModule()

        # 定义一个钩子函数
        def hook(mod, grad_input, grad_output):
            # 断言梯度输入列表的第一项为空
            self.assertIsNone(grad_input[0])
            # 断言梯度输入列表的第二项是张量类型
            self.assertIsInstance(grad_input[1], torch.Tensor)

            # 断言梯度输出列表的第一项是张量类型
            self.assertIsInstance(grad_output[0], torch.Tensor)
            # 断言梯度输出列表的第二项为空
            self.assertIsNone(grad_output[1])

        # 注册完全反向传播钩子
        mod.register_full_backward_hook(hook)
        # 运行模块的前向传播，获取输出和非张量输入
        out, _ = mod(True, inp)
        # 对输出结果求和并执行反向传播
        out.sum().backward()
    # 定义一个名为 test_hook_inplace 的测试方法
    def test_hook_inplace(self):
        # 定义一个继承自 nn.Module 的内部类 MyModule
        class MyModule(nn.Module):
            # 重写 forward 方法，接受输入 inp 和布尔值 do_inplace
            def forward(self, inp, do_inplace):
                # 将输入 inp 存储到当前对象的 inp 属性中
                self.inp = inp
                # 如果 do_inplace 为真，对输入 inp 原地加一
                if do_inplace:
                    inp += 1
                # 返回 inp 的克隆
                return inp.clone()

        # 初始化一个列表 hook_called，用于记录钩子函数调用次数
        hook_called = [0]

        # 定义一个钩子函数 hook，用于接收模块、梯度输入和梯度输出
        def hook(mod, grad_input, grad_output):
            hook_called[0] += 1

        # 定义一个预钩子函数 hook_pre，用于接收模块和梯度输出
        def hook_pre(mod, grad_output):
            hook_called[0] += 1

        # 生成一个随机张量 inp，形状为 (10,)，并设置 requires_grad=True
        inp = torch.rand(10, requires_grad=True)
        # 实例化 MyModule 类得到模块 mod
        mod = MyModule()

        # 遍历包含钩子函数和注册函数元组的列表
        for hook_fn, register_fn in [
            (hook, mod.register_full_backward_hook),
            (hook_pre, mod.register_full_backward_pre_hook),
        ]:
            # 每次循环前将 hook_called[0] 归零，用于统计钩子函数调用次数
            hook_called[0] = 0
            # 使用 register_fn 注册 hook_fn 作为钩子函数
            with register_fn(hook_fn):
                # 调用模块的 forward 方法，不进行原地操作，计算梯度并反向传播
                mod(inp, False).sum().backward()
                # 断言钩子函数被调用一次
                self.assertEqual(hook_called[0], 1)

                # 复制输入 inp，并进行原地操作，预期抛出运行时错误
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Output 0 of BackwardHookFunctionBackward is "
                    "a view and is being modified inplace.",
                ):
                    mod(inp.clone(), True)

                # 如果尝试在视图被修改后重新使用它，预期抛出运行时错误
                local_inp = inp.clone()
                out = mod(local_inp, False)
                local_inp[0] *= 1
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Output 0 of BackwardHookFunctionBackward is "
                    "a view and its base or another view",
                ):
                    # 任何涉及视图的操作都会在此失败
                    mod.inp + 2

                # 对输出进行原地操作，预期抛出运行时错误
                out = mod(inp, False)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "BackwardHookFunctionBackward is a view "
                    "and is being modified inplace.",
                ):
                    out += 1
    def test_hook_non_full_warning(self):
        def noop(*args):
            pass

        a = torch.rand(2, requires_grad=True)
        b = torch.rand(2, requires_grad=True)

        # Check invalid input container
        class MyModule(nn.Module):
            def forward(self, l):
                return l[0].clone(), l[1].clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning,
            "does not take as input a single Tensor or a tuple of Tensors",
        ):
            m([a, b])

        # Check invalid output container
        class MyModule(nn.Module):
            def forward(self, a, b):
                return [a.clone(), b.clone()]

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning, "does not return a single Tensor or a tuple of Tensors"
        ):
            m(a, b)

        # Check invalid output from different Nodes
        class MyModule(nn.Module):
            def forward(self, a, b):
                return a.clone(), b.clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning, "outputs are generated by different autograd Nodes"
        ):
            m(a, b)

        # Check invalid forward with multiple Nodes
        class MyModule(nn.Module):
            def forward(self, a):
                return a.clone().clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning, "the forward contains multiple autograd Nodes"
        ):
            m(a)

    def test_hook_backward_size(self):
        # Make module with multiple operations in forward
        # And different size for input and outputs
        class MyModule(nn.Module):
            def forward(self, arg1, arg2):
                # Compute intermediate results
                tmp = arg1.sum() * arg2
                tmp = tmp + arg2.sum() * arg1.sum()
                tmp = tmp.sum().view(1)
                tmp = tmp.expand(8).contiguous()
                return tmp

        module = MyModule()
        inp1 = torch.randn(5, 5, requires_grad=True)
        inp2 = torch.randn(10, 10, requires_grad=True)

        def bw_hook(module, grad_input, grad_output):
            # Check backward input size
            self.assertEqual(len(grad_input), 2)
            self.assertEqual(grad_input[0].size(), torch.Size([5, 5]))
            self.assertEqual(grad_input[1].size(), torch.Size([10, 10]))
            # Check backward output size
            self.assertEqual(len(grad_output), 1)
            self.assertEqual(grad_output[0].size(), torch.Size([8]))

        # Register backward hook and perform backward pass
        with module.register_full_backward_hook(bw_hook):
            module(inp1, inp2).sum().backward()
    # 定义一个测试方法，验证反向传播是否可写
    def test_hook_backward_writeable(self):
        # 创建一个 Sigmoid 激活函数的模块
        module = nn.Sigmoid()
        # 创建一个随机张量作为输入，并设置 requires_grad=True，表示需要计算梯度
        input = torch.randn(5, 5, requires_grad=True)
        # 对输入数据应用 Sigmoid 函数
        sig_x = torch.nn.functional.sigmoid(input)

        # 定义一个反向传播钩子函数
        def bw_hook(module, grad_input, grad_output):
            # 检查每个梯度是否为 Tensor 类型
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            # 返回修改后的梯度输入
            return tuple(gi * 2 for gi in grad_input)

        # 注册反向传播钩子函数到 Sigmoid 模块
        module.register_backward_hook(bw_hook)
        # 对模块进行前向传播和反向传播
        module(input).backward(torch.ones(5, 5))
        # 计算预期的梯度值
        expected_grad = sig_x * (1 - sig_x) * 2
        # 断言输入的梯度与预期的梯度值相等
        self.assertEqual(input.grad, expected_grad)

    # 定义一个测试方法，验证前向传播前钩子和前向传播钩子是否可写
    def test_hook_forward_preforward_writable(self):
        # 创建一个 Sigmoid 激活函数的模块
        module = nn.Sigmoid()
        # 创建一个随机张量作为输入，并设置 requires_grad=True，表示需要计算梯度
        input = torch.randn(5, 5, requires_grad=True)
        # 对输入数据应用 Sigmoid 函数
        sig_x = torch.nn.functional.sigmoid(input)

        # 定义一个前向传播前钩子函数
        def forward_pre_hook(m, input):
            # 返回输入的第一个元素经过 ReLU 函数处理后的结果
            return torch.nn.functional.relu(input[0])

        # 定义一个前向传播钩子函数
        def forward_hook(m, input, output):
            # 返回输出的负值
            return -output

        # 注册前向传播前钩子函数到 Sigmoid 模块
        module.register_forward_pre_hook(forward_pre_hook)
        # 注册前向传播钩子函数到 Sigmoid 模块
        module.register_forward_hook(forward_hook)
        # 对模块进行前向传播
        output = module(input)
        # 计算预期的输出结果
        expected_res = -torch.nn.functional.sigmoid(torch.nn.functional.relu(input))
        # 断言模块的输出与预期的输出结果相等
        self.assertEqual(output, expected_res)
        # 对模块的输出进行反向传播，保留计算图
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        # 创建一个掩码，选择输入大于零的元素
        mask = input > 0
        # 计算预期的梯度值
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        # 断言输入的梯度与预期的梯度值相等
        self.assertEqual(input.grad, expected_grad)

    # 定义一个测试方法，验证缓冲区注册钩子是否可用
    def test_hook_buffer_registration(self):
        # 遍历每个返回缓冲区的情况
        for return_buffer in (True, False):

            # 定义一个缓冲区注册钩子函数
            def buffer_registration_hook(module, name, buffer):
                # 注册缓冲区并设置 registered 属性为 True
                buffer.registered = True
                # 如果需要返回缓冲区，则返回它
                if return_buffer:
                    return buffer

            # 注册模块缓冲区注册钩子函数
            handle = torch.nn.modules.module.register_module_buffer_registration_hook(
                buffer_registration_hook
            )
            try:
                # 创建一个基本网络并获取其所有缓冲区
                l, n, s = _create_basic_net()
                for b in s.buffers():
                    # 断言每个缓冲区的 registered 属性为 True
                    self.assertTrue(getattr(b, "registered", False))
            finally:
                # 移除模块缓冲区注册钩子函数
                handle.remove()

    # 定义一个测试方法，验证子模块注册钩子是否可用
    def test_hook_submodule_registration(self):
        # 遍历每个返回子模块的情况
        for return_submodule in (True, False):

            # 定义一个子模块注册钩子函数
            def module_registration_hook(module, name, submodule):
                # 设置模块和子模块的 registered 属性为 True
                module.registered = True
                submodule.registered = True
                # 如果需要返回子模块，则返回它
                if return_submodule:
                    return submodule

            # 注册模块子模块注册钩子函数
            handle = torch.nn.modules.module.register_module_module_registration_hook(
                module_registration_hook
            )
            try:
                # 创建一个基本网络并获取其所有模块
                l, n, s = _create_basic_net()
                for m in s.modules():
                    # 断言每个模块和子模块的 registered 属性为 True
                    self.assertTrue(getattr(m, "registered", False))
            finally:
                # 移除模块子模块注册钩子函数
                handle.remove()
    def test_hook_parameter_registration(self):
        # 针对返回参数为True和False两种情况，进行循环测试
        for return_parameter in (True, False):

            # 定义一个参数注册钩子函数
            def parameter_registration_hook(module, name, parameter):
                # 将参数的registered属性设置为True
                parameter.registered = True
                # 如果return_parameter为True，则返回当前参数
                if return_parameter:
                    return parameter

            # 注册参数注册钩子函数到torch.nn.modules.module模块，获取句柄
            handle = (
                torch.nn.modules.module.register_module_parameter_registration_hook(
                    parameter_registration_hook
                )
            )
            try:
                # 创建一个基本网络，获取其层、节点、参数的元组
                l, n, s = _create_basic_net()
                # 遍历网络s的所有参数
                for p in s.parameters():
                    # 断言每个参数的registered属性为True
                    self.assertTrue(getattr(p, "registered", False))
            finally:
                # 移除参数注册钩子函数的句柄，确保清理工作
                handle.remove()
# 实例化参数化测试，针对 TestModuleHooks 类
instantiate_parametrized_tests(TestModuleHooks)

# 实例化参数化测试，针对 TestStateDictHooks 类
instantiate_parametrized_tests(TestStateDictHooks)

# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```