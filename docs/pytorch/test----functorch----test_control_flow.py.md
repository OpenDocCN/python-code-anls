# `.\pytorch\test\functorch\test_control_flow.py`

```
# 导入所需的模块和库
import contextlib
import functools
import unittest
import torch
import torch.utils._pytree as pytree
from functorch.experimental import control_flow
from functorch.experimental.control_flow import cond, UnsupportedAliasMutationException
from torch._higher_order_ops.while_loop import while_loop
from torch._subclasses.functional_tensor import (
    CppFunctionalizeAPI,
    FunctionalTensor,
    FunctionalTensorMode,
    PythonFunctionalizeAPI,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xfailIfTorchDynamo,
)

# 定义转换函数
def to_fun(t):
    if isinstance(t, torch.Tensor):
        return FunctionalTensor.to_functional(t)
    return t

# 定义反向转换函数
def from_fun(t):
    if not isinstance(t, FunctionalTensor):
        # 快速检查断言
        if isinstance(t, torch.Tensor):
            assert not torch._is_functional_tensor(t)
        return t
    torch._sync(t)
    return torch._from_functional_tensor(t.elem)

# 定义旧版本的转换函数
def to_fun_old(t):
    if isinstance(t, torch.Tensor) and not torch._is_functional_tensor(t):
        out = torch._to_functional_tensor(t)
        torch._mirror_autograd_meta_to(t, out)
        return out
    return t

# 定义旧版本的反向转换函数
def from_fun_old(t):
    # 快速检查断言
    if isinstance(t, torch.Tensor):
        assert torch._is_functional_tensor(t)
        torch._sync(t)
        return torch._from_functional_tensor(t)
    return t

# 定义伪造的映射函数
def _fake_map(f, x, *args):
    from functorch.experimental.control_flow import _stack_pytree, _unstack_pytree
    x_pytrees = _unstack_pytree(x)
    zs = []
    for xp in x_pytrees:
        zs.append(f(xp, *args))
    return _stack_pytree(zs)

# 定义伪造的循环函数
def _fake_while_loop(cond_fn, body_fn, operands):
    while cond_fn(*operands):
        operands = body_fn(*operands)
    return operands

# 定义简单的循环测试函数
def _while_loop_tests():
    def simple(x):
        def cond_fn(x):
            return x.sum() < 10

        def body_fn(x):
            return (x + 1,)

        return while_loop(cond_fn, body_fn, (x,))

    def simple_with_mutation(x):
        def cond_fn(x):
            y = x.clone().add_(1).add_(-1)
            return y.sum() < 10

        def body_fn(x):
            y = x.clone().add_(1).add_(-1)
            return (y + 1,)

        return while_loop(cond_fn, body_fn, (x,))
    def nested(out_iter, it, y):
        def cond_fn(out_iter, it, y):
            return it.sum() < 10

        def body_fn(out_iter, it, y):
            return (out_iter.clone(), it + y, y + 1)

        def outer_cond_fn(out_iter, it, y):
            return out_iter.sum() < 2

        def outer_body_fn(out_iter, it, y):
            # 调用 while_loop 函数，执行内部循环 cond_fn 和 body_fn，更新 out_iter, it, y
            out_iter, it, y = while_loop(cond_fn, body_fn, (out_iter, it, y))
            return (out_iter + 1, it, y)

        # 调用 while_loop 函数，执行外部循环 outer_cond_fn 和 outer_body_fn，返回更新后的 out_iter
        return while_loop(outer_cond_fn, outer_body_fn, (out_iter, it, y))

    class Nested(torch.nn.Module):
        def forward(self, ci, cj, a, b):
            def cond_fn(i1, j1, x1, y1):
                return i1 > 0

            def body_fn(i1, j1, x1, y1):
                def cond_fn_nested(i2, j2, x2, y2):
                    return j2 > 0

                def body_fn_nested(i2, j2, x2, y2):
                    # 执行内部循环 cond_fn_nested 和 body_fn_nested，更新 i2, j2, x2, y2
                    return i2.clone(), j2 - 1, x2 + 3.14, y2 - 2.71

                # 调用 while_loop 函数，执行内部循环 cond_fn_nested 和 body_fn_nested，更新 i1, j1, x1, y1
                i1, j1, x1, y1 = while_loop(
                    cond_fn_nested, body_fn_nested, [i1, j1, x1, y1]
                )
                return i1 - 1, j1.clone(), x1 * 2, y1 / 2

            # 调用 while_loop 函数，执行外部循环 cond_fn 和 body_fn，更新 ci, cj, a, b
            return while_loop(cond_fn, body_fn, (ci, cj, a, b))

    class SimpleWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("dec", torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                # 调用线性层 self.linear 处理输入 x
                return it - 1, self.linear(x)

            # 调用 while_loop 函数，执行循环 cond_fn 和 body_fn，更新 iter, x
            return while_loop(cond_fn, body_fn, (iter, x))

    class NestedWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = SimpleWithLinear()
            self.outer_linear = torch.nn.Linear(2, 2)
            self.register_buffer("dec", torch.tensor(1))

        def forward(self, iter, x):
            def cond_fn(it, x):
                return it - self.dec > 0

            def body_fn(it, x):
                # 调用内部模块 self.mod 的 forward 方法，处理输入 iter, x，并用外部线性层 self.outer_linear 处理输出
                return it - 1, self.outer_linear(self.mod(iter, x)[1])

            # 调用 while_loop 函数，执行循环 cond_fn 和 body_fn，更新 iter, x
            return while_loop(cond_fn, body_fn, (iter, x))

    nested2 = Nested()
    simple_with_linear = SimpleWithLinear()
    nested_with_linear = NestedWithLinear()

    x = torch.zeros(1)
    y = torch.zeros(1)
    z = torch.zeros(1)
    return {
        "simple": (simple, (x,)),
        "nested": (nested, (x, y, z)),
        "nested2": (
            nested2,
            (torch.tensor(2), torch.tensor(2), torch.ones(2, 2), torch.ones(2, 2)),
        ),
        "simple_with_mutation": (simple_with_mutation, (x,)),
        "simple_with_linear": (
            simple_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
        "nested_with_linear": (
            nested_with_linear,
            (torch.tensor(3), torch.randn(2, 2)),
        ),
    }
# 调用函数 _while_loop_tests() 并将其结果赋值给 WHILE_LOOP_TESTS 变量
WHILE_LOOP_TESTS = _while_loop_tests()

# 收集经过筛选的节点的元数据信息
def collect_meta_for_filtered_nodes(
    gm: torch.fx.GraphModule, node_names, meta_field_name
):
    # 初始化返回结果列表
    ret = []
    # 遍历 GraphModule 实例 gm 中的所有模块
    for mod in gm.modules():
        # 遍历每个模块的图中的所有节点
        for node in mod.graph.nodes:
            # 如果节点的名称在给定的 node_names 中
            if node.name in node_names:
                # 遍历每个元数据字段名
                for field_name in meta_field_name:
                    # 将节点的特定字段名的元数据值添加到返回结果列表中
                    ret.append(node.meta.get(field_name))
    # 返回收集到的所有节点的特定元数据值列表
    return ret


# 定义一个简单的求和函数 reduce_func，接受任意数量的参数并返回它们的和
def reduce_func(*operands):
    acc = 0
    for operand in operands:
        acc += operand
    return acc


# 定义一个可调用的类 ReduceObj，用于将其实例视为函数，并在调用时调用 reduce_func 函数
class ReduceObj:
    def __call__(self, *operands):
        return reduce_func(*operands)


# 定义一个继承自 torch.nn.Module 的类 ReduceMod，实现了 _reduce 方法用于调用 reduce_func 函数
class ReduceMod(torch.nn.Module):
    def _reduce(self, *operands):
        return reduce_func(*operands)

    # 实现 Module 类的 forward 方法，用于调用 _reduce 方法
    def forward(self, *operands):
        return self._reduce(*operands)


# 使用 unittest 模块的装饰器 @unittest.skipIf，用于标记测试类 TestControlFlow
# 当 IS_WINDOWS 为 True 时跳过测试，注明 Windows 系统不支持该测试
@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
@skipIfNoDynamoSupport
# 定义测试类 TestControlFlow，继承自 unittest.TestCase
class TestControlFlow(TestCase):
    # 在每个测试方法运行之前调用，重置 torch._dynamo 状态
    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

    # 定义测试方法 test_cond_no_trace，测试条件分支控制流的无跟踪情况
    def test_cond_no_trace(self):
        # 定义一个返回输入张量的正弦值的函数 true_fn
        def true_fn(x):
            return x.sin()

        # 定义一个返回输入张量的余弦值的函数 false_fn
        def false_fn(x):
            return x.cos()

        # 生成一个形状为 (4,) 的随机张量 x
        x = torch.randn(4)
        # 调用 cond 函数，根据条件调用 true_fn 或 false_fn 函数，并传入参数 x
        result = cond(False, true_fn, false_fn, [x])
        # 断言 result 结果等于 x 的余弦值
        self.assertEqual(result, torch.cos(x))

    # 使用 unittest 模块的装饰器 @unittest.skipIf，用于标记测试方法
    # 当 CUDA 不可用时跳过该测试，注明该测试需要 CUDA 环境
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    # 定义测试方法 test_cond_gpu，测试条件分支控制流在 GPU 上的情况
    def test_cond_gpu(self):
        # 定义一个返回输入张量的正弦值的函数 true_fn
        def true_fn(x):
            return x.sin()

        # 定义一个返回输入张量的余弦值的函数 false_fn
        def false_fn(x):
            return x.cos()

        # 生成一个形状为 (4,) 的随机张量 x，位于 CUDA 设备上
        x = torch.randn(4, device="cuda")
        # 生成一个位于 CUDA 设备上的布尔张量 pred
        pred = torch.tensor(False, device="cuda")
        # 调用 cond 函数，根据条件调用 true_fn 或 false_fn 函数，并传入参数 x
        result = cond(pred, true_fn, false_fn, [x])
        # 断言 result 结果等于 x 的余弦值
        self.assertEqual(result, torch.cos(x))

    # 使用 unittest 模块的装饰器 @unittest.skipIf，用于标记测试方法
    # 当 CUDA 不可用时跳过该测试，注明该测试需要 CUDA 环境
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    # 定义测试方法 test_map_gpu，测试在 GPU 上的映射操作
    def test_map_gpu(self):
        # 定义一个接受两个输入张量并返回它们的和的函数 f
        def f(x, y):
            return x + y

        # 生成一个形状为 (3, 2, 2) 的全 1 张量 xs，位于 CUDA 设备上
        xs = torch.ones(3, 2, 2, device="cuda")
        # 生成一个形状为 (2,) 的全 1 张量 y，位于 CUDA 设备上
        y = torch.ones(2, device="cuda")
        # 调用 control_flow 模块的 map 函数，对 xs 和 y 进行映射操作
        res = control_flow.map(f, xs, y)
        # 调用 _fake_map 函数，对 xs 和 y 进行模拟的映射操作
        expected = _fake_map(f, xs, y)
        # 断言 res 结果等于 expected
        self.assertEqual(expected, res)

    # 使用 unittest 模块的装饰器 @unittest.skipIf，用于标记测试方法
    # 当 CUDA 不可用时跳过该测试，注明该测试需要 CUDA 环境
    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA.")
    # 定义测试方法 test_while_loop_gpu，测试在 GPU 上的 while 循环操作
    def test_while_loop_gpu(self):
        # 定义一个接受输入张量并返回是否满足条件的函数 cond_fn
        def cond_fn(x):
            return x.sum() < 10

        # 定义一个接受输入张量并返回循环体计算结果的函数 body_fn
        def body_fn(x):
            return (x + 1,)

        # 生成一个形状为 (1,) 的全 0 张量 x，位于 CUDA 设备上
        x = torch.zeros(1, device="cuda")
        # 调用 while_loop 函数，在 CUDA 设备上执行循环，根据条件调用 body_fn 函数
        res = while_loop(cond_fn, body_fn, (x,))
        # 调用 _fake_while_loop 函数，对 CUDA 设备上的循环进行模拟
        expected = _fake_while_loop(cond_fn, body_fn, (x,))
        # 断言 res 结果等于 expected
        self.assertEqual(expected, res)
    def test_map_illegal_inputs(self):
        # 定义一个函数 f，计算两个元素和以及一个额外的输入，并返回结果
        def f(x, y):
            return x[0] + x[1] + y

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误消息是否匹配特定模式
        with self.assertRaisesRegex(
            RuntimeError,
            r"Mapped xs can only consist of tensors\. Got xs \[3, tensor\(\[1\., 1\.\]\)\]\.",
        ):
            # 调用 control_flow.map 函数，期望抛出 RuntimeError 异常
            _ = control_flow.map(f, (3, torch.ones(2)), torch.ones(2))

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误消息是否匹配特定模式
        with self.assertRaisesRegex(
            RuntimeError, r"Leading dimensions of mapped xs cannot be 0\."
        ):
            # 调用 control_flow.map 函数，期望抛出 RuntimeError 异常
            _ = control_flow.map(
                f, (torch.ones(0, 1, 2), torch.ones(0, 1, 2)), torch.ones(2)
            )

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误消息是否匹配特定模式
        with self.assertRaisesRegex(
            RuntimeError,
            r"Leading dimensions of mapped xs must be consistent\. "
            r"Got shapes \[torch\.Size\(\[3, 4, 5\]\), torch\.Size\(\[4, 4, 5\]\)\]\.",
        ):
            # 调用 control_flow.map 函数，期望抛出 RuntimeError 异常
            _ = control_flow.map(
                f, (torch.ones(3, 4, 5), torch.ones(4, 4, 5)), torch.ones(5)
            )

    def test_map_illegal_outputs(self):
        # 定义一个函数 f，返回第一个输入的标量值
        def f(x, y):
            return x.item()

        # 定义一个函数 f1，返回第二个输入的尺寸
        def f1(x, y):
            return y.size()

        # 定义一个函数 f2，返回 None
        def f2(x, y):
            return None

        # 创建张量 x 和 y
        x = torch.ones([3])
        y = torch.ones([1, 2, 3])
        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误消息是否匹配特定模式
        with self.assertRaisesRegex(
            RuntimeError, r"Expect outputs of map only contains tensors or None\."
        ):
            # 调用 control_flow.map 函数，期望抛出 RuntimeError 异常
            _ = control_flow.map(f, x, y)

        # 使用 assertRaisesRegex 断言捕获 RuntimeError 异常，并验证错误消息是否匹配特定模式
        with self.assertRaisesRegex(
            RuntimeError, r"Expect outputs of map only contains tensors or None\."
        ):
            # 调用 control_flow.map 函数，期望抛出 RuntimeError 异常
            out = control_flow.map(f1, x, y)

        # 对于返回 None 的情况，不应该抛出异常
        _ = control_flow.map(f2, x, y)

    def test_map_list_in_out(self):
        # 定义一个函数 f，接收两个输入并返回一个包含一个列表的列表
        def f(x, y):
            return [[x[0][0] + y]]

        # 创建输入列表 xs 和张量 y
        xs = [[torch.ones(3, 2, 2)]]
        y = torch.ones(2)
        # 调用 control_flow.map 函数，对结果进行验证
        res = control_flow.map(f, xs, y)
        # 调用 _fake_map 函数以获取期望结果
        expected = _fake_map(f, xs, y)
        # 使用 self.assertEqual 断言检查结果的长度和内容
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(expected, res)

    def test_map_dict_in_out(self):
        # 定义一个函数 f，接收两个输入并返回一个包含单个键值对的字典
        def f(x, y):
            return {"c": x["a"]["b"] + y}

        # 创建输入字典 xs 和张量 y
        xs = {"a": {"b": torch.ones(3, 2, 2)}}
        y = torch.ones(2)
        # 调用 control_flow.map 函数，对结果进行验证
        res = control_flow.map(f, xs, y)
        # 调用 _fake_map 函数以获取期望结果
        expected = _fake_map(f, xs, y)
        # 使用 self.assertEqual 断言检查结果的长度和内容
        self.assertEqual(len(res), 1)
        self.assertTrue("c" in res)
        self.assertEqual(expected, res)

    def test_map_autograd_simple(self):
        # 定义一个函数 f，对两个输入张量执行一系列数学运算
        def f(x, y):
            return x.sin().cos() * y.cos().sin()

        # 创建输入张量 xs 和 y，并启用梯度跟踪
        xs = torch.ones(3, 2, 2, requires_grad=True)
        y = torch.ones(2, requires_grad=True)
        # 调用 control_flow.map 函数，对结果进行验证
        res = control_flow.map(f, xs, y)
        # 调用 _fake_map 函数以获取期望结果
        expected_res = _fake_map(f, xs, y)
        # 创建梯度输出张量
        grad_out = torch.ones_like(res)
        # 计算结果张量和梯度张量的梯度
        grads = torch.autograd.grad(res, (xs, y), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs, y), grad_out)
        # 使用 self.assertEqual 断言检查结果和梯度的一致性
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)
    def test_map_autograd_simple_partial_grad(self):
        # 定义一个函数 f，接受两个参数 x 和 y，返回它们的复合函数结果
        def f(x, y):
            return x.sin().cos() * y.cos().sin()

        # 创建一个形状为 (3, 2, 2) 的张量 xs，要求计算梯度
        xs = torch.ones(3, 2, 2, requires_grad=True)
        # 创建一个形状为 (2,) 的张量 y，并禁用其梯度计算
        y = torch.ones(2, requires_grad=False)
        # 调用控制流函数 map，将函数 f 应用于 xs 和 y 上
        res = control_flow.map(f, xs, y)
        # 调用一个辅助函数 _fake_map，对比其返回结果和 res
        expected_res = _fake_map(f, xs, y)
        # 创建一个与 res 相同形状的张量 grad_out，梯度全为 1
        grad_out = torch.ones_like(res)
        # 计算 res 对 xs 的梯度
        grads = torch.autograd.grad(res, (xs,), grad_out)
        # 计算 expected_res 对 xs 的梯度
        expected_grads = torch.autograd.grad(expected_res, (xs,), grad_out)
        # 使用断言检查 res 和 expected_res 是否相等
        self.assertEqual(expected_res, res)
        # 使用断言检查 grads 和 expected_grads 是否相等
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_no_grad_output(self):
        # 定义一个函数 f，接受两个参数 x 和 y，返回一个元组
        def f(x, y):
            return x[0].sin().cos() + y, y.cos().sin()

        # 创建一个包含两个张量的列表 xs，都要求计算梯度
        xs = [torch.ones(3, 2, 2, requires_grad=True), torch.ones(3, 3)]
        # 创建一个形状为 (2,) 的张量 y，并禁用其梯度计算
        y = torch.ones(2, requires_grad=False)
        # 调用控制流函数 map，将函数 f 应用于 xs 和 y 上
        res = control_flow.map(f, xs, y)
        # 调用一个辅助函数 _fake_map，对比其返回结果和 res
        expected_res = _fake_map(f, xs, y)
        # 创建一个与 res 的第一个元素形状相同的张量 grad_out，梯度全为 1
        grad_out = torch.ones_like(res[0])
        # 计算 res 的第一个元素对 xs[0] 的梯度
        grads = torch.autograd.grad(res[0], (xs[0],), grad_out)
        # 计算 expected_res 的第一个元素对 xs[0] 的梯度
        expected_grads = torch.autograd.grad(expected_res[0], (xs[0],), grad_out)
        # 使用断言检查 res 和 expected_res 是否相等
        self.assertEqual(expected_res, res)
        # 使用断言检查 grads 和 expected_grads 是否相等
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_nested_list(self):
        # 导入 pytree 模块中的 tree_leaves 函数
        import torch.utils._pytree as pytree

        # 定义一个函数 f，接受两个参数 x 和 y，返回一个嵌套列表
        def f(x, y):
            a, b = x
            c, d = a
            return [[b.sin() * c.cos()], d.sin() * y.cos()]

        # 定义一个函数 fwbw，接受 map_op、f、x 和 y 作为参数
        def fwbw(map_op, f, x, y):
            # 调用 map_op 函数，将函数 f 应用于 x 和 y 上
            z = map_op(f, x, y)
            # 对 x 和 z 进行展平操作
            flat_x = pytree.tree_leaves(x)
            flat_z = pytree.tree_leaves(z)
            # 计算 z 对 flat_x 的梯度，grad_out 是一个全为 1 的张量列表
            grads = torch.autograd.grad(
                flat_z, flat_x, [torch.ones_like(z) for z in flat_z]
            )
            # 返回 z 和 grads
            return z, grads

        # 创建一个嵌套列表 x，包含两个张量
        x = [
            [
                torch.randn(3, 2, 2, requires_grad=True),
                torch.randn(3, 2, 1, requires_grad=True),
            ],
            torch.ones(3, 1, 2, requires_grad=True),
        ]
        # 创建一个形状为 (1,) 的张量 y，要求计算梯度
        y = torch.ones(1, requires_grad=True)
        # 调用 test_map_autograd_nested_list 函数，获取真实输出和虚假输出
        true_outs = fwbw(control_flow.map, f, x, y)
        fake_outs = fwbw(_fake_map, f, x, y)
        # 使用断言检查 true_outs 和 fake_outs 是否相等
        self.assertEqual(true_outs, fake_outs)
# 如果在 Windows 上运行测试，则跳过，因为此测试不支持 Windows
@unittest.skipIf(IS_WINDOWS, "Windows not supported for this test")
# 如果没有 Dynamo 支持，则跳过此测试
@skipIfNoDynamoSupport
# 定义测试类 TestControlFlowTraced，继承自 TestCase
class TestControlFlowTraced(TestCase):
    # 设置测试环境的初始化方法
    def setUp(self):
        # 重置 Torch 的 Dynamo 状态
        torch._dynamo.reset()
        # 调用父类的 setUp 方法初始化
        super().setUp()

    # 检查追踪功能的私有方法，fn 是函数，args 是参数列表，allow_non_fake_inputs 表示是否允许非虚拟输入
    def _check_tracing(self, fn, args, allow_non_fake_inputs=False):
        # 用传入的参数调用原始函数 fn，并保存结果
        eager_res = fn(*args)
        # 创建一个空字典来保存不同追踪模式的图
        graphs = {}
        # 遍历三种追踪模式：符号化、真实、虚拟
        for tracing_mode in ["symbolic", "real", "fake"]:
            # 使用 make_fx 函数创建追踪后的图形，传入函数 fn、追踪模式、是否允许非虚拟输入
            graph = make_fx(
                fn,
                tracing_mode=tracing_mode,
                _allow_non_fake_inputs=allow_non_fake_inputs,
            )(*args)
            # 将当前追踪模式的图保存到字典中
            graphs[tracing_mode] = graph
            # 断言当前模式下的图执行结果与原始函数的执行结果相等
            self.assertEqual(graph(*args), eager_res)
        # 返回保存所有追踪模式图的字典
        return graphs

    # 检查编译功能的私有方法，fn 是函数，args 是参数列表，backend 是编译后端，默认为 "eager"
    def _check_compile(self, fn, args, *, backend="eager"):
        # 用传入的参数调用原始函数 fn，并保存结果
        eager_res = fn(*args)
        # 编译函数 fn 使用指定的后端
        compiled_fn = torch.compile(fn, backend=backend)
        # 断言编译后的函数执行结果与原始函数的执行结果相等
        self.assertEqual(compiled_fn(*args), eager_res)

    # 测试条件控制追踪不嵌套的情况
    def test_cond_traced_not_nested(self):
        # 定义一个返回输入张量正弦值的真函数
        def true_fn(x):
            return x.sin()

        # 定义一个返回输入张量余弦值的假函数
        def false_fn(x):
            return x.cos()

        # 定义一个条件函数 f，根据 y 的值选择调用真函数或假函数
        def f(x, y):
            return cond(y, true_fn, false_fn, [x])

        # 创建一个随机张量 x
        x = torch.randn(4)
        # 使用 make_fx 函数创建符号化追踪的图形
        graph = make_fx(f)(x, torch.tensor(False))
        # 使用符号化追踪的图形计算结果，分别使用 True 和 False 作为 y 的值
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        # 断言不相等，即使用 True 和 False 作为 y 的值时结果应该不同
        self.assertFalse(torch.allclose(result_true, result_false))
        # 断言使用 True 作为 y 的值时的结果应为 x 的正弦值
        self.assertEqual(result_true, torch.sin(x))
        # 断言使用 False 作为 y 的值时的结果应为 x 的余弦值
        self.assertEqual(result_false, torch.cos(x))

        # 使用符号化追踪模式创建图形，并断言其执行结果与原始函数的执行结果相等
        graph = make_fx(f, tracing_mode="symbolic")(x, torch.tensor(False))
        self.assertEqual(graph(x, torch.tensor(True)), f(x, torch.tensor(True)))

    # 测试嵌套的追踪循环
    def test_while_loop_nested_traced(self):
        # 从 WHILE_LOOP_TESTS 字典中获取测试函数和输入数据
        fn, inp = WHILE_LOOP_TESTS["nested"]
        # 使用 _check_tracing 方法检查追踪结果，并获取不同追踪模式下的图形
        graphs = self._check_tracing(fn, inp)
        # 断言符号化追踪模式下的 while 循环体的代码与预期的代码匹配
        self.assertExpectedInline(
            graphs["symbolic"].code.strip("\n"),
            """\
def forward(self, out_iter_1, it_1, y_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (out_iter_1, it_1, y_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = out_iter_1 = it_1 = y_1 = None
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2];  while_loop = None
    return (getitem, getitem_1, getitem_2)
    """,  # noqa: B950
        )
        # 断言符号化追踪模式下的 while 循环条件的代码与预期的代码匹配
        self.assertExpectedInline(
            graphs["symbolic"].while_loop_cond_graph_0.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    sum_1 = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
    lt = torch.ops.aten.lt.Scalar(sum_1, 2);  sum_1 = None
    return lt
    """,
        )
        # 断言符号化追踪模式下的 while 循环体的代码与预期的代码匹配
        self.assertExpectedInline(
            graphs["symbolic"].while_loop_body_graph_0.code.strip("\n"),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    # 从当前对象中获取名为 while_loop_body_graph_0 的属性，赋值给变量 while_loop_body_graph_0

    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1), ());
    # 调用 Torch 的高阶操作函数 while_loop，传入条件函数 while_loop_cond_graph_0、循环体函数 while_loop_body_graph_0、参数元组 (arg0_1, arg1_1, arg2_1)，返回结果赋给 while_loop

    while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = None
    # 将 while_loop_cond_graph_0、while_loop_body_graph_0、arg0_1、arg1_1、arg2_1 置为 None

    getitem = while_loop[0]
    # 从 while_loop 结果中获取第一个元素，赋值给 getitem

    getitem_1 = while_loop[1]
    # 从 while_loop 结果中获取第二个元素，赋值给 getitem_1

    getitem_2 = while_loop[2];
    # 从 while_loop 结果中获取第三个元素，赋值给 getitem_2

    while_loop = None
    # 将 while_loop 变量置为 None

    add = torch.ops.aten.add.Tensor(getitem, 1);
    # 调用 Torch 的 aten.add.Tensor 函数，将 getitem 和标量值 1 相加，结果赋给 add

    getitem = None
    # 将 getitem 变量置为 None

    return (add, getitem_1, getitem_2)
    # 返回包含 add、getitem_1 和 getitem_2 的元组作为结果
# 定义一个方法 `forward`，接受一个参数 `x_1`
def forward(self, x_1):
    # 将实例属性 `while_loop_cond_graph_0` 赋给局部变量 `while_loop_cond_graph_0`
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    # 将实例属性 `while_loop_body_graph_0` 赋给局部变量 `while_loop_body_graph_0`
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    # 调用 `torch.ops.higher_order.while_loop` 方法进行循环操作，传入条件函数、循环体函数、初始参数列表和空元组
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (x_1,), ())
    # 清空不再需要的变量，释放内存
    while_loop_cond_graph_0 = while_loop_body_graph_0 = x_1 = None
    # 从循环结果中获取第一个元素
    getitem = while_loop[0]
    # 清空不再需要的循环对象，释放内存
    while_loop = None
    # 返回包含获取的元素的元组
    return (getitem,)
        """
        检查条件，如果满足条件，则执行以下代码块；否则，执行另一段代码块。
        """
        if not hasattr(_grammar, "classDef"):
            # 断言符号图的内联部分符合预期，去掉结尾的换行符
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """
                期望符号图的代码块符合预期，去掉结尾的换行符
                """,
            )
        else:
            # 断言预期内联结果
            self.assertExpectedInline(
                graphs["symbolic"].code.strip("\n"),
                """
                期望符号图的代码块符合预期，去掉结尾的换行符
                """,
            )
# 定义一个方法用于实现前向传播
def forward(self, x_1):
    # 获取当前对象中定义的循环条件图和循环体图
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    # 调用高阶操作库中的while_loop函数，执行循环
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (x_1,), ())
    # 解构循环结果，获取第一个元素
    getitem = while_loop[0]
    # 清空变量，释放内存
    while_loop_cond_graph_0 = while_loop_body_graph_0 = x_1 = None
    # 返回结果的元组形式
    return (getitem,)
    # 定义一个测试方法，用于测试带有线性编译检查图的简单 while 循环
    def test_while_loop_simple_with_linear_compile_check_graph(self):
        # 从 WHILE_LOOP_TESTS 中获取简单线性测试用例的函数和输入
        fn, inp = WHILE_LOOP_TESTS["simple_with_linear"]
        # 导入 EagerAndRecordGraphs 类用于测试
        from torch._dynamo.testing import EagerAndRecordGraphs
        
        # 创建一个 EagerAndRecordGraphs 的实例作为后端
        backend = EagerAndRecordGraphs()
        # 调用 torch.compile 函数，使用指定的后端编译函数 fn，并传入输入 inp
        torch.compile(fn, backend=backend)(*inp)
        # 断言生成的图的数量为 1
        self.assertEqual(len(backend.graphs), 1)
        # 获取第一个生成的图
        gm = backend.graphs[0]
        # 如果配置允许内联内置的 nn 模块
        if torch._dynamo.config.inline_inbuilt_nn_modules:
            # 断言生成的代码与预期的代码匹配（去除首尾空格）
            self.assertExpectedInline(
                gm.code.strip(),
                """\
# 定义一个方法 `forward`，接收多个参数，并用类型提示说明参数类型为 torch.Tensor
def forward(self, L_iter_ : torch.Tensor, L_x_ : torch.Tensor, L_self_buffers_dec_ : torch.Tensor, L_self_modules_linear_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_linear_parameters_bias_ : torch.nn.parameter.Parameter):
    # 将参数赋值给本地变量，名字保持一致
    l_iter_ = L_iter_
    l_x_ = L_x_
    l_self_buffers_dec_ = L_self_buffers_dec_
    l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
    l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_
    # 获取类实例的 cond_fn_0 和 body_fn_0 方法
    cond_fn_0 = self.cond_fn_0
    body_fn_0 = self.body_fn_0
    # 调用 torch.ops.higher_order.while_loop 方法执行循环，传入条件函数和循环体函数
    while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (l_iter_, l_x_), (l_self_buffers_dec_, l_self_modules_linear_parameters_bias_, l_self_modules_linear_parameters_weight_))
    # 清空本地变量
    cond_fn_0 = body_fn_0 = l_iter_ = l_x_ = l_self_buffers_dec_ = l_self_modules_linear_parameters_bias_ = l_self_modules_linear_parameters_weight_ = None
    # 获取循环结束后的结果
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    # 清空循环结果
    while_loop = None
    # 返回结果元组
    return (getitem, getitem_1)
    return (getitem, getitem_1)""",  # noqa: B950
            )
            # 断言检查，验证 gm.cond_fn_0.code 的预期内联内容
            self.assertExpectedInline(
                gm.cond_fn_0.code.strip(),
                """\
# 定义一个方法，处理输入参数并执行一系列计算和操作，返回结果元组
def forward(self, l_iter_, l_x_, l__self___dec_cond_fn, l__self___linear_bias_body_fn, l__self___linear_weight_body_fn):
    # 计算 l_iter_ 减去 l__self___dec_cond_fn 的结果，并将其赋给 sub，然后清空 l_iter_ 和 l__self___dec_cond_fn 的引用
    sub = l_iter_ - l__self___dec_cond_fn;  l_iter_ = l__self___dec_cond_fn = None
    # 判断 sub 是否大于 0，并将结果赋给 gt，然后清空 sub 的引用
    gt = sub > 0;  sub = None
    # 返回 gt，即上述比较的结果
    return gt""",  # noqa: B950
        )
        # 调用自定义的断言方法，验证生成的内联代码与预期的匹配
        self.assertExpectedInline(
            gm.body_fn_0.code.strip(),
            """\
def forward(self, l_iter_, l_x_, l__self___dec_cond_fn, l__self___linear_bias_body_fn, l__self___linear_weight_body_fn):
    # 将 l_iter_ 减去 1 的结果赋给 sub，并清空 l_iter_ 的引用
    sub = l_iter_ - 1;  l_iter_ = None
    # 使用 torch._C._nn.linear 函数对 l_x_ 进行线性变换，使用 l__self___linear_weight_body_fn 和 l__self___linear_bias_body_fn 作为参数，并将结果赋给 linear
    linear = torch._C._nn.linear(l_x_, l__self___linear_weight_body_fn, l__self___linear_bias_body_fn);  l_x_ = l__self___linear_weight_body_fn = l__self___linear_bias_body_fn = None
    # 返回 sub 和 linear 构成的元组
    return (sub, linear)""",  # noqa: B950
        )

# 定义一个测试方法，用于测试带有嵌套 while 循环的函数的符号化图
def test_while_loop_nested2_traced(self):
    # 获取待测试的函数和输入数据
    fn, inp = WHILE_LOOP_TESTS["nested2"]
    # 调用内部方法检查追踪结果，返回符号化图集合
    graphs = self._check_tracing(fn, inp)
    # 从符号化图集合中获取符号化图对象
    gm = graphs["symbolic"]
    # 获取外部 while 循环的主体和条件符号化图
    outer_body = gm.while_loop_body_graph_0
    outer_cond = gm.while_loop_cond_graph_0
    # 获取内部 while 循环的主体和条件符号化图
    inner_body = outer_body.while_loop_body_graph_0
    inner_cond = outer_body.while_loop_cond_graph_0
    # 调用自定义的断言方法，验证生成的内联代码与预期的匹配
    self.assertExpectedInline(
        gm.code.strip("\n"),
        """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    # 将 self.while_loop_cond_graph_0 和 self.while_loop_body_graph_0 赋给 while_loop_cond_graph_0 和 while_loop_body_graph_0
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    # 使用 torch.ops.higher_order.while_loop 函数执行 while 循环，参数为 while_loop_cond_graph_0、while_loop_body_graph_0 和 (arg0_1, arg1_1, arg2_1, arg3_1)，
    # 并将结果赋给 while_loop，然后清空所有变量的引用
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = None
    # 分别从 while_loop 中获取四个元素并赋给 getitem、getitem_1、getitem_2、getitem_3，然后清空 while_loop 的引用
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    # 返回四个元素构成的元组
    return (getitem, getitem_1, getitem_2, getitem_3)
    """,  # noqa: B950
    )
    # 调用自定义的断言方法，验证生成的内联代码与预期的匹配
    self.assertExpectedInline(
        outer_body.code.strip("\n"),
        """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    # 将 self.while_loop_cond_graph_0 和 self.while_loop_body_graph_0 赋给 while_loop_cond_graph_0 和 while_loop_body_graph_0
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    # 使用 torch.ops.higher_order.while_loop 函数执行 while 循环，参数为 while_loop_cond_graph_0、while_loop_body_graph_0 和 (arg0_1, arg1_1, arg2_1, arg3_1)，
    # 并将结果赋给 while_loop，然后清空所有变量的引用
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), ());  while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = None
    # 分别从 while_loop 中获取四个元素并赋给 getitem、getitem_1、getitem_2、getitem_3，然后清空 while_loop 的引用
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3];  while_loop = None
    # 使用 torch.ops.aten.sub.Tensor 函数计算 getitem 减去 1 的结果，并将结果赋给 sub，然后清空 getitem 的引用
    sub = torch.ops.aten.sub.Tensor(getitem, 1);  getitem = None
    # 使用 torch.ops.aten.clone.default 函数对 getitem_1 进行克隆操作，并将结果赋给 clone，然后清空 getitem_1 的引用
    clone = torch.ops.aten.clone.default(getitem_1);  getitem_1 = None
    # 使用 torch.ops.aten.mul.Tensor 函数计算 getitem_2 乘以 2 的结果，并将结果赋给 mul，然后清空 getitem_2 的引用
    mul = torch.ops.aten.mul.Tensor(getitem_2, 2);  getitem_2 = None
    # 使用 torch.ops.aten.div.Tensor 函数计算 getitem_3 除以 2 的结果，并将结果赋给 div，然后清空 getitem_3 的引用
    div = torch.ops.aten.div.Tensor(getitem_3, 2);  getitem_3 = None
    # 返回 sub、clone、mul 和 div 构成的元组
    return (sub, clone, mul, div)
    """,  # noqa: B950
    )
    # 调用自定义的断言方法，验证生成的内联代码与预期的匹配
    self.assertExpectedInline(
        outer_body.code.strip("\n"),
        """\
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    while_loop_cond_graph_0 = self.while_loop_cond_graph_0
    while_loop_body_graph_0 = self.while_loop_body_graph_0
    # 调用 Torch 模块中的高阶函数 while_loop，传入条件和循环体图形，初始参数为 (arg0_1, arg1_1, arg2_1, arg3_1)
    while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (arg0_1, arg1_1, arg2_1, arg3_1), ())
    # 清空变量以释放内存
    while_loop_cond_graph_0 = while_loop_body_graph_0 = arg0_1 = arg1_1 = arg2_1 = arg3_1 = None
    # 获取 while_loop 的返回结果的各个元素
    getitem = while_loop[0]
    getitem_1 = while_loop[1]
    getitem_2 = while_loop[2]
    getitem_3 = while_loop[3]
    # 释放 while_loop 变量以释放内存
    while_loop = None
    # 对第一个返回结果执行减法操作
    sub = torch.ops.aten.sub.Tensor(getitem, 1)
    # 对第二个返回结果执行默认克隆操作
    clone = torch.ops.aten.clone.default(getitem_1)
    # 对第三个返回结果执行乘法操作
    mul = torch.ops.aten.mul.Tensor(getitem_2, 2)
    # 对第四个返回结果执行除法操作
    div = torch.ops.aten.div.Tensor(getitem_3, 2)
    # 返回操作后的结果元组
    return (sub, clone, mul, div)
def forward(self, arg0_1, arg1_1, arg2_1, arg3_1):
    # 调用 torch.ops.aten.clone.default 对 arg0_1 进行克隆操作，并清空 arg0_1
    clone = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    # 调用 torch.ops.aten.sub.Tensor 对 arg1_1 减去标量 1，并清空 arg1_1
    sub = torch.ops.aten.sub.Tensor(arg1_1, 1);  arg1_1 = None
    # 调用 torch.ops.aten.add.Tensor 对 arg2_1 加上标量 3.14，并清空 arg2_1
    add = torch.ops.aten.add.Tensor(arg2_1, 3.14);  arg2_1 = None
    # 调用 torch.ops.aten.sub.Tensor 对 arg3_1 减去标量 2.71，并清空 arg3_1
    sub_1 = torch.ops.aten.sub.Tensor(arg3_1, 2.71);  arg3_1 = None
    # 返回四个操作的结果，分别是克隆结果、减法结果、加法结果和第二个减法结果
    return (clone, sub, add, sub_1)
    def test_cond_functionalized_hah(self):
        def true_fn(x):
            # 计算输入张量 x 的正弦值
            y = x.sin()
            # 将 y 中的每个元素加 4
            y.add_(4)
            # 返回 x 的正弦值的最大值，加上 y 中所有元素的和
            return x.sin().max() + y.sum()

        def false_fn(x):
            # 返回输入张量 x 的余弦值的最小值
            return x.cos().min()

        def f(x):
            # 判断输入张量 x 的第一个维度是否为 1
            pred = x.shape[0] == 1
            # 根据条件 pred 调用 true_fn 或 false_fn，并返回结果
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        # 使用 torch.func.functionalize 将函数 f 转化为可序列化的函数
        functional_f = torch.func.functionalize(f)
        # 断言 functional_f 在 example_inputs 上的输出与 f 在相同输入上的输出相等
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        # 使用 make_fx 将 functional_f 转化为 Torch 的图模块，并在 example_inputs 上运行
        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        # 断言图模块在 example_inputs 上的输出与 f 在相同输入上的输出相等
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        # 收集图模块中 true 分支中所有的操作
        all_ops_in_true_branch = []
        for node in graph_module.true_graph_0.graph.nodes:
            if node.op == "call_function":
                all_ops_in_true_branch.append(node.target)

        # 断言所有 true 分支中的操作都不可变
        self.assertFalse(any(op._schema.is_mutable for op in all_ops_in_true_branch))

        # 使用 tracing_mode="symbolic" 将 functional_f 转化为符号化的图模块
        graph_module = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        # 断言符号化的图模块在 example_inputs 上的输出与 f 在相同输入上的输出相等
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

    def test_cond_accepts_torch_function_as_inputs(self):
        a = torch.randn(3, 4)
        b = torch.randn(3, 4)

        def f(a, b):
            # 根据 a.sum() > 0 的结果，调用 torch.add 或 torch.mul，并返回结果
            return cond(a.sum() > 0, torch.add, torch.mul, (a, b))

        # 进行追踪并获取符号化的图模块
        gm = self._check_tracing(f, (a, b))["symbolic"]
        # 断言生成的符号化图模块的代码与预期的代码匹配
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, a_1, b_1):
    # 调用 torch.ops.aten.sum.default 方法对 a_1 进行求和操作
    sum_1 = torch.ops.aten.sum.default(a_1)
    # 调用 torch.ops.aten.gt.Scalar 方法比较 sum_1 是否大于 0
    gt = torch.ops.aten.gt.Scalar(sum_1, 0);  sum_1 = None
    # 从 self 对象中获取 true_graph_0 和 false_graph_0 属性
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    # 调用 torch.ops.higher_order.cond 方法根据 gt 的值选择执行 true_graph_0 或 false_graph_0 中的函数，并传入 [a_1, b_1] 参数
    conditional = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, [a_1, b_1]);  gt = true_graph_0 = false_graph_0 = a_1 = b_1 = None
    # 从 conditional 结果中获取第一个元素
    getitem = conditional[0];  conditional = None
    # 返回 getitem 结果
    return getitem
    def test_cond_functionalized_nested(self):
        def true_true_fn(x):
            # 计算输入张量 x 的余弦值，然后加上常数 4
            y = x.cos()
            y.add_(4)
            # 返回 x 的正弦值的最大值与 y 的正弦值的最大值之和
            return x.sin().max() + y.sin().max()

        def true_false_fn(x):
            # 返回输入张量 x 的余弦值的最小值
            return x.cos().min()

        def true_fn(x):
            # 判断张量 x 是否是形状为 (1, ...) 的张量
            pred = x.shape[0] == 1
            # 如果满足条件 pred，则调用 true_true_fn 函数，否则调用 true_false_fn 函数
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            # 返回输入张量 x 的所有元素的和
            return x.sum()

        def f(x):
            # 判断张量 x 是否是形状为 (1, ...) 的张量
            pred = x.shape[0] == 1
            # 如果满足条件 pred，则调用 true_fn 函数，否则调用 false_fn 函数
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        # 将函数 f 转换为函数式表示，并验证其在示例输入上的输出是否与直接调用 f 函数的输出一致
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        # 将函数 f 转换为 Torch Script GraphModule，并验证其在示例输入上的输出是否与直接调用 f 函数的输出一致
        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

        # 获取 Torch Script GraphModule 中的 true_true 分支
        gm_true_true_branch = graph_module.true_graph_0.true_graph_0

        # 使用符号化跟踪模式，将函数 f 转换为 Torch Script GraphModule，并验证其在示例输入上的输出是否与直接调用 f 函数的输出一致
        graph_module1 = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        self.assertEqual(graph_module1(*example_inputs), f(*example_inputs))

        # 收集 true_true 分支中所有的操作节点，并检查是否有可变操作
        all_ops = []
        for node in gm_true_true_branch.graph.nodes:
            if node.op == "call_function":
                all_ops.append(node.target)

        # 断言所有操作节点中没有可变操作
        self.assertFalse(any(op._schema.is_mutable for op in all_ops))

    def test_cond_functionalized_data_dependent_pred(self):
        def true_fn(x):
            # 返回输入张量 x 的正弦值的总和
            return x.sin().sum()

        def false_fn(x):
            # 返回输入张量 x 的余弦值的总和
            return x.cos().sum()

        def f(x):
            # 判断张量 x 是否具有非零元素，并且这些非零元素形成的张量是否是形状为 (1, ...) 的张量
            pred = x.nonzero().shape[0] == 1
            # 如果满足条件 pred，则调用 true_fn 函数，否则调用 false_fn 函数
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        # 将函数 f 转换为函数式表示，并验证其在示例输入上的输出是否与直接调用 f 函数的输出一致
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))

        # 将函数 f 转换为 Torch Script GraphModule，并验证其在示例输入上的输出是否与直接调用 f 函数的输出一致
        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

    # https://github.com/pytorch/pytorch/issues/126988
    @xfailIfTorchDynamo
    def test_cond_functionalized_input_mutation_on_true_branch(self):
        def true_fn(x):
            # 创建张量 x 的视图，并加上常数 1，然后返回其正弦值的总和
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.sin().sum()

        def false_fn(x):
            # 返回输入张量 x 的余弦值的总和
            return x.cos().sum()

        def f(x):
            # 判断张量 x 是否是形状为 (4, ...) 的张量
            pred = x.shape[0] == 4
            # 如果满足条件 pred，则调用 true_fn 函数，否则调用 false_fn 函数
            return cond(pred, true_fn, false_fn, [x])

        example_inputs = (torch.ones(4, 5),)
        # 将函数 f 转换为函数式表示，并断言其在示例输入上的执行会引发异常
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            functional_f(*example_inputs)

        # 将函数 f 转换为 Torch Script GraphModule，并断言其在示例输入上的执行会引发异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            make_fx(torch.func.functionalize(f))(*example_inputs)
    def test_cond_functionalized_input_mutation_on_false_branch(self):
        # 定义条件为真时的函数
        def true_fn(x):
            return x.sin().sum()

        # 定义条件为假时的函数
        def false_fn(x):
            # 创建 x 的视图并增加1，然后返回视图的余弦之和
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.cos().sum()

        # 定义主函数 f，根据条件执行 true_fn 或 false_fn，并传递 x 作为参数
        def f(x):
            pred = x.shape[0] == 4  # 检查 x 是否有4行
            return cond(pred, true_fn, false_fn, [x])  # 使用 cond 函数根据 pred 执行不同的函数

        example_inputs = (torch.ones(5, 5),)
        functional_f = torch.func.functionalize(f)
        # 测试函数 functional_f 的行为是否符合预期，应当抛出异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            functional_f(*example_inputs)

        # 测试经过 make_fx 包装的 functional_f 是否也能正确抛出异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    # https://github.com/pytorch/pytorch/issues/126988
    @xfailIfTorchDynamo
    def test_cond_functionalized_output_alias_input(self):
        # 定义条件为真时的函数
        def true_fn(x):
            return x

        # 定义条件为假时的函数
        def false_fn(x):
            # 创建 x 的视图并返回
            view_x = x.view(x.shape)
            return view_x

        # 定义主函数 f，根据条件执行 true_fn 或 false_fn，并传递 x 作为参数
        def f(x):
            pred = x.shape[0] == 4  # 检查 x 是否有4行
            return cond(pred, true_fn, false_fn, [x])  # 使用 cond 函数根据 pred 执行不同的函数

        example_inputs = (torch.ones(5, 5),)
        functional_f = torch.func.functionalize(f)

        # 测试函数 functional_f 的行为是否符合预期，应当抛出异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException,
            "One of torch.cond branch might be aliasing",
        ):
            functional_f(*example_inputs)

        # 测试经过 make_fx 包装的 functional_f 是否也能正确抛出异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException,
            "One of torch.cond branch might be aliasing",
        ):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    # https://github.com/pytorch/pytorch/issues/126988
    @xfailIfTorchDynamo
    def test_cond_functionalized_nested_input_mutation(self):
        # 定义内层条件为真时的函数
        def true_true_fn(x):
            x.add_(4)  # 修改 x 的值，增加4
            return x.sin().max()  # 返回修改后 x 的正弦最大值

        # 定义内层条件为假时的函数
        def true_false_fn(x):
            return x.cos().min()  # 返回 x 的余弦最小值

        # 定义外层条件为真时的函数
        def true_fn(x):
            pred = x.shape[0] == 1  # 检查 x 是否有1行
            return cond(pred, true_true_fn, true_false_fn, [x])  # 使用 cond 函数根据 pred 执行不同的函数

        # 定义外层条件为假时的函数
        def false_fn(x):
            return x.sum()  # 返回 x 的总和

        # 定义主函数 f，根据条件执行 true_fn 或 false_fn，并传递 x 作为参数
        def f(x):
            pred = x.shape[0] == 1  # 检查 x 是否有1行
            return cond(pred, true_fn, false_fn, [x])  # 使用 cond 函数根据 pred 执行不同的函数

        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)

        # 测试函数 functional_f 的行为是否符合预期，应当抛出异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            functional_f(*example_inputs)

        # 测试经过 make_fx 包装的 functional_f 是否也能正确抛出异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            make_fx(torch.func.functionalize(f))(*example_inputs)
    # 定义一个测试函数，用于测试条件函数功能化嵌套输入变异与AOT函数
    def test_cond_functionalized_nested_input_mutation_with_aot_func(self):
        # 定义一个函数，在输入张量上执行真真分支操作，即加4后求正弦函数的最大值
        def true_true_fn(x):
            x.add_(4)  # 在张量上原地加4
            return x.sin().max()  # 返回正弦函数的最大值

        # 定义一个函数，在输入张量上执行真假分支操作，即求余弦函数的最小值
        def true_false_fn(x):
            return x.cos().min()  # 返回余弦函数的最小值

        # 定义一个函数，根据条件预测选择真函数或假函数进行操作
        def true_fn(x):
            pred = x.shape[0] == 1  # 判断张量行数是否为1
            return cond(pred, true_true_fn, true_false_fn, [x])  # 调用条件函数处理张量

        # 定义一个函数，在输入张量上执行假函数操作，即返回张量的总和
        def false_fn(x):
            return x.sum()  # 返回张量的总和

        # 定义一个函数f，根据条件预测选择真函数或假函数进行操作
        def f(x):
            pred = x.shape[0] == 1  # 判断张量行数是否为1
            return cond(pred, true_fn, false_fn, [x])  # 调用条件函数处理张量

        # 创建一个示例输入张量，全为1，形状为(4, 5)
        example_input = torch.ones(4, 5)
        
        try:
            # 将示例输入张量转换为旧的函数对象
            example_input_func = to_fun_old(example_input)
            # 启用函数功能化，禁用重新应用视图
            torch._enable_functionalization(reapply_views=False)
            # 断言调用f(example_input_func)时抛出UnsupportedAliasMutationException异常
            with self.assertRaisesRegex(
                UnsupportedAliasMutationException, "One of torch.cond branch"
            ):
                f(example_input_func)

            # 断言调用make_fx(f)(example_input_func)时抛出UnsupportedAliasMutationException异常
            with self.assertRaisesRegex(
                UnsupportedAliasMutationException, "One of torch.cond branch"
            ):
                make_fx(f)(example_input_func)
        finally:
            # 禁用函数功能化
            torch._disable_functionalization()

        # 定义一个函数包装器f_wrapper，用于启用和禁用函数功能化，并执行func函数
        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    return func(*args, **kwargs)
                finally:
                    torch._disable_functionalization()

            return wrapper

        # 断言调用make_fx(f_wrapper(f))(example_input_func)时抛出UnsupportedAliasMutationException异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "One of torch.cond branch"
        ):
            make_fx(f_wrapper(f))(example_input_func)

    # 标记：https://github.com/pytorch/pytorch/issues/126988
    @xfailIfTorchDynamo
    # 定义一个测试方法，用于验证条件函数化输入别名化与AOT函数的功能
    def test_cond_functionalized_input_aliasing_with_aot_func(self):
        # 定义一个返回输入参数的函数
        def true_fn(x):
            return x

        # 定义一个对输入参数做视图操作后返回的函数
        def false_fn(x):
            view_x = x.view(x.shape)
            return view_x

        # 定义一个函数f，根据输入张量x的形状是否为(4, ...)，选择不同的函数进行处理
        def f(x):
            pred = x.shape[0] == 4  # 判断输入张量x的行数是否为4
            return cond(pred, true_fn, false_fn, [x])  # 根据pred条件调用true_fn或false_fn函数处理x

        # 创建一个5x5的全1张量作为示例输入
        example_input = torch.ones(5, 5)
        try:
            # 将example_input转换为函数化的表示形式
            example_input_func = to_fun_old(example_input)
            # 开启函数化的运行环境，禁止视图重新应用
            torch._enable_functionalization(reapply_views=False)
            # 使用断言检查调用f函数时是否会引发UnsupportedAliasMutationException异常
            with self.assertRaisesRegex(
                UnsupportedAliasMutationException,
                "One of torch.cond branch might be aliasing",
            ):
                f(example_input_func)
        finally:
            # 禁用函数化的运行环境
            torch._disable_functionalization()

        # 定义一个装饰器函数，用于包装函数，使其在运行前后开启和关闭函数化的运行环境
        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)
                try:
                    # 对函数的输入参数args和kwargs中的张量进行转换为函数化张量的处理
                    func_args = pytree.tree_map(
                        lambda x: torch._to_functional_tensor(x)
                        if isinstance(x, torch.Tensor)
                        else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: torch._to_functional_tensor(x)
                        if isinstance(x, torch.Tensor)
                        else x,
                        kwargs,
                    )
                    # 调用原始的func函数，并返回其结果
                    return func(*func_args, **func_kwargs)
                finally:
                    # 在finally块中禁用函数化的运行环境
                    torch._disable_functionalization()

            return wrapper

        # 使用断言检查调用make_fx(f_wrapper(f))(example_input)时是否会引发UnsupportedAliasMutationException异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException,
            "One of torch.cond branch might be aliasing",
        ):
            make_fx(f_wrapper(f))(example_input)
    # 定义测试函数，用于检查条件函数化后的功能
    def test_cond_functionalized_aot_func_check_functional(self):
        # 定义返回输入张量的余弦函数
        def true_fn(x):
            return x.cos()

        # 定义返回输入张量的正弦函数，并在结果上加上常数5
        def false_fn(x):
            y = x.sin()
            y.add_(5)
            return y

        # 定义条件函数 f，根据输入张量 x 的形状是否为4，调用 true_fn 或 false_fn
        def f(x):
            pred = x.shape[0] == 4  # 判断输入张量 x 的行数是否为4
            return cond(pred, true_fn, false_fn, [x])

        example_input = torch.ones(5, 5)  # 创建一个全为1的5x5张量作为示例输入

        # 函数包装器 f_wrapper，用于在调用函数期间启用和禁用函数化
        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)  # 启用函数化
                try:
                    # 将输入参数中的张量转换为旧的函数表示形式（非张量）
                    func_args = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        kwargs,
                    )
                    # 调用原始函数，并将结果从旧的函数表示形式转换回新的函数表示形式（张量）
                    return pytree.tree_map(
                        from_fun_old, func(*func_args, **func_kwargs)
                    )
                finally:
                    torch._disable_functionalization()  # 禁用函数化

            return wrapper

        # 使用函数包装器 f_wrapper 包装函数 f，并应用于示例输入 example_input
        result_gm = make_fx(f_wrapper(f))(example_input)

        # 遍历函数化后的 true 分支的计算图中的节点，并检查是否不可变
        for node in result_gm.true_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        # 遍历函数化后的 false 分支的计算图中的节点，并检查是否不可变
        for node in result_gm.false_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)

        # 验证函数化结果与未函数化的结果是否相等
        self.assertEqual(result_gm(torch.ones(5, 5)), f(torch.ones(5, 5)))

    # 测试嵌套条件函数的功能，并验证其他输入的情况
    def test_cond_nested_traced_other_inputs(self):
        # 定义在 true 分支中的嵌套函数，返回输入的平方
        def true_nested(y):
            return y * y

        # 定义在 false 分支中的嵌套函数，返回输入的加法结果
        def false_nested(y):
            return y + y

        # 定义 true 分支的主函数，根据 pred2 调用 true_nested 或 false_nested
        def true_fn(k, pred2):
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([0.25, 0.25]), z)

        # 定义 false 分支的主函数，返回输入张量的余弦函数
        def false_fn(k, _):
            return k.cos()

        # 主函数 f，根据 pred 调用 true_fn 或 false_fn
        def f(k, pred, pred2):
            return cond(pred, true_fn, false_fn, [k, pred2])

        x = torch.tensor([0.5, 0.5])  # 创建输入张量 x，包含两个0.5的元素
        # 创建函数化后的图形对象，并应用于输入 x 和两个 False 标志
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))

        a = torch.tensor([1.0, 1.0])  # 创建输入张量 a，包含两个1.0的元素
        # 在图中执行 true 分支的计算，并验证结果是否符合预期
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (a * a) + torch.tensor([0.25, 0.25]))

        b = torch.tensor([2.0, 2.0])  # 创建输入张量 b，包含两个2.0的元素
        # 在图中执行 true 分支的计算，并验证结果是否符合预期
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (b * b) + torch.tensor([0.25, 0.25]))
    def test_cond_nested_traced_multi(self):
        # 定义函数 true_a，计算输入 y 的平方
        def true_a(y):
            return y * y
        
        # 定义函数 false_a，计算输入 y 的两倍
        def false_a(y):
            return y + y
        
        # 定义函数 true_b，对输入 y 和 z 进行加法操作
        def true_b(y, z):
            return y + z
        
        # 定义函数 false_b，对输入 y 和 z 进行乘法操作
        def false_b(y, z):
            return y * z
        
        # 定义函数 f，根据两个条件 pred 和 pred2 分别调用不同的函数，并返回结果的和
        def f(x, pred, pred2):
            a_out = cond(pred, true_a, false_a, [x])
            b_out = cond(pred2, true_b, false_b, [x, x])
            return a_out + b_out
        
        # 生成一个包含4个随机数的张量 x
        x = torch.randn(4)
        # 使用 make_fx 函数将 f 转换为一个计算图 graph，并传入 False 和 False 作为条件
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))
        
        # 使用断言验证生成的计算图代码是否符合预期，去除首尾空白字符后进行比较
        self.assertExpectedInline(
            graph.code.strip(),
            """\
# 定义一个名为 forward 的方法，接受三个参数 self, x_1, pred_1, pred2_1
def forward(self, x_1, pred_1, pred2_1):
    # 从当前对象中获取 true_graph_0 和 false_graph_0
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    # 使用 torch.ops.higher_order.cond 方法，根据 pred_1 条件选择 true_graph_0 或 false_graph_0，并传入 [x_1] 作为参数
    conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);  pred_1 = true_graph_0 = false_graph_0 = None
    # 从条件执行的结果中获取第一个元素，并清空 conditional 变量
    getitem = conditional[0];  conditional = None
    # 从当前对象中获取 true_graph_1 和 false_graph_1
    true_graph_1 = self.true_graph_1
    false_graph_1 = self.false_graph_1
    # 使用 torch.ops.higher_order.cond 方法，根据 pred2_1 条件选择 true_graph_1 或 false_graph_1，并传入 [x_1] 作为参数
    conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1]);  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
    # 从条件执行的结果中获取第一个元素，并清空 conditional_1 变量
    getitem_1 = conditional_1[0];  conditional_1 = None
    # 使用 torch.ops.aten.add.Tensor 方法，将 getitem 和 getitem_1 进行张量加法操作
    add = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
    # 返回 add 结果
    return add
    def test_cond_nested_traced_fake_tensor(self):
        # 定义一个嵌套函数，返回输入的平方
        def true_nested(y):
            return y * y

        # 定义一个嵌套函数，返回输入的两倍
        def false_nested(y):
            return y + y

        # 定义一个函数，根据条件选择调用 true_nested 或 false_nested 函数，并返回结果
        def true_fn(x, pred2):
            z = cond(pred2, true_nested, false_nested, [x])
            return x + z

        # 定义一个函数，直接返回输入的余弦值
        def false_fn(x, _):
            return x.cos()

        # 定义一个函数 f，根据条件选择调用 true_fn 或 false_fn 函数，并返回结果
        def f(x, pred, pred2):
            return cond(pred, true_fn, false_fn, [x, pred2])

        # 生成一个包含4个随机数的张量 x
        x = torch.randn(4)
        # 使用指定的追踪模式（fake），构建函数图
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        # 测试用例：pred=True, pred2=True，期望结果是 x*x + x
        result_true_true = graph.forward(
            x, torch.tensor(True), torch.tensor(True)
        )  # True + True -> x * x
        # 测试用例：pred=True, pred2=False，期望结果是 x + x
        result_true_false = graph.forward(
            x, torch.tensor(True), torch.tensor(False)
        )  # True + False -> x + x
        # 测试用例：pred=False, pred2=True，期望结果是 torch.cos(x)
        result_false_true = graph.forward(
            x, torch.tensor(False), torch.tensor(True)
        )  # False + True -> cos
        # 测试用例：pred=False, pred2=False，期望结果是 torch.cos(x)
        result_false_false = graph.forward(
            x, torch.tensor(False), torch.tensor(False)
        )  # False + False -> cos

        # 断言不相等
        self.assertNotEqual(result_true_true, result_true_false)
        # 断言 torch.allclose 结果为 False
        self.assertFalse(torch.allclose(result_false_true, result_true_true))

        # 断言相等
        self.assertEqual(result_false_true, result_false_false)

        # 断言相等，期望结果是 x*x + x
        self.assertEqual(result_true_true, (x * x) + x)
        # 断言相等，期望结果是 x + x + x
        self.assertEqual(result_true_false, x + x + x)

        # 断言相等，期望结果是 torch.cos(x)
        self.assertEqual(result_false_true, torch.cos(x))

    def test_cond_nested_traced_other_inputs_fake_tensor(self):
        # 定义一个嵌套函数，返回输入的平方
        def true_nested(y):
            return y * y

        # 定义一个嵌套函数，返回输入的两倍
        def false_nested(y):
            return y + y

        # 定义一个函数，根据条件选择调用 true_nested 或 false_nested 函数，并返回结果
        def true_fn(k, pred2):
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([0.25, 0.25]), z)

        # 定义一个函数，直接返回输入的余弦值
        def false_fn(k, _):
            return k.cos()

        # 定义一个函数 f，根据条件选择调用 true_fn 或 false_fn 函数，并返回结果
        def f(k, pred, pred2):
            return cond(pred, true_fn, false_fn, [k, pred2])

        # 生成一个包含两个数值的张量 x
        x = torch.tensor([0.5, 0.5])
        # 使用指定的追踪模式（fake），构建函数图
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        # 创建一个张量 a，测试用例：pred=True, pred2=True，期望结果是 a*a + [0.25, 0.25]
        a = torch.tensor([1.0, 1.0])
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (a * a) + torch.tensor([0.25, 0.25]))

        # 创建一个张量 b，测试用例：pred=True, pred2=True，期望结果是 b*b + [0.25, 0.25]
        b = torch.tensor([2.0, 2.0])
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, (b * b) + torch.tensor([0.25, 0.25]))
    # 定义一个测试函数，用于测试条件嵌套和跟踪多个假张量的情况
    def test_cond_nested_traced_multi_fake_tensor(self):
        # 定义一个函数，返回输入参数的平方
        def true_a(y):
            return y * y

        # 定义一个函数，返回输入参数的两倍
        def false_a(y):
            return y + y

        # 定义一个函数，返回两个输入参数的和
        def true_b(y, z):
            return y + z

        # 定义一个函数，返回两个输入参数的乘积
        def false_b(y, z):
            return y * z

        # 定义一个函数 f，根据两个条件选择不同的函数进行计算，然后返回结果的和
        def f(x, pred, pred2):
            # 根据 pred 条件选择 true_a 或 false_a 函数计算 x
            a_out = cond(pred, true_a, false_a, [x])
            # 根据 pred2 条件选择 true_b 或 false_b 函数计算 x 和 x
            b_out = cond(pred2, true_b, false_b, [x, x])
            # 返回 a_out 和 b_out 的和作为函数 f 的结果
            return a_out + b_out

        # 生成一个随机的张量 x
        x = torch.randn(4)
        # 调用 make_fx 函数生成一个函数图，并使用 "fake" 跟踪模式
        graph = make_fx(f, tracing_mode="fake")(
            x, torch.tensor(False), torch.tensor(False)
        )

        # 使用 self.assertExpectedInline 方法断言生成的函数图的代码是否符合预期
        self.assertExpectedInline(
            graph.code.strip(),
            """\
    # 定义一个方法 `forward`，接收三个参数 `x_1`, `pred_1`, `pred2_1`
    def forward(self, x_1, pred_1, pred2_1):
        # 获取 self 对象的 true_graph_0 属性
        true_graph_0 = self.true_graph_0
        # 获取 self 对象的 false_graph_0 属性
        false_graph_0 = self.false_graph_0
        # 使用 torch.ops.higher_order.cond 方法进行条件判断，根据 pred_1 的值选择 true_graph_0 或 false_graph_0 的图形
        conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);  pred_1 = true_graph_0 = false_graph_0 = None
        # 获取条件判断结果的第一个元素
        getitem = conditional[0];  conditional = None
        # 获取 self 对象的 true_graph_1 属性
        true_graph_1 = self.true_graph_1
        # 获取 self 对象的 false_graph_1 属性
        false_graph_1 = self.false_graph_1
        # 使用 torch.ops.higher_order.cond 方法进行条件判断，根据 pred2_1 的值选择 true_graph_1 或 false_graph_1 的图形
        conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1]);  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None
        # 获取条件判断结果的第一个元素
        getitem_1 = conditional_1[0];  conditional_1 = None
        # 使用 torch.ops.aten.add.Tensor 方法对 getitem 和 getitem_1 进行张量相加操作
        add = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        # 返回张量相加的结果
        return add""",  # noqa: B950
    # 定义测试函数，用于测试符号化追踪模式下的 map 函数的简单情况
    def test_tracing_map_symbolic_simple(self):
        # 定义简单的函数 f，对两个输入进行加法操作
        def f(x, y):
            return x + y

        # 定义函数 g，使用 control_flow.map 对 xs 中的每个元素应用 f 函数，并传递 y 作为额外参数
        def g(xs, y):
            return control_flow.map(f, xs, y)

        # 使用 make_fx 函数创建一个符号化追踪的函数 gm，并传入 torch.ones 作为参数
        gm = make_fx(g, tracing_mode="symbolic")(torch.ones(3, 2, 4), torch.ones(4))
        # 创建随机张量 x 和 y
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        # 计算 gm(x, y) 的结果
        res = gm(x, y)
        # 断言 gm(x, y) 的结果与直接调用 g(x, y) 的结果相等
        self.assertEqual(res, g(x, y))
        # 检查 gm 内部 map 函数的调用次数是否为 1
        self.check_map_count(gm, 1)

    # 定义测试函数，用于测试符号化追踪模式下的 map 函数的列表情况
    def test_tracing_map_symbolic_list(self):
        # 定义函数 f，对输入列表的元素进行不同的加法和乘法操作
        def f(x, y):
            return [x[0][0] + y, x[1] * y]

        # 定义函数 g，使用 control_flow.map 对 xs 中的每个元素应用 f 函数，并传递 y 和 z 作为额外参数
        def g(xs, y, z):
            out = control_flow.map(f, xs, y)
            return out[0] + z, out[1] * z

        # 定义示例输入 example_x
        example_x = [[torch.ones(3, 4, 5)], torch.ones(3, 4, 5)]
        # 使用 make_fx 函数创建一个符号化追踪的函数 gm，并传入 example_x, torch.ones(5), torch.ones(5) 作为参数
        gm = make_fx(g, tracing_mode="symbolic")(
            example_x, torch.ones(5), torch.ones(5)
        )
        # 创建随机张量 x, y 和 z
        x = [[torch.randn(4, 5, 6)], torch.ones(4, 5, 6)]
        y = torch.randn(6)
        z = torch.ones(6)
        # 计算 gm(x, y, z) 的结果
        res = gm(x, y, z)
        # 断言 gm(x, y, z) 的结果与直接调用 g(x, y, z) 的结果相等
        self.assertEqual(res, g(x, y, z))
        # 检查 gm 内部 map 函数的调用次数是否为 1
        self.check_map_count(gm, 1)

    # 定义测试函数，用于测试符号化追踪模式下的 map 函数的字典情况
    def test_tracing_map_symbolic_dict(self):
        # 定义函数 f，对输入字典的元素进行加法和乘法操作，并返回新的字典
        def f(x, y):
            return {"d": x["b"]["a"] + y, "e": x["c"] * y}

        # 定义函数 g，使用 control_flow.map 对 xs 中的每个元素应用 f 函数，并传递 y 和 z 作为额外参数
        def g(xs, y, z):
            out = control_flow.map(f, xs, y)
            return {"f": out["d"] + z, "g": out["e"] * z}

        # 定义示例输入 example_x
        example_x = {"b": {"a": torch.ones(3, 4, 5)}, "c": torch.ones(3, 4, 5)}
        # 使用 make_fx 函数创建一个符号化追踪的函数 gm，并传入 example_x, torch.ones(5), torch.ones(5) 作为参数
        gm = make_fx(g, tracing_mode="symbolic")(
            example_x, torch.ones(5), torch.ones(5)
        )
        # 创建随机张量 x, y 和 z
        x = {"b": {"a": torch.randn(4, 5, 6)}, "c": torch.ones(4, 5, 6)}
        y = torch.randn(6)
        z = torch.ones(6)
        # 计算 gm(x, y, z) 的结果
        res = gm(x, y, z)
        # 断言 gm(x, y, z) 的结果与直接调用 g(x, y, z) 的结果相等
        self.assertEqual(res, g(x, y, z))
        # 检查 gm 内部 map 函数的调用次数是否为 1
        self.check_map_count(gm, 1)

    # 定义测试函数，用于测试符号化追踪模式下的 map 函数的自动求导的简单情况
    def test_tracing_map_autograd_symbolic_simple(self):
        # 定义简单的函数 f，对两个输入进行加法操作
        def f(x, y):
            return x + y

        # 定义函数 g，使用 control_flow.map 对 xs 中的每个元素应用 f 函数，并传递 y 作为额外参数
        # 然后计算结果的梯度
        def g(xs, y):
            out = control_flow.map(f, xs, y)
            return torch.autograd.grad(out, (xs, y), torch.ones_like(out))

        # 使用 make_fx 函数创建一个符号化追踪的函数 gm，并传入 torch.ones 的张量，并设置 requires_grad=True
        gm = make_fx(g, tracing_mode="symbolic")(
            torch.ones(3, 4, 5, requires_grad=True), torch.ones(5, requires_grad=True)
        )
        # 创建随机张量 x 和 y，并设置 requires_grad=True
        x = torch.randn(4, 5, 6, requires_grad=True)
        y = torch.randn(6, requires_grad=True)
        # 计算 gm(x, y) 的结果
        res = gm(x, y)
        # 断言 gm(x, y) 的结果与直接调用 g(x, y) 的结果相等
        self.assertEqual(res, g(x, y))
        # 检查 gm 内部 map 函数的调用次数是否为 2
        self.check_map_count(gm, 2)
    # 定义一个测试函数，用于测试追踪映射自动微分符号列表
    def test_tracing_map_autograd_symbolic_list(self):
        # 导入必要的模块
        import torch.utils._pytree as pytree

        # 定义一个函数 f，接受两个参数 x 和 y，返回一个列表
        def f(x, y):
            return [x[0].cos() + y.sin(), x[1].sin() * y.cos()]

        # 定义一个函数 g，接受参数 xs 和 y，进行控制流映射操作
        def g(xs, y):
            # 对函数 f 进行映射操作
            out = control_flow.map(f, xs, y)
            # 将输出展平为一维数组
            flat_out = pytree.tree_leaves(out)
            # 将输入展平为一维数组
            flat_inp = pytree.tree_leaves((xs, y))
            # 获取需要梯度的输入
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            # 计算梯度
            return torch.autograd.grad(
                flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out]
            )

        # 使用 make_fx 函数创建一个追踪模式为“symbolic”的函数
        gm = make_fx(g, tracing_mode="symbolic")(
            [torch.ones(3, 4, 5), torch.ones(3, 4, 5, requires_grad=True)],
            torch.ones(5, requires_grad=True),
        )
        # 创建输入数据 x 和 y
        x = [torch.randn(4, 5, 6), torch.ones(4, 5, 6, requires_grad=True)]
        y = torch.randn(6, requires_grad=True)
        # 调用 gm 函数并保存结果
        res = gm(x, y)
        # 断言结果与 g 函数的结果相等
        self.assertEqual(res, g(x, y))
        # 检查映射次数
        self.check_map_count(gm, 2)

    # 定义一个测试函数，用于测试追踪映射自动微分符号字典
    def test_tracing_map_autograd_symbolic_dict(self):
        # 定义一个函数 f，接受两个参数 x 和 y，返回一个列表
        def f(x, y):
            return [x["a"] + y, x["b"] * y]

        # 定义一个函数 g，接受参数 xs 和 y，进行控制流映射操作
        def g(xs, y):
            # 对函数 f 进行映射操作
            out = control_flow.map(f, xs, y)
            # 将输出展平为一维数组
            flat_out = pytree.tree_leaves(out)
            # 将输入展平为一维数组
            flat_inp = pytree.tree_leaves((xs, y))
            # 获取需要梯度的输入
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            # 计算梯度
            return torch.autograd.grad(
                flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out]
            )

        # 创建一个字典 traced_x，包含两个键值对
        traced_x = {
            "a": torch.ones(3, 4, 5, requires_grad=True),
            "b": torch.ones(3, 4, 5, requires_grad=True),
        }
        # 使用 make_fx 函数创建一个追踪模式为“symbolic”的函数
        gm = make_fx(g, tracing_mode="symbolic")(
            traced_x, torch.ones(5, requires_grad=True)
        )
        # 创建输入数据 x 和 y
        x = {
            "a": torch.randn(4, 5, 6, requires_grad=True),
            "b": torch.ones(4, 5, 6, requires_grad=True),
        }
        y = torch.randn(6, requires_grad=True)
        # 调用 gm 函数并保存结果
        res = gm(x, y)
        # 断言结果与 g 函数的结果相等
        self.assertEqual(res, g(x, y))
        # 检查映射次数
        self.check_map_count(gm, 2)
    # 定义一个测试函数，用于追踪映射、自动微分和功能化的操作
    def test_tracing_map_autograd_aot_functionalized(self):
        # 内部函数 inner 接收两个参数 x 和 y，执行减法和加法操作，返回乘积结果
        def inner(x, y):
            z = x - 1  # 计算 x - 1
            z.add_(1)  # 在 z 上加 1
            return z * y  # 返回 z 与 y 的乘积

        # 函数 f 接收 xs 和 y 两个参数，通过控制流映射 inner 函数来计算结果，并返回 xs 和 y 的梯度
        def f(xs, y):
            res = control_flow.map(inner, xs, y)  # 映射 inner 函数到 xs 和 y 上，得到结果 res
            grads = torch.autograd.grad(res, (xs, y), torch.ones_like(res))  # 计算 res 对 xs 和 y 的梯度
            return grads  # 返回梯度

        # 函数 f_wrapper 接收一个函数 func，并创建一个包装函数，用于在执行过程中启用和禁用函数化
        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                torch._enable_functionalization(reapply_views=False)  # 启用函数化
                try:
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))  # 调用 func 函数并应用函数化
                finally:
                    torch._disable_functionalization()  # 禁用函数化

            return wrapper

        # 示例输入，包含两个 torch 张量，用于测试
        example_inputs = (
            torch.ones(3, 2, 4, requires_grad=True),
            torch.ones(2, 4, requires_grad=True),
        )
        # 使用 make_fx 函数将 f 和 f_wrapper 函数进行功能化，并设置追踪模式为 "symbolic"
        gm = make_fx(f, tracing_mode="symbolic")(*example_inputs)
        fgm = make_fx(f_wrapper(f), tracing_mode="symbolic")(*example_inputs)
        xs = torch.ones(3, 4, 5, requires_grad=True)  # 创建一个需要梯度的 torch 张量 xs
        y = torch.ones(4, 5, requires_grad=True)  # 创建一个需要梯度的 torch 张量 y

        self.assertEqual(gm(xs, y), f(xs, y))  # 断言 gm(xs, y) 和 f(xs, y) 的结果相等

        # 计算图中可变节点的计数函数，用于计算 gm 中的可变节点数量
        def count_mutable(gm):
            c = 0
            for node in gm.graph.nodes:
                if node.op == "call_function":
                    if node.target == torch.ops.higher_order.map_impl:
                        c += count_mutable(getattr(gm, str(node.args[0])))
                    elif schema := getattr(node.target, "_schema", None):
                        c += int(schema.is_mutable)
            return c

        self.assertEqual(count_mutable(fgm), 0)  # 断言 fgm 中的可变节点数量为 0
        # 断言 gm 中的可变节点数量为 2，一个用于前向计算，一个用于反向重计算逻辑
        self.assertEqual(count_mutable(gm), 2)

    # 定义一个测试函数，用于测试功能化映射
    def test_map_functionalized(self):
        # 内部函数 map_fn 接收两个参数 x 和 y，执行加法和加法操作，并返回结果
        def map_fn(x, y):
            z = x + y  # 计算 x + y
            z.add_(4)  # 在 z 上加 4
            return z  # 返回结果 z

        # 函数 f 接收 xs 和 y 两个参数，通过控制流映射 map_fn 函数来计算结果，并返回结果
        def f(xs, y):
            return control_flow.map(map_fn, xs, y)  # 映射 map_fn 函数到 xs 和 y 上，并返回结果

        # 示例输入，包含两个 torch 张量，用于测试
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)  # 使用 torch.func.functionalize 将 f 函数功能化
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))  # 断言 functional_f 和 f 的结果相等

        # 使用 make_fx 函数将功能化的 f 函数功能化，并设置追踪模式为 "symbolic"
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(gm(*example_inputs), f(*example_inputs))  # 断言 gm 和 f 的结果相等

        # 使用 make_fx 函数将功能化的 f 函数功能化，并设置追踪模式为 "symbolic"
        gm = make_fx(torch.func.functionalize(f), tracing_mode="symbolic")(
            *example_inputs
        )
        self.assertEqual(gm(*example_inputs), f(*example_inputs))  # 断言 gm 和 f 的结果相等

        # 遍历 gm.body_graph_0 中的节点，如果节点的操作为 "call_function"
        # 则断言其目标函数的 _schema.is_mutable 属性为假
        for node in gm.body_graph_0.graph.nodes:
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)
        
        # 检查 gm 中映射节点的数量是否为 1
        self.check_map_count(gm, 1)
    # 定义测试方法，用于测试 functionalized_aot_func 函数的行为
    def test_map_functionalized_aot_func(self):
        # 定义 map_fn 函数，接受两个参数 x 和 y，并返回对它们的操作结果
        def map_fn(x, y):
            # 计算 x + y，并将结果保存到 z
            z = x + y
            # 对 z 执行原地加法，添加常数 4
            z.add_(4)
            return z
        
        # 定义 f 函数，接受 xs 和 y 两个参数，调用 control_flow.map 函数，并返回结果
        def f(xs, y):
            return control_flow.map(map_fn, xs, y)
        
        # 定义 f_wrapper 函数，接受一个函数作为参数 func，返回一个装饰器函数 wrapper
        def f_wrapper(func):
            # 定义 wrapper 函数，用于包装 func 函数的执行过程
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 在执行 func 函数之前，启用函数化
                torch._enable_functionalization(reapply_views=False)
                try:
                    # 调用 func 函数，并对返回值应用 from_fun_old 函数
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))
                finally:
                    # 执行完毕后，禁用函数化
                    torch._disable_functionalization()
            
            return wrapper
        
        # 定义示例输入 example_inputs
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        
        # 调用 make_fx 函数，传入 f_wrapper(f) 作为参数，并使用 example_inputs 运行返回结果
        gm = make_fx(f_wrapper(f))(*example_inputs)
        
        # 遍历 gm 对象的图节点
        for node in gm.body_graph_0.graph.nodes:
            # 检查节点操作是否为 "call_function"，并验证 node.target._schema.is_mutable 为假
            if node.op == "call_function":
                self.assertTrue(not node.target._schema.is_mutable)
        
        # 断言 gm(*example_inputs) 的结果与 f(*example_inputs) 的结果相等
        self.assertEqual(gm(*example_inputs), f(*example_inputs))

    # https://github.com/pytorch/pytorch/issues/126988
    # 标记为预期失败，用于测试 functionalized_arg_mutation 函数的特定问题
    @xfailIfTorchDynamo
    def test_map_functionalized_arg_mutation(self):
        # 定义 map_fn 函数，接受两个参数 x 和 y，对 y 执行原地加法，并返回 x + y 的结果
        def map_fn(x, y):
            y.add_(4)
            return x + y
        
        # 定义 f 函数，接受 xs 和 y 两个参数，调用 control_flow.map 函数，并返回结果
        def f(xs, y):
            return control_flow.map(map_fn, xs, y)
        
        # 定义示例输入 example_inputs
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        
        # 使用 torch.func.functionalize 包装 f 函数
        functional_f = torch.func.functionalize(f)
        
        # 使用断言检查 functional_f(*example_inputs) 是否引发异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "torch.map is mutating the input!"
        ):
            functional_f(*example_inputs)

    # https://github.com/pytorch/pytorch/issues/126988
    # 标记为预期失败，用于测试 functionalized_elem_mutation 函数的特定问题
    @xfailIfTorchDynamo
    def test_map_functionalized_elem_mutation(self):
        # 定义 map_fn 函数，接受两个参数 x 和 y，对 x 执行原地加法，并返回 x + y 的结果
        def map_fn(x, y):
            x.add_(4)
            return x + y
        
        # 定义 f 函数，接受 xs 和 y 两个参数，调用 control_flow.map 函数，并返回结果
        def f(xs, y):
            return control_flow.map(map_fn, xs, y)
        
        # 定义示例输入 example_inputs
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        
        # 使用 torch.func.functionalize 包装 f 函数
        functional_f = torch.func.functionalize(f)
        
        # 使用断言检查 functional_f(*example_inputs) 是否引发异常
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "torch.map is mutating the input!"
        ):
            functional_f(*example_inputs)

    # 定义测试方法，用于测试 cond_autograd_fail 函数的行为
    def test_cond_autograd_fail(self):
        # 定义 true_fn 函数，接受 x 一个参数，返回 x 的余弦值
        def true_fn(x):
            return x.cos()
        
        # 定义 false_fn 函数，接受 x 一个参数，返回 x 的正弦值
        def false_fn(x):
            return x.sin()
        
        # 定义 f 函数，接受 x 和 y 两个参数，调用 control_flow.cond 函数，并返回结果
        def f(x, y):
            return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [y])
        
        # 定义示例输入 example_inputs，包含两个张量，都要求梯度计算
        example_inputs = (
            torch.ones(3, 2, 4, requires_grad=True),
            torch.ones(4, requires_grad=True),
        )
        
        # 使用断言检查运行 f(*example_inputs).sum().backward() 是否引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Autograd not implemented for cond"):
            f(*example_inputs).sum().backward()
        
        # 确保在不运行反向传播时不会引发错误
        f(*example_inputs)
    def test_map_functionalized_elem_alias(self):
        # 定义一个映射函数，对输入张量进行视图操作后返回原始输入
        def map_fn(x):
            x.view(x.shape)
            return x

        # 定义一个函数，使用控制流模块中的map函数对输入列表中的每个元素应用map_fn函数
        def f(xs):
            return control_flow.map(map_fn, xs)

        # 定义示例输入
        example_inputs = (torch.ones(3, 2, 4),)
        # 将函数f转换为torch.func.functionalize类型的函数
        functional_f = torch.func.functionalize(f)
        # 使用断言检查是否捕获到UnsupportedAliasMutationException异常，错误消息为"torch.map is aliasing the input!"
        with self.assertRaisesRegex(
            UnsupportedAliasMutationException, "torch.map is aliasing the input!"
        ):
            # 调用functional_f函数，并传入示例输入
            functional_f(*example_inputs)

    def test_nested_map_cond_real(self):
        # 定义一个返回两个输入张量元素对应位置相乘的函数
        def true_fn(x, y):
            return x * y

        # 定义一个返回两个输入张量元素对应位置相加的函数
        def false_fn(x, y):
            return x + y

        # 定义一个函数，根据输入的pred张量选择调用true_fn或false_fn函数
        def f(x, pred, y):
            return cond(pred, true_fn, false_fn, [x, y])

        # 定义一个函数，使用控制流模块中的map函数对输入列表中的每个元素应用f函数
        def g(pred, xs, y):
            return control_flow.map(f, xs, pred, y)

        # 使用make_fx函数创建一个功能化对象gm，用于模拟实际运行模式
        gm = make_fx(g, tracing_mode="real")(
            torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        # 定义pred张量为False
        pred = torch.tensor(False)
        # 生成一个形状为(3, 2, 4)的随机张量x
        x = torch.randn(3, 2, 4)
        # 生成一个形状为(4,)的随机张量y
        y = torch.randn(4)
        # 调用gm对象，并传入pred, x, y，得到结果res
        res = gm(pred, x, y)
        # 使用断言检查res是否等于调用g函数时的结果
        self.assertEqual(res, g(pred, x, y))
        # 使用check_map_count函数检查gm对象中map函数调用的次数是否为1
        self.check_map_count(gm, 1)

    def test_nested_map_cond_symbolic(self):
        # 定义一个返回两个输入张量元素对应位置相乘的函数
        def true_fn(x, y):
            return x * y

        # 定义一个返回两个输入张量元素对应位置相加的函数
        def false_fn(x, y):
            return x + y

        # 定义一个函数，根据输入的pred张量选择调用true_fn或false_fn函数
        def f(x, pred, y):
            return cond(pred, true_fn, false_fn, [x, y])

        # 定义一个函数，使用控制流模块中的map函数对输入列表中的每个元素应用f函数
        def g(pred, xs, y):
            return control_flow.map(f, xs, pred, y)

        # 使用make_fx函数创建一个功能化对象gm，用于模拟符号化运行模式
        gm = make_fx(g, tracing_mode="symbolic")(
            torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        # 定义pred张量为False
        pred = torch.tensor(False)
        # 生成一个形状为(3, 2, 2)的随机张量x
        x = torch.randn(3, 2, 2)
        # 生成一个形状为(2,)的随机张量y
        y = torch.randn(2)
        # 调用gm对象，并传入pred, x, y，得到结果res
        res = gm(pred, x, y)
        # 使用断言检查res是否等于调用g函数时的结果
        self.assertEqual(res, g(pred, x, y))
        # 使用check_map_count函数检查gm对象中map函数调用的次数是否为1
        self.check_map_count(gm, 1)

    def test_nested_cond_map_cond_symbolic(self):
        # 定义一个返回两个输入张量元素对应位置相乘的函数
        def true_fn(x, y):
            return x * y

        # 定义一个返回两个输入张量元素对应位置相加的函数
        def false_fn(x, y):
            return x + y

        # 定义一个函数，根据输入的pred张量选择调用true_fn或false_fn函数
        def f(x, pred, y):
            return cond(pred, true_fn, false_fn, [x, y])

        # 定义一个函数，使用控制流模块中的map函数对输入列表中的每个元素应用f函数
        def g(pred, xs, y):
            return control_flow.map(f, xs, pred, y)

        # 定义一个函数，根据输入的pred张量选择调用g中的true_fn或false_fn函数
        def main_true_fn(pred, xs, y):
            return g(pred, xs, y) * 2

        # 定义一个函数，根据输入的pred张量选择调用g中的true_fn或false_fn函数
        def main_false_fn(pred, xs, y):
            return g(pred, xs, y) + 1

        # 定义一个函数，根据输入的p张量选择调用main_true_fn或main_false_fn函数
        def main(p, pred, xs, y):
            return cond(p, main_true_fn, main_false_fn, [pred, xs, y])

        # 使用make_fx函数创建一个功能化对象gm，用于模拟符号化运行模式
        gm = make_fx(main, tracing_mode="symbolic")(
            torch.tensor(True), torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4)
        )
        # 定义p张量为False
        p = torch.tensor(False)
        # 定义pred张量为False
        pred = torch.tensor(False)
        # 生成一个形状为(3, 2, 2)的随机张量xs
        xs = torch.randn(3, 2, 2)
        # 生成一个形状为(2,)的随机张量y
        y = torch.randn(2)
        # 调用gm对象，并传入p, pred, xs, y，得到结果res
        res = gm(p, pred, xs, y)
        # 使用断言检查res是否等于调用main函数时的结果
        self.assertEqual(res, main(p, pred, xs, y))
        # 使用check_map_count函数检查gm对象中map函数调用的次数是否为2
        self.check_map_count(gm, 2)
    # 定义一个测试函数，使用 self 参数，表示这是一个类方法
    def test_cond_with_sym_pred(self):
        
        # 定义一个返回参数加倍的函数 true_fn
        def true_fn(x):
            return x + x
        
        # 定义一个返回参数平方的函数 false_fn
        def false_fn(x):
            return x * x
        
        # 定义一个函数 foo，根据条件选择 true_fn 或 false_fn 来处理参数 x
        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        
        # 使用 make_fx 函数生成一个图模式函数 gm，并设置跟踪模式为 "symbolic"，传入参数 torch.ones(3, 2, 1)
        gm = make_fx(foo, tracing_mode="symbolic")(torch.ones(3, 2, 1))
        
        # 断言图模式函数 gm 的 shape_env 中的 guards 数量为 0
        self.assertEqual(len(gm.shape_env.guards), 0)
        
        # 使用 self 断言 gm.code 去除首尾空格后与预期的内联代码匹配
        self.assertExpectedInline(
            gm.code.strip(),
            """\
# 定义一个方法用于模型前向传播，接受一个输入参数 x_1
def forward(self, x_1):
    # 调用 Torch 操作符 aten.sym_size.int 获取张量 x_1 的第一个维度大小
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    # 检查 sym_size_int 是否等于 4，并将结果存储在 eq 变量中
    eq = sym_size_int == 4;  sym_size_int = None
    # 从 self 对象中获取 true_graph_0 和 false_graph_0
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    # 调用 Torch 操作符 higher_order.cond，根据 eq 的值选择 true_graph_0 或 false_graph_0，并传入参数列表 [x_1]
    conditional = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, [x_1]);  eq = true_graph_0 = false_graph_0 = x_1 = None
    # 从 conditional 结果中获取第一个元素并存储在 getitem 变量中
    getitem = conditional[0];  conditional = None
    # 返回 getitem 作为该方法的输出
    return getitem
    # 调用 Torch 模块中的高阶条件运算函数 cond，传入参数为 False，true_graph_0，false_graph_0，以及一个列表包含 x_1, _tensor_constant0, _tensor_constant1
    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [x_1, _tensor_constant0, _tensor_constant1]);
    # 将 cond 函数返回值的第一个元素赋值给 getitem
    getitem = conditional[0];
    # 清空变量 conditional，释放其所占用的内存
    conditional = None
    # 返回 getitem 变量的值作为函数的结果
    return getitem
def forward(self, arg0_1, arg1_1, arg2_1):
    # 调用 PyTorch 的底层 ATen 操作库，执行张量相加操作，并将结果存储在变量 add 中
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    # 返回 add 的元组作为结果
    return (add,)



    def test_cond_with_module_param_closure(self):
        # 定义一个继承自 torch.nn.Module 的子类 Mod
        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 注册一个参数 param，形状为 (2, 3)，不需要梯度
                self.register_parameter(
                    "param", torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                )
                # 注册一个缓冲区 buffer，形状为 (2, 3)，值为 2
                self.register_buffer("buffer", torch.ones(2, 3) + 1)

        # 创建 Mod 类的实例 my_mode
        my_mode = Mod()

        # 定义 true_fn 函数，将输入 x 与 my_mode.param 相加
        def true_fn(x):
            return x + my_mode.param

        # 定义 false_fn 函数，将输入 x 与 my_mode.buffer 相加
        def false_fn(x):
            return x + my_mode.buffer

        # 定义 foo 函数，根据条件判断 x 的行数是否为 4，选择执行 true_fn 或 false_fn
        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # 创建输入张量 inp，形状为 (2, 3)
        inp = torch.ones(2, 3)
        # 调用 _check_closure_correctly_lifted_with_mutation 方法验证闭包 lift 是否正确
        self._check_closure_correctly_lifted_with_mutation(
            foo, (my_mode.param, my_mode.buffer), args=(inp,), exp_arg_num=3
        )



    def test_cond_with_module_python_scalar_closure(self):
        # 定义 foo 函数，内部定义了标量张量 a 和 Python 标量 b
        def foo(x):
            a = torch.ones(1, 1)
            b = 1

            # 定义 true_fn 函数，将输入 x 与 a 相加
            def true_fn(x):
                return x + a

            # 定义 false_fn 函数，将输入 x 与 b 相加
            def false_fn(x):
                return x + b

            # 调用 cond 函数，根据条件判断 x 的行数是否为 4，选择执行 true_fn 或 false_fn
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # 创建输入张量 inp，形状为 (2, 3)
        inp = torch.ones(2, 3)
        # 预期结果为 inp + 1
        res = inp + 1
        # 调用 _check_closure_correctly_lifted 方法验证闭包 lift 是否正确
        self._check_closure_correctly_lifted(
            foo, args=(inp,), exp_res=res, exp_arg_num=2
        )



    def test_cond_nested_with_closure(self):
        # 创建张量 a，形状为 (1, 1)，值为 1
        a = torch.ones(1, 1)
        # 创建张量 b，形状为 (1, 1)，值为 2
        b = torch.ones(1, 1) + 1

        # 定义 inner_true_fn 函数，将输入 x 与 a 相加
        def inner_true_fn(x):
            return x + a

        # 定义 inner_false_fn 函数，将输入 x 与 b 相加
        def inner_false_fn(x):
            return x + b

        # 定义 foo 函数，根据条件判断 x 的行数是否为 4，选择执行 true_fn 或 false_fn
        def foo(x):
            # 定义 true_fn 函数，根据条件判断 x 的行数是否为 2，选择执行 inner_true_fn 或 inner_false_fn
            def true_fn(x):
                return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

            # 定义 false_fn 函数，根据条件判断 x 的行数是否大于 4，选择执行 inner_true_fn 或 inner_false_fn
            def false_fn(x):
                return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])

            # 调用 cond 函数，根据条件判断 x 的行数是否为 4，选择执行 true_fn 或 false_fn
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # 创建输入张量 inp，形状为 (2, 3)
        inp = torch.ones(2, 3)
        # 调用 _check_closure_correctly_lifted_with_mutation 方法验证闭包 lift 是否正确
        self._check_closure_correctly_lifted_with_mutation(
            foo, (a, b), args=(inp,), exp_arg_num=3
        )
        def test_cond_nested_with_closure_graph_module(self):
            # 创建一个形状为 [1, 1] 的张量，所有元素值为 1
            a = torch.ones(1, 1)
            # 创建一个形状为 [1, 1] 的张量，所有元素值为 2
            b = torch.ones(1, 1) + 1

            # 定义一个内部函数，接受一个参数 x，返回 x 与张量 a 的和
            def inner_true_fn(x):
                return x + a

            # 定义一个内部函数，接受一个参数 x，返回 x 与张量 b 的和
            def inner_false_fn(x):
                return x + b

            # 定义一个函数 foo，接受一个参数 x
            def foo(x):
                # 定义一个内部函数 true_fn，接受一个参数 x，根据条件判断选择调用 inner_true_fn 或 inner_false_fn
                def true_fn(x):
                    return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

                # 定义一个内部函数 false_fn，接受一个参数 x，根据条件判断选择调用 inner_true_fn 或 inner_false_fn
                def false_fn(x):
                    return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])

                # 根据条件判断选择调用 true_fn 或 false_fn
                return cond(x.shape[0] == 4, true_fn, false_fn, [x])

    def test_map_unfunc_boolean_tensor_for_nested_map_cond(self):
        # 定义一个函数 map_fn，接受两个参数 pred 和 x
        def map_fn(pred, x):
            # 定义一个内部函数 fn，接受参数 x 和 pred，根据 pred 的值选择执行 x * 2 或 x / 2
            def fn(x, pred):
                return control_flow.cond(pred, lambda x: x * 2, lambda x: x / 2, (x,))

            # 对输入的 x 列表进行 map 操作，应用 fn 函数
            return control_flow.map(fn, x, pred)

        # 定义一个装饰器函数 f_wrapper，接受一个函数 func 作为参数
        def f_wrapper(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 禁用视图重新应用功能
                torch._enable_functionalization(reapply_views=False)
                try:
                    # 将 args 和 kwargs 中的张量转换为旧的功能版本（假设函数 to_fun_old 存在）
                    func_args = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        args,
                    )
                    func_kwargs = pytree.tree_map(
                        lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x,
                        kwargs,
                    )
                    # 调用 func，并将结果转换回新的功能版本（假设函数 from_fun_old 存在）
                    return pytree.tree_map(
                        from_fun_old, func(*func_args, **func_kwargs)
                    )
                finally:
                    # 禁用功能化
                    torch._disable_functionalization()

            return wrapper

        # 使用 f_wrapper 装饰 map_fn 函数，返回一个图模块（假设 make_fx 和 assertExpectedInline 是已定义的函数）
        gm = make_fx(f_wrapper(map_fn))(
            torch.tensor(True), torch.ones([2, 3], requires_grad=False)
        )
        # 断言生成的图模块的代码与预期一致
        self.assertExpectedInline(
            gm.code.strip(),
            """\
# 定义一个方法 forward，接受参数 self, pred_1, x_1
def forward(self, pred_1, x_1):
    # 将 self.body_graph_0 赋值给变量 body_graph_0
    body_graph_0 = self.body_graph_0
    # 调用 torch.ops.higher_order.map_impl 方法，传入 body_graph_0, [x_1], [pred_1] 参数，并将结果赋值给 map_impl
    map_impl = torch.ops.higher_order.map_impl(body_graph_0, [x_1], [pred_1])
    # 清空变量 body_graph_0, x_1, pred_1
    body_graph_0 = x_1 = pred_1 = None
    # 获取 map_impl 的第一个元素，赋值给 getitem
    getitem = map_impl[0]
    # 清空 map_impl 变量
    map_impl = None
    # 返回 getitem
    return getitem
        # 定义一个返回 x - x.cos() 的函数 true_fn，用于条件为真时的处理
        def true_fn(x):
            return x - x.cos()

        # 定义一个返回 x + x.sin() 的函数 false_fn，用于条件为假时的处理
        def false_fn(x):
            return x + x.sin()

        # 定义一个函数 foo，根据条件判断 x 的形状是否为 (4,)，选择不同的处理函数
        def foo(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        # 定义输入的几种不同形状的张量
        inps = (torch.ones(3, 4), torch.ones(3, 5), torch.ones(5, 4), torch.ones(5, 3))
        # 对每一个输入张量进行处理
        for inp in inps:
            # 使用 make_fx 函数将 foo 转换为图模式，并传入张量 torch.ones(3, 4)
            gm = make_fx(foo, tracing_mode="symbolic")(torch.ones(3, 4))
            # 使用断言检查生成的图模式代码是否符合预期
            self.assertExpectedInline(
                gm.code.strip(),
                """\
def forward(self, x_1):
    # 调用自定义的 Torch 操作符 'sym_size.int' 获取张量 x_1 在第 0 维的大小，并赋值给 sym_size_int
    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)
    # 检查 sym_size_int 是否等于 4，将结果赋给 eq；清空 sym_size_int 的引用
    eq = sym_size_int == 4;  sym_size_int = None
    # 获取当前对象的 true_graph_0 和 false_graph_0 属性
    true_graph_0 = self.true_graph_0
    false_graph_0 = self.false_graph_0
    # 根据条件 eq 调用 torch.ops.higher_order.cond 运算符，选择执行 true_graph_0 或 false_graph_0 中的一个，传入 x_1 作为参数；清空 eq, true_graph_0, false_graph_0, x_1 的引用
    conditional = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, [x_1]);  eq = true_graph_0 = false_graph_0 = x_1 = None
    # 从条件运算结果 conditional 中获取第一个元素并赋给 getitem；清空 conditional 的引用
    getitem = conditional[0];  conditional = None
    # 返回 getitem
    return getitem
    # 根据 inner_fn_type 初始化内部函数对象
    def _init_fn(self, inner_fn_type):
        # 如果 inner_fn_type 是 "function"，返回 reduce_func 函数
        if inner_fn_type == "function":
            return reduce_func
        # 如果 inner_fn_type 是 "module"，返回 ReduceMod 类的实例
        elif inner_fn_type == "module":
            return ReduceMod()
        # 如果 inner_fn_type 是 "object"，返回 ReduceObj 类的实例
        elif inner_fn_type == "object":
            return ReduceObj()
        # 如果 inner_fn_type 不在已定义的类型中，抛出 NotImplementedError 异常
        else:
            raise NotImplementedError

    # 使用参数化测试执行多个测试用例，测试条件跟踪函数在不同输入下的行为
    @parametrize("predType", ["bool", "intTensor", "floatTensor", "boolTensor"])
    @parametrize("innerFnType", ["function", "module", "object"])
    @parametrize("nOperands", [0, 1])
    @parametrize("nClosure", [0, 1])
    @parametrize("nesting", [0, 2])
    def test_cond_tracing_with_valid_inputs(
        self, predType, innerFnType, nOperands, nClosure, nesting
    ):
        # 初始化条件断言函数
        pred = self._init_predicate(predType)
        # 根据 innerFnType 初始化内部函数对象
        inner_fn = self._init_fn(innerFnType)
        # 创建操作数列表，其中包含若干个大小为 (2, 3) 的张量，数量由 nOperands 决定
        operands = [torch.ones(2, 3) + i for i in range(nOperands)]
        # 创建闭包列表，其中包含若干个大小为 (2, 3) 的张量，数量由 nClosure 决定
        closure = [torch.ones(2, 3) - i for i in range(nClosure)]
        # 创建测试函数的参数和函数
        args, fn = self._create_test_fns_for_cond(
            pred, inner_fn, operands, closure, nesting
        )
        # 在 eager 模式下执行测试函数，获取其结果
        eager_res = fn(*args)
        # 对于三种追踪模式分别执行符号化追踪
        for tracing_mode in ["symbolic", "fake", "real"]:
            # 在子测试中设置 tracing_mode，并允许 fake 通过闭包传播
            with self.subTest(tracing_mode=tracing_mode):
                # 使用 make_fx 函数进行符号化追踪，并传入允许 fake 输入的标志
                gm = make_fx(
                    fn, tracing_mode=tracing_mode, _allow_non_fake_inputs=True
                )(*args)
                # 断言符号化追踪函数执行的结果与 eager 模式下的结果一致
                self.assertEqual(gm(*args), eager_res)

    # 使用参数化测试执行多个测试用例，测试 torch.vmap 函数在不同输入下的行为
    @parametrize("predType", ["boolTensor"])
    @parametrize("innerFnType", ["function", "module", "object"])
    @parametrize("nOperands", [1, 2])
    @parametrize("nClosure", [0, 1])
    @parametrize("nesting", [0])
    def test_cond_vmap(self, predType, innerFnType, nOperands, nClosure, nesting):
        # 初始化条件断言函数
        pred = self._init_predicate(predType)
        # 根据 innerFnType 初始化内部函数对象
        inner_fn = self._init_fn(innerFnType)
        # 创建操作数列表，其中包含若干个大小为 (2, 3) 的张量，数量由 nOperands 决定
        operands = [torch.ones(2, 3) + i for i in range(nOperands)]
        # 创建闭包列表，其中包含若干个大小为 (2, 3) 的张量，数量由 nClosure 决定
        closure = [torch.ones(2, 3) - i for i in range(nClosure)]
        # 创建测试函数的参数和函数
        args, fn = self._create_test_fns_for_cond(
            pred, inner_fn, operands, closure, nesting
        )
        # 在 eager 模式下执行测试函数，获取其结果
        eager_res = fn(*args)
        # 使用 torch.vmap 函数对测试函数进行向量化映射
        out = torch.vmap(fn)(*args)
        # 根据闭包的数量进行不同的断言处理
        if nClosure == 0:
            # 如果没有闭包，直接比较 eager 模式下的结果和向量化映射的结果
            self.assertEqual(eager_res, out)
        else:
            # 如果有闭包，分别比较 eager 模式下的结果和向量化映射的两个元素的结果
            self.assertEqual(eager_res, out[0])
            self.assertEqual(eager_res, out[1])

    # 测试简单的 torch.vmap 函数行为
    def test_cond_vmap_simple(self):
        # 定义简单的条件函数 fn，根据条件返回不同的结果
        def fn(x):
            return torch.cond(
                pred=torch.tensor([True]),
                true_fn=lambda x: x + 100,
                false_fn=lambda x: x,
                operands=(x,),
            )

        # 创建大小为 (3, 5) 的张量 a
        a = torch.arange(15).reshape((3, 5))
        # 使用 torch.vmap 函数对 fn 进行向量化映射
        res = torch.vmap(fn, in_dims=(0,))(a)
        # 断言向量化映射后的结果形状为 (3, 5)
        self.assertEqual(res.shape, (3, 5))
        # 断言向量化映射后的结果与预期结果 a + 100 相等
        self.assertEqual(res, a + 100)
    # 定义一个测试函数，测试 torch.cond 结合 torch.vmap 对多个输入进行条件操作
    def test_cond_vmap_multiple_inputs(self):
        # 定义一个函数 fn，接受两个输入 x 和 y
        def fn(x, y):
            # 使用 torch.cond 根据条件判断 x.sum() < y.sum()，选择 true_fn 或 false_fn
            return torch.cond(
                pred=x.sum() < y.sum(),  # 判断条件：x 的总和是否小于 y 的总和
                true_fn=lambda x, y: x + 100,  # 如果条件为真，返回 x + 100
                false_fn=lambda x, y: y,  # 如果条件为假，返回 y
                operands=(x, y),  # 操作数，传递给 true_fn 或 false_fn 的参数
            )

        # 创建一个 3x5 的张量 a，其中元素为 0 到 14
        a = torch.arange(15).reshape(3, 5)
        # 创建一个与 a 同样大小的张量 b，元素值为 a 的各元素加 3
        b = torch.ones_like(a) + 3
        # 使用 torch.vmap 对函数 fn 进行向量化映射，in_dims=(0, 0) 表示 fn 的两个输入都是第 0 维批处理
        res = torch.vmap(fn, in_dims=(0, 0))(a, b)
        # 创建一个预期结果的张量，与 res 结果对比
        expected = torch.tensor(
            [[100, 101, 102, 103, 104], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]
        )
        # 断言 res 的形状为 (3, 5)
        self.assertEqual(res.shape, (3, 5))
        # 断言 res 等于预期的结果张量 expected
        self.assertEqual(expected, res)

    # 定义一个测试函数，测试 torch.cond 结合 torch.vmap 对单个输入进行条件操作，使用了闭包
    def test_cond_vmap_single_input_with_closure(self):
        # 创建一个 3x5 的张量 a，元素值为 4
        a = torch.ones((3, 5)) + 3
        # 创建一个长度为 5 的张量 c，元素为 0 到 4
        c = torch.arange(5)

        # 定义一个函数 fn，接受一个输入 x
        def fn(x):
            # 使用 torch.cond 固定为 pred=torch.tensor([True])，选择 true_fn 或 false_fn
            return torch.cond(
                pred=torch.tensor([True]),  # 固定为 True
                true_fn=lambda x: x + c,  # 返回 x 加上 c 中各元素的结果
                false_fn=lambda x: x - c,  # 返回 x 减去 c 中各元素的结果
                operands=(x,),  # 操作数，传递给 true_fn 或 false_fn 的参数
            )

        # 使用 torch.vmap 对函数 fn 进行向量化映射，in_dims=(0,) 表示 fn 的输入在第 0 维批处理
        res = torch.vmap(fn, in_dims=(0,))(a)
        # 断言 res 等于 a 加上 c 中各元素的结果
        self.assertEqual(a + c, res)

    # 定义一个测试函数，测试 torch.cond 结合 torch.vmap 对多个输入进行条件操作，使用了闭包
    def test_cond_vmap_multiple_args_with_closure(self):
        # 创建一个 3x5 的张量 a，元素值为 4
        a = torch.ones((3, 5), dtype=torch.int64) + 3
        # 创建一个 3x5 的张量 b，元素为 0 到 14
        b = torch.arange(15).reshape(3, 5)
        # 创建一个长度为 5 的张量 c，元素为 0 到 4
        c = torch.arange(5)

        # 定义一个函数 fn，接受两个输入 x 和 y
        def fn(x, y):
            # 使用 torch.cond 固定为 pred=torch.tensor([False])，选择 true_fn 或 false_fn
            return torch.cond(
                pred=torch.tensor([False]),  # 固定为 False
                true_fn=lambda x, y: x + c,  # 返回 x 加上 c 中各元素的结果
                false_fn=lambda x, y: y - c,  # 返回 y 减去 c 中各元素的结果
                operands=(x, y),  # 操作数，传递给 true_fn 或 false_fn 的参数
            )

        # 使用 torch.vmap 对函数 fn 进行向量化映射
        res = torch.vmap(fn)(a, b)
        # 断言 res 等于 b 减去 c 中各元素的结果
        self.assertEqual(b - c, res)

    # 参数化测试函数，测试 torch.cond 结合 torch.vmap 对多个输出进行条件操作，根据 nClosure 的值选择不同的闭包
    @parametrize("nClosure", [0, 1])
    def test_cond_vmap_multiple_outputs(self, nClosure):
        if nClosure:
            # 创建一个长度为 5 的张量 c，元素为 6
            c = torch.ones(5, dtype=torch.int64) + 5

            # 定义一个函数 fn，接受一个输入 x
            def fn(x):
                # 使用 torch.cond 固定为 pred=torch.tensor([True])，选择 true_fn 或 false_fn
                return torch.cond(
                    pred=torch.tensor([True]),  # 固定为 True
                    true_fn=lambda x: (x + c, x - c),  # 返回 x 加上 c 和 x 减去 c 的结果
                    false_fn=lambda x: (x, x),  # 返回 x 和 x 的结果
                    operands=(x,),  # 操作数，传递给 true_fn 或 false_fn 的参数
                )

        else:
            # 定义一个函数 fn，接受一个输入 x
            def fn(x):
                # 使用 torch.cond 固定为 pred=torch.tensor([True])，选择 true_fn 或 false_fn
                return torch.cond(
                    pred=torch.tensor([True]),  # 固定为 True
                    true_fn=lambda x: (x + 1, x - 1),  # 返回 x 加 1 和 x 减 1 的结果
                    false_fn=lambda x: (x, x),  # 返回 x 和 x 的结果
                    operands=(x,),  # 操作数，传递给 true_fn 或 false_fn 的参数
                )

        # 创建一个 3x5 的张量 a，其中元素为 0 到 14
        a = torch.arange(15).reshape(3, 5)
        # 使用 torch.vmap 对函数 fn 进行向量化映射
        res = torch.vmap(fn)(
            a,
        )
        # 断言 res 的长度为 2
        self.assertEqual(len(res), 2)
        if nClosure:
            # 如果 nClosure 为真，则断言 res 等于 a 加上 c 和 a 减去 c 的结果
            self.assertEqual(res, (a + c, a - c))
        else:
            # 如果 nClosure 为假，则断言 res 等于 a 加 1 和 a 减 1 的结果
            self.assertEqual(res, (a + 1, a - 1))
    # 定义一个测试方法，测试 torch.vmap 的使用
    def test_vmap_vmap(self):
        # 定义一个操作函数 fn，根据条件选择对输入 x 进行加一或减一的操作
        def fn(x):
            return torch.cond(
                pred=torch.tensor([True]),  # 条件为始终为真的张量
                true_fn=lambda x: x + 1,    # 如果条件为真，则 x 加一
                false_fn=lambda x: x - 1,   # 如果条件为假，则 x 减一
                operands=(x,),             # 操作的输入为 x
            )

        # 定义一个包装函数 wrapper，对输入 x 应用 torch.vmap
        def wrapper(x):
            return torch.vmap(fn)(x)

        # 创建一个维度为 (3, 4, 5) 的全为 1 的张量 a
        a = torch.ones((3, 4, 5))
        # 对 wrapper 函数应用 torch.vmap，将操作应用于 a
        res = torch.vmap(wrapper)(a)
        # 断言结果 res 等于 a 加一
        self.assertEqual(res, a + 1)

    # 定义一个测试方法，测试在条件语句中使用 set_ 方法和张量变量的变化
    def test_cond_trace_set__and_mutate_input(self):
        # 定义函数 f，接受两个张量 a 和 tmp 作为输入
        def f(a, tmp):
            # 将 a 展平为一维视图
            a_view = a.view(-1)
            # 使用 torch.no_grad 禁止梯度跟踪
            with torch.no_grad():
                # 使用 tmp 设置 a 的值，此处会导致图形中断
                a.set_(tmp)
                # 将 a_view 中的所有元素乘以 2
                a_view.mul_(2)
            # 返回 a 加上 tmp 的结果
            return a + tmp

        # 创建一个全为 1 的 3x3 张量 inp，并开启梯度跟踪
        inp = torch.ones(3, 3, requires_grad=True)
        # 创建一个全为 1 的 3x3 张量 tmp，并开启梯度跟踪
        tmp = torch.ones(3, 3, requires_grad=True)

        # 使用 torch.cond 调用函数 f，并期望捕获错误
        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Cond doesn't work unless it is captured completely with torch.compile",
        ):
            # 当 inp 的总和大于 0 时，调用函数 f
            torch.cond(inp.sum() > 0, f, f, (inp, tmp))

    # 定义一个测试方法，测试在条件语句中使用 set_ 方法和修改中间变量的影响
    def test_cond_trace_set__and_mutate_intermediate(self):
        # 定义函数 f，接受两个张量 a 和 tmp 作为输入
        def f(a, tmp):
            # 克隆张量 a
            a = a.clone()
            # 将 a 展平为一维视图
            a_view = a.view(-1)
            # 克隆张量 tmp
            tmp = tmp.clone()
            # 使用 torch.no_grad 禁止梯度跟踪
            with torch.no_grad():
                # 使用 tmp 设置 a 的值，此处会导致图形中断
                a.set_(tmp)
                # 将 a_view 中的所有元素乘以 2
                a_view.mul_(2)
            # 返回 a 加上 tmp 的结果
            return a + tmp

        # 创建一个全为 1 的 3x3 张量 inp，并开启梯度跟踪
        inp = torch.ones(3, 3, requires_grad=True)
        # 创建一个全为 1 的 3x3 张量 tmp，并开启梯度跟踪
        tmp = torch.ones(3, 3, requires_grad=True)

        # 定义一个继承自 torch.nn.Module 的类 Mod
        class Mod(torch.nn.Module):
            # 定义 Mod 类的前向传播方法
            def forward(self, inp: torch.Tensor, tmp: torch.Tensor) -> torch.Tensor:
                # 使用 torch.cond 调用函数 f，并期望捕获错误
                return torch.cond(inp.sum() > 0, f, f, (inp, tmp))

        # 使用 torch.compile 编译 Mod 类，并期望捕获错误
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            # 使用 AOT eager 后端编译 Mod 类
            out = torch.compile(Mod(), backend="aot_eager")(inp, tmp)

        # 使用 torch.compile 编译 Mod 类，并期望捕获错误
        with self.assertRaisesRegex(
            RuntimeError, "cannot mutate tensors with frozen storage"
        ):
            # 使用 Inductor 后端编译 Mod 类
            out = torch.compile(Mod(), backend="inductor")(inp, tmp)

        # 导入 EagerAndRecordGraphs 测试模块
        from torch._dynamo.testing import EagerAndRecordGraphs

        # 创建 EagerAndRecordGraphs 后端实例
        backend = EagerAndRecordGraphs()
        # 使用 backend 编译 Mod 类
        out = torch.compile(Mod(), backend=backend)(inp, tmp)
        # 断言 backend.graphs[0].cond_true_0.code 的结果
        self.assertExpectedInline(
            backend.graphs[0].cond_true_0.code.strip("\n"),
            """\
# 定义一个类方法 `forward`，接收两个参数 `l_inp_` 和 `l_tmp_`
def forward(self, l_inp_, l_tmp_):
    # 复制 `l_inp_` 并赋给 `l_inp__1`，同时清空 `l_inp_`
    l_inp__1 = l_inp_
    # 复制 `l_tmp_` 并赋给 `l_tmp__1`，同时清空 `l_tmp_`
    l_tmp__1 = l_tmp_
    # 克隆 `l_inp__1` 并赋给 `clone`，然后清空 `l_inp__1`
    clone = l_inp__1.clone();  l_inp__1 = None
    # 将 `clone` 进行形状重塑成一维
    view = clone.view(-1)
    # 克隆 `l_tmp__1` 并赋给 `clone_1`，然后清空 `l_tmp__1`
    clone_1 = l_tmp__1.clone();  l_tmp__1 = None
    # 关闭梯度追踪
    _set_grad_enabled = torch._C._set_grad_enabled(False)
    # 将 `clone` 设置为 `clone_1` 的值
    set_ = clone.set_(clone_1)
    # 将 `view` 中的每个元素乘以 2，并清空 `view`
    mul_ = view.mul_(2);  view = None
    # 开启梯度追踪
    _set_grad_enabled_1 = torch._C._set_grad_enabled(True)
    # 将 `clone` 和 `clone_1` 相加，然后清空 `clone` 和 `clone_1`
    add = clone + clone_1;  clone = clone_1 = None
    # 返回一个包含 `add` 的元组
    return (add,)
```