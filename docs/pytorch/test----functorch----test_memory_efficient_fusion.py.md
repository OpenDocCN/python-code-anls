# `.\pytorch\test\functorch\test_memory_efficient_fusion.py`

```
# Owner(s): ["module: functorch"]

# 导入必要的模块和库
import inspect
import random
import unittest
from typing import Callable

import torch
import torch.fx as fx  # 导入 torch.fx 模块，用于函数级别的分析和转换
import torch.nn as nn

from functorch import make_fx  # 导入 functorch 库中的 make_fx 函数
from functorch.compile import memory_efficient_fusion  # 导入 functorch 库中的内存高效融合函数
from torch._functorch.compile_utils import fx_graph_cse  # 导入 torch._functorch.compile_utils 库中的 fx_graph_cse 函数
from torch.nn import functional as F  # 导入 torch.nn.functional 模块，并使用 F 作为别名
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关的函数和类

HAS_CUDA = torch.cuda.is_available()  # 检查当前系统是否支持 CUDA


def _num_args(fn: Callable):
    # 返回给定函数的参数数量
    return len(inspect.signature(fn).parameters)


def gelu_bias(bias, y):
    # 实现带有偏置的 GELU 激活函数
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def swish(x):
    # 实现 Swish 激活函数
    return x * torch.sigmoid(x)


def mish(x):
    # 实现 Mish 激活函数
    return x.mul(torch.tanh(F.softplus(x)))


def hard_sigmoid(x):
    # 实现 Hard Sigmoid 激活函数
    return (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_swish(x):
    # 实现 Hard Swish 激活函数
    return x * (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)


def hard_mish(x):
    # 实现 Hard Mish 激活函数
    return 0.5 * x * (x + 2.0).clamp(min=0.0, max=2.0)


# todo: convert these into tests
# 以下两个函数用于计算 Group Standard Deviation，目前暂时注释掉，未来可能用于测试
# def group_std(x, groups: int = 32, eps: float = 1e-5, flatten: bool = False):
#     B, C, H, W = x.shape
#     x_dtype = x.dtype
#     if flatten:
#         x = x.reshape(B, groups, -1)  # FIXME simpler shape causing TPU / XLA issues
#         std = x.float().var(dim=2, unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     else:
#         x = x.reshape(B, groups, C // groups, H, W)
#         std = x.float().var(dim=(2, 3, 4), unbiased=False, keepdim=True).add(eps).sqrt().to(x_dtype)
#     return std.expand(x.shape).reshape(B, C, H, W)

# class EvoNorm2dS0(nn.Module):
#     def __init__(self, num_features, groups=32, group_size=None, apply_act=True, eps=1e-5, **_):
#         super().__init__()
#         self.apply_act = apply_act  # 是否应用激活函数（非线性）
#         if group_size:
#             assert num_features % group_size == 0
#             self.groups = num_features // group_size
#         else:
#             self.groups = groups
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(num_features))
#         self.bias = nn.Parameter(torch.zeros(num_features))
#         self.v = nn.Parameter(torch.ones(num_features)) if apply_act else None
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.ones_(self.weight)
#         nn.init.zeros_(self.bias)
#         if self.v is not None:
#             nn.init.ones_(self.v)

#     def forward(self, x):
#         x_dtype = x.dtype
#         v_shape = (1, -1, 1, 1)
#         if self.v is not None:
#             v = self.v.view(v_shape).to(dtype=x_dtype)
#             x = x * (x * v).sigmoid() / group_std(x, self.groups, self.eps)
#         return x * self.weight.view(v_shape).to(dtype=x_dtype) + self.bias.view(v_shape).to(dtype=x_dtype)


# device = "cuda"
# dtype = torch.float

# evo_norm = EvoNorm2dS0(2048)
# evo_norm_inp = [(128, 2048, 8, 8)]


def run_and_compare_activation(self, fn, inps):
    # 这个函数可能用于运行和比较激活函数，但未完全实现，暂时保留
    # 使用指定的torch.jit.fuser设置张量操作的融合策略为"fuser1"
    with torch.jit.fuser("fuser1"):
        # 将设备类型设为"cuda"
        device = "cuda"
        # 将数据类型设为torch.float
        dtype = torch.float
        # 如果fn是nn.Module的实例，则将其移动到指定设备并设置数据类型
        if isinstance(fn, nn.Module):
            fn = fn.to(device=device, dtype=dtype)

        # 生成参考输入张量列表，每个张量在指定设备上，数据类型为指定类型，并要求梯度计算
        ref_args = [
            torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            for shape in inps
        ]
        # 复制参考输入张量列表，并将其分离出来，同时要求保留梯度计算的状态
        res_args = [i.clone().detach().requires_grad_(True) for i in ref_args]

        # 使用fn计算参考输出张量
        ref = fn(*ref_args)
        # 对参考输出张量的所有元素求和，并执行反向传播
        ref.sum().backward()

        # 使用memory_efficient_fusion函数优化内存后的fn
        mem_optimized_fn = memory_efficient_fusion(fn)
        # 迭代5次
        for _ in range(5):
            # 将结果张量的梯度置为None
            for i in res_args:
                i.grad = None
            # 使用优化后的函数计算结果张量
            res = mem_optimized_fn(*res_args)
            # 对结果张量的所有元素求和，并执行反向传播
            res.sum().backward()

        # 断言参考输出与结果输出张量相等
        self.assertEqual(ref, res)
        # 对比参考输入张量和结果输入张量的梯度是否相等
        for ref_arg, res_arg in zip(ref_args, res_args):
            self.assertEqual(ref_arg.grad, res_arg.grad)
# 如果 CUDA 不可用，则跳过测试
@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
# 定义测试类 TestMemoryEfficientOpAuthoring，继承自 TestCase
class TestMemoryEfficientOpAuthoring(TestCase):
    
    # 测试 gelu_bias 函数
    def test_gelu_bias(self):
        # 调用 run_and_compare_activation 函数测试 gelu_bias 函数
        run_and_compare_activation(self, gelu_bias, [(1024,), (1024,)])

    # 测试 mish 函数
    def test_mish(self):
        # 调用 run_and_compare_activation 函数测试 mish 函数
        run_and_compare_activation(self, mish, [(1024,)])

    # 测试 swish 函数
    def test_swish(self):
        # 调用 run_and_compare_activation 函数测试 swish 函数
        run_and_compare_activation(self, swish, [(1024,)])

    # 测试 hard_sigmoid 函数
    def test_hard_sigmoid(self):
        # 调用 run_and_compare_activation 函数测试 hard_sigmoid 函数
        run_and_compare_activation(self, hard_sigmoid, [(1024,)])

    # 测试 hard_swish 函数
    def test_hard_swish(self):
        # 调用 run_and_compare_activation 函数测试 hard_swish 函数
        run_and_compare_activation(self, hard_swish, [(1024,)])

    # 测试 layer_norm 函数
    def test_layer_norm(self):
        # 定义 layer_norm 函数，对输入数据进行层归一化
        def layer_norm(x, weight, bias):
            dim = -1
            eps = 1e-5
            # 计算均值
            mean = torch.mean(x, dim, keepdim=True)
            # 中心化数据
            centered = x - mean
            # 计算方差
            var = torch.sum(centered * centered, dim, keepdim=True) / x.size(-1)
            # 计算标准差的倒数
            rvar = 1.0 / torch.sqrt(var + eps)
            # 归一化
            normed = (x - mean) * rvar
            # 返回归一化后的结果加权和偏置
            return normed * weight + bias
        
        # 定义 batch size 和层大小
        bs = 10
        ln_size = 16
        # 定义 layer_norm 函数的输入参数
        layer_norm_inps = [(bs, ln_size), (ln_size,), (ln_size,)]
        # 调用 run_and_compare_activation 函数测试 layer_norm 函数
        run_and_compare_activation(self, layer_norm, layer_norm_inps)

    # 测试 rmsnorm 函数
    def test_rmsnorm(self):
        # 定义 T5LayerNorm 类，实现 T5 模型风格的层归一化
        class T5LayerNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-6):
                """
                构建 T5 风格的层归一化模块，无偏置和均值减法。
                """
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                # 层归一化应始终在 float32 中计算
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(
                    variance + self.variance_epsilon
                )

                # 如果需要，将结果转换为半精度
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)

                return self.weight * hidden_states
        
        # 定义 batch size、序列长度和隐藏单元数
        bs = 256
        seq = 256
        hidden = 1024
        # 创建 T5LayerNorm 实例
        t5_norm = T5LayerNorm(hidden)
        # 定义 t5_norm 函数的输入参数
        t5_norm_inputs = [(bs, seq, hidden)]
        # 调用 run_and_compare_activation 函数测试 t5_norm 函数
        run_and_compare_activation(self, t5_norm, t5_norm_inputs)

    # TODO - 断言失败
    # def test_hard_mish(self):
    #   for compiler in compilers:
    #     run_and_compare_activation(hard_mish, 1024)

# 检查函数 f 是否在图形 t 中具有减少的节点数，第二次传递时不再减少节点数。
# delta 是大于等于 -1 的整数。如果 delta = -1，则仅检查新图形是否具有少于或等于的节点数。
def check(f, t, delta, check_val=True, graph_input=False):
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)
    # 运行 CSE（公共子表达式消除）优化的函数图形
    new_graph = fx_graph_cse(fx_g.graph)
    # 创建新的 GraphModule，结合原始函数和优化后的函数图形
    new_g = fx.GraphModule(fx_g, new_graph)
    # 记录原始图和新图的节点数目
    old_num_nodes = len(fx_g.graph.nodes)  # 获取原始图中节点的数量
    new_num_nodes = len(new_graph.nodes)  # 获取新图中节点的数量

    # 根据 delta 值进行节点数量的检查
    if delta == -1:
        assert (
            old_num_nodes >= new_num_nodes
        ), f"节点数量增加了 {old_num_nodes}, {new_num_nodes}"
    else:
        assert (
            old_num_nodes == new_num_nodes + delta
        ), f"节点数量不一致 {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}"

    # 第二次遍历不应减少更多节点
    pass_2_graph = fx_graph_cse(new_graph)  # 对新图执行第二遍图形优化
    pass_2_num_nodes = len(pass_2_graph.nodes)  # 获取第二遍优化后的图中节点的数量
    assert (
        pass_2_num_nodes == new_num_nodes
    ), f"第二遍优化后的图节点数少于原始节点数 {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}"

    # 检查计算结果的正确性
    if check_val:
        true_result = fx_g(t)  # 获取原始函数的计算结果
        our_result = new_g(t)  # 获取优化后函数的计算结果
        if true_result is None:  # 如果原始结果为 None，则两者都应该是 None
            assert (
                our_result is None
            ), f"原始结果为 None，但优化结果为 {our_result}"
        else:  # 否则，检查两者的计算结果是否完全相同
            assert torch.all(
                true_result == our_result
            ), f"计算结果不一致 {true_result}, {our_result}"  # 检查结果是否相同
class NoChangeTestCase(TestCase):
    def test_nochange(self):
        # 定义一个函数 f，接受一个参数 x
        def f(x):
            # 计算 a = x + 1
            a = x + 1
            # 计算 b = x + a
            b = x + a
            # 重新赋值 a = x
            a = x
            # 计算 d = x + a
            d = x + a
            # 返回 b + d 的结果
            return b + d
        
        # 生成一个大小为 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入函数 f 和张量 t，还有额外参数 0
        check(f, t, 0)

    def test_empty(self):
        # 定义一个空函数 f，没有实际操作
        def f(x):
            pass
        
        # 生成一个大小为 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入函数 f 和张量 t，还有额外参数 0
        check(f, t, 0)

    def test_rand_like(self):
        # 定义一个函数 f，接受一个参数 x
        def f(x):
            # 生成一个与输入张量 x 相同大小的随机张量 a
            a = torch.rand_like(x)
            # 生成一个与输入张量 x 相同大小的随机张量 b
            b = torch.rand_like(x)
            # 返回 a + b 的结果
            return a + b
        
        # 生成一个大小为 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入函数 f 和张量 t，还有额外参数 0，并且关闭值的检查
        check(f, t, 0, check_val=False)

    def test_rand_n(self):
        # 定义一个函数 f，接受一个参数 x
        def f(x):
            # 生成一个大小为 4 的随机张量 a
            a = torch.randn(4)
            # 生成一个大小为 4 的随机张量 b
            b = torch.randn(4)
            # 返回 a + b 的结果
            return a + b
        
        # 生成一个大小为 2x2 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入函数 f 和张量 t，还有额外参数 0，并且关闭值的检查
        check(f, t, 0, check_val=False)

    def test_hash_with_numbers(self):
        # 测试用例，重现在 fx_graph_cse 中的问题
        # 当 hash((primals_2, 1.0)) == hash((primals_2, 1)) 时的情况
        
        # 如果处于编译状态，则跳过此测试
        if torch._dynamo.is_compiling():
            self.skipTest("Unsupported if test run is compiled")
        
        # 定义一个函数 f，接受两个参数 inpt 和 osize
        def f(inpt, osize):
            # 获取输入张量的最后一个维度的大小
            size = inpt.shape[-1]
            # 计算 s1 = size - 1
            s1 = size - 1
            # 计算 s2 = size - 1.0
            s2 = size - 1.0
            # 计算比例 scale = s2 / (osize - 1.0)
            scale = s2 / (osize - 1.0)
            # 将输入张量 inpt 限制在范围 [0, s1] 内
            inpt = torch.clamp(inpt, 0, s1)
            # 返回 scale 乘以 inpt 的结果
            return scale * inpt
        
        # 初始化动态图列表
        gms = []

        # 定义一个玩具后端函数，将生成的图形添加到 gms 列表中
        def toy_backend(gm, _):
            gms.append(gm)
            return gm.forward
        
        # 重置动态环境
        torch._dynamo.reset()
        # 编译函数 f，使用 toy_backend 作为后端，启用动态图
        fn = torch.compile(backend=toy_backend, dynamic=True)(f)
        
        # 生成一个大小为 [3, 100] 的随机张量 t
        t = torch.rand(3, 100)
        # 调用编译后的函数 fn，传入张量 t 和参数 50
        _ = fn(t, 50)
        # 断言 gms 列表长度为 1
        assert len(gms) == 1, gms
        # 获取第一个动态图 fx_g
        fx_g = gms[0]
        # 调用 check 函数，传入 fx_g，还有额外参数 0，关闭值的检查，并指定输入为图形输入
        check(fx_g, None, 0, check_val=False, graph_input=True)
    # 定义一个测试函数，接受一个参数 x
    def test_two_args(self):
        # 定义内部函数 f，接受一个参数 x
        def f(x):
            # 计算 x 按第一维度求和，结果保存在变量 a 中
            a = x.sum(dim=1)
            # 再次计算 x 按第一维度求和，保持维度，结果保存在变量 b 中
            b = x.sum(dim=1, keepdim=True)
            # 同上，计算 x 按第一维度求和，保持维度，结果保存在变量 c 中
            c = x.sum(dim=1, keepdim=True)
            # 再次计算 x 按第一维度求和，结果保存在变量 d 中
            d = x.sum(dim=1)
            # 返回四个求和结果的总和
            return a + b + c + d

        # 生成一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入 f 函数、张量 t 和数值 2 进行检查
        check(f, t, 2)

    # 定义一个测试函数，接受一个参数 x
    def test_simple_multiple_same_ops(self):
        # 定义内部函数 f，接受一个参数 x
        def f(x):
            # 计算 x 的总和，结果保存在变量 a 中
            a = x.sum()
            # 再次计算 x 的总和，结果保存在变量 b 中
            b = x.sum()
            # 同上，计算 x 的总和，结果保存在变量 c 中
            c = x.sum()
            # 再次计算 x 的总和，结果保存在变量 d 中
            d = x.sum()
            # 返回四次求和结果的总和
            return a + b + c + d

        # 生成一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入 f 函数、张量 t 和数值 3 进行检查
        check(f, t, 3)

    # 定义一个测试函数，接受一个参数 x
    def test_nested_immutable_list_type(self):
        # 定义内部函数 f，接受一个参数 x
        def f(x):
            # 将 x 与自身在第一维度上拼接，结果保存在变量 a 中
            a = torch.cat((x, x))
            # 同上，将 x 与自身在第一维度上拼接，结果保存在变量 b 中
            b = torch.cat((x, x))
            # 返回两次拼接操作的总和
            return a + b

        # 生成一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入 f 函数、张量 t 和数值 1 进行检查
        check(f, t, 1)

    # 定义一个测试函数，接受一个参数 x
    def test_kwarg(self):
        # 定义内部函数 f，接受一个参数 x
        def f(x):
            # 生成一个与 x 同形状的全为 1 的张量，结果保存在变量 a 中
            a = torch.ones_like(x)
            # 同上，生成一个与 x 同形状的全为 1 的张量，结果保存在变量 b 中
            b = torch.ones_like(x)
            # 返回两个全为 1 张量的总和
            return a + b

        # 生成一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)
        # 调用 check 函数，传入 f 函数、张量 t 和数值 1 进行检查
        check(f, t, 1)
# 定义一个测试类 RandomOpTestCase，继承自 TestCase
class RandomOpTestCase(TestCase):

    # 定义测试方法 test_random
    def test_random(self):
        
        # 定义内部函数 f，接受参数 x
        def f(x):
            # 初始化 vals 列表，包含参数 x 的值
            vals = [x]
            # 定义 ops 列表，包含四个函数：torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu
            ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
            
            # 循环执行 100 次
            for _ in range(100):
                # 在 ops 列表中随机选择一个操作函数，并对 vals 列表中随机选择的元素执行该操作，将结果添加到 vals 列表末尾
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            
            # 返回 vals 列表中的最后一个元素作为函数 f 的结果
            return vals[-1]

        # 对函数 f 进行符号跟踪
        fx_g = fx.symbolic_trace(f)
        # 精简跟踪得到的图的死代码
        fx_g.graph.eliminate_dead_code()
        # 重新编译符号跟踪后的函数
        fx_g.recompile()
        
        # 创建一个形状为 (2, 2) 的随机张量 t
        t = torch.randn(2, 2)

        # 循环执行 30 次
        for _ in range(30):
            # 调用 check 函数，检查符号跟踪后的函数 fx_g 在输入张量 t 上的输出
            check(fx_g, t, -1, graph_input=True)

# 如果当前脚本被直接执行，则运行测试
if __name__ == "__main__":
    run_tests()
```