# `.\pytorch\test\inductor\test_perf.py`

```py
# 引入上下文管理工具
import contextlib
# 从 unittest.mock 模块中引入 patch 函数
from unittest.mock import patch

# 引入 functorch 库
import functorch

# 引入 torch 库及其子模块
import torch
# 引入 torch._inductor.config 模块
import torch._inductor.config as config
# 引入 torch.autograd 模块
import torch.autograd
# 从 torch._inductor 中引入 metrics 模块
from torch._inductor import metrics
# 从 torch._inductor.compile_fx 模块中引入 compile_fx 和 compile_fx_inner 函数
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
# 从 torch._inductor.test_case 模块中引入 TestCase 类并重命名为 InductorTestCase
from torch._inductor.test_case import TestCase as InductorTestCase

########################
# Explanation of Tests #
########################
# 这些测试用例主要测试 TorchInductor 的内存访问。
# 它们旨在进行确定性能量化测试。
# 期望的测试用例都测量 Inductor 生成的代码读取/写入的内存字节数。
#
# 如果测试失败是因为数字变小了，可以考虑降低它。
# 另一方面，如果测试失败是因为数字变大了，
# 那意味着您的更改导致在这个测试用例中进行了更多的内存访问。
#
# 这可能仍然可以接受，但请注意您可能会降低该设置的性能。

# 从 torch.testing._internal.triton_utils 模块中导入 HAS_CUDA 和 requires_cuda 函数
from torch.testing._internal.triton_utils import HAS_CUDA, requires_cuda

# 如果 HAS_CUDA 为 True，则从 torch.testing._internal.triton_utils 模块中导入 add_kernel 函数
if HAS_CUDA:
    from torch.testing._internal.triton_utils import add_kernel

# 从 torch.ops.aten 中导入 aten 对象
aten = torch.ops.aten


def compile_but_use_eager(gm, example_inputs):
    def inner_compile(gm, *args, **kwargs):
        # 调用 compile_fx_inner 函数进行内部编译
        compile_fx_inner(gm, *args, **kwargs)
        return gm

    # 调用 compile_fx 函数，进行编译并返回结果
    return compile_fx(gm, example_inputs, inner_compile=inner_compile)


def count_numel(f, *args):
    """
    Assumes all inputs are fp32
    """
    # 重置性能度量指标
    metrics.reset()
    # 使用编译的函数计算
    torch.compile(f, backend=compile_but_use_eager)(*args)
    # 打印节点数
    print(metrics.nodes_num_elem)
    # 返回访问的内存字节数，并将其除以4以获取元素数
    return str(metrics.num_bytes_accessed // 4)


def count_numel_train(f, *args):
    """
    Assumes all inputs are fp32
    """
    # 重置性能度量指标
    metrics.reset()

    # 使用编译的函数计算
    f = torch.compile(f, backend=compile_but_use_eager)
    out = f(*args)
    res = 0
    for o in out:
        res += o.mean()
    res.backward()
    # 打印节点数
    print(metrics.nodes_num_elem)
    # 返回访问的内存字节数，并将其除以4以获取元素数
    return str(metrics.num_bytes_accessed // 4)


# 设备设置为 "cuda"
DEVICE = "cuda"


def T(*size, dtype=torch.float32, device=DEVICE, grad=False):
    # 返回指定大小的随机张量
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


def TI(*size, mx=10, dtype=torch.int32, device=DEVICE):
    # 返回指定大小的随机整数张量
    return torch.randint(0, mx, size, dtype=dtype, device=device)


class TestCase(InductorTestCase):
    # 设备设置为 "cuda"
    device = DEVICE
    pass


class NumBytesMetricTests(TestCase):
    """
    Primarily used for sanity testing that the num_bytes_accessed metrics is correct.
    """
    # 定义测试函数 test_pointwise，用于测试按元素操作的函数
    def test_pointwise(self):
        
        # 定义函数 f，对输入 x 执行余弦函数
        def f(x):
            return x.cos()

        # 准备输入数据 inp，包含一个张量 T(10)
        inp = (T(10),)
        # 断言函数 count_numel 返回的结果与期望值 "20" 相符
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        # 重新定义函数 f，接收两个参数 x 和 y，返回它们的和
        def f(x, y):
            return x + y

        # 更新输入数据 inp，包含两个张量 T(10)
        inp = (T(10), T(10))
        # 断言函数 count_numel 返回的结果与期望值 "30" 相符
        self.assertExpectedInline(count_numel(f, *inp), """30""")

        # 重新定义函数 f，接收一个大小为 (10, 10) 的张量 x 和一个大小为 (10,) 的张量 y，返回它们的和
        def f(x, y):
            return x + y

        # 更新输入数据 inp，包含一个大小为 (10, 10) 的张量 T(10, 10) 和一个大小为 (10,) 的张量 T(10)
        inp = (T(10, 10), T(10))
        # 断言函数 count_numel 返回的结果与期望值 "210" 相符
        self.assertExpectedInline(count_numel(f, *inp), """210""")

        # 重新定义函数 f，接收一个参数 x，返回 x 与自身的和
        def f(x):
            return x + x

        # 更新输入数据 inp，包含一个张量 T(10)
        inp = (T(10),)
        # 断言函数 count_numel 返回的结果与期望值 "20" 相符
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        # 重新定义函数 f，接收一个大小为 (10, 10) 的张量 x，返回 x 与其转置矩阵的和
        def f(x):
            return x + x.t()

        # 更新输入数据 inp，包含一个大小为 (10, 10) 的张量 T(10, 10)
        inp = (T(10, 10),)
        # 断言函数 count_numel 返回的结果与期望值 "200" 相符
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        # 重新定义函数 f，接收三个参数 a, b, c，返回 a 的余弦值与 b 和 c 的正弦值之和
        def f(a, b, c):
            return a.cos(), b.sin() + c.sin()

        # 更新输入数据 inp，包含三个张量 T(10)
        inp = (T(10), T(10), T(10))
        # 断言函数 count_numel 返回的结果与期望值 "50" 相符
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    # 定义测试函数 test_reduction，用于测试张量的降维操作函数
    def test_reduction(self):
        
        # 定义函数 f，对输入 x 按行求和
        def f(x):
            return x.sum(dim=1)

        # 准备输入数据 inp，包含一个大小为 (10, 10) 的张量 T(10, 10)
        inp = (T(10, 10),)
        # 断言函数 count_numel 返回的结果与期望值 "110" 相符
        self.assertExpectedInline(count_numel(f, *inp), """110""")

        # 重新定义函数 f，对输入 x 按列求和
        def f(x):
            return x.sum(dim=0)

        # 更新输入数据 inp，包含一个大小为 (10, 10) 的张量 T(10, 10)
        inp = (T(10, 10),)
        # 断言函数 count_numel 返回的结果与期望值 "110" 相符
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    # 定义测试函数 test_extern，用于测试外部库函数的使用情况
    def test_extern(self):
        
        # 定义函数 f，对输入 x 进行矩阵乘法 x * x
        def f(x):
            return torch.mm(x, x)

        # 准备输入数据 inp，包含一个大小为 (10, 10) 的张量 T(10, 10)
        inp = (T(10, 10),)
        # 断言函数 count_numel 返回的结果与期望值 "200" 相符
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        # 重新定义函数 f，对输入 a, b 进行矩阵乘法 a * b
        def f(a, b):
            return torch.mm(a, b)

        # 更新输入数据 inp，包含两个大小为 (10, 10) 的张量 T(10, 10)
        inp = (T(10, 10), T(10, 10))
        # 断言函数 count_numel 返回的结果与期望值 "300" 相符
        self.assertExpectedInline(count_numel(f, *inp), """300""")

        # 重新定义函数 f，对输入 x 进行余弦函数、矩阵乘法 x * x 和再次余弦函数
        def f(x):
            x = x.cos()
            x = torch.mm(x, x)
            x = x.cos()
            return x

        # 更新输入数据 inp，包含一个大小为 (10, 10) 的张量 T(10, 10)
        inp = (T(10, 10),)
        # 断言函数 count_numel 返回的结果与期望值 "600" 相符
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        # 重新定义函数 f，对输入 x 进行余弦函数得到 a，正弦函数得到 b，然后进行矩阵乘法 a * b
        def f(x):
            a = x.cos()
            b = x.sin()
            x = torch.mm(a, b)
            return x

        # 更新输入数据 inp，包含一个大小为 (10, 10) 的张量 T(10, 10)
        inp = (T(10, 10),)
        # 断言函数 count_numel 返回的结果与期望值 "600" 相符
        self.assertExpectedInline(count_numel(f, *inp), """600""")
    # 定义测试函数 test_cat，用于测试 torch.cat 函数的不同用法
    def test_cat(self):
        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([a.sin(), b.sin()]) 的结果
        def f(a, b):
            return torch.cat([a.sin(), b.sin()])

        # 准备输入数据 inp，包含两个大小为 10 的张量
        inp = (T(10), T(10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 40，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([a, b]) 的结果
        def f(a, b):
            return torch.cat([a, b])

        # 准备输入数据 inp，包含两个大小为 10 的张量
        inp = (T(10), T(10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 40，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([a.cos(), b]) 的结果
        def f(a, b):
            return torch.cat([a.cos(), b])

        # 准备输入数据 inp，包含两个大小为 10 的张量
        inp = (T(10), T(10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 40，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        # 定义函数 f，接受一个参数 a，返回 torch.cat([a.cos(), a.sin()]) 的结果
        def f(a):
            return torch.cat([a.cos(), a.sin()])

        # 准备输入数据 inp，包含一个大小为 10 的张量
        inp = (T(10),)
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 30，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """30""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([torch.mm(a, a), b.sin()]) 的结果
        def f(a, b):
            return torch.cat([torch.mm(a, a), b.sin()])

        # 准备输入数据 inp，包含两个大小为 (10, 10) 的张量
        inp = (T(10, 10), T(10, 10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 400，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        # 定义函数 f，接受三个参数 a、b、c，返回 torch.cat((a + 1, b + 2, c + 3)) + 10 的结果
        def f(a, b, c):
            return torch.cat((a + 1, b + 2, c + 3)) + 10

        # 准备输入数据 inp，包含三个大小为 (10, 10) 的张量
        inp = (T(10, 10), T(10, 10), T(10, 10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 600，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        # 定义函数 f，接受五个参数 a、b、c、d、e，返回 torch.cat((a + 1, b + 2, c + 3, d + 4, e + 5)) + 10 的结果
        def f(a, b, c, d, e):
            return torch.cat((a + 1, b + 2, c + 3, d + 4, e + 5)) + 10

        # 准备输入数据 inp，包含五个大小为 (10, 10) 的张量
        inp = [T(10, 10) for _ in range(5)]
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 1000，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """1000""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([a.sum(dim=0), b.sum(dim=0)]) + 10 的结果
        def f(a, b):
            return torch.cat([a.sum(dim=0), b.sum(dim=0)]) + 10

        # 准备输入数据 inp，包含两个大小为 (10, 10, 10) 的张量
        inp = [T(10, 10, 10), T(10, 10, 10)]
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 2600，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """2600""")

    # 定义测试函数 test_cat_pointwise，用于测试带有 pointwise 操作的 torch.cat 使用情况
    def test_cat_pointwise(self):
        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)]) 的结果
        def f(a, b):
            return torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)])

        # 准备输入数据 inp，包含两个大小为 (10, 10) 的张量
        inp = (T(10, 10), T(10, 10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 400，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)]).cos() 的结果
        def f(a, b):
            return torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)]).cos()

        # 准备输入数据 inp，包含两个大小为 (10, 10) 的张量
        inp = (T(10, 10), T(10, 10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 680，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """680""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([a.cos(), torch.mm(b, b)]) 的结果，然后再对结果执行 cos() 操作
        def f(a, b):
            out = torch.cat([a.cos(), torch.mm(b, b)])
            return out.cos()

        # 准备输入数据 inp，包含两个大小为 (10, 10) 的张量
        inp = (T(10, 10), T(10, 10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 600，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([torch.mm(a, a), torch.mm(b, b)]) 的结果，然后再对结果执行 cos() 操作
        def f(a, b):
            out = torch.cat([torch.mm(a, a), torch.mm(b, b)])
            return out.cos()

        # 准备输入数据 inp，包含两个大小为 (10, 10) 的张量
        inp = (T(10, 10), T(10, 10))
        # 调用 count_numel 函数，检查 f 返回的张量的元素个数是否为 800，并断言结果
        self.assertExpectedInline(count_numel(f, *inp), """800""")

        # 定义函数 f，接受两个参数 a 和 b，返回 torch.cat([a, b]) 的结果，然后再对结果执行 cos() 操作
        def f(a, b):
            out = torch.cat([a, b])
            return out.cos()

        # 准备输入数据 inp，包含两个大小为 (10, 10) 的张量
        inp = (T(10
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    @patch.object(config, "post_grad_fusion_options", {})


    # 设置预梯度融合选项，将多个批量操作合并为一个操作
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    # 设置后梯度融合选项为空字典
    @patch.object(config, "post_grad_fusion_options", {})



    def test_cat_pointwise_many_complex_inputs(self):
        def f(*inputs):
            input = [torch.nn.functional.gelu(val) for val in inputs]
            return torch.cat(input) + 10


    # 测试函数，对多个复杂输入执行 torch.nn.functional.gelu 激活，然后拼接并加上 10
    def test_cat_pointwise_many_complex_inputs(self):
        def f(*inputs):
            input = [torch.nn.functional.gelu(val) for val in inputs]
            return torch.cat(input) + 10



        inp = (T(10, 10) for _ in range(16))
        self.assertExpectedInline(count_numel(f, *inp), """6400""")


        # 定义输入为 16 个大小为 (10, 10) 的 Tensor
        inp = (T(10, 10) for _ in range(16))
        # 断言函数 f 在给定输入上的计算结果为 6400
        self.assertExpectedInline(count_numel(f, *inp), """6400""")



    @patch.object(config, "split_cat_fx_passes", False)
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    @patch.object(config, "post_grad_fusion_options", {})


    # 设置不分割 cat 操作的 FX 通道
    @patch.object(config, "split_cat_fx_passes", False)
    # 设置预梯度融合选项，将多个批量操作合并为一个操作
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    # 设置后梯度融合选项为空字典
    @patch.object(config, "post_grad_fusion_options", {})



    def test_cat_pointwise_many_simple_inputs(self):
        def f(*inputs):
            input = [torch.nn.functional.relu(val) for val in inputs]
            return torch.cat(input) + 10


    # 测试函数，对多个简单输入执行 torch.nn.functional.relu 激活，然后拼接并加上 10
    def test_cat_pointwise_many_simple_inputs(self):
        def f(*inputs):
            input = [torch.nn.functional.relu(val) for val in inputs]
            return torch.cat(input) + 10



        inp = (T(10, 10) for _ in range(16))
        self.assertExpectedInline(count_numel(f, *inp), """9600""")


        # 定义输入为 16 个大小为 (10, 10) 的 Tensor
        inp = (T(10, 10) for _ in range(16))
        # 断言函数 f 在给定输入上的计算结果为 9600
        self.assertExpectedInline(count_numel(f, *inp), """9600""")



    @patch.object(config, "max_pointwise_cat_inputs", 0)


    # 设置点积 cat 输入的最大数量为 0
    @patch.object(config, "max_pointwise_cat_inputs", 0)



    def test_cat_pointwise_config_option(self):
        def f(a, b):
            return torch.cat([a + 1, b + 2]) + 3


    # 测试函数，对两个输入 a 和 b 分别加 1 和 2，然后拼接并加上 3
    def test_cat_pointwise_config_option(self):
        def f(a, b):
            return torch.cat([a + 1, b + 2]) + 3



        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """400""")


        # 定义输入为两个大小为 (10, 10) 的 Tensor
        inp = (T(10, 10), T(10, 10))
        # 断言函数 f 在给定输入上的计算结果为 400
        self.assertExpectedInline(count_numel(f, *inp), """400""")



    def test_index(self):
        def f(a, b):
            return a[b]


    # 测试函数，返回 a 中索引为 b 的元素
    def test_index(self):
        def f(a, b):
            return a[b]



        inp = (T(10), TI(10, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """30""")


        # 定义输入为一个大小为 10 的 Tensor 和一个大小为 (10, max=10) 的整数 Tensor
        inp = (T(10), TI(10, mx=10))
        # 断言函数 f 在给定输入上的计算结果为 30
        self.assertExpectedInline(count_numel(f, *inp), """30""")
# 定义一个名为 FusionTests 的测试类，继承自 TestCase，用于测试核心融合功能
class FusionTests(TestCase):
    """
    Tests that things can be fused into a single kernel
    """

    # 定义测试方法 test_horizontal_reduction_pointwise，测试水平方向的逐点减少操作
    def test_horizontal_reduction_pointwise(self):
        # 定义函数 f，接受一个参数 a
        def f(a):
            # 计算 a 在第一维度上的和
            b = a.sum(dim=1)
            # 计算 a 的余弦值
            c = a.cos()
            # 返回 b 和 c
            return b, c

        # 定义输入 inp，包含一个大小为 (10, 10) 的张量元组
        inp = (T(10, 10),)
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """210""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    # 定义测试方法 test_horizontal_reduction_reduction，测试水平方向的逐渐减少操作
    def test_horizontal_reduction_reduction(self):
        # 定义函数 f，接受一个参数 a
        def f(a):
            # 计算 a 在第一维度上的和
            b = a.sum(dim=1)
            # 计算 a 在第一维度上的最大值
            c = a.amax(dim=1)
            # 返回 b 和 c
            return b, c

        # 定义输入 inp，包含一个大小为 (10, 10) 的张量元组
        inp = (T(10, 10),)
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """120""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    # 定义测试方法 test_horizontal_reduction_pointwise2，测试水平方向的逐点减少操作2
    def test_horizontal_reduction_pointwise2(self):
        # 定义函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 计算 a 在第一维度上的和
            c = a.sum(dim=1)
            # 计算 b 的余弦值
            b = b.cos()
            # 返回 b 加上 c
            return b + c

        # 定义输入 inp，包含一个大小为 (10, 10) 和一个大小为 (10,) 的张量元组
        inp = (T(10, 10), T(10))
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """120""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    # 定义测试方法 test_horizontal_reduction_outer_pointwise，测试外部的水平方向逐点减少操作
    def test_horizontal_reduction_outer_pointwise(self):
        # 定义函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 计算 a 在第零维度上的和
            c = a.sum(dim=0)
            # 计算 b 的余弦值
            b = b.cos()
            # 返回 b 加上 c
            return b + c

        # 定义输入 inp，包含一个大小为 (10, 10) 和一个大小为 (10,) 的张量元组
        inp = (T(10, 10), T(10))
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """120""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    # 定义测试方法 test_horizontal_sum_pw_broadcast，测试广播的水平方向和逐点操作
    def test_horizontal_sum_pw_broadcast(self):
        # 定义函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 计算 a 在第一维度上的和，保持维度
            a = a.sum(dim=1, keepdim=True)
            # 计算 b 的余弦值
            b = b.cos()
            # 返回 a 乘以 b
            return a * b

        # 定义输入 inp，包含一个大小为 (10, 10) 和一个大小为 (10,) 的张量元组
        inp = (T(10, 10), T(10))
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """210""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    # 定义测试方法 test_vertical_sum_pw，测试垂直方向的和逐点操作
    def test_vertical_sum_pw(self):
        # 定义函数 f，接受一个参数 a
        def f(a):
            # 计算 a 的余弦值
            a = a.cos()
            # 计算 a 在第一维度上的和
            a = a.sum(dim=1)
            # 返回 a 的余弦值
            return a.cos()

        # 定义输入 inp，包含一个大小为 (10, 10) 的张量元组
        inp = (T(10, 10),)
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """110""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    # 定义测试方法 test_norm_chain，测试规范链
    def test_norm_chain(self):
        # 定义函数 f，接受一个参数 a
        def f(a):
            # 计算 a 在第一维度上的和，保持维度
            b = a.sum(dim=1, keepdim=True)
            # a 乘以 b
            a = a * b
            # 计算 a 在第一维度上的和，保持维度
            b = a.sum(dim=1, keepdim=True)
            # a 乘以 b
            a = a * b
            # 计算 a 在第一维度上的和，保持维度
            b = a.sum(dim=1, keepdim=True)
            # a 乘以 b
            a = a * b
            # 返回 a
            return a

        # 定义输入 inp，包含一个大小为 (10, 10) 的张量元组
        inp = (T(10, 10),)
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """200""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    # 定义测试方法 test_softmax_inner，测试内部 softmax
    def test_softmax_inner(self):
        # 定义函数 f，接受一个参数 a
        def f(a):
            # 对 a 进行在第一维度上的 softmax 操作
            return torch.softmax(a, dim=1)

        # 定义输入 inp，包含一个大小为 (10, 10) 的张量元组
        inp = (T(10, 10),)
        # 断言函数 count_numel 调用 f 函数后的结果与期望结果 """200""" 相等
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    # 定义测试方法 test_layer_norm，测试层归一化
    def test_layer_norm(self):
        # TODO: Suboptimal! We shouldn't need to save normalization stats.
        # 创建一个大小为 10 的层归一化模块，使用 self.device 指定的设备
        mod = torch.nn.LayerNorm(10, device=self.device)

        # 定义函数 f，接受一个参数 x
        def f(x):
            # 对 x 进行层归一化操作
            return mod(x)

        # 定义输入 inp，包含一个大小为 (10, 10) 的张量元组
        inp = (T(10, 10),)
        # 在没有梯度的情况下，断言函数 count_numel 调用 f 函数后的结果与期望结果 """220""" 相等
        with torch.no_grad():
            self.assertExpectedInline(count_numel(f, *inp), """220""")

    # 定义测试方法 test_double_softmax，测试双重 softmax
    def test_double_softmax(self):
        # 定义函数 f，接受一个参数 x
        def f(x):
            # 对 x 进行在第一维度
    def test_neighbor(self):
        # 定义一个内部函数f，计算两个张量a和b的元素差的平方，然后按最后一个维度求和，并在第一维度求最大值
        def f(a, b):
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        # 输入数据为两个张量，形状分别为(10, 1, 4)和(1, 10, 4)，调用count_numel函数验证函数f的输出是否符合预期结果"90"
        inp = (T(10, 1, 4), T(1, 10, 4))
        self.assertExpectedInline(count_numel(f, *inp), """90""")

    def test_factory_reduction(self):
        # 定义一个内部函数f，创建两个张量a和b，形状分别为(10,)和(10, 10)，返回这两个张量相加后按最后一个维度求和的结果
        def f():
            a = torch.ones(10, device=self.device)
            b = torch.ones(10, 10, device=self.device)
            return (a + b).sum(dim=-1)

        # 输入为空元组，调用count_numel函数验证函数f的输出是否符合预期结果"10"
        inp = ()
        self.assertExpectedInline(count_numel(f, *inp), """10""")

    def test_index_pointwise(self):
        # 定义一个内部函数f，接受两个张量a和b，返回张量a按照索引张量b的值取cos函数的结果
        def f(a, b):
            return a[b].cos()

        # 输入数据为两个张量，形状分别为(10, 10)和(20,)，调用count_numel函数验证函数f的输出是否符合预期结果"320"
        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """320""")

    def test_index_reduction(self):
        # 定义一个内部函数f，接受两个张量a和b，返回张量a按照索引张量b的值取cos函数的结果，并在最后一个维度上求和
        def f(a, b):
            return a[b].cos().sum(dim=1)

        # 输入数据为两个张量，形状分别为(10, 10)和(20,)，调用count_numel函数验证函数f的输出是否符合预期结果"140"
        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """140""")

    def test_mutation_fusion(self):
        # 定义一个内部函数f，接受三个张量a、b、c，依次进行张量操作，并在最后调用copy_()方法实现原地复制
        def f(a, b, c):
            a0 = a.add(c)
            b0 = b.add(a0)
            b.copy_(b0)
            a.copy_(a0)

        # 输入数据为三个形状为(10, 10)的张量，调用count_numel函数验证函数f的输出是否符合预期结果"500"
        inp = (T(10, 10), T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """500""")

    def test_reduction_pointwise_multi_level_reduction(self):
        # 设置隐藏层大小为4096，创建一个在GPU上的LayerNorm对象
        hidden_size = 4096
        layer_norm = torch.nn.LayerNorm(hidden_size).cuda().float()

        # 定义一个包含两个内核的函数f，使用torch.inference_mode()进行推断模式设置
        @torch.inference_mode()
        def f(x, scale, amax_keep_dim):
            # 对输入张量x进行LayerNorm归一化，转换为浮点类型
            x = layer_norm(x.to(dtype=torch.float))
            # 计算x的绝对值的最大值，并根据参数amax_keep_dim是否为True保持维度
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            # 将x按元素乘以scale
            x_scaled = x * scale
            # 对x_scaled应用sigmoid函数
            y = torch.nn.functional.sigmoid(x_scaled)
            # 返回元组(y, amax)
            return (y, amax)

        # 输入数据为两个张量，形状分别为(4, 2048, hidden_size)和(1,)，计算预期的总元素数量并验证
        inp = (T(4, 2048, hidden_size, dtype=torch.float), T(1, dtype=torch.float))
        expected_numel = (
            1 + hidden_size * 2 + 4 * 2048 * hidden_size * 2 + 4 * 2048 * 2 + 1
        )
        self.assertExpectedInline(count_numel(f, *inp, True), str(expected_numel))
        self.assertExpectedInline(count_numel(f, *inp, False), str(expected_numel))
    # 定义一个测试方法，用于测试多层级减少的点对点操作
    def test_pointwise_multi_level_reduction(self):
        # TODO: 可以通过让第一个点对点内核利用块大小来优化这段代码
        # 的作用是定义隐藏大小为4096
        hidden_size = 4096

        # 定义一个内部函数f，接受x，scale和amax_keep_dim作为参数
        def f(x, scale, amax_keep_dim):
            # 将x乘以1.1
            x = x * 1.1
            # 计算x的绝对值的最大值，如果amax_keep_dim为True，则保持维度
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            # 将x按比例scale进行缩放
            x_scaled = x * scale
            # 对x_scaled进行sigmoid操作
            y = torch.nn.functional.sigmoid(x_scaled)
            # 返回y和amax
            return (y, amax)

        # 构造输入元组inp，包含T(4, 2048, hidden_size, dtype=torch.float)和T(1, dtype=torch.float)
        inp = (T(4, 2048, hidden_size, dtype=torch.float), T(1, dtype=torch.float))

        # 使用torch.compile编译函数f
        compiled_f = torch.compile(f)
        # 调用编译后的函数compiled_f，传入inp和True作为参数
        compiled_f(*inp, True)

        # 3个内核：
        # 内核1：(输入 = X, scale, 输出 = pointwise(X))
        # 内核2：(输入 = X, 输出 = 第一级amax)
        # 内核3：(输入 = 第一级amax, 输出 = 最终amax)
        # scale (1) + X (4*2048*hidden_size) * 3 + amax (num_splits * 2 + 1)
        # num_splits取决于SM架构。
        # 预期的元素数量为：1 + 4 * 2048 * hidden_size * 3 + 1
        expected_numel = 1 + 4 * 2048 * hidden_size * 3 + 1
        # 使用count_numel函数计算带有amax_keep_dim=True的实际元素数量
        actual_numel_amax_keep_dim = count_numel(f, *inp, True)
        # 使用count_numel函数计算带有amax_keep_dim=False的实际元素数量
        actual_numel_amax_no_keep_dim = count_numel(f, *inp, False)
        # 断言实际的带有amax_keep_dim=True的元素数量等于实际的带有amax_keep_dim=False的元素数量
        self.assertEqual(actual_numel_amax_keep_dim, actual_numel_amax_no_keep_dim)
        # 断言实际的带有amax_keep_dim=True的元素数量大于或等于预期的元素数量
        self.assertGreaterAlmostEqual(actual_numel_amax_keep_dim, str(expected_numel))
class SchedulerFusionTests(TestCase):
    """
    Testing the fusion group creation heuristic (i.e. cases where we can't fuse
    everything into a single kernel)
    Disables inductor rematerialization for easier reasoning of tests.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._stack = contextlib.ExitStack()  # 创建一个上下文管理器栈
        cls._stack.enter_context(patch.object(config, "realize_opcount_threshold", 0))  # 将配置项的实现操作计数阈值设置为0

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()  # 关闭上下文管理器栈
        super().tearDownClass()

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice1(self):
        # Doesn't matter where we break fusion group here
        def f(a):
            c = a.cos()  # 计算输入张量的余弦值
            d = torch.mm(c, c)  # 执行矩阵乘法
            e = c.cos()  # 计算余弦值
            return d + e  # 返回矩阵乘积和余弦值的和

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """700""")  # 断言期望的张量元素数量为700

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice2(self):
        # We should materialize e (it's smaller!)
        # [c, e]: 210, [f]: 210, [d]: 200
        def f(a):
            c = a.cos()  # 计算输入张量的余弦值
            d = torch.mm(c, c)  # 执行矩阵乘法
            e = c.sum(dim=1)  # 按指定维度求和
            f = d + e  # 返回矩阵乘积和求和结果的和
            return f  # 返回结果

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """620""")  # 断言期望的张量元素数量为620

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice3(self):
        # We should materialize e.
        # [c, e]: 300, [f]: 300, [d]: 200
        def f(a):
            c = a.cos()  # 计算输入张量的余弦值
            d = torch.mm(c, c)  # 执行矩阵乘法
            e = c + a  # 张量间的加法操作
            f = d + e  # 返回矩阵乘积和加法结果的和
            return f, e  # 返回结果和中间张量 e

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """800""")  # 断言期望的张量元素数量为800

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice4_cpu(self):
        # Fuse nodes with same number of elements and compatible orginal var ranges
        # [buf0: {d0: 60, d1: 11}, buf1: {d0: 660}] -> buf0_buf1
        def f(x, w):
            o1 = x * w  # 张量乘法操作
            output = o1 + 1.0  # 张量加法操作
            return output  # 返回结果

        inp = (T(2, 3, 10, 11, device="cpu"), T(11, device="cpu"))
        self.assertExpectedInline(count_numel(f, *inp), """1331""")  # 断言期望的张量元素数量为1331

        # [buf0_buf1: {d0: 60, d1: 11}, buf2: {d0: 660}] -> buf0_buf1_buf2
        def f(x, w1, w2):
            o1 = x * w1  # 张量乘法操作
            o2 = x * w2  # 张量乘法操作
            output = o1 + o2  # 张量加法操作
            return output  # 返回结果

        inp = (T(2, 3, 10, 11, device="cpu"), T(11, device="cpu"), T(11, device="cpu"))
        self.assertExpectedInline(count_numel(f, *inp), """1342""")  # 断言期望的张量元素数量为1342


class TilingTests(TestCase):
    def test_tiling_simple(self):
        def f(a, b):
            return a + b.t()  # 张量加法和转置操作

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")  # 断言期望的张量元素数量为300

        def f(a, b):
            return a.t() + b  # 转置和张量加法操作

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")  # 断言期望的张量元素数量为300
    def test_tiling_three(self):
        # 定义一个嵌套函数 f，接受三个参数 a, b, c，并返回它们的加权和
        def f(a, b, c):
            return a + b.permute(1, 2, 0) + c.permute(2, 0, 1)

        # 创建输入元组 inp，包含三个相同形状的 Tensors (10, 10, 10)
        inp = (T(10, 10, 10), T(10, 10, 10), T(10, 10, 10))
        # 调用 count_numel 函数计算 f 返回值的元素个数，并与预期结果 "4000" 断言比较
        self.assertExpectedInline(count_numel(f, *inp), """4000""")
class MinCutPartitioningTests(TestCase):
    # MinCutPartitioningTests 类，用于测试切分和分区功能

    def test_partitioning_full_remat(self):
        # 测试完全重新材料化的分区功能
        def f(x):
            return x.cos().cos().cos()

        # 构造输入数据
        inp = (T(10, grad=True),)
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """50""")

    def test_partitioning_partial_remat(self):
        # 测试部分重新材料化的分区功能
        def f(a, b, c, d):
            x = a + b + c + d
            return x.cos().cos()

        # 构造输入数据
        inp = (T(10, grad=True), T(10, grad=True), T(10, grad=True), T(10, grad=True))
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """90""")

    def test_partitioning_dtype(self):
        # 测试数据类型对分区功能的影响
        def f(x):
            return (x < 0) * x

        # 构造输入数据
        inp = (T(100, grad=True),)
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """450""")

    @patch.object(functorch.compile.config, "max_dist_from_bw", 1000)
    def test_partitioning_unremat_bw(self):
        # 测试未重新材料化时带宽限制对分区功能的影响
        def f(x):
            return torch.mm(x, x.new_ones(x.shape)).tanh().tanh()

        # 构造输入数据
        inp = (T(10, 10, grad=True),)
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """1300""")

    @patch.object(config, "pattern_matcher", False)
    def test_partitioning_unremat_bw2(self):
        # 测试未重新材料化时带宽限制2对分区功能的影响
        def f(a):
            a = torch.mm(a, a)
            a = a + 1
            b = a + 2
            c = torch.mm(a, b)
            return c

        # 构造输入数据
        inp = (T(10, 10, grad=True),)
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """2600""")

    def test_partitioning_keops(self):
        # 测试使用KeOps库的分区功能
        def f(a, b):
            return (a * b).cos().sum(dim=1)

        # 构造输入数据
        inp = (T(20, 1, grad=True), T(1, 20, grad=True))
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """220""")

    def test_partitioning_cat(self):
        # 测试使用 torch.cat 的分区功能
        def f(a, b):
            a = torch.tanh(a)
            return torch.cat([a, b])

        # 构造输入数据
        inp = (T(10, grad=True), T(10, grad=True))
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """70""")

    def test_partitioning_with_view(self):
        # 测试包含视图操作的分区功能
        class Foo(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x.sin()
                x = x.cos()
                x = x.view(10, 10)
                ctx.save_for_backward(x, y)
                x = x.cos()
                return x

            @staticmethod
            def backward(ctx, gradOut):
                x, y = ctx.saved_tensors
                return torch.mm(gradOut, x).view(100) * y

        def f(a):
            return Foo.apply(a)

        # 构造输入数据
        inp = (T(100, grad=True),)
        # 断言期望结果与实际结果相符
        self.assertExpectedInline(count_numel_train(f, *inp), """900""")

    @patch.object(config, "pattern_matcher", False)
    # 定义一个函数 f(x)，该函数对输入的张量进行多次操作
    def test_partitioning_long_chain_add(self):
        # 内部函数 f(x) 的定义开始
        def f(x):
            # 将输入值保存到 orig 变量中
            orig = x
            # 循环两次，对输入值进行复杂的数学操作
            for _ in range(2):
                # 对输入值进行平方操作
                x = x * x
                # 使用 torch.mm 函数进行矩阵乘法操作
                x = torch.mm(x, x)
                # 对结果乘以 2
                x = x * 2
                # 将原始输入值 orig 加上变换后的值 x
                x = orig + x
                # 更新 orig 变量为新的 x 值
                orig = x
            # 返回最终结果 x
            return x

        # 准备输入数据 inp，是一个元组，包含一个形状为 (10, 10) 的张量，并启用梯度跟踪
        inp = (T(10, 10, grad=True),)
        # 断言调用 count_numel_train 函数后返回的结果与预期值 "3900" 相符
        self.assertExpectedInline(count_numel_train(f, *inp), """3900""")
# 定义一个函数 unfusible，用于进行 noop 测试。在这里，我们希望强制 inductor 切换到急切模式，因此我们使用一个没有分解或降低的 aten 运算符：
def unfusible(x):
    return aten._lazy_clone(x)

# 定义 NoopTests 类，用于测试 unfusible 函数的行为
class NoopTests(TestCase):

    # 测试函数 test_noop_clones
    def test_noop_clones(self):
        # 定义函数 f(a)，其中 a 是输入
        def f(a):
            # 克隆输入 a，得到 b
            b = a.clone()
            # 对 b 应用 unfusible 函数
            b = unfusible(b)
            return b

        # 创建输入 inp
        inp = T(10)
        # 断言函数 count_numel(f, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f, inp), """20""")

        # 定义函数 f(a)，其中 a 是输入
        def f(a):
            # 克隆输入 a，得到 b
            b = a.clone()
            # 对 b 应用 unfusible 函数，得到 c
            c = unfusible(b)
            return b, c

        # 断言函数 count_numel(f, inp) 的行内预期输出为 "40"
        self.assertExpectedInline(count_numel(f, inp), """40""")

    # 测试函数 test_noop_slice_scatter
    def test_noop_slice_scatter(self):
        # 定义函数 f(a)，其中 a 是输入
        def f(a):
            # 使用 aten.slice_scatter 对输入 a 进行处理，得到 b
            b = aten.slice_scatter(a, a)
            # 对 b 应用 unfusible 函数，得到 c
            c = unfusible(b)
            return c

        # 创建输入 inp
        inp = T(10)
        # 断言函数 count_numel(f, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f, inp), """20""")

    # 测试函数 test_noop_dtype_conversion
    def test_noop_dtype_conversion(self):
        # 定义函数 f(a)，其中 a 是输入
        def f(a):
            # 使用 torch.ops.prims.convert_element_type 将输入 a 转换为 torch.float32 类型，得到 b
            b = torch.ops.prims.convert_element_type(a, torch.float32)
            # 对 b 应用 unfusible 函数，得到 c
            c = unfusible(b)
            return c

        # 创建输入 inp
        inp = T(10)
        # 断言函数 count_numel(f, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f, inp), """20""")

    # 测试函数 test_noop_device_conversion
    def test_noop_device_conversion(self):
        # 定义函数 f(a)，其中 a 是输入
        def f(a):
            # 使用 torch.ops.prims.device_put 将输入 a 放置在 "cuda" 设备上，得到 b
            b = torch.ops.prims.device_put(a, "cuda")
            # 对 b 应用 unfusible 函数，得到 c
            c = unfusible(b)
            return c

        # 创建输入 inp
        inp = T(10)
        # 断言函数 count_numel(f, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f, inp), """20""")

    # 测试函数 test_noop_int_ops
    def test_noop_int_ops(self):
        # 定义函数 f1(a)，其中 a 是输入
        def f1(a):
            # 对输入 a 应用 torch.ceil 函数，得到 b
            b = torch.ceil(a)
            # 对 b 应用 unfusible 函数，得到 c
            c = unfusible(b)
            return c

        # 定义函数 f2(a)，其中 a 是输入
        def f2(a):
            # 对输入 a 应用 torch.floor 函数，得到 d
            d = torch.floor(a)
            # 对 d 应用 unfusible 函数，得到 e
            e = unfusible(d)
            return e

        # 定义函数 f3(a)，其中 a 是输入
        def f3(a):
            # 对输入 a 应用 torch.round 函数，得到 f
            f = torch.round(a)
            # 对 f 应用 unfusible 函数，得到 g
            g = unfusible(f)
            return g

        # 定义函数 f4(a)，其中 a 是输入
        def f4(a):
            # 对输入 a 应用 torch.pow(a, 1) 函数，得到 f
            f = torch.pow(a, 1)
            # 对 f 应用 unfusible 函数，得到 g
            g = unfusible(f)
            return g

        # 创建输入 inp
        inp = TI(10)
        # 断言函数 count_numel(f1, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f1, inp), """20""")
        # 断言函数 count_numel(f2, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f2, inp), """20""")
        # 断言函数 count_numel(f3, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f3, inp), """20""")
        # 断言函数 count_numel(f4, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f4, inp), """20""")

    # 测试函数 test_noop_cat
    def test_noop_cat(self):
        # 定义函数 f1(a)，其中 a 是输入
        def f1(a):
            # 对输入 a 进行 torch.cat([a]) 操作，得到 b，然后应用 unfusible 函数
            return unfusible(torch.cat([a]))

        # 创建输入 inp
        inp = T(10)
        # 断言函数 count_numel(f1, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f1, inp), """20""")

        # 定义函数 f2(a)，其中 a 是输入
        def f2(a):
            # 对输入 a 进行 torch.cat([a]) 操作，得到 b
            b = torch.cat([a])
            # 对 b 再次进行 torch.cat([b]) 操作，得到 c
            c = torch.cat([b])
            return c

        # 断言函数 count_numel(f2, inp) 的行内预期输出为 "20"
        self.assertExpectedInline(count_numel(f2, inp), """20""")
    def test_inplace_scatter(self):
        # 定义函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 对 a 求余弦并覆盖原始值
            a = a.cos()
            # 在索引 b 处设置值为 1
            a[b] = 1
            return a

        # 定义输入 inp 为两个张量 T(10) 和 TI(2, mx=5)
        inp = (T(10), TI(2, mx=5))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "26" 相符
        self.assertExpectedInline(count_numel(f, *inp), """26""")

        # 定义函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 使用 aten.index_put 在索引 (b,) 处插入值为 1.0 的张量，并返回原始张量的副本
            out = aten.index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)

        # 重新定义输入 inp 为两个张量 T(10) 和 TI(2, mx=5)
        inp = (T(10), TI(2, mx=5))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "6" 相符
        self.assertExpectedInline(count_numel(f, *inp), """6""")

        # 定义函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 使用 aten._unsafe_index_put 在索引 (b,) 处插入值为 1.0 的张量，并返回原始张量的副本
            out = aten._unsafe_index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)

        # 重新定义输入 inp 为两个张量 T(10) 和 TI(2, mx=5)
        inp = (T(10), TI(2, mx=5))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "6" 相符
        self.assertExpectedInline(count_numel(f, *inp), """6""")

    def test_inplace_scatter_noop_view(self):
        # 定义函数 f，接受两个参数 a 和 b
        def f(a, b):
            # 在 a 的所有行中的索引 b 处设置值为 1
            a[:, b] = 1
            return a

        # 定义输入 inp 为两个张量 T(10, 10) 和 TI(2, mx=5)
        inp = (T(10, 10), TI(2, mx=5))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "42" 相符
        self.assertExpectedInline(count_numel(f, *inp), """42""")

    @requires_cuda
    def test_inplace_triton_kernel_v1(self):
        # 定义函数 f，接受两个参数 x 和 y，都是 torch.Tensor 类型
        def f(x: torch.Tensor, y: torch.Tensor):
            # 创建一个与 x 相同形状的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 张量中元素的总数
            n_elements = output.numel()
            # 定义一个由元素总数组成的 grid
            grid = (n_elements,)
            # 调用 CUDA 内核函数 add_kernel 对 x, y, output 进行操作
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        # 定义输入 inp 为两个张量 T(10) 和 T(10)
        inp = (T(10), T(10))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "40" 相符
        self.assertExpectedInline(count_numel(f, *inp), """40""")

    @requires_cuda
    def test_inplace_triton_kernel_v2(self):
        # 定义函数 f，接受两个参数 x 和 y，都是 torch.Tensor 类型
        def f(x: torch.Tensor, y: torch.Tensor):
            # 创建一个与 x 相同形状的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 张量中元素的总数
            n_elements = output.numel()
            # 定义一个由元素总数组成的 grid
            grid = (n_elements,)
            # 将 x 加上 1 并存储在 tmp 中
            tmp = torch.add(x, 1)
            # 调用 CUDA 内核函数 add_kernel 对 x, y, output 进行操作
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output, tmp

        # 定义输入 inp 为两个张量 T(10) 和 T(10)
        inp = (T(10), T(10))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "60" 相符
        self.assertExpectedInline(count_numel(f, *inp), """60""")

    @requires_cuda
    def test_inplace_triton_kernel_v3(self):
        # 定义函数 f，接受两个参数 x 和 y，都是 torch.Tensor 类型
        def f(x: torch.Tensor, y: torch.Tensor):
            # 创建一个与 x 相同形状的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 张量中元素的总数
            n_elements = output.numel()
            # 定义一个由元素总数组成的 grid
            grid = (n_elements,)
            # 调用 CUDA 内核函数 add_kernel 对 x, y, output 进行操作
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            # 在原地对 x 中的所有元素都加上 1
            x.add_(1)
            return output

        # 定义输入 inp 为两个张量 T(10) 和 T(10)
        inp = (T(10), T(10))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "60" 相符
        self.assertExpectedInline(count_numel(f, *inp), """60""")

    @requires_cuda
    def test_inplace_triton_kernel_v4(self):
        # 定义函数 f，接受两个参数 x 和 y，都是 torch.Tensor 类型
        def f(x: torch.Tensor, y: torch.Tensor):
            # 将 x 视图重塑为一维张量 x_view
            x_view = x.view(-1)
            # 创建一个与 x 相同形状的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 张量中元素的总数
            n_elements = output.numel()
            # 定义一个由元素总数组成的 grid
            grid = (n_elements,)
            # 调用 CUDA 内核函数 add_kernel 对 x, y, output 进行操作
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            # 对 x_view 中的所有元素乘以 2 并存储在 output2 中
            output2 = x_view.mul(2)
            return output, output2

        # 定义输入 inp 为两个张量 T(10) 和 T(10)
        inp = (T(10), T(10))
        # 断言调用 count_numel 函数对 f(*inp) 返回的值与期望结果 "60" 相符
        self.assertExpectedInline(count_numel(f, *inp), """60""")
    # 定义一个名为 test_inplace_triton_kernel_v5 的测试方法
    def test_inplace_triton_kernel_v5(self):
        # 定义一个函数 f，接受两个类型为 torch.Tensor 的参数 x 和 y
        def f(x: torch.Tensor, y: torch.Tensor):
            # 将 x 展平为一维张量
            x_view = x.view(-1)
            # 创建一个与 x 相同大小的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 中元素的总数
            n_elements = output.numel()
            # 定义一个包含 n_elements 的元组 grid
            grid = (n_elements,)
            # 调用名为 add_kernel 的 CUDA kernel 函数，对 x、y、output 进行计算
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            # 将 x_view 中的元素乘以 2
            x_view.mul_(2)
            # 返回计算结果 output
            return output

        # 输入参数 inp 是两个大小为 10 的张量组成的元组
        inp = (T(10), T(10))
        # 使用 assertExpectedInline 方法验证 f 函数计算结果与预期结果 "60" 是否一致

    # 标记需要在 CUDA 环境下执行的测试方法
    @requires_cuda
    # 定义一个名为 test_inplace_triton_kernel_v6 的测试方法
    def test_inplace_triton_kernel_v6(self):
        # 定义一个函数 f，接受两个类型为 torch.Tensor 的参数 x 和 y
        def f(x: torch.Tensor, y: torch.Tensor):
            # 创建一个与 x 相同大小的全零张量 output
            output = torch.zeros_like(x)
            # 计算 output 中元素的总数
            n_elements = output.numel()
            # 定义一个包含 n_elements 的元组 grid
            grid = (n_elements,)
            # 调用名为 add_kernel 的 CUDA kernel 函数，对 x、y、output 进行计算
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            # 返回计算结果 output
            return output

        # 创建一个大小为 10 的张量 t
        t = T(10)
        # 将 t 展平为一维张量，并与 t 组成的元组作为输入参数 inp
        inp = (t, t.view(-1))
        # 使用 assertExpectedInline 方法验证 f 函数计算结果与预期结果 "40" 是否一致

    # 定义一个名为 test_inplace_randperm_scatter 的测试方法
    def test_inplace_randperm_scatter(self):
        # 定义一个函数 scaled_index_add，接受三个参数 x、y 和 scale_y
        def scaled_index_add(x, y, scale_y):
            # 在 x 的第一个维度上生成一个随机排列的索引 index
            index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
            # 将 y 乘以 scale_y，然后按照 index 所指示的位置将结果加到 x 上
            out = x.index_add_(dim=0, source=y * scale_y, index=index)
            # 返回计算结果 out
            return out

        # 输入参数 inp 包含三个张量：大小为 (10, 10) 的 x，大小为 (5, 10) 的 y，以及大小为 10 的 scale_y
        inp = (T(10, 10), T(5, 10), T(10))
        # 使用 assertExpectedInline 方法验证 scaled_index_add 函数计算结果与预期结果 "240" 是否一致
# Test cases where we don't do the right thing yet.
class WouldBeNiceIfItWorked:
    # 定义一个测试方法，计算输入张量沿着第一个维度的和及其余弦值
    def test_horizontal(self):
        def f(a):
            # 计算张量沿着第一个维度的和
            b = a.sum(dim=0)
            # 计算张量的余弦值
            c = a.cos()
            return b, c

        # 定义输入为一个形状为(10, 10)的张量元组
        inp = (T(10, 10),)
        # 断言调用函数 count_numel 计算 f 函数的输出元素个数，并期望结果为 "210"
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    # TODO: We aren't fusing outer dim softmaxes
    # 测试外部维度 softmax 合并策略
    def test_softmax_outer(self):
        def f(a):
            # 对输入张量在第一个维度上应用 softmax
            return torch.softmax(a, dim=0)

        # 定义输入为一个形状为(10, 10)的张量元组
        inp = (T(10, 10),)
        # 断言调用函数 count_numel 计算 f 函数的输出元素个数，并期望结果为 "200"
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    # TODO: The greedy fusion strategy results in suboptimal grouping
    # 测试贪婪融合策略下的优化组合
    @patch.object(config, "realize_opcount_threshold", 0)
    def test_fusion_choice4(self):
        def f(a, b, b2):
            # 计算两个张量的和、乘积及其他组合
            c = a + b
            d = torch.mm(c, c)
            e = c + b + b2
            f = d + e + b2
            return f, e

        # 定义输入为三个张量元组，分别为形状为(10, 10)、(10, 10)、(10, 10)的张量
        inp = (T(10, 10), T(10, 10, dtype=torch.float16), T(10, 10))
        # 断言调用函数 count_numel 计算 f 函数的输出元素个数，并期望结果为 "1000"
        self.assertExpectedInline(count_numel(f, *inp), """1000""")

    # TODO: We materialize the intermediate if we don't unroll the reduction
    # 测试邻域操作中的中间结果展开与非展开的影响
    def test_neighbor(self):
        def f(a, b):
            # 计算两个张量之差的平方，沿着最后一个维度求和，然后沿着第一个维度取最大值
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        # 定义输入为两个张量元组，分别为形状为(10, 1, 8)、(1, 10, 8)的张量
        inp = (T(10, 1, 8), T(1, 10, 8))
        # 断言调用函数 count_numel 计算 f 函数的输出元素个数，并期望结果为 "170"
        self.assertExpectedInline(count_numel(f, *inp), """170""")


if __name__ == "__main__":
    # 导入测试框架中的 run_tests 方法
    from torch._inductor.test_case import run_tests

    # 如果存在 CUDA 支持，则运行需要文件锁的测试
    if HAS_CUDA:
        run_tests(needs="filelock")
```