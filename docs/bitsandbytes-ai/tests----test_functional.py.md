# `bitsandbytes\tests\test_functional.py`

```py
# 导入所需的模块
from itertools import product
import math
import random
import time

import einops
import numpy as np
import pytest
from scipy.stats import norm
import torch

# 导入自定义模块和函数
import bitsandbytes as bnb
from bitsandbytes import functional as F
from tests.helpers import (
    BOOLEAN_TUPLES,
    TRUE_FALSE,
    describe_dtype,
    get_test_dims,
    id_formatter,
)

# 设置打印选项
torch.set_printoptions(
    precision=5, sci_mode=False, linewidth=120, edgeitems=20, threshold=10000
)
# 定义常量 k
k = 20

# 自定义函数，用于检查两个张量是否近似相等
def assert_all_approx_close(a, b, rtol=1e-3, atol=1e-3, count=0, throw=True):
    idx = torch.isclose(a, b, rtol, atol)
    sumval = (idx == 0).sum().item()
    if sumval > count:
        if throw:
            print(f"Too many values not close: assert {sumval} < {count}")
            torch.testing.assert_close(a, b, rtol, atol)

    return sumval

# 定义一个全连接神经网络类
class FFN(torch.nn.Module):
    def __init__(self, input_features, hidden_size, bias=True):
        super().__init__()
        # 定义两个全连接层
        self.fc1 = torch.nn.Linear(input_features, hidden_size, bias=bias)
        self.fc2 = torch.nn.Linear(hidden_size, input_features, bias=bias)

        # 使用 Xavier 初始化权重
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)

    # 前向传播函数
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义一个计时器类
class Timer:
    def __init__(self):
        self.starts = {}
        self.ends = {}
        self.agg = {}

    # 记录时间开始
    def tick(self, name="default"):
        if name not in self.starts:
            self.starts[name] = torch.cuda.Event(enable_timing=True)
            self.ends[name] = torch.cuda.Event(enable_timing=True)
            self.starts[name].record()
        else:
            ms = self.tock(name, evict=True, print_ms=False)
    # 记录时间戳，用于计算时间间隔
    def tock(self, name="default", evict=True, print_ms=True):
        # 如果名称在结束时间字典中
        if name in self.ends:
            # 记录结束时间
            self.ends[name].record()
            # 同步 CUDA 设备
            torch.cuda.synchronize()
            # 计算时间间隔
            ms = self.starts[name].elapsed_time(self.ends[name])
            # 如果名称不在聚合字典中
            if name not in self.agg:
                self.agg[name] = 0.0
            # 累加时间间隔
            self.agg[name] += ms
            # 如果需要清除数据
            if evict:
                # 移除开始时间和结束时间
                self.starts.pop(name)
                self.ends.pop(name)

        # 如果需要打印时间间隔并且名称在聚合字典中
        if print_ms and name in self.agg:
            # 打印名称和时间间隔
            print(f"{name} took: {self.agg[name] / 1000.0:.5f}s")

        # 返回聚合时间间隔
        return self.agg[name]

    # 重置所有数据
    def reset(self):
        # 清空开始时间字典
        self.starts = {}
        # 清空结束时间字典
        self.ends = {}
        # 清空聚合时间间隔字典
        self.agg = {}
        # 打印重置信息
        print("Resetting benchmark data")
# 设置函数，暂时不做任何操作
def setup():
    pass

# 清理函数，暂时不做任何操作
def teardown():
    pass

# 使用参数化装饰器，测试估计分位数函数
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16], ids=["float", "half"]
)
def test_estimate_quantiles(dtype):
    # 在 CUDA 设备上生成随机数据 A
    A = torch.rand(1024, 1024, device="cuda")
    # 将数据类型转换为指定类型
    A = A.to(dtype)
    # 调用估计分位数函数，得到结果 code
    code = F.estimate_quantiles(A)

    # 生成一组分位数值
    percs = torch.linspace(1 / 512, 511 / 512, 256, device=A.device)
    # 使用测试函数检查结果是否接近期望值
    torch.testing.assert_close(percs, code, atol=1e-3, rtol=1e-2)

    # 重新生成随机数据 A
    A = torch.randn(1024, 1024, device="cuda")
    # 将数据类型转换为指定类型
    A = A.to(dtype)
    # 调用估计分位数函数，得到结果 code
    code = F.estimate_quantiles(A)

    # 使用 torch.quantile 计算真实分位数
    quantiles = torch.quantile(A.float(), percs)
    # 计算估计值与真实值的差异
    diff = torch.abs(code - quantiles)
    # 断言差异小于指定阈值
    assert (diff > 5e-02).sum().item() == 0

# 测试分位数量化函数
def test_quantile_quantization():
    for i in range(100):
        # 生成随机数据 A1
        A1 = torch.randn(1024, 1024, device="cuda")
        # 调用估计分位数函数，得到结果 code
        code = F.estimate_quantiles(A1)
        # 使用分位数进行量化
        C = F.quantize_no_absmax(A1, code)
        # 使用分位数进行反量化
        A2 = F.dequantize_no_absmax(C, code)
        # 计算 A1 和 A2 之间的平均绝对差异
        diff = torch.abs(A1 - A2).mean().item()
        # 断言差异小于指定阈值
        assert diff < 0.0075

        # 生成随机数据 A1
        A1 = torch.rand(1024, 1024, device="cuda")
        # 调用估计分位数函数，得到结果 code
        code = F.estimate_quantiles(A1)
        # 使用分位数进行量化
        C = F.quantize_no_absmax(A1, code)
        # 使用分位数进行反量化
        A2 = F.dequantize_no_absmax(C, code)
        # 计算 A1 和 A2 之间的平均绝对差异
        diff = torch.abs(A1 - A2).mean().item()
        # 使用测试函数检查结果是否接近期望值
        torch.testing.assert_close(A1, A2, atol=5e-3, rtol=0)
        # 断言差异小于指定阈值
        assert diff < 0.001

# 测试动态量化函数
def test_dynamic_quantization():
    diffs = []
    reldiffs = []
    for i in range(100):
        # 生成随机数据 A1
        A1 = torch.randn(1024, 1024, device="cuda")
        # 调用动态量化函数，得到量化结果 C 和缩放因子 S
        C, S = F.quantize(A1)
        # 使用反量化函数还原数据
        A2 = F.dequantize(C, S)
        # 计算 A1 和 A2 之间的绝对差异
        diff = torch.abs(A1 - A2)
        # 计算相对差异
        reldiff = diff / torch.abs(A1 + 1e-8)
        # 记录绝对差异的平均值
        diffs.append(diff.mean().item())
        # 记录相对差异的平均值
        reldiffs.append(reldiff.mean().item())
        # 断言绝对差异的平均值小于指定阈值
        assert diff.mean().item() < 0.0135
    # 打印绝对差异的平均值
    print(sum(diffs)/len(diffs))
    # 打印相对差异的平均值
    print(sum(reldiffs)/len(reldiffs))
    # 循环100次
    for i in range(100):
        # 在CUDA设备上生成一个随机的1024x1024张量
        A1 = torch.rand(1024, 1024, device="cuda")
        # 对A1进行量化操作，返回量化后的结果C和缩放因子S
        C, S = F.quantize(A1)
        # 对量化后的结果C和缩放因子S进行反量化操作，得到A2
        A2 = F.dequantize(C, S)
        # 计算A1和A2之间的绝对值差的平均值
        diff = torch.abs(A1 - A2).mean().item()
        # 使用指定的绝对误差和相对误差检查A1和A2是否相等
        torch.testing.assert_close(A1, A2, atol=1e-2, rtol=0)
        # 断言A1和A2之间的差异小于0.004
        assert diff < 0.004
# 使用 pytest.mark.parametrize 装饰器为 test_dynamic_blockwise_quantization 函数添加参数化测试
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
# 参数化 nested 参数，取值为 True 和 False
@pytest.mark.parametrize("nested", TRUE_FALSE, ids=id_formatter("nested"))
# 参数化 blocksize 参数，取值为 4096, 2048, 1024, 512, 256, 128, 64
@pytest.mark.parametrize("blocksize", [4096, 2048, 1024, 512, 256, 128, 64])
# 参数化 signed 参数，取值为 True 和 False
@pytest.mark.parametrize("signed", TRUE_FALSE, ids=id_formatter("signed"))
# 定义测试函数 test_dynamic_blockwise_quantization，接受 dtype, nested, blocksize, signed 四个参数
def test_dynamic_blockwise_quantization(dtype, nested, blocksize, signed):
    # 初始化空列表 diffs 和 reldiffs
    diffs = []
    reldiffs = []
    # 循环 100 次
    for i in range(100):
        # 生成随机数据 A1
        A1 = torch.randn(1024, 1024, device="cuda", dtype=dtype)
        # 对 A1 进行分块量化
        C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested)
        # 对量化后的数据进行反量化
        A2 = F.dequantize_blockwise(C, S)
        # 计算 A1 和 A2 之间的绝对差值
        diff = torch.abs(A1 - A2).float()
        # 计算相对差值
        reldiff = diff / torch.abs(A1.float() + 1e-8)
        # 计算绝对差值的平均值并添加到 diffs 列表中
        diffs.append(diff.mean().item())
        # 计算相对差值的平均值并添加到 reldiffs 列表中
        reldiffs.append(reldiff.mean().item())
    # 计算绝对误差和相对误差
    abserr = sum(diffs)/len(diffs)
    relerr = sum(reldiffs)/len(reldiffs)
    # 断言绝对误差小于 0.011
    assert abserr < 0.011
    # 断言相对误差小于 0.018
    assert relerr < 0.018
    # 断言 A2 的数据类型为 dtype

    # 重新初始化 diffs 列表
    diffs = []
    # 生成动态映射 code
    code = F.create_dynamic_map(signed=signed)
    # 再次循环 100 次
    for i in range(100):
        # 生成随机数据 A1
        A1 = torch.rand(1024, 1024, device="cuda", dtype=dtype)
        # 对 A1 进行分块量化
        C, S = F.quantize_blockwise(A1, blocksize=blocksize, nested=nested, code=code)
        # 对量化后的数据进行反量化
        A2 = F.dequantize_blockwise(C, S)
        # 计算 A1 和 A2 之间的绝对差值
        diff = torch.abs(A1 - A2).float()
        # 计算相对差值
        reldiff = diff / torch.abs(A1.float() + 1e-8)
        # 计算绝对差值的平均值并添加到 diffs 列表中
        diffs.append(diff.mean().item())
        # 计算相对差值的平均值并添加到 reldiffs 列表中
        reldiffs.append(reldiff.mean().item())
    # 计算绝对误差和相对误差
    abserr = sum(diffs)/len(diffs)
    relerr = sum(reldiffs)/len(reldiffs)
    # 根据 signed 参数进行不同的断言
    if signed:
        assert abserr < 0.0035
        assert relerr < 0.015
    else:
        assert abserr < 0.00175
        assert relerr < 0.012
    # 断言 A2 的数据类型为 dtype
    # 打印 signed、nested、rand 和 blocksize 的值以及 diffs 列表的平均值
    #print('signed=', signed, 'nested=', nested, 'rand', blocksize, sum(diffs)/len(diffs))
    # 打印 signed、nested、rand 和 blocksize 的值以及 reldiffs 列表的平均值
    #print('signed=', signed, 'nested=', nested, 'rand', blocksize, sum(reldiffs)/len(reldiffs))
# 使用 pytest.mark.parametrize 装饰器为 test_percentile_clipping 函数添加参数化测试，测试 torch.float32 和 torch.float16 两种数据类型
@pytest.mark.parametrize(
    "gtype", [torch.float32, torch.float16], ids=["float", "half"]
)
# 定义测试函数 test_percentile_clipping，用于测试百分位剪裁功能
def test_percentile_clipping(gtype):
    # 在 GPU 上创建两个全零张量 gnorm_vec1 和 gnorm_vec2
    gnorm_vec1 = torch.zeros(100, device="cuda")
    gnorm_vec2 = torch.zeros(100, device="cuda")
    n = 4
    step = 0
    percentile = 5
    # 循环 k 次
    for i in range(k):
        step += 1
        # 在 GPU 上创建一个随机张量 g，数据类型为 gtype
        g = torch.randn(n, n, dtype=gtype, device="cuda")
        # 调用 F.percentile_clipping 函数进行百分位剪裁
        gnorm1, clip2, gnorm_scale = F.percentile_clipping(
            g, gnorm_vec2, step, percentile=percentile
        )
        # 断言 gnorm_scale 的值为 1.0（如果 gnorm1 小于 clip2），否则为 clip2 除以 gnorm1
        assert gnorm_scale == 1.0 if gnorm1 < clip2 else clip2 / gnorm1

        # 计算 g 的 L2 范数
        gnorm2 = torch.norm(g.float())
        # 如果 step 为 1，则将 gnorm2 赋值给 gnorm_vec1，否则将 gnorm2 存储在 gnorm_vec1 的特定位置
        if step == 1:
            gnorm_vec1[:] = gnorm2
        else:
            gnorm_vec1[step % 100] = gnorm2

        # 对 gnorm_vec1 进行排序，并获取百分位处的值
        vals, idx = torch.sort(gnorm_vec1)
        clip1 = vals[percentile]

        # 使用 torch.testing.assert_close 函数检查 gnorm_vec1 和 gnorm_vec2 的平方根是否接近
        torch.testing.assert_close(gnorm_vec1, torch.sqrt(gnorm_vec2))
        # 使用 torch.testing.assert_close 函数检查 clip1 和 clip2 是否接近
        torch.testing.assert_close(clip1, clip2)
        # 使用 torch.testing.assert_close 函数检查 gnorm1 和 gnorm2 是否接近

# 定义 quant 函数，用于对输入张量进行量化
def quant(x):
    # 计算张量 x 的绝对值的最大值
    max1 = torch.abs(x).max()
    # 将张量 x 除以 max1 乘以 127 并四舍五入，然后转换为 torch.int8 类型
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)

# 定义 dequant 函数，用于对量化后的张量进行反量化
def dequant(c, maxC):
    return c.float() * (maxC / 127)

# 定义 mm_dequant 函数，用于对两个量化后的张量进行矩阵乘法并反量化
def mm_dequant(maxA, maxB, C):
    return C.float() * (maxA / 127) * (maxB / 127)

# 定义 quant_multi 函数，用于对输入张量的指定维度进行量化
def quant_multi(x, dim):
    # 计算张量 x 在指定维度上的绝对值的最大值
    max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
    max1[max1 == 0] = 1.0
    # 将张量 x 在指定维度上除以 max1 乘以 127 并四舍五入，然后转换为 torch.int8 类型
    x = torch.round(x / max1 * 127)
    return max1, x.to(torch.int8)

# 定义 quant_multi_chunk 函数，用于对输入张量的指定维度进行分块量化
def quant_multi_chunk(x, dim, chunk_size=32):
    if dim == 1:
        # 将张量 x 按照指定维度分块
        x_chunked = einops.rearrange(x, "(c a) b -> c a b", c=chunk_size)
        # 计算分块后张量 x 在指定维度上的绝对值的最大值
        max1 = torch.amax(torch.abs(x_chunked), dim=dim + 1, keepdim=True)
        max1 = torch.tile(max1, (1, 1, x.shape[1]))
        max1 = max1.view(x.shape)
    elif dim == 0:
        # 将张量 x 按照指定维度分块
        x_chunked = einops.rearrange(x, "a (b c) -> a b c", c=chunk_size)
        # 计算分块后张量 x 在指定维度上的绝对值的最大值
        max1 = torch.amax(torch.abs(x_chunked), dim=dim, keepdim=True)
        max1 = torch.tile(max1, (x.shape[0], 1, 1))
        max1 = max1.view(x.shape)
    max1[max1 == 0] = 1.0
    # 将张量 x 中的值除以 max1，然后四舍五入并乘以 127，将结果赋值给 x
    x = torch.round(x / max1 * 127)
    # 将张量 x 转换为 int8 类型，并返回 max1 和转换后的张量 x
    return max1, x.to(torch.int8)
# 定义一个函数，用于计算给定数组的最小值和最大值
def quant_minmax(A):
    # 计算数组 A 的最小值
    minA = A.min()
    # 计算数组 A 的最大值
    maxA = A.max()


# 定义一个函数，用于计算给定数组的平均值
def mean(xx):
    # 返回数组 xx 的总和除以数组长度的浮点数值
    return sum(xx) / float(len(xx))


# 定义一个包含不同方法的字典
methods = {
    "linear": (
        lambda x, dim: quant(x),
        lambda x, dim: quant(x),
        dequant,
        dequant,
        mm_dequant,
    ),
    "vectorwise": (quant_multi, quant_multi, dequant, dequant, mm_dequant),
}


# 使用 pytest.mark.parametrize 注册测试参数
@pytest.mark.parametrize("dim1", [1024 * 2], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [1024 * 16], ids=id_formatter("dim2"))
@pytest.mark.parametrize("quant_methods", methods.values(), ids=methods.keys())
@pytest.mark.parametrize("batched", TRUE_FALSE, ids=id_formatter("batched"))
def test_approx_igemm(dim1, dim2, quant_methods, batched):
    # 将 dim1 和 dim2 调整为能被 32 整除的值
    dim1 = dim1 - (dim1 % 32)
    dim2 = dim2 - (dim2 % 32)
    errors = []
    relerrors = []
    #print("")
    # 循环 5 次
    for i in range(5):
        # 根据 batched 参数选择不同的数据生成方式
        if batched:
            A = torch.normal(0, 0.5, size=(32, dim1, dim2 // 32), device="cuda")
            B = torch.normal(0, 0.5, size=(32, dim2 // 32, dim1), device="cuda")
            maxA, Ac = quant_methods[0](A, 2)
            maxB, Bc = quant_methods[1](B, 1)
        else:
            A = torch.normal(0, 0.5, size=(dim1, dim2), device="cuda")
            B = torch.normal(0, 0.5, size=(dim2, dim1), device="cuda")
            maxA, Ac = quant_methods[0](A, 1)
            maxB, Bc = quant_methods[1](B, 0)
        # 断言两个张量的值在一定误差范围内接近
        torch.testing.assert_close(
            quant_methods[2](maxA, Ac), A, atol=0.025, rtol=0.05
        )
        if batched:
            out2 = torch.bmm(A, B)
            C = torch.bmm(Ac.float(), Bc.float())
        else:
            out2 = torch.mm(A, B)
            C = F.igemm(Ac, Bc)
        out = quant_methods[4](maxA, maxB, C)
        std = out2.std()
        out /= std
        out2 /= std
        err = torch.abs(out - out2)
        relerr = err / torch.abs(out2)
        errors.append(err.mean().item())
        relerrors.append(relerr.mean().item())
    #print(mean(errors))
    #print(mean(relerrors))
# 测试稳定嵌入层的初始化和重置参数
def test_stable_embedding():
    # 创建稳定嵌入层对象，输入和输出维度均为1024
    layer = bnb.nn.StableEmbedding(1024, 1024)
    # 重置参数
    layer.reset_parameters()

# 参数化测试，测试不同维度和转置情况下的整数矩阵乘法
@pytest.mark.parametrize("hidden_dim", get_test_dims(32, 256, n=2), ids=id_formatter("hidden_dim"))
@pytest.mark.parametrize("batch_dim", get_test_dims(16, 256, n=2), ids=id_formatter("batch_dim"))
@pytest.mark.parametrize("seq_dim", get_test_dims(16, 256, n=2), ids=id_formatter("seq_dim"))
@pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
def test_igemm(hidden_dim, batch_dim, transpose, seq_dim):
    # 调整维度大小，使其能够整除32或16
    hidden_dim = hidden_dim - (hidden_dim % 32)
    batch_dim = batch_dim - (batch_dim % 16)
    seq_dim = seq_dim - (seq_dim % 16)
    # 循环执行k次
    for i in range(k):
        # 根据转置情况确定矩阵A的形状
        shapeA = (
            (batch_dim, hidden_dim)
            if not transpose[0]
            else (hidden_dim, batch_dim)
        )
        # 根据转置情况确定矩阵B的形状
        shapeB = (
            (32 * random.randint(1, 4), hidden_dim)
            if transpose[1]
            else (hidden_dim, 32 * random.randint(1, 4))
        )
        # 生成随机整数矩阵A和B，数据类型为int8，存储在GPU上
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
        # 根据转置情况进行整数矩阵乘法运算
        if not transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.matmul(A.float(), B.t().float())
            out = F.igemm(A, B.t())
        elif transpose[0] and not transpose[1]:
            out2 = torch.matmul(A.t().float(), B.float())
            out = F.igemm(A.t(), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.matmul(A.t().float(), B.t().float())
            out = F.igemm(A.t(), B.t())
        
        # 断言两个矩阵乘法的结果非常接近
        torch.testing.assert_close(out.float(), out2)
    # 循环 k 次
    for i in range(k):
        # 定义矩阵 A 的形状为 (batch_dim, seq_dim, hidden_dim)
        shapeA = (batch_dim, seq_dim, hidden_dim)
        # 根据 transpose[1] 的值确定矩阵 B 的形状
        shapeB = (
            (32 * random.randint(1, 4), hidden_dim)
            if transpose[1]
            else (hidden_dim, 32 * random.randint(1, 4))
        )
        # 生成随机整数矩阵 A，数据范围在 -128 到 127 之间，存储在 GPU 上
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        # 生成随机整数矩阵 B，数据范围在 -128 到 127 之间，存储在 GPU 上
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)
        
        # 根据 transpose[0] 和 transpose[1] 的值选择不同的矩阵乘法方式
        if not transpose[0] and not transpose[1]:
            # 执行矩阵乘法操作，结果存储在 out2 中
            out2 = torch.matmul(A.float(), B.float())
            # 调用 F.igemm 函数执行整数矩阵乘法操作，结果存储在 out 中
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            # 执行矩阵乘法操作，其中 B 需要转置，结果存储在 out2 中
            out2 = torch.matmul(A.float(), B.t().float())
            # 调用 F.igemm 函数执行整数矩阵乘法操作，其中 B 需要转置，结果存储在 out 中
            out = F.igemm(A, B.t())

        # 检查 out 和 out2 的结果是否接近
        torch.testing.assert_close(out.float(), out2)
# 使用参数化测试，对 seq_dim 进行测试
@pytest.mark.parametrize("seq_dim", get_test_dims(32, 512, n=3), ids=id_formatter("seq_dim"))
# 使用参数化测试，对 hidden_dim 进行测试
@pytest.mark.parametrize("hidden_dim", get_test_dims(32, 1024 * 4, n=3), ids=id_formatter("hidden_dim"))
# 使用参数化测试，对 batch_dim 进行测试
@pytest.mark.parametrize("batch_dim", get_test_dims(2, 16, n=3), ids=id_formatter("batch_dim"))
# 定义测试函数 test_dim3_igemm，接受 seq_dim, hidden_dim, batch_dim 作为参数
def test_dim3_igemm(seq_dim, hidden_dim, batch_dim):
    # 将 seq_dim 调整为 32 的倍数
    seq_dim = seq_dim - (seq_dim % 32)
    # 将 hidden_dim 调整为 32 的倍数
    hidden_dim = hidden_dim - (hidden_dim % 32)
    # 将 batch_dim 调整为 2 的倍数
    batch_dim = batch_dim - (batch_dim % 2)
    # 循环 25 次
    for i in range(25):
        # 生成随机整数张量 A
        A = torch.randint(
            -128, 127, size=(batch_dim, seq_dim, hidden_dim), device="cuda"
        ).to(torch.int8)
        # 生成随机整数张量 B
        B = torch.randint(
            -128, 127, size=(batch_dim, seq_dim, 1024), device="cuda"
        ).to(torch.int8)
        # 使用 einsum 计算矩阵乘积
        out2 = torch.einsum("bsi, bso->io", A.float(), B.float())
        # 创建空张量 iout
        iout = torch.empty(
            A.shape[2], B.shape[2], dtype=torch.int32, device=A.device
        )
        # 使用 F.igemm 计算矩阵乘积
        out = F.igemm(A, B, out=iout)
        # 断言两个结果张量相近
        torch.testing.assert_close(out.float(), out2)

# 使用参数化测试，对 seq_dim 进行测试
@pytest.mark.parametrize("seq_dim", get_test_dims(32, 512, n=2), ids=id_formatter("seq_dim"))
# 使用参数化测试，对 hidden_dim 进行测试
@pytest.mark.parametrize("hidden_dim", get_test_dims(32, 1024 * 4, n=2), ids=id_formatter("hidden_dim"))
# 使用参数化测试，对 batch_dim 进行测试
@pytest.mark.parametrize("batch_dim", get_test_dims(2, 16, n=2), ids=id_formatter("batch_dim"))
# 使用参数化测试，对 transpose 进行测试
@pytest.mark.parametrize("transpose", TRUE_FALSE, ids=id_formatter("transpose"))
# 定义测试函数 test_minmax_igemm，接受 seq_dim, hidden_dim, batch_dim, transpose 作为参数
def test_minmax_igemm(seq_dim, hidden_dim, batch_dim, transpose):
    # 定义函数 min_max，用于计算最大最小值
    def min_max(x):
        # 计算最大值
        maxA = torch.amax(x, dim=2, keepdim=True)
        # 计算最小值
        minA = torch.amin(x, dim=2, keepdim=True)
        # 计算缩放比例
        scale = (maxA - minA) / 2.0
        return (127 * (x - minA - scale) / scale).to(torch.int8), minA, scale

    # 将 seq_dim 调整为 16 的倍数
    seq_dim = seq_dim - (seq_dim % 16)
    # 将 hidden_dim 调整为 16 的倍数
    hidden_dim = hidden_dim - (hidden_dim % 16)
    # 将 batch_dim 调整为 2 的倍数
    batch_dim = batch_dim - (batch_dim % 2)
    # 初始化错误列表
    errs = []
    relerrs = []
    errs2 = []
    relerrs2 = []
    # 对于给定的范围 k，循环执行以下操作
    for i in range(k):
        # 生成服从正态分布的张量 A，大小为(batch_dim, seq_dim, hidden_dim)，存储在 GPU 上
        A = torch.normal(
            0.0, 0.5, size=(batch_dim, seq_dim, hidden_dim), device="cuda"
        )
        # 如果需要转置
        if transpose:
            # 生成服从正态分布的张量 B，大小为(256, hidden_dim)，存储在 GPU 上
            B = torch.normal(0, 0.5, size=(256, hidden_dim), device="cuda")
        else:
            # 生成服从正态分布的张量 B，大小为(hidden_dim, 256)，存储在 GPU 上
            B = torch.normal(0, 0.5, size=(hidden_dim, 256), device="cuda")
        # 对张量 A 进行最小最大值归一化，返回归一化后的张量 Ac、最小值 minA 和缩放比例 scale
        Ac, minA, scale = min_max(A)
        # 如果需要转置
        if transpose:
            # 对张量 B 进行量化，返回最大值 maxB 和量化后的张量 Bc
            maxB, Bc = quant_multi(B, dim=(1 if transpose else 0))
            # 使用 F.igemm 函数执行矩阵乘法操作，结果存储在 out 中
            out = F.igemm(Ac, Bc.t())
            # 使用 torch.matmul 函数执行矩阵乘法操作，结果存储在 out2 中
            out2 = torch.matmul(A, B.t())
            # 计算偏移量 offset
            offset = B.t().sum(0) * (minA + scale)
            # 将 out 转换为浮点型
            out = out.float()
            # 对 out 进行缩放和偏移操作
            out = (out * maxB.t() * scale / (127 * 127)) + offset

            # 对张量 A 进行量化，返回最大值 maxA 和量化后的张量 Ac
            maxA, Ac = quant_multi(A, dim=2)
            # 使用 F.igemm 函数执行矩阵乘法操作，结果存储在 out3 中
            out3 = F.igemm(Ac, Bc.t())
            # 对 out3 进行反量化操作
            out3 = mm_dequant(maxA, maxB.t(), out3)
        else:
            # 对张量 B 进行量化，返回最大值 maxB 和量化后的张量 Bc
            maxB, Bc = quant_multi(B, dim=0)
            # 计算偏移量 offset
            offset = B.sum(0) * (minA + scale)
            # 使用 F.igemm 函数执行矩阵乘法操作，结果存储在 out 中
            out = F.igemm(Ac, Bc)
            # 使用 torch.matmul 函数执行矩阵乘法操作，结果存储在 out2 中
            out2 = torch.matmul(A, B)
            # 将 out 转换为浮点型
            out = out.float()
            # 对 out 进行缩放和偏移操作
            out = (out * maxB * scale / (127 * 127)) + offset

            # 对张量 A 进行量化，返回最大值 maxA 和量化后的张量 Ac
            maxA, Ac = quant_multi(A, dim=2)
            # 使用 F.igemm 函数执行矩阵乘法操作，结果存储在 out3 中
            out3 = F.igemm(Ac, Bc)
            # 对 out3 进行反量化操作
            out3 = mm_dequant(maxA, maxB, out3)

        # 计算 out2 的标准差
        std = out2.std()
        # 对 out2 进行标准化
        out2 /= std
        out /= std
        out3 /= std

        # 计算 out 与 out2 之间的绝对误差
        err = torch.abs(out - out2)
        # 计算相对误差
        relerr = err / (torch.abs(out2) + 1e-7)

        # 计算 out3 与 out2 之间的绝对误差
        err2 = torch.abs(out3 - out2)
        # 计算相对误差
        relerr2 = err2 / (torch.abs(out2) + 1e-7)

        # 将绝对误差的均值添加到 errs 列表中
        errs.append(err.mean().item())
        # 将相对误差的均值添加到 relerrs 列表中
        relerrs.append(relerr.mean().item())
        # 将第二组绝对误差的均值添加到 errs2 列表中
        errs2.append(err2.mean().item())
        # 将第二组相对误差的均值添加到 relerrs2 列表中
        relerrs2.append(relerr2.mean().item()
    # 断言第一组绝对误差的均值小于0.015
    assert mean(errs) < 0.015
    # 断言第一组相对误差的均值小于0.3
    assert mean(relerrs) < 0.3
# 使用参数化测试，对 dim1 进行测试，范围为 1 到 64，共 2 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim1", get_test_dims(1, 64, n=2), ids=id_formatter("dim1"))
# 使用参数化测试，对 dim2 进行测试，范围为 32 到 128，共 2 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim2", get_test_dims(32, 128, n=2), ids=id_formatter("dim2"))
# 使用参数化测试，对 dim3 进行测试，范围为 32 到 256，共 2 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim3", get_test_dims(32, 256, n=2), ids=id_formatter("dim3"))
# 使用参数化测试，对 dim4 进行测试，范围为 32 到 256，共 2 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim4", get_test_dims(32, 256, n=2), ids=id_formatter("dim4"))
# 使用参数化测试，对 transpose 进行测试，使用 BOOLEAN_TUPLES 中的值，共 4 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("transpose", BOOLEAN_TUPLES, ids=id_formatter("transpose"))
# 定义测试函数 test_ibmm，接受 dim1、dim2、dim3、dim4、transpose 作为参数
def test_ibmm(dim1, dim2, dim3, dim4, transpose):
    # 对 dim2、dim3、dim4 进行调整，使其为 16 的倍数
    dim2 = dim2 - (dim2 % 16)
    dim3 = dim3 - (dim3 % 16)
    dim4 = dim4 - (dim4 % 16)
    # 循环 k 次
    for i in range(k):
        # 根据 transpose 的值确定 shapeA 和 shapeB 的形状
        shapeA = (dim1, dim3, dim2) if transpose[0] else (dim1, dim2, dim3)
        shapeB = (dim1, dim4, dim3) if transpose[1] else (dim1, dim3, dim4)
        # 生成随机整数张量 A 和 B，设备为 cuda，数据类型为 torch.int8
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        B = torch.randint(-128, 127, size=shapeB, device="cuda").to(torch.int8)

        # 根据 transpose 的值选择不同的计算方式
        if not transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.float(), B.float())
            out = F.igemm(A, B)
        elif not transpose[0] and transpose[1]:
            out2 = torch.bmm(A.float(), B.permute([0, 2, 1]).float())
            out = F.igemm(A, B.permute([0, 2, 1]))
        elif transpose[0] and not transpose[1]:
            out2 = torch.bmm(A.permute([0, 2, 1]).float(), B.float())
            out = F.igemm(A.permute([0, 2, 1]), B)
        elif transpose[0] and transpose[1]:
            out2 = torch.bmm(
                A.permute([0, 2, 1]).float(), B.permute([0, 2, 1]).float()
            )
            out = F.igemm(A.permute([0, 2, 1]), B.permute([0, 2, 1]))
        # 断言两个张量的值接近
        torch.testing.assert_close(out.float(), out2.float())

# 使用参数化测试，对 dim1 进行测试，范围为 1 到 64，共 1 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim1", get_test_dims(1, 64, n=1), ids=id_formatter("dim1"))
# 使用参数化测试，对 dim2 进行测试，范围为 32 到 128，共 1 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim2", get_test_dims(32, 128, n=1), ids=id_formatter("dim2"))
# 使用参数化测试，对 dim3 进行测试，范围为 32 到 256，共 1 个值，使用 id_formatter 函数生成测试用例的标识符
@pytest.mark.parametrize("dim3", get_test_dims(32, 256, n=1), ids=id_formatter("dim3"))
# 定义测试函数 test_vector_quant，接受 dim1、dim2、dim3 作为参数
def test_vector_quant(dim1, dim2, dim3):
    # 将 dim2 调整为最接近的 16 的倍数
    dim2 = dim2 - (dim2 % 16)
    # 将 dim3 调整为最接近的 16 的倍数
    dim3 = dim3 - (dim3 % 16)
    # 遍历 k 次
    for i in range(k):
        # 在 GPU 上生成一个 dim2 x dim3 大小的随机张量 A
        A = torch.randn(size=(dim2, dim3), device="cuda")
        # 对张量 A 进行向量化量化，得到量化后的张量 qA 和量化参数 SA
        qA, SA = F.vectorwise_quant(A, dim=0)
        # 对量化后的张量 qA 进行反向量化，得到还原后的张量 A1
        A1 = F.vectorwise_dequant(qA, SA)
        # 计算张量 A1 和原始张量 A 之间的近似误差，确保它们在一定的误差范围内
        n = A1.numel()
        assert_all_approx_close(A1, A, atol=0.01, rtol=0.1, count=int(n*0.002))
# 使用参数化测试，对 dim1 进行测试
@pytest.mark.parametrize("dim1", get_test_dims(2, 256, n=2), ids=id_formatter("dim1"))
# 使用参数化测试，对 dim2 进行测试
@pytest.mark.parametrize("dim2", get_test_dims(2, 256, n=2), ids=id_formatter("dim2"))
# 使用参数化测试，对 dim3 进行测试
@pytest.mark.parametrize("dim3", get_test_dims(2, 256, n=2), ids=id_formatter("dim3"))
# 使用参数化测试，对 dtype 进行测试
@pytest.mark.parametrize("dtype", [torch.int8, torch.int32], ids=describe_dtype)
# 使用参数化测试，对 orderA 进行测试
@pytest.mark.parametrize("orderA", ["row"], ids=id_formatter("orderA"))
# 使用参数化测试，对 orderOut 进行测试
@pytest.mark.parametrize("orderOut", ["col", "row", "col32"], ids=id_formatter("orderOut"))
# 使用参数化测试，对 transpose 进行测试
@pytest.mark.parametrize("transpose", [False], ids=id_formatter("transpose"))
# 使用参数化测试，对 dims 进行测试
@pytest.mark.parametrize("dims", [2, 3], ids=id_formatter("dims"))
# 定义测试函数 test_nvidia_transform，传入参数 dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose
def test_nvidia_transform(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    # 如果 dims 为 3 且 orderOut 不为 "col32"，则返回
    if dims == 3 and orderOut != "col32":
        return
    # 如果 dtype 为 torch.int32 且 orderOut 不为 "col32"，则返回
    if dtype == torch.int32 and orderOut != "col32":
        return
    try:
        # 获取转换函数 func
        func = F.get_transform_func(dtype, orderA, orderOut, transpose)
    except ValueError as ve:
        # 如果出现异常，跳过测试
        pytest.skip(str(ve))  # skip if not supported

    # 根据 dims 的值生成不同维度的随机张量 A
    if dims == 2:
        A = torch.randint(-128, 127, size=(dim1, dim2), device="cuda").to(dtype)
    elif dims == 3:
        A = torch.randint(-128, 127, size=(dim1, dim2, dim3), device="cuda").to(
            dtype
        )

    # 对 A 进行转换，得到输出张量 out 和转换矩阵 S
    out, S = F.nvidia_transform(A, to_order=orderOut)

    # 根据 orderOut 的值进行断言
    if orderOut == "row":
        torch.testing.assert_close(A.flatten(), out.flatten())
    elif orderOut == "col":
        torch.testing.assert_close(A.t().flatten(), out.flatten())
    elif orderOut == "col32":
        # 如果 dims 为 2，则计算 n
        if dims == 2:
            n = A.shape[0] * (A.shape[1] + (32 - (A.shape[1] % 32)))
        # 如果 dims 为 3，则计算 n
        elif dims == 3:
            n = (
                A.shape[0]
                * A.shape[1]
                * (A.shape[2] + (32 - (A.shape[2] % 32)))
            )
        # 断言输出张量的元素数量为 n
        assert out.numel() == n
    elif orderOut == "col_turing":
        # 计算需要的总元素个数，保证填充到32列8行的矩阵
        n = (A.shape[0] + (8 - A.shape[0] % 8)) * (
            A.shape[1] + (32 - (A.shape[1] % 32))
        )
        # 断言输出的元素个数等于计算得到的总元素个数
        assert out.numel() == n
        # 计算总共的列瓦片数
        total_coltile = (A.shape[1] // 32) + (1 if A.shape[1] % 32 != 0 else 0)
        # 遍历矩阵A的行和列
        for row in range(A.shape[0]):
            for col in range(A.shape[1]):
                i = row * A.shape[1]
                j = col

                # 计算当前列所在的瓦片数
                coltile = (col // 32) + (1 if col % 32 != 0 else 0)
                # 计算当前行所在的瓦片数
                rowtile = (
                    (row // 8) + (1 if row % 8 != 0 else 0)
                ) * total_coltile
                # 计算偏移量
                offset = 32 * 8 * (rowtile + coltile)
                col2 = col % 32
                row2 = (row % 8) * 32

                # 断言矩阵A的扁平化后的元素等于A[row, col]
                assert A.flatten()[i + j] == A[row, col]
                # assert A.flatten()[i+j] == out.flatten()[row2+col2]
                # torch.testing.assert_close(A.flatten()[i+j], A[row, col])
                # torch.testing.assert_close(A.flatten()[i+j], out.flatten()[row2+ col2+block_offset])

    if orderOut == "col32":
        # 对输出进行转换，从列优先到行优先
        out2, S = F.nvidia_transform(
            out, from_order=orderOut, to_order="row", state=S
        )
        # 断言矩阵A和转换后的输出out2相等
        torch.testing.assert_close(A, out2)
# 使用参数化测试，测试不同维度的矩阵乘法
@pytest.mark.parametrize("dim1", get_test_dims(1, 256, n=1), ids=id_formatter("dim1"))
# 参数化测试，测试不同维度的矩阵乘法
@pytest.mark.parametrize("dim2", get_test_dims(32, 512, n=1), ids=id_formatter("dim2"))
# 参数化测试，测试不同维度的矩阵乘法
@pytest.mark.parametrize("dim3", get_test_dims(32, 1024, n=1), ids=id_formatter("dim3"))
# 参数化测试，测试不同维度的矩阵乘法
@pytest.mark.parametrize("dim4", get_test_dims(32, 1024, n=1), ids=id_formatter("dim4"))
# 参数化测试，测试不同维度的矩阵乘法
@pytest.mark.parametrize("dims", (2, 3), ids=id_formatter("dims"))
# 参数化测试，测试不同维度的矩阵乘法
@pytest.mark.parametrize("ldb", (0,), ids=id_formatter("ldb"))
# 定义测试函数，测试整数矩阵乘法
def test_igemmlt_int(dim1, dim2, dim3, dim4, dims, ldb):
    # 循环执行测试
    for i in range(k):
        # 根据维度生成随机整数矩阵A
        if dims == 2:
            A = torch.randint(-128, 127, size=(dim1, dim3), device="cuda").to(
                torch.int8
            )
        elif dims == 3:
            A = torch.randint(
                -128, 127, size=(dim1, dim2, dim3), device="cuda"
            ).to(torch.int8)
        # 根据维度生成随机整数矩阵B
        B = torch.randint(-128, 127, size=(dim4, dim3), device="cuda").to(
            torch.int8
        )
        # 计算矩阵乘法结果C1
        C1 = torch.matmul(A.float(), B.t().float())

        # 对矩阵A进行转换
        A2, SA = F.transform(A, "col32")
        # 对矩阵B进行转换
        B2, SB = F.transform(B, "col_turing")
        # 执行整数矩阵乘法
        C2, SC = F.igemmlt(A2, B2, SA, SB)
        # 对结果进行转换
        C3, S = F.nvidia_transform(C2, "row", state=SC)
        # 断言结果是否一致
        torch.testing.assert_close(C1, C3.float())

        # 转置矩阵B
        B = torch.randint(-128, 127, size=(dim3, dim4), device="cuda").to(
            torch.int8
        )
        # 计算矩阵乘法结果C1
        C1 = torch.matmul(A.float(), B.float())

        # 对转置后的矩阵B进行转换
        B2t, SBt = F.transform(B, "col_turing", transpose=True)
        # 执行整数矩阵乘法
        C2, SC = F.igemmlt(A2, B2t, SA, SBt)
        # 对结果进行转换
        C3, S = F.nvidia_transform(C2, "row", state=SC)
        # 断言结果是否一致
        torch.testing.assert_close(C1, C3.float())


# 参数化测试，测试特定维度的矩阵乘法
@pytest.mark.parametrize("dim1", [32], ids=id_formatter("dim1"))
# 参数化测试，测试特定维度的矩阵乘法
@pytest.mark.parametrize("dim2", [32], ids=id_formatter("dim2"))
# 参数化测试，测试特定维度的矩阵乘法
@pytest.mark.parametrize("dim3", [32], ids=id_formatter("dim3"))
# 参数化测试，测试特定维度的矩阵乘法
@pytest.mark.parametrize("dim4", [32], ids=id_formatter("dim4"))
# 参数化测试，测试特定维度的矩阵乘法
@pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
# 定义一个测试函数，用于测试8位整数矩阵乘法加速器的性能
def test_igemmlt_half(dim1, dim2, dim3, dim4, dims):
    # 获取特殊格式字符串
    formatB = F.get_special_format_str()
    # 循环k次
    for i in range(k):
        # 根据维度dims的不同，生成不同维度的随机数矩阵A
        if dims == 2:
            A = torch.normal(0, 0.5, size=(dim1, dim3), device="cuda").half()
        elif dims == 3:
            A = torch.normal(
                0, 0.5, size=(dim1, dim2, dim3), device="cuda"
            ).half()
        # 生成随机数矩阵B，并进行Xavier初始化
        B = torch.randn((dim4, dim3), device="cuda").half()
        torch.nn.init.xavier_uniform_(B)
        # 计算矩阵乘积C1和C2
        C1 = torch.matmul(A, B.t())
        C2 = bnb.matmul(A, B.t())

        # 将A视为一维向量
        A = A.view(-1, A.shape[-1])

        # 对A和B进行双精度量化
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        CB, CBt, statsB, statsBt, coo_tensor = F.double_quant(B)
        # 对CA和CB进行格式转换
        C32A, SA = F.transform(CA, "col32")
        CxB, SB = F.transform(CB, to_order=formatB)
        # 进行整数矩阵乘法运算
        out1_32, Sout1_32 = F.igemmlt(C32A, CxB, SA, SB)
        # 对结果进行反量化
        output = F.mm_dequant(out1_32, Sout1_32, statsAt, statsBt)

        # 打印输出结果的前10个元素
        # print('')
        # print(output.flatten()[:10])
        # print(C1.flatten()[:10])
        # print(C2.flatten()[:10])

        # 断言C1和output的近似相等性
        # torch.testing.assert_close(C1.view(-1, C1.shape[-1]), output, atol=0.025, rtol=0.05)

        # 转置操作
        # B = torch.randint(-128, 127, size=(dim3, dim4), device='cuda').to(torch.int8)
        # C1 = torch.matmul(A.float(), B.float())

        # 对B进行格式转换，同时进行转置
        # B2t, SBt = F.transform2(B, 'col_turing', transpose=True)
        # 进行整数矩阵乘法运算
        # C2, SC = F.igemmlt(A2, B2t, SA, SBt)
        # 对结果进行格式转换
        # C3, S = F.transform(C2, 'row', state=SC)
        # 断言C1和C3的近似相等性
        # torch.testing.assert_close(C1, C3.float())

# 参数化测试函数，测试不同参数下的性能
@pytest.mark.parametrize(
    ("batch", "seq", "model", "hidden"),
    [
        pytest.param(2, 512, 4 * 1024, 3 * 4 * 1024, id="batch=2, seq=512, model=4k, hidden=12k"),
        pytest.param(2, 512, 5120, 3 * 5120, id="batch=2, seq=512, model=5k, hidden=15k"),
        pytest.param(2, 512, 12 * 1024, 4 * 12 * 1024, id="batch=2, seq=512, model=12k, hidden=48k"),
    ],
)
# 标记为性能测试
@pytest.mark.benchmark
def test_bench_8bit_training(batch, seq, model, hidden):
    # 获取特殊格式字符串
    formatB = F.get_special_format_str()
    # 生成指定大小的随机张量 A 和 grad，并使用半精度，在 GPU 上运行
    A = torch.randn(batch, seq, model, device="cuda").half()
    grad = torch.randn(batch, seq, model, device="cuda").half()
    # 生成指定大小的随机整数张量 w1 和 w2，并使用半精度，在 GPU 上运行
    w1 = torch.randint(-128, 127, size=(hidden, model), device="cuda").half()
    w2 = torch.randint(-128, 127, size=(model, hidden), device="cuda").half()
    # 打印空行
    print("")

    # 进行预热操作
    # torch.cuda.synchronize()
    ## warmup
    # for i in range(100):
    #    torch.matmul(A, w1.t())
    # torch.cuda.synchronize()

    # 设置数据类型为 int8
    dtype = torch.int8
    # 将张量 A 和 grad 进行形状变换，并保证内存连续性
    A = A.view(-1, A.shape[-1]).contiguous()
    grad = grad.view(-1, grad.shape[-1]).contiguous()
    # 同步 GPU 计算
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(k):

        out1 = torch.matmul(A, w1.t())  # fc1
        # out2 = torch.matmul(out1, w2.t())# fc2

        # d1 = torch.matmul(grad, w2) # delta1
        # d2 = torch.matmul(d1, w1) # delta2

        # grad1 = torch.einsum('bo,bh->oh', out1, grad) # grad w2
        # grad2 = torch.einsum('bh,bo->ho', A, d2) # grad w1

    torch.cuda.synchronize()
    t16 = time.time() - t0
    # 打印时间
    print(t16)

    # 释放 GPU 缓存
    # torch.cuda.empty_cache()

    # 对 w1 和 w2 进行双精度量化
    # Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)

    # 对 w1 和 w2 进行特殊格式转换
    # CTw1, Sw1 = F.transform2(Cw1, formatB)
    # CTw2, Sw2 = F.transform2(Cw2, formatB)
    # CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
    # CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)

    # 对 A 进行双精度量化
    # CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
    # 对 CA 进行特殊格式转换
    # C32A, SA = F.transform2(CA, 'col32')
    ## fc1
    # out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, dtype=dtype)
    ##out1 = F.mm_dequant(out1_32, Sout1_32, statsAt, statsw1t)

    ## fc2
    # Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
    # C32out1, Sout1 = F.transform2(Cout1, 'col32')
    # out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, dtype=dtype)
    ##out2 = F.mm_dequant(out2_32, Sout2_32, statsout1t, statsw2t)

    ## delta1
    # Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
    # 对梯度进行双精度量化，得到量化后的梯度和统计信息
    # C32grad, Sgrad = F.transform2(Cgrad, 'col32')
    # 将梯度转换为col32格式
    ##d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, dtype=dtype)
    ## 使用igemm进行矩阵乘法计算d1_32，其中包括输入矩阵、权重矩阵的转置、输入矩阵的统计信息、权重矩阵的统计信息
    ##d1 = F.mm_dequant(d1_32, Sd1_32, statsgradt, statsw2)
    ## 对d1_32进行反量化操作，得到d1

    ## delta2
    # Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
    # 对d1进行双精度量化，得到量化后的d1和统计信息
    # C32d1, Sd1 = F.transform2(Cd1, 'col32')
    # 将d1转换为col32格式
    ##d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, dtype=dtype)
    ## 使用igemm进行矩阵乘法计算d2_32，其中包括输入矩阵、权重矩阵的转置、输入矩阵的统计信息、权重矩阵的统计信息
    ##d2 = F.mm_dequant(d2_32, Sd2_32, statsd1t, statsw1)

    ## grad1
    # C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
    # 将Cout1t转换为col32格式，并进行转置
    # CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
    # 将Cgradt转换为formatB格式，并进行转置
    ##grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, dtype=dtype)
    ## 使用igemm进行矩阵乘法计算grad1_32，其中包括输入矩阵、权重矩阵的转置、输入矩阵的统计信息、权重矩阵的统计信息
    ##grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1, statsgrad)

    ## grad2
    # C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
    # 将CAt转换为col32格式，并进行转置
    # CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
    # 将Cd1t转换为formatB格式，并进行转置
    ##grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, dtype=dtype)
    ## 使用igemm进行矩阵乘法计算grad2_32，其中包括输入矩阵、权重矩阵的转置、输入矩阵的统计信息、权重矩阵的统计信息
    ##grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsA, statsd1)

    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)
    # 对w2进行双精度量化，得到量化后的w2和统计信息

    # Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    # 对w1进行双精度量化，得到量化后的w1和统计信息
    # Cw2, Cw2t, statsw2, statsw2t, coo_tensor = F.double_quant(w2)
    # 对w2进行双精度量化，得到量化后的w2和统计信息

    # CTw1, Sw1 = F.transform2(Cw1, formatB)
    # 将Cw1转换为formatB格式
    # CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)
    # 将Cw1t转换为formatB格式，并进行转置
    # CTw2, Sw2 = F.transform2(Cw2, formatB)
    # 将Cw2转换为formatB格式
    # CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)
    # 将Cw2t转换为formatB格式，并进行转置
    # torch.cuda.synchronize()
    # 同步CUDA设备
    # t0 = time.time()
    # 记录当前时间
    # for i in range(k):
    #    #Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    #    #CTw1, Sw1 = F.transform2(Cw1, formatB)
    #    #Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
    #    #CTw1, Sw1 = F.transform2(Cw1, formatB)
    #    循环k次，执行注释部分的操作

    #    #CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A, threshold=3.5)
    #    对A进行双精度量化，得到量化后的A和统计信息，其中包括阈值为3.5
    #    CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
    #    #CTw1t, Sw1t = F.transform2(Cw1t, formatB, transpose=True)
    #    #CTw2, Sw2 = F.transform2(Cw2, formatB)
    #    #CTw2t, Sw2t = F.transform2(Cw2t, formatB, transpose=True)

    #    C32A, SA = F.transform2(CA, 'col32')

    #    # fc1
    #    out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1, dtype=dtype)
    #    #out1dn = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)

    #    #print(coo_tensor.nnz)
    #    #out1sp = F.spmm_coo(coo_tensor, w1.t())
    #    #print(w1.t().shape)
    #    #out1 = out1dn + out1sp

    #    # fc2
    #    Cout1, Cout1t, statsout1, statsout1t, coo_tensor = F.double_quant(out1)
    #    C32out1, Sout1 = F.transform2(Cout1, 'col32')
    #    out2_32, Sout2_32 = F.igemmlt(C32out1, CTw2, Sout1, Sw2, dtype=dtype)
    #    #out2 = F.mm_dequant(out2_32, Sout2_32, statsout1, statsw2)

    #    # delta1
    #    Cgrad, Cgradt, statsgrad, statsgradt, coo_tensor = F.double_quant(grad)
    #    C32grad, Sgrad = F.transform2(Cgrad, 'col32')
    #    d1_32, Sd1_32 = F.igemmlt(C32grad, CTw2t, Sgrad, Sw2t, dtype=dtype)
    #    #d1 = F.mm_dequant(d1_32, Sd1_32, statsgrad, statsw2t)

    #    # delta2
    #    Cd1, Cd1t, statsd1, statsd1t, coo_tensor = F.double_quant(d1)
    #    C32d1, Sd1 = F.transform2(Cd1, 'col32')
    #    d2_32, Sd2_32 = F.igemmlt(C32d1, CTw1t, Sd1, Sw1t, dtype=dtype)
    #    #d2 = F.mm_dequant(d2_32, Sd2_32, statsd1, statsw1t)

    #    # grad1
    #    #C32out1t, Sout1t = F.transform2(Cout1t, 'col32', transpose=True)
    #    #CTgradt, Sgradt = F.transform2(Cgradt, formatB, transpose=True)
    #    #grad1_32, Sgrad1_32 = F.igemmlt(C32out1t, CTgradt, Sout1t, Sgradt, dtype=dtype)
    #    #grad1 = F.mm_dequant(grad1_32, Sgrad1_32, statsout1t, statsgradt)

    ## grad2
    #    #C32At, SAt = F.transform2(CAt, 'col32', transpose=True)
    #    #CTd1t, Sd1t = F.transform2(Cd1t, formatB, transpose=True)
    #    #grad2_32, Sgrad2_32 = F.igemmlt(C32At, CTd1t, SAt, Sd1t, dtype=dtype)
    #    #grad2 = F.mm_dequant(grad2_32, Sgrad2_32, statsAt, statsd1t)
    
    # torch.cuda.synchronize()
    # t8 = time.time() - t0
    # print(t8)
# 使用参数化测试，测试维度为64到256之间的两个值
@pytest.mark.parametrize("dim1", get_test_dims(64, 256, n=2), ids=id_formatter("dim1"))
# 使用参数化测试，测试维度为64到1024之间的两个值
@pytest.mark.parametrize("dim4", get_test_dims(64, 1024, n=2), ids=id_formatter("dim4"))
# 使用参数化测试，测试维度为2
@pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
# 使用参数化测试，测试不同的格式
@pytest.mark.parametrize("formatB", ["col_turing", "col_ampere"], ids=id_formatter("formatB"))
# 使用参数化测试，测试是否有偏置
@pytest.mark.parametrize("has_bias", TRUE_FALSE, ids=id_formatter("has_bias"))
def test_dequant_mm(dim1, dim4, dims, formatB, has_bias):
    # 生成一个随机整数
    inner = torch.randint(1, 128, size=(1,)).item()
    bias = None
    # 如果有偏置，则生成一个随机偏置
    if has_bias: bias = torch.randn(dim4, device='cuda', dtype=torch.float16)
    # 获取特殊格式字符串
    formatB = F.get_special_format_str()
    # 循环一次
    for i in range(1):
        # 生成随机张量A和B
        A = torch.randn(dim1, inner, device="cuda")
        B = torch.randn(dim4, inner, device="cuda")
        # 计算矩阵乘法
        C1 = torch.matmul(A.half(), B.t().half())
        if has_bias: C1 += bias

        # 对A和B进行向量量化
        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)

        # 对A1和B1进行特殊转换
        A2, SA = F.nvidia_transform(A1, "col32")
        B2, SB = F.nvidia_transform(B1, formatB)
        # 使用igemmlt函数计算矩阵乘法
        C2, SC = F.igemmlt(A2, B2, SA, SB)

        # 对C2进行特殊转换
        C3, S = F.nvidia_transform(C2, "row", state=SC)
        # 对C3进行反量化
        C4 = F.vectorwise_mm_dequant(C3.float(), maxA, maxB.t())
        if has_bias: C4 += bias

        # 标准化C1和C4
        std = C1.std(0).view(1, -1)
        C1 /= std
        C4 /= std

        # 使用assert_all_approx_close函数比较C1和C4
        C5 = F.mm_dequant(C2, SC, maxA.flatten(), maxB.flatten(), bias=bias)
        n = C5.numel()
        assert_all_approx_close(C1, C4, atol=0.015, rtol=0.1, count=int(0.01 * n))

# 使用参数化测试，测试维度为1024
@pytest.mark.parametrize("dim1", [1 * 1024], ids=id_formatter("dim1"))
# 使用 pytest.mark.parametrize 注册测试参数，dim2 固定为 1024
@pytest.mark.parametrize("dim2", [1 * 1024], ids=id_formatter("dim2"))
# 使用 pytest.mark.parametrize 注册测试参数，dims 固定为 2
@pytest.mark.parametrize("dims", (2,), ids=id_formatter("dims"))
# 定义测试函数 test_colrow_absmax，参数为 dim1, dim2, dims
def test_colrow_absmax(dim1, dim2, dims):
    # 循环执行 k 次
    for i in range(k):
        # 设置阈值为 3.0
        threshold = 3.0
        # 生成一个在 CUDA 设备上的随机张量 A，数据类型为 half
        A = torch.randn(dim1, dim2, device="cuda").half()
        # 克隆张量 A 为 A_truncated
        A_truncated = A.clone()
        # 将 A_truncated 中绝对值大于等于 3.0 的元素置为 0.0
        A_truncated[torch.abs(A_truncated) >= 3.0] = 0.0
        # 如果 dims 为 2
        if dims == 2:
            # 计算 A 的每行绝对值的最大值和每列绝对值的最大值
            row_stats1, _ = torch.abs(A.float()).max(1)
            col_stats1, _ = torch.abs(A.float()).max(0)
            row_stats1_trunc, _ = torch.abs(A_truncated.float()).max(1)
            col_stats1_trunc, _ = torch.abs(A_truncated.float()).max(0)
        else:
            # 如果 dims 不为 2，则断言为 False
            assert False

        # 调用 F.get_colrow_absmax 函数，获取 A 的列和行的绝对值最大值以及非零块指针
        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(
            A, threshold=threshold
        )

        # 对 A 进行重新排列，得到 A_blocked
        A_blocked = einops.rearrange(
            torch.abs(A),
            "(rows row_tiles) (cols block_size)-> rows cols row_tiles block_size",
            row_tiles=16,
            block_size=64 * 4,
        )
        # 计算 A_blocked 中大于等于阈值的元素数量
        nnz_rows1_counts = (torch.abs(A_blocked) >= threshold).sum(3).flatten()
        # 创建全零的 nnz_block_ptr1 张量
        nnz_block_ptr1 = torch.zeros(
            nnz_rows1_counts.shape[0] + 1,
            dtype=nnz_rows1_counts.dtype,
            device=nnz_rows1_counts.device,
        )
        # 计算 nnz_rows1_counts 的累积和，存入 nnz_block_ptr1
        nnz_block_ptr1[1:] = nnz_rows1_counts.cumsum(0)

        # 断言 col_stats1_trunc 与 col_stats2 相等
        torch.testing.assert_close(col_stats1_trunc, col_stats2)
        # 断言 row_stats1_trunc 与 row_stats2 相等
        torch.testing.assert_close(row_stats1_trunc, row_stats2)
        # 断言 nnz_block_ptr1 与 nnz_block_ptr2 相等
        torch.testing.assert_close(nnz_block_ptr1.int(), nnz_block_ptr2)

        # 再次调用 F.get_colrow_absmax 函数，获取 A 的列和行的绝对值最大值以及非零块指针
        row_stats2, col_stats2, nnz_block_ptr2 = F.get_colrow_absmax(
            A, threshold=0.0
        )

        # 断言 col_stats1 与 col_stats2 相等
        torch.testing.assert_close(col_stats1, col_stats2)
        # 断言 row_stats1 与 row_stats2 相等
        torch.testing.assert_close(row_stats1, row_stats2)
        # 断言 nnz_block_ptr2 为 None
        assert nnz_block_ptr2 is None

# 使用 pytest.mark.parametrize 注册测试参数，dim1 为 get_test_dims(1, 4 * 1024, n=2) 的结果
@pytest.mark.parametrize("dim1", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim1"))
# 使用 pytest.mark.parametrize 注册测试参数，dim2 为 get_test_dims(1, 4 * 1024, n=2) 的结果
@pytest.mark.parametrize("dim2", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim2"))
# 定义一个测试函数，用于测试双量化操作
def test_double_quant(dim1, dim2):
    # 对于范围在 0 到 k 之间的整数 i
    for i in range(k):
        # 生成一个在 CUDA 设备上的随机张量 A，数据类型为 half
        A = torch.randn(dim1, dim2, device="cuda").half()
        # 对张量 A 进行向量化量化操作，dim=0 表示按行进行量化
        out_col1, Scol = F.vectorwise_quant(A, dim=0)
        # 对张量 A 进行向量化量化操作，dim=1 表示按列进行量化
        out_row1, Srow = F.vectorwise_quant(A, dim=1)

        # 对张量 A 进行双量化操作，返回结果张量 CA、CAt，以及统计信息 statsA、statsAt 和稀疏张量 coo_tensor
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)

        # 断言 CA 与 out_row1 之间的最大差异为 1，由于舍入误差
        torch.testing.assert_close(CA, out_row1, atol=1, rtol=0)
        # 断言 CAt 与 out_col1 之间的最大差异为 1，由于舍入误差
        torch.testing.assert_close(CAt, out_col1, atol=1, rtol=0)

        # 计算 CAt 中元素的数量
        n = CAt.numel()
        # 计算 CA 与 out_row1 中不相近的元素数量
        num_not_close_rows = (
            (torch.isclose(CA, out_row1, atol=1) == 0).sum().item()
        )
        # 计算 CAt 与 out_col1 中不相近的元素数量
        num_not_close_cols = (
            (torch.isclose(CAt, out_col1, atol=1) == 0).sum().item()
        )

        # 允许由于舍入误差导致的 1:500 误差
        min_error = 1 / 500
        if num_not_close_cols > (min_error * n):
            # 如果超过最小误差，则打印错误信息并断言失败
            print(
                f"Min error exceeded {num_not_close_cols} elements are different. Error: {num_not_close_cols/n:.4f}"
            )
            assert False
        if num_not_close_rows > (min_error * n):
            # 如果超过最小误差，则打印错误信息并断言失败
            print(
                f"Min error exceeded {num_not_close_rows} elements are different. Error: {num_not_close_rows/n:.4f}"
            )
            assert False

        # 断言 Srow 展平后的张量与 statsA 之间的接近程度
        torch.testing.assert_close(Srow.flatten().float(), statsA)
        # 断言 Scol 展平后的张量与 statsAt 之间的接近程度


# 使用参数化测试，测试集成的 igemmlt 函数
@pytest.mark.parametrize(
    ("dim1", "dim4", "inner"),
    (
        # 生成参数化测试的参数组合
        pytest.param(dim1, dim4, inner, id=f"{dim1=},{dim4=},{inner=}")
        for (dim1, dim4, inner)
        in zip(
            get_test_dims(1, 4 * 1024, n=4),
            get_test_dims(1, 4 * 1024, n=4),
            get_test_dims(1, 4 * 1024, n=4),
        )
    )
)
def test_integrated_igemmlt(dim1, dim4, inner):
    # 循环 k 次
    for i in range(k):
        # 在 GPU 上生成随机的半精度张量 A 和 B
        A = torch.randn(dim1, inner, device="cuda").half()
        B = torch.randn(dim4, inner, device="cuda").half()

        # 计算 A 和 B 的矩阵乘法
        out1 = torch.matmul(A.half(), B.t().half())

        # 对 A 和 B 进行双精度量化
        C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
        C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
        
        # 对 A 和 B 进行向量化量化
        A1, maxA = F.vectorwise_quant(A, dim=1)
        B1, maxB = F.vectorwise_quant(B, dim=1)

        # 断言量化后的统计数据与最大值
        torch.testing.assert_close(maxA.flatten().float(), stats1a)
        torch.testing.assert_close(maxB.flatten().float(), stats2a)
        torch.testing.assert_close(C1a, A1, rtol=0, atol=1)
        torch.testing.assert_close(C2a, B1, rtol=0, atol=1)

        # 对量化后的 A 和 B 进行 NVIDIA 转换
        A2, SA = F.nvidia_transform(C1a, "col32")
        B2, SB = F.nvidia_transform(C2a, "col_turing")
        
        # 使用 iGEMM 进行矩阵乘法
        outC32, SC = F.igemmlt(A2, B2, SA, SB)
        out2 = F.mm_dequant(outC32, SC, stats1a, stats2a)

        # 再次进行 NVIDIA 转换和 iGEMM 运算
        A2, SA = F.nvidia_transform(A1, "col32")
        B2, SB = F.nvidia_transform(B1, "col_turing")
        C2, SC = F.igemmlt(A2, B2, SA, SB)

        # 对结果进行反向 NVIDIA 转换
        C3, S = F.nvidia_transform(C2, "row", state=SC)
        out3 = F.vectorwise_mm_dequant(C3.float(), maxA, maxB.t())

        # 计算两种方法的误差
        err1 = torch.abs(out1 - out2).mean().item()
        err2 = torch.abs(out1 - out3).mean().item()
        
        # 断言第二种方法的误差小于等于第一种方法误差的1.025倍
        assert err2 <= err1 * 1.025
# 使用 pytest.mark.parametrize 装饰器为 test_igemmlt_row_scale 函数参数化测试用例
@pytest.mark.parametrize(
    ("dim1", "dim4", "inner"),
    (
        # 使用 pytest.param 为参数化的测试用例设置参数和 ID
        pytest.param(dim1, dim4, inner, id=f"{dim1=},{dim4=},{inner=}")
        # 使用 zip 函数将三个维度的测试数据打包成元组
        for (dim1, dim4, inner)
        in zip(
            get_test_dims(1, 4 * 1024, n=6),  # 获取维度为 1 到 4*1024 的 6 个测试数据
            get_test_dims(1, 4 * 1024, n=6),  # 获取维度为 1 到 4*1024 的 6 个测试数据
            get_test_dims(1, 4 * 1024, n=6),  # 获取维度为 1 到 4*1024 的 6 个测试数据
        )
    )
)
# 标记该测试用例为跳过状态，原因是 Row scale 在 ampere 上存在一些 bug
@pytest.mark.skip("Row scale has some bugs for ampere")
def test_igemmlt_row_scale(dim1, dim4, inner):
    formatB = F.get_special_format_str()  # 获取特殊格式字符串
    err1, err2, err3 = [], [], []  # 初始化错误列表
    relerr1, relerr2 = [], []  # 初始化相对错误列表
    scale = 1  # 初始化缩放比例
    print("")  # 打印空行
    print(sum(err1) / len(err1))  # 打印 err1 列表的平均值
    print(sum(err2) / len(err2))  # 打印 err2 列表的平均值
    print(sum(err3) / len(err3))  # 打印 err3 列表的平均值

# 使用 pytest.mark.parametrize 装饰器为 test_row_scale_bench 函数参数化测试用例
@pytest.mark.parametrize(
    ("dim1", "dim4", "inner"),
    [
        # 使用 pytest.param 为参数化的测试用例设置参数和 ID
        pytest.param(1024, 12288 * 4, 12288, id="1024, 12288*4, 12288"),
        pytest.param(2048, 4096 * 4, 4096, id="2048, 4096*4, 4096"),
    ],
)
# 标记该测试用例为跳过状态，原因是 Row scale 在 ampere 上存在一些 bug
@pytest.mark.skip("Row scale has some bugs for ampere")
# 标记该测试用例为性能测试
@pytest.mark.benchmark
def test_row_scale_bench(dim1, dim4, inner):
    formatB = F.get_special_format_str()  # 获取特殊格式字符串
    err1, err2, err3 = [], [], []  # 初始化错误列表
    relerr1, relerr2 = [], []  # 初始化相对错误列表
    scale = 1  # 初始化缩放比例
    A = torch.randn(dim1, inner, device="cuda").half()  # 在 CUDA 设备上生成随机张量 A
    B = torch.randn(dim4, inner, device="cuda").half()  # 在 CUDA 设备上生成随机张量 B
    torch.nn.init.xavier_uniform_(B)  # 使用 Xavier 初始化方法初始化张量 B
    # 预热
    for i in range(k):
        C1 = torch.matmul(A, B.t())

    torch.cuda.synchronize()  # 同步 CUDA 设备
    t0 = time.time()  # 记录当前时间
    # 进行矩阵乘法运算
    for i in range(k):
        C1 = torch.matmul(A, B.t())
    torch.cuda.synchronize()  # 同步 CUDA 设备
    print("16", time.time() - t0)  # 打印时间差

    # 对张量 A 进行双精度量化
    C1a, C1b, stats1a, stats1b, coo_tensor = F.double_quant(A)
    # 对张量 B 进行向量量化
    CB, absmaxB = F.vectorwise_quant(B, quant_type="linear")
    # 对张量 A 进行 NVIDIA 转换
    A2, SA = F.nvidia_transform(C1a, "col32")
    # 对张量 B 进行 NVIDIA 转换
    B2, SB = F.nvidia_transform(CB, formatB)
    # 对张量 A 进行向量量化
    A1, maxA = F.vectorwise_quant(A, dim=1)

    c = 10.0 * inner * scale  # 计算 c 值
    row_scale = maxA / c  # 计算行缩放比例
    torch.cuda.synchronize()  # 同步 CUDA 设备
    t0 = time.time()  # 记录当前时间
    # 对于给定的范围 k，执行下面的操作
    for i in range(k):
        # 调用 F.igemmlt 函数，传入参数 A2, B2, SA, SB, 指定数据类型为 torch.int8，同时传入行缩放因子 row_scale
        outC32, SC = F.igemmlt(
            A2, B2, SA, SB, dtype=torch.int8, row_scale=row_scale
        )
    # 同步 CUDA 设备上的所有流
    torch.cuda.synchronize()
    # 打印 "row-wise" 和计算时间
    print("row-wise", time.time() - t0)

    # 调用 F.double_quant 函数，传入参数 B，返回 C2a, C2b, stats2a, stats2b, coo_tensor
    C2a, C2b, stats2a, stats2b, coo_tensor = F.double_quant(B)
    # 调用 F.nvidia_transform 函数，传入参数 C2a, 指定格式 formatB，返回 B2, SB
    B2, SB = F.nvidia_transform(C2a, formatB)
    # 同步 CUDA 设备上的所有流
    torch.cuda.synchronize()
    # 记录当前时间
    t0 = time.time()
    # 对于给定的范围 k，执行下面的操作
    for i in range(k):
        # 调用 F.igemmlt 函数，传入参数 A2, B2, SA, SB
        outC32, SC = F.igemmlt(A2, B2, SA, SB)
    # 同步 CUDA 设备上的所有流
    torch.cuda.synchronize()
    # 打印 "vector-wise" 和计算时间
    print("vector-wise", time.time() - t0)
# 使用参数化测试，对 dim1 进行测试
@pytest.mark.parametrize("dim1", get_test_dims(2, 1024, n=2), ids=id_formatter("dim1"))
# 使用参数化测试，对 dim2 进行测试
@pytest.mark.parametrize("dim2", get_test_dims(2, 1024, n=2), ids=id_formatter("dim2"))
# 使用参数化测试，对 dim3 进行测试
@pytest.mark.parametrize("dim3", [0], ids=id_formatter("dim3"))
# 使用参数化测试，对 dims 进行测试
@pytest.mark.parametrize("dims", [2], ids=id_formatter("dims"))
# 使用参数化测试，对 dtype 进行测试
@pytest.mark.parametrize("dtype", [torch.int8], ids=describe_dtype)
# 使用参数化测试，对 orderA 进行测试
@pytest.mark.parametrize("orderA", ["row"], ids=id_formatter("orderA"))
# 使用参数化测试，对 orderOut 进行测试
@pytest.mark.parametrize("orderOut", ["col32", "col_turing", "col_ampere"], ids=id_formatter("orderOut"))
# 使用参数化测试，对 transpose 进行测试
@pytest.mark.parametrize("transpose", TRUE_FALSE, ids=id_formatter("transpose"))
def test_transform(dim1, dim2, dim3, dims, dtype, orderA, orderOut, transpose):
    # 循环执行 k 次
    for i in range(k):
        # 根据 dims 的值选择不同维度的随机整数张量 A
        if dims == 2:
            A = torch.randint(10, 99, size=(dim1, dim2), device="cuda").to(
                dtype
            )
        elif dims == 3:
            A = torch.randint(
                10, 99, size=(dim1, dim2, dim3), device="cuda"
            ).to(dtype)

        # 修改张量 A 的最后一个元素为 -1
        A.view(-1)[-1] = -1
        # 根据 transpose 的值选择是否转置张量 A
        if transpose:
            At = A.t().contiguous()
            out1, S1 = F.nvidia_transform(At, to_order=orderOut)
        else:
            out1, S1 = F.nvidia_transform(A, to_order=orderOut)
        out2, S2 = F.transform(A, to_order=orderOut, transpose=transpose)

        # 断言两次转换的结果的第一个元素相等
        assert S1[0][0] == S2[0][0]
        # 断言两次转换的结果的第二个元素相等
        assert S1[0][1] == S2[0][1]
        # 使用 torch.testing.assert_close 函数检查两次转换的结果是否接近

# 测试溢出情况
def test_overflow():
    # 获取特殊格式字符串
    formatB = F.get_special_format_str()
    # 打印特殊格式字符串
    print(formatB)
    # 循环执行两次
    for i in range(2):
        # 创建两个整数张量 a 和 b
        a = torch.arange(5, 15).cuda().to(torch.int8).view(-1, 1)
        b = torch.arange(5, 15).cuda().to(torch.int8).view(-1, 1)

        # 对张量 a 和 b 进行 NVIDIA 转换
        Ca, Sa = F.nvidia_transform(a, "col32")
        Cb, Sb = F.nvidia_transform(b, formatB)

        # 使用 igemmlt 函数对 Ca 和 Cb 进行矩阵乘法
        c = F.igemmlt(Ca, Cb, Sa, Sb, dtype=torch.int8)
        # 使用 torch.matmul 函数对 a 和 b 进行矩阵乘法
        c2 = torch.matmul(a.float(), b.float().t())
# 使用参数化测试，测试维度为1到4*1024之间的两个维度，用于测试COO双精度量化
@pytest.mark.parametrize("dim1", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(1, 4 * 1024, n=2), ids=id_formatter("dim2"))
def test_coo_double_quant(dim1, dim2):
    # 设置阈值为3.00
    threshold = 3.00
    # 对于k次循环
    for i in range(k):
        # 在CUDA设备上生成维度为dim1和dim2的随机张量A，并转换为半精度
        A = torch.randn(dim1, dim2, device="cuda").half()

        # 计算绝对值大于等于阈值的索引
        idx = torch.abs(A) >= threshold
        # 调用double_quant函数对A进行双精度量化
        CA2, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        # 调用double_quant函数对A进行双精度量化，指定阈值为threshold
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(
            A, threshold=threshold
        )

        # 如果coo_tensor不为空
        if coo_tensor is not None:
            # 根据索引A1和COO张量构建A2，并进行张量相等性断言
            A1 = A * idx
            A2 = torch.zeros_like(A)
            A2[
                coo_tensor.rowidx.long(), coo_tensor.colidx.long()
            ] = coo_tensor.values
            torch.testing.assert_close(A1, A2)

            # 根据索引和量化后的张量构建A2，并进行张量相等性断言
            A1 = A * (idx == 0)
            A2 = (CA.float() * statsA.unsqueeze(1) / 127).half()
            torch.testing.assert_close(
                A * (idx == 0), A2, rtol=0.05, atol=1.5e-2
            )

# 使用参数化测试，测试维度为1到1*1024之间的两个维度，以及是否转置B的情况，用于测试COO稀疏矩阵乘法
@pytest.mark.parametrize("dim1", get_test_dims(1, 1 * 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(1, 1 * 1024, n=2), ids=id_formatter("dim2"))
@pytest.mark.parametrize("transposed_B", TRUE_FALSE, ids=id_formatter("transposed_B"))
def test_spmm_coo(dim1, dim2, transposed_B):
    # 设置阈值为1.5
    threshold = 1.5
    # 生成一个随机整数作为dim3
    dim3 = torch.randint(32, 128, size=(1,)).item()
    # dim3 = 17
    # 循环k次，进行下面的操作
    for i in range(k):
        # 生成一个dim1 x dim2大小的随机张量A，并将其转移到GPU上，使用半精度
        A = torch.randn(dim1, dim2).cuda().half()
        # 如果B需要转置
        if transposed_B:
            # 生成一个dim3 x dim2大小的随机张量B，并将其转移到GPU上，使用半精度
            B = torch.randn(dim3, dim2).cuda().half()
        else:
            # 生成一个dim2 x dim3大小的随机张量B，并将其转移到GPU上，使用半精度
            B = torch.randn(dim2, dim3).cuda().half()

        # 找到A中绝对值大于等于阈值的元素的索引
        idx = torch.abs(A) >= threshold
        # 统计非零元素的数量
        nnz = (idx == 1).sum().item()
        # 找到非零元素的行列索引
        rows, cols = torch.where(idx)
        # 找到非零元素的值
        values = A[idx]
        # 使用行列索引和值创建COO格式的稀疏张量cooA
        cooA = F.COOSparseTensor(
            A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values
        )
        # 将A中非零元素位置的值置为0
        A2 = A * idx

        # 如果B需要转置
        if transposed_B:
            # 使用COO格式的稀疏张量cooA和B的转置进行稀疏矩阵乘法，得到out2
            out2 = F.spmm_coo(cooA, B.t())
            # 使用A2和B的转置进行矩阵乘法，得到out1
            out1 = torch.matmul(A2, B.t())
        else:
            # 使用COO格式的稀疏张量cooA和B进行稀疏矩阵乘法，得到out2
            out2 = F.spmm_coo(cooA, B)
            # 使用A2和B进行矩阵乘法，得到out1
            out1 = torch.matmul(A2, B)

        # 断言out1和out2的近似相等性，允许的相对误差为0.01，绝对误差为3.0e-2，重复检查30次
        assert_all_approx_close(out1, out2, rtol=0.01, atol=3.0e-2, count=30)
# 标记为基准测试
@pytest.mark.benchmark
def test_spmm_bench():
    # 定义变量
    batch = 2
    model = 1024 * 1
    hidden = model * 4
    seq = 1024
    dim1 = batch * seq
    dim2 = model
    dim3 = hidden
    threshold = 4
    # 生成随机张量 A 和 B，存储在 GPU 上
    A = torch.randn(dim1, dim2, device="cuda").half()
    B = torch.randn(dim2, dim3, device="cuda").half()
    # 进行矩阵乘法运算
    for i in range(10):
        C1 = bnb.matmul(A, B.t())

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 进行矩阵乘法运算
    for i in range(k):
        C1 = bnb.matmul(A, B.t())
    torch.cuda.synchronize()
    t8 = time.time() - t0

    # 计算绝对值大于等于阈值的元素索引
    idx = torch.abs(A) >= threshold
    # 统计非零元素个数
    nnz = (idx == 1).sum().item()
    print(nnz / idx.numel())
    # 获取非零元素的行列索引和数值
    rows, cols = torch.where(idx)
    values = A[idx]
    # 创建 COO 格式的稀疏张量 cooA
    cooA = F.COOSparseTensor(
        A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values
    )

    # 进行稀疏矩阵乘法运算
    for i in range(10):
        out2 = F.spmm_coo(cooA, B)

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 进行稀疏矩阵乘法运算
    for i in range(k):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    tsp = time.time() - t0
    print(tsp, t8)
    print(tsp / t8)


# 参数化测试，测试维度 dim1 和 dim2
@pytest.mark.parametrize("dim1", get_test_dims(256, 1024, n=2), ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", get_test_dims(256, 1024, n=2), ids=id_formatter("dim2"))
def test_integrated_sparse_decomp(dim1, dim2):
    # 定义阈值和格式化参数 formatB
    threshold = 3.0
    formatB = "col_turing"
    # 循环k次
    for i in range(k):
        # 生成一个dim1 x dim2的随机张量A，并将其移动到GPU上并转换为半精度
        A = torch.randn(dim1, dim2).cuda().half()
        # 生成一个dim1 x dim2的随机张量w1，并将其移动到GPU上并转换为半精度
        w1 = torch.randn(dim1, dim2).cuda().half()
        # 计算A和w1的矩阵乘法，并将结果存储在out1中
        out1 = torch.matmul(A, w1.t())

        # 对w1进行双精度量化
        Cw1, Cw1t, statsw1, statsw1t, coo_tensor = F.double_quant(w1)
        # 对量化后的w1进行格式转换
        CTw1, Sw1 = F.transform(Cw1, formatB)

        # 对A进行双精度量化
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(A)
        # 对量化后的A进行格式转换
        C32A, SA = F.transform(CA, "col32")

        # 使用igemmlt算法进行矩阵乘法计算
        out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1)
        # 对计算结果进行反量化
        out2 = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)

        # 对A进行带有阈值的双精度量化
        CA, CAt, statsA, statsAt, coo_tensor = F.double_quant(
            A, threshold=threshold
        )
        # 对量化后的A进行格式转换
        C32A, SA = F.transform(CA, "col32")

        # 使用igemmlt算法进行矩阵乘法计算
        out1_32, Sout1_32 = F.igemmlt(C32A, CTw1, SA, Sw1)
        # 对计算结果进行反量化
        out3 = F.mm_dequant(out1_32, Sout1_32, statsA, statsw1)

        # 断言coo_tensor不为空
        assert coo_tensor is not None

        # 使用coo_tensor进行稀疏矩阵乘法计算
        out4 = F.spmm_coo(coo_tensor, w1.t())
        # 将out3和out4相加
        out5 = out3 + out4

        # 计算out1和out2之间的平均绝对误差
        err1 = torch.abs(out1 - out2).mean().item()
        # 计算out1和out5之间的平均绝对误差
        err2 = torch.abs(out1 - out5).mean().item()
        # 断言err2小于err1
        assert err2 < err1
# 定义一个测试函数，用于测试矩阵乘法操作
def test_matmuls():
    # 生成随机的半精度浮点数矩阵，并将其移动到 GPU 上
    a = torch.randn(256, 512).half().cuda()
    b = torch.randn(256, 512).half().cuda()
    # 执行矩阵乘法操作
    c1 = torch.matmul(a, b.t())
    c2 = bnb.matmul(a, b)
    c3 = bnb.matmul_cublas(a, b.t())

    # 计算两个结果之间的误差，并取平均值
    err1 = torch.abs(c1 - c2).mean().item()
    err2 = torch.abs(c1 - c3).mean().item()
    # 断言误差小于0.2
    assert err1 < 0.2
    assert err2 < 0.2
    # 打印误差值
    print(err1, err2)


# 使用参数化测试，测试稀疏矩阵乘法操作
@pytest.mark.parametrize("dim1", [1 * 2048], ids=id_formatter("dim1"))
@pytest.mark.parametrize("dim2", [12288], ids=id_formatter("dim2"))
@pytest.mark.parametrize("dtype", [torch.float16], ids=describe_dtype)
@pytest.mark.parametrize("out_func", ["zeros", "ones"], ids=id_formatter("out_func"))
def test_spmm_coo_very_sparse(dim1, dim2, dtype, out_func):
    # 获取指定的输出函数
    out_func = getattr(torch, out_func)

    threshold = 3.3
    # threshold = 2.8
    # threshold = 0.0
    # 生成随机的半精度浮点数矩阵，并将其移动到 GPU 上
    A = torch.randn(dim1, dim2, device="cuda").half()
    if dtype == torch.float16:
        B = torch.randn(dim2, dim2 * 4, device="cuda").half()
        torch.nn.init.xavier_uniform_(B)
    else:
        B = torch.randn(dim2, dim2 * 4, device="cuda").half()
        torch.nn.init.xavier_uniform_(B)
        B, SB = F.vectorwise_quant(B, quant_type="linear")
        # B = torch.randint(-127, 127, size=(dim2, dim2*4), device='cuda').to(torch.int8)

    print("")
    # 根据阈值筛选出稀疏矩阵的非零元素
    idx = torch.abs(A) >= threshold
    nnz = (idx == 1).sum().item()
    rows, cols = torch.where(idx)
    values = A[idx]
    # 创建 COO 格式的稀疏矩阵
    cooA = F.COOSparseTensor(
        A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values
    )
    A2 = A * idx
    # 执行矩阵乘法操作
    out1 = torch.matmul(A2.half(), B.half())
    out = out_func(out1.shape, dtype=torch.float16, device=out1.device)
    out1 += out.clone()
    out2 = F.spmm_coo_very_sparse(cooA, B, out=out)
    # print(B)
    # print(out1)
    # print(out2)
    p = 200 / (2048 * 12288 * 4)
    n = out1.numel()
    count = math.ceil(p * n)
    std = out1.std()
    out1 /= std
    out2 /= std
    # 使用 assert_all_approx_close 函数比较 out1 和 out2.half() 的近似相等性，设置相对误差和绝对误差的阈值
    assert_all_approx_close(
        out1, out2.half(), rtol=0.01, atol=3.0e-2, count=count
    )
    # 使用 torch.testing.assert_close 函数比较 out1 和 out2.half() 的近似相等性，设置相对误差和绝对误差的阈值
    # assert_all_approx_close(out1, out2.half(), rtol=0.05, atol=0.01, count=count)

    # 生成一个包含随机整数的张量，用于索引 A2 的最后一个维度
    idx_col = torch.randint(0, A2.shape[-1], size=(15,))

    # 注释掉的代码块，未被使用，不会执行

    # 注释掉的代码块，未被使用，不会执行

    # 注释掉的代码块，未被使用，不会执行

    # 注释掉的代码块，未被使用，不会执行
# 定义一个测试函数，用于测试 COO 转 CSR 的功能
def test_coo2csr():
    # 设置阈值
    threshold = 1
    # 生成一个随机的半精度张量 A，并将其移动到 GPU 上
    A = torch.randn(128, 128).half().cuda()
    # 根据阈值生成一个索引张量 idx
    idx = torch.abs(A) >= threshold
    # 统计 nnz（非零元素）的数量
    nnz = (idx == 1).sum().item()
    # 获取非零元素的行列索引
    rows, cols = torch.where(idx)
    # 获取非零元素的值
    values = A[idx]
    # 使用行列索引和值创建 COO 格式的稀疏张量 cooA
    cooA = F.COOSparseTensor(
        A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values
    )
    # 将 A 与 idx 相乘得到 A2
    A2 = A * idx
    # 将 COO 格式的稀疏张量 cooA 转换为 CSR 格式的稀疏张量 csrA
    csrA = F.coo2csr(cooA)
    # 计算每行非零元素的数量
    counts = csrA.rowptr[1:] - csrA.rowptr[:-1]
    # 断言每行非零元素的数量等于 A 的行数
    assert counts.numel() == A.shape[0]

    # 使用 torch.testing.assert_close 函数检查 counts 与 (A2 != 0).sum(1) 的一致性
    torch.testing.assert_close(counts.long(), (A2 != 0).sum(1))
    # 更新 idx 为 A2 中不为 0 的元素的索引
    idx = A2 != 0
    # 使用 torch.testing.assert_close 函数检查 A2 中不为 0 的元素与 csrA 的值的一致性


# 定义一个测试函数，用于测试 COO 转 CSC 的功能
def test_coo2csc():
    # 设置阈值
    threshold = 1
    # 生成一个随机的半精度张量 A，并将其移动到 GPU 上
    A = torch.randn(128, 128).half().cuda()
    # 根据阈值生成一个索引张量 idx
    idx = torch.abs(A) >= threshold
    # 统计 nnz（非零元素）的数量
    nnz = (idx == 1).sum().item()
    # 获取非零元素的行列索引
    rows, cols = torch.where(idx)
    # 获取非零元素的值
    values = A[idx]
    # 使用行列索引和值创建 COO 格式的稀疏张量 cooA
    cooA = F.COOSparseTensor(
        A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values
    )
    # 将 A 与 idx 相乘得到 A2
    A2 = A * idx
    # 将 COO 格式的稀疏张量 cooA 转换为 CSC 格式的稀疏张量 cscA
    cscA = F.coo2csc(cooA)
    # 计算每列非零元素的数量
    counts = cscA.colptr[1:] - cscA.colptr[:-1]
    # 断言每列非零元素的数量等于 A 的列数
    assert counts.numel() == A.shape[1]

    # 使用 torch.testing.assert_close 函数检查 counts 与 (A2 != 0).sum(0) 的一致性
    torch.testing.assert_close(counts.long(), (A2 != 0).sum(0)
    # 将 A2 转置后的非零元素索引更新为 idx
    idx = A2.t() != 0
    # 使用 torch.testing.assert_close 函数检查 A2 转置后的非零元素与 cscA 的值的一致性


# 使用 pytest.mark.parametrize 装饰器定义测试参数
@pytest.mark.parametrize("dim1", [1 * 2048])
@pytest.mark.parametrize("dim2", [2048])
@pytest.mark.parametrize("dtype", [torch.int8])
# 定义一个测试函数，用于测试 COO 转 CSR 的功能
def test_spmm_coo_dequant(dim1, dim2, dtype):
    # 设置阈值
    threshold = 6.0
    # 生成一个随机的半精度张量 A，并将其移动到 GPU 上
    A = torch.randn(dim1, dim2, device="cuda").half()
    # 生成一个空的张量 B，并使用 xavier_uniform_ 初始化
    B = torch.empty(dim2, dim2 * 4, device="cuda", dtype=torch.float16)
    torch.nn.init.xavier_uniform_(B)
    # 将 B 转置并连续化
    Bt = B.t().contiguous()

    # 对 B 进行双量化操作，得到 CB, CBt, statsB, statsBt, coo_tensor
    CB, CBt, statsB, statsBt, coo_tensor = F.double_quant(B)

    # 生成一个随机的行索引 rowidx
    rowidx = torch.randint(0, A.shape[-1], size=(15,))

    # 将 A 的指定行设置为 8.0
    A[:, rowidx] = 8.0

    # 根据阈值生成一个索引张量 idx
    idx = torch.abs(A) >= threshold
    # 统计 nnz（非零元素）的数量
    nnz = (idx == 1).sum().item()
    # 获取非零元素的行列索引
    rows, cols = torch.where(idx)
    # 获取非零元素的值
    values = A[idx]
    # 创建一个 COO 格式的稀疏张量，用于存储稀疏矩阵 A 的数据
    cooA = F.COOSparseTensor(
        A.shape[0], A.shape[1], nnz, rows.int(), cols.int(), values
    )
    # 计算 A 与 idx 的乘积
    A2 = A * idx
    # 使用非常稀疏的 COO 格式进行稀疏矩阵乘法操作
    out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
    # 计算 A2 与 B 的矩阵乘法
    out1 = torch.matmul(A2, B.half())
    # 使用非常稀疏的 COO 格式进行稀疏矩阵乘法操作
    out3 = F.spmm_coo_very_sparse(cooA, CBt.half())
    # 对 out3 进行处理
    out3 = out3 * statsBt.half() / 127

    # 找出 cooA.rowidx 中的唯一值和对应的计数
    values, counts = torch.unique(cooA.rowidx, return_counts=True)
    # 计算偏移量
    offset = counts.cumsum(0).int()
    # 找出最大计数和对应的索引
    max_count, max_idx = torch.sort(counts, descending=True)
    # 打印最大计数的中位数
    print(torch.median(max_count.float()))

    # 断言 out2 与 out3 在一定误差范围内相等
    torch.testing.assert_close(out2, out3, rtol=0.05, atol=0.001)

    # 计算 count 的值
    p = 200 / (2048 * 12288 * 4)
    n = out1.numel()
    count = math.ceil(p * n)
    # 断言 out1 与 out2 在一定误差范围内相等
    assert_all_approx_close(out1, out2, rtol=0.01, atol=3.0e-2, count=count)

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 进行 100 次稀疏矩阵乘法操作，并计时
    for i in range(100):
        out2 = F.spmm_coo(cooA, B)
    torch.cuda.synchronize()
    print("cusparse fp16", time.time() - t0)

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 进行 100 次稀疏矩阵乘法操作，并计时
    for i in range(100):
        out2 = F.spmm_coo_very_sparse(cooA, CBt)
    torch.cuda.synchronize()
    print("int8", time.time() - t0)

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 进行 100 次稀疏矩阵乘法操作，并计时
    for i in range(100):
        out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
    torch.cuda.synchronize()
    print("int8+dequant", time.time() - t0)

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 进行 100 次矩阵乘法操作，并计时
    for i in range(100):
        out2 = torch.matmul(A, B)
    torch.cuda.synchronize()
    print("matmul", time.time() - t0)

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 进行 100 次矩阵乘法操作，并计时
    for i in range(100):
        out1 = bnb.matmul(A, Bt)
        out2 = F.spmm_coo_very_sparse(cooA, CBt, dequant_stats=statsBt)
        out = out1 + out2
    torch.cuda.synchronize()
    # 打印"sparse+ matmul"和计算时间差
    print("sparse+ matmul", time.time() - t0)
    
    # 同步CUDA设备，记录当前时间
    torch.cuda.synchronize()
    t0 = time.time()
    # 循环100次
    for i in range(100):
        # 计算A和Bt的矩阵乘积，结果存储在out1中
        out1 = bnb.matmul(A, Bt)
        # 使用torch.matmul计算A的部分行与Bt的部分列的矩阵乘积，结果存储在out1中
        torch.matmul(A[:, rowidx], Bt.t()[rowidx], out=out1)
    # 同步CUDA设备
    torch.cuda.synchronize()
    # 打印"partial matmul"和计算时间差
    print("partial matmul", time.time() - t0)
    
    # 同步CUDA设备
    torch.cuda.synchronize()
    t0 = time.time()
    # 循环100次
    for i in range(100):
        # 计算A和Bt的矩阵乘积，结果存储在out1中
        out1 = bnb.matmul(A, Bt)
    # 同步CUDA设备
    torch.cuda.synchronize()
    # 打印"partial matmul"和计算时间差
    print("partial matmul", time.time() - t0)
# 使用 pytest.mark.parametrize 装饰器为测试用例提供参数化，参数包括 batch, seq, model, hidden
@pytest.mark.parametrize(
    ("batch", "seq", "model", "hidden"),
    [pytest.param(1, 1, 6656, 4*6656, id="batch=1, seq=1, model=6656, hidden=26k")],
)
# 使用 pytest.mark.benchmark 装饰器标记性能测试
@pytest.mark.benchmark
def test_bench_matmul(batch, seq, model, hidden):
    # 设置迭代次数
    iters = 1000
    # 获取特殊格式字符串
    formatB = F.get_special_format_str()

    # 生成随机张量 A，数据类型为半精度浮点数，存储在 GPU 上
    A = torch.randn(batch, seq, model, device="cuda").half()
    # 创建空张量 B，数据类型为半精度浮点数，存储在 GPU 上
    B = torch.empty(hidden, model, dtype=torch.float16, device="cuda")
    # 使用 Xavier 初始化方法初始化张量 B
    torch.nn.init.xavier_uniform_(B)

    # 对张量 B 进行 4 位浮点数量化
    B_fp4, state = F.quantize_fp4(B)
    # 对张量 B 进行 4 位浮点数量化，并压缩统计信息
    B_fp4_c, state_c = F.quantize_fp4(B, compress_statistics=True)

    # 对张量 B 进行非均匀 4 位浮点数量化
    B_nf4, state_nf4 = F.quantize_nf4(B)
    # 对张量 B 进行非均匀 4 位浮点数量化，并压缩统计信息
    B_nf4_c, state_nf4_c = F.quantize_nf4(B, compress_statistics=True)

    # 创建 Linear8bitLt 模型，用于 8 位整数线性变换，存储在 GPU 上
    linear8bit = bnb.nn.Linear8bitLt(model, hidden, False, False).cuda().half()
    # 设置模型为评估模式
    linear8bit.eval()

    # 生成随机异常值索引，存储在 GPU 上
    outliers = torch.randint(0, model, size=(5,)).cuda()
    # 将张量 A 中的异常值索引位置设置为 8.0

    # 创建 Linear8bitLt 模型，用于 8 位整数线性变换，设置阈值为 6.0，存储在 GPU 上
    linearMixedBit = (bnb.nn.Linear8bitLt(model, hidden, False, False, threshold=6.0).cuda().half())
    # 设置模型为评估模式

    # 创建 Linear8bitLt 模型，用于 8 位整数线性变换，存储在 GPU 上
    linear8bit_train = bnb.nn.Linear8bitLt(model, hidden, False).cuda().half()
    # 创建 Linear8bitLt 模型，用于 8 位整数线性变换，设置阈值为 6.0，存储在 GPU 上
    linear8bit_train_thresh = bnb.nn.Linear8bitLt(model, hidden, False, threshold=6.0).cuda().half()
    # 使用 4 位整数矩阵乘法计算 A 和 B_nf4 转置的乘积，使用量化状态 state_nf4

    # 预热
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    print("")

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        torch.matmul(A, B.t())
    torch.cuda.synchronize()
    # 打印 pytorch fp16 矩阵乘法的执行时间

    # 注释部分代码未完整，暂不添加注释
    # 同步 CUDA 设备，确保前面的操作已经完成
    torch.cuda.synchronize()
    # 记录当前时间
    t0 = time.time()
    # 循环执行指定次数的矩阵乘法操作
    for i in range(iters):
        # 使用 bnb.matmul_4bit 函数执行矩阵乘法操作
        bnb.matmul_4bit(A, B_nf4.t(), quant_state=state_nf4)
    # 再次同步 CUDA 设备
    torch.cuda.synchronize()
    # 打印执行时间信息
    print( f"bnb nf4: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )

    # 同步 CUDA 设备
    torch.cuda.synchronize()
    # 记录当前时间
    t0 = time.time()
    # 循环执行指定次数的矩阵乘法操作
    for i in range(iters):
        # 使用 bnb.matmul_4bit 函数执行矩阵乘法操作
        bnb.matmul_4bit(A, B_nf4_c.t(), quant_state=state_nf4_c)
    # 再次同步 CUDA 设备
    torch.cuda.synchronize()
    # 打印执行时间信息
    print( f"bnb nf4+DQ: [{batch},{seq},{model}], [{model},{hidden}]->[{batch},{seq},{hidden}]: {time.time()-t0:.4f}s" )
    # 对 CB 进行 NVIDIA 转换，得到 CxB 和 SB
    # 同步 CUDA 设备
    # 记录当前时间
    # 循环执行指定次数
    # 将 A 重塑为二维数组，保证内存连续性
    # 对 A 进行向量化量化，得到 CA 和 statsA
    # 对 CA 进行 NVIDIA 转换，得到 C32A 和 SA
    # 使用 NVIDIA 进行整数 GEMM 运算，得到 out32 和 Sout32
    # 对 out32 进行 NVIDIA 转换，得到 Cout 和 Sout
    # 对 Cout 进行向量化矩阵乘法的反量化操作
    # 同步 CUDA 设备
    # 打印执行时间信息
    
    # 对 B 进行向量化量化，得到 BA 和 statsB
    # 对 CB 进行 NVIDIA 转换，得到 CxB 和 SB
    # 同步 CUDA 设备
    # 记录当前时间
    # 循环执行指定次数
    # 将 A 重塑为二维数组，保证内存连续性
    # 对 A 进行向量化量化，得到 CA 和 statsA
    # 对 CA 进行 NVIDIA 转换，得到 C32A 和 SA
    # 使用 NVIDIA 进行整数 GEMM 运算，得到 out32 和 Sout32
    # 对 out32 进行 NVIDIA 转换，得到 Cout 和 Sout
    # 对 Cout 进行反量化操作，乘以 statsB 和 statsA，并除以 (127 * 127)
    # 同步 CUDA 设备
    # 打印执行时间信息
    
    # 对 A 进行 8 位线性量化
    # 同步 CUDA 设备
    # 记录当前时间
    # 循环执行指定次数
    # 对 A 进行 8 位线性量化
    # 同步 CUDA 设备
    # 打印执行时间信息
    
    # 对 A 进行混合位宽线性量化
    # 同步 CUDA 设备
    # 记录当前时间
    # 循环执行指定次数
    # 对 A 进行混合位宽线性量化
    # 同步 CUDA 设备
    # 打印执行时间信息
    
    # 对 A 进行 8 位线性量化训练
    # 使用 torch.cuda.synchronize() 来同步 CUDA 设备上的操作
    # 记录当前时间 t0
    # 循环执行 linear8bit_train(A) 函数 iters 次
    # 再次使用 torch.cuda.synchronize() 来同步 CUDA 设备上的操作
    # 打印出线性量化训练的时间信息
    # 调用 linear8bit_train_thresh(A) 函数
    # 再次使用 torch.cuda.synchronize() 来同步 CUDA 设备上的操作
    # 记录当前时间 t0
    # 循环执行 linear8bit_train(A) 函数 iters 次
    # 再次使用 torch.cuda.synchronize() 来同步 CUDA 设备上的操作
    # 打印出带有阈值的线性量化训练的时间信息
# 定义一个测试函数，用于测试零点量化
def test_zeropoint():
    # 定义一个内部函数，用于对输入进行零点量化
    def quant_zp(x):
        # 获取输入的数据类型
        dtype = x.dtype
        # 将输入转换为浮点数类型
        x = x.float()
        # 计算输入数据的动态范围
        dyna = x.max() - x.min()
        # 如果动态范围为0，则设置为1
        if dyna == 0:
            dyna = 1
        # 计算量化因子
        qx = 254.0 / dyna
        # 获取输入数据的最小值
        minx = x.min()
        # 计算零点值
        zpx = torch.round(x.min() * qx) - 127
        # 对输入进行零点量化
        x = (qx * x) + zpx
        return x, qx, zpx

    # 定义一些参数
    batch = 2
    seq = 512
    model = 1024
    hidden = 4 * model
    # 生成随机输入数据 A 和 B
    A = torch.randn(batch * seq, model, device="cuda").half() * 0.1
    B = torch.randn(model, hidden, device="cuda").half() * 0.1

    # 计算矩阵乘积 C0
    C0 = torch.matmul(A, B)

    # 对输入数据 A 和 B 进行零点量化
    A = A.float()
    B = B.float()

    # 计算矩阵乘积 C1
    C1 = torch.matmul(A, B)
    # 使用 bnb.matmul 计算矩阵乘积 C3
    C3 = bnb.matmul(A.half(), B.t().contiguous().half())

    # 设置零点值为1，计算矩阵乘积 C2
    zp = 1
    C2 = torch.matmul(A, B - zp)
    C2 -= A.sum(1).view(-1, 1) * zp

    # 对输入数据 A 进行零点量化
    ca, cqa, cza = quant_zp(A)

    # 设置零点值和缩放因子，计算矩阵乘积 C5
    zp = 1
    scale = 2.0
    C5 = torch.matmul((A * scale) - zp, B)
    C5 += B.sum(0) * zp
    C5 /= scale

    # 对输入数据 A 进行零点量化
    CA, qa, zpa = quant_zp(A)
    # 计算矩阵乘积 C4
    C4 = torch.matmul(CA, B)
    C4 -= B.sum(0) * zpa
    C4 /= qa

    # 设置零点值和量化因子，计算矩阵乘积 C6
    zpb = 1
    zpa = 1
    qa = 2
    qb = 2
    C6 = torch.matmul((A * qa) + zpa, (B * qb) + zpb)
    C6 -= (qb * B.sum(0).view(1, -1) * zpa) + (qa * A.sum(1).view(-1, 1) * zpb)
    C6 -= zpa * zpb * A.shape[1]
    C6 /= qa * qb

    # 对输入数据 A 和 B 进行零点量化
    CA, qa, zpa = quant_zp(A)
    CB, qb, zpb = quant_zp(B)
    # 计算矩阵乘积 C7
    C7 = torch.matmul(CA, CB)
    C7 -= (qb * B.sum(0).view(1, -1) * zpa) + (qa * A.sum(1).view(-1, 1) * zpb)
    C7 -= zpa * zpb * A.shape[1]
    C7 /= qa * qb
    # 计算张量 C1 与 C2 之间的平均绝对误差
    err1 = torch.abs(C1 - C2).mean().item()
    # 计算张量 C1 与 C3 之间的平均绝对误差
    err2 = torch.abs(C1 - C3).mean().item()
    # 计算张量 C1 与 C4 之间的平均绝对误差
    err3 = torch.abs(C1 - C4).mean().item()
    # 计算张量 C1 与 C5 之间的平均绝对误差
    err4 = torch.abs(C1 - C5).mean().item()
    # 计算张量 C1 与 C6 之间的平均绝对误差
    err5 = torch.abs(C1 - C6).mean().item()
    # 计算张量 C1 与 C7 之间的平均绝对误差
    err6 = torch.abs(C1 - C7).mean().item()
    # 打印各个误差值
    print(err1, err2, err3, err4, err5, err6)
# 测试函数，用于提取异常值
def test_extract_outliers():
    # 循环 k 次
    for i in range(k):
        # 定义矩阵形状 shapeA
        shapeA = (4096, 4096 * 4)
        # 生成随机索引 idx，并转移到 GPU 上
        idx = torch.unique(torch.randint(0, shapeA[1], size=(10,)).int()).cuda()
        # 生成随机矩阵 A，并转移到 GPU 上
        A = torch.randint(-128, 127, size=shapeA, device="cuda").to(torch.int8)
        # 提取 A 中指定列的数据，形成 outliers1
        outliers1 = A[:, idx.long()]

        # 对 A 进行列变换，得到 CA 和 SA
        CA, SA = F.transform(A, "col_turing")

        # 提取 CA 和 SA 中指定列的数据，形成 outliers2
        outliers2 = F.extract_outliers(CA, SA, idx)

        # 断言 outliers2 的形状
        assert outliers2.shape[0] == shapeA[0]
        assert outliers2.shape[1] == idx.numel()

        # 断言 outliers1 和 outliers2 的值接近
        torch.testing.assert_close(outliers1, outliers2)

        # 对 A 进行另一种列变换，得到 CA 和 SA
        CA, SA = F.transform(A, "col_ampere")

        # 提取 CA 和 SA 中指定列的数据，形成 outliers2
        outliers2 = F.extract_outliers(CA, SA, idx)

        # 断言 outliers2 的形状
        assert outliers2.shape[0] == shapeA[0]
        assert outliers2.shape[1] == idx.numel()

        # 断言 outliers1 和 outliers2 的值接近
        torch.testing.assert_close(outliers1, outliers2)


# 测试函数，用于在 CPU 上进行大规模块状量化
def test_blockwise_cpu_large():
    # 初始化空列表 diffs 和 reldiffs
    diffs = []
    reldiffs = []
    # 定义 batch 和 seq
    batch = 128
    seq = 128
    # 循环遍历不同的 hidden 和 blocksize
    for hidden in [128]:#, 14336]:
        for blocksize in [4096, 16384]:
            for i in range(2):
                # 生成随机矩阵 A1，并转移到 CPU 上
                A1 = torch.randn(batch, seq, hidden, device='cpu')
                # 计时开始
                t0 = time.time()
                # 对 A1 进行块状量化，得到 C 和 S
                C, S = F.quantize_blockwise(A1, blocksize=blocksize)
                # 对 C 和 S 进行块状反量化，得到 A2
                A2 = F.dequantize_blockwise(C, S, blocksize=blocksize)
                # 打印时间差
                print(time.time() - t0)
                # 计算 A1 和 A2 之间的差异
                diff = torch.abs(A1 - A2)
                # 计算相对差异
                reldiff = diff / torch.abs(A1 + 1e-8)
                # 计算并记录平均差异和相对差异
                diffs.append(diff.mean().item())
                reldiffs.append(reldiff.mean().item())
                # 断言差异小于阈值
                assert diffs[-1] < 0.011
            # 打印平均差异和相对差异
            # print(sum(diffs)/len(diffs))
            # print(sum(reldiffs)/len(reldiffs)


# 测试函数，用于测试 fp8 量化
def test_fp8_quant():
    # 循环遍历从1到6的整数，表示e_bits的取值范围
    for e_bits in range(1, 7):
        # 计算p_bits的值，使得e_bits + p_bits = 7
        p_bits = 7-e_bits
        # 调用F模块的create_fp8_map方法，生成一个CUDA版本的code
        code = F.create_fp8_map(True, e_bits, p_bits).cuda()

        # 初始化用于存储绝对误差和相对误差的列表
        abserr = []
        relerr = []
        # 循环执行100次
        for i in range(100):
            # 生成一个CUDA版本的1024x1024的随机张量A1
            A1 = torch.randn(1024, 1024, device="cuda")
            # 对A1进行分块量化，得到量化后的张量C和量化参数SC
            C, SC = F.quantize_blockwise(A1, code=code)
            # 对量化后的张量C进行反量化，得到A2
            A2 = F.dequantize_blockwise(C, SC)
            # 计算A1和A2之间的绝对差异
            diff = torch.abs(A1 - A2)
            # 计算相对差异
            reldiff = diff/torch.abs(A1+1e-8)
            # 将绝对差异的均值添加到abserr列表中
            abserr.append(diff.mean().item())
            # 将相对差异的均值添加到relerr列表中
            relerr.append(reldiff.mean().item())
            # 断言绝对差异小于0.0075
            #assert diff < 0.0075
        # 打印绝对误差的平均值
        #print(sum(abserr)/len(abserr)
        # 打印相对误差的平均值
        #print(sum(relerr)/len(relerr)

        # 重置绝对误差和相对误差的列表
        abserr = []
        relerr = []
        # 循环执行100次
        for i in range(100):
            # 生成一个CUDA版本的1024x1024的随机张量A1
            A1 = torch.rand(1024, 1024, device="cuda")
            # 对A1进行分块量化，得到量化后的张量C和量化参数SC
            C, SC = F.quantize_blockwise(A1, code=code)
            # 对量化后的张量C进行反量化，得到A2
            A2 = F.dequantize_blockwise(C, SC)
            # 计算A1和A2之间的绝对差异
            diff = torch.abs(A1 - A2)
            # 计算相对差异
            reldiff = diff/torch.abs(A1+1e-8)
            # 将绝对差异的均值添加到abserr列表中
            abserr.append(diff.mean().item())
            # 将相对差异的均值添加到relerr列表中
            relerr.append(reldiff.mean().item())
            # 断言绝对差异小于0.0075
            #assert diff < 0.0075
        # 打印绝对误差的平均值
        #print(sum(abserr)/len(abserr)
        # 打印相对误差的平均值
        #print(sum(relerr)/len(relerr)

        # 重置绝对误差和相对误差的列表
        abserr = []
        relerr = []
        # 循环执行100次
        for i in range(100):
            # 生成一个CUDA版本的1024x1024的随机张量A1
            A1 = torch.randn(1024, 1024, device="cuda")
            # 对A1进行分块量化，得到量化后的张量C和量化参数SC
            C, SC = F.quantize_blockwise(A1)
            # 对量化后的张量C进行反量化，得到A2
            A2 = F.dequantize_blockwise(C, SC)
            # 计算A1和A2之间的绝对差异
            diff = torch.abs(A1 - A2)
            # 计算相对差异
            reldiff = diff/torch.abs(A1+1e-8)
            # 将绝对差异的均值添加到abserr列表中
            abserr.append(diff.mean().item())
            # 将相对差异的均值添加到relerr列表中
            relerr.append(reldiff.mean().item())
            # 断言绝对差异小于0.0075
            #assert diff < 0.0075
        # 打印绝对误差的平均值
        #print(3, sum(abserr)/len(abserr))
        # 打印相对误差的平均值
        #print(3, sum(relerr)/len(relerr)
def test_few_bit_quant():

    # 这是一个空函数，用于测试少量位的量化
    #print('')
    #assert False


def test_kbit_quantile_estimation():
    # 循环100次
    for i in range(100):
        # 生成一个大小为1024x1024的随机张量，存储在CUDA设备上
        data = torch.randn(1024, 1024, device='cuda')
        # 循环遍历2到8位的量化
        for bits in range(2, 9):
            # 生成等间隔的概率值
            p = np.linspace(1.3e-4, 1-1.3e-4, 2**bits)
            # 使用正态分布的逆函数生成张量val1，存储在CUDA设备上
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            # 估计数据的分位数，存储在val2中
            val2 = F.estimate_quantiles(data, offset=0, num_quantiles=2**bits)
            # 计算估计值与真实值之间的绝对误差
            err = torch.abs(val1-val2).mean()
            # 断言误差小于0.038
            assert err < 0.038

    # 再次循环100次
    for i in range(100):
        # 生成一个大小为1024x1024的随机张量，存储在CUDA设备上
        data = torch.randn(1024, 1024, device='cuda')
        # 循环遍历2到4位的量化
        for bits in range(2, 4):
            # 计算总值数量
            total_values = 2**bits-1
            # 生成等间隔的概率值
            p = np.linspace(0, 1, 2*total_values+1)
            # 选择奇数索引
            idx = np.arange(1, 2*total_values+1, 2)
            p = p[idx]
            offset = 1/(2*total_values)
            # 生成等间隔的概率值
            p = np.linspace(offset, 1-offset, total_values)
            # 使用正态分布的逆函数生成张量val1，存储在CUDA设备上
            val1 = torch.Tensor(norm.ppf(p)).cuda()
            # 估计数据的分位数，存储在val2中
            val2 = F.estimate_quantiles(data, num_quantiles=2**bits-1)
            # 计算估计值与真实值之间的绝对误差
            err = torch.abs(val1-val2).mean()
            # 断言误差小于0.035
            assert err < 0.035


@pytest.mark.benchmark
def test_bench_dequantization():
    # 生成一个大小为1024x1024的随机张量，存储在CUDA设备上，数据类型为half
    a = torch.rand(1024, 1024, device='cuda').half()
    # 创建一个FP8映射
    code =F.create_fp8_map(True, 3, 0, 4).cuda()
    # 对张量a进行分块量化
    qa, SA = F.quantize_blockwise(a, code=code)
    # 打印qa的最大值

    max_theoretical_mu =  1024*1024*2/1024**3/672*1000*1000
    #print(max_theoretical_mu)

    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(100):
        qa, SA = F.quantize_blockwise(a)
    torch.cuda.synchronize()
    #print((time.time()-t0)/1e6)



@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=describe_dtype)
def test_fp4_quant(dtype):
    # 生成一个包含0和1的列表，长度为4的笛卡尔积
    vals = list(product([0, 1], repeat=4))

    code = {}
    # 遍历 vals 列表中的每个元素
    for bits in vals:
        # 初始化结果为0
        result = 0
        # 设置偏置值为3
        bias = 3
        # 解析元组中的四个值
        sign, e1, e2, p1 = bits
        # 根据四个值计算索引
        idx = sign*8 + e1*4 + e2*2 + p1*1
        # 根据 sign 值确定符号
        sign = -1.0 if sign else 1.0
        # 计算指数值
        exp = e1*2 + e2*1
        # 如果指数值为0，处理为 sub-normal
        if exp == 0:
            if p1 == 0: result = 0
            else: result = sign*0.0625
        else:
            # 处理为 normal
            exp = 2**(-exp + bias + 1)
            frac = 1.5 if p1 else 1.0
            result = sign*exp*frac
        # 将计算结果存入 code 列表中对应的索引位置
        code[idx] = result

    # 生成一个在 CUDA 设备上的随机张量 A1
    A1 = torch.randn(1024, 1024, device='cuda', dtype=dtype)
    # 对 A1 进行 FP4 格式的量化
    qa, SA = F.quantize_fp4(A1, blocksize=64)
    # 对量化后的结果进行反量化
    A2 = F.dequantize_fp4(qa, SA)

    # 计算 A1 和 A2 之间的绝对误差
    err = (A1 - A2).abs().float()
    # 计算相对误差
    relerr = (err/(A1.abs().float()+1e-8)).mean()
    # 找出绝对误差大于1.0的索引
    idx = err > 1.0
    # 计算绝对误差的平均值
    err = err.mean()

    # 断言 A2 的数据类型为指定的 dtype
    assert A2.dtype == dtype
    # 断言绝对误差小于0.1
    assert err.item() < 0.1
    # 断言相对误差小于0.28
    assert relerr.item() < 0.28
# 使用 pytest 的参数化装饰器，指定测试函数的参数为 'fp4' 和 'nf4'
@pytest.mark.parametrize("quant_type", ['fp4', 'nf4'])
# 定义测试函数，测试4位压缩统计信息
def test_4bit_compressed_stats(quant_type):
    # 遍历不同的块大小
    for blocksize in [128, 64]:
        # 初始化两个错误列表
        errs1 = []
        errs2 = []
        # 循环10次
        for i in range(10):
            # 生成一个随机的半精度张量 A1
            A1 = torch.randn(1024, 1024, device='cuda').half()
            # 对 A1 进行4位量化，得到量化后的张量 q2 和统计信息 SA2
            q2, SA2 = F.quantize_4bit(A1, blocksize=blocksize, quant_type=quant_type)
            # 对 A1 进行4位量化，启用压缩统计信息，得到量化后的张量 q3 和统计信息 SA3
            q3, SA3= F.quantize_4bit(A1, blocksize=blocksize, compress_statistics=True, quant_type=quant_type)
            # 对 q2 进行4位反量化，得到反量化后的张量 A2
            A2 = F.dequantize_4bit(q2, SA2, quant_type=quant_type)
            # 对 q3 进行4位反量化，得到反量化后的张量 A3
            A3 = F.dequantize_4bit(q3, SA3, quant_type=quant_type)

            # 计算 A1 和 A2 之间的绝对误差
            err = (A1 - A2).abs().float()
            # 计算相对误差
            relerr = (err/(A1.abs().float()+1e-15)).mean()
            # 计算平均误差
            err = err.mean()

            # 将平均误差添加到 errs1 列表中
            errs1.append(err.item())

            # 断言平均误差小于0.11
            assert err.item() < 0.11
            # 断言相对误差小于0.28
            assert relerr.item() < 0.28

            # 计算 A1 和 A3 之间的绝对误差
            err = (A1 - A3).abs().float()
            # 计算相对误差
            relerr = (err/(A1.abs().float()+1e-15)).mean()
            # 计算平均误差
            err = err.mean()

            # 将平均误差添加到 errs2 列表中
            errs2.append(err.item())

            # 断言平均误差小于0.11
            assert err.item() < 0.11
            # 断言相对误差小于0.28

        # 打印平均误差和块大小、量化类型
        #print(sum(errs1)/len(errs1), blocksize, quant_type)
        #print(sum(errs2)/len(errs2), blocksize, quant_type)

# 使用 pytest 的参数化装饰器，指定测试函数的参数为 'nf4'
@pytest.mark.parametrize("quant_type", ['nf4'])
# 使用 pytest 的性能测试装饰器
@pytest.mark.benchmark
# 定义性能测试函数，测试4位反量化
def test_bench_4bit_dequant(quant_type):
    # 设置块大小为256
    blocksize = 256
    # 生成一个随机的半精度张量 a
    a = torch.rand(1024*12*4, 1024*12, device='cuda').half()
    # 对 a 进行4位量化，得到量化后的张量 qa 和统计信息 SA
    qa, SA = F.quantize_4bit(a, blocksize=blocksize, quant_type=quant_type)

    # 计算输入大小、输出大小和字节数
    input_size = a.numel()/2
    output_size = a.numel()*2
    num_bytes = input_size+output_size
    GB = num_bytes/1e9
    max_theoretical_s =  GB/768
    #print(max_theoretical_s*1e6)
    # 生成一个随机的半精度张量 b
    b = torch.randn(128, 1024*12, device='cuda').half()

    # 迭代100次
    iters = 100
    # 同步 CUDA 设备
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(iters):
        # 对 qa 进行4位反量化
        F.dequantize_4bit(qa, SA, blocksize=blocksize, quant_type=quant_type)
        #b.copy_(a)
    # 同步 CUDA 设备，确保前面的所有操作都已完成
    torch.cuda.synchronize()
    # 打印注释掉的代码块的执行时间，单位为微秒
    #print((time.time()-t0)/iters*1e6)
    
    # 以下是注释掉的代码块，用于计算矩阵乘法的执行时间
    #torch.cuda.synchronize()
    #t0 = time.time()
    #for i in range(iters):
    #    torch.matmul(b, a.t())
    #torch.cuda.synchronize()
    # 打印矩阵乘法的执行时间，单位为微秒
    #print((time.time()-t0)/iters*1e6)
# 测试普通映射树
def test_normal_map_tree():
    # 创建普通映射
    code = F.create_normal_map()
    # 从映射中取出前8个和后8个值
    values = code[:8].tolist() + code[-8:].tolist()
    # 初始化枢轴数量为1
    num_pivots = 1
    # 循环直到枢轴数量达到16
    while num_pivots < 16:
        # 计算枢轴的索引
        idx = list(range(16 // num_pivots // 2, 16, 16 // num_pivots))
        # 更新枢轴数量
        num_pivots *= 2
        # 初始化枢轴列表
        pivots = []
        # 计算枢轴的值
        for i in idx:
            pivots.append((values[i - 1] + values[i]) / 2)


# 使用参数化测试框架，测试gemv_4bit函数
@pytest.mark.parametrize("double_quant", TRUE_FALSE, ids=lambda double_quant: f"DQ_{double_quant}")
@pytest.mark.parametrize("storage_type", ['nf4', 'fp4'])
@pytest.mark.parametrize("kind", ['fc1', 'fc2', 'attn', 'attn_packed'])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
@pytest.mark.parametrize("quant_storage", [torch.uint8, torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
def test_gemv_4bit(dtype, storage_type, quant_storage, double_quant, kind):
    # 遍历不同的维度
    for dim in [128, 256, 512, 1024]:
        # 标记跳过测试
        @pytest.mark.skip("Row scale has some bugs for ampere")
        def test_managed():
            # 初始化矩阵A，B，B2
            n = 32 * 10
            A = F.get_paged(n, n, dtype=torch.float32)
            B = F.get_paged(n, n, dtype=torch.uint8)
            B2 = F.get_paged(n, n, dtype=torch.float32)
            # 断言A和B是分页的
            assert A.is_paged
            assert B.is_paged
            assert A.page_deviceid == 0
            assert B.page_deviceid == 0
            # 填充A和B的值
            F.fill(A, 17.0)
            F.fill(B, 17)
            F.fill(B2, 2)
            # 断言A和B的值
            assert (A == 17).sum().item() == n * n
            assert (B == 17).sum().item() == n * n
            # 计算矩阵C
            C = A * B.float()
            assert (C == 289).sum().item() == n * n
            # 调用_mul函数
            F._mul(A, B2)
            F._mul(A, B2)
            F._mul(A, B2)
            assert (A == 17 * (2 ** 3)).sum().item() == n * n
            # 预取张量
            # F.prefetch_tensor(A)
            # F.prefetch_tensor(B)
            # F.fill(B2, 17.0)
            # F._mul(A, B2)
            # F.prefetch_tensor(A, to_cpu=True)
            # F.prefetch_tensor(B, to_cpu=True)
            # F.prefetch_tensor(B2, to_cpu=True)
            # torch.cuda.synchronize()
            # assert (A == 17).sum().item() == n * n
            # torch.testing.assert_close(A, torch.ones(A.shape) * 289)
# 使用 pytest.mark.parametrize 装饰器为测试用例 test_gemv_eye_4bit 添加参数化，storage_type 参数取值为 'nf4' 和 'fp4'，对应的标识为 'nf4' 和 'fp4'
# 使用 pytest.mark.parametrize 装饰器为测试用例 test_gemv_eye_4bit 添加参数化，dtype 参数取值为 torch.float16, torch.bfloat16, torch.float32，对应的标识为 describe_dtype
# 使用 pytest.mark.parametrize 装饰器为测试用例 test_gemv_eye_4bit 添加参数化，double_quant 参数取值为 False，对应的标识为 'DQ_True'
def test_gemv_eye_4bit(storage_type, dtype, double_quant):
    # 设置维度为 10
    dims = 10
    # 设置随机种子
    torch.random.manual_seed(np.random.randint(0, 412424242))
    # 获取测试维度，范围为 [0, 8192]，个数为 dims
    dims = get_test_dims(0, 8192, n=dims)
    # 将维度调整为 64 的倍数
    dims = [dim + (64-(dim % 64)) for dim in dims]
    # 遍历 dims 中的维度
    for dim in dims:
        # 生成均值为 0，标准差为 0.1 的正态分布随机数，维度为 (1, 1, dim)，数据类型为 dtype，设备为 'cuda'
        A = torch.normal(0, 0.1, size=(1, 1, dim), dtype=dtype, device='cuda')
        # 生成维度为 dim 的单位矩阵，数据类型为 dtype，设备为 'cuda'
        B = torch.eye(dim, dtype=dtype, device='cuda')

        # 对 B 进行 4 位量化，quant_type 为 storage_type，compress_statistics 为 double_quant
        qB, state = F.quantize_4bit(B, quant_type=storage_type, compress_statistics=double_quant)
        # 计算 A 与 B 转置矩阵的矩阵乘法
        C3 = torch.matmul(A, B.t())
        # 调用 bnb.matmul_4bit 计算 A 与 qB 转置矩阵的矩阵乘法
        C2 = bnb.matmul_4bit(A, qB.t(), state)
        # 设置 A 的 requires_grad 为 True
        A.requires_grad = True
        # 调用 bnb.matmul_4bit 计算 A 与 qB 转置矩阵的矩阵乘法
        C1 = bnb.matmul_4bit(A, qB.t(), state)

        # 断言 A 与 C3 的值接近
        torch.testing.assert_close(A, C3)
        # 断言 A 与 C1 的值接近
        torch.testing.assert_close(A, C1)
        # 断言 A 与 C2 的值接近
        torch.testing.assert_close(A, C2)
        # 断言 A 与 C1 的值接近，相对误差不超过 1e-5，绝对误差不超过 0.00001
        #torch.testing.assert_close(A, C1, rtol=1e-5, atol=0.00001)
        # 断言 A 与 C2 的值接近，相对误差不超过 1e-5，绝对误差不超过 0.080
        #torch.testing.assert_close(A, C2, rtol=1e-5, atol=0.080)
```