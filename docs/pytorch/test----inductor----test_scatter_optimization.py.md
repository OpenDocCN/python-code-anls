# `.\pytorch\test\inductor\test_scatter_optimization.py`

```py
#`
# Owner(s): ["module: inductor"]

import copy  # 导入 copy 模块，提供对象的复制功能
import os  # 导入 os 模块，用于与操作系统交互
import unittest  # 导入 unittest 模块，用于单元测试

import torch  # 导入 torch 模块，PyTorch 的核心库
from torch import nn  # 从 torch 模块导入 nn 子模块，包含神经网络相关功能
from torch._dynamo.utils import counters, same  # 导入 counters 和 same 函数，分别用于计数和比较函数的相等性
from torch._inductor import metrics  # 导入 metrics 模块，包含与性能相关的指标
from torch._inductor.runtime.runtime_utils import do_bench_gpu as do_bench  # 从 runtime_utils 模块导入 do_bench_gpu 函数，并重命名为 do_bench
from torch._inductor.test_case import TestCase  # 从 test_case 模块导入 TestCase 类，作为测试基类
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU  # 从 inductor_utils 模块导入 GPU_TYPE 和 HAS_GPU，用于判断 GPU 是否可用

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"  # 从环境变量获取 DO_PERF_TEST，判断是否进行性能测试

class TestScatterOpt(TestCase):  # 定义 TestScatterOpt 类，继承自 TestCase
    def setUp(self):  # 设置测试环境
        super().setUp()  # 调用基类的 setUp 方法
        metrics.reset()  # 重置 metrics 模块的状态
        counters.clear()  # 清空 counters 的计数

    def check_metric(self, val=1):  # 检查 scatter 操作的计数值
        self.assertEqual(val, metrics.num_matches_for_scatter_upon_const_tensor)  # 断言计数值是否等于 val

    def do_acc_test(self, f, *args):  # 执行准确性测试，比较期望结果与实际结果
        expect = f(*args)  # 调用函数 f 获取期望结果
        actual = torch.compile(f)(*args)  # 使用 torch.compile 编译函数 f，获取实际结果
        self.assertTrue(same(expect, actual, tol=1e-3), f"{expect=}\n{actual=}\n")  # 断言期望结果和实际结果在误差范围内相同

    def test_3d_tensor(self):  # 测试 3D 张量的 scatter 操作
        L, M, N = 2, 1024, 2048  # 定义张量的维度

        def f(x):  # 定义函数 f，执行 scatter 操作
            y = torch.full([L, M, N], 3.14, dtype=torch.float)  # 创建全 3.14 的 3D 张量 y
            y.scatter_(2, x.unsqueeze(2), 2.718)  # 在第 2 维上进行 scatter 操作
            return y  # 返回结果张量 y

        x = torch.randint(0, N, (L, M), dtype=torch.int64)  # 创建随机整数张量 x，范围 [0, N)，维度为 [L, M]
        self.do_acc_test(f, x)  # 执行准确性测试
        expected_num_bytes = (L * M * N * torch.float.itemsize + L * M * torch.int64.itemsize)  # 计算期望的内存访问字节数
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)  # 断言实际内存访问字节数是否等于期望值

    def test_non_last_dim(self):  # 测试非最后一维的 scatter 操作
        """
        Test the case that the scatter dimension is not the last one.
        """
        M, N = 1024, 2048  # 定义张量的维度

        def f(x):  # 定义函数 f，执行 scatter 操作
            y = torch.full([M, N], 3.14, dtype=torch.float)  # 创建全 3.14 的 2D 张量 y
            y.scatter_(0, x.unsqueeze(0), 2.718)  # 在第 0 维上进行 scatter 操作
            return y  # 返回结果张量 y

        x = torch.randint(0, M, (N,), dtype=torch.int64)  # 创建随机整数张量 x，范围 [0, M)，维度为 [N]
        self.do_acc_test(f, x)  # 执行准确性测试
        expected_num_bytes = M * N * torch.float.itemsize + N * torch.int64.itemsize  # 计算期望的内存访问字节数
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)  # 断言实际内存访问字节数是否等于期望值

    def test_neg_scatter_dim(self):  # 测试负数索引的 scatter 操作
        M, N = 1024, 2048  # 定义张量的维度

        def f(x):  # 定义函数 f，执行 scatter 操作
            y = torch.full([M, N], 3.14, dtype=torch.float)  # 创建全 3.14 的 2D 张量 y
            y.scatter_(-1, x.unsqueeze(1), 2.718)  # 在负数索引的维度进行 scatter 操作
            return y  # 返回结果张量 y

        x = torch.randint(0, N, (M,), dtype=torch.int64)  # 创建随机整数张量 x，范围 [0, N)，维度为 [M]
        self.do_acc_test(f, x)  # 执行准确性测试
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize  # 计算期望的内存访问字节数
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes)  # 断言实际内存访问字节数是否等于期望值

    def test_shorter_index_tensor(self):  # 测试索引张量长度不足的 scatter 操作
        M, N = 1024, 2048  # 定义张量的维度

        def f(x):  # 定义函数 f，执行 scatter 操作
            y = torch.full([M, N], 3.14, dtype=torch.float)  # 创建全 3.14 的 2D 张量 y
            y.scatter_(1, x.unsqueeze(1), 2.718)  # 在第 1 维上进行 scatter 操作
            return y  # 返回结果张量 y

        x = torch.randint(0, N, (M // 2,), dtype=torch.int64)  # 创建随机整数张量 x，范围 [0, N)，维度为 [M // 2]
        self.do_acc_test(f, x)  # 执行准确性测试

        # 断言计数器中的模式匹配次数是否为 0，因为索引张量长度不足
        self.assertEqual(0, counters["inductor"]["pattern_matcher_count"])
    # 测试一个非零常数张量的情况
    def test_nonzero_const_tensor(self):
        # 定义常量
        M, N = 1024, 2048

        # 定义函数 f，接受参数 x
        def f(x):
            # 创建一个大小为 MxN 的张量，所有元素值为 3.14，数据类型为浮点数
            y = torch.full([M, N], 3.14, dtype=torch.float)
            # 在张量 y 的第一维上根据索引 x，将元素设为 2.718
            y.scatter_(1, x.unsqueeze(1), 2.718)
            return y

        # 创建一个大小为 M 的长整型张量 x，元素值在 [0, N) 之间随机取值
        x = torch.randint(0, N, (M,), dtype=torch.int64)
        # 执行累积测试 do_acc_test，并传入函数 f 和参数 x
        self.do_acc_test(f, x)
        # 预期访问的字节数，包括张量 y 和索引 x 的内存消耗
        expected_num_bytes = M * N * torch.float.itemsize + M * torch.int64.itemsize
        # 断言 metrics.num_bytes_accessed 大于等于预期的字节数
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

    # 测试由于稠密矩阵而无法进行优化的情况
    def test_can_not_optimize_due_to_dense(self):
        # 定义常量
        M, N = 1024, 2048

        # 定义函数 f，接受参数 x
        def f(x):
            # 创建一个大小为 MxN 的张量，所有元素值为 0，数据类型为浮点数
            y = torch.full([M, N], 0, dtype=torch.float)
            # 在张量 y 的第一维上根据索引 x，将元素设为 0.618
            y.scatter_(1, x, 0.618)
            return y

        # 创建一个大小为 Mx(N/2) 的长整型张量 x，元素值在 [0, N) 之间随机取值
        x = torch.randint(0, N, (M, N // 2), dtype=torch.int64)
        # 执行累积测试 do_acc_test，并传入函数 f 和参数 x
        self.do_acc_test(f, x)
        # 预期访问的字节数，包括张量 y 和索引 x 的内存消耗
        expected_num_bytes = M * N * torch.float.itemsize + M * (N // 2) * (
            torch.int64.itemsize + torch.float.itemsize
        )
        # 使用 assertGreaterEqual 检查 metrics.num_bytes_accessed 是否大于等于预期的字节数
        # 由于 StarDep 的问题，这里使用 assertGreaterEqual 而不是 assertEqual
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

    # 测试由于非常数张量而无法进行优化的情况
    def test_can_not_optimize_due_to_non_const(self):
        # 定义常量
        M, N = 1024, 2048

        # 定义函数 f，接受参数 x 和 y
        def f(x, y):
            # 在张量 y 的第一维上根据索引 x，将元素设为 0.618
            y.scatter_(1, x, 0.618)
            return y

        # 创建一个大小为 Mx1 的长整型张量 x，元素值在 [0, N) 之间随机取值
        x = torch.randint(0, N, (M, 1), dtype=torch.int64)
        # 创建一个大小为 MxN 的浮点数张量 y，元素值为随机生成的标准正态分布值
        y = torch.randn([M, N])
        # 执行累积测试 do_acc_test，并传入函数 f、参数 x 和 y
        self.do_acc_test(f, x, y)

        # 预期访问的字节数，包括张量 y 和索引 x 的内存消耗
        expected_num_bytes = 4 * M * N * torch.float.itemsize + M * (
            torch.int64.itemsize + torch.float.itemsize
        )
        # 使用 assertGreaterEqual 检查 metrics.num_bytes_accessed 是否大于等于预期的字节数
        self.assertGreaterEqual(metrics.num_bytes_accessed, expected_num_bytes)

        # 第二个和第三个内核都是变异内核。因此，我们对内存访问进行了过估计。
        # 一旦过估计问题得到修复，更新测试。
        over_estimate = M * torch.float.itemsize + M * N * torch.float.itemsize
        # 断言 metrics.num_bytes_accessed 等于预期的字节数加上过估计的字节数
        self.assertEqual(metrics.num_bytes_accessed, expected_num_bytes + over_estimate)
    def test_cross_entropy_loss(self):
        """
        Match full+scatter in CEL and replaces it with a pointwise.

        Perf data on an A100 GPU:
        Without the scatter optimization:
          ms=47.340, peak_mem=10.524 GB
        With the scatter optimization:
          ms=42.768, peak_mem=7.227 GB
        """
        # 设置测试用的批量大小（B）、时间步（T）、特征维度（D）、词汇量大小（V）
        B, T, D, V = 32, 1024, 768, 50257
        # 如果不进行性能测试，为了避免在 CI 中出现内存溢出，将 V 缩小为原来的 100 倍
        if not DO_PERF_TEST:
            V = V // 100
        # 创建一个参考模型，是一个线性层，输入特征维度 D，输出特征维度 V，使用 bfloat16 类型
        ref_model = nn.Linear(D, V).to(torch.bfloat16)
        # 使用深拷贝复制参考模型以创建优化模型
        opt_model = copy.deepcopy(ref_model)
        # 创建交叉熵损失函数对象
        ce = nn.CrossEntropyLoss()

        # 定义一个函数 f，接收模型 m、输入数据 x 和标签 label，计算损失并进行反向传播
        def f(m, x, label):
            ce(m(x).view(-1, V), label.view(-1)).backward()

        # 使用 torch.compile 对函数 f 进行优化编译
        opt_f = torch.compile(f)

        # 生成随机输入数据 x，使用 bfloat16 类型
        x = torch.randn(B, T, D).to(torch.bfloat16)
        # 生成随机标签数据 label，使用 int64 类型
        label = torch.randint(0, V, (B, T)).to(torch.int64)

        # 在参考模型上执行前向传播、计算损失并进行反向传播，获得参考模型的梯度
        f(ref_model, x, label)
        ref_grad = ref_model.weight.grad
        # 在优化模型上执行优化编译后的函数，进行前向传播、计算损失并进行反向传播，获得优化模型的梯度
        opt_f(opt_model, x, label)
        act_grad = opt_model.weight.grad
        # 断言参考模型的梯度和优化模型的梯度在一定的数值容差内相等
        assert torch.allclose(
            ref_grad, act_grad, atol=1e-3, rtol=1e-3
        ), f"{ref_grad=}\n{act_grad=}"

        # 检查额外的度量指标
        self.check_metric()

        # 如果进行性能测试
        if DO_PERF_TEST:
            # 如果 GPU 类型为 "xpu"，则跳过测试
            if GPU_TYPE == "xpu":
                raise unittest.SkipTest(
                    "torch.xpu.reset_peak_memory_stats not implemented."
                )
            # 重置 CUDA 内存统计的峰值
            torch.cuda.reset_peak_memory_stats()
            # 多次执行优化编译后的函数，测量平均执行时间和内存峰值
            for _ in range(3):
                opt_f(opt_model, x, label)
            ms = do_bench(lambda: opt_f(opt_model, x, label))
            peak_mem = torch.cuda.max_memory_allocated() / 10**9
            print(f"{ms=:.3f}, {peak_mem=:.3f} GB")
# 如果系统中有 GPU 可用，则将默认设备设置为指定的 GPU 类型
if HAS_GPU:
    torch.set_default_device(GPU_TYPE)

# 如果这个脚本是作为主程序被执行
if __name__ == "__main__":
    # 从 torch 库中导入测试用例运行函数
    from torch._inductor.test_case import run_tests

    # 如果系统中有 GPU 可用
    if HAS_GPU:
        # 运行 GPU 相关的测试用例
        run_tests()
```