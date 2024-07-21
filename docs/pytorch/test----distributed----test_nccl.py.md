# `.\pytorch\test\distributed\test_nccl.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块
import re  # 导入正则表达式模块
import sys  # 导入系统相关模块

import torch  # 导入PyTorch主模块
import torch.cuda  # 导入PyTorch CUDA模块
import torch.cuda.nccl as nccl  # 导入PyTorch NCCL模块
import torch.distributed as c10d  # 导入PyTorch分布式模块
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU  # 导入测试相关模块
from torch.testing._internal.common_device_type import (  # 导入设备类型相关模块
    dtypes,
    instantiate_device_type_tests,
)

from torch.testing._internal.common_utils import (  # 导入通用测试相关工具
    IS_WINDOWS,
    load_tests,
    NoTest,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
    TestCase,
)

HIP_VERSION = (
    0.0  # 设置HIP版本号初始值为0.0
    if torch.version.hip is None
    else float(re.search(r"^\d+\.\d+", torch.version.hip)[0])  # 获取HIP版本号
)

# load_tests用于在sandcastle上自动过滤测试，以下代码行抑制flake警告
load_tests = load_tests

nGPUs = torch.cuda.device_count()  # 获取CUDA可用的GPU数量
if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)  # 如果CUDA不可用，输出错误信息
    TestCase = NoTest  # 如果没有CUDA测试，则使用NoTest类进行标记

datatypes = [torch.float]  # 初始化数据类型列表为torch.float
if (
    TEST_CUDA and c10d.is_nccl_available() and nccl.version() >= (2, 10)
) or TEST_WITH_ROCM:
    datatypes.append(torch.bfloat16)  # 如果满足条件，添加torch.bfloat16数据类型到列表中


class TestNCCL(TestCase):
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    def test_unique_id(self, device):
        uid = nccl.unique_id()  # 调用nccl模块生成唯一ID
        self.assertIsInstance(uid, bytes)  # 断言uid的类型为bytes
        self.assertGreater(len(uid), 1)  # 断言uid长度大于1

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM and HIP_VERSION < 3.5, "Skip NCCL tests for ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_broadcast(self, device, dtype):
        expected = torch.zeros(128).uniform_().to(dtype=dtype)  # 创建期望的tensor
        tensors = [expected.cuda()]  # 将期望的tensor放到第一个GPU上
        for device in range(1, torch.cuda.device_count()):  # 遍历其他GPU
            tensors.append(torch.zeros(128, dtype=dtype, device=device))  # 创建其他GPU上的tensor

        nccl.broadcast(tensors)  # 调用nccl广播函数
        for i in range(torch.cuda.device_count()):  # 遍历每个GPU
            self.assertEqual(tensors[i], expected)  # 断言每个GPU上的tensor与期望的tensor相等

        # 使用tuple进行测试
        tensors = [expected.cuda()]  # 将期望的tensor放到第一个GPU上
        for device in range(1, torch.cuda.device_count()):  # 遍历其他GPU
            tensors.append(torch.zeros(128, dtype=dtype, device=device))  # 创建其他GPU上的tensor

        nccl.broadcast(tuple(tensors))  # 调用nccl广播函数
        for i in range(torch.cuda.device_count()):  # 遍历每个GPU
            self.assertEqual(tensors[i], expected)  # 断言每个GPU上的tensor与期望的tensor相等

    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM and HIP_VERSION < 3.5, "Skip NCCL tests for ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    # 定义一个测试方法，用于测试 reduce 方法
    def test_reduce(self, device, dtype):
        # 创建多个 CPU 上的张量列表，每个张量为长度为 128 的随机张量，转移到指定数据类型上
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        # 创建一个期望的张量，长度为 128，数据类型为指定类型，初始值为零张量
        expected = torch.zeros(128, dtype=dtype)
        # 将每个 CPU 张量加到期望张量上
        for t in cpu_tensors:
            expected.add_(t)

        # 将每个 CPU 张量转移到对应 GPU 上，并创建张量列表
        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        # 在 NCCL 上执行 reduce 操作
        nccl.reduce(tensors)

        # 断言第一个 GPU 上的张量与期望的张量相等
        self.assertEqual(tensors[0], expected)

        # 使用元组形式测试 reduce 方法
        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        nccl.reduce(tuple(tensors))

        # 再次断言第一个 GPU 上的张量与期望的张量相等
        self.assertEqual(tensors[0], expected)

    # 添加条件装饰器，根据操作系统和 GPU 数量决定是否跳过测试
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM and HIP_VERSION < 3.5 and dtype == torch.bfloat16,  # noqa: F821
        "Skip bfloat16 test for ROCm < 3.5",
    )
    @dtypes(*datatypes)
    # 定义一个测试方法，用于测试 all_reduce 方法
    def test_all_reduce(self, device, dtype):
        # 创建多个 CPU 上的张量列表，每个张量为长度为 128 的随机张量，转移到指定数据类型上
        cpu_tensors = [
            torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        # 创建一个期望的张量，长度为 128，数据类型为指定类型，初始值为零张量
        expected = torch.zeros(128, dtype=dtype)
        # 将每个 CPU 张量加到期望张量上
        for t in cpu_tensors:
            expected.add_(t)

        # 将每个 CPU 张量转移到对应 GPU 上，并创建张量列表
        tensors = [cpu_tensors[i].cuda(i) for i in range(nGPUs)]
        # 在 NCCL 上执行 all_reduce 操作
        nccl.all_reduce(tensors)

        # 断言每个 GPU 上的张量与期望的张量相等
        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # 使用元组形式测试 all_reduce 方法
        tensors = tuple(cpu_tensors[i].cuda(i) for i in range(nGPUs))
        nccl.all_reduce(tensors)

        # 再次断言每个 GPU 上的张量与期望的张量相等
        for tensor in tensors:
            self.assertEqual(tensor, expected)

        # 使用集合形式测试 all_reduce 方法
        tensors = {cpu_tensors[i].cuda(i) for i in range(nGPUs)}
        nccl.all_reduce(tensors)

        # 再次断言每个 GPU 上的张量与期望的张量相等
        for tensor in tensors:
            self.assertEqual(tensor, expected)

    # 添加条件装饰器，根据操作系统和 ROCm 版本决定是否跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM and HIP_VERSION < 3.5, "Skip NCCL tests for ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    # 定义一个测试方法，用于测试 collective errors
    def test_collective_errors(self, device):
        # 创建一个长度为 10 的随机张量，并将其转移到第一个 GPU 上
        t = torch.rand(10).cuda(0)
        # 测试 all_reduce 方法是否抛出预期的 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_reduce(t)

        # 测试 reduce 方法是否抛出预期的 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce(t)

        # 测试 broadcast 方法是否抛出预期的 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.broadcast(t)

        # 测试 all_gather 方法是否抛出预期的 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.all_gather(t, t)

        # 测试 reduce_scatter 方法是否抛出预期的 TypeError 异常
        with self.assertRaisesRegex(
            TypeError, "Inputs should be a collection of tensors"
        ):
            nccl.reduce_scatter(t, t)

    # 添加条件装饰器，根据 ROCm 版本决定是否跳过测试
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM and HIP_VERSION < 3.5, "Skip NCCL tests for ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    # 在满足条件时跳过，否则在沙堡中继续运行测试。条件为 IS_WINDOWS 为真，说明当前环境是 Windows，NCCL 不支持 Windows。
    # 同时，如果 TEST_MULTIGPU 为假，表明只检测到一个 GPU，因此测试也会被跳过。
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_all_gather(self, device, dtype):
        # 创建 nGPUs 个大小为 128 的零张量列表，并转换为指定数据类型 dtype，并移至设备 device
        cpu_inputs = [torch.zeros(128).uniform_().to(dtype=dtype) for i in range(nGPUs)]
        # 将上述 cpu_inputs 中的张量连接成一个期望的张量
        expected = torch.cat(cpu_inputs, 0)
    
        # 将 cpu_inputs 中的张量移至对应的 GPU，并创建 nGPUs 个设备为 i、数据类型为 dtype 的零张量列表
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        # 创建 nGPUs 个设备为 i、数据类型为 dtype、大小为 128 * nGPUs 的零张量列表
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        # 使用 NCCL 库的 all_gather 方法，将 inputs 中的数据聚集到 outputs 中
        nccl.all_gather(inputs, outputs)
    
        # 检查每个输出张量是否与期望的张量 expected 相等
        for tensor in outputs:
            self.assertEqual(tensor, expected)
    
        # 测试元组输入情况
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [
            torch.zeros(128 * nGPUs, device=i, dtype=dtype) for i in range(nGPUs)
        ]
        nccl.all_gather(tuple(inputs), tuple(outputs))
    
        # 再次检查每个输出张量是否与期望的张量 expected 相等
        for tensor in outputs:
            self.assertEqual(tensor, expected)
    
    # 在满足条件时跳过，否则在沙堡中继续运行测试。条件为 TEST_WITH_ROCM 为真且 HIP_VERSION 小于 3.5，表明当前环境是 ROCm 且版本低于 3.5，
    # NCCL 测试会被跳过。另外，如果 IS_WINDOWS 为真，说明当前环境是 Windows，NCCL 不支持 Windows。同时，如果 TEST_MULTIGPU 为假，
    # 表明只检测到一个 GPU，因此测试也会被跳过。
    @skip_but_pass_in_sandcastle_if(
        TEST_WITH_ROCM and HIP_VERSION < 3.5, "Skip NCCL tests for ROCm"
    )
    @skip_but_pass_in_sandcastle_if(IS_WINDOWS, "NCCL doesn't support Windows")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "only one GPU detected")
    @dtypes(*datatypes)
    def test_reduce_scatter(self, device, dtype):
        # 输入张量的大小为 32 * nGPUs
        in_size = 32 * nGPUs
        # 输出张量的大小为 32
        out_size = 32
    
        # 创建 nGPUs 个大小为 in_size 的零张量列表，并转换为指定数据类型 dtype
        cpu_inputs = [
            torch.zeros(in_size).uniform_().to(dtype=dtype) for i in range(nGPUs)
        ]
        # 创建期望的张量，将上述 cpu_inputs 中的张量相加，并按视图重塑为 nGPUs 行，32 列的张量
        expected = torch.zeros(in_size, dtype=dtype)
        for t in cpu_inputs:
            expected.add_(t)
        expected = expected.view(nGPUs, 32)
    
        # 将 cpu_inputs 中的张量移至对应的 GPU，并创建 nGPUs 个设备为 i、数据类型为 dtype 的零张量列表
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        # 创建 nGPUs 个设备为 i、数据类型为 dtype、大小为 out_size 的零张量列表
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        # 使用 NCCL 库的 reduce_scatter 方法，将 inputs 中的数据按列相加，并分散到 outputs 中
        nccl.reduce_scatter(inputs, outputs)
    
        # 检查每个输出张量是否与期望的张量 expected 相等
        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])
    
        # 测试元组输入情况
        inputs = [cpu_inputs[i].cuda(i) for i in range(nGPUs)]
        outputs = [torch.zeros(out_size, device=i, dtype=dtype) for i in range(nGPUs)]
        nccl.reduce_scatter(tuple(inputs), tuple(outputs))
    
        # 再次检查每个输出张量是否与期望的张量 expected 相等
        for i in range(nGPUs):
            self.assertEqual(outputs[i], expected[i])
# 根据给定的测试类 TestNCCL 和全局变量，实例化设备类型测试
instantiate_device_type_tests(TestNCCL, globals(), only_for="cuda")

# 如果当前脚本被直接运行，则执行测试
if __name__ == "__main__":
    run_tests()
```