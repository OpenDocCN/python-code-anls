# `.\pytorch\test\test_indexing.py`

```py
# Owner(s): ["module: tests"]

import operator  # 导入 operator 模块，用于操作符的函数
import random  # 导入 random 模块，用于生成随机数

import unittest  # 导入 unittest 模块，用于编写和运行单元测试
import warnings  # 导入 warnings 模块，用于警告控制
from functools import reduce  # 从 functools 模块导入 reduce 函数

import numpy as np  # 导入 NumPy 库，用于数值计算

import torch  # 导入 PyTorch 库
from torch import tensor  # 从 torch 模块导入 tensor 函数

from torch.testing import make_tensor  # 从 torch.testing 模块导入 make_tensor 函数
from torch.testing._internal.common_device_type import (
    dtypes,  # 导入 dtypes 类型列表，包括所有设备类型
    dtypesIfCPU,  # 导入仅在 CPU 上使用的 dtypes 类型列表
    dtypesIfCUDA,  # 导入仅在 CUDA 上使用的 dtypes 类型列表
    instantiate_device_type_tests,  # 导入用于实例化设备类型测试的函数
    onlyCUDA,  # 导入仅在 CUDA 上运行的装饰器
    onlyNativeDeviceTypes,  # 导入仅在本地设备类型上运行的装饰器
    skipXLA,  # 导入用于跳过 XLA 的装饰器
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,  # 导入用于确定性控制的类
    run_tests,  # 导入运行测试的函数
    serialTest,  # 导入用于串行测试的装饰器
    skipIfTorchDynamo,  # 导入用于在 Torch Dynamo 下跳过测试的装饰器
    TEST_CUDA,  # 导入 CUDA 测试标志
    TestCase,  # 导入测试用例基类
    xfailIfTorchDynamo,  # 导入在 Torch Dynamo 下失败测试的装饰器
)


class TestIndexing(TestCase):
    @onlyNativeDeviceTypes  # 装饰器：仅在本地设备类型上运行测试
    @dtypes(torch.half, torch.double)  # 装饰器：指定测试使用的数据类型为 torch.half 和 torch.double
    def test_advancedindex_big(self, device):
        reference = torch.arange(0, 123344, dtype=torch.int, device=device)

        self.assertEqual(
            reference[[0, 123, 44488, 68807, 123343],],
            torch.tensor([0, 123, 44488, 68807, 123343], dtype=torch.int),
        )

    def test_set_item_to_scalar_tensor(self, device):
        m = random.randint(1, 10)  # 生成随机整数 m
        n = random.randint(1, 10)  # 生成随机整数 n
        z = torch.randn([m, n], device=device)  # 在指定设备上生成大小为 [m, n] 的随机张量 z
        a = 1.0
        w = torch.tensor(a, requires_grad=True, device=device)  # 创建一个需要梯度的标量张量 w
        z[:, 0] = w  # 将张量 w 赋值给张量 z 的所有行的第一列
        z.sum().backward()  # 对张量 z 的和进行反向传播
        self.assertEqual(w.grad, m * a)  # 断言 w 的梯度为 m * a

    def test_single_int(self, device):
        v = torch.randn(5, 7, 3, device=device)  # 在指定设备上生成大小为 [5, 7, 3] 的随机张量 v
        self.assertEqual(v[4].shape, (7, 3))  # 断言取出 v 的第 5 个元素的形状为 (7, 3)

    def test_multiple_int(self, device):
        v = torch.randn(5, 7, 3, device=device)  # 在指定设备上生成大小为 [5, 7, 3] 的随机张量 v
        self.assertEqual(v[4].shape, (7, 3))  # 断言取出 v 的第 5 个元素的形状为 (7, 3)
        self.assertEqual(v[4, :, 1].shape, (7,))  # 断言取出 v 的第 5 个元素的所有行的第 2 列的形状为 (7,)

    def test_none(self, device):
        v = torch.randn(5, 7, 3, device=device)  # 在指定设备上生成大小为 [5, 7, 3] 的随机张量 v
        self.assertEqual(v[None].shape, (1, 5, 7, 3))  # 断言在 v 上增加一个维度后的形状为 (1, 5, 7, 3)
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))  # 断言在 v 上增加两个维度后的形状为 (5, 1, 7, 3)
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))  # 断言在 v 上增加三个维度后的形状为 (5, 1, 1, 7, 3)
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))  # 断言在 v 上在末尾增加一个维度后的形状为 (5, 7, 3, 1)

    def test_step(self, device):
        v = torch.arange(10, device=device)  # 在指定设备上生成从 0 到 9 的张量 v
        self.assertEqual(v[::1], v)  # 断言取 v 的所有元素（步长为 1）与 v 相等
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])  # 断言取 v 的步长为 2 的所有元素组成的列表与给定列表相等
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])  # 断言取 v 的步长为 3 的所有元素组成的列表与给定列表相等
        self.assertEqual(v[::11].tolist(), [0])  # 断言取 v 的步长为 11 的所有元素组成的列表与给定列表相等
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])  # 断言取 v 的索引从 1 到 6，步长为 2 的所有元素组成的列表与给定列表相等

    def test_step_assignment(self, device):
        v = torch.zeros(4, 4, device=device)  # 在指定设备上生成大小为 [4, 4] 的零张量 v
        v[0, 1::2] = torch.tensor([3.0, 4.0], device=device)  # 将张量 [3.0, 4.0] 赋值给 v 的第一行的第 2 列及其后的偶数列
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])  # 断言取 v 的第一行的所有元素组成的列表与给定列表相等
        self.assertEqual(v[1:].sum(), 0)  # 断言取 v 的第二行及其后的所有元素的和为 0
    # 测试使用布尔索引选择张量的子集，要求与预期形状和数值相等
    def test_bool_indices(self, device):
        # 创建一个形状为 (5, 7, 3) 的随机张量 v
        v = torch.randn(5, 7, 3, device=device)
        # 创建一个布尔张量 boolIndices，指定哪些索引为 True
        boolIndices = torch.tensor(
            [True, False, True, True, False], dtype=torch.bool, device=device
        )
        # 断言选择的子集张量形状与预期相等
        self.assertEqual(v[boolIndices].shape, (3, 7, 3))
        # 断言选择的子集张量与手动堆叠的张量在数值上相等
        self.assertEqual(v[boolIndices], torch.stack([v[0], v[2], v[3]]))

        # 创建一个布尔张量 v
        v = torch.tensor([True, False, True], dtype=torch.bool, device=device)
        # 创建一个布尔张量 boolIndices 和一个 uint8 张量 uint8Indices
        boolIndices = torch.tensor(
            [True, False, False], dtype=torch.bool, device=device
        )
        uint8Indices = torch.tensor([1, 0, 0], dtype=torch.uint8, device=device)
        # 使用警告记录检测异常情况
        with warnings.catch_warnings(record=True) as w:
            # 分别使用布尔索引和 uint8 索引选择张量的子集，并比较它们的形状
            v1 = v[boolIndices]
            v2 = v[uint8Indices]
            self.assertEqual(v1.shape, v2.shape)
            # 断言选择的子集张量在数值上相等
            self.assertEqual(v1, v2)
            # 断言布尔索引选择的子集张量的值
            self.assertEqual(
                v[boolIndices], tensor([True], dtype=torch.bool, device=device)
            )
            # 断言警告的数量
            self.assertEqual(len(w), 1)

    # 测试使用布尔索引累加操作
    def test_bool_indices_accumulate(self, device):
        # 创建一个全 False 的布尔掩码 mask 和一个全为 1 的张量 y
        mask = torch.zeros(size=(10,), dtype=torch.bool, device=device)
        y = torch.ones(size=(10, 10), device=device)
        # 使用布尔索引累加 y 中 mask 对应位置的值
        y.index_put_((mask,), y[mask], accumulate=True)
        # 断言 y 的值仍然全部为 1
        self.assertEqual(y, torch.ones(size=(10, 10), device=device))

    # 测试使用多个布尔索引选择子集
    def test_multiple_bool_indices(self, device):
        # 创建一个形状为 (5, 7, 3) 的随机张量 v
        v = torch.randn(5, 7, 3, device=device)
        # 创建两个布尔掩码 mask1 和 mask2
        # 注意：这两个掩码会进行广播并转置到第一个维度
        mask1 = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool, device=device)
        mask2 = torch.tensor([1, 1, 1], dtype=torch.bool, device=device)
        # 断言选择的子集张量的形状与预期相等
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))

    # 测试使用字节掩码进行选择
    def test_byte_mask(self, device):
        # 创建一个形状为 (5, 7, 3) 的随机张量 v
        v = torch.randn(5, 7, 3, device=device)
        # 创建一个字节掩码 mask
        mask = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        # 使用字节掩码选择张量的子集，同时记录警告
        with warnings.catch_warnings(record=True) as w:
            res = v[mask]
            # 断言选择的子集张量的形状与预期相等
            self.assertEqual(res.shape, (3, 7, 3))
            # 断言选择的子集张量与手动堆叠的张量在数值上相等
            self.assertEqual(res, torch.stack([v[0], v[2], v[3]]))
            # 断言警告的数量
            self.assertEqual(len(w), 1)

        # 创建一个包含单个元素的张量 v
        v = torch.tensor([1.0], device=device)
        # 断言对于选择 v 中值为 0 的元素，结果应为空张量
        self.assertEqual(v[v == 0], torch.tensor([], device=device))

    # 测试使用字节掩码进行累加操作
    def test_byte_mask_accumulate(self, device):
        # 创建一个全零的字节掩码 mask 和一个全为 1 的张量 y
        mask = torch.zeros(size=(10,), dtype=torch.uint8, device=device)
        y = torch.ones(size=(10, 10), device=device)
        # 使用字节掩码累加 y 中 mask 对应位置的值，同时记录警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y.index_put_((mask,), y[mask], accumulate=True)
            # 断言 y 的值仍然全部为 1
            self.assertEqual(y, torch.ones(size=(10, 10), device=device))
            # 断言警告的数量
            self.assertEqual(len(w), 2)

    # 在 TorchDynamo 环境中跳过此测试
    @skipIfTorchDynamo(
        "This test causes SIGKILL when running with dynamo, https://github.com/pytorch/pytorch/issues/88472"
    )
    @serialTest(TEST_CUDA)
    # 定义一个测试方法，用于测试大型张量的索引操作
    def test_index_put_accumulate_large_tensor(self, device):
        # 设置一个超过 INT_MAX (2^31 - 1) 元素数量的张量大小
        N = (1 << 31) + 5
        # 指定张量的数据类型为 int8
        dt = torch.int8
        # 创建一个大小为 N 的全 1 张量，指定数据类型和设备
        a = torch.ones(N, dtype=dt, device=device)
        # 指定索引张量，包含要修改的位置
        indices = torch.tensor(
            [-2, 0, -2, -1, 0, -1, 1], device=device, dtype=torch.long
        )
        # 指定值张量，包含要加到相应位置的值
        values = torch.tensor([6, 5, 6, 6, 5, 7, 11], dtype=dt, device=device)

        # 使用索引方式在张量 a 中放置值 values，并累积到原始值上
        a.index_put_((indices,), values, accumulate=True)

        # 断言操作后张量 a 中的特定位置的值是否正确
        self.assertEqual(a[0], 11)
        self.assertEqual(a[1], 12)
        self.assertEqual(a[2], 1)
        self.assertEqual(a[-3], 1)
        self.assertEqual(a[-2], 13)
        self.assertEqual(a[-1], 14)

        # 创建一个大小为 (2, N) 的全 1 张量，指定数据类型和设备
        a = torch.ones((2, N), dtype=dt, device=device)
        # 指定两个索引张量，分别表示要修改的位置
        indices0 = torch.tensor([0, -1, 0, 1], device=device, dtype=torch.long)
        indices1 = torch.tensor([-2, -1, 0, 1], device=device, dtype=torch.long)
        # 指定值张量，包含要加到相应位置的值
        values = torch.tensor([12, 13, 10, 11], dtype=dt, device=device)

        # 使用多维索引方式在张量 a 中放置值 values，并累积到原始值上
        a.index_put_((indices0, indices1), values, accumulate=True)

        # 断言操作后张量 a 中的特定位置的值是否正确
        self.assertEqual(a[0, 0], 11)
        self.assertEqual(a[0, 1], 1)
        self.assertEqual(a[1, 0], 1)
        self.assertEqual(a[1, 1], 12)
        self.assertEqual(a[:, 2], torch.ones(2, dtype=torch.int8))
        self.assertEqual(a[:, -3], torch.ones(2, dtype=torch.int8))
        self.assertEqual(a[0, -2], 13)
        self.assertEqual(a[1, -2], 1)
        self.assertEqual(a[-1, -1], 14)
        self.assertEqual(a[0, -1], 1)
    # 定义一个测试方法，用于验证在 CUDA 设备上执行 index_put_ 操作时的累积值问题
    def test_index_put_accumulate_expanded_values(self, device):
        # 检查与 CUDA 相关的问题：https://github.com/pytorch/pytorch/issues/39227
        # 并验证与 CPU 结果的一致性
        t = torch.zeros((5, 2))  # 创建一个大小为 5x2 的全零张量
        t_dev = t.to(device)  # 将张量移动到指定的设备（CPU或CUDA）

        # 定义多个索引张量
        indices = [
            torch.tensor([0, 1, 2, 3]),  # 长度为 4 的张量索引
            torch.tensor([1]),  # 长度为 1 的张量索引
        ]
        indices_dev = [i.to(device) for i in indices]  # 将所有索引张量移动到设备上

        # 定义不同维度的值张量
        values0d = torch.tensor(1.0)  # 标量值张量
        values1d = torch.tensor([1.0])  # 长度为 1 的向量值张量

        # 在 CUDA 设备上执行 index_put_ 操作，累积替换值
        out_cuda = t_dev.index_put_(indices_dev, values0d.to(device), accumulate=True)
        # 在 CPU 上执行 index_put_ 操作，累积替换值
        out_cpu = t.index_put_(indices, values0d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)  # 断言 CUDA 和 CPU 的结果一致性

        # 再次执行 index_put_ 操作，使用长度为 1 的向量值张量
        out_cuda = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)  # 断言 CUDA 和 CPU 的结果一致性

        # 创建一个大小为 4x3x2 的全零张量，并移动到 CUDA 设备上
        t = torch.zeros(4, 3, 2)
        t_dev = t.to(device)

        # 定义多个不同维度的索引张量
        indices = [
            torch.tensor([0]),  # 长度为 1 的张量索引
            torch.arange(3)[:, None],  # 3 行 1 列的张量索引
            torch.arange(2)[None, :],  # 1 行 2 列的张量索引
        ]
        indices_dev = [i.to(device) for i in indices]  # 将所有索引张量移动到设备上

        # 定义不同维度的值张量
        values1d = torch.tensor([-1.0, -2.0])  # 长度为 2 的向量值张量
        values2d = torch.tensor([[-1.0, -2.0]])  # 1 行 2 列的矩阵值张量

        # 在 CUDA 设备上执行 index_put_ 操作，累积替换值
        out_cuda = t_dev.index_put_(indices_dev, values1d.to(device), accumulate=True)
        # 在 CPU 上执行 index_put_ 操作，累积替换值
        out_cpu = t.index_put_(indices, values1d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)  # 断言 CUDA 和 CPU 的结果一致性

        # 再次执行 index_put_ 操作，使用 1x2 的矩阵值张量
        out_cuda = t_dev.index_put_(indices_dev, values2d.to(device), accumulate=True)
        out_cpu = t.index_put_(indices, values2d, accumulate=True)
        self.assertEqual(out_cuda.cpu(), out_cpu)  # 断言 CUDA 和 CPU 的结果一致性

    @onlyCUDA
    # 定义一个测试方法，用于验证在非连续张量上执行 index_put_ 操作时的累积值问题
    def test_index_put_accumulate_non_contiguous(self, device):
        t = torch.zeros((5, 2, 2))  # 创建一个大小为 5x2x2 的全零张量
        t_dev = t.to(device)  # 将张量移动到指定的设备（CPU或CUDA）

        t1 = t_dev[:, 0, :]  # 选择张量的部分切片，并移动到设备上
        t2 = t[:, 0, :]  # 选择相同切片的张量，但保留在 CPU 上
        self.assertTrue(not t1.is_contiguous())  # 断言 t1 张量是非连续的
        self.assertTrue(not t2.is_contiguous())  # 断言 t2 张量是非连续的

        # 定义一个长度为 2 的张量索引
        indices = [
            torch.tensor([0, 1]),
        ]
        indices_dev = [i.to(device) for i in indices]  # 将索引张量移动到设备上

        # 定义一个随机值张量
        value = torch.randn(2, 2)

        # 在非连续的 t1 张量上执行 index_put_ 操作，累积替换值
        out_cuda = t1.index_put_(indices_dev, value.to(device), accumulate=True)
        # 在非连续的 t2 张量上执行 index_put_ 操作，累积替换值
        out_cpu = t2.index_put_(indices, value, accumulate=True)
        self.assertTrue(not t1.is_contiguous())  # 断言 t1 张量仍然是非连续的
        self.assertTrue(not t2.is_contiguous())  # 断言 t2 张量仍然是非连续的

        self.assertEqual(out_cuda.cpu(), out_cpu)  # 断言 CUDA 和 CPU 的结果一致性

    @onlyCUDA
    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    # 使用 torch.jit.script 装饰器将 func 函数编译成 TorchScript 形式，用于优化执行效率
    @torch.jit.script
    def func(x, i, v):
        # 创建包含 None 和索引 i 的列表 idx
        idx = [None, i]
        # 在张量 x 上使用索引 idx 进行累加操作，将值 v 插入对应位置
        x.index_put_(idx, v, accumulate=True)
        return x

    # 设置测试中使用的张量大小 n
    n = 4
    # 创建一个形状为 (n, 2) 的浮点型张量 t
    t = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
    # 将张量 t 移动到指定设备上
    t_dev = t.to(device)
    # 创建索引张量 indices，用于指定 func 函数中的索引 i
    indices = torch.tensor([1, 0])
    # 将索引张量 indices 移动到指定设备上
    indices_dev = indices.to(device)
    # 创建标量张量 value0d，用于作为值 v 的输入
    value0d = torch.tensor(10.0)
    # 创建一维张量 value1d，用于作为值 v 的输入
    value1d = torch.tensor([1.0, 2.0])

    # 在 CUDA 设备上调用 func 函数，输入 t_dev、indices_dev 和 value0d.cuda()，并获得输出 out_cuda
    out_cuda = func(t_dev, indices_dev, value0d.cuda())
    # 在 CPU 上调用 func 函数，输入 t、indices 和 value0d，并获得输出 out_cpu
    out_cpu = func(t, indices, value0d)
    # 断言 out_cuda 和 out_cpu 的值相等
    self.assertEqual(out_cuda.cpu(), out_cpu)

    # 在 CUDA 设备上调用 func 函数，输入 t_dev、indices_dev 和 value1d.cuda()，并获得输出 out_cuda
    out_cuda = func(t_dev, indices_dev, value1d.cuda())
    # 在 CPU 上调用 func 函数，输入 t、indices 和 value1d，并获得输出 out_cpu
    out_cpu = func(t, indices, value1d)
    # 断言 out_cuda 和 out_cpu 的值相等
    self.assertEqual(out_cuda.cpu(), out_cpu)


```    
    @onlyNativeDeviceTypes
    def test_index_put_accumulate_duplicate_indices(self, device):
        # 遍历从 1 到 511 的整数范围，用于生成索引 delta
        for i in range(1, 512):
            # 创建 delta 张量，包含 i 个元素，数据类型为 double，存储在指定设备上，并初始化为 -1 到 1 之间的均匀分布随机数
            delta = torch.empty(i, dtype=torch.double, device=device).uniform_(-1, 1)
            # 对 delta 张量进行累积求和，并转换为长整型索引张量 indices
            indices = delta.cumsum(0).long()

            # 创建输入张量 input，形状为 indices.abs().max() + 1，存储在指定设备上，并初始化为标准正态分布随机数
            input = torch.randn(indices.abs().max() + 1, device=device)
            # 创建值张量 values，形状为 indices.size(0)，存储在指定设备上，并初始化为标准正态分布随机数
            values = torch.randn(indices.size(0), device=device)
            # 在输入张量 input 上使用索引 indices，将值张量 values 插入对应位置并进行累加操作，得到输出 output
            output = input.index_put((indices,), values, accumulate=True)

            # 将输入张量 input 转换为 Python 列表 input_list
            input_list = input.tolist()
            # 将索引张量 indices 转换为 Python 列表 indices_list
            indices_list = indices.tolist()
            # 将值张量 values 转换为 Python 列表 values_list
            values_list = values.tolist()
            # 遍历 indices_list 和 values_list，并根据其值更新 input_list 中对应索引的元素
            for i, v in zip(indices_list, values_list):
                input_list[i] += v

            # 断言输出 output 与更新后的 input_list 相等
            self.assertEqual(output, input_list)



    @onlyNativeDeviceTypes
    def test_index_ind_dtype(self, device):
        # 创建形状为 (4, 4) 的标准正态分布张量 x，存储在指定设备上
        x = torch.randn(4, 4, device=device)
        # 创建形状为 (4,) 的长整型随机张量 ind_long，数据类型为 long，存储在指定设备上
        ind_long = torch.randint(4, (4,), dtype=torch.long, device=device)
        # 将 ind_long 转换为整型张量 ind_int
        ind_int = ind_long.int()
        # 创建形状为 (4,) 的标准正态分布张量 src，存储在指定设备上
        src = torch.randn(4, device=device)

        # 使用 ind_long 作为索引，在张量 x 中获取参考值 ref
        ref = x[ind_long, ind_long]
        # 使用 ind_int 作为索引，在张量 x 中获取结果值 res
        res = x[ind_int, ind_int]
        # 断言 ref 和 res 的值相等
        self.assertEqual(ref, res)

        # 使用 ind_long 作为索引，在张量 x 中获取参考值 ref
        ref = x[ind_long, :]
        # 使用 ind_int 作为索引，在张量 x 中获取结果值 res
        res = x[ind_int, :]
        # 断言 ref 和 res 的值相等
        self.assertEqual(ref, res)

        # 使用 ind_long 作为索引，在张量 x 中获取参考值 ref
        ref = x[:, ind_long]
        # 使用 ind_int 作为索引，在张量 x 中获取结果值 res
        res = x[:, ind_int]
        # 断言 ref 和 res 的值相等
        self.assertEqual(ref, res)

        # 遍历布尔值 (True, False)，分别进行以下操作
        for accum in (True, False):
            # 复制张量 x 到 inp_ref 和 inp_res
            inp_ref = x.clone()
            inp_res = x.clone()
            # 使用 ind_long 作为索引，在 inp_ref 中使用 src 进行累加更新
            torch.index_put_(inp_ref, (ind_long, ind_long), src, accum)
            # 使用 ind_int 作为索引，在 inp_res 中使用 src 进行累加更新
            torch.index_put_(inp_res, (ind_int, ind_int), src, accum)
            # 断言 inp_ref 和 inp_res 的值相等
            self.assertEqual(inp_ref, inp_res)
    def test_index_put_accumulate_empty(self, device):
        # 测试空索引情况下的 index_put 方法
        # 这是针对 https://github.com/pytorch/pytorch/issues/94667 的回归测试
        # 创建一个随机的标量张量 input，数据类型为 float32，在指定设备上
        input = torch.rand([], dtype=torch.float32, device=device)
        # 使用 assertRaises 检查是否抛出 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            # 对空索引调用 index_put 方法，传入一个张量和参数 True
            input.index_put([], torch.tensor([1.0], device=device), True)

    def test_multiple_byte_mask(self, device):
        # 测试多个字节掩码的情况
        # 创建一个在指定设备上的张量 v，形状为 (5, 7, 3)
        v = torch.randn(5, 7, 3, device=device)
        # 创建两个字节张量 mask1 和 mask2，并转移到指定设备
        mask1 = torch.ByteTensor([1, 0, 1, 1, 0]).to(device)
        mask2 = torch.ByteTensor([1, 1, 1]).to(device)
        # 使用警告过滤器捕获警告信息
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # 断言使用多个字节掩码对张量 v 进行索引操作后的形状
            self.assertEqual(v[mask1, :, mask2].shape, (3, 7))
            # 断言捕获到的警告数量为 2
            self.assertEqual(len(w), 2)

    def test_byte_mask2d(self, device):
        # 测试二维字节掩码的情况
        # 创建一个在指定设备上的张量 v，形状为 (5, 7, 3)
        v = torch.randn(5, 7, 3, device=device)
        # 创建一个在指定设备上的张量 c，形状为 (5, 7)
        c = torch.randn(5, 7, device=device)
        # 计算张量 c 中大于 0 的元素数量
        num_ones = (c > 0).sum()
        # 根据 c 中大于 0 的元素创建一个新的张量 r
        r = v[c > 0]
        # 断言张量 r 的形状与大于 0 的元素数量和 3 维度相匹配
        self.assertEqual(r.shape, (num_ones, 3))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_jit_indexing(self, device):
        # 测试 JIT 编译的索引操作
        # 定义两个函数 fn1 和 fn2，用于修改输入张量的值
        def fn1(x):
            x[x < 50] = 1.0
            return x

        def fn2(x):
            x[0:50] = 1.0
            return x

        # 对函数 fn1 和 fn2 进行 JIT 编译
        scripted_fn1 = torch.jit.script(fn1)
        scripted_fn2 = torch.jit.script(fn2)
        # 创建一个设备上的张量数据
        data = torch.arange(100, device=device, dtype=torch.float)
        # 使用 JIT 编译的函数处理数据，并生成参考结果 ref
        out = scripted_fn1(data.detach().clone())
        ref = torch.tensor(
            np.concatenate((np.ones(50), np.arange(50, 100))),
            device=device,
            dtype=torch.float,
        )
        # 断言 JIT 处理后的结果与参考结果 ref 相等
        self.assertEqual(out, ref)
        # 对第二个 JIT 编译的函数进行类似的测试
        out = scripted_fn2(data.detach().clone())
        self.assertEqual(out, ref)

    def test_int_indices(self, device):
        # 测试使用整数索引的情况
        # 创建一个在指定设备上的张量 v，形状为 (5, 7, 3)
        v = torch.randn(5, 7, 3, device=device)
        # 断言使用整数索引操作后的张量形状
        self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
        self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
        self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

    @dtypes(
        torch.cfloat, torch.cdouble, torch.float, torch.bfloat16, torch.long, torch.bool
    )
    @dtypesIfCPU(
        torch.cfloat, torch.cdouble, torch.float, torch.long, torch.bool, torch.bfloat16
    )
    @dtypesIfCUDA(
        torch.cfloat,
        torch.cdouble,
        torch.half,
        torch.long,
        torch.bool,
        torch.bfloat16,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    )
    def test_index_put_src_datatype(self, device, dtype):
        # 测试 index_put 方法的源数据类型
        # 创建一个在指定设备上的全为 1 的张量 src，并指定数据类型 dtype
        src = torch.ones(3, 2, 4, device=device, dtype=dtype)
        # 创建一个与 src 形状相同的全为 1 的张量 vals，并指定数据类型 dtype
        vals = torch.ones(3, 2, 4, device=device, dtype=dtype)
        # 定义索引 indices，用于指定操作的位置
        indices = (torch.tensor([0, 2, 1]),)
        # 使用 index_put_ 方法对 src 进行操作，并启用累加模式
        res = src.index_put_(indices, vals, accumulate=True)
        # 断言操作后的结果张量形状与 src 相同
        self.assertEqual(res.shape, src.shape)

    @dtypes(torch.float, torch.bfloat16, torch.long, torch.bool)
    @dtypesIfCPU(torch.float, torch.long, torch.bfloat16, torch.bool)
    # 使用 @dtypesIfCUDA 装饰器指定在 CUDA 下的数据类型
    @dtypesIfCUDA(torch.half, torch.long, torch.bfloat16, torch.bool)
    # 定义测试函数，测试索引操作的源数据类型
    def test_index_src_datatype(self, device, dtype):
        # 创建一个指定设备和数据类型的全为 1 的张量
        src = torch.ones(3, 2, 4, device=device, dtype=dtype)
        # 测试索引操作，选择部分元素组成新的张量 res
        res = src[[0, 2, 1], :, :]
        # 断言新张量的形状与原始张量相同
        self.assertEqual(res.shape, src.shape)
        # 测试索引赋值操作，将 res 的值赋回原张量的指定位置
        src[[0, 2, 1], :, :] = res
        # 再次断言修改后的张量形状与原始张量相同
        self.assertEqual(res.shape, src.shape)

    # 定义测试函数，测试二维整数索引
    def test_int_indices2d(self, device):
        # 创建一个形状为 (4, 3) 的张量 x，其值从 0 到 11
        x = torch.arange(0, 12, device=device).view(4, 3)
        # 创建行索引和列索引的张量
        rows = torch.tensor([[0, 0], [3, 3]], device=device)
        columns = torch.tensor([[0, 2], [0, 2]], device=device)
        # 断言按指定索引取出的值与预期结果相同
        self.assertEqual(x[rows, columns].tolist(), [[0, 2], [9, 11]])

    # 定义测试函数，测试广播的整数索引
    def test_int_indices_broadcast(self, device):
        # 创建一个形状为 (4, 3) 的张量 x，其值从 0 到 11
        x = torch.arange(0, 12, device=device).view(4, 3)
        # 创建行索引和列索引的张量
        rows = torch.tensor([0, 3], device=device)
        columns = torch.tensor([0, 2], device=device)
        # 使用广播形式的索引操作，获取结果张量
        result = x[rows[:, None], columns]
        # 断言按指定广播索引取出的值与预期结果相同
        self.assertEqual(result.tolist(), [[0, 2], [9, 11]])

    # 定义测试函数，测试空索引的情况
    def test_empty_index(self, device):
        # 创建一个形状为 (4, 3) 的张量 x，其值从 0 到 11
        x = torch.arange(0, 12, device=device).view(4, 3)
        # 创建一个空的索引张量
        idx = torch.tensor([], dtype=torch.long, device=device)
        # 断言空索引取出的元素个数为 0
        self.assertEqual(x[idx].numel(), 0)

        # 使用空索引进行赋值操作，应该没有效果但不会引发异常
        y = x.clone()
        y[idx] = -1
        # 断言赋值后张量值与原张量相同
        self.assertEqual(x, y)

        # 使用布尔掩码进行空索引赋值操作，应该没有效果但不会引发异常
        mask = torch.zeros(4, 3, device=device).bool()
        y[mask] = -1
        # 断言赋值后张量值与原张量相同
        self.assertEqual(x, y)

    # 定义测试函数，测试多维空索引的情况
    def test_empty_ndim_index(self, device):
        # 创建一个形状为 (5,) 的张量 x，其值为随机数
        x = torch.randn(5, device=device)
        # 断言按空索引取出的张量形状符合预期
        self.assertEqual(
            torch.empty(0, 2, device=device),
            x[torch.empty(0, 2, dtype=torch.int64, device=device)],
        )

        # 创建一个形状为 (2, 3, 4, 5) 的张量 x，其值为随机数
        x = torch.randn(2, 3, 4, 5, device=device)
        # 断言按空索引取出的张量形状符合预期
        self.assertEqual(
            torch.empty(2, 0, 6, 4, 5, device=device),
            x[:, torch.empty(0, 6, dtype=torch.int64, device=device)],
        )

        # 创建一个形状为 (10, 0) 的空张量 x
        x = torch.empty(10, 0, device=device)
        # 断言空索引取出的张量形状符合预期
        self.assertEqual(x[[1, 2]].shape, (2, 0))
        # 断言空索引取出的张量形状符合预期
        self.assertEqual(x[[], []].shape, (0,))
        # 断言索引超出维度大小会引发 IndexError 异常
        with self.assertRaisesRegex(IndexError, "for dimension with size 0"):
            x[:, [0, 1]]

    # 定义测试函数，测试布尔空索引的情况
    def test_empty_ndim_index_bool(self, device):
        # 创建一个形状为 (5,) 的张量 x，其值为随机数
        x = torch.randn(5, device=device)
        # 断言使用布尔类型的空索引会引发 IndexError 异常
        self.assertRaises(
            IndexError, lambda: x[torch.empty(0, 2, dtype=torch.uint8, device=device)]
        )

    # 定义测试函数，测试空切片的情况
    def test_empty_slice(self, device):
        # 创建一个形状为 (2, 3, 4, 5) 的张量 x，其值为随机数
        x = torch.randn(2, 3, 4, 5, device=device)
        # 按指定切片取出子张量 y
        y = x[:, :, :, 1]
        # 按空切片切取子张量 z
        z = y[:, 1:1, :]
        # 断言切取后张量 z 的形状符合预期
        self.assertEqual((2, 0, 4), z.shape)
        # 断言 z 张量的步幅符合 NumPy 的计算规则
        # 这虽然不是必要的，但与 NumPy 的步幅计算匹配
        self.assertEqual((60, 20, 5), z.stride())
        # 断言 z 张量是连续的
        self.assertTrue(z.is_contiguous())
    def test_index_getitem_copy_bools_slices(self, device):
        # 创建 torch.uint8 类型的张量 true 和 false，分别表示 1 和 0
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        # 创建包含不同类型张量的列表 tensors
        tensors = [torch.randn(2, 3, device=device), torch.tensor(3.0, device=device)]

        # 遍历 tensors 中的每个张量 a
        for a in tensors:
            # 断言使用索引 True 不会得到与原张量相同的数据指针
            self.assertNotEqual(a.data_ptr(), a[True].data_ptr())
            # 断言使用索引 False 会返回一个空张量，形状与 a 相同
            self.assertEqual(torch.empty(0, *a.shape), a[False])
            # 断言使用变量 true 作为索引不会得到与原张量相同的数据指针
            self.assertNotEqual(a.data_ptr(), a[true].data_ptr())
            # 断言使用变量 false 作为索引会返回一个空张量，形状与 a 相同
            self.assertEqual(torch.empty(0, *a.shape), a[false])
            # 断言使用索引 None 会得到与原张量相同的数据指针
            self.assertEqual(a.data_ptr(), a[None].data_ptr())
            # 断言使用省略号索引 (...) 会得到与原张量相同的数据指针
            self.assertEqual(a.data_ptr(), a[...].data_ptr())

    def test_index_setitem_bools_slices(self, device):
        # 创建 torch.uint8 类型的张量 true 和 false，分别表示 1 和 0
        true = torch.tensor(1, dtype=torch.uint8, device=device)
        false = torch.tensor(0, dtype=torch.uint8, device=device)

        # 创建包含不同类型张量的列表 tensors
        tensors = [torch.randn(2, 3, device=device), torch.tensor(3, device=device)]

        # 遍历 tensors 中的每个张量 a
        for a in tensors:
            # 创建全为 -1 的张量 neg_ones，并扩展其维度
            neg_ones = torch.ones_like(a) * -1
            neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
            # 使用索引 True 进行赋值操作
            a[True] = neg_ones_expanded
            # 断言张量 a 的值与 neg_ones 相等
            self.assertEqual(a, neg_ones)
            # 使用索引 False 进行赋值操作，预期不会改变张量 a 的值
            a[False] = 5
            self.assertEqual(a, neg_ones)
            # 使用变量 true 作为索引进行赋值操作
            a[true] = neg_ones_expanded * 2
            # 断言张量 a 的值为 neg_ones 的两倍
            self.assertEqual(a, neg_ones * 2)
            # 使用变量 false 作为索引进行赋值操作，预期不会改变张量 a 的值
            a[false] = 5
            self.assertEqual(a, neg_ones * 2)
            # 使用索引 None 进行赋值操作
            a[None] = neg_ones_expanded * 3
            # 断言张量 a 的值为 neg_ones 的三倍
            self.assertEqual(a, neg_ones * 3)
            # 使用省略号索引 (...) 进行赋值操作
            a[...] = neg_ones_expanded * 4
            # 断言张量 a 的值为 neg_ones 的四倍
            self.assertEqual(a, neg_ones * 4)
            # 如果张量 a 的维度为 0，则检查赋值操作是否引发 IndexError
            if a.dim() == 0:
                with self.assertRaises(IndexError):
                    a[:] = neg_ones_expanded * 5

    def test_index_scalar_with_bool_mask(self, device):
        # 创建一个标量张量 a
        a = torch.tensor(1, device=device)
        # 创建 torch.uint8 类型的张量 uintMask 和 boolMask，值均为 True
        uintMask = torch.tensor(True, dtype=torch.uint8, device=device)
        boolMask = torch.tensor(True, dtype=torch.bool, device=device)
        # 使用 uintMask 和 boolMask 作为索引访问张量 a，预期结果相等
        self.assertEqual(a[uintMask], a[boolMask])
        # 断言 uintMask 和 boolMask 访问结果的数据类型相同
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

        # 将标量张量 a 重新赋值为 True，数据类型为 torch.bool
        a = torch.tensor(True, dtype=torch.bool, device=device)
        # 使用 uintMask 和 boolMask 作为索引访问张量 a，预期结果相等
        self.assertEqual(a[uintMask], a[boolMask])
        # 断言 uintMask 和 boolMask 访问结果的数据类型相同
        self.assertEqual(a[uintMask].dtype, a[boolMask].dtype)

    def test_setitem_expansion_error(self, device):
        # 创建 torch.uint8 类型的张量 true
        true = torch.tensor(True, device=device)
        # 创建一个张量 a，形状为 (2, 3)
        a = torch.randn(2, 3, device=device)
        # 创建一个扩展了维度的张量 a_expanded，其形状为 [5, 1] + a.size()
        a_expanded = a.expand(torch.Size([5, 1]) + a.size())
        # 使用索引 True 进行赋值操作，预期引发 RuntimeError
        with self.assertRaises(RuntimeError):
            a[True] = a_expanded
        # 使用变量 true 作为索引进行赋值操作，预期引发 RuntimeError
        with self.assertRaises(RuntimeError):
            a[true] = a_expanded
    # 测试使用标量进行索引和赋值的情况
    def test_getitem_scalars(self, device):
        # 创建值为0和1的整数张量，数据类型为int64，存储在指定设备上
        zero = torch.tensor(0, dtype=torch.int64, device=device)
        one = torch.tensor(1, dtype=torch.int64, device=device)

        # 创建一个形状为(2, 3)的张量，元素服从标准正态分布，存储在指定设备上
        a = torch.randn(2, 3, device=device)

        # 断言索引操作的结果相等：标量索引和整数张量索引的比较
        self.assertEqual(a[0], a[zero])
        self.assertEqual(a[0][1], a[zero][one])
        self.assertEqual(a[0, 1], a[zero, one])
        self.assertEqual(a[0, one], a[zero, 1])

        # 断言标量索引操作后的数据指针相等，表明标量索引是切片而非复制
        self.assertEqual(a[0, 1].data_ptr(), a[zero, one].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.int()].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.short()].data_ptr())

        # 标量索引和标量赋值的情况
        r = torch.randn((), device=device)  # 创建一个标量张量，存储在指定设备上
        with self.assertRaises(IndexError):
            r[:]  # 不支持切片赋值
        with self.assertRaises(IndexError):
            r[zero]  # 不支持标量索引
        self.assertEqual(r, r[...])  # 索引为...的操作应返回自身

    # 测试使用标量进行赋值的情况
    def test_setitem_scalars(self, device):
        zero = torch.tensor(0, dtype=torch.int64)

        # 创建一个形状为(2, 3)的张量，元素服从标准正态分布，存储在指定设备上
        a = torch.randn(2, 3, device=device)
        a_set_with_number = a.clone()
        a_set_with_scalar = a.clone()
        b = torch.randn(3, device=device)

        # 使用数值和标量进行索引和赋值
        a_set_with_number[0] = b
        a_set_with_scalar[zero] = b
        self.assertEqual(a_set_with_number, a_set_with_scalar)
        a[1, zero] = 7.7
        self.assertEqual(7.7, a[1, 0])

        # 标量张量进行标量赋值的情况
        r = torch.randn((), device=device)
        with self.assertRaises(IndexError):
            r[:] = 8.8  # 不支持切片赋值
        with self.assertRaises(IndexError):
            r[zero] = 8.8  # 不支持标量索引
        r[...] = 9.9
        self.assertEqual(9.9, r)

    # 测试基本和高级索引结合使用的情况
    def test_basic_advanced_combined(self, device):
        # 从 NumPy 索引示例中创建张量
        x = torch.arange(0, 12, device=device).view(4, 3)
        self.assertEqual(x[1:2, 1:3], x[1:2, [1, 2]])
        self.assertEqual(x[1:2, 1:3].tolist(), [[4, 5]])

        # 断言切片操作是副本
        unmodified = x.clone()
        x[1:2, [1, 2]].zero_()
        self.assertEqual(x, unmodified)

        # 但是赋值操作应修改原始张量
        unmodified = x.clone()
        x[1:2, [1, 2]] = 0
        self.assertNotEqual(x, unmodified)

    # 测试整数赋值的情况
    def test_int_assignment(self, device):
        x = torch.arange(0, 4, device=device).view(2, 2)
        x[1] = 5
        self.assertEqual(x.tolist(), [[0, 1], [5, 5]])

        x = torch.arange(0, 4, device=device).view(2, 2)
        x[1] = torch.arange(5, 7, device=device)
        self.assertEqual(x.tolist(), [[0, 1], [5, 6]])
    # 测试字节张量的赋值操作，使用指定的设备
    def test_byte_tensor_assignment(self, device):
        # 创建一个4x4的张量x，元素从0到15，使用指定设备
        x = torch.arange(0.0, 16, device=device).view(4, 4)
        # 创建一个字节张量b，指定元素为True、False、True、False，并转移到指定设备
        b = torch.ByteTensor([True, False, True, False]).to(device)
        # 创建一个张量value，包含元素3.0、4.0、5.0、6.0，并使用指定设备
        value = torch.tensor([3.0, 4.0, 5.0, 6.0], device=device)

        # 用警告捕获功能记录警告信息
        with warnings.catch_warnings(record=True) as w:
            # 使用字节张量b对张量x进行赋值操作，赋值为张量value
            x[b] = value
            # 断言捕获的警告数量为1
            self.assertEqual(len(w), 1)

        # 断言张量x的第一行等于张量value
        self.assertEqual(x[0], value)
        # 断言张量x的第二行等于从4.0到7.0的序列张量，使用指定设备
        self.assertEqual(x[1], torch.arange(4.0, 8, device=device))
        # 断言张量x的第三行等于张量value
        self.assertEqual(x[2], value)
        # 断言张量x的第四行等于从12.0到15.0的序列张量，使用指定设备
        self.assertEqual(x[3], torch.arange(12.0, 16, device=device))

    # 测试变量切片功能，使用指定的设备
    def test_variable_slicing(self, device):
        # 创建一个4x4的张量x，元素从0到15，使用指定设备
        x = torch.arange(0, 16, device=device).view(4, 4)
        # 创建一个整型张量indices，包含元素0和1，并转移到指定设备
        indices = torch.IntTensor([0, 1]).to(device)
        # 将indices中的元素分别赋值给变量i和j
        i, j = indices
        # 断言张量x中i到j的切片等于x中0到1的切片
        self.assertEqual(x[i:j], x[0:1])

    # 测试省略符号的张量操作，使用指定的设备
    def test_ellipsis_tensor(self, device):
        # 创建一个3x3的张量x，元素从0到8，使用指定设备
        x = torch.arange(0, 9, device=device).view(3, 3)
        # 创建一个张量idx，包含元素0和2，并使用指定设备
        idx = torch.tensor([0, 2], device=device)
        # 断言张量x中所有维度的所有切片与idx对应的列组成的列表相等
        self.assertEqual(x[..., idx].tolist(), [[0, 2], [3, 5], [6, 8]])
        # 断言张量x中idx对应的行与所有维度的所有切片组成的列表相等
        self.assertEqual(x[idx, ...].tolist(), [[0, 1, 2], [6, 7, 8]])

    # 测试unravel_index函数的错误处理功能，使用指定的设备
    def test_unravel_index_errors(self, device):
        # 断言抛出TypeError异常，其中错误消息包含"expected 'indices' to be integer"
        with self.assertRaisesRegex(TypeError, r"expected 'indices' to be integer"):
            torch.unravel_index(torch.tensor(0.5, device=device), (2, 2))

        # 断言抛出TypeError异常，其中错误消息包含"expected 'indices' to be integer"
        with self.assertRaisesRegex(TypeError, r"expected 'indices' to be integer"):
            torch.unravel_index(torch.tensor([], device=device), (10, 3, 5))

        # 断言抛出TypeError异常，其中错误消息包含"expected 'shape' to be int or sequence"
        with self.assertRaisesRegex(
            TypeError, r"expected 'shape' to be int or sequence"
        ):
            torch.unravel_index(
                torch.tensor([1], device=device, dtype=torch.int64),
                torch.tensor([1, 2, 3]),
            )

        # 断言抛出TypeError异常，其中错误消息包含"expected 'shape' sequence to only contain ints"
        with self.assertRaisesRegex(
            TypeError, r"expected 'shape' sequence to only contain ints"
        ):
            torch.unravel_index(
                torch.tensor([1], device=device, dtype=torch.int64), (1, 2, 2.0)
            )

        # 断言抛出ValueError异常，其中错误消息包含"'shape' cannot have negative values, but got (2, -3)"
        with self.assertRaisesRegex(
            ValueError, r"'shape' cannot have negative values, but got \(2, -3\)"
        ):
            torch.unravel_index(torch.tensor(0, device=device), (2, -3))

    # 测试无效索引的处理，使用指定的设备
    def test_invalid_index(self, device):
        # 创建一个4x4的张量x，元素从0到15，使用指定设备
        x = torch.arange(0, 16, device=device).view(4, 4)
        # 断言抛出TypeError异常，错误消息包含"slice indices"
        self.assertRaisesRegex(TypeError, "slice indices", lambda: x["0":"1"])
    # 定义一个测试方法，用于测试在给定设备上的索引超出边界情况
    def test_out_of_bound_index(self, device):
        # 创建一个张量 x，其值从 0 到 99，在指定设备上，形状为 2x5x10
        x = torch.arange(0, 100, device=device).view(2, 5, 10)
        # 断言在索引超出维度 1 大小（5）的情况下会抛出 IndexError 异常
        self.assertRaisesRegex(
            IndexError,
            "index 5 is out of bounds for dimension 1 with size 5",
            lambda: x[0, 5],
        )
        # 断言在索引超出维度 0 大小（2）的情况下会抛出 IndexError 异常
        self.assertRaisesRegex(
            IndexError,
            "index 4 is out of bounds for dimension 0 with size 2",
            lambda: x[4, 5],
        )
        # 断言在索引超出维度 2 大小（10）的情况下会抛出 IndexError 异常
        self.assertRaisesRegex(
            IndexError,
            "index 15 is out of bounds for dimension 2 with size 10",
            lambda: x[0, 1, 15],
        )
        # 断言使用切片索引超出维度 2 大小（10）的情况下会抛出 IndexError 异常
        self.assertRaisesRegex(
            IndexError,
            "index 12 is out of bounds for dimension 2 with size 10",
            lambda: x[:, :, 12],
        )

    # 定义一个测试方法，用于测试在零维张量上的索引操作
    def test_zero_dim_index(self, device):
        # 创建一个零维张量 x，其值为 10，在指定设备上
        x = torch.tensor(10, device=device)
        # 断言从零维张量中获取单一索引时会抛出 IndexError 异常
        self.assertEqual(x, x.item())

        def runner():
            # 尝试从零维张量中获取索引 0
            print(x[0])
            return x[0]

        self.assertRaisesRegex(IndexError, "invalid index", runner)

    # 仅在 CUDA 设备上执行的测试方法
    @onlyCUDA
    def test_invalid_device(self, device):
        # 创建一个索引张量 idx，包含 [0, 1]
        idx = torch.tensor([0, 1])
        # 创建一个全零张量 b，形状为 5，在指定设备上
        b = torch.zeros(5, device=device)
        # 创建一个张量 c，包含 [1.0, 2.0]，在 CPU 上
        c = torch.tensor([1.0, 2.0], device="cpu")

        # 针对每个累加标志进行迭代
        for accumulate in [True, False]:
            # 断言在使用索引张量 idx 在 b 上进行 index_put_ 操作时会抛出 RuntimeError 异常
            self.assertRaises(
                RuntimeError,
                lambda: torch.index_put_(b, (idx,), c, accumulate=accumulate),
            )

    # 仅在 CUDA 设备上执行的测试方法
    @onlyCUDA
    def test_cpu_indices(self, device):
        # 创建一个索引张量 idx，包含 [0, 1]
        idx = torch.tensor([0, 1])
        # 创建一个全零张量 b，形状为 2，在指定设备上
        b = torch.zeros(2, device=device)
        # 创建一个全一张量 x，形状为 10，在指定设备上
        x = torch.ones(10, device=device)
        # 使用索引张量 idx 将张量 b 的值放入张量 x 中，使用 index_put_ 操作
        x[idx] = b
        # 创建一个参考张量 ref，形状为 10，在指定设备上，前两个元素为 0，其余为 1
        ref = torch.ones(10, device=device)
        ref[:2] = 0
        # 断言张量 x 等于参考张量 ref，允许的绝对误差为 0，相对误差为 0
        self.assertEqual(x, ref, atol=0, rtol=0)
        # 从张量 x 中获取索引张量 idx 的值，使用 index 操作
        out = x[idx]
        # 断言获取的 out 等于一个全零张量，允许的绝对误差为 0，相对误差为 0
        self.assertEqual(out, torch.zeros(2, device=device), atol=0, rtol=0)

    # 在特定数据类型上执行的测试装饰器
    @dtypes(torch.long, torch.float32)
    # 定义一个测试方法，用于测试 `torch.take_along_dim` 函数在不同条件下的行为
    def test_take_along_dim(self, device, dtype):
        # 定义内部函数 `_test_against_numpy`，用于与 NumPy 的 `np.take_along_axis` 函数进行比较
        def _test_against_numpy(t, indices, dim):
            # 调用 `torch.take_along_dim` 函数，获取实际结果
            actual = torch.take_along_dim(t, indices, dim=dim)
            # 将 tensor 转换为 NumPy 数组
            t_np = t.cpu().numpy()
            indices_np = indices.cpu().numpy()
            # 使用 NumPy 的 `take_along_axis` 函数计算期望结果
            expected = np.take_along_axis(t_np, indices_np, axis=dim)
            # 断言实际结果与期望结果相等
            self.assertEqual(actual, expected, atol=0, rtol=0)

        # 遍历不同的 tensor 形状
        for shape in [(3, 2), (2, 3, 5), (2, 4, 0), (2, 3, 1, 4)]:
            # 遍历是否为非连续存储的情况
            for noncontiguous in [True, False]:
                # 生成指定形状的 tensor
                t = make_tensor(
                    shape, device=device, dtype=dtype, noncontiguous=noncontiguous
                )
                # 遍历 tensor 的各个维度及 None（表示对整体排序）
                for dim in list(range(t.ndim)) + [None]:
                    if dim is None:
                        # 如果 dim 为 None，则对 tensor 展开后进行排序
                        indices = torch.argsort(t.view(-1))
                    else:
                        # 否则，在指定维度上对 tensor 进行排序
                        indices = torch.argsort(t, dim=dim)

                    # 调用比较函数，比较 `torch.take_along_dim` 和 `np.take_along_axis` 的结果
                    _test_against_numpy(t, indices, dim)

        # 测试广播操作
        t = torch.ones((3, 4, 1), device=device)
        indices = torch.ones((1, 2, 5), dtype=torch.long, device=device)

        # 调用比较函数，比较 `torch.take_along_dim` 和 `np.take_along_axis` 的结果
        _test_against_numpy(t, indices, 1)

        # 测试空 indices 的情况
        t = torch.ones((3, 4, 5), device=device)
        indices = torch.ones((3, 0, 5), dtype=torch.long, device=device)

        # 调用比较函数，比较 `torch.take_along_dim` 和 `np.take_along_axis` 的结果
        _test_against_numpy(t, indices, 1)

    # 使用装饰器指定数据类型为 torch.long 和 torch.float 的无效测试方法
    @dtypes(torch.long, torch.float)
    def test_take_along_dim_invalid(self, device, dtype):
        # 定义 tensor 的形状和指定的维度
        shape = (2, 3, 1, 4)
        dim = 0
        # 生成指定形状和数据类型的 tensor
        t = make_tensor(shape, device=device, dtype=dtype)
        # 在指定维度上对 tensor 进行排序
        indices = torch.argsort(t, dim=dim)

        # 测试当 `t` 和 `indices` 的维度不匹配时的异常情况
        with self.assertRaisesRegex(
            RuntimeError, "input and indices should have the same number of dimensions"
        ):
            torch.take_along_dim(t, indices[0], dim=0)

        # 测试 `indices` 的数据类型不合法的异常情况
        with self.assertRaisesRegex(RuntimeError, r"dtype of indices should be Long"):
            torch.take_along_dim(t, indices.to(torch.bool), dim=0)

        with self.assertRaisesRegex(RuntimeError, r"dtype of indices should be Long"):
            torch.take_along_dim(t, indices.to(torch.float), dim=0)

        with self.assertRaisesRegex(RuntimeError, r"dtype of indices should be Long"):
            torch.take_along_dim(t, indices.to(torch.int32), dim=0)

        # 测试维度超出范围的异常情况
        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            torch.take_along_dim(t, indices, dim=-7)

        with self.assertRaisesRegex(IndexError, "Dimension out of range"):
            torch.take_along_dim(t, indices, dim=7)

    # 使用装饰器指定仅在 CUDA 上运行的测试方法，数据类型为 torch.float
    @onlyCUDA
    @dtypes(torch.float)
    def test_gather_take_along_dim_cross_device(self, device, dtype):
        # 定义张量的形状
        shape = (2, 3, 1, 4)
        # 指定操作的维度
        dim = 0
        # 创建指定设备上的张量
        t = make_tensor(shape, device=device, dtype=dtype)
        # 对张量按指定维度进行排序，返回排序后的索引
        indices = torch.argsort(t, dim=dim)

        # 测试torch.gather函数在不同设备上执行时是否引发RuntimeError异常
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            torch.gather(t, 0, indices.cpu())

        # 测试torch.take_along_dim函数在不符合预期条件下是否引发RuntimeError异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected tensor to have .* but got tensor with .* torch.take_along_dim()",
        ):
            torch.take_along_dim(t, indices.cpu(), dim=0)

        # 测试torch.gather函数在不同设备上执行时是否引发RuntimeError异常
        with self.assertRaisesRegex(
            RuntimeError, "Expected all tensors to be on the same device"
        ):
            torch.gather(t.cpu(), 0, indices)

        # 测试torch.take_along_dim函数在不符合预期条件下是否引发RuntimeError异常
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected tensor to have .* but got tensor with .* torch.take_along_dim()",
        ):
            torch.take_along_dim(t.cpu(), indices, dim=0)

    @onlyCUDA
    def test_cuda_broadcast_index_use_deterministic_algorithms(self, device):
        # 启用确定性算法保护上下文
        with DeterministicGuard(True):
            # 定义不同的索引张量
            idx1 = torch.tensor([0])
            idx2 = torch.tensor([2, 6])
            idx3 = torch.tensor([1, 5, 7])

            # 创建并复制张量到指定设备
            tensor_a = torch.rand(13, 11, 12, 13, 12).cpu()
            tensor_b = tensor_a.to(device=device)
            # 使用索引修改张量的部分元素值
            tensor_a[idx1] = 1.0
            tensor_a[idx1, :, idx2, idx2, :] = 2.0
            tensor_a[:, idx1, idx3, :, idx3] = 3.0
            tensor_b[idx1] = 1.0
            tensor_b[idx1, :, idx2, idx2, :] = 2.0
            tensor_b[:, idx1, idx3, :, idx3] = 3.0
            # 检查两个张量在CPU上的值是否一致
            self.assertEqual(tensor_a, tensor_b.cpu(), atol=0, rtol=0)

            # 创建并复制张量到指定设备
            tensor_a = torch.rand(10, 11).cpu()
            tensor_b = tensor_a.to(device=device)
            # 使用索引修改张量的部分元素值
            tensor_a[idx3] = 1.0
            tensor_a[idx2, :] = 2.0
            tensor_a[:, idx2] = 3.0
            tensor_a[:, idx1] = 4.0
            tensor_b[idx3] = 1.0
            tensor_b[idx2, :] = 2.0
            tensor_b[:, idx2] = 3.0
            tensor_b[:, idx1] = 4.0
            # 检查两个张量在CPU上的值是否一致
            self.assertEqual(tensor_a, tensor_b.cpu(), atol=0, rtol=0)

            # 创建并复制张量到指定设备
            tensor_a = torch.rand(10, 10).cpu()
            tensor_b = tensor_a.to(device=device)
            # 使用索引修改张量的部分元素值
            tensor_a[[8]] = 1.0
            tensor_b[[8]] = 1.0
            # 检查两个张量在CPU上的值是否一致
            self.assertEqual(tensor_a, tensor_b.cpu(), atol=0, rtol=0)

            # 创建并复制张量到指定设备
            tensor_a = torch.rand(10).cpu()
            tensor_b = tensor_a.to(device=device)
            # 使用索引修改张量的部分元素值
            tensor_a[6] = 1.0
            tensor_b[6] = 1.0
            # 检查两个张量在CPU上的值是否一致
            self.assertEqual(tensor_a, tensor_b.cpu(), atol=0, rtol=0)

    def test_index_limits(self, device):
        # 对空张量进行测试，验证是否会引发IndexError异常
        t = torch.tensor([], device=device)
        # 获取64位整型的最小值和最大值
        idx_min = torch.iinfo(torch.int64).min
        idx_max = torch.iinfo(torch.int64).max
        # 使用lambda表达式验证是否会引发IndexError异常
        self.assertRaises(IndexError, lambda: t[idx_min])
        self.assertRaises(IndexError, lambda: t[idx_max])
# NumPy 测试类，继承自 Python 的 TestCase 类，用于定义和执行 NumPy 库的单元测试
class NumpyTests(TestCase):
    # 定义测试函数，用于测试索引操作中不允许使用浮点数的情况
    def test_index_no_floats(self, device):
        # 创建一个张量 `a`，包含一个浮点数 5.0，指定设备为参数 `device`
        a = torch.tensor([[[5.0]]], device=device)

        # 以下每行代码都测试引发 IndexError 异常，因为使用了浮点数索引，这在 PyTorch 中是不允许的
        self.assertRaises(IndexError, lambda: a[0.0])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[0, 0.0])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[0.0, 0])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[0.0, :])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[:, 0.0])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[:, 0.0, :])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[0.0, :, :])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[0, 0, 0.0])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[0.0, 0, 0])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[0, 0.0, 0])  # 索引值为浮点数 0.0
        self.assertRaises(IndexError, lambda: a[-1.4])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[0, -1.4])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[-1.4, 0])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[-1.4, :])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[:, -1.4])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[:, -1.4, :])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[-1.4, :, :])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[0, 0, -1.4])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[-1.4, 0, 0])  # 索引值为浮点数 -1.4
        self.assertRaises(IndexError, lambda: a[0, -1.4, 0])  # 索引值为浮点数 -1.4
        # 下面两行代码被注释掉，因为它们会引发语法错误，并不是实际测试的一部分
        # self.assertRaises(IndexError, lambda: a[0.0:, 0.0])
        # self.assertRaises(IndexError, lambda: a[0.0:, 0.0,:])
    # 测试用例：测试使用省略号索引操作符在PyTorch张量上的行为
    def test_ellipsis_index(self, device):
        # 创建一个张量a，包含3行3列的数据，存储在指定设备上
        a = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
        # 断言省略号索引不是同一对象
        self.assertIsNot(a[...], a)
        # 断言省略号索引和原张量a相等
        self.assertEqual(a[...], a)
        # 断言省略号索引的数据指针和原张量a的数据指针相同
        self.assertEqual(a[...].data_ptr(), a.data_ptr())

        # 使用省略号进行切片可以跳过任意数量的维度
        self.assertEqual(a[0, ...], a[0])
        self.assertEqual(a[0, ...], a[0, :])
        self.assertEqual(a[..., 0], a[:, 0])

        # 在NumPy中，使用省略号进行切片结果是0维数组。在PyTorch中，我们没有单独的0维数组和标量。
        self.assertEqual(a[0, ..., 1], torch.tensor(2, device=device))

        # 对0维数组使用`(Ellipsis,)`进行赋值
        b = torch.tensor(1)
        b[(Ellipsis,)] = 2
        self.assertEqual(b, 2)
    def test_boolean_assignment_value_mismatch(self, device):
        # 测试布尔赋值在数值形状无法广播到订阅时应该失败。参见 gh-3458
        # 创建一个在指定设备上的张量，包含从0到3的整数
        a = torch.arange(0, 4, device=device)

        def f(a, v):
            # 将条件为真的元素用给定的张量 v 替换
            a[a > -1] = tensor(v).to(device)

        # 断言调用 f 函数时应抛出异常并包含 "shape mismatch" 字符串
        self.assertRaisesRegex(Exception, "shape mismatch", f, a, [])
        self.assertRaisesRegex(Exception, "shape mismatch", f, a, [1, 2, 3])
        self.assertRaisesRegex(Exception, "shape mismatch", f, a[:1], [1, 2, 3])

    def test_boolean_indexing_twodim(self, device):
        # 使用二维布尔数组索引二维数组
        # 创建一个在指定设备上的张量，包含指定的整数二维数组
        a = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
        # 创建一个在指定设备上的张量，包含指定的二维布尔数组
        b = tensor(
            [[True, False, True], [False, True, False], [True, False, True]],
            device=device,
        )
        # 断言通过布尔索引得到的结果张量与预期结果相等
        self.assertEqual(a[b], tensor([1, 3, 5, 7, 9], device=device))
        self.assertEqual(a[b[1]], tensor([[4, 5, 6]], device=device))
        self.assertEqual(a[b[0]], a[b[2]])

        # 布尔赋值
        a[b] = 0
        # 断言修改后的张量与预期结果相等
        self.assertEqual(a, tensor([[0, 2, 0], [4, 0, 6], [0, 8, 0]], device=device))

    def test_boolean_indexing_weirdness(self, device):
        # 处理奇怪的布尔索引情况
        # 创建一个在指定设备上的全为1的三维张量
        a = torch.ones((2, 3, 4), device=device)
        # 断言使用 False, True, ... 的布尔索引得到的张量形状符合预期
        self.assertEqual((0, 2, 3, 4), a[False, True, ...].shape)
        # 断言使用 True, [0, 1], True, True, [1], [[2]] 的复杂布尔索引得到的张量与预期结果相等
        self.assertEqual(
            torch.ones(1, 2, device=device), a[True, [0, 1], True, True, [1], [[2]]]
        )
        # 断言使用 False, [0, 1], ... 的索引会抛出 IndexError 异常
        self.assertRaises(IndexError, lambda: a[False, [0, 1], ...])

    def test_boolean_indexing_weirdness_tensors(self, device):
        # 处理奇怪的布尔索引情况，使用张量定义 True 和 False
        false = torch.tensor(False, device=device)
        true = torch.tensor(True, device=device)
        # 创建一个在指定设备上的全为1的三维张量
        a = torch.ones((2, 3, 4), device=device)
        # 断言使用 False, True, ... 的布尔索引得到的张量形状符合预期
        self.assertEqual((0, 2, 3, 4), a[False, True, ...].shape)
        # 断言使用 true, [0, 1], true, true, [1], [[2]] 的复杂布尔索引得到的张量与预期结果相等
        self.assertEqual(
            torch.ones(1, 2, device=device), a[true, [0, 1], true, true, [1], [[2]]]
        )
        # 断言使用 false, [0, 1], ... 的索引会抛出 IndexError 异常
        self.assertRaises(IndexError, lambda: a[false, [0, 1], ...])

    def test_boolean_indexing_alldims(self, device):
        true = torch.tensor(True, device=device)
        # 创建一个在指定设备上的全为1的二维张量
        a = torch.ones((2, 3), device=device)
        # 断言使用 True, True 的布尔索引得到的张量形状符合预期
        self.assertEqual((1, 2, 3), a[True, True].shape)
        self.assertEqual((1, 2, 3), a[true, true].shape)

    def test_boolean_list_indexing(self, device):
        # 使用布尔列表索引二维数组
        # 创建一个在指定设备上的张量，包含指定的整数二维数组
        a = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device)
        # 创建布尔列表 b 和 c
        b = [True, False, False]
        c = [True, True, False]
        # 断言使用布尔列表 b 和 c 的索引得到的张量与预期结果相等
        self.assertEqual(a[b], tensor([[1, 2, 3]], device=device))
        self.assertEqual(a[b, b], tensor([1], device=device))
        self.assertEqual(a[c], tensor([[1, 2, 3], [4, 5, 6]], device=device))
        self.assertEqual(a[c, c], tensor([1, 5], device=device))
    def test_everything_returns_views(self, device):
        # 创建一个张量 `a`，包含一个元素 [5]，使用指定的设备
        a = tensor([5], device=device)

        # 断言：确保 `a` 不是其自身的零切片
        self.assertIsNot(a, a[()])

        # 断言：确保 `a` 不是其自身的省略号切片
        self.assertIsNot(a, a[...])

        # 断言：确保 `a` 不是其自身的全切片
        self.assertIsNot(a, a[:])

    def test_broaderrors_indexing(self, device):
        # 创建一个形状为 5x5 的全零张量 `a`，使用指定的设备
        a = torch.zeros(5, 5, device=device)

        # 断言：调用 `a.__getitem__` 时，期望引发 IndexError 异常，错误信息为 "shape mismatch"
        self.assertRaisesRegex(
            IndexError, "shape mismatch", a.__getitem__, ([0, 1], [0, 1, 2])
        )

        # 断言：调用 `a.__setitem__` 时，期望引发 IndexError 异常，错误信息为 "shape mismatch"
        self.assertRaisesRegex(
            IndexError, "shape mismatch", a.__setitem__, ([0, 1], [0, 1, 2]), 0
        )

    def test_trivial_fancy_out_of_bounds(self, device):
        # 创建一个形状为 5 的全零张量 `a`，使用指定的设备
        a = torch.zeros(5, device=device)

        # 创建一个长度为 20 的全一张量 `ind`，数据类型为 torch.int64，使用指定的设备
        ind = torch.ones(20, dtype=torch.int64, device=device)

        # 如果张量 `a` 在 CUDA 设备上，则跳过测试，抛出 unittest.SkipTest 异常
        if a.is_cuda:
            raise unittest.SkipTest("CUDA asserts instead of raising an exception")

        # 修改张量 `ind` 中倒数第一个元素为 10
        ind[-1] = 10

        # 断言：调用 `a.__getitem__` 时，期望引发 IndexError 异常
        self.assertRaises(IndexError, a.__getitem__, ind)

        # 断言：调用 `a.__setitem__` 时，期望引发 IndexError 异常
        self.assertRaises(IndexError, a.__setitem__, ind, 0)

        # 重新创建一个长度为 20 的全一张量 `ind`，数据类型为 torch.int64，使用指定的设备
        ind = torch.ones(20, dtype=torch.int64, device=device)

        # 修改张量 `ind` 中第一个元素为 11
        ind[0] = 11

        # 断言：调用 `a.__getitem__` 时，期望引发 IndexError 异常
        self.assertRaises(IndexError, a.__getitem__, ind)

        # 断言：调用 `a.__setitem__` 时，期望引发 IndexError 异常
        self.assertRaises(IndexError, a.__setitem__, ind, 0)

    def test_index_is_larger(self, device):
        # 创建一个形状为 (5, 5) 的全零张量 `a`，使用指定的设备
        a = torch.zeros((5, 5), device=device)

        # 对 `a` 的指定位置应用复杂索引广播赋值
        a[[[0], [1], [2]], [0, 1, 2]] = tensor([2.0, 3.0, 4.0], device=device)

        # 断言：确保 `a[:3, :3]` 中的所有元素都等于张量 [2.0, 3.0, 4.0]，使用指定的设备
        self.assertTrue((a[:3, :3] == tensor([2.0, 3.0, 4.0], device=device)).all())

    def test_broadcast_subspace(self, device):
        # 创建一个形状为 (100, 100) 的全零张量 `a`，使用指定的设备
        a = torch.zeros((100, 100), device=device)

        # 创建一个张量 `v`，包含从 0 到 99 的浮点数，每个值作为列向量，使用指定的设备
        v = torch.arange(0.0, 100, device=device)[:, None]

        # 创建一个从 99 到 0 的长整型张量 `b`，使用指定的设备
        b = torch.arange(99, -1, -1, device=device).long()

        # 使用张量 `v` 对张量 `a` 的子空间进行广播赋值
        a[b] = v

        # 创建一个预期结果的张量 `expected`，其形状为 (100, 100)，使用指定的设备
        expected = b.float().unsqueeze(1).expand(100, 100)

        # 断言：确保张量 `a` 等于预期的张量 `expected`
        self.assertEqual(a, expected)

    def test_truncate_leading_1s(self, device):
        # 创建一个形状为 (1, 4) 的随机张量 `col_max`
        col_max = torch.randn(1, 4)

        # 复制张量 `col_max` 生成 `kernel`，形状为 [4, 4]
        kernel = col_max.T * col_max

        # 设置 `kernel` 的对角线元素为 `col_max` 的平方
        kernel[range(len(kernel)), range(len(kernel))] = torch.square(col_max)

        # 复制 `col_max` 的视图到 `kernel2`
        kernel2 = kernel.clone()

        # 使用 `torch.square(col_max.view(4))` 设置 `kernel2` 的对角线元素
        torch.diagonal(kernel2).copy_(torch.square(col_max.view(4)))

        # 断言：确保 `kernel` 等于 `kernel2`
        self.assertEqual(kernel, kernel2)
# 使用函数 instantiate_device_type_tests 实例化 TestIndexing 类的测试，并将其添加到全局命名空间中，排除 "meta" 测试
instantiate_device_type_tests(TestIndexing, globals(), except_for="meta")

# 使用函数 instantiate_device_type_tests 实例化 NumpyTests 类的测试，并将其添加到全局命名空间中，排除 "meta" 测试
instantiate_device_type_tests(NumpyTests, globals(), except_for="meta")

# 如果当前脚本作为主程序运行，则执行测试函数 run_tests()
if __name__ == "__main__":
    run_tests()
```