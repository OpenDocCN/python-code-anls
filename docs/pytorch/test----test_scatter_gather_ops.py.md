# `.\pytorch\test\test_scatter_gather_ops.py`

```
# Owner(s): ["module: scatter & gather ops"]

# 引入必要的库和模块
import random

import torch

# 引入测试相关的函数和类
from torch.testing import make_tensor
from torch.testing._internal.common_utils import \
    (parametrize, run_tests, TestCase, DeterministicGuard)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCPU, dtypes, dtypesIfCUDA,
     toleranceOverride, tol,)
from torch.testing._internal.common_dtype import \
    (get_all_dtypes,)

# 防止包含文件意外设置默认数据类型
assert torch.get_default_dtype() is torch.float32


# Note: test_scatter_gather_ops.py
# 这个测试文件测试scatter和gather等操作，
#   如torch.scatter和torch.gather。

# 测试类，测试scatter和gather操作
class TestScatterGather(TestCase):

    # 填充索引张量以包含有效索引
    def _fill_indices(self, idx, dim, dim_size, elems_per_row, m, n, o, unique_indices=True):
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, idx.size(dim) + 1)
                    if unique_indices:
                        idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]
                    else:
                        idx[tuple(ii)] = torch.randint(dim_size, (elems_per_row,))

    # 使用指定数据类型进行gather操作的测试
    @dtypes(torch.float32, torch.complex64)
    def test_gather(self, device, dtype):
        # 随机生成尺寸和参数
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        # 创建随机数据源和索引张量
        src = make_tensor((m, n, o), device=device, dtype=dtype)
        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = make_tensor(idx_size, device=device, dtype=torch.long)
        self._fill_indices(idx, dim, src.size(dim), elems_per_row, m, n, o)

        # 执行torch.gather操作并验证结果
        actual = torch.gather(src, dim, idx)
        expected = torch.zeros(idx_size, device=device, dtype=dtype)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[i, j, k] = src[tuple(ii)]
        self.assertEqual(actual, expected, atol=0, rtol=0)

        # 由于复数类型不支持torch.max，因此需加以保护
        if not dtype.is_complex:
            src = make_tensor((3, 4, 5), device=device, dtype=dtype)
            expected, idx = src.max(2, True)
            actual = torch.gather(src, 2, idx)
            self.assertEqual(actual, expected, atol=0, rtol=0)

    # 使用布尔类型进行测试
    @dtypes(torch.bool)
    # 测试 torch.gather 函数对布尔类型数据的功能
    def test_gather_bool(self, device, dtype):
        # 创建一个布尔类型的张量作为源数据
        src = torch.tensor(((False, True), (True, True)), device=device, dtype=dtype)
        # 创建一个索引张量，用于指定从源数据中收集的元素
        idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
        # 使用 torch.gather 函数根据索引在源数据中收集元素
        actual = torch.gather(src, 1, idx)
        # 期望的结果，根据索引收集到的元素
        expected = torch.tensor(((False, False), (True, True)), device=device, dtype=dtype)
        # 断言实际结果与期望结果相等
        self.assertEqual(actual, expected, atol=0, rtol=0)

    # 参数化测试函数，测试 torch.gather 函数在使用空索引张量时的反向传播行为
    @parametrize("sparse_grad", [False, True])
    @dtypes(torch.float32, torch.float64)
    def test_gather_backward_with_empty_index_tensor(self, device, dtype, sparse_grad):
        # 指定操作的维度
        dim = -1
        # 创建一个随机张量作为输入，要求其梯度计算
        input = torch.rand([10, 5], dtype=dtype, device=device, requires_grad=True)
        # 创建一个空的索引张量
        index = torch.randint(0, 2, [3, 0], dtype=torch.int64, device=device)
        # 使用 torch.gather 函数在指定维度上根据索引收集元素，同时考虑稀疏梯度选项
        res = torch.gather(input, dim, index, sparse_grad=sparse_grad)
        # 对结果求和并进行反向传播
        res.sum().backward()
        # 获取输入张量的梯度，并转换为稠密格式（如果使用了稀疏梯度选项）
        grad = input.grad.to_dense() if sparse_grad else input.grad
        # 期望的梯度，全为零
        expected_grad = torch.zeros_like(input, requires_grad=False)
        # 断言实际计算得到的梯度与期望的全零梯度相等
        self.assertEqual(grad, expected_grad, atol=0, rtol=0)

    # 参数化测试函数，测试 torch.Tensor.scatter_ 函数的基础功能
    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_(self, device, dtype):
        # 循环测试是否具有确定性行为和非确定性行为
        for deterministic in [False, True]:
            # 使用 DeterministicGuard 确保当前上下文的确定性状态
            with DeterministicGuard(deterministic):
                # 调用基础的 scatter 函数测试方法
                self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                        is_scalar=False, reduction=None)

    # 参数化测试函数，测试 torch.Tensor.scatter_ 函数在处理标量数据时的功能
    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__scalar(self, device, dtype):
        # 调用基础的 scatter 函数测试方法，处理标量数据的情况
        self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                is_scalar=True, reduction=None)

    # FIXME: 运行时错误提示 "cuda_scatter_gather_base_kernel_reduce_multiply" 未对 'ComplexFloat' 类型实现
    @toleranceOverride({torch.float16: tol(atol=1e-2, rtol=0)})
    @dtypesIfCUDA(torch.float16, torch.float32)
    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__reductions(self, device, dtype):
        # 循环测试不同的减少操作类型
        for reduction in ("add", "multiply"):
            # 调用基础的 scatter 函数测试方法，处理非标量数据和不同减少操作类型的情况
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=False, reduction=reduction)
            # 再次调用基础的 scatter 函数测试方法，处理标量数据和不同减少操作类型的情况
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=True, reduction=reduction)

    # 参数化测试函数，测试 torch.Tensor.scatter_add_ 函数的基础功能
    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_add_(self, device, dtype):
        # 循环测试是否具有确定性行为和非确定性行为
        for deterministic in [False, True]:
            # 使用 DeterministicGuard 确保当前上下文的确定性状态
            with DeterministicGuard(deterministic):
                # 调用基础的 scatter_add 函数测试方法
                self._test_scatter_base(torch.Tensor.scatter_add_, device=device, dtype=dtype,
                                        is_scalar=False, reduction=None)

    # 参数化测试函数，仅测试 torch.float32 类型的情况
    @dtypes(torch.float32)
    # 定义测试函数，用于测试 scatter_add_mult_index_base 方法
    def test_scatter_add_mult_index_base(self, device, dtype):
        # 循环遍历 deterministic 参数的值：False 和 True
        for deterministic in [False, True]:
            # 根据 deterministic 参数设置上下文管理器 DeterministicGuard
            with DeterministicGuard(deterministic):
                # 定义 m 和 n 的值分别为 30 和 40
                m, n = 30, 40
                # 创建一个大小为 m x n 的零张量 idx，设备为 device，数据类型为 torch.long
                idx = torch.zeros(m, n, device=device, dtype=torch.long)
                # 创建一个大小为 m x n 的全一张量 src，设备为 device，数据类型为 dtype
                src = torch.ones(m, n, device=device, dtype=dtype)
                # 在一个大小为 m x n 的零张量上执行 scatter_add_ 操作，沿着维度 0，索引为 idx，值为 src
                res0 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(0, idx, src)
                # 在一个大小为 m x n 的零张量上执行 scatter_add_ 操作，沿着维度 1，索引为 idx，值为 src
                res1 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(1, idx, src)

                # 断言结果 res0 的第一行应为 m 倍的全一张量，设备为 device，数据类型为 dtype，容差为 0
                self.assertEqual(res0[0, :], m * torch.ones(n, device=device, dtype=dtype), atol=0, rtol=0)
                # 断言结果 res1 的第一列应为 n 倍的全一张量，设备为 device，数据类型为 dtype，容差为 0
                self.assertEqual(res1[:, 0], n * torch.ones(m, device=device, dtype=dtype), atol=0, rtol=0)

    # FIXME: discrepancy between bool ReduceAdd on CUDA and CPU (a + b on CPU and buggy a && b on CUDA)
    # 根据数据类型 dtype 参数执行 scatter_reduce_ 方法的测试，使用 reduction='sum' 方式
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    def test_scatter_reduce_sum(self, device, dtype):
        # 循环遍历 include_self 参数的值：True 和 False
        for include_self in (True, False):
            # 循环遍历 deterministic 参数的值：False 和 True
            for deterministic in [False, True]:
                # 根据 deterministic 参数设置上下文管理器 DeterministicGuard
                with DeterministicGuard(deterministic):
                    # 调用 _test_scatter_base 方法进行 scatter_reduce_ 方法的基础测试
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='sum', unique_indices=False,
                                            include_self=include_self)

    # 根据数据类型 dtype 参数执行 scatter_reduce_ 方法的测试，使用 reduction='prod' 方式
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_prod(self, device, dtype):
        # 循环遍历 include_self 参数的值：True 和 False
        for include_self in (True, False):
            # 调用 _test_scatter_base 方法进行 scatter_reduce_ 方法的基础测试
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='prod', unique_indices=False,
                                    include_self=include_self)

    # 根据数据类型 dtype 参数执行 scatter_reduce_ 方法的测试，使用 reduction='mean' 方式
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_mean(self, device, dtype):
        # 循环遍历 include_self 参数的值：True 和 False
        for include_self in (True, False):
            # 循环遍历 deterministic 参数的值：False 和 True
            for deterministic in [False, True]:
                # 根据 deterministic 参数设置上下文管理器 DeterministicGuard
                with DeterministicGuard(deterministic):
                    # 调用 _test_scatter_base 方法进行 scatter_reduce_ 方法的基础测试
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='mean', unique_indices=False,
                                            include_self=include_self)

    # 根据数据类型 dtype 参数执行 scatter_reduce_ 方法的测试，使用 reduction='sum' 方式
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    # 定义一个测试方法，用于测试 scatter_reduce_ 方法的 'amax' 情况
    def test_scatter_reduce_amax(self, device, dtype):
        # 遍历 include_self 参数的 True 和 False 两种情况
        for include_self in (True, False):
            # 调用 _test_scatter_base 方法进行 scatter_reduce_ 方法的基本测试
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amax', unique_indices=False,
                                    include_self=include_self)
            # 对 nan/inf 传播进行简单测试
            if (dtype.is_floating_point):
                # 创建一个设备为 device，dtype 为 dtype 的零张量 input
                input = torch.zeros(3, device=device, dtype=dtype)
                # 创建一个包含特定值的源张量 src，其中包括 nan 和 inf
                src = torch.tensor([1, float('nan'), -float('inf'), -float('inf'), 2, float('inf')], device=device, dtype=dtype)
                # 创建索引张量 idx，用于 scatter 操作
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                # 在 input 上进行 scatter_reduce_ 操作，使用 'amax' 作为 reduction 方法
                input.scatter_reduce_(0, idx, src, 'amax', include_self=include_self)
                # 创建预期结果张量 expected_result，根据 include_self 参数调整预期结果
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[1] = 0
                # 断言 input 是否等于 expected_result
                self.assertEqual(input, expected_result)

    # 定义一个测试方法，用于测试 scatter_reduce_ 方法的 'amin' 情况
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_amin(self, device, dtype):
        # 遍历 include_self 参数的 True 和 False 两种情况
        for include_self in (True, False):
            # 调用 _test_scatter_base 方法进行 scatter_reduce_ 方法的基本测试
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amin', unique_indices=False,
                                    include_self=include_self)
            # 对 nan/inf 传播进行简单测试
            if (dtype.is_floating_point):
                # 创建一个设备为 device，dtype 为 dtype 的零张量 input
                input = torch.zeros(3, device=device, dtype=dtype)
                # 创建一个包含特定值的源张量 src，其中包括 nan 和 inf
                src = torch.tensor([1, float('nan'), -2, -float('inf'), float('inf'), float('inf')], device=device, dtype=dtype)
                # 创建索引张量 idx，用于 scatter 操作
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                # 在 input 上进行 scatter_reduce_ 操作，使用 'amin' 作为 reduction 方法
                input.scatter_reduce_(0, idx, src, 'amin', include_self=include_self)
                # 创建预期结果张量 expected_result，根据 include_self 参数调整预期结果
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[2] = 0
                # 断言 input 是否等于 expected_result
                self.assertEqual(input, expected_result)

    # 仅在 CPU 上运行的装饰器
    @onlyCPU
    # 指定测试类型为 float32、float64、bfloat16、float16 的数据类型
    @dtypes(torch.float32, torch.float64, torch.bfloat16, torch.float16)
    def test_scatter_expanded_index(self, device, dtype):
        # 定义一个辅助函数，用于测试 scatter 操作中索引扩展的情况
        def helper(input_size, idx_size):
            # 生成指定大小的随机张量 input，并将其转移到指定设备并指定数据类型
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            # 克隆 input 以备后用
            input2 = input.clone()

            # 构造一个形状与 input_size 相同的全 1 列表，并将其第一个元素设置为 idx_size
            shape = [1] * len(input_size)
            shape[0] = idx_size
            # 获取 input_size 的第一个维度大小
            dim_size = input_size[0]
            # 生成一个在 [0, dim_size) 范围内的随机整数张量，形状为 shape
            idx = torch.randint(0, dim_size, shape)

            # 当索引值在 (1, 4) 之间时，将其置为 0，以创建一些空行
            mask = (idx > 1) * (idx < 4)
            idx[mask] = 0

            # 将 idx 扩展为 expanded_shape 的形状，并保证其连续性
            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()
            # 生成一个与 expanded_shape 相同大小的随机张量 src
            src = torch.randn(expanded_shape, device=device).to(dtype=dtype)

            # 在 input 上执行 scatter_add 操作，将 src 按照 idx 指定的位置累加到 input 中
            out = input.scatter_add(0, idx, src)
            out2 = input2.scatter_add(0, idx2, src)
            # 断言两个 scatter_add 操作的结果相等
            self.assertEqual(out, out2)

            # 遍历多种 reduce 操作和 include_self 的组合
            for reduce in ["sum", "prod", "mean", "amax", "amin"]:
                for include_self in [True, False]:
                    # 在 input 上执行 scatter_reduce 操作，将 src 按照 idx 指定的位置根据指定 reduce 方法聚合
                    out = input.scatter_reduce(0, idx, src, reduce=reduce, include_self=include_self)
                    out2 = input2.scatter_reduce(0, idx2, src, reduce=reduce, include_self=include_self)
                    # 断言两个 scatter_reduce 操作的结果相等
                    self.assertEqual(out, out2)

        # 分别测试不同的 input_size 和 idx_size 组合
        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)

    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    def test_gather_expanded_index(self, device, dtype):
        # 当索引是 [N, 1] 时，其步长为 [1, 0]，在索引扩展时应排除快速路径
        # 使用 torch.arange 生成一个 5x5 的张量 input
        input = torch.arange(25).view(5, 5)
        input2 = input.to(dtype=dtype)

        # 使用 torch.arange 生成一个形状为 [5, 1] 的索引张量 idx
        idx = torch.arange(5).view(5, 1)
        # 在 input 上执行 gather 操作，根据 idx 中的索引从 input 中聚集数据
        out = torch.gather(input, 0, idx)
        out2 = torch.gather(input2, 0, idx)

        # 断言两个 gather 操作的结果相等，并转换为指定的数据类型 dtype
        self.assertEqual(out.to(dtype=dtype), out2)

        # 定义一个辅助函数，用于测试 gather 操作中索引扩展的情况
        def helper(input_size, idx_size):
            # 生成指定大小的随机张量 input，并将其转移到指定设备并指定数据类型
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            # 克隆 input 以备后用
            input2 = input.clone()

            # 构造一个形状与 input_size 相同的全 1 列表，并将其第一个元素设置为 idx_size
            shape = [1] * len(input_size)
            shape[0] = idx_size
            # 获取 input_size 的第一个维度大小
            dim_size = input_size[0]
            # 生成一个在 [0, dim_size) 范围内的随机整数张量，形状为 shape
            idx = torch.randint(0, dim_size, shape)

            # 将 idx 扩展为 expanded_shape 的形状，并保证其连续性
            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()

            # 在 input 上执行 gather 操作，根据 idx 中的索引从 input 中聚集数据
            out = torch.gather(input, 0, idx)
            out2 = torch.gather(input2, 0, idx2)

            # 断言两个 gather 操作的结果相等
            self.assertEqual(out, out2)

        # 分别测试不同的 input_size 和 idx_size 组合
        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)
# 导入通用设备测试框架，并实例化测试用例 TestScatterGather，参考
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
#   获取更多细节。
instantiate_device_type_tests(TestScatterGather, globals())

# 如果作为主程序运行，则执行测试。
if __name__ == '__main__':
    run_tests()
```