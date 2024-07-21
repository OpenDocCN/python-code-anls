# `.\pytorch\test\test_linalg.py`

```py
# Owner(s): ["module: linear algebra"]

# 导入必要的库
import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库

import unittest  # 导入单元测试模块
import itertools  # 导入迭代工具模块
import warnings  # 导入警告模块
import math  # 导入数学函数模块
from math import inf, nan, isnan  # 导入特定数学常量和函数
import random  # 导入随机数生成模块
from random import randrange  # 导入特定的随机数生成函数
from itertools import product  # 导入迭代工具中的笛卡尔积函数
from functools import reduce, partial  # 导入函数式编程工具

# 导入PyTorch内部测试工具中的函数和类
from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_SCIPY, IS_MACOS, IS_WINDOWS, slowTest,
     TEST_WITH_ROCM, IS_FBCODE, IS_REMOTE_GPU, iter_indices,
     make_fullrank_matrices_with_distinct_singular_values,
     freeze_rng_state, IS_ARM64, IS_SANDCASTLE, TEST_OPT_EINSUM, parametrize, skipIfTorchDynamo,
     setBlasBackendsToDefaultFinally, setLinalgBackendsToDefaultFinally, serialTest,
     xfailIfTorchDynamo)

# 导入PyTorch测试设备类型相关的函数和类
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, has_cusolver, has_hipsolver,
     onlyCPU, skipCUDAIf, skipCUDAIfNoMagma, skipCPUIfNoLapack, precisionOverride,
     skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, onlyNativeDeviceTypes, dtypesIfCUDA,
     onlyCUDA, skipCUDAVersionIn, skipMeta, skipCUDAIfNoCusolver, skipCUDAIfNotRocm,
     dtypesIfMPS, largeTensorTest)

# 导入PyTorch张量生成工具
from torch.testing import make_tensor

# 导入PyTorch内部测试中关于数据类型的函数
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex_and, floating_and_complex_types, integral_types,
    floating_and_complex_types_and, floating_types_and, complex_types,
)

# 导入PyTorch内部测试中关于CUDA的函数
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, SM90OrLater, tf32_on_and_off, _get_magma_version, \
    _get_torch_cuda_version

# 导入PyTorch内部测试中关于量化的函数
from torch.testing._internal.common_quantization import _group_quantize_tensor, _dynamically_quantize_per_channel

# 导入PyTorch内部测试中关于MKLDNN的函数
from torch.testing._internal.common_mkldnn import bf32_on_and_off

# 导入二项分布
from torch.distributions.binomial import Binomial

# 导入PyTorch的opt_einsum后端
import torch.backends.opt_einsum as opt_einsum

import operator  # 导入运算符模块

# 确保默认数据类型不会意外地被更改
assert torch.get_default_dtype() is torch.float32

# 如果需要测试SciPy，则导入SciPy库
if TEST_SCIPY:
    import scipy

# 函数用于检查是否支持BLAS LT的设备
def blaslt_supported_device():
    if torch.cuda.is_available():  # 如果CUDA可用
        if torch.version.hip:  # 如果是HIP环境
            for arch in ['gfx90a', 'gfx94']:  # 遍历指定的GPU架构
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True  # 如果设备支持指定的GPU架构，则返回True
        else:  # 如果不是HIP环境
            return True  # 直接返回True
    return False  # 如果以上条件都不满足，则返回False

# 测试线性代数相关功能的测试类
class TestLinalg(TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        torch.backends.cuda.matmul.allow_tf32 = False  # 设置CUDA矩阵乘法不允许使用TF32

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True  # 恢复CUDA矩阵乘法允许使用TF32
        super(self.__class__, self).tearDown()  # 调用父类的tearDown方法

    exact_dtype = True  # 设置精确的数据类型检查

    @dtypes(torch.float, torch.cfloat)  # 测试使用的数据类型
    @precisionOverride({torch.float: 1e-06, torch.cfloat: 1e-06})  # 设置精度覆盖
    @tf32_on_and_off(5e-3)  # 设置TF32开启和关闭
    @bf32_on_and_off(5e-3)  # 设置BF16开启和关闭
    # 定义内部测试函数，接受设备和数据类型参数
    def test_inner(self, device, dtype):
        # 定义内部函数check，用于检查不同维度组合下的torch.inner运算结果与numpy.inner的一致性
        def check(a_sizes_, b_sizes_):
            # 对a_sizes和b_sizes进行迭代，交换顺序以确保对称性
            for a_sizes, b_sizes in ((a_sizes_, b_sizes_), (b_sizes_, a_sizes_)):
                # 生成指定维度大小的随机张量a和b
                a = torch.randn(a_sizes, dtype=dtype, device=device)
                b = torch.randn(b_sizes, dtype=dtype, device=device)
                # 计算torch.inner的结果
                res = torch.inner(a, b)
                # 计算numpy.inner的结果（在CPU上执行）
                ref = np.inner(a.cpu().numpy(), b.cpu().numpy())
                # 断言torch.inner和numpy.inner的结果是否相等
                self.assertEqual(res.cpu(), torch.from_numpy(np.array(ref)))
                # 创建一个和res相同大小的零张量out，用于接收torch.inner的输出
                out = torch.zeros_like(res)
                # 将torch.inner的结果存入out中
                torch.inner(a, b, out=out)
                # 断言torch.inner的输出res与out相等
                self.assertEqual(res, out)

        # 各种维度组合的测试用例
        check([], [])                       # 标量 x 标量
        check([], [0])                      # 标量 x 空张量
        check([], [3])                      # 标量 x 1维张量
        check([], [2, 3, 4])                # 标量 x 3维张量

        check([0], [0])                     # 空张量 x 空张量
        check([0], [2, 0])                  # 空张量 x 2维张量

        check([2], [2])                     # 1维张量 x 1维张量
        check([2], [3, 1, 2])               # 1维张量 x 3维张量
        check([2], [3, 0, 2])               # 1维张量 x 3维空张量

        check([1, 2], [3, 2])               # 2维张量 x 2维张量
        check([1, 2], [3, 4, 2])            # 2维张量 x 3维张量
        check([2, 1, 3, 2], [1, 3, 2, 2])   # 4维张量 x 4维张量

        # 测试错误消息
        with self.assertRaisesRegex(RuntimeError,
                                    r"inner\(\) the last dimension must match on both "
                                    r"input tensors but got shapes \[2, 3\] and \[2, 2\]"):
            # 断言在不匹配维度时调用torch.inner会引发RuntimeError异常
            torch.randn(2, 3, device=device, dtype=dtype).inner(torch.randn(2, 2, device=device, dtype=dtype))

    # 测试torch.outer及其别名torch.ger与NumPy的比较
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_outer(self, device, dtype):
        # 定义内部函数 run_test_case，用于运行测试用例 a 和 b 是输入张量
        def run_test_case(a, b):
            # 根据 dtype 类型进行不同的处理
            if dtype == torch.bfloat16:
                # 将 a 和 b 转换为双精度张量，并转换为 NumPy 数组
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                # 将 a 和 b 转换为 CPU 上的 NumPy 数组
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                exact_dtype = True
            # 计算期望的外积结果
            expected = np.outer(a_np, b_np)

            # 使用 torch.outer 方法进行外积计算并断言结果是否正确
            self.assertEqual(torch.outer(a, b), expected, exact_dtype=False)
            # 使用 torch.Tensor.outer 方法进行外积计算并断言结果是否正确
            self.assertEqual(torch.Tensor.outer(a, b), expected, exact_dtype=False)

            # 使用 torch.ger 方法进行外积计算并断言结果是否正确
            self.assertEqual(torch.ger(a, b), expected, exact_dtype=False)
            # 使用 torch.Tensor.ger 方法进行外积计算并断言结果是否正确
            self.assertEqual(torch.Tensor.ger(a, b), expected, exact_dtype=False)

            # 测试使用输出张量的变体
            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.outer(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

            out = torch.empty(a.size(0), b.size(0), device=device, dtype=dtype)
            torch.ger(a, b, out=out)
            self.assertEqual(out, expected, exact_dtype=False)

        # 生成随机张量 a 和 b 进行测试
        a = torch.randn(50).to(device=device, dtype=dtype)
        b = torch.randn(50).to(device=device, dtype=dtype)
        run_test_case(a, b)

        # 测试零步幅张量
        zero_strided = torch.randn(1).to(device=device, dtype=dtype).expand(50)
        run_test_case(zero_strided, b)
        run_test_case(a, zero_strided)

    def test_matrix_rank_removed_error(self, device):
        # 生成指定设备上的随机张量 a
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        # 断言调用 torch.matrix_rank 方法抛出指定异常信息
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.matrix_rank(a)

    def test_solve_removed_error(self, device):
        # 生成指定设备上的随机张量 a 和 b
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        b = make_tensor(5, 1, device=device, dtype=torch.float32)
        # 断言调用 torch.solve 方法抛出指定异常信息
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.solve(b, a)
        # 断言调用 b.solve(a) 方法抛出指定异常信息
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            b.solve(a)

    def test_eig_removed_error(self, device):
        # 生成指定设备上的随机张量 a
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        # 断言调用 torch.eig 方法抛出指定异常信息
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.eig(a)
        # 断言调用 a.eig() 方法抛出指定异常信息
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.eig()
    # 测试函数，验证 torch.symeig 函数在移除后是否能正确引发 RuntimeError 异常
    def test_symeig_removed_error(self, device):
        # 创建一个大小为 5x5 的张量，在指定设备上
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        # 使用 assertRaisesRegex 验证调用 torch.symeig(a) 是否引发特定异常
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.symeig(a)
        # 使用 assertRaisesRegex 验证调用 a.symeig() 是否引发特定异常
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.symeig()

    # 测试函数，验证 torch.lstsq 函数在移除后是否能正确引发 RuntimeError 异常
    def test_lstsq_removed_error(self, device):
        # 创建一个大小为 5x5 的张量，在指定设备上
        a = make_tensor(5, 5, device=device, dtype=torch.float32)
        # 使用 assertRaisesRegex 验证调用 torch.lstsq(a, a) 是否引发特定异常
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            torch.lstsq(a, a)
        # 使用 assertRaisesRegex 验证调用 a.lstsq(a) 是否引发特定异常
        with self.assertRaisesRegex(RuntimeError, "This function was deprecated since version 1.9 and is now removed"):
            a.lstsq(a)

    # 测试函数，验证 torch.linalg.lstsq 在批处理和广播情况下的正确性
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipIfTorchDynamo("flaky, needs investigation")
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_linalg_lstsq_batch_broadcasting(self, device, dtype):
        # 导入随机生成良好条件矩阵的函数
        from torch.testing._internal.common_utils import random_well_conditioned_matrix

        # 检查函数定义，验证给定的矩阵 a 和 b 的解是否正确
        def check_correctness(a, b):
            # 使用 torch.linalg.lstsq 求解线性最小二乘问题，获取解 solution
            sol = torch.linalg.lstsq(a, b).solution
            # 使用 a 的伪逆和矩阵 b 的乘积作为参考解
            sol2 = a.pinverse() @ b
            # 断言 sol 与 sol2 相等，允许的相对误差和绝对误差均为 1e-5
            self.assertEqual(sol, sol2, rtol=1e-5, atol=1e-5)

        # 定义不同尺寸的矩阵 m 和批处理情况
        ms = [2 ** i for i in range(5)]
        batches = [(), (0,), (2,), (2, 2), (2, 2, 2)]
        
        # 单个矩阵在 rhs 上进行批处理广播的情况
        for m, batch in itertools.product(ms, batches):
            # 创建一个随机良好条件的 m x m 矩阵，并将其视图转换为指定批次形状
            a = random_well_conditioned_matrix(m, m, dtype=dtype, device=device).view(*([1] * len(batch)), m, m)
            # 创建一个与 a 的形状兼容的随机张量 b
            b = torch.rand(*(batch + (m, m)), dtype=dtype, device=device)
            # 调用 check_correctness 验证解的正确性
            check_correctness(a, b)

        # 具有可广播形状的其他情况
        for m in ms:
            # 创建一个具有特定形状的随机良好条件矩阵 a
            a = random_well_conditioned_matrix(1, 3, 1, 3, m, m, dtype=dtype, device=device)
            # 创建一个具有特定形状的随机张量 b
            b = torch.rand(3, 1, 3, 1, m, m // 2, dtype=dtype, device=device)
            # 调用 check_correctness 验证解的正确性
            check_correctness(a, b)

            # 在这个测试中，rhs 是向量而不是矩阵，需要对 b 进行 unsqueeze
            b = torch.rand(3, 1, 3, 1, m, dtype=dtype, device=device)
            # 由于 check_correctness 检查 a.pinverse() @ b，需要将 b unsqueeze
            check_correctness(a, b.unsqueeze(-1))

            # 创建一个具有特定形状的随机良好条件矩阵 a
            a = random_well_conditioned_matrix(3, 1, 3, 1, m, m, dtype=dtype, device=device)
            # 创建一个具有特定形状的随机张量 b
            b = torch.rand(1, 3, 1, 3, m, m // 2, dtype=dtype, device=device)
            # 调用 check_correctness 验证解的正确性
            check_correctness(a, b)

            # 在这个测试中，rhs 是向量而不是矩阵，需要对 b 进行 unsqueeze
            b = torch.rand(1, 3, 1, 3, m, dtype=dtype, device=device)
            # 调用 check_correctness 验证解的正确性
            check_correctness(a, b.unsqueeze(-1))
    # 标记：如果没有 LAPACK 库，则跳过测试
    @skipCPUIfNoLapack
    # 标记：测试支持的浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 定义测试函数 test_cholesky，接受设备和数据类型作为参数
    def test_cholesky(self, device, dtype):
        # 导入随机生成 Hermitian 正定矩阵的工具函数
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # 定义内部函数 run_test，用于运行具体的测试
        def run_test(shape, batch, contiguous):
            # 生成随机 Hermitian 正定矩阵 A
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            # 如果 A 的元素数量大于 0 且不连续，则转置 A 并断言其不连续
            if A.numel() > 0 and not contiguous:
                A = A.mT
                self.assertFalse(A.is_contiguous())
            # 使用 NumPy 计算 A 的 Cholesky 分解作为期望结果
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            # 使用 PyTorch 计算 A 的 Cholesky 分解作为实际结果
            actual_L = torch.linalg.cholesky(A)

            # 对于单精度浮点数和复数类型，可能会出现单个元素之间的差异，因此比较矩阵的范数而非元素
            if A.numel() > 0 and dtype in [torch.float32, torch.complex64]:
                # 计算期望和实际结果的矩阵范数
                expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
                actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
                # 使用标准容差比较范数
                self.assertEqual(actual_norm, expected_norm)
                # 使用较高的容差比较单个元素的值
                self.assertEqual(actual_L, expected_L, atol=1e-2, rtol=1e-5)
            else:
                # 比较实际和期望的 Cholesky 分解结果
                self.assertEqual(actual_L, expected_L)

        # 定义不同的形状、批次和连续性情况的测试参数
        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        larger_input_case = [(100, (5, ), True)]
        # 遍历所有形状、批次和连续性的组合进行测试
        for shape, batch, contiguous in list(itertools.product(shapes, batches, (True, False))) + larger_input_case:
            run_test(shape, batch, contiguous)

        # 检查 out= 变体的情况
        A = random_hermitian_pd_matrix(3, 3, dtype=dtype, device=device)
        out = torch.empty_like(A)
        ans = torch.linalg.cholesky(A, out=out)
        self.assertEqual(ans, out)
        expected = torch.linalg.cholesky(A)
        self.assertEqual(expected, out)

        # 检查 upper= 变体的情况
        # 计算期望的上三角矩阵
        expected = torch.linalg.cholesky(A).mH
        # 计算实际的上三角矩阵
        actual = torch.linalg.cholesky(A, upper=True)
        self.assertEqual(expected, actual)

    # 标记：如果没有 MAGMA 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 标记：如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 标记：测试支持的双精度浮点数类型
    @dtypes(*floating_and_complex_types())
    # 注释：旧的 Cholesky 分解测试从 test_torch.py 和 test_autograd.py 转移到这里
    # 标记：标记为慢速测试
    @slowTest
    # 标记：如果没有 MAGMA 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 标记：如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 标记：测试支持的双精度浮点数类型
    @dtypes(torch.double)
    # 定义一个测试方法，用于测试批量处理的旧版 Cholesky 分解
    def test_old_cholesky_batched_many_batches(self, device, dtype):
        # 导入必要的随机对称正定矩阵生成函数
        from torch.testing._internal.common_utils import random_symmetric_pd_matrix

        # 定义 Cholesky 测试辅助函数
        def cholesky_test_helper(n, batchsize, device, upper):
            # 生成随机对称正定矩阵 A
            A = random_symmetric_pd_matrix(n, batchsize, dtype=dtype, device=device)
            # 进行 Cholesky 分解
            chol_fact = torch.cholesky(A, upper=upper)
            if upper:
                # 如果是上三角分解，则进行正确性检查
                self.assertEqual(A, chol_fact.mT.matmul(chol_fact))
                # 上三角矩阵检查
                self.assertEqual(chol_fact, chol_fact.triu())
            else:
                # 如果是下三角分解，则进行正确性检查
                self.assertEqual(A, chol_fact.matmul(chol_fact.mT))
                # 下三角矩阵检查
                self.assertEqual(chol_fact, chol_fact.tril())

        # 遍历参数组合进行测试
        for upper, batchsize in itertools.product([True, False], [262144, 524288]):
            cholesky_test_helper(2, batchsize, device, upper)

    # 设置精度覆盖装饰器，指定 torch.float32 和 torch.complex64 精度为 1e-4
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    # 如果没有 magma 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 设置数据类型装饰器，用于指定测试数据类型为浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 定义旧版 Cholesky 分解的批量测试方法
    def test_old_cholesky_batched(self, device, dtype):
        # 导入必要的随机 Hermite 正定矩阵生成函数
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # 定义 Cholesky 测试辅助函数
        def cholesky_test_helper(n, batch_dims, upper):
            # 生成随机 Hermite 正定矩阵 A
            A = random_hermitian_pd_matrix(n, *batch_dims, dtype=dtype, device=device)
            # 对 A 进行 Cholesky 分解
            cholesky_exp = torch.stack([m.cholesky(upper=upper) for m in A.reshape(-1, n, n)])
            cholesky_exp = cholesky_exp.reshape_as(A)
            # 检查 Cholesky 分解的正确性
            self.assertEqual(cholesky_exp, torch.cholesky(A, upper=upper))

        # 遍历参数组合进行测试
        for upper, batchsize in itertools.product([True, False], [(3,), (3, 4), (2, 3, 4)]):
            cholesky_test_helper(3, batchsize, upper)

    # 设置精度覆盖装饰器，指定 torch.float32 和 torch.complex64 精度为 1e-4
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    # 如果没有 magma 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 设置数据类型装饰器，用于指定测试数据类型为浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 设置 TF32 精度开关，概率为 0.01
    @tf32_on_and_off(0.01)
    # 设置 BF32 精度开关，概率为 0.01
    @bf32_on_and_off(0.01)
    # 定义旧版 Cholesky 分解的基本测试方法
    def test_old_cholesky(self, device, dtype):
        # 导入必要的随机 Hermite 正定矩阵生成函数
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # 生成随机 Hermite 正定矩阵 A
        A = random_hermitian_pd_matrix(10, dtype=dtype, device=device)

        # 默认情况测试
        # 对 A 进行 Cholesky 分解
        C = torch.cholesky(A)
        # 重建原始矩阵 B
        B = torch.mm(C, C.t().conj())
        # 检查分解后的结果是否与原始矩阵 A 相等
        self.assertEqual(A, B, atol=1e-14, rtol=0)

        # 测试上三角分解
        # 对 A 进行 Cholesky 分解，指定为上三角分解
        U = torch.cholesky(A, True)
        # 重建原始矩阵 B
        B = torch.mm(U.t().conj(), U)
        # 检查分解后的结果是否与原始矩阵 A 相等
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (upper) did not allow rebuilding the original matrix')

        # 测试下三角分解
        # 对 A 进行 Cholesky 分解，指定为下三角分解
        L = torch.cholesky(A, False)
        # 重建原始矩阵 B
        B = torch.mm(L, L.t().conj())
        # 检查分解后的结果是否与原始矩阵 A 相等
        self.assertEqual(A, B, atol=1e-14, rtol=0, msg='cholesky (lower) did not allow rebuilding the original matrix')

    # 如果没有 magma 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 设置数据类型装饰器，用于指定测试数据类型为浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 定义一个测试方法，用于测试空矩阵的 Cholesky 分解
    def test_old_cholesky_empty(self, device, dtype):
        # 定义内部函数 run_test，参数为 upper
        def run_test(upper):
            # 创建一个空的 dtype 类型的 tensor A，位于指定的 device 上
            A = torch.empty(0, 0, dtype=dtype, device=device)
            # 对 A 进行 Cholesky 分解，返回上三角或下三角矩阵 chol
            chol = torch.cholesky(A, upper)
            # 计算 chol 与其共轭转置的乘积，得到重构的矩阵 chol_A
            chol_A = torch.matmul(chol, chol.t().conj())
            # 断言 A 与 chol_A 相等
            self.assertEqual(A, chol_A)
        
        # 对 upper 分别为 True 和 False 运行 run_test 函数
        for upper in [True, False]:
            run_test(upper)

    # 对 GitHub 上的问题进行测试
    # https://github.com/pytorch/pytorch/issues/57032
    # 测试 torch.cholesky 在批处理的 CUDA 输入中 upper=True 时的错误
    # 它使用了下三角部分而不是上三角部分
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    def test_old_cholesky_batched_upper(self, device, dtype):
        # 导入所需的函数 random_hermitian_pd_matrix
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # 设定 batchsize 为 2
        batchsize = 2
        # 生成一个 dtype 类型的随机埃尔米特正定矩阵 A，形状为 (3, batchsize)，位于指定的 device 上
        A = random_hermitian_pd_matrix(3, batchsize, dtype=dtype, device=device)
        # 取 A 的上三角部分，下三角部分填充为零，得到 A_triu
        A_triu = A.triu()

        # 对 A_triu 进行 Cholesky 分解，要求返回上三角矩阵 U
        U = torch.cholesky(A_triu, upper=True)

        # 重构原始矩阵 A，使用 U 的共轭转置与 U 的乘积
        reconstruct_A = U.mH @ U
        # 断言重构后的 A 与原始 A 相等
        self.assertEqual(A, reconstruct_A)

    # 跳过不支持 Magma 和 Cusolver 的 CUDA 平台
    # 如果没有 LAPACK 支持则跳过 CPU 平台
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_cholesky_ex(self, device, dtype):
        # 导入所需的函数 random_hermitian_pd_matrix
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # 定义内部函数 run_test，参数为 n 和 batch
        def run_test(n, batch):
            # 生成一个大小为 n x n 的随机埃尔米特正定矩阵 A，批处理维度为 batch
            A = random_hermitian_pd_matrix(n, *batch, dtype=dtype, device=device)
            # 使用 NumPy 计算 A 的 Cholesky 分解，得到期望的下三角矩阵 expected_L
            expected_L = np.linalg.cholesky(A.cpu().numpy())
            # 创建一个与 A 形状相同的零张量 expected_info，数据类型为 torch.int32，位于指定的 device 上
            expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
            # 使用 PyTorch 的 torch.linalg.cholesky_ex 函数计算 A 的 Cholesky 分解和信息
            actual_L, actual_info = torch.linalg.cholesky_ex(A)

            # 对于单个浮点数和复数的情况，PyTorch 和 NumPy 的矩阵可能会有微小差异
            # 因此比较矩阵的范数而不是逐个元素进行比较
            if A.numel() > 0 and dtype in [torch.float32, torch.complex64]:
                # 计算期望的矩阵 expected_L 的范数，按批处理维度计算矩阵范数
                expected_norm = np.linalg.norm(expected_L, ord=1, axis=(-2, -1))
                # 计算实际的矩阵 actual_L 的范数，按批处理维度计算矩阵范数
                actual_norm = torch.linalg.norm(actual_L, ord=1, axis=(-2, -1))
                # 使用标准的容差比较范数
                self.assertEqual(actual_norm, expected_norm)
                # 使用较高的容差比较单个值
                self.assertEqual(actual_L, expected_L, atol=1e-2, rtol=1e-5)
            else:
                # 比较实际的矩阵 actual_L 与期望的矩阵 expected_L
                self.assertEqual(actual_L, expected_L)
            # 比较实际的信息 actual_info 与期望的信息 expected_info
            self.assertEqual(actual_info, expected_info)

        # 定义测试用例的参数组合
        ns = (0, 3, 5)
        batches = ((), (2,), (2, 1))
        # 对参数组合进行遍历，并运行 run_test 函数
        for n, batch in itertools.product(ns, batches):
            run_test(n, batch)

    # 跳过不支持 Magma 和 Cusolver 的 CUDA 平台
    # 如果没有 LAPACK 支持则跳过 CPU 平台
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，测试 cholesky_ex 方法在处理非正定矩阵时的行为
    def test_cholesky_ex_non_pd(self, device, dtype):
        # 创建一个单位矩阵 A，数据类型为 dtype，在指定设备上
        A = torch.eye(3, 3, dtype=dtype, device=device)
        # 修改 A 的最后一个元素为 0，使得 A 成为奇异矩阵（非正定）
        A[-1, -1] = 0  # Now A is singular
        # 调用 torch.linalg.cholesky_ex 计算 A 的 Cholesky 分解，返回结果和 info
        _, info = torch.linalg.cholesky_ex(A)
        # 断言 info 的值为 3，表示 A 非正定
        self.assertEqual(info, 3)
        # 使用断言确保调用 torch.linalg.cholesky_ex(A, check_errors=True) 会抛出 LinAlgError
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'minor of order 3 is not positive-definite'):
            torch.linalg.cholesky_ex(A, check_errors=True)

        # 将 A 转换成形状为 (1, 3, 3) 的批量矩阵
        A = torch.eye(3, 3, dtype=dtype, device=device)
        A = A.reshape((1, 3, 3))
        A = A.repeat(5, 1, 1)
        # 修改 A[3] 的倒数第二行和倒数第二列的元素为 0，使得 A[3] 成为奇异矩阵
        A[3, -2, -2] = 0  # Now A[3] is singular
        # 再次调用 torch.linalg.cholesky_ex 计算批量矩阵 A 的 Cholesky 分解，返回结果和 batched info
        _, info = torch.linalg.cholesky_ex(A)

        # 创建期望的 info，形状与 A 的批量形状相同，所有元素初始化为 0
        expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
        # 设置第 3 个元素的 info 为 2，表示 A[3] 非正定
        expected_info[3] = 2
        # 断言计算得到的 info 与期望的 expected_info 相等
        self.assertEqual(info, expected_info)
        # 使用断言确保调用 torch.linalg.cholesky_ex(A, check_errors=True) 会抛出 LinAlgError
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 3\): The factorization could not be completed'):
            torch.linalg.cholesky_ex(A, check_errors=True)
    # 定义一个测试方法，用于比较 `torch.addr` 函数和 NumPy 的效果
    def _test_addr_vs_numpy(self, device, dtype, beta=1, alpha=1):
        
        # 定义内部函数 check，用于执行具体的测试
        def check(m, a, b, beta, alpha):
            # 如果数据类型是 torch.bfloat16，则将张量转换为双精度并转换为 NumPy 数组
            if dtype == torch.bfloat16:
                a_np = a.to(torch.double).cpu().numpy()
                b_np = b.to(torch.double).cpu().numpy()
                m_np = m.to(torch.double).cpu().numpy()
                exact_dtype = False
            else:
                # 否则直接将张量转换为 NumPy 数组
                a_np = a.cpu().numpy()
                b_np = b.cpu().numpy()
                m_np = m.cpu().numpy()
                exact_dtype = True
            
            # 根据 beta 的值计算期望的输出
            if beta == 0:
                expected = alpha * np.outer(a_np, b_np)
            else:
                expected = beta * m_np + alpha * np.outer(a_np, b_np)
            
            # 使用 torch.addr 计算结果
            res = torch.addr(m, a, b, beta=beta, alpha=alpha)
            
            # 断言计算结果与期望结果相等
            self.assertEqual(res, expected, exact_dtype=exact_dtype)
            
            # 测试输出张量的变体
            out = torch.empty_like(res)
            torch.addr(m, a, b, beta=beta, alpha=alpha, out=out)
            self.assertEqual(out, expected, exact_dtype=exact_dtype)

        # 创建测试所需的张量 m, a, b
        m = make_tensor((50, 50), device=device, dtype=dtype, low=-2, high=2)
        a = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)
        b = make_tensor((50,), device=device, dtype=dtype, low=-2, high=2)

        # 执行 check 函数，对 m, a, b 进行测试
        check(m, a, b, beta, alpha)

        # 测试转置后的 m
        m_transpose = torch.transpose(m, 0, 1)
        check(m_transpose, a, b, beta, alpha)

        # 测试零步幅张量
        zero_strided = make_tensor((1,), device=device, dtype=dtype, low=-2, high=2).expand(50)
        check(m, zero_strided, b, beta, alpha)

        # 测试标量张量 m
        m_scalar = torch.tensor(1, device=device, dtype=dtype)
        check(m_scalar, a, b, beta, alpha)

        # 当 beta == 0 时，测试 NaN 和 Inf 是否不会传播到输出中
        float_and_complex_dtypes = floating_and_complex_types_and(torch.half, torch.bfloat16)
        if beta == 0 and dtype in float_and_complex_dtypes:
            m[0][10] = m[10][10] = m[20][20] = float('inf')
            m[1][10] = m[11][10] = m[21][20] = float('nan')
        check(m, a, b, 0, alpha)

    # 使用 torch.bool 类型进行测试的装饰器方法
    @dtypes(torch.bool)
    def test_addr_bool(self, device, dtype):
        # 分别使用不同的 beta 和 alpha 值调用 _test_addr_vs_numpy 方法进行测试
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=True)
        self._test_addr_vs_numpy(device, dtype, beta=False, alpha=False)
        self._test_addr_vs_numpy(device, dtype, beta=True, alpha=True)

    # 使用所有整数类型进行测试的装饰器方法
    # 测试函数：测试地址向量化操作的整数参数
    def test_addr_integral(self, device, dtype):
        # 断言：当 beta 参数为浮点数时，抛出 RuntimeError 异常，错误信息为 'argument beta must not be a floating point number.'
        with self.assertRaisesRegex(RuntimeError,
                                    'argument beta must not be a floating point number.'):
            self._test_addr_vs_numpy(device, dtype, beta=2., alpha=1)
        # 断言：当 alpha 参数为浮点数时，抛出 RuntimeError 异常，错误信息为 'argument alpha must not be a floating point number.'
        with self.assertRaisesRegex(RuntimeError,
                                    'argument alpha must not be a floating point number.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=1.)
        # 断言：当 beta 参数为布尔值时，抛出 RuntimeError 异常，错误信息为 'Boolean beta only supported for Boolean results.'
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean beta only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        # 断言：当 alpha 参数为布尔值时，抛出 RuntimeError 异常，错误信息为 'Boolean alpha only supported for Boolean results.'
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean alpha only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # 当 beta 参数为零时，调用测试函数 self._test_addr_vs_numpy 进行测试，alpha 参数为 2
        self._test_addr_vs_numpy(device, dtype, beta=0, alpha=2)
        # 当 beta 参数非零时，调用测试函数 self._test_addr_vs_numpy 进行测试，beta 和 alpha 参数都为 2
        self._test_addr_vs_numpy(device, dtype, beta=2, alpha=2)

    # 链接到 GitHub 问题编号 127043 的注释
    @xfailIfTorchDynamo
    @precisionOverride({torch.bfloat16: 1e-1})
    @dtypes(*floating_and_complex_types_and(torch.half, torch.bfloat16))
    # 测试函数：测试地址向量化操作的浮点数和复数参数
    def test_addr_float_and_complex(self, device, dtype):
        # 断言：当 beta 参数为布尔值时，抛出 RuntimeError 异常，错误信息为 'Boolean beta only supported for Boolean results.'
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean beta only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=True, alpha=1)
        # 断言：当 alpha 参数为布尔值时，抛出 RuntimeError 异常，错误信息为 'Boolean alpha only supported for Boolean results.'
        with self.assertRaisesRegex(RuntimeError,
                                    'Boolean alpha only supported for Boolean results.'):
            self._test_addr_vs_numpy(device, dtype, beta=2, alpha=True)

        # 当 beta 参数为零时，调用测试函数 self._test_addr_vs_numpy 进行测试，alpha 参数为 2.0
        self._test_addr_vs_numpy(device, dtype, beta=0., alpha=2)
        # 当 beta 参数非零时，调用测试函数 self._test_addr_vs_numpy 进行测试，beta 参数为 0.5，alpha 参数为 2
        self._test_addr_vs_numpy(device, dtype, beta=0.5, alpha=2)
        # 如果 dtype 属于复数类型，调用测试函数 self._test_addr_vs_numpy 进行测试，beta 和 alpha 参数分别为复数
        if dtype in complex_types():
            self._test_addr_vs_numpy(device, dtype, beta=(0 + 0.1j), alpha=(0.2 - 0.2j))

    # 测试函数：测试外积操作的类型提升
    @dtypes(*itertools.product(all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
                               all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool)))
    def test_outer_type_promotion(self, device, dtypes):
        # 生成设备和指定数据类型的随机张量 a 和 b
        a = torch.randn(5).to(device=device, dtype=dtypes[0])
        b = torch.randn(5).to(device=device, dtype=dtypes[1])
        # 对于 torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger 中的每一个操作 op
        for op in (torch.outer, torch.Tensor.outer, torch.ger, torch.Tensor.ger):
            # 执行外积操作 op(a, b)，并断言其结果的数据类型等于张量 a 和 b 的结果类型
            result = op(a, b)
            self.assertEqual(result.dtype, torch.result_type(a, b))

    # 注释：不使用 @dtypes 装饰器，以避免在每个设备上生成大约 1700 个测试
    # 测试地址类型提升功能，使用给定的设备执行测试
    def test_addr_type_promotion(self, device):
        # 遍历所有数据类型组合，包括 torch.half, torch.bfloat16, torch.bool 和所有其他类型
        for dtypes0, dtypes1, dtypes2 in product(all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), repeat=3):
            # 创建大小为 (5,) 的张量 a，使用指定的设备和数据类型，数值范围在 [-2, 2] 之间
            a = make_tensor((5,), device=device, dtype=dtypes0, low=-2, high=2)
            # 创建大小为 (5,) 的张量 b，使用指定的设备和数据类型，数值范围在 [-2, 2] 之间
            b = make_tensor((5,), device=device, dtype=dtypes1, low=-2, high=2)
            # 创建大小为 (5, 5) 的张量 m，使用指定的设备和数据类型，数值范围在 [-2, 2] 之间
            m = make_tensor((5, 5), device=device, dtype=dtypes2, low=-2, high=2)

            # 计算所需的数据类型，将 dtypes0 和 dtypes1 提升到一个更高级别，然后再与 dtypes2 提升
            desired_dtype = torch.promote_types(torch.promote_types(dtypes0, dtypes1),
                                                dtypes2)
            # 对于 torch.addr 和 torch.Tensor.addr 中的每个操作符 op
            for op in (torch.addr, torch.Tensor.addr):
                # 执行地址操作，计算结果
                result = op(m, a, b)
                # 断言结果张量的数据类型与期望的数据类型一致
                self.assertEqual(result.dtype, desired_dtype)

    # 从 test_torch.py 迁移的测试用例
    # 1) 测试当输入张量为空时，结果张量的形状
    # 2) 测试当输入张量为标量时，是否会抛出运行时异常
    def test_outer_ger_addr_legacy_tests(self, device):
        # 遍历不同大小的输入对
        for size in ((0, 0), (0, 5), (5, 0)):
            # 创建大小为 size[0] 的随机张量 a，使用指定的设备
            a = torch.rand(size[0], device=device)
            # 创建大小为 size[1] 的随机张量 b，使用指定的设备
            b = torch.rand(size[1], device=device)

            # 断言 torch.outer(a, b) 的形状与 size 相同
            self.assertEqual(torch.outer(a, b).shape, size)
            # 断言 torch.ger(a, b) 的形状与 size 相同
            self.assertEqual(torch.ger(a, b).shape, size)

            # 创建空的大小为 size 的张量 m，使用指定的设备
            m = torch.empty(size, device=device)
            # 断言 torch.addr(m, a, b) 的形状与 size 相同
            self.assertEqual(torch.addr(m, a, b).shape, size)

        # 创建大小为 (5, 6) 的随机张量 m，使用指定的设备
        m = torch.randn(5, 6, device=device)
        # 创建大小为 5 的随机张量 a，使用指定的设备
        a = torch.randn(5, device=device)
        # 创建标量张量 b，使用指定的设备
        b = torch.tensor(6, device=device)
        # 断言运行时会抛出异常，因为 torch.outer(a, b) 期望 a 和 b 都是向量而不是标量
        self.assertRaises(RuntimeError, lambda: torch.outer(a, b))
        self.assertRaises(RuntimeError, lambda: torch.outer(b, a))
        self.assertRaises(RuntimeError, lambda: torch.ger(a, b))
        self.assertRaises(RuntimeError, lambda: torch.ger(b, a))
        # 断言运行时会抛出异常，因为 torch.addr(m, a, b) 期望 a 和 b 都是向量而不是标量
        self.assertRaises(RuntimeError, lambda: torch.addr(m, a, b))
        self.assertRaises(RuntimeError, lambda: torch.addr(m, b, a))

    # 测试 torch.det 及其别名 torch.linalg.det 与 NumPy 的比较
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.double, torch.cdouble)
    def test_det(self, device, dtype):
        # 定义不同形状的张量列表
        tensors = (
            torch.randn((2, 2), device=device, dtype=dtype),
            torch.randn((129, 129), device=device, dtype=dtype),
            torch.randn((3, 52, 52), device=device, dtype=dtype),
            torch.randn((4, 2, 26, 26), device=device))

        # 定义要测试的操作列表，包括 torch.det, torch.Tensor.det, torch.linalg.det
        ops = (torch.det, torch.Tensor.det,
               torch.linalg.det)
        # 遍历所有张量
        for t in tensors:
            # 使用 NumPy 计算预期的行列式值
            expected = np.linalg.det(t.cpu().numpy())
            # 对于每个操作 op，计算实际的行列式值，并断言与预期值一致
            for op in ops:
                actual = op(t)
                self.assertEqual(actual, expected)
                # 使用自定义方法 compare_with_numpy 比较 op 的结果与 NumPy 计算的结果
                self.compare_with_numpy(op, np.linalg.det, t)

        # 注意：torch.det 要求张量的维度至少为 2
        t = torch.randn(1, device=device, dtype=dtype)
        # 断言运行时会抛出异常，因为 t 的维度不符合要求
        with self.assertRaises(RuntimeError):
            op(t)

    # 跳过没有 Magma 支持的 CUDA 测试和没有 Lapack 支持的 CPU 测试
    # 使用浮点数和复数类型进行测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    # 定义一个测试方法，用于测试 torch.linalg.eigh 函数在不同设备和数据类型下的行为
    def test_eigh(self, device, dtype):
        # 从 torch.testing._internal.common_utils 导入 random_hermitian_matrix 函数
        from torch.testing._internal.common_utils import random_hermitian_matrix

        # 定义一个内部函数 run_test，用于执行测试用例
        def run_test(shape, batch, uplo):
            # 生成一个随机的 Hermite 矩阵作为测试数据
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            # 使用 numpy 中的 np.linalg.eigh 函数计算期望的特征值和特征向量
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            # 使用 torch.linalg.eigh 函数计算实际的特征值和特征向量
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            # 断言实际的特征值与期望的特征值相等
            self.assertEqual(actual_w, expected_w)
            # 由于特征向量的符号不唯一，因此比较它们的绝对值
            self.assertEqual(abs(actual_v), abs(expected_v))
            # 此外，我们可以将特征向量乘以一个相位因子 e^{i\phi} 后再比较其值
            # 选择 torch 和 numpy 中特征向量的第一个元素相同作为约定
            # 对于实数输入，这个相位因子是加减一
            if matrix.numel() > 0:
                # 计算相位因子
                phase = torch.from_numpy(expected_v[..., 0, :]).to(device=device).div(actual_v[..., 0, :])
                # 对实际的特征向量应用相位因子进行旋转
                actual_v_rotated = actual_v * phase.unsqueeze(-2).expand_as(actual_v)
                # 断言旋转后的实际特征向量与期望的特征向量相等
                self.assertEqual(actual_v_rotated, expected_v)

            # 检查使用 out= 参数的情况
            out_w = torch.empty_like(actual_w)
            out_v = torch.empty_like(actual_v)
            ans_w, ans_v = torch.linalg.eigh(matrix, UPLO=uplo, out=(out_w, out_v))
            # 断言 out= 参数返回的特征值与预期相等
            self.assertEqual(ans_w, out_w)
            # 断言 out= 参数返回的特征向量与预期相等
            self.assertEqual(ans_v, out_v)
            # 断言 out= 参数返回的特征值与直接调用时的特征值相等
            self.assertEqual(ans_w, actual_w)
            # 断言 out= 参数返回的特征向量的绝对值与直接调用时的特征向量的绝对值相等
            self.assertEqual(abs(ans_v), abs(actual_v))

        # 定义测试用例中的形状、批次和上三角/下三角矩阵类型
        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        uplos = ["U", "L"]
        # 使用 itertools.product 对所有形状、批次和上三角/下三角矩阵类型进行排列组合
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    # 如果设备上没有支持的 MAGMA，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果设备上没有支持的 LAPACK，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 使用浮点数和复数类型进行测试
    @dtypes(*floating_and_complex_types())
    # 为特定的数据类型设置精度覆盖
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    # 定义测试 torch.linalg.eigh 函数的下三角矩阵测试用例
    def test_eigh_lower_uplo(self, device, dtype):
        # 定义一个内部函数 run_test，用于执行测试用例
        def run_test(shape, batch, uplo):
            # 检查小写的 uplo 参数
            # 使用非对称输入检查 uplo 参数是否按预期工作
            matrix = torch.randn(shape, shape, *batch, dtype=dtype, device=device)
            # 使用 numpy 中的 np.linalg.eigh 函数计算期望的特征值和特征向量
            expected_w, expected_v = np.linalg.eigh(matrix.cpu().numpy(), UPLO=uplo)
            # 使用 torch.linalg.eigh 函数计算实际的特征值和特征向量
            actual_w, actual_v = torch.linalg.eigh(matrix, UPLO=uplo)
            # 断言实际的特征值与期望的特征值相等
            self.assertEqual(actual_w, expected_w)
            # 断言实际的特征向量的绝对值与期望的特征向量的绝对值相等
            self.assertEqual(abs(actual_v), abs(expected_v))

        # 定义测试用例中的上三角/下三角矩阵类型
        uplos = ["u", "l"]
        # 对所有上三角/下三角矩阵类型进行测试
        for uplo in uplos:
            run_test(3, (2, 2), uplo)

    # 如果设备上没有支持的 MAGMA，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果设备上没有支持的 LAPACK，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 使用浮点数和复数类型进行测试
    @dtypes(*floating_and_complex_types())
    # 定义测试方法来检测 torch.linalg.eigh 函数的错误和警告
    def test_eigh_errors_and_warnings(self, device, dtype):
        # 导入生成随机埃尔米特矩阵的方法
        from torch.testing._internal.common_utils import random_hermitian_matrix

        # eigh 函数要求输入是方阵
        t = torch.randn(2, 3, device=device, dtype=dtype)
        # 使用 assertRaisesRegex 断言捕获 RuntimeError，验证错误信息包含特定文本
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigh(t)

        # eigh 函数要求 'uplo' 参数为 'U' 或 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        # 遍历不合法的 'uplo' 参数值
        for uplo in ["a", "wrong"]:
            # 使用 assertRaisesRegex 断言捕获 RuntimeError，验证错误信息包含特定文本
            with self.assertRaisesRegex(RuntimeError, "be 'L' or 'U'"):
                torch.linalg.eigh(t, UPLO=uplo)
            # 使用 assertRaisesRegex 断言捕获 ValueError，验证错误信息包含特定文本
            with self.assertRaisesRegex(ValueError, "be 'L' or 'U'"):
                np.linalg.eigh(t.cpu().numpy(), UPLO=uplo)

        # 若传入非空且形状错误的输出张量，将触发警告
        a = random_hermitian_matrix(3, dtype=dtype, device=device)
        # 确定实部数据类型，若输入数据类型为复数，则输出数据类型为其实部类型
        real_dtype = a.real.dtype if dtype.is_complex else dtype
        # 创建形状不匹配的输出张量
        out_w = torch.empty(7, 7, dtype=real_dtype, device=device)
        out_v = torch.empty(7, 7, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.linalg.eigh(a, out=(out_w, out_v))
            # 验证是否触发了两次警告
            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[-2].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 输出张量的数据类型应安全转换
        out_w = torch.empty(0, dtype=real_dtype, device=device)
        out_v = torch.empty(0, dtype=torch.int, device=device)
        # 使用 assertRaisesRegex 断言捕获 RuntimeError，验证错误信息包含特定文本
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        out_w = torch.empty(0, dtype=torch.int, device=device)
        out_v = torch.empty(0, dtype=dtype, device=device)
        # 使用 assertRaisesRegex 断言捕获 RuntimeError，验证错误信息包含特定文本
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigh(a, out=(out_w, out_v))

        # 输出张量的设备应一致
        if torch.cuda.is_available():
            # 根据当前设备类型选择错误的设备
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            # 创建设备不匹配的输出张量
            out_w = torch.empty(0, device=wrong_device, dtype=dtype)
            out_v = torch.empty(0, device=device, dtype=dtype)
            # 使用 assertRaisesRegex 断言捕获 RuntimeError，验证错误信息包含特定文本
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))
            out_w = torch.empty(0, device=device, dtype=dtype)
            out_v = torch.empty(0, device=wrong_device, dtype=dtype)
            # 使用 assertRaisesRegex 断言捕获 RuntimeError，验证错误信息包含特定文本
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigh(a, out=(out_w, out_v))
    def test_eigh_svd_illcondition_matrix_input_should_not_crash(self, device, dtype):
        # 此测试函数用于验证处理条件糟糕的矩阵输入时不会崩溃的情况
        # 参考链接 https://github.com/pytorch/pytorch/issues/94772, https://github.com/pytorch/pytorch/issues/105359
        # 在 CUDA 11.8 上会因 `cusolver error: CUSOLVER_STATUS_EXECUTION_FAILED` 而崩溃，
        # 但在 CUDA 12.1 更新 1 或更高版本上可以通过测试。

        # 创建一个大小为 512x512 的张量，元素类型为 dtype，设备为 device，所有元素初始化为 1
        a = torch.ones(512, 512, dtype=dtype, device=device)
        # 设置第一个元素为 1.0e-5
        a[0, 0] = 1.0e-5
        # 设置最后一个元素为 1.0e5
        a[-1, -1] = 1.0e5

        # 使用 torch.linalg.eigh 计算特征值和特征向量
        eigh_out = torch.linalg.eigh(a)
        # 使用 torch.linalg.svd 计算奇异值分解
        svd_out = torch.linalg.svd(a)

        # 矩阵输入 a 过于条件糟糕
        # 我们只比较前两个奇异值/特征值。它们分别是 1.0e5 和 511.0
        # 使用容差为 1.0 的精度覆盖是有意义的，因为条件糟糕的输入很难收敛到精确值。
        self.assertEqual(eigh_out.eigenvalues.sort(descending=True).values[:2], [1.0e5, 511.0], atol=1.0, rtol=1.0e-2)
        self.assertEqual(svd_out.S[:2], [1.0e5, 511.0], atol=1.0, rtol=1.0e-2)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-4, torch.complex64: 1e-4})
    def test_eigvalsh(self, device, dtype):
        # 导入随机生成 Hermitian 矩阵的函数
        from torch.testing._internal.common_utils import random_hermitian_matrix

        def run_test(shape, batch, uplo):
            # 生成随机的 Hermitian 矩阵
            matrix = random_hermitian_matrix(shape, *batch, dtype=dtype, device=device)
            # 使用 numpy.linalg.eigvalsh 计算 Hermitian 矩阵的特征值
            expected_w = np.linalg.eigvalsh(matrix.cpu().numpy(), UPLO=uplo)
            # 使用 torch.linalg.eigvalsh 计算 Hermitian 矩阵的特征值
            actual_w = torch.linalg.eigvalsh(matrix, UPLO=uplo)
            # 断言计算得到的特征值与预期值一致
            self.assertEqual(actual_w, expected_w)

            # 检查带有 out 参数的情况
            out = torch.empty_like(actual_w)
            ans = torch.linalg.eigvalsh(matrix, UPLO=uplo, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, actual_w)

        # 定义不同的矩阵形状、批处理形状和 uplo 参数的组合
        shapes = (0, 3, 5)
        batches = ((), (3, ), (2, 2))
        uplos = ["U", "L"]
        for shape, batch, uplo in itertools.product(shapes, batches, uplos):
            run_test(shape, batch, uplo)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，用于测试 eigvalsh 函数的错误和警告
    def test_eigvalsh_errors_and_warnings(self, device, dtype):
        # eigvalsh 需要输入一个方阵
        t = torch.randn(2, 3, device=device, dtype=dtype)
        # 使用 assertRaisesRegex 确保运行时错误包含特定信息，表明必须是批量的方阵
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvalsh(t)

        # eigvalsh 需要 'uplo' 参数为 'U' 或 'L'
        t = torch.randn(3, 3, device=device, dtype=dtype)
        for uplo in ["a", "wrong"]:
            # 使用 assertRaisesRegex 确保运行时错误包含特定信息，表明 uplo 参数必须为 'L' 或 'U'
            with self.assertRaisesRegex(RuntimeError, "be 'L' or 'U'"):
                torch.linalg.eigvalsh(t, UPLO=uplo)
            # 使用 assertRaisesRegex 确保值错误包含特定信息，表明 uplo 参数必须为 'L' 或 'U'
            with self.assertRaisesRegex(ValueError, "be 'L' or 'U'"):
                np.linalg.eigvalsh(t.cpu().numpy(), UPLO=uplo)

        # 如果传入非空的输出张量并且形状错误，则会给出警告
        real_dtype = t.real.dtype if dtype.is_complex else dtype
        out = torch.empty_like(t).to(real_dtype)
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.linalg.eigvalsh(t, out=out)
            # 检查是否出现警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 数据类型应当安全转换
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.eigvalsh(t, out=out)

        # 设备应当匹配
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigvalsh(t, out=out)

    # 使用装饰器指定的数据类型，定义了一个测试 kron 函数
    @dtypes(*floating_and_complex_types())
    def test_kron(self, device, dtype):

        # 定义一个运行测试案例的内部函数，接受两个张量形状作为参数
        def run_test_case(a_shape, b_shape):
            # 创建两个随机张量 a 和 b，指定数据类型和设备
            a = torch.rand(a_shape, dtype=dtype, device=device)
            b = torch.rand(b_shape, dtype=dtype, device=device)

            # 使用 numpy 的 kron 函数计算预期结果
            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            # 使用 torch 的 kron 函数计算结果
            result = torch.kron(a, b)
            # 断言结果相等
            self.assertEqual(result, expected)

            # 检查带有 out 参数的变体
            out = torch.empty_like(result)
            ans = torch.kron(a, b, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        # 定义不同形状的张量对，使用 itertools.product 生成它们的组合
        shapes = [(4,), (2, 2), (1, 2, 3), (1, 2, 3, 3)]
        for a_shape, b_shape in itertools.product(shapes, reversed(shapes)):
            run_test_case(a_shape, b_shape)

    # 使用装饰器指定的数据类型，这里是一个测试函数的结束
    @dtypes(*floating_and_complex_types())
    # 测试空张量情况下的 torch.kron 函数
    def test_kron_empty(self, device, dtype):

        # 定义运行测试用例的内部函数，参数为空张量的形状 empty_shape
        def run_test_case(empty_shape):
            # 创建单位矩阵 a，数据类型为 dtype，在指定设备上
            a = torch.eye(3, dtype=dtype, device=device)
            # 创建空张量 b，数据类型为 dtype，在指定设备上，形状由 empty_shape 指定
            b = torch.empty(empty_shape, dtype=dtype, device=device)
            # 使用 torch.kron 计算 a 和 b 的 Kronecker 乘积
            result = torch.kron(a, b)
            # 使用 NumPy 计算 a 和 b 的 Kronecker 乘积，结果作为期望值
            expected = np.kron(a.cpu().numpy(), b.cpu().numpy())
            # 断言 torch.kron 计算结果与期望值相等
            self.assertEqual(result, expected)

            # 当第一个参数为空时，NumPy 的计算会失败
            result = torch.kron(b, a)
            # 断言 torch.kron 计算的结果形状与期望值形状相等
            self.assertEqual(result.shape, expected.shape)

        # 定义测试的空张量形状列表
        empty_shapes = [(0,), (2, 0), (1, 0, 3)]
        # 对每个空张量形状进行测试
        for empty_shape in empty_shapes:
            run_test_case(empty_shape)

    # 使用浮点数和复数数据类型执行测试
    @dtypes(*floating_and_complex_types())
    def test_kron_errors_and_warnings(self, device, dtype):
        # 如果输出张量不为空且形状错误，会触发警告
        a = torch.eye(3, dtype=dtype, device=device)
        b = torch.ones((2, 2), dtype=dtype, device=device)
        # 创建与 a 相同形状的空张量 out
        out = torch.empty_like(a)
        # 使用 warnings 捕获警告
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.kron(a, b, out=out)
            # 检查是否触发了警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 输出张量的数据类型应与输入张量的数据类型匹配
        out = torch.empty_like(a).to(torch.int)
        # 使用 self.assertRaisesRegex 断言捕获到特定的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "can't be cast to the desired output type"):
            torch.kron(a, b, out=out)

    # 此测试确认 torch.linalg.norm 函数的 dtype 参数按照文档预期工作
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble, torch.bfloat16, torch.float16)
    # 定义一个测试函数，用于测试 torch.linalg.norm 函数在不同数据类型和设备上的行为
    def test_norm_dtype(self, device, dtype):
        # 创建一个偏函数，用于生成指定数据类型和设备的张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        # 定义一个内部函数，运行单个测试用例
        def run_test_case(input_size, ord, keepdim, to_dtype):
            # 构建测试用例的信息消息
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'dtype={dtype}, to_dtype={to_dtype}')
            # 生成指定大小的输入张量
            input = make_arg(input_size)
            # 计算输入张量的指定范数，并断言结果的数据类型与输入实部的数据类型相同
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            self.assertEqual(result.dtype, input.real.dtype, msg=msg)

            # 创建一个空张量 result_out 作为输出参数，并计算指定范数，再次断言结果与 result 相同
            result_out = torch.empty((0), dtype=result.dtype, device=device)
            torch.linalg.norm(input, ord, keepdim=keepdim, out=result_out)
            self.assertEqual(result, result_out, msg=msg)

            # 将输入张量转换为指定数据类型，计算其范数，并与未转换的计算结果进行断言比较
            result = torch.linalg.norm(input.to(to_dtype), ord, keepdim=keepdim)
            result_with_dtype = torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype)
            self.assertEqual(result, result_with_dtype, msg=msg)

            # 创建与指定数据类型的结果张量相同的空张量 result_out_with_dtype，并计算范数，进行断言比较
            result_out_with_dtype = torch.empty_like(result_with_dtype)
            torch.linalg.norm(input, ord, keepdim=keepdim, dtype=to_dtype, out=result_out_with_dtype)
            self.assertEqual(result_with_dtype, result_out_with_dtype, msg=msg)

        # 定义一个范数的向量序列，包括整数、浮点数和特殊值
        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]

        # 对于非半精度数据类型，添加额外的浮点数范数值
        if dtype != torch.float16 and dtype != torch.bfloat16:
            ord_vector.extend([0.1, -0.1])
        
        # 定义矩阵范数的序列
        ord_matrix = ['fro', 'nuc', 1, -1, 2, -2, inf, -inf, None]
        
        # 定义输入张量的大小 S
        S = 10

        # 根据数据类型选择正确的标准化数据类型组合
        if dtype == torch.cfloat:
            norm_dtypes = (torch.cfloat, torch.cdouble)
        elif dtype == torch.cdouble:
            norm_dtypes = (torch.cdouble,)
        elif dtype in (torch.float16, torch.bfloat16, torch.float):
            norm_dtypes = (torch.float, torch.double)
        elif dtype == torch.double:
            norm_dtypes = (torch.double,)
        else:
            raise RuntimeError("Unsupported dtype")

        # 对 ord_vector 和 norm_dtypes 的笛卡尔积进行测试
        for ord, keepdim, norm_dtype in product(ord_vector, (True, False), norm_dtypes):
            run_test_case((S,) , ord, keepdim, norm_dtype)

        # 对 ord_matrix 和 norm_dtypes 的笛卡尔积进行测试
        for ord, keepdim, norm_dtype in product(ord_matrix, (True, False), norm_dtypes):
            # 对于特定的 ord 值，需要特定的函数或库支持，否则跳过测试
            if ord in [2, -2, 'nuc']:
                # 需要 torch.svdvals 支持
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue

                # 需要 LAPACK 或等效库支持
                if ((torch.device(device).type == 'cuda' and not torch.cuda.has_magma and not has_cusolver()) or
                   (torch.device(device).type == 'cpu' and not torch._C.has_lapack)):
                    continue
            run_test_case((S, S) , ord, keepdim, norm_dtype)

    # 此测试确认 torch.linalg.norm 在 bfloat16 和 half 数据类型上得到正确结果
    @dtypes(torch.bfloat16, torch.float16)
    # 定义一个测试函数，用于测试 torch.linalg.norm 在 bfloat16 和 half 类型上的行为
    def test_norm_bfloat16_and_half(self, device, dtype):
        # 使用偏函数 make_tensor 来创建指定设备和数据类型的张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        # 定义一个内部函数，运行单个测试案例
        def run_test_case(input_size, ord, keepdim):
            # 组装测试消息字符串，包含输入大小、ord、keepdim、数据类型等信息
            msg = (
                f'input_size={input_size}, ord={ord}, keepdim={keepdim}, '
                f'dtype={dtype}')
            # 创建指定大小的填充为 1 的输入张量
            input = make_arg(input_size).fill_(1)
            # 使用 float 类型计算参考结果
            result_ref = torch.linalg.norm(input.float(), ord, keepdim=keepdim).to(dtype=dtype)
            # 使用 torch.linalg.norm 计算当前数据类型的结果
            result = torch.linalg.norm(input, ord, keepdim=keepdim)
            # 断言参考结果与当前结果相等
            self.assertEqual(result_ref, result, msg=msg)

        # 定义 ord 的测试向量，包含不同的范数值和特殊情况如 inf、-inf、None
        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf, None]
        # 针对每种输入大小、ord、keepdim 的组合进行测试
        for S, ord, keepdim in product((10, 2049), ord_vector, (True, False)):
            run_test_case((S,) , ord, keepdim, )

    # 使用 dtypes 装饰器指定多种数据类型，对 vector_norm_dim_tuple_arg 函数进行测试
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble, torch.bfloat16, torch.float16)
    def test_vector_norm_dim_tuple_arg(self, device):
        # 定义测试案例列表，每个案例包含输入大小、维度元组、预期错误类型和错误消息
        test_cases = [
            # input size, dim, error, error message
            ((4, ), (0, ), None, None),
            ((4, ), (1, ), IndexError, r'Dimension out of range'),
            ((4, ), (-2, ), IndexError, r'Dimension out of range'),
            ((4, 3), (0, -1), None, None),
            ((4, 3), (0, 0), RuntimeError, r'dim 0 appears multiple times in the list of dims'),
            ((4, 3), (0, -2), RuntimeError, r'dim 0 appears multiple times in the list of dims'),
            ((4, 3), (0, 1.0), TypeError, r"argument 'dim' must be tuple of ints"),
            ((4, 3), (None, ), TypeError, r"argument 'dim' must be tuple of ints"),
        ]
        # 遍历每个测试案例
        for input_size, dim_tuple, error, error_msg in test_cases:
            # 在指定设备上生成随机张量作为输入
            input = torch.randn(input_size, device=device)
            # vector_norm 应该接受一个整数元组或列表作为 dim 参数
            for dim in [dim_tuple, list(dim_tuple)]:
                if error is None:
                    # 如果没有预期错误，调用 vector_norm 函数
                    torch.linalg.vector_norm(input, dim=dim)
                else:
                    # 否则使用断言捕获预期错误
                    with self.assertRaises(error):
                        torch.linalg.vector_norm(input, dim=dim)

    # 此测试比较 torch.linalg.norm 和 numpy.linalg.norm 的向量范数结果，确保它们一致
    @dtypes(torch.float, torch.double)
    # 定义一个测试方法，用于测试向量的规范化函数
    def test_norm_vector(self, device, dtype):
        # 定义一个内部函数，用于运行测试案例
        def run_test_case(input, ord, dim, keepdim):
            # 使用 torch.linalg.norm 计算输入张量的规范化
            result = torch.linalg.norm(input, ord, dim, keepdim)
            # 将输入张量转换为 numpy 数组，并使用 np.linalg.norm 计算规范化
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            # 准备测试失败时的消息
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            # 断言 torch.linalg.norm 的结果与 np.linalg.norm 的结果应该相等
            self.assertEqual(result, result_numpy, msg=msg)

            # 创建一个和 result 维度相同的空张量 result_out
            result_out = torch.empty_like(result)
            # 使用 torch.linalg.norm 将结果存储在 result_out 中
            torch.linalg.norm(input, ord, dim, keepdim, out=result_out)
            # 断言 result 和 result_out 应该相等
            self.assertEqual(result, result_out, msg=msg)

        # 定义不同的 ord 参数的向量
        ord_vector = [0, 1, -1, 2, -2, 3, -3, 4.5, -4.5, inf, -inf]
        S = 10
        # 定义多个测试案例，每个案例包括输入大小、ord 设置和维度
        test_cases = [
            # 一维输入，不同的 ord 设置，无维度限制
            ((S, ), ord_vector, None),
            # 一维输入，不同的 ord 设置，维度为 0
            ((S, ), ord_vector, 0),
            # 三维输入，不同的 ord 设置，维度为 0
            ((S, S, S), ord_vector, 0),
            # 三维输入，不同的 ord 设置，维度为 1
            ((S, S, S), ord_vector, 1),
            # 三维输入，不同的 ord 设置，维度为 2
            ((S, S, S), ord_vector, 2),
            # 三维输入，不同的 ord 设置，维度为 -1（倒数第一维）
            ((S, S, S), ord_vector, -1),
            # 三维输入，不同的 ord 设置，维度为 -2（倒数第二维）
            ((S, S, S), ord_vector, -2),
        ]
        L = 1_000_000
        # 如果 dtype 是 torch.double，则添加一个大尺寸输入的测试案例
        if dtype == torch.double:
            test_cases.append(((L, ), ord_vector, None))
        # 遍历 keepdim 参数的 True 和 False 值
        for keepdim in [True, False]:
            # 遍历所有的输入大小、ord 设置和维度组合
            for input_size, ord_settings, dim in test_cases:
                # 根据指定的设备和 dtype 创建一个随机输入张量
                input = torch.randn(*input_size, dtype=dtype, device=device)
                # 遍历所有的 ord 设置
                for ord in ord_settings:
                    # 运行测试案例
                    run_test_case(input, ord, dim, keepdim)

    # 此测试函数用于比较 torch.linalg.norm、torch.linalg.matrix_norm 和 numpy.linalg.norm
    # 确保它们的矩阵范数计算结果一致
    @skipMeta  # 在 GitHub 上的问题跟踪页面说明了这个装饰器的作用 https://github.com/pytorch/pytorch/issues/54082
    @skipCUDAIfNoMagma
    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 2e-4})
    # 定义一个测试函数，用于测试标准化矩阵操作
    def test_norm_matrix(self, device, dtype):
        # partial 函数用于生成指定设备和数据类型的张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        # 定义一个运行测试用例的函数
        def run_test_case(input, ord, dim, keepdim):
            # 构建测试用例的信息消息字符串
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            # 使用 torch.linalg.norm 计算输入张量的范数
            result = torch.linalg.norm(input, ord, dim, keepdim)
            # 将输入张量转为 NumPy 数组后使用 np.linalg.norm 计算范数
            input_numpy = input.cpu().numpy()
            result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)

            # 使用 torch.linalg.norm 再次计算输入张量的范数
            result = torch.linalg.norm(input, ord, dim, keepdim)
            # 断言两种范数计算结果相等，否则输出消息 msg
            self.assertEqual(result, result_numpy, msg=msg)
            # 如果 ord 和 dim 都不为 None，则使用 torch.linalg.matrix_norm 计算矩阵范数
            if ord is not None and dim is not None:
                result = torch.linalg.matrix_norm(input, ord, dim, keepdim)
                # 再次断言两种范数计算结果相等，否则输出消息 msg
                self.assertEqual(result, result_numpy, msg=msg)

        # 定义一组矩阵范数的测试用例
        ord_matrix = [1, -1, 2, -2, inf, -inf, 'nuc', 'fro']
        S = 10
        test_cases = [
            # input size, dim
            ((S, S), None),
            ((S, S), (0, 1)),
            ((S, S), (1, 0)),
            ((S, S, S, S), (2, 0)),
            ((S, S, S, S), (-1, -2)),
            ((S, S, S, S), (-1, -3)),
            ((S, S, S, S), (-3, 2)),
        ]

        # 使用 itertools.product 对测试用例进行排列组合，生成完整的测试集
        for (shape, dim), keepdim, ord in product(test_cases, [True, False], ord_matrix):
            # 若 ord 是 2, -2, 'nuc' 中的一种，则需要特定的函数支持
            if ord in [2, -2, 'nuc']:
                # 需要 torch.svdvals 支持
                if dtype == torch.float16 or dtype == torch.bfloat16:
                    continue
                # 需要 LAPACK 或其等效库的支持
                if ((torch.device(device).type == 'cuda' and not torch.cuda.has_magma and not has_cusolver()) or
                   (torch.device(device).type == 'cpu' and not torch._C.has_lapack)):
                    continue
            # 运行测试用例
            run_test_case(make_arg(shape), ord, dim, keepdim)


    # 仅在 CUDA 上运行的测试函数装饰器
    @onlyCUDA
    # 指定数据类型为 torch.bfloat16 和 torch.float16 的测试函数装饰器
    @dtypes(torch.bfloat16, torch.float16)
    def test_norm_fused_type_promotion(self, device, dtype):
        # 生成指定设备和数据类型的随机张量
        x = torch.randn(10, device=device, dtype=dtype)

        # 定义一个性能分析并检查结果的函数
        def profile_and_check(fn, x, kwargs):
            # 使用 torch.profiler.profile 开始性能分析
            with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU,)) as p:
                # 调用指定的函数 fn 进行测试，并记录使用 torch.float 类型
                fn(x, **kwargs, dtype=torch.float)
            # 确认性能分析器返回了一些事件
            self.assertTrue("aten::linalg_vector_norm" in (e.name for e in p.events()))
            # 检查性能分析中没有显式的数据拷贝操作
            self.assertFalse("aten::to" in (e.name for e in p.events()))

        # 遍历测试的函数和参数组合，分别进行性能分析和结果检查
        for f, kwargs, in zip((torch.linalg.vector_norm, torch.norm), ({}, {"p" : 2})):
            profile_and_check(f, x, kwargs)


    # 标记为跳过的元测试，因为存在已知问题 https://github.com/pytorch/pytorch/issues/53739
    @skipMeta  # https://github.com/pytorch/pytorch/issues/53739
    # 如果没有 LAPACK 库，则在 CPU 上跳过测试
    @skipCPUIfNoLapack
    # 如果没有 MAGMA 库，则在 CUDA 上跳过测试
    @skipCUDAIfNoMagma
    # 使用浮点数和复数类型进行测试的装饰器
    @dtypes(*floating_and_complex_types())
    # 针对 torch.float32 的精度覆盖测试
    @precisionOverride({torch.float32: 1e-3})
    # 定义测试方法，接受设备和数据类型作为参数
    def test_cond(self, device, dtype):
        # 定义运行测试用例的内部函数，接受输入和范数类型作为参数
        def run_test_case(input, p):
            # 使用 torch.linalg.cond 计算输入张量的条件数，并存储结果
            result = torch.linalg.cond(input, p)
            # 使用 numpy.linalg.cond 计算输入张量的条件数（需要先将输入张量转为 NumPy 数组），并存储结果
            result_numpy = np.linalg.cond(input.cpu().numpy(), p)
            # 使用 self.assertEqual 断言两个结果的近似相等性，指定相对和绝对误差容差，确保不要求精确的数据类型匹配
            self.assertEqual(result, result_numpy, rtol=1e-2, atol=self.precision, exact_dtype=False)
            # 使用 self.assertEqual 断言两个结果的形状相等
            self.assertEqual(result.shape, result_numpy.shape)

            # 测试 out= 变体的情况
            out = torch.empty_like(result)
            # 使用 torch.linalg.cond 计算输入张量的条件数，将结果存储在预先分配的 out 张量中，并存储 ans
            ans = torch.linalg.cond(input, p, out=out)
            # 使用 self.assertEqual 断言 ans 和 out 的相等性
            self.assertEqual(ans, out)
            # 使用 self.assertEqual 断言 ans 和 result 的相等性
            self.assertEqual(ans, result)

        # 定义各种范数类型的列表
        norm_types = [1, -1, 2, -2, inf, -inf, 'fro', 'nuc', None]
        # 定义输入张量的尺寸列表
        input_sizes = [(32, 32), (2, 3, 3, 3)]
        # 遍历输入张量尺寸列表
        for input_size in input_sizes:
            # 使用指定设备和数据类型生成随机输入张量
            input = torch.randn(*input_size, dtype=dtype, device=device)
            # 遍历范数类型列表
            for p in norm_types:
                # 运行测试用例
                run_test_case(input, p)

        # 测试空批量尺寸的情况
        input_sizes = [(0, 3, 3), (0, 2, 5, 5)]
        # 遍历空批量尺寸列表
        for input_size in input_sizes:
            # 使用指定设备和数据类型生成随机输入张量
            input = torch.randn(*input_size, dtype=dtype, device=device)
            # 遍历范数类型列表
            for p in norm_types:
                # 运行测试用例
                run_test_case(input, p)

        # 测试非方阵输入的情况
        input_sizes = [(16, 32), (32, 16), (2, 3, 5, 3), (2, 3, 3, 5)]
        # 遍历非方阵输入尺寸列表
        for input_size in input_sizes:
            # 使用指定设备和数据类型生成随机输入张量
            input = torch.randn(*input_size, dtype=dtype, device=device)
            # 遍历指定范数类型列表
            for p in [2, -2, None]:
                # 运行测试用例
                run_test_case(input, p)

        # 测试奇异输入的情况
        a = torch.eye(3, dtype=dtype, device=device)
        a[-1, -1] = 0  # 使 'a' 成为奇异矩阵
        # 遍历范数类型列表
        for p in norm_types:
            try:
                # 尝试运行测试用例
                run_test_case(a, p)
            except np.linalg.LinAlgError:
                # 如果 numpy 抛出奇异值错误，这在 BLAS 后端中很少见
                # 参考 https://github.com/pytorch/pytorch/issues/67675
                pass

        # 测试 0x0 矩阵的情况。对于这样的输入，NumPy 无法计算，我们返回 0
        input_sizes = [(0, 0), (2, 5, 0, 0)]
        # 遍历 0x0 矩阵尺寸列表
        for input_size in input_sizes:
            # 使用指定设备和数据类型生成零填充的输入张量
            input = torch.randn(*input_size, dtype=dtype, device=device)
            # 遍历范数类型列表，仅对 'fro' 和 2 范数进行测试
            for p in ['fro', 2]:
                # 根据输入尺寸生成期望的输出 dtype
                expected_dtype = a.real.dtype if dtype.is_complex else dtype
                # 生成期望的结果张量，全为零
                expected = torch.zeros(input_size[:-2], dtype=expected_dtype, device=device)
                # 使用 torch.linalg.cond 计算输入张量的条件数，与期望结果进行断言
                actual = torch.linalg.cond(input, p)
                self.assertEqual(actual, expected)

    # 跳过元信息的测试装饰器
    @skipMeta  # 参考 https://github.com/pytorch/pytorch/issues/53739
    # 如果没有 LAPACK 库，跳过 CPU 测试的装饰器
    @skipCPUIfNoLapack
    # 如果没有 MAGMA 库，跳过 CUDA 测试的装饰器
    @skipCUDAIfNoMagma
    # 使用浮点数和复数类型进行数据类型的装饰器
    @dtypes(*floating_and_complex_types())
    # 覆盖精度，针对 torch.float32 设置较小的容差
    @precisionOverride({torch.float32: 1e-3})
    # 这个测试调用 torch.linalg.norm 和 numpy.linalg.norm，使用非法参数确保它们都抛出错误
    @dtypes(torch.float, torch.double)
    # 定义一个测试方法，用于测试 torch.linalg.norm 函数的错误情况
    def test_norm_errors(self, device, dtype):
        
        # 定义运行单个错误测试用例的函数
        def run_error_test_case(input, ord, dim, keepdim, error_type, error_regex):
            # 构建测试用例的信息字符串，包括输入大小、ord、dim、keepdim、dtype
            test_case_info = (
                f'test case input.size()={input.size()}, ord={ord}, dim={dim}, '
                f'keepdim={keepdim}, dtype={dtype}')
            
            # 使用 assertRaisesRegex 验证 torch.linalg.norm 是否会抛出指定的错误类型和正则表达式匹配的异常消息
            with self.assertRaisesRegex(error_type, error_regex, msg=test_case_info):
                torch.linalg.norm(input, ord, dim, keepdim)
            
            # 将 PyTorch 张量转换为 NumPy 数组，用于验证 NumPy 是否会相同的异常信息
            input_numpy = input.cpu().numpy()
            
            # 验证 NumPy 是否也会抛出异常，如果没有则抛出自定义异常消息
            msg = f'numpy does not raise error but pytorch does, for case "{test_case_info}"'
            with self.assertRaises(Exception, msg=test_case_info):
                np.linalg.norm(input_numpy, ord, dim, keepdim)

        # 定义一个大小常量 S
        S = 10
        
        # 定义多个错误测试用例，每个用例包括输入大小、p 设置、dim、错误类型、错误正则表达式
        error_test_cases = [
            # input size, p settings, dim, error type, error regex
            ((S, ), ['fro', 'nuc'], None, RuntimeError, r'A must have at least 2 dimensions'),
            ((S, S), [3.5], None, RuntimeError, r'matrix_norm: Order 3.5 not supported'),
            ((S, S), [0], None, RuntimeError, r'matrix_norm: Order 0 not supported'),
            ((S, S), ['fail'], None, RuntimeError, r'matrix_norm: Order fail not supported'),
            ((S, S), ['fro', 'nuc'], 0, RuntimeError, r'matrix_norm: dim must be a 2-tuple'),
            ((S, S), ['fro', 'nuc', 2], (0, 0), RuntimeError, r'dims must be different'),
            ((S, S), ['fro', 'nuc', 2], (-1, 1), RuntimeError, r'dims must be different'),
            ((S, S), ['fro', 'nuc', 2], (0, 4), IndexError, r'Dimension out of range'),
            ((S, ), [0], (4, ), IndexError, r'Dimension out of range'),
            ((S, ), [None], (0, 0), RuntimeError, r'dim 0 appears multiple times'),
            ((S, S, S), [1], (0, 1, 2), RuntimeError, r"If dim is specified, it must be of length 1 or 2."),
            ((S, S, S), [1], None, RuntimeError, r"If dim is not specified but ord is, the input must be 1D or 2D"),
        ]
        
        # 遍历所有 keepdim 的可能取值，True 和 False
        for keepdim in [True, False]:
            # 遍历所有错误测试用例
            for input_size, ord_settings, dim, error_type, error_regex in error_test_cases:
                # 根据输入大小和数据类型在指定设备上生成一个随机张量
                input = torch.randn(*input_size, dtype=dtype, device=device)
                
                # 对于每种 p 设置，运行单个错误测试用例
                for ord in ord_settings:
                    run_error_test_case(input, ord, dim, keepdim, error_type, error_regex)
    # 测试复杂情况下的标准化函数
    def test_norm_complex(self, device, dtype):
        # 生成错误信息函数，用于生成标准化失败时的错误消息
        def gen_error_message(input_size, ord, keepdim, dim=None):
            return f"complex norm failed for input size {input_size}, ord={ord}, keepdim={keepdim}, dim={dim}"

        # 向量的不同标准化方式列表
        vector_ords = [None, 0, 1, 2, 3, inf, -1, -2, -3, -inf]
        # 矩阵的不同标准化方式列表
        matrix_ords = [None, 'fro', 'nuc', 1, 2, inf, -1, -2, -inf]

        # 测试支持的标准化方式
        for keepdim in [False, True]:
            # 向量标准化
            x = torch.randn(25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in vector_ords:
                # 计算向量的标准化结果
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                # 断言标准化结果形状和期望相同
                self.assertEqual(res.shape, expected.shape, msg=msg)
                # 断言标准化结果值和期望相同
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                # 使用预分配的张量计算标准化结果
                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                # 断言预分配张量的标准化结果形状和期望相同
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                # 断言预分配张量的标准化结果值和期望相同
                self.assertEqual(res_out, expected, msg=msg)

            # 矩阵标准化
            x = torch.randn(25, 25, device=device, dtype=dtype)
            xn = x.cpu().numpy()
            for ord in matrix_ords:
                # 计算矩阵的标准化结果
                res = torch.linalg.norm(x, ord, keepdim=keepdim).cpu()
                expected = np.linalg.norm(xn, ord, keepdims=keepdim)
                msg = gen_error_message(x.size(), ord, keepdim)
                # 断言标准化结果形状和期望相同
                self.assertEqual(res.shape, expected.shape, msg=msg)
                # 断言标准化结果值和期望相同
                self.assertEqual(res, expected, msg=msg, exact_dtype=False)

                # 使用预分配的张量计算标准化结果
                res_out = torch.tensor([], device=device, dtype=res.dtype)
                torch.linalg.norm(x, ord, keepdim=keepdim, out=res_out)
                # 断言预分配张量的标准化结果形状和期望相同
                self.assertEqual(res_out.shape, expected.shape, msg=msg)
                # 断言预分配张量的标准化结果值和期望相同
                self.assertEqual(res_out, expected, msg=msg)

    # 测试当输入包含极端值（inf, -inf, nan）时，linal.vector_norm 与 numpy 的结果是否一致
    def test_vector_norm_extreme_values(self, device):
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        vectors = []
        # 生成包含极端值的向量组合
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
        # 对每个向量进行测试
        for vector in vectors:
            x = torch.tensor(vector, device=device)
            x_n = x.cpu().numpy()
            for ord in vector_ords:
                msg = f'ord={ord}, vector={vector}'
                # 计算向量的标准化结果并断言与 numpy 的结果一致
                result = torch.linalg.vector_norm(x, ord=ord)
                result_n = np.linalg.norm(x_n, ord=ord)
                self.assertEqual(result, result_n, msg=msg)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 定义一个测试方法，用于测试一维向量的规范化
    def test_vector_norm_reduce_over_1D_vector(self, device, dtype):
        # 定义不同的输入大小和维度配置
        input_sizes_and_dims = [
            ((6, 1), -1),  # 向量大小为 (6, 1)，维度为 -1
            ((3, 1, 2, 1), (1, 3)),  # 向量大小为 (3, 1, 2, 1)，维度为 (1, 3)
            ((1,), None),  # 向量大小为 (1,)，无特定维度
        ]
        # 定义不同的规范化阶数
        orders = [float('inf'), -float('inf'), 0, 1, -1, 2, -2]
        # 定义是否保持维度的选项
        keepdims = [True, False]

        # 对于每个输入大小、规范化阶数和保持维度的组合进行迭代测试
        for input_size_and_dim, ord, keepdim in product(input_sizes_and_dims, orders, keepdims):
            # 提取当前输入的大小和维度
            input_size = input_size_and_dim[0]
            dim = input_size_and_dim[1]
            # 如果维度是元组且规范化阶数为 0，则跳过测试，因为 np.linalg.norm 会引发异常 'ValueError: Invalid norm order for matrices.'
            if type(dim) is tuple and ord == 0:
                continue
            # 生成指定设备和数据类型的输入向量
            input = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)
            # 计算输入向量的规范化
            result = torch.linalg.vector_norm(input, ord, dim, keepdim)
            # 使用 numpy 计算相同输入向量的规范化结果
            result_numpy = np.linalg.norm(input.cpu().numpy(), ord, dim, keepdim)

            # 设置测试消息，包含输入向量的尺寸、规范化阶数、维度、保持维度选项和数据类型
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            # 断言 torch 的规范化结果与 numpy 的结果相等
            self.assertEqual(result, result_numpy, msg=msg)

    # 使用装饰器定义跳过 CUDA 环境没有 Magma 和没有 Cusolver 的测试
    @skipCUDAIfNoMagmaAndNoCusolver
    # 使用装饰器定义跳过 CPU 环境没有 Lapack 的测试
    @skipCPUIfNoLapack
    # 使用装饰器定义数据类型为 float 和 double 的测试
    @dtypes(torch.float, torch.double)
    # 覆盖 float32 数据类型的测试精度
    @precisionOverride({torch.float32: 2e-5})
    # 定义测试矩阵的规范化方法
    def test_matrix_norm(self, device, dtype):
        # 测试 torch.linalg.matrix_norm 与 torch.linalg.norm 不一致的输入
        A = make_tensor((2, 2, 2), dtype=dtype, device=device)

        # 使用断言检查是否会引发指定异常：'linalg.matrix_norm:.*must have at least 2 dimensions.*'
        with self.assertRaisesRegex(RuntimeError, r'linalg.matrix_norm:.*must have at least 2 dimensions.*'):
            torch.linalg.matrix_norm(make_tensor((2,), dtype=dtype, device=device))
        # 使用断言检查是否会引发指定异常：'linalg.matrix_norm:.*must be a 2-tuple.*'
        with self.assertRaisesRegex(RuntimeError, r'linalg.matrix_norm:.*must be a 2-tuple.*'):
            torch.linalg.matrix_norm(A, dim=(0,))
        # 使用断言检查是否会引发指定异常：'.*not supported.*'
        with self.assertRaisesRegex(RuntimeError, r'.*not supported.*'):
            torch.linalg.matrix_norm(A, ord=0)
        # 使用断言检查是否会引发指定异常：'.*not supported.*'
        with self.assertRaisesRegex(RuntimeError, r'.*not supported.*'):
            torch.linalg.matrix_norm(A, ord=3.0)

        # 测试 dim=None 的行为是否与 torch.linalg.norm 一致
        ref = torch.linalg.norm(A, dim=(-2, -1))
        res = torch.linalg.matrix_norm(A)
        # 断言 torch.linalg.matrix_norm 的结果与 torch.linalg.norm 的结果相等
        self.assertEqual(ref, res)

    # 测试当输入包含极端值（inf、-inf、nan）时，torch.linalg.norm 是否与 numpy 的结果一致
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    @unittest.skipIf(IS_MACOS, "Skipped on MacOS!")
    # 使用装饰器定义跳过 CUDA 环境没有 Magma 的测试
    @skipCUDAIfNoMagma
    # 使用装饰器定义跳过 CPU 环境没有 Lapack 的测试
    @skipCPUIfNoLapack
    # 定义一个测试函数，用于测试在极端值下的向量和矩阵的归一化情况
    def test_norm_extreme_values(self, device):
        # 定义包含各种数值的向量范数，包括无穷大和负无穷大
        vector_ords = [0, 1, 2, 3, inf, -1, -2, -3, -inf]
        # 目前跳过了 'nuc', 2, -2，参见问题 https://github.com/pytorch/pytorch/issues/71911
        matrix_ords = ['fro', 1, inf, -1, -inf]
        vectors = []
        matrices = []
        # 生成所有可能的向量和矩阵对，并将其存储在列表中
        for pair in itertools.product([inf, -inf, 0.0, nan, 1.0], repeat=2):
            vectors.append(list(pair))
            matrices.append([[pair[0], pair[1]]])
            matrices.append([[pair[0]], [pair[1]]])
        # 对每个向量进行测试
        for vector in vectors:
            # 将向量转换为指定设备上的张量
            x = torch.tensor(vector).to(device)
            # 将张量转换为 numpy 数组
            x_n = x.cpu().numpy()
            # 遍历所有向量范数，生成相应的消息
            for ord in vector_ords:
                msg = f'ord={ord}, vector={vector}'
                # 计算张量的范数
                result = torch.linalg.norm(x, ord=ord)
                # 计算 numpy 数组的范数
                result_n = np.linalg.norm(x_n, ord=ord)
                # 使用断言比较两者是否相等
                self.assertEqual(result, result_n, msg=msg)

        # TODO: 一旦修复了有问题的案例，删除此函数
        def is_broken_matrix_norm_case(ord, x):
            # 如果设备类型是 'cuda'
            if self.device_type == 'cuda':
                # 如果张量的大小为 [1, 2]
                if x.size() == torch.Size([1, 2]):
                    # 如果 ord 在 ['nuc', 2, -2] 中，且第一个元素为 NaN，第二个元素为 1
                    if ord in ['nuc', 2, -2] and isnan(x[0][0]) and x[0][1] == 1:
                        # 这些情况由于 SVD 存在问题而出现错误
                        # 参见 https://github.com/pytorch/pytorch/issues/43567
                        return True
                # 如果 ord 在 ['nuc', 2, -2] 中，存在另一个与 SVD 相关的问题
                if ord in ['nuc', 2, -2]:
                    # 参见 https://github.com/pytorch/pytorch/issues/52633
                    return True
            return False

        # 对每个矩阵进行测试
        for matrix in matrices:
            # 将矩阵转换为指定设备上的张量
            x = torch.tensor(matrix).to(device)
            # 将张量转换为 numpy 数组
            x_n = x.cpu().numpy()
            # 遍历所有矩阵范数，生成相应的消息
            for ord in matrix_ords:
                msg = f'ord={ord}, matrix={matrix}'
                # 检查是否为有问题的矩阵范数案例，如果是则跳过
                if is_broken_matrix_norm_case(ord, x):
                    continue
                else:
                    # 计算 numpy 数组的范数
                    result_n = np.linalg.norm(x_n, ord=ord)
                    # 计算张量的范数
                    result = torch.linalg.norm(x, ord=ord)
                    # 使用断言比较两者是否相等
                    self.assertEqual(result, result_n, msg=msg)

    # 使用 numpy 计算的 linalg.norm 向量范数进行测试，验证其与 PyTorch 结果的一致性
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 定义测试函数，用于测试正常化向量的各种边缘情况
    def test_norm_vector_degenerate_shapes(self, device, dtype):
        # 定义内部函数，运行测试用例
        def run_test_case(input, ord, dim, keepdim):
            # 构建测试信息字符串，包括输入大小、ord值、维度dim、keepdim标志和数据类型dtype
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            # 如果输入为空且ord为负数或inf，并且维度dim为None或者指定维度大小为0，则预期引发运行时错误
            if (input.numel() == 0 and
                (ord < 0. or ord == inf) and
                (dim is None or input.shape[dim] == 0)):
                with self.assertRaises(RuntimeError):
                    torch.linalg.norm(input, ord, dim, keepdim)
            else:
                # 将输入张量转换为numpy数组
                input_numpy = input.cpu().numpy()
                # 使用numpy计算输入数据的norm，与torch的结果进行比较
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                result = torch.linalg.norm(input, ord, dim, keepdim)
                # 断言torch计算的结果与numpy计算的结果相等，否则抛出错误，打印msg信息
                self.assertEqual(result, result_numpy, msg=msg)

        # 定义测试的ord向量，包含各种可能的ord值
        ord_vector = [0, 0.5, 1, 2, 3, inf, -0.5, -1, -2, -3, -inf]
        # 设置测试维度的大小
        S = 10
        # 定义测试用例，每个元素为一个元组，包含输入大小和指定的维度dim
        test_cases = [
            # input size, dim
            ((0, ), None),
            ((0, S), 0),
            ((0, S), 1),
            ((S, 0), 0),
            ((S, 0), 1),
        ]
        # 遍历keepdim的True和False两种情况
        for keepdim in [True, False]:
            # 遍历测试用例集合
            for input_size, dim in test_cases:
                # 生成指定大小的随机输入张量
                input = torch.randn(*input_size, dtype=dtype, device=device)
                # 遍历ord向量，对每个输入进行测试
                for ord in ord_vector:
                    run_test_case(input, ord, dim, keepdim)

    # 使用skipCUDAIfNoMagma装饰器，跳过没有Magma支持的CUDA环境
    # 使用skipCPUIfNoLapack装饰器，跳过没有Lapack支持的CPU环境
    # 使用dtypes装饰器，指定测试浮点数、双精度浮点数、复数浮点数和复数双精度浮点数
    def test_norm_matrix_degenerate_shapes(self, device, dtype):
        # 定义测试用例函数，测试输入形状的正常和异常情况
        def run_test_case(input, ord, dim, keepdim, should_error):
            # 构建用于显示错误消息的字符串
            msg = f'input.size()={input.size()}, ord={ord}, dim={dim}, keepdim={keepdim}, dtype={dtype}'
            # 将输入转换为 NumPy 数组
            input_numpy = input.cpu().numpy()
            # 定义要测试的操作列表，包括 torch.linalg.norm
            ops = [torch.linalg.norm]

            # 如果 ord 和 dim 均不为 None，则添加 torch.linalg.matrix_norm 到操作列表
            if ord is not None and dim is not None:
                ops.append(torch.linalg.matrix_norm)

            # 如果应该出现错误，则使用 assertRaises 检测错误
            if should_error:
                # 使用 NumPy 的 norm 函数预期出现 ValueError
                with self.assertRaises(ValueError):
                    np.linalg.norm(input_numpy, ord, dim, keepdim)
                # 对于每个操作，预期出现 IndexError
                for op in ops:
                    with self.assertRaises(IndexError):
                        op(input, ord, dim, keepdim)
            else:
                # 否则，计算 NumPy 的 norm 结果
                result_numpy = np.linalg.norm(input_numpy, ord, dim, keepdim)
                # 对于每个操作，比较其结果与 NumPy 的结果是否一致
                for op in ops:
                    result = op(input, ord, dim, keepdim)
                    self.assertEqual(result, result_numpy, msg=msg)

        # ord_matrix 列表包含了测试的不同矩阵范数设置
        ord_matrix = ['fro', 'nuc', 1, 2, inf, -1, -2, -inf, None]
        S = 10
        # test_cases 列表包含了各种输入大小和设置下的测试用例
        test_cases = [
            # input size, p settings that cause error, dim
            ((0, 0), [1, 2, inf, -1, -2, -inf], None),
            ((0, S), [2, inf, -2, -inf], None),
            ((S, 0), [1, 2, -1, -2], None),
            ((S, S, 0), [], (0, 1)),
            ((1, S, 0), [], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (0, 1)),
            ((0, 0, S), [1, 2, inf, -1, -2, -inf], (1, 0)),
        ]

        # 遍历测试用例
        for keepdim in [True, False]:
            for input_size, error_ords, dim in test_cases:
                # 生成指定大小的随机输入张量
                input = torch.randn(*input_size, dtype=dtype, device=device)
                # 遍历 ord_matrix 中的每个矩阵范数设置
                for ord in ord_matrix:
                    # 运行测试用例
                    run_test_case(input, ord, dim, keepdim, ord in error_ords)

    def test_norm_fastpaths(self, device):
        # 生成指定设备上的随机张量 x
        x = torch.randn(3, 5, device=device)

        # slow path
        # 使用 ord=4.5 运行 torch.linalg.norm 的慢速计算路径
        result = torch.linalg.norm(x, 4.5, 1)
        # 计算预期结果
        expected = torch.pow(x.abs().pow(4.5).sum(1), 1.0 / 4.5)
        # 检查结果是否相等
        self.assertEqual(result, expected)

        # fast 0-norm
        # 使用 ord=0 运行 torch.linalg.norm 的快速计算路径
        result = torch.linalg.norm(x, 0, 1)
        # 计算预期结果
        expected = (x != 0).type_as(x).sum(1)
        # 检查结果是否相等
        self.assertEqual(result, expected)

        # fast 1-norm
        # 使用 ord=1 运行 torch.linalg.norm 的快速计算路径
        result = torch.linalg.norm(x, 1, 1)
        # 计算预期结果
        expected = x.abs().sum(1)
        # 检查结果是否相等
        self.assertEqual(result, expected)

        # fast 2-norm
        # 使用 ord=2 运行 torch.linalg.norm 的快速计算路径
        result = torch.linalg.norm(x, 2, 1)
        # 计算预期结果
        expected = torch.sqrt(x.pow(2).sum(1))
        # 检查结果是否相等
        self.assertEqual(result, expected)

        # fast 3-norm
        # 使用 ord=3 运行 torch.linalg.norm 的快速计算路径
        result = torch.linalg.norm(x, 3, 1)
        # 计算预期结果
        expected = torch.pow(x.pow(3).abs().sum(1), 1.0 / 3.0)
        # 检查结果是否相等
        self.assertEqual(result, expected)

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    # NumPy computes only in float64 and complex128 precisions
    # for float32 or complex64 results might be very different from float64 or complex128
    @dtypes(torch.float64, torch.complex128)
    # 定义一个测试函数，用于在给定的设备和数据类型上测试 numpy 版本的特征值分解
    def test_eig_numpy(self, device, dtype):
        # 定义一个内部函数，用于运行特定形状的特征值分解测试
        def run_test(shape, *, symmetric=False):
            # 导入需要使用的函数：random_symmetric_matrix 函数用于生成随机对称矩阵
            from torch.testing._internal.common_utils import random_symmetric_matrix
            
            # 根据情况生成输入矩阵 a
            if not dtype.is_complex and symmetric:
                # 对于非复数且对称的情况，生成随机对称矩阵
                # 注意：与 NumPy 不同，结果不会在此情况下转换为 float32 或 float64 类型
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                # 对于其他情况，使用给定的 shape、dtype 和 device 生成张量 a
                a = make_tensor(shape, dtype=dtype, device=device)
            
            # 调用 torch.linalg.eig 对矩阵 a 进行特征值分解，返回实际结果
            actual = torch.linalg.eig(a)

            # 与 NumPy 的特征值分解结果进行比较
            # 特征值的顺序不一定相同，因此 NumPy 和 PyTorch 的顺序可能不同
            expected = np.linalg.eig(a.cpu().numpy())

            # 对 NumPy 输出进行排序
            ind = np.argsort(expected[0], axis=-1)[::-1]
            expected = (np.take_along_axis(expected[0], ind, axis=-1), np.take_along_axis(expected[1], ind[:, None], axis=-1))

            # 对 PyTorch 输出进行排序
            # torch.argsort 无法处理复数输入，因此使用 CPU 上的 NumPy 排序
            ind = np.argsort(actual[0].cpu().numpy(), axis=-1)[::-1]
            actual_np = [x.cpu().numpy() for x in actual]
            sorted_actual = (
                np.take_along_axis(actual_np[0], ind, axis=-1),
                np.take_along_axis(actual_np[1], ind[:, None], axis=-1))

            # 使用 self.assertEqual 进行期望结果和排序后的实际结果的比较
            self.assertEqual(expected[0], sorted_actual[0], exact_dtype=False)
            self.assertEqual(abs(expected[1]), abs(sorted_actual[1]), exact_dtype=False)

        # 定义不同形状的测试用例
        shapes = [(0, 0),  # 空矩阵
                  (5, 5),  # 单个矩阵
                  (0, 0, 0), (0, 5, 5),  # 零批处理维度的张量
                  (2, 5, 5),  # 三维张量
                  (2, 1, 5, 5)]  # 四维张量
        
        # 遍历每种形状，分别运行测试
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    # 以下装饰器为 test_eig_numpy 函数添加了一些额外的功能和约束
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，用于比较不同后端的特征值分解
    def test_eig_compare_backends(self, device, dtype):
        # 定义内部函数，运行特征值分解的测试
        def run_test(shape, *, symmetric=False):
            # 导入随机对称矩阵生成函数
            from torch.testing._internal.common_utils import random_symmetric_matrix

            # 如果数据类型不是复数且要求对称，则生成对称实数矩阵
            if not dtype.is_complex and symmetric:
                # 对于对称的实数输入，特征值和特征向量的虚部为零
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                # 否则生成指定形状的张量
                a = make_tensor(shape, dtype=dtype, device=device)

            # 执行特征值分解
            actual = torch.linalg.eig(a)

            # 设置比较用的CPU设备
            complementary_device = 'cpu'

            # 使用CPU上的数据进行比较
            expected = torch.linalg.eig(a.to(complementary_device))
            # 断言实际特征值与期望特征值相等
            self.assertEqual(expected[0], actual[0])
            # 断言实际特征向量与期望特征向量相等
            self.assertEqual(expected[1], actual[1])

        # 不同形状的输入数据
        shapes = [(0, 0),  # 空矩阵
                  (5, 5),  # 单一矩阵
                  (0, 0, 0), (0, 5, 5),  # 零批次维度张量
                  (2, 5, 5),  # 三维张量
                  (2, 1, 5, 5)]  # 四维张量
        # 对每种形状的输入执行测试
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    # 标记为慢速测试，并仅在CUDA环境下运行，同时跳过没有MAGMA支持的情况
    @slowTest
    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(torch.float32)
    def test_eig_check_magma(self, device, dtype):
        # 仅对大于2048x2048大小的矩阵调用MAGMA库
        shape = (2049, 2049)
        a = make_tensor(shape, dtype=dtype, device=device)
        w, v = torch.linalg.eig(a)
        # 使用特征分解身份验证正确性
        self.assertEqual(a.to(v.dtype) @ v, w * v, atol=1e-3, rtol=1e-3)

    # 跳过没有MAGMA支持的情况，并且当没有LAPACK支持时在CPU上跳过
    # NumPy仅在float64和complex128精度上计算
    # 对于float32或complex64，结果可能与float64或complex128非常不同
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_eig_with_nan(self, device, dtype):
        # 对于无穷大和NaN的情况进行测试
        for val in [np.inf, np.nan]:
            for batch_dim in [(), (10,)]:
                # 生成具有NaN值的张量
                a = make_tensor((*batch_dim, 5, 5), device=device, dtype=dtype)
                a[..., -1, -1] = val

                # 使用断言捕获运行时错误，确保特征值分解不接受NaN值
                with self.assertRaisesRegex(RuntimeError, "torch.linalg.eig: input tensor should not"):
                    torch.linalg.eig(a)

    # 跳过没有MAGMA支持的情况，并且当没有LAPACK支持时在CPU上跳过
    # NumPy仅在float64和complex128精度上计算
    # 对于float32或complex64，结果可能与float64或complex128非常不同
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.float64, torch.complex128)
    # 定义一个测试函数，用于测试使用 NumPy 返回的特征值与 PyTorch 的 linalg.eigvals 函数计算的结果是否一致
    def test_eigvals_numpy(self, device, dtype):
        # 定义内部函数 run_test，用于运行特定形状的特征值测试
        def run_test(shape, *, symmetric=False):
            # 导入必要的函数 random_symmetric_matrix 用于生成随机对称矩阵
            from torch.testing._internal.common_utils import random_symmetric_matrix

            # 根据输入的形状和对称性生成测试用的张量 a
            if not dtype.is_complex and symmetric:
                # 对于非复数且对称的情况，特征值和特征向量的虚部为零
                # 与 NumPy 不同的是，在这种情况下结果不会被转换为 float32 或 float64 类型
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                a = make_tensor(shape, dtype=dtype, device=device)

            # 使用 PyTorch 的 linalg.eigvals 函数计算实际的特征值
            actual = torch.linalg.eigvals(a)

            # 与 NumPy 的比较
            # 特征值的顺序不一定相同，因此 NumPy 和 PyTorch 的顺序可能不同
            expected = np.linalg.eigvals(a.cpu().numpy())

            # 对 NumPy 输出进行排序
            ind = np.argsort(expected, axis=-1)[::-1]
            expected = np.take_along_axis(expected, ind, axis=-1)

            # 对 PyTorch 输出进行排序
            # torch.argsort 不适用于复数输入，因此使用在 CPU 上的 NumPy 排序
            ind = np.argsort(actual.cpu().numpy(), axis=-1)[::-1]
            actual_np = actual.cpu().numpy()
            sorted_actual = np.take_along_axis(actual_np, ind, axis=-1)

            # 使用 self.assertEqual 断言排序后的 PyTorch 输出与 NumPy 的期望输出一致
            self.assertEqual(expected, sorted_actual, exact_dtype=False)

        # 不同的形状列表，用于测试
        shapes = [(0, 0),  # 空矩阵
                  (5, 5),  # 单个矩阵
                  (0, 0, 0), (0, 5, 5),  # 零批处理维度张量
                  (2, 5, 5),  # 三维张量
                  (2, 1, 5, 5)]  # 四维张量
        for shape in shapes:
            # 运行非对称情况下的测试
            run_test(shape)
            # 运行对称情况下的测试
            run_test(shape, symmetric=True)

    # 标记仅在 CUDA 上运行的装饰器
    @onlyCUDA
    # 如果没有 magma 库则跳过 CUDA 测试的装饰器
    @skipCUDAIfNoMagma
    # 使用浮点数和复数类型作为参数的装饰器
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，用于比较不同后端计算环境下的特征值计算结果
    def test_eigvals_compare_backends(self, device, dtype):
        # 定义内部函数，执行特定形状的特征值计算测试
        def run_test(shape, *, symmetric=False):
            # 导入必要的函数，random_symmetric_matrix 用于生成随机对称矩阵
            from torch.testing._internal.common_utils import random_symmetric_matrix

            # 根据数据类型和设备生成对称或非对称的输入张量 a
            if not dtype.is_complex and symmetric:
                # 对于非复数类型且要求对称的情况，生成随机对称矩阵
                a = random_symmetric_matrix(shape[-1], *shape[:-2], dtype=dtype, device=device)
            else:
                # 否则，生成指定形状的张量 a
                a = make_tensor(shape, dtype=dtype, device=device)

            # 计算张量 a 的特征值
            actual = torch.linalg.eigvals(a)

            # 设置用于比较的 CPU 设备
            complementary_device = 'cpu'

            # 将张量 a 移到 CPU 上，并计算其特征值作为期望值
            expected = torch.linalg.eigvals(a.to(complementary_device))
            # 使用断言检查计算结果是否一致
            self.assertEqual(expected, actual)

            # 检查带有 out 参数的情况
            complex_dtype = dtype
            if not dtype.is_complex:
                # 对于非复数类型，使用适当的复数类型作为 complex_dtype
                complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
            # 创建一个空的输出张量 out，计算 a 的特征值并将结果存储在 out 中
            out = torch.empty(0, dtype=complex_dtype, device=device)
            ans = torch.linalg.eigvals(a, out=out)
            # 使用断言检查计算结果是否与 out 中的值一致
            self.assertEqual(ans, out)
            # 再次使用断言检查计算结果是否与之前计算的 expected 一致
            self.assertEqual(expected.to(complex_dtype), out)

            # 检查非连续的 out 张量
            if a.numel() > 0:
                # 创建一个非连续的复数类型张量 out，计算 a 的特征值并将结果存储在 out 中
                out = torch.empty(2 * shape[0], *shape[1:-1], dtype=complex_dtype, device=device)[::2]
                # 使用断言检查 out 是否为非连续张量
                self.assertFalse(out.is_contiguous())
                ans = torch.linalg.eigvals(a, out=out)
                # 使用断言检查计算结果是否与 out 中的值一致
                self.assertEqual(ans, out)
                # 再次使用断言检查计算结果是否与之前计算的 expected 一致
                self.assertEqual(expected.to(complex_dtype), out)

        # 定义不同形状的输入张量形状列表
        shapes = [(0, 0),  # 空矩阵
                  (5, 5),  # 单个矩阵
                  (0, 0, 0), (0, 5, 5),  # 零批次维度的张量
                  (2, 5, 5),  # 3 维张量
                  (2, 1, 5, 5)]  # 4 维张量
        # 遍历形状列表，并对每种形状执行特征值计算测试
        for shape in shapes:
            run_test(shape)
            run_test(shape, symmetric=True)

    # 跳过没有 Magma 库的 CUDA 测试装饰器
    @skipCUDAIfNoMagma
    # 跳过没有 Lapack 库的 CPU 测试装饰器
    @skipCPUIfNoLapack
    # 对于浮点数和复数类型，执行所有的数据类型测试装饰器
    @dtypes(*floating_and_complex_types())
    # 测试 torch.linalg.eigvals 在不同情况下的错误和警告处理

    def test_eigvals_errors_and_warnings(self, device, dtype):
        # eig 函数要求输入至少是二维张量
        a = make_tensor(2, dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.eigvals(a)

        # eig 函数要求输入是方阵
        a = make_tensor((2, 3), dtype=dtype, device=device)
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.eigvals(a)

        # 如果传递了具有浮点 dtype 的 out 张量以获取复数输出，则会抛出错误
        if not dtype.is_complex:
            # 特征方程为 p(lambda) = lambda^2 - 2lambda + 5 = 0，其根为 lambda = 1[+-]2i
            a = torch.tensor([[3., -2.], [4., -1.]], dtype=dtype, device=device)
            out = torch.empty(0, device=device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected eigenvalues to be safely castable"):
                torch.linalg.eigvals(a, out=out)

        # dtype 应该可以安全转换
        a = make_tensor((3, 3), dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got eigenvalues with dtype Int"):
            torch.linalg.eigvals(a, out=out)

        # 如果传递了非空的形状不匹配的 out 张量，则会发出警告
        out = torch.empty(1, device=device, dtype=torch.complex128)
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.linalg.eigvals(a, out=out)
            # 检查是否发生了警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 设备应该匹配
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out_w = torch.empty(0, device=wrong_device, dtype=torch.complex128)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.eigvals(a, out=out_w)
    # 定义一个测试方法，用于验证复杂情况下的 torch.norm 函数的行为
    def test_norm_complex_old(self, device):
        # 定义一个生成错误消息的内部函数，根据输入大小、p值、keepdim标志和dim维度
        def gen_error_message(input_size, p, keepdim, dim=None):
            return f"complex norm failed for input size {input_size}, p={p}, keepdim={keepdim}, dim={dim}"

        # 对于每个 keepdim 标志，进行测试
        for keepdim in [False, True]:
            # 测试向量的情况
            x = torch.randn(25, device=device) + 1j * torch.randn(25, device=device)
            xn = x.cpu().numpy()
            # 遍历不同的 p 值
            for p in [0, 1, 2, 3, inf, -1, -2, -3, -inf]:
                # 计算向量 x 的 p 范数，并转移到 CPU 上
                res = x.norm(p, keepdim=keepdim).cpu()
                # 计算 numpy 中向量的 p 范数
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                # 生成错误消息
                msg = gen_error_message(x.size(), p, keepdim)
                # 断言 torch.norm 计算结果的形状与 numpy 中计算结果的形状相同
                self.assertEqual(res.shape, expected.shape, msg=msg)
                # 断言 torch.norm 计算结果与 numpy 中计算结果相等
                self.assertEqual(res, expected, msg=msg)

            # 测试矩阵的情况
            x = torch.randn(25, 25, device=device) + 1j * torch.randn(25, 25, device=device)
            xn = x.cpu().numpy()
            # 遍历不同的 p 值，包括 'nuc' 和 'fro'
            for p in ['nuc', 'fro']:
                # 计算矩阵 x 的 p 范数，并转移到 CPU 上
                res = x.norm(p, keepdim=keepdim).cpu()
                # 计算 numpy 中矩阵的 p 范数
                expected = np.linalg.norm(xn, p, keepdims=keepdim)
                # 生成错误消息
                msg = gen_error_message(x.size(), p, keepdim)
                # 断言 torch.norm 计算结果的形状与 numpy 中计算结果的形状相同，并设置数值容差和相对容差
                self.assertEqual(res.shape, expected.shape, msg=msg)
                self.assertEqual(res, expected, msg=msg, rtol=4e-6, atol=6e-4)

    # 确保 torch.norm 在 p='fro' 和 p=2 时对相互支持的输入组合给出相同的结果
    @dtypes(torch.float)
    def test_norm_fro_2_equivalence_old(self, device, dtype):
        # 定义不同的输入尺寸
        input_sizes = [
            (0,),
            (10,),
            (0, 0),
            (4, 30),
            (0, 45),
            (100, 0),
            (45, 10, 23),
            (0, 23, 59),
            (23, 0, 37),
            (34, 58, 0),
            (0, 0, 348),
            (0, 3434, 0),
            (0, 0, 0),
            (5, 3, 8, 1, 3, 5)]

        # 对每个输入尺寸进行测试
        for input_size in input_sizes:
            # 创建一个指定尺寸的 tensor
            a = make_tensor(input_size, dtype=dtype, device=device, low=-9, high=9)

            # 尝试全维度归约
            dim_settings = [None]

            # 尝试所有可能的 1-D 归约
            dim_settings += list(range(-a.dim(), a.dim()))

            # 定义一个辅助函数，用于处理负数维度索引
            def wrap_dim(dim, ndims):
                assert (dim < ndims) and (dim >= -ndims)
                if dim >= 0:
                    return dim
                else:
                    return dim + ndims

            # 尝试所有可能的 2-D 归约
            dim_settings += [
                (d0, d1) for d0, d1 in itertools.combinations(range(-a.dim(), a.dim()), 2)
                if wrap_dim(d0, a.dim()) != wrap_dim(d1, a.dim())]

            # 遍历所有的维度设置
            for dim in dim_settings:
                # 对每种 keepdim 标志进行测试
                for keepdim in [True, False]:
                    # 计算 p=2 时的范数和 p='fro' 时的范数
                    a_norm_2 = torch.norm(a, p=2, dim=dim, keepdim=keepdim)
                    a_norm_fro = torch.norm(a, p='fro', dim=dim, keepdim=keepdim)
                    # 断言两者相等
                    self.assertEqual(a_norm_fro, a_norm_2)

    # 使用条件跳过装饰器，不适合 TorchDynamo 的测试
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    # 使用条件跳过装饰器，如果没有 Magma，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 使用条件跳过装饰器，如果没有 Lapack，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 装饰器，用于测试函数，如果没有 CUDA 和 Magma 库，则跳过测试
    @skipCUDAIfNoMagma
    # 定义一个测试函数，测试 nuclear norm 的异常情况
    def test_nuclear_norm_exceptions_old(self, device):
        # 对空列表、单元素列表、两个元素列表分别进行测试
        for lst in [], [1], [1, 2]:
            # 创建一个张量 x，根据给定的设备，数据类型为双精度
            x = torch.tensor(lst, dtype=torch.double, device=device)
            # 对于空轴和单个轴，调用 torch.norm 函数期望引发 RuntimeError 异常
            for axes in (), (0,):
                self.assertRaises(RuntimeError, torch.norm, x, "nuc", axes)
            # 调用 torch.norm 函数期望引发 RuntimeError 异常，指定轴为 (0, 1)
            self.assertRaises(RuntimeError, torch.norm, x, "nuc", (0, 1))
    
        # 创建一个二维张量 x，数据类型为双精度，指定设备
        x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.double, device=device)
        # 调用 torch.norm 函数期望引发 RuntimeError 异常，指定轴为 (0, 0)，并匹配指定正则表达式 "must be different"
        self.assertRaisesRegex(RuntimeError, "must be different", torch.norm, x, "nuc", (0, 0))
        # 调用 torch.norm 函数期望引发 IndexError 异常，指定轴为 (0, 2)，并匹配指定正则表达式 "Dimension out of range"
        self.assertRaisesRegex(IndexError, "Dimension out of range", torch.norm, x, "nuc", (0, 2))
    
    # 装饰器，用于测试函数，如果没有 CUDA 和 Cusolver 库，则跳过测试
    @skipCUDAIfNoCusolver
    # 装饰器，用于测试函数，如果没有 LAPACK 库，则跳过测试
    @skipCPUIfNoLapack
    # 指定测试数据类型为双精度和复数双精度
    @dtypes(torch.double, torch.cdouble)
    # 定义一个测试方法，用于测试 SVD 低秩分解的功能
    def test_svd_lowrank(self, device, dtype):
        # 导入需要的函数和类
        from torch.testing._internal.common_utils import random_lowrank_matrix, random_sparse_matrix

        # 定义一个子测试函数，运行 SVD 低秩分解的实际测试
        def run_subtest(actual_rank, matrix_size, batches, device, svd_lowrank, **options):
            # 从 options 中取出密度参数，默认为 1
            density = options.pop('density', 1)
            # 根据 matrix_size 的类型确定行数和列数
            if isinstance(matrix_size, int):
                rows = columns = matrix_size
            else:
                rows, columns = matrix_size
            # 根据密度选择生成稠密矩阵或稀疏矩阵
            if density == 1:
                # 生成低秩随机矩阵，或者使用已有的输入
                a_input = random_lowrank_matrix(actual_rank, rows, columns, *batches, device=device, dtype=dtype)
                a = a_input
            else:
                # 对于稀疏输入，确保 batches 为空
                assert batches == ()
                # 生成稀疏矩阵，并转换为稠密形式
                a_input = random_sparse_matrix(rows, columns, density, device=device, dtype=dtype)
                a = a_input.to_dense()

            # 确定要计算的奇异值的数量
            q = min(*size)
            # 进行 SVD 低秩分解
            u, s, v = svd_lowrank(a_input, q=q, **options)

            # 检查 u, s, v 是否满足 SVD 分解条件
            u, s, v = u[..., :q], s[..., :q], v[..., :q]
            A = (u * s.unsqueeze(-2)).matmul(v.mH)
            self.assertEqual(A, a, rtol=1e-7, atol=2e-7)

            # 检查 svd_lowrank 是否产生与 torch.linalg.svdvals 相同的奇异值
            U, S, Vh = torch.linalg.svd(a, full_matrices=False)
            V = Vh.mH
            self.assertEqual(s, S)

            if density == 1:
                # 对于稠密输入，检查 u 和 U、v 和 V 是否张成相同的子空间
                #
                # 检查 (u, U) 和 (v, V) 是否分别张成相同的子空间
                u, v = u[..., :actual_rank], v[..., :actual_rank]
                U, V = U[..., :actual_rank], V[..., :actual_rank]
                expected_ones = u.mH.matmul(U).det().abs()
                self.assertEqual(expected_ones, torch.ones_like(expected_ones))
                self.assertEqual(v.mH.matmul(V).det().abs(), torch.ones_like(expected_ones))

        # 定义所有可能的批次组合
        all_batches = [(), (1,), (3,), (2, 3)]
        # 遍历不同的测试参数组合进行测试
        for actual_rank, size, all_batches in [  # noqa: B020
                (2, (17, 4), all_batches),
                (4, (17, 4), all_batches),
                (4, (17, 17), all_batches),
                (10, (100, 40), all_batches),
                (7, (1000, 1000), [()]),
        ]:
            # 对于稠密输入，测试所有可能的批次组合
            for batches in all_batches:
                run_subtest(actual_rank, size, batches, device, torch.svd_lowrank)
                # 如果 size 不是 size 的翻转，则再次运行测试
                if size != size[::-1]:
                    run_subtest(actual_rank, size[::-1], batches, device, torch.svd_lowrank)

        # 对于稀疏输入，测试不同的矩阵大小和密度
        for size in [(17, 4), (4, 17), (17, 17), (100, 40), (40, 100), (1000, 1000)]:
            for density in [0.005, 0.1]:
                run_subtest(None, size, (), device, torch.svd_lowrank, density=density)

        # 支持 JIT 编译的测试
        jitted = torch.jit.script(torch.svd_lowrank)
        actual_rank, size, batches = 2, (17, 4), ()
        run_subtest(actual_rank, size, batches, device, jitted)

    # 如果没有 Magma 和 Cusolver 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagmaAndNoCusolver
    # 如果没有 Lapack 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 使用装饰器设置 torch.float 和 torch.cfloat 的精度覆盖
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 2e-4})
    # 使用装饰器将线性代数后端设置为默认值
    @setLinalgBackendsToDefaultFinally
    # 使用装饰器设置测试的数据类型为浮点数和复数类型的所有可能组合
    @dtypes(*floating_and_complex_types())
    # 使用装饰器标记为序列化测试
    @serialTest()
    # 定义测试函数 test_svd，接受设备和数据类型作为参数
    def test_svd(self, device, dtype):
        # 测试 linalg.svd, svd, linalg.svdvals
        # 部分应用函数 make_tensor，使用指定的数据类型和设备创建张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        # 初始化后端列表，默认包含 "default"
        backends = ["default"]

        # 如果设备是 CUDA，检查是否支持 MAGMA 后端，支持则加入列表
        if torch.device(device).type == 'cuda':
            if torch.cuda.has_magma:
                backends.append("magma")
            # 检查是否有 cusolver 或 hipsolver 支持，支持则加入列表
            if has_cusolver() or has_hipsolver():
                backends.append("cusolver")

        # 定义多个测试用例参数
        ns = (12, 4, 2, 0)
        batches = ((), (0,), (1,), (2,), (2, 1), (0, 2))
        drivers = (None, 'gesvd', 'gesvdj', 'gesvda')

        # 遍历后端列表
        for backend in backends:
            # 设置 CUDA 的首选线性代数库为当前后端
            torch.backends.cuda.preferred_linalg_library(backend)

            # 使用 product 组合测试参数
            for batch, m, n, driver in product(batches, ns, ns, drivers):
                # 如果不是 cusolver 后端且驱动程序不为 None，则跳过当前测试用例
                if not (backend == 'cusolver' or driver is None):
                    # 只测试以下情况，否则跳过：
                    # - backend == 'cusolver'（驱动程序可以是任何值）
                    # - backend != 'cusolver' 且 driver 应为 None
                    continue

                # 构建张量的形状
                shape = batch + (m, n)
                k = min(m, n)
                # 创建测试用的张量 A
                A = make_arg(shape)
                
                # 执行 linalg.svd 函数，获取 U, S, Vh 并断言 U @ S.diag() @ Vh == A
                U, S, Vh = torch.linalg.svd(A, full_matrices=False, driver=driver)
                self.assertEqual((U @ S.to(A.dtype).diag_embed()) @ Vh, A)

                # 执行 full_matrices=True 的 linalg.svd 函数，验证 S 的一致性
                U_f, S_f, Vh_f = torch.linalg.svd(A, full_matrices=True, driver=driver)
                self.assertEqual(S_f, S)
                # 验证 U_f @ S_f.diag() @ Vh_f[..., :k, :] == A 的一致性
                self.assertEqual((U_f[..., :k] @ S_f.to(A.dtype).diag_embed()) @ Vh_f[..., :k, :], A)

                # 执行 linalg.svdvals 函数，验证 S_s == S
                S_s = torch.linalg.svdvals(A, driver=driver)
                self.assertEqual(S_s, S)

                # 执行 torch.svd 函数，获取 U, S, V 并断言 U @ S.diag() @ V.mH == A
                U, S, V = torch.svd(A, some=True)
                self.assertEqual((U @ S.to(A.dtype).diag_embed()) @ V.mH, A)

                # 执行 torch.svd 函数，获取 full_matrices=False 的 U, S, V 并验证 S 的一致性
                U_f, S_f, V_f = torch.svd(A, some=False)
                self.assertEqual(S_f, S)
                # 验证 U_f[..., :k] @ S_f.diag() @ V_f[..., :k].mH == A 的一致性
                self.assertEqual((U_f[..., :k] @ S_f.to(A.dtype).diag_embed()) @ V_f[..., :k].mH, A)

                # 执行 torch.svd 函数，仅计算奇异值并验证 S_s == S
                S_s = torch.svd(A, compute_uv=False).S
                self.assertEqual(S_s, S)

    # 如果没有 MAGMA 和 cusolver 支持，则跳过 CUDA 测试
    @skipCUDAIfNoMagmaAndNoCusolver
    # 如果没有 LAPACK 支持，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 设置数据类型为 torch.complex128 的测试用例
    @dtypes(torch.complex128)
    # 测试奇异值分解函数 `torch.linalg.svd` 的稳定性和错误处理能力
    def test_invariance_error_spectral_decompositions(self, device, dtype):
        # 创建一个部分应用了设备、数据类型和梯度要求的张量生成函数
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        
        # 生成一个随机的 m × n 的张量 A
        A = make_arg((3, 3))
        
        # 在运行中捕获 RuntimeError 异常，其错误消息包含 "ill-defined"
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            # 对 A 进行奇异值分解，获取其左奇异向量 U、奇异值 S 和右奇异向量的共轭转置 Vh
            U, _, Vh = torch.linalg.svd(A, full_matrices=False)
            # 对 (U + Vh) 的绝对值求和进行反向传播
            (U + Vh).sum().abs().backward()

        # 重新生成一个随机的 m × n 的张量 A
        A = make_arg((3, 3))
        
        # 再次捕获 RuntimeError 异常，错误消息包含 "ill-defined"
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            # 对 A 进行特征值分解，获取其特征向量 V
            V = torch.linalg.eig(A).eigenvectors
            # 对 V 的绝对值求和进行反向传播
            V.sum().abs().backward()

        # 重新生成一个随机的 m × n 的张量 A
        A = make_arg((3, 3))
        
        # 将 A 与其共轭转置相加
        A = A + A.mH
        
        # 第三次捕获 RuntimeError 异常，错误消息包含 "ill-defined"
        with self.assertRaisesRegex(RuntimeError, "ill-defined"):
            # 对 A 进行厄米特矩阵的特征分解，获取其特征向量 Q
            Q = torch.linalg.eigh(A).eigenvectors
            # 对 Q 的绝对值求和进行反向传播
            Q.sum().abs().backward()

    # 跳过不支持 Cusolver 的 CUDA 测试，用 MAGMA 后端不适用于此情况
    # 对浮点数和复数类型的测试，使用指定的精度覆盖
    @skipCUDAIfNoCusolver
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_svd_memory_allocation(self, device, dtype):
        # 测试 https://github.com/pytorch/pytorch/issues/61949 的问题
        # 问题是错误大小的张量被分配然后被缩小
        m = 3
        n = 2**20
        
        # 生成一个 m × n 的张量 a
        a = make_tensor((m, n), dtype=dtype, device=device)
        
        # 使用 torch.linalg.svdvals 计算 a 的奇异值 S
        S = torch.linalg.svdvals(a)
        
        # 使用 torch.linalg.svd 计算 a 的奇异值分解结果
        result = torch.linalg.svd(a, full_matrices=False)
        
        # 断言两种方法得到的奇异值 S 相等
        self.assertEqual(result.S, S)

    # Cholesky 分解求解测试辅助函数
    def cholesky_solve_test_helper(self, A_dims, b_dims, upper, device, dtype):
        # 导入随机生成 Hermitian 正定矩阵的函数
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix
        
        # 生成一个随机的 b 张量
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        
        # 生成一个随机的 Hermitian 正定矩阵 A
        A = random_hermitian_pd_matrix(*A_dims, dtype=dtype, device=device)
        
        # 对 A 进行 Cholesky 分解，生成下三角矩阵 L
        L = torch.cholesky(A, upper=upper)
        
        # 返回生成的 b 张量、A 矩阵和 L 矩阵
        return b, A, L

    # 跳过不支持 Magma 的 CUDA 测试，跳过不支持 LAPACK 的 CPU 测试
    # 对浮点数和复数类型的测试，使用指定的精度覆盖
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_cholesky_solve(self, device, dtype):
        # 遍历每对 (k, n) 和每种上三角矩阵的情况
        for (k, n), upper in itertools.product(zip([2, 3, 5], [3, 5, 7]), [True, False]):
            # 使用辅助函数生成 Cholesky 分解测试的输入
            b, A, L = self.cholesky_solve_test_helper((n,), (n, k), upper, device, dtype)
            
            # 使用 torch.cholesky_solve 求解线性系统
            x = torch.cholesky_solve(b, L, upper=upper)
            
            # 断言解 x 与预期解 np.matmul(A.cpu(), b.cpu()) 相等
            self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))
    # 定义一个测试方法，用于批量测试 Cholesky 分解求解的情况
    def test_cholesky_solve_batched(self, device, dtype):
        # 辅助函数，用于执行批量 Cholesky 分解求解的测试
        def cholesky_solve_batch_helper(A_dims, b_dims, upper):
            # 调用辅助方法执行 Cholesky 分解求解测试，并返回测试结果
            b, A, L = self.cholesky_solve_test_helper(A_dims, b_dims, upper, device, dtype)
            x_exp_list = []
            # 遍历每个批次中的数据，使用 Cholesky 分解求解方程组
            for i in range(b_dims[0]):
                x_exp_list.append(torch.cholesky_solve(b[i], L[i], upper=upper))
            # 将所有批次的求解结果堆叠成一个张量
            x_exp = torch.stack(x_exp_list)  # Stacked output
            # 使用 Cholesky 分解求解整个批次的方程组
            x_act = torch.cholesky_solve(b, L, upper=upper)  # Actual output
            # 断言预期的求解结果与实际求解结果相等
            self.assertEqual(x_act, x_exp)  # Equality check
            # 计算矩阵乘积 Ax，并断言其与 b 相等，验证求解正确性
            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)  # Correctness check

        # 使用 itertools.product 生成所有的 upper 和 batchsize 组合，并执行测试
        for upper, batchsize in itertools.product([True, False], [1, 3, 4]):
            cholesky_solve_batch_helper((5, batchsize), (batchsize, 5, 10), upper)

    # 装饰器，标记为慢速测试
    @slowTest
    # 如果没有 MAGMA 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 装饰器，指定测试数据类型为浮点数和复数类型的所有组合
    @dtypes(*floating_and_complex_types())
    # 装饰器，覆盖特定数据类型的精度要求
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 测试方法，用于测试处理多个批次的 Cholesky 分解求解情况
    def test_cholesky_solve_batched_many_batches(self, device, dtype):
        # 遍历每个 A_dims 和 b_dims 的组合，执行多批次的 Cholesky 分解求解测试
        for A_dims, b_dims in zip([(5, 256, 256), (5,)], [(5, 10), (512, 512, 5, 10)]):
            # 遍历每个 upper 的值，执行 Cholesky 分解求解测试，并断言结果
            for upper in [True, False]:
                # 调用测试辅助方法执行 Cholesky 分解求解测试，并获取结果
                b, A, L = self.cholesky_solve_test_helper(A_dims, b_dims, upper, device, dtype)
                # 使用 Cholesky 分解求解方程组
                x = torch.cholesky_solve(b, L, upper)
                # 计算矩阵乘积 Ax
                Ax = torch.matmul(A, x)
                # 断言矩阵乘积 Ax 与 b 的展开张量相等，验证求解正确性
                self.assertEqual(Ax, b.expand_as(Ax))

    # 如果没有 MAGMA 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 装饰器，指定测试数据类型为浮点数和复数类型的所有组合
    @dtypes(*floating_and_complex_types())
    # 装饰器，覆盖特定数据类型的精度要求
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 定义一个测试方法，用于批处理和广播的 Cholesky 分解求解测试
    def test_cholesky_solve_batched_broadcasting(self, device, dtype):
        # 导入 solve 函数用于 NumPy 数组的解算，以及随机生成 Hermite 正定矩阵的工具函数
        from numpy.linalg import solve
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # 定义一个内部函数，运行 Cholesky 分解求解测试
        def run_test(A_dims, b_dims, upper):
            # 获取 A 的矩阵大小和批处理维度
            A_matrix_size = A_dims[-1]
            A_batch_dims = A_dims[:-2]
            # 生成随机的 Hermite 正定矩阵 A 和随机向量 b
            A = random_hermitian_pd_matrix(A_matrix_size, *A_batch_dims,
                                           dtype=dtype, device='cpu')
            b = torch.randn(*b_dims, dtype=dtype, device='cpu')
            # 计算 NumPy solve 函数的期望结果，并将其转换为 PyTorch 张量
            x_exp = torch.tensor(solve(A.numpy(), b.numpy()), dtype=dtype, device=device)
            # 将 A 和 b 转换为指定设备和数据类型的 PyTorch 张量
            A, b = A.to(dtype=dtype, device=device), b.to(dtype=dtype, device=device)
            # 对 A 进行 Cholesky 分解，返回下三角矩阵 L
            L = torch.linalg.cholesky(A, upper=upper)
            # 使用 Cholesky 分解后的下三角矩阵 L 解算线性系统 Ax=b，返回解 x
            x = torch.cholesky_solve(b, L, upper=upper)
            # 断言解 x 等于期望结果 x_exp
            self.assertEqual(x, x_exp)
            # 使用已有的输出张量 x 执行 Cholesky 分解求解，用于验证 GitHub 上的问题
            x = torch.cholesky_solve(b, L, upper=upper, out=x)
            self.assertEqual(x, x_exp)

        # 对于每种上三角选项，执行测试
        for upper in [True, False]:
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), upper)  # 没有广播
            run_test((2, 1, 3, 4, 4), (4, 6), upper)  # 广播 b
            run_test((4, 4), (2, 1, 3, 4, 2), upper)  # 广播 A
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), upper)  # 广播 A 和 b

    # 使用 skipCUDAIfNoMagma 装饰器，跳过没有 Magma 支持的 CUDA 测试
    # 使用 skipCPUIfNoLapack 装饰器，跳过没有 LAPACK 支持的 CPU 测试
    # 使用 dtypes 装饰器，指定测试包括浮点数和复数类型
    def test_cholesky_solve_out_errors_and_warnings(self, device, dtype):
        # 确保可以安全地转换数据类型
        a = torch.eye(2, dtype=dtype, device=device)
        b = torch.randn(2, 1, dtype=dtype, device=device)
        # 准备一个输出张量 out，数据类型不匹配，预期引发 RuntimeError
        out = torch.empty(0, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.cholesky_solve(b, a, out=out)

        # 确保设备类型匹配
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.cholesky_solve(b, a, out=out)

        # 如果传递了形状错误的输出张量，将会产生警告
        with warnings.catch_warnings(record=True) as w:
            out = torch.empty(1, dtype=dtype, device=device)
            # 触发警告
            torch.cholesky_solve(b, a, out=out)
            # 检查是否发出了警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))
    # 定义一个测试函数，用于测试 Cholesky 分解求解的反向传播
    def test_cholesky_solve_backward(self, device, dtype):
        # 定义 b 的维度为 (5, 2)
        b_dims = (5, 2)
        # 定义 L 的维度为 (5, 5)
        L_dims = (5, 5)

        # 循环测试两种情况：test_L_grad 为 False 和 True
        for test_L_grad in (False, True):
            # 随机生成 b，并指定其数据类型、设备，并要求计算梯度
            b = torch.randn(*b_dims, dtype=dtype, device=device, requires_grad=True)
            # 随机生成 L，并指定其数据类型、设备，根据 test_L_grad 决定是否要求计算梯度
            L = torch.randn(*L_dims, dtype=dtype, device=device, requires_grad=test_L_grad)
            # 如果 test_L_grad 为 True，则进行梯度检查，使用 torch.cholesky_solve 求解
            if test_L_grad:
                torch.autograd.gradcheck(lambda b, L: torch.cholesky_solve(b, torch.tril(L), upper=False), (b, L))
            # 如果 test_L_grad 为 False，则进行梯度检查，使用 torch.cholesky_solve 求解
            else:
                torch.autograd.gradcheck(lambda b: torch.cholesky_solve(b, L, upper=False), (b,))
    
    # 根据条件跳过不支持 CUDA 的测试，同时也要求 LAPACK 库支持
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    # 根据一组浮点数和复数数据类型执行测试
    @dtypes(*floating_and_complex_types())
    # 设置精度覆盖，不同数据类型有不同的精度要求
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 根据条件跳过不支持 CUDA 的测试，同时也要求 LAPACK 库支持
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    # 根据一组浮点数和复数数据类型执行测试
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，测试 torch.linalg.inv_ex 在指定设备上的行为
    def test_inv_ex_info_device(self, device, dtype):
        # 创建一个单位矩阵 A，数据类型为 dtype，存储设备为 device
        A = torch.eye(3, 3, dtype=dtype, device=device)
        # 使用 torch.linalg.inv_ex 计算 A 的逆矩阵信息，将结果存储在 info 中
        info = torch.linalg.inv_ex(A).info
        # 断言 info 的设备与 A 的设备相同
        self.assertTrue(info.device == A.device)

    # 根据条件跳过不支持 CUDA 的测试，同时也要求 LAPACK 库支持
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    # 根据一组浮点数和复数数据类型执行测试
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，测试当输入矩阵是奇异矩阵时的行为
    def test_inv_ex_singular(self, device, dtype):
        # 创建一个单位矩阵 A，数据类型为 dtype，存储设备为 device
        A = torch.eye(3, 3, dtype=dtype, device=device)
        # 将 A 的最后一个对角线元素设为 0，使得 A 变成奇异矩阵
        A[-1, -1] = 0  # 现在 A 是奇异的
        # 使用 torch.linalg.inv_ex 计算 A 的逆矩阵信息，将结果存储在 info 中
        info = torch.linalg.inv_ex(A).info
        # 断言 info 的值为 3，表示奇异性信息
        self.assertEqual(info, 3)
        # 使用断言确保当矩阵 A 是奇异的时候，抛出特定的异常信息
        with self.assertRaisesRegex(torch.linalg.LinAlgError,
                                    r'diagonal element 3 is zero, the inversion could not be completed'):
            torch.linalg.inv_ex(A, check_errors=True)

        # 创建一个单位矩阵 A，数据类型为 dtype，存储设备为 device
        A = torch.eye(3, 3, dtype=dtype, device=device)
        # 将 A 变形为 (1, 3, 3) 的形状
        A = A.reshape((1, 3, 3))
        # 将 A 复制为一个大小为 (5, 3, 3) 的批次
        A = A.repeat(5, 1, 1)
        # 将 A[3] 的倒数第二行、倒数第二列的元素设为 0，使得 A[3] 变成奇异矩阵
        A[3, -2, -2] = 0  # 现在 A[3] 是奇异的
        # 使用 torch.linalg.inv_ex 计算 A 的逆矩阵信息，将结果存储在 info 中
        info = torch.linalg.inv_ex(A).info

        # 创建一个预期的信息张量，大小与 A 的批次维度相同，数据类型为 torch.int32，存储设备为 device
        expected_info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=device)
        # 将第四个元素的信息设为 2，表示 A[3] 是奇异的
        expected_info[3] = 2
        # 断言 info 与预期信息张量 expected_info 相等
        self.assertEqual(info, expected_info)
        # 使用断言确保当批次中至少一个矩阵是奇异的时候，抛出特定的异常信息
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 3\): The diagonal element 2 is zero'):
            torch.linalg.inv_ex(A, check_errors=True)

    # 标记为慢速测试
    @slowTest
    # 根据条件跳过不支持 CUDA 的测试，同时也要求 LAPACK 库支持
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    # 根据一组浮点数和复数数据类型执行测试
    @dtypes(*floating_and_complex_types())
    # 设置精度覆盖，不同数据类型有不同的精度要求
    @precisionOverride({torch.float32: 2e-3, torch.complex64: 2e-3,
                        torch.float64: 1e-5, torch.complex128: 1e-5})
    # 定义测试函数，用于测试批量求逆操作，测试运行在指定设备上，使用指定数据类型
    def test_inverse_many_batches(self, device, dtype):
        # 别名函数，生成具有不同奇异值的满秩矩阵
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        # 带设备和数据类型参数的部分函数别名
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        # 辅助函数：测试批量求逆操作
        def test_inverse_many_batches_helper(torch_inverse, b, n):
            # 生成满秩矩阵
            matrices = make_arg(b, n, n)
            # 使用 Torch 提供的求逆函数对矩阵进行求逆
            matrices_inverse = torch_inverse(matrices)

            # 与 NumPy 求逆结果进行比较
            expected = np.linalg.inv(matrices.cpu().numpy())
            # 断言 Torch 求逆结果与 NumPy 求逆结果在给定精度下相等
            self.assertEqual(matrices_inverse, expected, atol=self.precision, rtol=1e-3)

        # 遍历两种 Torch 求逆函数进行测试
        for torch_inverse in [torch.inverse, torch.linalg.inv]:
            # 分别使用不同的参数进行测试
            test_inverse_many_batches_helper(torch_inverse, 5, 256)
            test_inverse_many_batches_helper(torch_inverse, 3, 512)

    # 跳过没有 Magma 和 Cusolver 的 CUDA 环境，以及没有 Lapack 的 CPU 环境
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyNativeDeviceTypes   # 只在原生设备类型上运行，XLA 不会引发异常
    @dtypes(*floating_and_complex_types())
    # 测试逆运算的错误情况
    def test_inverse_errors(self, device, dtype):
        # inverse 函数期望批量输入的方阵
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.inverse(torch.randn(2, 3, 4, 3))

        # 如果输入矩阵不可逆，则引发 RuntimeError，指明第一个不可逆批次
        def run_test_singular_input(batch_dim, n):
            # 生成单位矩阵，并复制成批次矩阵
            x = torch.eye(3, 3, dtype=dtype, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
            # 使第 n 批次的最后一个对角元素为零
            x[n, -1, -1] = 0
            # 断言调用 inverse 函数引发特定的 LinAlgError，指明不可逆的原因
            with self.assertRaisesRegex(torch.linalg.LinAlgError, rf'\(Batch element {n}\): The diagonal element 3 is zero'):
                torch.inverse(x)

        # 使用不同参数运行测试不可逆输入的情况
        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

    # 在特定条件下跳过测试：在 IS_FBCODE 或 IS_SANDCASTLE 为真时跳过，因为在 Meta 基础设施上 GPU (P100, V100) 上的 float64 类型测试失败
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Test fails for float64 on GPU (P100, V100) on Meta infra")
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    @onlyNativeDeviceTypes   # 只在原生设备类型上运行，XLA 不会引发异常
    @dtypes(*floating_and_complex_types())
    # 测试大规模矩阵的逆运算错误
    def test_inverse_errors_large(self, device, dtype):
        # 测试对奇异矩阵的批量逆运算，确保报告错误而不崩溃 (gh-51930)
        x = torch.empty((8, 10, 616, 616), dtype=dtype, device=device)
        # 填充成单位矩阵，其中第 0 批次的第 11 个对角元素为零
        x[:] = torch.eye(616, dtype=dtype, device=device)
        x[..., 10, 10] = 0
        # 断言调用 inverse 函数引发特定的 LinAlgError，指明不可逆的原因
        with self.assertRaisesRegex(torch.linalg.LinAlgError, r'\(Batch element 0\): The diagonal element 11 is zero'):
            torch.inverse(x)

    # 设置精度覆盖参数，针对不同的数据类型设置不同的精度
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3, torch.float64: 1e-7, torch.complex128: 1e-7})
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    # 跳过没有 Magma 的 CUDA 环境，以及没有 Lapack 的 CPU 环境
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    # 测试 pinv 函数在错误和警告方面的表现，需要指定设备和数据类型

    # 创建一个形状为 (1,) 的随机张量，pinv 要求至少是 2 维张量，会引发 RuntimeError
    a = torch.randn(1, device=device, dtype=dtype)
    with self.assertRaisesRegex(RuntimeError, "expected a tensor with 2 or more dimensions"):
        torch.linalg.pinv(a)

    # 创建一个形状为 (3, 3) 的随机张量 a 和一个形状为 (7, 7) 的空张量 out
    # 如果传入形状不匹配的输出张量，会触发警告
    a = torch.randn(3, 3, dtype=dtype, device=device)
    out = torch.empty(7, 7, dtype=dtype, device=device)
    with warnings.catch_warnings(record=True) as w:
        # 触发警告
        torch.linalg.pinv(a, out=out)
        # 检查是否触发了警告
        self.assertEqual(len(w), 1)
        self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

    # 创建一个与 a 相同形状的空张量 out，但是将其类型转换为 torch.int
    # 如果输出张量的类型无法安全转换，则会引发 RuntimeError
    out = torch.empty_like(a).to(torch.int)
    with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
        torch.linalg.pinv(a, out=out)

    # 如果 CUDA 可用，则测试输出张量与输入张量的设备是否匹配
    # 创建一个与 a 相同形状的空张量 out，并将其放在错误的设备上
    # 如果输出张量和输入张量不在同一设备上，则会引发 RuntimeError
    if torch.cuda.is_available():
        wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
        out = torch.empty_like(a).to(wrong_device)
        with self.assertRaisesRegex(RuntimeError, "Expected result and input tensors to be on the same device"):
            torch.linalg.pinv(a, out=out)

        # 创建一个标量 rcond，放在错误的设备上
        # 如果 rcond 张量和输入张量不在同一设备上，则会引发 RuntimeError
        wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
        rcond = torch.full((), 1e-2, device=wrong_device)
        with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
            torch.linalg.pinv(a, rcond=rcond)

    # 创建一个复数类型的标量 rcond
    # pinv 函数不支持复数类型的 rcond 张量，会引发 RuntimeError
    rcond = torch.full((), 1j, device=device)
    with self.assertRaisesRegex(RuntimeError, "rcond tensor of complex type is not supported"):
        torch.linalg.pinv(a, rcond=rcond)

    # 创建一个复数类型的标量 atol
    # pinv 函数不支持复数类型的 atol 张量，会引发 RuntimeError
    atol = torch.full((), 1j, device=device)
    with self.assertRaisesRegex(RuntimeError, "atol tensor of complex type is not supported"):
        torch.linalg.pinv(a, atol=atol)

    # 创建一个复数类型的标量 rtol
    # pinv 函数不支持复数类型的 rtol 张量，会引发 RuntimeError
    rtol = torch.full((), 1j, device=device)
    with self.assertRaisesRegex(RuntimeError, "rtol tensor of complex type is not supported"):
        torch.linalg.pinv(a, rtol=rtol)
    # 测试在给定设备和数据类型下，torch.linalg.inv 函数的错误和警告情况
    def test_inv_errors_and_warnings(self, device, dtype):
        # inv 函数期望输入是批次的方阵
        a = torch.randn(2, 3, 4, 3, dtype=dtype, device=device)
        # 使用 assertRaisesRegex 检查是否抛出 RuntimeError，并验证错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.linalg.inv(a)

        # inv 函数要求输入至少是二维张量
        a = torch.randn(2, device=device, dtype=dtype)
        # 使用 assertRaisesRegex 检查是否抛出 RuntimeError，并验证错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.linalg.inv(a)

        # 如果输入不可逆，将引发 RuntimeError，其中会提到第一个不可逆的批次
        def run_test_singular_input(batch_dim, n):
            # 创建单位矩阵张量，并重复以形成批次
            a = torch.eye(3, 3, dtype=dtype, device=device).reshape((1, 3, 3)).repeat(batch_dim, 1, 1)
            # 修改指定批次的某个元素，使其不可逆
            a[n, -1, -1] = 0
            # 使用 assertRaisesRegex 检查是否抛出 LinAlgError，并验证错误消息中包含特定字符串
            with self.assertRaisesRegex(torch.linalg.LinAlgError, rf"\(Batch element {n}\): The diagonal element 3 is zero"):
                torch.linalg.inv(a)

        # 遍历测试参数，运行不可逆输入测试函数
        for params in [(1, 0), (2, 0), (2, 1), (4, 0), (4, 2), (10, 2)]:
            run_test_singular_input(*params)

        # 数据类型应该匹配
        a = torch.eye(2, dtype=dtype, device=device)
        out = torch.empty(0, dtype=torch.int, device=device)
        # 使用 assertRaisesRegex 检查是否抛出 RuntimeError，并验证错误消息中包含特定字符串
        with self.assertRaisesRegex(RuntimeError, "but got int instead"):
            torch.linalg.inv(a, out=out)

        # 设备应该匹配
        if torch.cuda.is_available():
            # 根据当前设备类型选择一个错误的设备
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            # 使用 assertRaisesRegex 检查是否抛出 RuntimeError，并验证错误消息中包含特定字符串
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.inv(a, out=out)

        # 如果传递了形状不正确的输出张量，则会发出警告
        with warnings.catch_warnings(record=True) as w:
            a = torch.eye(2, dtype=dtype, device=device)
            out = torch.empty(1, dtype=dtype, device=device)
            # 触发警告
            torch.linalg.inv(a, out=out)
            # 检查是否有警告发生
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 如果传递了批次列主格式的输出张量，但形状不正确，则会发出警告
        with warnings.catch_warnings(record=True) as w:
            a = torch.eye(2, dtype=dtype, device=device)
            out = torch.empty(3, 3, dtype=dtype, device=device)
            # 将输出张量转换为列主格式
            out = out.t().clone(memory_format=torch.contiguous_format)
            out = out.t()
            self.assertTrue(out.t().is_contiguous())
            # 触发警告
            torch.linalg.inv(a, out=out)
            # 检查是否有警告发生
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))
    # 定义一个辅助函数，用于生成具有不同奇异值的全秩矩阵
    def solve_test_helper(self, A_dims, b_dims, device, dtype):
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_A = partial(make_fullrank, device=device, dtype=dtype)

        # 生成一个随机的张量 b，其维度为 b_dims，数据类型为 dtype，放置在指定设备上
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        # 生成一个全秩矩阵 A，其维度为 A_dims，放置在指定设备上
        A = make_A(*A_dims)
        return b, A

    # 测试函数修饰器：如果没有安装 MAGMA 库，则跳过测试
    @skipCUDAIfNoMagma
    # 测试函数修饰器：如果没有 LAPACK 库，则跳过测试
    @skipCPUIfNoLapack
    # 参数化测试：测试包含所有浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 精度覆盖修饰器：指定浮点数和复数类型的精度
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3})
    def test_solve(self, device, dtype):
        # 定义内部函数 run_test，用于运行单个解决方案测试
        def run_test(n, batch, rhs):
            # 计算 A 的维度，将 batch 维度扩展到 A_dims
            A_dims = (*batch, n, n)
            # 计算 b 的维度，将 batch 维度扩展到 b_dims，并添加 rhs 维度
            b_dims = (*batch, n, *rhs)
            # 使用 solve_test_helper 函数生成测试数据 b 和 A
            b, A = self.solve_test_helper(A_dims, b_dims, device, dtype)

            # 正确性测试
            # 使用 torch.linalg.solve 解 A x = b 的方程
            x = torch.linalg.solve(A, b)
            if rhs == ():
                # 如果 rhs 是空元组，进行 Ax 和 b 的乘积，使用 NumPy 进行比较
                Ax = np.matmul(A.cpu(), x.unsqueeze(-1).cpu())
                Ax.squeeze_(-1)
            else:
                # 否则直接计算 Ax
                Ax = np.matmul(A.cpu(), x.cpu())
            # 断言 b 与 Ax 的扩展相等
            self.assertEqual(b.expand_as(Ax), Ax)

            # 使用 NumPy 对比结果
            expected = np.linalg.solve(A.cpu().numpy(), b.expand_as(x).cpu().numpy())
            self.assertEqual(x, expected)

        # 定义测试参数的组合
        batches = [(), (0, ), (3, ), (2, 3)]
        ns = [0, 5, 32]
        nrhs = [(), (1, ), (5, )]
        # 对参数进行组合，并执行测试
        for n, batch, rhs in itertools.product(ns, batches, nrhs):
            run_test(n, batch, rhs)

    # 测试函数修饰器：如果没有安装 MAGMA 和 CUSOLVER 库，则跳过测试
    @skipCUDAIfNoMagmaAndNoCusolver
    # 测试函数修饰器：如果没有 LAPACK 库，则跳过测试
    @skipCPUIfNoLapack
    # 参数化测试：测试包含所有浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    def test_solve_batched_broadcasting(self, device, dtype):
        from numpy.linalg import solve

        # 定义内部函数 run_test，用于运行批量广播解决方案测试
        def run_test(A_dims, B_dims):
            # 提取 A 矩阵的尺寸
            A_matrix_size = A_dims[-1]
            # 提取 A 批次的维度
            A_batch_dims = A_dims[:-2]
            # 使用 solve_test_helper 函数生成测试数据 B 和 A
            B, A = self.solve_test_helper(A_batch_dims + (A_matrix_size, A_matrix_size), B_dims, device, dtype)
            # 使用 torch.linalg.solve 解 A x = B 的方程
            actual = torch.linalg.solve(A, B)
            # 使用 NumPy 对比结果
            expected = solve(A.cpu().numpy(), B.cpu().numpy())
            self.assertEqual(actual, expected)

        # 对不同维度的 A 和 B 进行批量广播测试
        run_test((5, 5), (2, 0, 5, 3))  # broadcasting with 0 batch dim
        run_test((2, 0, 5, 5), (5, 3))  # broadcasting with 0 batch dim
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting B
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & B

    # 测试函数修饰器：如果没有安装 MAGMA 库，则跳过测试
    @skipCUDAIfNoMagma
    # 测试函数修饰器：如果没有 LAPACK 库，则跳过测试
    @skipCPUIfNoLapack
    # 参数化测试：指定浮点数和复数类型
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 精度覆盖修饰器：指定浮点数和复数类型的精度
    @precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})
    # 定义测试函数 test_tensorsolve，测试 torch.linalg.tensorsolve 函数在不同条件下的行为
    def test_tensorsolve(self, device, dtype):
        # 定义内部函数 run_test，用于执行单个测试
        def run_test(a_shape, dims):
            # 生成指定形状和数据类型的随机张量 a 和 b
            a = torch.randn(a_shape, dtype=dtype, device=device)
            b = torch.randn(a_shape[:2], dtype=dtype, device=device)
            # 使用 torch.linalg.tensorsolve 求解张量方程 a x = b，并记录结果
            result = torch.linalg.tensorsolve(a, b, dims=dims)
            # 使用 NumPy 的 tensorsolve 进行相同的操作，以便验证结果
            expected = np.linalg.tensorsolve(a.cpu().numpy(), b.cpu().numpy(), axes=dims)
            # 断言 torch 和 NumPy 的结果应当一致
            self.assertEqual(result, expected)

            # 检查带有 out 参数的变体
            out = torch.empty_like(result)
            ans = torch.linalg.tensorsolve(a, b, dims=dims, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, result)

        # 待测试的张量形状和维度参数
        a_shapes = [(2, 3, 6), (3, 4, 4, 3)]
        dims = [None, (0, 2)]
        # 使用 itertools.product 对所有形状和维度组合进行迭代测试
        for a_shape, d in itertools.product(a_shapes, dims):
            run_test(a_shape, d)

    # 标记装饰器，如果没有 Magma 库支持则跳过测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 定义测试函数 test_tensorsolve_empty，测试处理空输入的情况
    def test_tensorsolve_empty(self, device, dtype):
        # 检查空输入情况，NumPy 无法处理此类情况
        a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
        b = torch.empty(a.shape[:2], dtype=dtype, device=device)
        # 使用 torch.linalg.tensorsolve 解决张量方程 a x = b，并验证结果
        x = torch.linalg.tensorsolve(a, b)
        self.assertEqual(torch.tensordot(a, x, dims=len(x.shape)), b)

    # 标记装饰器，如果没有 Magma 库支持则跳过测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32)
    # 定义测试函数 test_tensorsolve_errors_and_warnings，测试处理错误和警告的情况
    def test_tensorsolve_errors_and_warnings(self, device, dtype):
        # tensorsolve 函数要求输入可以重塑为方阵
        a = torch.eye(2 * 3 * 4, dtype=dtype, device=device).reshape((2 * 3, 4, 2, 3, 4))
        b = torch.randn(8, 4, dtype=dtype, device=device)
        # 断言输入是否满足方阵要求
        self.assertTrue(np.prod(a.shape[2:]) != np.prod(b.shape))
        with self.assertRaisesRegex(RuntimeError, r'Expected self to satisfy the requirement'):
            torch.linalg.tensorsolve(a, b)

        # 如果传入具有错误形状的非空输出张量，则会发出警告
        out = torch.empty_like(a)
        b = torch.randn(6, 4, dtype=dtype, device=device)
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.linalg.tensorsolve(a, b, out=out)
            # 检查是否发出了警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # dtype 应当可以安全地进行转换
        out = torch.empty_like(a).to(torch.int)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.tensorsolve(a, b, out=out)

        # 设备应当匹配
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.tensorsolve(a, b, out=out)
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float: 1e-3, torch.cfloat: 1e-3})
    def test_tensorinv(self, device, dtype):
        # 定义内部函数，用于执行测试
        def run_test(a_shape, ind):
            # 生成指定形状和数据类型的随机张量 a，放置在指定设备上
            a = torch.randn(a_shape, dtype=dtype, device=device)
            # 将张量 a 转为 NumPy 数组
            a_numpy = a.cpu().numpy()
            # 使用 torch.linalg.tensorinv 计算张量 a 的逆，结果保存在 result 中
            result = torch.linalg.tensorinv(a, ind=ind)
            # 使用 np.linalg.tensorinv 计算 NumPy 数组 a_numpy 的逆，结果保存在 expected 中
            expected = np.linalg.tensorinv(a_numpy, ind=ind)
            # 断言 torch 和 NumPy 计算得到的逆矩阵结果相等
            self.assertEqual(result, expected)

            # 检查指定 out 参数的情况
            out = torch.empty_like(result)
            # 使用 torch.linalg.tensorinv 计算张量 a 的逆，并将结果保存在预先分配的 out 张量中
            ans = torch.linalg.tensorinv(a, ind=ind, out=out)
            # 断言计算结果与 out 张量内容相等
            self.assertEqual(ans, out)
            # 再次断言计算结果与之前 result 的比较结果相等
            self.assertEqual(ans, result)

        # 与 NumPy 输出进行比较的测试用例
        run_test((12, 3, 4), ind=1)
        run_test((3, 8, 24), ind=2)
        run_test((18, 3, 3, 2), ind=1)
        run_test((1, 4, 2, 2), ind=2)
        run_test((2, 3, 5, 30), ind=3)
        run_test((24, 2, 2, 3, 2), ind=1)
        run_test((3, 4, 2, 3, 2), ind=2)
        run_test((1, 2, 3, 2, 3), ind=3)
        run_test((3, 2, 1, 2, 12), ind=4)

    @skipMeta  # See https://github.com/pytorch/pytorch/issues/53739
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_tensorinv_empty(self, device, dtype):
        # 遍历指标范围，测试空输入情况。这些情况下 NumPy 无法正常工作。
        for ind in range(1, 4):
            # 创建一个空的张量 a，指定形状和数据类型，并放置在指定设备上
            a = torch.empty(0, 0, 1, 2, 3, 0, dtype=dtype, device=device)
            # 使用 torch.linalg.tensorinv 计算张量 a 的逆，指定逆的维度 ind
            a_inv = torch.linalg.tensorinv(a, ind=ind)
            # 断言计算得到的逆矩阵形状与张量 a 的形状和指标 ind 相对应
            self.assertEqual(a_inv.shape, a.shape[ind:] + a.shape[:ind])

    @skipMeta  # See https://github.com/pytorch/pytorch/issues/53739
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，用于测试 torch.linalg.tensorinv 方法的错误和警告情况
    def test_tensorinv_errors_and_warnings(self, device, dtype):

        # 定义一个内部函数，检查输入形状是否满足 tensorinv 方法的要求
        def check_shape(a_shape, ind):
            # 创建一个随机张量 a，其形状为 a_shape，数据类型为 dtype，设备为 device
            a = torch.randn(a_shape, dtype=dtype, device=device)
            # 使用断言检查调用 tensorinv 方法时是否会引发 RuntimeError 异常，异常信息包含特定字符串
            with self.assertRaisesRegex(RuntimeError, "Expected self to satisfy the requirement"):
                torch.linalg.tensorinv(a, ind=ind)

        # 定义一个内部函数，检查传入的索引 ind 是否有效
        def check_ind(a_shape, ind):
            # 创建一个随机张量 a，其形状为 a_shape，数据类型为 dtype，设备为 device
            a = torch.randn(a_shape, dtype=dtype, device=device)
            # 使用断言检查调用 tensorinv 方法时是否会引发 RuntimeError 异常，异常信息包含特定字符串
            with self.assertRaisesRegex(RuntimeError, "Expected a strictly positive integer"):
                torch.linalg.tensorinv(a, ind=ind)

        # 定义一个内部函数，检查传入的输出张量 out 是否符合要求
        def check_out(a_shape, ind):
            # 创建一个随机张量 a，其形状为 a_shape，数据类型为 dtype，设备为 device
            a = torch.randn(a_shape, dtype=dtype, device=device)
            # 创建一个和 a 同样形状的空张量 out
            out = torch.empty_like(a)
            # 使用警告捕获器记录警告信息
            with warnings.catch_warnings(record=True) as w:
                # 调用 tensorinv 方法，使用指定的 ind 和 out
                torch.linalg.tensorinv(a, ind=ind, out=out)
                # 断言捕获到的警告数量为 1
                self.assertEqual(len(w), 1)
                # 断言捕获到的最后一条警告消息包含特定字符串
                self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

            # 创建一个形状为 (0,) 的空张量 out，数据类型为 torch.int，设备为 device
            out = torch.empty(0, dtype=torch.int, device=device)
            # 使用断言检查调用 tensorinv 方法时是否会引发 RuntimeError 异常，异常信息包含特定字符串
            with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
                torch.linalg.tensorinv(a, ind=ind, out=out)

            # 如果 CUDA 可用，创建一个形状为 (0,) 的空张量 out，数据类型为 dtype，但设备类型错误
            if torch.cuda.is_available():
                wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
                out = torch.empty(0, dtype=dtype, device=wrong_device)
                # 使用断言检查调用 tensorinv 方法时是否会引发 RuntimeError 异常，异常信息包含特定字符串
                with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                    torch.linalg.tensorinv(a, ind=ind, out=out)

        # 测试不合法形状的输入情况
        check_shape((2, 3, 4), ind=1)
        check_shape((1, 2, 3, 4), ind=3)

        # 测试不合法索引的输入情况
        check_ind((12, 3, 4), ind=-1)
        check_ind((18, 3, 3, 2), ind=0)

        # 测试不合法输出张量的情况
        check_out((12, 3, 4), ind=1)
        check_out((3, 8, 24), ind=2)


    # 使用装饰器指定条件下跳过测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    # 使用装饰器指定数据类型为浮点数或复数的测试用例
    @dtypes(*floating_and_complex_types())
    # 定义测试函数，用于测试 tensorinv 方法在奇异输入情况下的行为
    def test_tensorinv_singular_input(self, device, dtype):

        # 定义一个内部函数，检查输入是否为奇异矩阵
        def check_singular_input(a_shape, ind):
            # 计算尾部乘积 prod_ind_end
            prod_ind_end = np.prod(a_shape[ind:])
            # 创建一个单位矩阵 a，形状为 (prod_ind_end, prod_ind_end)，数据类型为 dtype，设备为 device
            a = torch.eye(prod_ind_end, dtype=dtype, device=device)
            # 使 a 成为奇异矩阵，将最后一个对角线元素置为 0
            a[-1, -1] = 0   # 现在 `a` 是奇异矩阵
            # 将 a 重新调整形状为 a_shape
            a = a.reshape(a_shape)
            # 使用断言检查调用 tensorinv 方法时是否会引发 torch.linalg.LinAlgError 异常，异常信息包含特定字符串
            with self.assertRaisesRegex(torch.linalg.LinAlgError, "The diagonal element"):
                torch.linalg.tensorinv(a, ind=ind)

        # 测试输入为奇异矩阵的情况
        check_singular_input((12, 3, 4), ind=1)
        check_singular_input((3, 6, 18), ind=2)
    def _test_dot_vdot_vs_numpy(self, device, dtype, torch_fn, np_fn):
        def check(x, y):
            # 使用给定的 torch_fn 和输入 x, y 计算结果
            res = torch_fn(x, y)
            # 根据 x 的数据类型选择参考值的计算方式
            if x.dtype == torch.bfloat16:
                ref = torch.from_numpy(np.array(np_fn(x.cpu().float().numpy(), y.cpu().float().numpy())))
            else:
                ref = torch.from_numpy(np.array(np_fn(x.cpu().numpy(), y.cpu().numpy())))
            # 检查计算结果与参考值是否一致，考虑到 bfloat16 数据类型的特殊处理
            if res.dtype == torch.bfloat16:
                self.assertEqual(res.cpu(), ref.bfloat16())
            else:
                self.assertEqual(res.cpu(), ref)

            # 测试输出参数的情况
            out = torch.empty_like(res)
            torch_fn(x, y, out=out)
            self.assertEqual(out, res)

        # 测试空张量的情况
        x = torch.tensor([], dtype=dtype, device=device)
        y = torch.tensor([], dtype=dtype, device=device)
        check(x, y)

        # 测试连续内存布局的情况
        x = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        y = 0.1 * torch.randn(5000, dtype=dtype, device=device)
        check(x, y)

        # 测试非连续内存布局，但维度一致的情况
        y = 0.1 * torch.randn(1, dtype=dtype, device=device).expand(5000)
        check(x, y)

        # 测试非连续内存布局，并且步长不同的情况
        check(x[::2], y[::2])

    @dtypes(torch.float, torch.cfloat, torch.bfloat16, torch.float16)
    @dtypesIfCUDA(torch.float, torch.cfloat)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5, torch.bfloat16: 1e-0})
    def test_dot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.dot, np.dot)

    @dtypes(torch.float, torch.cfloat)
    @precisionOverride({torch.cfloat: 1e-4, torch.float32: 5e-5})
    def test_vdot_vs_numpy(self, device, dtype):
        self._test_dot_vdot_vs_numpy(device, dtype, torch.vdot, np.vdot)

    def _test_dot_vdot_invalid_args(self, device, torch_fn, complex_dtypes=False):
        def check(x, y, regex):
            # 使用给定的 torch_fn 和输入 x, y 检查是否引发预期的异常
            with self.assertRaisesRegex(RuntimeError, regex):
                torch_fn(x, y)

        if complex_dtypes:
            # 复杂数据类型的测试
            x = torch.randn(1, dtype=torch.cfloat, device=device)
            y = torch.randn(3, dtype=torch.cdouble, device=device)
        else:
            # 普通数据类型的测试
            x = torch.randn(1, dtype=torch.float, device=device)
            y = torch.randn(3, dtype=torch.double, device=device)

        # 检查不同情况下是否会引发异常
        check(x, y, 'dot : expected both vectors to have same dtype')
        check(x.reshape(1, 1), y, '1D tensors expected')
        check(x.expand(9), y.to(x.dtype), 'inconsistent tensor size')

        if self.device_type != 'cpu':
            # GPU 上的设备类型不一致性测试
            x_cpu = x.expand(3).cpu()
            check(x_cpu, y.to(x.dtype), 'Expected all tensors to be on the same device')

    @onlyNativeDeviceTypes
    def test_vdot_invalid_args(self, device):
        self._test_dot_vdot_invalid_args(device, torch.vdot)
        self._test_dot_vdot_invalid_args(device, torch.vdot, complex_dtypes=True)

    @onlyNativeDeviceTypes


这里是对给定代码块的详细注释，按照规定的格式和要求完成了注释部分。
    # 测试函数，用于验证 torch.dot 函数的无效参数
    def test_dot_invalid_args(self, device):
        # 调用 _test_dot_vdot_invalid_args 函数测试 torch.dot
        self._test_dot_vdot_invalid_args(device, torch.dot)
        # 使用 complex_dtypes=True 参数再次调用 _test_dot_vdot_invalid_args 函数测试 torch.dot

    # 装饰器，如果没有 Magma 库则跳过 CUDA 测试
    # 如果没有 LAPACK 库则跳过 CPU 测试
    # 测试函数，用于测试 matrix_rank 函数
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank(self, device, dtype):
        # 获取 torch.linalg.matrix_rank 函数的引用
        matrix_rank = torch.linalg.matrix_rank

        # 内部函数，运行具体测试
        def run_test(shape0, shape1, batch):
            # 生成指定形状、数据类型的随机张量 a
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            # 计算张量 a 的秩
            rank_a = matrix_rank(a)

            # 断言：a 的秩与其共轭转置的秩相等
            self.assertEqual(rank_a, matrix_rank(a.mH))
            
            # 计算 a 与其共轭转置的乘积 aaH
            aaH = torch.matmul(a, a.mH)
            # 计算 aaH 的秩
            rank_aaH = matrix_rank(aaH)
            # 使用 hermitian=True 计算 aaH 的秩
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            # 断言：不使用和使用 hermitian=True 计算得到的 aaH 的秩相等
            self.assertEqual(rank_aaH, rank_aaH_hermitian)
            
            # 计算 a 的共轭转置与 a 的乘积 aHa
            aHa = torch.matmul(a.mH, a)
            # 断言：aHa 的秩与其共轭转置的秩相等
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            # 使用 NumPy 比较结果
            self.assertEqual(rank_a, np.linalg.matrix_rank(a.cpu().numpy()))
            self.assertEqual(matrix_rank(a, 0.01), np.linalg.matrix_rank(a.cpu().numpy(), 0.01))
            self.assertEqual(rank_aaH, np.linalg.matrix_rank(aaH.cpu().numpy()))
            self.assertEqual(matrix_rank(aaH, 0.01), np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01))

            # 如果 NumPy 版本 >= 1.14.0，则使用 hermitian=True 参数比较结果
            if np.lib.NumpyVersion(np.__version__) >= '1.14.0':
                self.assertEqual(rank_aaH_hermitian,
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), hermitian=True))
                self.assertEqual(matrix_rank(aaH, 0.01, True),
                                 np.linalg.matrix_rank(aaH.cpu().numpy(), 0.01, True))

            # 检查 out= 参数的变体
            out = torch.empty(a.shape[:-2], dtype=torch.int64, device=device)
            ans = matrix_rank(a, out=out)
            self.assertEqual(ans, out)
            self.assertEqual(ans, rank_a)

        # 待测试的形状和批处理大小
        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        # 对每对形状和批处理大小运行测试
        for (shape0, shape1), batch in zip(itertools.product(shapes, reversed(shapes)), batches):
            run_test(shape0, shape1, batch)

    # 装饰器，如果没有 Magma 库则跳过 CUDA 测试
    # 如果没有 LAPACK 库则跳过 CPU 测试
    # 测试函数，用于测试 matrix_rank 函数
    @dtypes(*floating_and_complex_types())
    # 定义一个测试方法，用于测试 matrix_rank 函数在给定设备和数据类型下的行为
    def test_matrix_rank_atol(self, device, dtype):

        # 定义一个内部方法，用于运行具有给定形状和批次的测试
        def run_test_atol(shape0, shape1, batch):
            # 创建一个指定形状、设备和数据类型的张量 a
            a = make_tensor((*batch, shape0, shape1), dtype=dtype, device=device)
            # 检查与 NumPy 输出的一致性
            # 测试浮点公差和每个矩阵的特定值
            tolerances = [float(torch.rand(1)), ]
            
            # 测试不同类型的公差张量
            for tol_type in all_types():
                tolerances.append(make_tensor(a.shape[:-2], dtype=tol_type, device=device, low=0))
            
            # 测试公差的广播性质
            if a.ndim > 2:
                tolerances.append(make_tensor(a.shape[-3], dtype=torch.float32, device=device, low=0))
            
            # 遍历所有公差值进行测试
            for tol in tolerances:
                # 使用 atol 参数计算实际的矩阵秩
                actual = torch.linalg.matrix_rank(a, atol=tol)
                # 使用 tol 参数计算实际的矩阵秩
                actual_tol = torch.linalg.matrix_rank(a, tol=tol)
                # 断言两种方式计算的秩应当相等
                self.assertEqual(actual, actual_tol)
                # 将公差转换为 NumPy 数组，如果不是 float 类型的话
                numpy_tol = tol if isinstance(tol, float) else tol.cpu().numpy()
                # 使用 NumPy 的 linalg.matrix_rank 计算预期的矩阵秩
                expected = np.linalg.matrix_rank(a.cpu().numpy(), tol=numpy_tol)
                # 断言 torch 和 NumPy 计算的矩阵秩应当相等
                self.assertEqual(actual, expected)

        # 定义形状和批次的组合
        shapes = (3, 13)
        batches = ((), (0, ), (4, ), (3, 5, ))
        
        # 遍历形状和批次的组合进行测试
        for (shape0, shape1), batch in zip(itertools.product(shapes, reversed(shapes)), batches):
            run_test_atol(shape0, shape1, batch)

    # 装饰器：如果没有 Magma 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 装饰器：如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 装饰器：指定测试数据类型为 torch.float64
    @dtypes(torch.float64)
    # 定义测试方法，测试 matrix_rank 函数在给定设备和数据类型下的行为，同时考虑 atol 和 rtol 参数
    def test_matrix_rank_atol_rtol(self, device, dtype):
        # 创建一个生成满秩矩阵的函数对象，并部分应用设备和数据类型参数
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        # 创建一个 n x n 的满秩矩阵 a
        n = 9
        a = make_arg(n, n)

        # 测试 float 和 tensor 变体的公差值
        for tol_value in [0.81, torch.tensor(0.81, device=device)]:
            # 使用 rtol (相对公差) 参数考虑最大奇异值（在这里为 1.5）的影响
            result = torch.linalg.matrix_rank(a, rtol=tol_value)
            self.assertEqual(result, 2)  # 存在两个奇异值大于 1.5*0.81 = 1.215

            # 使用 atol 直接与奇异值比较
            result = torch.linalg.matrix_rank(a, atol=tol_value)
            self.assertEqual(result, 7)  # 存在七个奇异值大于 0.81

            # 当同时指定 atol 和 rtol 时，使用最大的公差值
            result = torch.linalg.matrix_rank(a, atol=tol_value, rtol=tol_value)
            self.assertEqual(result, 2)  # 存在两个奇异值大于 max(0.81, 1.5*0.81)

    # 装饰器：如果没有 Magma 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 装饰器：如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 装饰器：如果 CUDA 版本在指定列表中，则跳过 CUDA 测试
    @skipCUDAVersionIn([(11, 6), (11, 7)])  # https://github.com/pytorch/pytorch/issues/75391
    # 装饰器：指定测试数据类型为所有浮点和复数类型
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_empty(self, device, dtype):
        # 导入 torch.linalg.matrix_rank 函数
        matrix_rank = torch.linalg.matrix_rank

        # NumPy doesn't work for input with no elements
        # 定义测试函数，用于不同形状和批次的矩阵计算
        def run_test(shape0, shape1, batch):
            # 生成指定形状的随机张量 a，设备为 device，数据类型为 dtype
            a = torch.randn(*batch, shape0, shape1, dtype=dtype, device=device)
            # 计算张量 a 的秩
            rank_a = matrix_rank(a)
            # 生成预期结果为全零的张量，数据类型为 torch.int64，设备为 device
            expected = torch.zeros(batch, dtype=torch.int64, device=device)

            # 断言张量 a 的秩与其共轭转置的秩相等
            self.assertEqual(rank_a, matrix_rank(a.mH))

            # 计算张量 a 与其共轭转置的乘积
            aaH = torch.matmul(a, a.mH)
            # 计算 aaH 的秩
            rank_aaH = matrix_rank(aaH)
            # 计算 aaH 的秩（假设是共轭的情况）
            rank_aaH_hermitian = matrix_rank(aaH, hermitian=True)
            # 断言两种方式计算的秩相等
            self.assertEqual(rank_aaH, rank_aaH_hermitian)

            # 计算 a 的共轭转置与 a 的乘积
            aHa = torch.matmul(a.mH, a)
            # 断言计算出的秩与设定的秩相等
            self.assertEqual(matrix_rank(aHa), matrix_rank(aHa, hermitian=True))

            # 断言计算出的秩与预期的全零张量相等
            self.assertEqual(rank_a, expected)
            # 断言计算出的秩（带有 tol 参数）与预期的全零张量相等
            self.assertEqual(matrix_rank(a, 0.01), expected)

            # 断言计算出的秩与预期的全零张量相等
            self.assertEqual(rank_aaH, expected)
            # 断言计算出的秩（带有 tol 参数）与预期的全零张量相等
            self.assertEqual(matrix_rank(aaH, 0.01), expected)

            # 断言计算出的秩与预期的全零张量相等
            self.assertEqual(rank_aaH_hermitian, expected)
            # 断言计算出的秩（带有 tol 和 hermitian 参数）与预期的全零张量相等
            self.assertEqual(matrix_rank(aaH, 0.01, True), expected)

        # 定义不同批次的测试参数
        batches = ((), (4, ), (3, 5, ))
        # 遍历不同批次的测试参数
        for batch in batches:
            # 分别测试形状为 (0, 0)，(0, 3)，(3, 0) 的情况
            run_test(0, 0, batch)
            run_test(0, 3, batch)
            run_test(3, 0, batch)

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    def test_matrix_rank_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        # 生成单位矩阵 a，数据类型为 dtype，设备为 device
        a = torch.eye(2, dtype=dtype, device=device)
        # 生成空的张量 out，数据类型为 torch.bool，设备为 device
        out = torch.empty(0, dtype=torch.bool, device=device)
        # 断言运行时错误，检测是否输出的 dtype 与预期不符合
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Bool"):
            torch.linalg.matrix_rank(a, out=out)

        # device should match
        # 如果 CUDA 可用，检查设备类型不匹配时是否会抛出运行时错误
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            # 生成空的张量 out，数据类型为 dtype，设备为 wrong_device
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            # 断言运行时错误，检测是否输出张量的设备类型与输入张量不一致
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.matrix_rank(a, out=out)

        # if out tensor with wrong shape is passed a warning is given
        # 如果输出张量的形状与预期不符合，会触发警告
        with warnings.catch_warnings(record=True) as w:
            # 生成形状为 (3,) 的空张量 out，数据类型为 dtype，设备为 device
            out = torch.empty(3, dtype=dtype, device=device)
            # 触发警告，计算矩阵的秩并将结果写入 out
            torch.linalg.matrix_rank(a, out=out)
            # 检查是否触发了警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    # 定义测试函数，用于测试基本的矩阵秩功能
    def test_matrix_rank_basic(self, device, dtype):
        # 引入 torch.linalg.matrix_rank 到当前命名空间
        matrix_rank = torch.linalg.matrix_rank

        # 创建一个 10x10 的单位矩阵 a，数据类型为 dtype，存储在指定设备上
        a = torch.eye(10, dtype=dtype, device=device)
        # 断言计算得到的矩阵秩为 10
        self.assertEqual(matrix_rank(a).item(), 10)
        # 断言计算得到的共轭传置矩阵秩为 10
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 10)

        # 修改矩阵 a 的元素 (5, 5) 为 0
        a[5, 5] = 0
        # 断言计算得到的矩阵秩为 9
        self.assertEqual(matrix_rank(a).item(), 9)
        # 断言计算得到的共轭传置矩阵秩为 9
        self.assertEqual(matrix_rank(a, hermitian=True).item(), 9)

    # 仅对本地设备类型执行的装饰器
    @onlyNativeDeviceTypes
    # 仅对双精度浮点数数据类型执行的装饰器
    @dtypes(torch.double)
    # 此测试用例仅涵盖 torch.chain_matmul 与其“别名” torch.linalg.multi_dot 不同的情况。
    def test_chain_matmul(self, device, dtype):
        # chain_matmul 接受单个输入张量，而 multi_dot 不接受
        t = make_tensor((2, 2), dtype=dtype, device=device)
        # 断言 t 与 torch.chain_matmul(t) 结果相等
        self.assertEqual(t, torch.chain_matmul(t))
        # 使用正则表达式断言 RuntimeError 中包含 "chain_matmul(): Expected one or more matrices" 的异常被引发
        with self.assertRaisesRegex(RuntimeError, r"chain_matmul\(\): Expected one or more matrices"):
            torch.chain_matmul()

        # chain_matmul 要求所有张量都是2D的，而 multi_dot 允许第一个和最后一个张量是1D或2D的
        # 使用正则表达式断言 RuntimeError 中包含 "Tensor dimension is 1, expected 2 instead" 的异常被引发
        with self.assertRaisesRegex(RuntimeError, r"Tensor dimension is 1, expected 2 instead"):
            torch.chain_matmul(make_tensor(1, dtype=dtype, device=device), make_tensor(1, dtype=dtype, device=device))

    # 仅对本地设备类型执行的装饰器
    @onlyNativeDeviceTypes
    # 仅对双精度浮点数和复双精度浮点数数据类型执行的装饰器
    @dtypes(torch.double, torch.cdouble)
    def test_multi_dot(self, device, dtype):
        # 定义内部函数 check，用于验证多个形状的输入
        def check(*shapes):
            # 根据输入的形状创建张量列表 tensors
            tensors = [make_tensor(shape, dtype=dtype, device=device) for shape in shapes]
            # 将张量转换为 numpy 数组，存储在 np_arrays 中
            np_arrays = [tensor.cpu().numpy() for tensor in tensors]
            # 使用 torch.linalg.multi_dot 计算张量列表的乘积，并转移到 CPU 上
            res = torch.linalg.multi_dot(tensors).cpu()
            # 使用 numpy.linalg.multi_dot 计算 np_arrays 数组的乘积，转换为 PyTorch 张量 ref
            ref = torch.from_numpy(np.array(np.linalg.multi_dot(np_arrays)))
            # 断言计算得到的结果 res 等于参考结果 ref
            self.assertEqual(res, ref)

        # 对空维度输入进行测试
        check([0], [0])
        check([2], [2, 0])
        check([1, 0], [0])
        check([0, 2], [2, 1])
        check([2, 2], [2, 0])
        check([2, 0], [0, 3])
        check([0, 0], [0, 1])
        check([4, 2], [2, 0], [0, 3], [3, 2])

        # 对变化的输出形状进行测试
        check([2], [2])
        check([1, 2], [2])
        check([2], [2, 1])
        check([1, 2], [2, 1])
        check([3, 2], [2, 4])

        # 对多个输入张量进行测试
        check([3], [3, 4], [4, 2], [2, 5], [5])
        check([1, 2], [2, 2], [2, 3], [3, 1])

        # 对大型张量进行测试
        check([10, 100], [100, 5], [5, 50])
        check([10, 20], [20, 30], [30, 5])
    # 定义一个测试方法，用于测试多重点乘运算的各种错误情况
    def test_multi_dot_errors(self, device, dtype):
        
        # 定义一个内部函数，用于检查点乘运算是否抛出预期的运行时错误消息
        def check(tensors, out, msg):
            # 使用断言验证是否抛出指定消息的运行时错误
            with self.assertRaisesRegex(RuntimeError, msg):
                torch.linalg.multi_dot(tensors, out=out)

        # 创建一个设备和数据类型指定的张量 a
        a = make_tensor(2, dtype=dtype, device=device)

        # 测试空列表作为输入时是否抛出 "expected at least 2 tensors" 错误
        check([], None, "expected at least 2 tensors")
        # 测试单个张量作为输入时是否抛出 "expected at least 2 tensors" 错误
        check([a], None, "expected at least 2 tensors")

        # 测试第一个张量为标量时是否抛出 "the first tensor must be 1D or 2D" 错误
        check([torch.tensor(1, device=device, dtype=dtype), a], None, "the first tensor must be 1D or 2D")
        # 测试最后一个张量为标量时是否抛出 "the last tensor must be 1D or 2D" 错误
        check([a, torch.tensor(1, device=device, dtype=dtype)], None, "the last tensor must be 1D or 2D")

        # 测试第一个张量不为二维时是否抛出 "tensor 1 must be 2D" 错误
        check([a, a, a], None, "tensor 1 must be 2D")
        # 测试第一个张量不为二维时是否抛出 "tensor 1 must be 2D" 错误
        check([a, make_tensor((2, 2, 2), dtype=dtype, device=device), a], None, "tensor 1 must be 2D")

        # 测试所有张量不具有相同数据类型时是否抛出 "all tensors must have be the same dtype" 错误
        check([a, make_tensor(2, dtype=torch.double, device=device)], None, "all tensors must have be the same dtype")
        # 测试输出张量数据类型不匹配时是否抛出 "expected out tensor to have dtype" 错误
        check([a, a], torch.empty(0, device=device, dtype=torch.double), "expected out tensor to have dtype")

        # 如果设备类型为 'cuda'，测试所有张量不在同一设备上时是否抛出 "all tensors must be on the same device" 错误
        if self.device_type == 'cuda':
            check([a, make_tensor(2, dtype=dtype, device="cpu")], None, "all tensors must be on the same device")
            # 测试输出张量不在期望设备上时是否抛出 "expected out tensor to be on device" 错误
            check([a, a], torch.empty(0, dtype=dtype), "expected out tensor to be on device")

        # 测试张量维度不支持点乘运算时是否抛出 "cannot be multiplied" 错误
        check([a, make_tensor(3, dtype=dtype, device=device)], None, "cannot be multiplied")
        # 测试张量维度不支持点乘运算时是否抛出 "cannot be multiplied" 错误
        check([a, make_tensor((3, 2), dtype=dtype, device=device), a], None, "cannot be multiplied")

    # 使用指定的精度覆盖装饰器修饰当前测试方法
    @precisionOverride({torch.float32: 5e-6, torch.complex64: 5e-6})
    # 如果没有 Cusolver，跳过当前测试方法装饰器修饰
    @skipCUDAIfNoCusolver
    # 如果没有 Lapack，跳过当前测试方法装饰器修饰
    @skipCPUIfNoLapack
    # 使用浮点数和复数类型的数据类型修饰当前测试方法
    @dtypes(*floating_and_complex_types())
    # 定义测试函数 test_qr，用于测试 torch.qr 函数的行为
    def test_qr(self, device, dtype):
        
        # 内部函数 run_test，执行具体的测试
        def run_test(tensor_dims, some):
            # 创建一个随机的张量 A，指定设备和数据类型
            A = torch.randn(*tensor_dims, dtype=dtype, device=device)
            # 对张量 A 进行 QR 分解，得到 Q 和 R
            Q, R = torch.qr(A, some=some)

            # Check0: 验证 Q 的形状是否为 (m, n_columns)，R 的形状是否为 (n_columns, n)
            m, n = tensor_dims[-2:]
            n_columns = m if (not some) and m > n else min(m, n)
            self.assertEqual(Q.size(-2), m)
            self.assertEqual(R.size(-1), n)
            self.assertEqual(Q.size(-1), n_columns)

            # 将张量 A 转换为 numpy 数组
            A_ = A.cpu().numpy()
            Q_ = Q.cpu().numpy()
            R_ = R.cpu().numpy()

            # Check1: 验证 A 是否等于 QR
            self.assertEqual(A_, np.matmul(Q_, R_))

            # Check2: 使用输出参数进行 QR 分解，验证 A 是否等于 QR
            Q_out, R_out = torch.full_like(Q, math.nan), torch.full_like(R, math.nan)
            torch.qr(A, some=some, out=(Q_out, R_out))
            Q_out_ = Q_out.cpu().numpy()
            R_out_ = R_out.cpu().numpy()
            self.assertEqual(A_, np.matmul(Q_out_, R_out_))

            # Check3: 验证 Q 和 Q_out，R 和 R_out 是否相等
            self.assertEqual(Q_, Q_out_)
            self.assertEqual(R_, R_out_)

            # Check4: 验证 Q 的转置乘以 Q 是否为单位矩阵，以及 R 的上三角部分是否等于 R 自身
            eye = torch.eye(n_columns, device=device, dtype=dtype).expand(Q.shape[:-2] + (n_columns, n_columns)).cpu().numpy()
            self.assertEqual(np.matmul(Q_.swapaxes(-1, -2).conj(), Q_), eye)
            self.assertEqual(R.triu(), R)

        # 定义不同维度的张量列表和 some 参数的组合
        tensor_dims_list = [(0, 5), (0, 0), (5, 0),  # 空张量
                            (2, 1, 0, 5), (2, 1, 0, 0), (2, 1, 5, 0), (2, 0, 5, 5),  # 批处理空张量
                            (3, 5), (5, 5), (5, 3),  # 单个矩阵
                            (7, 3, 5), (7, 5, 5), (7, 5, 3),  # 三维张量
                            (7, 5, 3, 5), (7, 5, 5, 5), (7, 5, 5, 3)]  # 四维张量
                            
        # 对每个 tensor_dims 和 some 的组合运行测试
        for tensor_dims, some in itertools.product(tensor_dims_list, [True, False]):
            run_test(tensor_dims, some)

    # 添加装饰器，如果没有 cusolver 库则跳过 CUDA 测试
    @skipCUDAIfNoCusolver
    # 添加装饰器，如果没有 lapack 库则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 添加数据类型装饰器，指定测试的数据类型
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 定义一个测试方法，用于比较 torch.linalg.qr 和 numpy.linalg.qr 的结果
    def test_qr_vs_numpy(self, device, dtype):
        """
        test torch.linalg.qr vs numpy.linalg.qr
        """
        # 定义要测试的矩阵大小的列表
        sizes_to_test = [
            (7, 5),     # 测试 (7, 5) 大小的矩阵
            (5, 7),     # 测试 (5, 7) 大小的矩阵
            (5, 0),     # 空矩阵
            (0, 5),     # 空矩阵
        ]
        # 遍历测试大小的列表
        for size in sizes_to_test:
            # 生成一个在指定设备上的随机张量 t
            t = torch.randn(size, device=device, dtype=dtype)
            # 将张量 t 转换为 numpy 数组 np_t
            np_t = t.cpu().numpy()
            # 遍历两种模式 'reduced' 和 'complete'
            for mode in ['reduced', 'complete']:
                # 使用 numpy 计算 QR 分解的期望结果 exp_q 和 exp_r
                exp_q, exp_r = np.linalg.qr(np_t, mode=mode)
                # 使用 torch 计算 QR 分解的结果 q 和 r
                q, r = torch.linalg.qr(t, mode=mode)
                # 断言 torch 的结果与 numpy 的结果相等
                self.assertEqual(q, exp_q)
                self.assertEqual(r, exp_r)
            #
            # 对于 mode='r' 需要特殊处理，因为 numpy 只返回 r
            exp_r = np.linalg.qr(np_t, mode='r')
            # 使用 torch 计算 QR 分解的结果 q 和 r
            q, r = torch.linalg.qr(t, mode='r')
            # 断言 q 是空张量
            self.assertEqual(q.shape, (0,))
            self.assertEqual(q.dtype, t.dtype)
            self.assertEqual(q.device, t.device)
            # 断言 r 与 numpy 的结果 exp_r 相等
            self.assertEqual(r, exp_r)

    # 标记如果没有 Cusolver 库则跳过测试
    @skipCUDAIfNoCusolver
    # 标记如果没有 Lapack 库则跳过测试
    @skipCPUIfNoLapack
    # 指定测试函数使用的数据类型为 torch.float
    @dtypes(torch.float)
    # 测试 torch.linalg.qr 在自动求导时的错误处理
    def test_linalg_qr_autograd_errors(self, device, dtype):
        # torch.linalg.qr(mode='r') 只返回 'r' 并丢弃 'q'，但是没有 'q' 无法进行反向传播。
        # 检查在这种情况下，是否能够正确抛出 linalg_qr_backward 的异常。
        inp = torch.randn((5, 7), device=device, dtype=dtype, requires_grad=True)
        # 使用 torch 计算 QR 分解的结果 q 和 r
        q, r = torch.linalg.qr(inp, mode='r')
        # 断言 q 是空张量
        self.assertEqual(q.shape, (0,))
        b = torch.sum(r)
        # 使用断言检查是否能够捕获到期望的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError,
                                    "The derivative of linalg.qr depends on Q"):
            b.backward()
        #
        inp = torch.randn((7, 5), device=device, dtype=dtype, requires_grad=True)
        # 使用 torch 计算 QR 分解的结果 q 和 r
        q, r = torch.linalg.qr(inp, mode='complete')
        b = torch.sum(r)
        # 使用断言检查是否能够捕获到期望的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError,
                                    "The QR decomposition is not differentiable when mode='complete' and nrows > ncols"):
            b.backward()

    # 标记如果没有 Cusolver 库则跳过测试
    @skipCUDAIfNoCusolver
    # 标记如果没有 Lapack 库则跳过测试
    @skipCPUIfNoLapack
    # 指定测试函数使用的数据类型为 torch.float, torch.double, torch.cfloat, torch.cdouble
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_qr_batched(self, device, dtype):
        """
        test torch.linalg.qr vs numpy.linalg.qr. We need some special logic
        because numpy does not support batched qr
        """
        def np_qr_batched(a, mode):
            """poor's man batched version of np.linalg.qr"""
            all_q = []
            all_r = []
            for matrix in a:
                # 调用 numpy 的 QR 分解函数，根据模式返回 Q 和 R 或者仅返回 R
                result = np.linalg.qr(matrix, mode=mode)
                if mode == 'r':
                    all_r.append(result)
                else:
                    q, r = result
                    all_q.append(q)
                    all_r.append(r)
            if mode == 'r':
                # 返回所有 R 组成的数组
                return np.array(all_r)
            else:
                # 返回所有 Q 和所有 R 组成的数组
                return np.array(all_q), np.array(all_r)

        # 创建一个形状为 (3, 7, 5) 的随机张量 t
        t = torch.randn((3, 7, 5), device=device, dtype=dtype)
        # 将 t 转换为 numpy 数组
        np_t = t.cpu().numpy()
        # 遍历模式列表 ['reduced', 'complete']
        for mode in ['reduced', 'complete']:
            # 调用自定义的 np_qr_batched 函数得到期望的 Q 和 R
            exp_q, exp_r = np_qr_batched(np_t, mode=mode)
            # 调用 torch.linalg.qr 函数得到计算结果的 Q 和 R
            q, r = torch.linalg.qr(t, mode=mode)
            # 断言 torch 计算的 Q 与期望的 Q 相等
            self.assertEqual(q, exp_q)
            # 断言 torch 计算的 R 与期望的 R 相等
            self.assertEqual(r, exp_r)
        
        # 对于 mode='r'，需要特殊处理，因为 numpy 只返回 R
        exp_r = np_qr_batched(np_t, mode='r')
        # 调用 torch.linalg.qr 函数计算 R
        q, r = torch.linalg.qr(t, mode='r')
        # 检查 Q 是否为空张量
        self.assertEqual(q.shape, (0,))
        # 检查 Q 的数据类型是否与 t 相同
        self.assertEqual(q.dtype, t.dtype)
        # 检查 Q 的设备是否与 t 相同
        self.assertEqual(q.device, t.device)
        # 检查 R 是否与期望的 R 相等
        self.assertEqual(r, exp_r)

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    @dtypes(torch.float)
    def test_qr_error_cases(self, device, dtype):
        # 创建一个形状为 (5,) 的随机张量 t1
        t1 = torch.randn(5, device=device, dtype=dtype)
        # 使用断言检查调用 torch.linalg.qr(t1) 是否会引发指定的异常
        with self.assertRaisesRegex(RuntimeError, 'linalg.qr: The input tensor A must have at least 2 dimensions.'):
            torch.linalg.qr(t1)
        
        # 创建一个形状为 (5, 7) 的随机张量 t2
        t2 = torch.randn((5, 7), device=device, dtype=dtype)
        # 使用断言检查调用 torch.linalg.qr(t2, mode='hello') 是否会引发指定的异常
        with self.assertRaisesRegex(RuntimeError, "qr received unrecognized mode 'hello'"):
            torch.linalg.qr(t2, mode='hello')

    def _check_einsum(self, *args, np_args=None):
        # 如果 np_args 为 None，则将每个参数转换为 numpy 数组
        if np_args is None:
            np_args = [arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
        # 使用 numpy 的 einsum 函数计算参考值 ref
        ref = np.einsum(*np_args)
        # 使用 torch 的 einsum 函数计算结果值 res
        res = torch.einsum(*args)
        # 断言 torch 的 einsum 计算结果与 numpy 的 einsum 计算结果相等
        self.assertEqual(ref, res)

        # 检查 opt_einsum 的其他变化是否也正常工作
        if TEST_OPT_EINSUM:
            with opt_einsum.flags(enabled=False):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

            with opt_einsum.flags(enabled=True, strategy='greedy'):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

            with opt_einsum.flags(enabled=True, strategy='optimal'):
                res = torch.einsum(*args)
                self.assertEqual(ref, res)

    @dtypes(torch.double, torch.cdouble)
    @dtypes(torch.double, torch.cdouble)
    # 测试 Einsum 函数对子列表格式的处理
    def test_einsum_sublist_format(self, device, dtype):
        # 创建长度为 5 的张量 x，指定设备和数据类型
        x = make_tensor((5,), dtype=dtype, device=device)
        # 创建长度为 7 的张量 y，指定设备和数据类型
        y = make_tensor((7,), dtype=dtype, device=device)
        # 创建形状为 (3, 5) 的张量 A，指定设备和数据类型
        A = make_tensor((3, 5), dtype=dtype, device=device)
        # 创建形状为 (2, 5) 的张量 B，指定设备和数据类型
        B = make_tensor((2, 5), dtype=dtype, device=device)
        # 创建形状为 (2, 1, 3, 1, 4) 的张量 C，指定设备和数据类型
        C = make_tensor((2, 1, 3, 1, 4), dtype=dtype, device=device)

        # 调用 _check_einsum 方法，对输入 x 进行运算，保留维度 [0] 的结果
        self._check_einsum(x, [0])
        # 调用 _check_einsum 方法，对输入 x 进行运算，不保留任何维度
        self._check_einsum(x, [0], [])
        # 调用 _check_einsum 方法，对输入 x 和 y 进行运算，保留维度 [0] 和 [1] 的结果
        self._check_einsum(x, [0], y, [1], [0, 1])
        # 调用 _check_einsum 方法，对输入 A 进行运算，交换维度 [0, 1] 的结果
        self._check_einsum(A, [0, 1], [1, 0])
        # 调用 _check_einsum 方法，对输入 A 和 x 进行运算，将维度 [0, 1] 映射到 [1] 的结果，保留维度 [0]
        self._check_einsum(A, [0, 1], x, [1], [0])
        # 调用 _check_einsum 方法，对输入 A 和 B 进行运算，将维度 [0, 1] 映射到 [2, 1] 的结果
        self._check_einsum(A, [0, 1], B, [2, 1])
        # 调用 _check_einsum 方法，对输入 A 和 B 进行运算，将维度 [0, 1] 映射到 [2, 1] 的结果，保留维度 [0, 2]
        self._check_einsum(A, [0, 1], B, [2, 1], [0, 2])
        # 调用 _check_einsum 方法，对输入 C 进行运算，映射维度 [0, 1, 2, 1, ...] 到 [0, 2, 1, ...] 的结果
        self._check_einsum(C, [0, 1, 2, 1, Ellipsis], [0, 2, 1, Ellipsis])
        # 调用 _check_einsum 方法，对 A 的转置和 B 进行运算，映射维度 [0, 1] 和 [Ellipsis, 0] 的结果，保留维度 [1, Ellipsis]
        self._check_einsum(A.t(), [0, 1], B, [Ellipsis, 0], [1, Ellipsis])
        # 调用 _check_einsum 方法，对 A 的转置和 B 进行运算，映射维度 [0, Ellipsis] 和 [1, 0] 的结果，保留维度 [Ellipsis]
        self._check_einsum(A.t(), [0, Ellipsis], B, [1, 0], [Ellipsis])

        # 使用 torch.bilinear 对非连续张量进行计算
        # 创建形状为 (5, 10) 的非连续张量 l，形状为 (5, 20) 的非连续张量 r，形状为 (15, 10, 20) 的权重张量 w
        l = make_tensor((5, 10), dtype=dtype, device=device, noncontiguous=True)
        r = make_tensor((5, 20), dtype=dtype, device=device, noncontiguous=True)
        w = make_tensor((15, 10, 20), dtype=dtype, device=device)
        # 调用 _check_einsum 方法，使用非连续张量 l, r 和权重张量 w 进行 bilinear 运算，映射维度 [40, 41], [2, 41, 50], [40, 50] 的结果，保留维度 [40, 2]
        self._check_einsum(l, [40, 41], w, [2, 41, 50], r, [40, 50], [40, 2])
    # 定义一个测试函数，用于测试 torch.einsum 的边界情况
    def test_einsum_corner_cases(self, device):
        # 定义一个内部函数，用于检查 einsum 的输出是否符合预期
        def check(equation, *operands, expected_output):
            # 根据传入的操作数，创建 Torch 张量列表
            tensors = [torch.tensor(operand, device=device, dtype=torch.float32) if not isinstance(operand, tuple)
                       else make_tensor(operand, dtype=torch.float32, device=device) for operand in operands]
            # 调用 einsum 函数，计算输出结果
            output = torch.einsum(equation, tensors)
            # 使用断言检查输出是否与预期结果一致
            self.assertEqual(output, torch.tensor(expected_output, dtype=torch.float32, device=device))

        # 测试空等式情况
        check(' ', 1, expected_output=1)
        # 测试只有箭头的等式
        check(' -> ', 1, expected_output=1)
        # 测试逗号分隔的等式
        check(' , ', 2, 2, expected_output=4)
        check(' , , ', 2, 2, 2, expected_output=8)
        check(' , -> ', 2, 2, expected_output=4)
        # 测试带索引的等式
        check(' i ', [1], expected_output=[1])
        check(' i -> ', [1], expected_output=1)
        check(' i -> i ', [1], expected_output=[1])
        check(' i , i ', [2], [2], expected_output=4)
        check(' i , i -> i ', [2], [2], expected_output=[4])

        # 测试零尺寸维度的张量
        check('i', [], expected_output=[])
        check(' i j -> j', [[], []], expected_output=[])
        check('ij->i', [[], []], expected_output=[0., 0.])
        check(' i j k  ,  k  -> i j ', (3, 0, 6), (6,), expected_output=[[], [], []])

        # 测试广播
        check('i,j', [2], [1, 2], expected_output=[[2, 4]])
        check('i,ij->ij', [1, 2], [[1, 2, 3], [2, 3, 4]], expected_output=[[1, 2, 3], [4, 6, 8]])

        # 测试省略号广播
        check('...', 1, expected_output=1)
        check('...->', 1, expected_output=1)
        check('...->...', 1, expected_output=1)
        check('...', [1], expected_output=[1])
        check('...->', [1], expected_output=1)
        check('z...->z', [1], expected_output=[1])
        check('Z...->...Z', [1], expected_output=[1])
        check('...a->', [[2], [4]], expected_output=6)
        check('a...b->ab', [[[1], [2]], [[3], [4]]], expected_output=[[3], [7]])
    # 定义一个测试函数 test_einsum_error_cases，用于测试 torch.einsum 函数在异常情况下的行为
    def test_einsum_error_cases(self, device):
        
        # 定义一个内部函数 check，用于检查 torch.einsum 在给定参数下是否抛出特定异常，并且异常信息匹配指定的正则表达式
        def check(*args, regex, exception=RuntimeError):
            # 使用 assertRaisesRegex 上下文管理器来检查是否抛出指定类型的异常，并且异常信息匹配指定的正则表达式
            with self.assertRaisesRegex(exception, r'einsum\(\):.*' + regex):
                torch.einsum(*args)

        # 创建一个大小为 (2,) 的浮点型张量 x，并放置在指定的设备上
        x = make_tensor((2,), dtype=torch.float32, device=device)
        # 创建一个大小为 (2, 3) 的浮点型张量 y，并放置在指定的设备上
        y = make_tensor((2, 3), dtype=torch.float32, device=device)

        # 测试空的子表达式情况下是否会抛出 ValueError 异常，异常信息包含指定的正则表达式内容
        check('', [], regex=r'at least one operand', exception=ValueError)
        # 测试包含非法 ellipsis 的表达式是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('. ..', [x], regex=r'found \'.\' for operand 0 that is not part of any ellipsis')
        # 测试多个 ellipsis 的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('... ...', [x], regex=r'found \'.\' for operand 0 for which an ellipsis was already found')
        # 测试单个非法子表达式是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('1', [x], regex=r'invalid subscript given at index 0')
        # 测试没有提供足够操作数的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check(',', [x], regex=r'fewer operands were provided than specified in the equation')
        # 测试提供过多操作数的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('', [x, x], regex=r'more operands were provided than specified in the equation')
        # 测试子表达式数量不匹配操作数维度的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('', [x], regex=r'the number of subscripts in the equation \(0\) does not match the number '
              r'of dimensions \(1\) for operand 0 and no ellipsis was given')
        # 测试子表达式数量与操作数维度不匹配且没有 ellipsis 的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('ai', [x], regex=r'the number of subscripts in the equation \(2\) does not match the number '
              r'of dimensions \(1\) for operand 0 and no ellipsis was given')
        # 测试子表达式数量超过操作数维度的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('ai...', [x], regex=r'the number of subscripts in the equation \(2\) is more than the number '
              r'of dimensions \(1\) for operand 0')
        # 测试输出中存在非法 ellipsis 的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('a->... .', [x], regex=r'found \'.\' for output but an ellipsis \(...\) was already found')
        # 测试输出中存在非法 ellipsis 的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('a->..', [x], regex=r'found \'.\' for output that is not part of any ellipsis \(...\)')
        # 测试输出中存在非法子表达式的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('a->1', [x], regex=r'invalid subscript given at index 3')
        # 测试输出中存在重复子表达式的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('a->aa', [x], regex=r'output subscript a appears more than once in the output')
        # 测试输出中存在无效子表达式的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('a->i', [x], regex=r'output subscript i does not appear in the equation for any input operand')
        # 测试操作数维度不匹配且子表达式中重复的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('aa', [y], regex=r'subscript a is repeated for operand 0 but the sizes don\'t match, 3 != 2')
        # 测试多个操作数无法广播的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('...,...', [x, y], regex=r'does not broadcast')
        # 测试多个操作数无法广播的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('a,a', [x, make_tensor((3,), dtype=torch.float32, device=device)], regex=r'does not broadcast')
        # 测试操作数维度不匹配且无法广播的情况下是否会抛出 RuntimeError 异常，异常信息包含指定的正则表达式内容
        check('a, ba', [x, y], regex=r'subscript a has size 3 for operand 1 which does not broadcast with previously'
              r' seen size 2')

        # 测试非法范围的子表达式是否会抛出 ValueError 异常，异常信息包含指定的正则表达式内容
        check(x, [-1], regex=r'not within the valid range \[0, 52\)', exception=ValueError)
        # 测试超出有效范围的子表达式是否会抛出 ValueError 异常，异常信息包含指定的正则表达式内容
        check(x, [52], regex=r'not within the valid range \[0, 52\)', exception=ValueError)
    # 生成用于 torch.linalg.solve_triangular 函数的输入数据集合
    def _gen_shape_inputs_linalg_triangular_solve(self, shape, dtype, device, well_conditioned=False):
        # 偏函数，用于创建指定类型和设备的张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)
        # 偏函数，创建具有不同奇异值的满秩矩阵
        make_fullrank = partial(make_fullrank_matrices_with_distinct_singular_values, dtype=dtype, device=device)
        # 解包形状元组
        b, n, k = shape
        # 对 left, uni, expand_a, tr_a, conj_a, expand_b, tr_b, conj_b 的每一种组合进行迭代
        for left, uni, expand_a, tr_a, conj_a, expand_b, tr_b, conj_b in product((True, False), repeat=8):
            # 如果要求共轭操作但数据类型不支持复数，则跳过
            if (conj_a or conj_b) and not dtype.is_complex:
                continue
            # 如果扩展操作且批次大小为1，则跳过
            if (expand_a or expand_b) and b == 1:
                continue

            # 根据左右操作选择 A 和 B 的大小
            size_a = (b, n, n) if left else (b, k, k)
            size_b = (b, n, k) if not tr_b else (b, k, n)

            # 如果批次大小为1或者需要扩展 A，则从大小中去除批次维度
            if b == 1 or expand_a:
                size_a = size_a[1:]
            # 如果批次大小为1或者需要扩展 B，则从大小中去除批次维度
            if b == 1 or expand_b:
                size_b = size_b[1:]

            # 如果要求生成良好条件的矩阵，则通过 LU 分解创建 A
            if well_conditioned:
                PLU = torch.linalg.lu(make_fullrank(*size_a))
                if uni:
                    # 如果要求单位三角形矩阵，则从 PLU 中选取 L 作为 A
                    A = PLU[1].transpose(-2, -1).contiguous()
                else:
                    # 否则选取 U 作为 A
                    A = PLU[2].contiguous()
            else:
                # 否则直接创建随机矩阵 A 并上三角化
                A = make_arg(size_a)
                A.triu_()

            # 获取 A 的对角线元素
            diag = A.diagonal(0, -2, -1)
            if uni:
                # 如果要求单位三角形矩阵，则将对角线元素设为 1
                diag.fill_(1.)
            else:
                # 否则将小于阈值的对角线元素设为 1
                diag[diag.abs() < 1e-6] = 1.

            # 创建随机矩阵 B
            B = make_arg(size_b)

            # 如果需要转置 A，则进行转置操作
            if tr_a:
                A.transpose_(-2, -1)
            # 如果需要转置 B，则进行转置操作
            if tr_b:
                B.transpose_(-2, -1)
            # 如果需要共轭 A，则进行共轭操作
            if conj_a:
                A = A.conj()
            # 如果需要共轭 B，则进行共轭操作
            if conj_b:
                B = B.conj()
            # 如果需要扩展 A，则进行扩展操作
            if expand_a:
                A = A.expand(b, *size_a)
            # 如果需要扩展 B，则进行扩展操作
            if expand_b:
                B = B.expand(b, n, k)
            
            # 生成 A, B, left, not tr_a, uni 作为生成器的下一个元素
            yield A, B, left, not tr_a, uni

    # 测试 torch.linalg.solve_triangular 函数
    def _test_linalg_solve_triangular(self, A, B, upper, left, uni):
        # 调用 torch.linalg.solve_triangular 求解线性系统，返回解 X
        X = torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=uni)
        # 根据 left 参数判断方程左右乘积是否正确
        if left:
            self.assertEqual(A @ X, B)
        else:
            self.assertEqual(X @ A, B)
        out = B
        # 如果 B 不是连续的且其转置也不是连续的，则克隆 B
        if not B.is_contiguous() and not B.transpose(-2, -1).is_contiguous():
            out = B.clone()
        # 调用 torch.linalg.solve_triangular 保持结果到 out 中，并比较结果与 X
        torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=uni, out=out)
        self.assertEqual(X, out)

    # 指定 CPU 下的容许误差范围
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3 if TEST_WITH_ROCM else 1e-1,
                        torch.float64: 1e-8,
                        torch.complex64: 1e-1,
                        torch.complex128: 1e-8})
    # 定义一个测试函数，用于测试 solve_triangular 函数在不同输入下的行为
    def test_linalg_solve_triangular(self, device, dtype):
        # This exercises the API + BLAS CPU + batched cuBLAS
        # 定义三组不同的测试参数
        ks = (3, 1, 0)  # 矩阵 A 的维度 (k, k)
        ns = (5, 0)     # 矩阵 B 的维度 (n, n)
        bs = (1, 2, 0)  # 批处理大小的不同组合

        # 生成输入数据的方法
        gen_inputs = self._gen_shape_inputs_linalg_triangular_solve
        
        # 遍历所有组合，对每组参数进行测试
        for b, n, k in product(bs, ns, ks):
            # 使用生成器生成输入数据 A, B，以及其他参数
            for A, B, left, upper, uni in gen_inputs((b, n, k), dtype, device, well_conditioned=True):
                # 调用 _test_linalg_solve_triangular 方法进行测试
                self._test_linalg_solve_triangular(A, B, upper, left, uni)

    # 标记测试在特定环境下跳过执行
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Test fails for float64 on GPU (P100, V100) on Meta infra")
    @onlyCUDA
    @skipCUDAIfNoMagma  # Magma needed for the PLU decomposition
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-2, torch.complex64: 1e-2,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 定义一个测试函数，用于测试 solve_triangular 函数在大尺寸输入下的行为
    def test_linalg_solve_triangular_large(self, device, dtype):
        # Exercises magma and cublas
        # 定义两种不同的尺寸参数
        magma = (9, 513, 1)
        iterative_cublas = (2, 64, 1)

        # 生成输入数据的方法
        gen_inputs = self._gen_shape_inputs_linalg_triangular_solve
        
        # 遍历所有组合，对每组参数进行测试
        for shape in (magma, iterative_cublas):
            # 使用生成器生成输入数据 A, B，以及其他参数
            for A, B, left, upper, uni in gen_inputs(shape, dtype, device, well_conditioned=True):
                # 调用 _test_linalg_solve_triangular 方法进行测试
                self._test_linalg_solve_triangular(A, B, upper, left, uni)

    # 定义一个测试函数，用于测试 solve_triangular 函数在广播输入下的行为
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-2, torch.complex64: 1e-2,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_linalg_solve_triangular_broadcasting(self, device, dtype):
        # Partial 函数创建一个根据指定类型和设备生成张量的函数
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        # 定义多组不同的输入大小
        sizes = (((2, 1, 3, 4, 4), (2, 1, 3, 4, 6)),
                 ((2, 1, 3, 4, 4), (4, 6)),
                 ((4, 4), (2, 1, 3, 4, 2)),
                 ((1, 3, 1, 4, 4), (2, 1, 3, 4, 5)))
        
        # 遍历所有组合，对每组参数进行测试
        for size_A, size_B in sizes:
            # 遍历 left, upper, uni 的所有组合
            for left, upper, uni in itertools.product([True, False], repeat=3):
                # 创建输入张量 A
                A = make_arg(size_A)
                # 根据 upper 参数将 A 转换为上三角或下三角矩阵
                if upper:
                    A.triu_()
                else:
                    A.tril_()
                
                # 对角线元素设为单位值或其他值
                diag = A.diagonal(0, -2, -1)
                if uni:
                    diag.fill_(1.)
                else:
                    diag[diag.abs() < 1e-6] = 1.
                
                # 创建输入张量 B
                B = make_arg(size_B)
                # 根据 left 参数转置 B
                if not left:
                    B.transpose_(-2, -1)

                # 调用 torch.linalg.solve_triangular 函数求解方程
                X = torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=uni)
                
                # 根据 left 参数计算另一个张量 B_other
                if left:
                    B_other = A @ X
                else:
                    B_other = X @ A
                
                # 断言 B 和 B_other 在广播后相等
                self.assertEqual(*torch.broadcast_tensors(B, B_other))
    # 定义一个辅助函数，用于测试解三角形方程
    def triangular_solve_test_helper(self, A_dims, b_dims, upper, unitriangular,
                                     device, dtype):
        # 根据参数决定是返回上三角矩阵还是下三角矩阵的函数
        triangle_function = torch.triu if upper else torch.tril
        # 生成随机张量 b，指定数据类型和设备
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        # 生成随机张量 A，指定数据类型和设备
        A = torch.randn(*A_dims, dtype=dtype, device=device)
        # 创建正定矩阵 A
        A = torch.matmul(A, A.t())
        # 将 A 转换为上或下三角形矩阵
        A_triangular = triangle_function(A)
        # 如果需要，将矩阵 A 设置为单位三角形矩阵
        if unitriangular:
            A_triangular.diagonal(dim1=-2, dim2=-1).fill_(1.)
        # 返回生成的向量 b 和三角形矩阵 A
        return b, A_triangular

    # 测试解三角形方程的方法
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @skipIfTorchDynamo("flaky, needs investigation")
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_triangular_solve(self, device, dtype):
        # 定义测试的 k 和 n 的值
        ks = [0, 1, 3]
        ns = [0, 5]
        # 使用 itertools.product 生成 k, n, upper, unitriangular, transpose 的所有组合
        for k, n, (upper, unitriangular, transpose) in itertools.product(ks, ns,
                                                                         itertools.product([True, False], repeat=3)):
            # 调用辅助函数生成测试用的 b 向量和 A 矩阵
            b, A = self.triangular_solve_test_helper((n, n), (n, k), upper,
                                                     unitriangular, device, dtype)
            # 解三角形方程，返回解向量 x
            x = torch.triangular_solve(b, A, upper=upper, unitriangular=unitriangular, transpose=transpose)[0]
            # 如果 transpose 为 True，比较 b 和 A 转置与解向量 x 的乘积
            if transpose:
                self.assertEqual(b, np.matmul(A.t().cpu(), x.cpu()))
            # 如果 transpose 为 False，比较 b 和 A 与解向量 x 的乘积
            else:
                self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

    # 另一个测试解三角形方程的方法，与上一个方法类似
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 定义一个测试函数，用于批量测试三角解法
    def test_triangular_solve_batched(self, device, dtype):
        # 辅助函数：批量三角解法测试助手
        def triangular_solve_batch_helper(A_dims, b_dims, upper, unitriangular, transpose):
            # 调用三角解法测试助手，获取解 x 和系数矩阵 A
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            x_exp_list = []
            # 对每个批次中的 b[i], A[i] 进行三角解法，并将结果存入列表
            for i in range(b_dims[0]):
                x_exp_list.append(torch.triangular_solve(b[i], A[i], upper=upper,
                                                         unitriangular=unitriangular,
                                                         transpose=transpose)[0])
            # 将列表中的解 x_exp 堆叠起来，形成一个张量
            x_exp = torch.stack(x_exp_list)  # Stacked output
            # 对整个批次的 b, A 进行三角解法，并获取解 x_act
            x_act = torch.triangular_solve(b, A, upper=upper,
                                           unitriangular=unitriangular,
                                           transpose=transpose)[0]  # Actual output
            # 断言 x_act 和 x_exp 相等
            self.assertEqual(x_act, x_exp)  # Equality check
            # 如果需要转置 A，进行转置操作
            if transpose:
                A = A.t()

            # 计算 A 与 x_act 的乘积 Ax，并与 b 进行比较
            Ax = np.matmul(A.cpu(), x_act.cpu())
            self.assertEqual(b, Ax)

        # 辅助函数：处理零批次情况的三角解法测试助手
        def triangular_solve_zero_batch_helper(A_dims, b_dims, upper, unitriangular, transpose):
            # 调用三角解法测试助手，获取解 x 和系数矩阵 A
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            # 对 b, A 进行三角解法，并获取解 x
            x = torch.triangular_solve(b, A, upper=upper,
                                       unitriangular=unitriangular,
                                       transpose=transpose)[0]
            # 断言 x 的形状与 b 相同
            self.assertTrue(x.shape == b.shape)

        # 对于每种 upper, unitriangular, transpose 的组合，使用 itertools 进行排列组合
        for upper, unitriangular, transpose in itertools.product([True, False], repeat=3):
            batchsize = 3
            # 测试正常情况下的批量三角解法
            triangular_solve_batch_helper((batchsize, 5, 5), (batchsize, 5, 10),
                                          upper, unitriangular, transpose)

            # 测试空输入情况
            triangular_solve_batch_helper((batchsize, 0, 0), (batchsize, 0, 10),
                                          upper, unitriangular, transpose)
            triangular_solve_batch_helper((batchsize, 0, 0), (batchsize, 0, 0),
                                          upper, unitriangular, transpose)

            # 测试零批次情况
            batchsize = 0
            triangular_solve_zero_batch_helper((batchsize, 5, 5), (batchsize, 5, 10),
                                               upper, unitriangular, transpose)

    # 标记为慢速测试
    @slowTest
    # 如果没有 Magma 库，则跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 如果没有 LAPACK 库，则跳过 CPU 测试
    @skipCPUIfNoLapack
    # 指定支持的浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 浮点数精度重写
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 定义测试函数，用于批量测试三角求解函数的多个情况
    def test_triangular_solve_batched_many_batches(self, device, dtype):
        # 使用 itertools.product 生成所有可能的组合 (upper, transpose, unitriangular)
        for upper, transpose, unitriangular in itertools.product([True, False], repeat=3):
            # 测试批量 A 的情况
            b, A = self.triangular_solve_test_helper((256, 256, 5, 5), (5, 1),
                                                     upper, unitriangular, device, dtype)
            # 使用 torch.triangular_solve 求解线性方程组
            x, _ = torch.triangular_solve(b, A,
                                          upper=upper, transpose=transpose, unitriangular=unitriangular)
            # 若 transpose 为 True，则对 A 进行转置
            if transpose:
                A = A.mT

            # 计算 Ax
            Ax = torch.matmul(A, x)

            # 根据数据类型设置相对误差容限
            rtol = 1e-2 if dtype in [torch.float32, torch.complex64] else self.precision
            # 使用 assertEqual 检查 Ax 与 b 的相等性
            self.assertEqual(Ax, b.expand_as(Ax), atol=self.precision, rtol=rtol)

            # 测试批量 b 的情况
            b, A = self.triangular_solve_test_helper((3, 3), (512, 512, 3, 1),
                                                     upper, unitriangular, device, dtype)
            # 使用 torch.triangular_solve 求解线性方程组
            x, _ = torch.triangular_solve(b, A, upper=upper, transpose=transpose,
                                          unitriangular=unitriangular)
            # 若 transpose 为 True，则对 A 进行转置
            if transpose:
                A = A.mT

            # 使用 assertEqual 检查 Ax 与 b 的相等性
            self.assertEqual(torch.matmul(A, x), b)

    # 根据条件选择性跳过 CUDA 没有 Magma 的测试
    @skipCUDAIfNoMagma
    # 根据条件选择性跳过 CPU 没有 Lapack 的测试
    @skipCPUIfNoLapack
    # 根据测试环境是否有 SciPy，选择性跳过测试
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    # 根据动态图模式是否开启，选择性跳过测试
    @skipIfTorchDynamo("flaky, needs investigation")
    # 使用装饰器指定支持的浮点和复数数据类型
    @dtypes(*floating_and_complex_types())
    # 定义批处理广播三角求解的测试函数，使用指定的设备和数据类型
    def test_triangular_solve_batched_broadcasting(self, device, dtype):
        # 导入 scipy 的 solve_triangular 函数
        from scipy.linalg import solve_triangular as tri_solve
        
        # 定义使用 scipy 进行批处理三角求解的函数
        def scipy_tri_solve_batched(A, B, upper, trans, diag):
            # 获取 A 和 B 的批处理维度和单个维度
            batch_dims_A, batch_dims_B = A.shape[:-2], B.shape[:-2]
            single_dim_A, single_dim_B = A.shape[-2:], B.shape[-2:]
            # 推断扩展维度，广播 A 和 B 到相同的形状
            expand_dims = tuple(torch._C._infer_size(torch.Size(batch_dims_A),
                                                     torch.Size(batch_dims_B)))
            expand_A = np.broadcast_to(A, expand_dims + single_dim_A)
            expand_B = np.broadcast_to(B, expand_dims + single_dim_B)
            # 将广播后的数组展平为二维数组
            flat_A = expand_A.reshape((-1,) + single_dim_A)
            flat_B = expand_B.reshape((-1,) + single_dim_B)
            # 对每对展平后的 A 和 B 执行 scipy 的三角求解，并垂直堆叠结果
            flat_X = np.vstack([tri_solve(a, b, lower=(not upper), trans=int(trans), unit_diagonal=diag)
                                for a, b in zip(flat_A, flat_B)])
            # 将结果重新形状为原始 B 的形状
            return flat_X.reshape(expand_B.shape)
        
        # 定义运行测试的函数，参数包括 A 和 b 的维度，设备类型，以及三角求解的参数
        def run_test(A_dims, b_dims, device, upper, transpose, unitriangular):
            # 使用测试辅助函数获取测试数据 A 和 b
            b, A = self.triangular_solve_test_helper(A_dims, b_dims, upper,
                                                     unitriangular, device, dtype)
            # 使用 scipy 的三角批处理求解作为期望结果
            x_exp = torch.as_tensor(scipy_tri_solve_batched(A.cpu().numpy(), b.cpu().numpy(),
                                                            upper, transpose, unitriangular))
            # 使用 PyTorch 的三角求解函数求解
            x = torch.triangular_solve(b, A, upper=upper, transpose=transpose, unitriangular=unitriangular)[0]
            # 断言 PyTorch 求解结果与 scipy 的期望结果一致
            self.assertEqual(x, x_exp.to(device))
        
        # 对 upper, transpose, unitriangular 参数的所有组合进行测试
        for upper, transpose, unitriangular in itertools.product([True, False], repeat=3):
            # 对不同的维度组合运行测试，没有广播
            run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6), device, upper, transpose, unitriangular)
            # 对不同的维度组合运行测试，b 被广播
            run_test((2, 1, 3, 4, 4), (4, 6), device, upper, transpose, unitriangular)
            # 对不同的维度组合运行测试，A 被广播
            run_test((4, 4), (2, 1, 3, 4, 2), device, upper, transpose, unitriangular)
            # 对不同的维度组合运行测试，A 和 b 被广播
            run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5), device, upper, transpose, unitriangular)

    # 仅在 CUDA 下运行的测试函数装饰器，使用单一浮点数据类型
    @onlyCUDA
    @dtypes(torch.float)
    def test_triangular_solve_large(self, device, dtype):
        # 重现 https://github.com/pytorch/pytorch/issues/79191 的问题
        # 创建随机矩阵 A，在设备上进行下三角化
        A = torch.randn(1, 2, 2, device=device, dtype=dtype).tril_()
        # 创建随机矩阵 B，在设备上进行求解
        B = torch.randn(1, 2, 524281, device=device, dtype=dtype)
        # 使用 PyTorch 的下三角求解函数求解方程 A @ X = B
        X = torch.linalg.solve_triangular(A, B, upper=False)
        # 断言解 X 满足方程 A @ X = B
        self.assertEqual(A @ X, B)

    # 如果没有 Magma 库，跳过 CUDA 测试的装饰器
    @skipCUDAIfNoMagma
    # 如果没有 LAPACK 库，跳过 CPU 测试的装饰器
    @skipCPUIfNoLapack
    # 支持所有浮点和复数数据类型
    @dtypes(*floating_and_complex_types())
    # 定义测试函数，用于验证 triangular_solve 函数的异常和警告处理
    def test_triangular_solve_out_errors_and_warnings(self, device, dtype):
        # dtypes should be safely castable
        # 创建单位矩阵 a，随机张量 b，以及空张量 out，dtype 为指定的数据类型，设备为指定的设备
        a = torch.eye(2, dtype=dtype, device=device)
        b = torch.randn(2, 1, dtype=dtype, device=device)
        out = torch.empty_like(b).to(torch.int)
        clone_a = torch.empty_like(a)
        # 预期引发 RuntimeError 异常，错误信息为 "Expected out tensor to have dtype"
        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            # 调用 triangular_solve 函数，指定 out 和 clone_a 作为输出
            torch.triangular_solve(b, a, out=(out, clone_a))

        # 重新初始化 out 和 clone_a
        out = torch.empty_like(b)
        clone_a = clone_a.to(torch.int)
        # 预期引发 RuntimeError 异常，错误信息为 "Expected out tensor to have dtype"
        with self.assertRaisesRegex(RuntimeError, "Expected out tensor to have dtype"):
            # 再次调用 triangular_solve 函数，指定 out 和 clone_a 作为输出
            torch.triangular_solve(b, a, out=(out, clone_a))

        # device should match
        # 如果 CUDA 可用，检查设备不匹配的情况
        if torch.cuda.is_available():
            # 根据当前设备类型确定错误的设备类型
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            # 初始化 out 和 clone_a，设备不匹配
            out = torch.empty(0, dtype=dtype, device=wrong_device)
            clone_a = torch.empty_like(a)
            # 预期引发 RuntimeError 异常，错误信息为 "tensors to be on the same device"
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                # 调用 triangular_solve 函数，指定 out 和 clone_a 作为输出
                torch.triangular_solve(b, a, out=(out, clone_a))
            # 重新初始化 out 和 clone_a，确保设备不匹配
            out = torch.empty(0, dtype=dtype, device=device)
            clone_a = torch.empty_like(a).to(wrong_device)
            # 预期引发 RuntimeError 异常，错误信息为 "tensors to be on the same device"
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                # 再次调用 triangular_solve 函数，指定 out 和 clone_a 作为输出
                torch.triangular_solve(b, a, out=(out, clone_a))

        # Trigger the WARN_ONCE deprecation error
        # 触发 WARN_ONCE 弃用警告
        torch.triangular_solve(b, a)

        # if out tensor with wrong shape is passed a warning is given
        # 如果传递了形状不正确的 out 张量，则会发出警告
        with warnings.catch_warnings(record=True) as w:
            # 初始化 out 和 clone_a，形状不正确
            out = torch.empty(1, dtype=dtype, device=device)
            clone_a = torch.empty(1, dtype=dtype, device=device)
            # 触发警告
            torch.triangular_solve(b, a, out=(out, clone_a))
            # 检查是否发出了警告
            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[0].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[1].message))
    def check_single_matmul(self, x, y):
        """
        Performs checks for matrix multiplication.

        Args:
            x (torch.Tensor): First matrix operand.
            y (torch.Tensor): Second matrix operand.
        """

        def assertEqual(answer, expected):
            """
            Helper function to assert equality between two matrices.

            Args:
                answer (torch.Tensor): The computed result.
                expected (torch.Tensor): The expected result.
            """
            if x.dtype.is_floating_point or x.dtype.is_complex:
                k = max(x.shape[-1], 1)  # Scale the atol with the size of the matrix
                self.assertEqual(answer, expected,
                                 msg=f"{x.shape} x {y.shape} = {answer.shape}",
                                 atol=k * 5e-5,
                                 rtol=1e-4)
            else:
                self.assertEqual(answer, expected, msg=f"{x.shape} x {y.shape} = {answer.shape}")

        # test x @ y
        expected = np.matmul(x.cpu(), y.cpu())  # Compute expected result using NumPy
        ans = torch.matmul(x, y)  # Compute result using PyTorch
        self.assertTrue(ans.is_contiguous())  # Ensure result tensor is contiguous
        assertEqual(ans, expected)  # Assert equality of computed and expected results

        # test out
        out = torch.empty_like(ans)  # Create an empty tensor like 'ans'
        ans = torch.matmul(x, y, out=out)  # Perform matrix multiplication with out tensor
        self.assertIs(ans, out)  # Assert that 'ans' is the same as 'out'
        self.assertTrue(ans.is_contiguous())  # Ensure result tensor is contiguous
        assertEqual(ans, expected)  # Assert equality of computed and expected results

    def gen_sizes_matmul(self, x_dim, y_dim=4, matrix_size=4, batch_size=3):
        """
        Generates compatible matrix dimensions for matrix multiplication.

        Args:
            x_dim (int): Dimension size for the first matrix.
            y_dim (int): Maximum dimension size for the second matrix (default is 4).
            matrix_size (int): Maximum size of matrices (default is 4).
            batch_size (int): Batch size for generating matrices (default is 3).

        Yields:
            tuple: Tuple of sizes (x_size, y_size) compatible for matrix multiplication.
        """
        assert x_dim >= 1
        assert y_dim >= 2
        x = x_dim
        for y in range(1, y_dim + 1):
            for batch, mn in product(product(range(batch_size), repeat=max(x - 2, y - 2, 0)),
                                     product(range(matrix_size), repeat=min(y, 2))):
                if x == 1:
                    size_x = mn[:1]
                    size_y = batch + mn
                    yield size_x, size_y
                else:
                    for k in range(matrix_size):
                        size_x = (k,) + mn[:1]
                        if x > 2:
                            size_x = batch[-(x - 2):] + size_x
                        size_y = mn
                        if y > 2:
                            size_y = batch[-(y - 2):] + size_y
                        yield size_x, size_y

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    @setBlasBackendsToDefaultFinally
    def test_matmul_small_brute_force_1d_Nd(self, device, dtype):
        """
        Tests matrix multiplication for small matrices using brute-force approach.

        Args:
            device (str): Device (CPU or CUDA) for running the test.
            dtype (torch.dtype): Data type for the tensors.
        """
        for backend in ["cublas", "cublaslt"]:
            if torch.device(device).type == 'cuda':
                torch.backends.cuda.preferred_blas_library(backend)

            make_arg = partial(make_tensor, device=device, dtype=dtype)

            for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(1), (True, False), (True, False)):
                x = make_arg(size_x, noncontiguous=nctg_x)
                y = make_arg(size_y, noncontiguous=nctg_y)
                self.check_single_matmul(x, y)

    @dtypesIfCUDA(torch.float, torch.complex64)  # Integer matmul just supported on CPU
    @dtypes(torch.int64, torch.float, torch.complex64)
    @setBlasBackendsToDefaultFinally
    # 在 CUDA 设备上测试二维张量的矩阵乘法，使用暴力方法
    def test_matmul_small_brute_force_2d_Nd(self, device, dtype):
        # 遍历不同的后端实现，如 cublas 和 cublaslt
        for backend in ["cublas", "cublaslt"]:
            # 如果当前设备是 CUDA 设备，则设置优选的 BLAS 库为指定的后端
            if torch.device(device).type == 'cuda':
                torch.backends.cuda.preferred_blas_library(backend)
    
            # 使用部分函数创建张量的函数 make_tensor，设置设备和数据类型
            make_arg = partial(make_tensor, device=device, dtype=dtype)
    
            # 遍历生成的二维矩阵大小及其非连续性的组合
            for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(2), (True, False), (True, False)):
                # 创建输入张量 x 和 y
                x = make_arg(size_x, noncontiguous=nctg_x)
                y = make_arg(size_y, noncontiguous=nctg_y)
                # 调用单独矩阵乘法检查函数
                self.check_single_matmul(x, y)
    
    @dtypesIfCUDA(torch.float, torch.complex64)  # 只有在 CPU 上支持整数矩阵乘法
    @dtypes(torch.int64, torch.float, torch.complex64)
    @setBlasBackendsToDefaultFinally
    # 在 CUDA 设备上测试三维及以上张量的矩阵乘法，使用暴力方法
    def test_matmul_small_brute_force_3d_Nd(self, device, dtype):
        # 遍历不同的后端实现，如 cublas 和 cublaslt
        for backend in ["cublas", "cublaslt"]:
            # 如果当前设备是 CUDA 设备，则设置优选的 BLAS 库为指定的后端
            if torch.device(device).type == 'cuda':
                torch.backends.cuda.preferred_blas_library(backend)
    
            # 使用部分函数创建张量的函数 make_tensor，设置设备和数据类型
            make_arg = partial(make_tensor, device=device, dtype=dtype)
    
            # 遍历生成的三维及以上张量大小及其非连续性的组合
            for (size_x, size_y), nctg_x, nctg_y in product(self.gen_sizes_matmul(3), (True, False), (True, False)):
                # 创建输入张量 x 和 y
                x = make_arg(size_x, noncontiguous=nctg_x)
                y = make_arg(size_y, noncontiguous=nctg_y)
                # 调用单独矩阵乘法检查函数
                self.check_single_matmul(x, y)
    
    @onlyCUDA
    @dtypes(*floating_types_and(torch.half))
    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.float)
    # 在 ROCm 环境下测试可调优操作的缓冲区旋转功能，默认开启，但会导致内存故障
    def test_bmm_tunableop_rocm(self, device, dtype):
        # 启用 CUDA 可调优功能
        torch.cuda.tunable.enable(True)
        # 获取当前 CUDA 设备的序数
        ordinal = torch.cuda.current_device()
        # 生成特定于设备序数的结果文件名
        filename = f"tunableop_results{ordinal}.csv"
        # 设置 CUDA 可调优结果文件名
        torch.cuda.tunable.set_filename(filename)
        # 获取当前 CUDA 可调优的最大迭代次数
        iterations = torch.cuda.tunable.get_max_tuning_iterations()
        # 设置 CUDA 可调优的最大迭代次数为 10 次
        torch.cuda.tunable.set_max_tuning_iterations(10)
        
        # 以下三个 case 涵盖了所有之前的失败案例，并用于捕捉回归错误
        # case 1
        B = 16
        N = M = K = 256
        # 指定张量数据类型为 torch.bfloat16
        dtype = torch.bfloat16
        # 指定设备为 CUDA 设备 0
        device = torch.device("cuda:0")
        # 生成随机张量 i1 和 i2，指定设备和数据类型，进行批量矩阵乘操作
        i1 = torch.randn((B, N, M), device=device, dtype=dtype)
        i2 = torch.randn((B, M, K), device=device, dtype=dtype)
        out = torch.bmm(i1, i2)
        
        # case 2
        # 重新生成随机张量 i1 和 i2，并对它们进行维度置换操作，再进行批量矩阵乘操作
        i1 = torch.randn((B, N, M), device=device, dtype=dtype)
        i1 = torch.permute(i1, (1, 2, 0))
        i2 = torch.randn((B, M, K), device=device, dtype=dtype)
        i2 = torch.permute(i2, (1, 0, 2))
        out = torch.bmm(i1, i2)
        
        # case 3
        # 重新生成随机张量 i1 和 i2，并对它们进行维度置换操作，再进行批量矩阵乘操作
        i1 = torch.randn((N, B, M), device=device, dtype=dtype)
        i1 = torch.permute(i1, (1, 0, 2))
        i2 = torch.randn((M, B, K), device=device, dtype=dtype)
        i2 = torch.permute(i2, (1, 2, 0))
        out = torch.bmm(i1, i2)
        
        # 清理操作，删除可能生成的任何文件
        try:
            import os
            os.remove(filename)
        except FileNotFoundError:
            # 如果文件不存在，则忽略异常
            pass
        
        # 恢复之前的 CUDA 可调优设置，包括最大迭代次数
        torch.cuda.tunable.set_max_tuning_iterations(iterations)
        # 禁用 CUDA 可调优功能
        torch.cuda.tunable.enable(False)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(torch.float)
    # 定义一个测试方法，用于检查数值计算操作在 ROCm 环境下的内存泄漏情况
    def test_numeric_check_leak_tunableop_rocm(self, device, dtype):
        # 导入必要的模块和类
        from torch.testing._internal.common_utils import CudaMemoryLeakCheck
        import os
        
        # 首先在不调整参数的情况下运行操作，确保所有 ROCm 库都被加载，
        # 否则可能会出现虚假的内存泄漏检测
        B = 16
        N = M = K = 256
        dtype = torch.bfloat16
        device = torch.device("cuda:0")
        
        # 创建输入张量 i1 和 i2，用随机数据填充
        i1 = torch.randn((B, N, M), device=device, dtype=dtype)
        i2 = torch.randn((B, M, K), device=device, dtype=dtype)
        
        # 执行矩阵乘法操作
        out = torch.bmm(i1, i2)
        
        # 设置环境变量以启用可调整操作的数值检查
        PYTORCH_TUNABLEOP_NUMERICAL_CHECK = "PYTORCH_TUNABLEOP_NUMERICAL_CHECK"
        prev_val = os.getenv(PYTORCH_TUNABLEOP_NUMERICAL_CHECK)
        try:
            os.environ[PYTORCH_TUNABLEOP_NUMERICAL_CHECK] = "1"
            # 启用可调整操作的数值检查
            torch.cuda.tunable.enable(True)
            
            # 获取当前设备的序号
            ordinal = torch.cuda.current_device()
            # 创建结果文件名
            filename = f"tunableop_results{ordinal}.csv"
            # 设置可调整操作的结果文件名
            torch.cuda.tunable.set_filename(filename)
            # 获取最大调整迭代次数
            iterations = torch.cuda.tunable.get_max_tuning_iterations()
            # 设置最大调整迭代次数为 10
            torch.cuda.tunable.set_max_tuning_iterations(10)
            
            # 使用 CudaMemoryLeakCheck 类来检查内存泄漏
            with CudaMemoryLeakCheck(self):
                # 再次执行矩阵乘法操作
                out = torch.bmm(i1, i2)
                
                # 恢复最大调整迭代次数设置
                torch.cuda.tunable.set_max_tuning_iterations(iterations)
                # 禁用可调整操作的数值检查
                torch.cuda.tunable.enable(False)
                
                # 清理操作，删除生成的文件
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass
        finally:
            # 恢复原来的环境变量设置
            if prev_val is None:
                del os.environ[PYTORCH_TUNABLEOP_NUMERICAL_CHECK]
            else:
                os.environ[PYTORCH_TUNABLEOP_NUMERICAL_CHECK] = prev_val


    # 使用指定的数据类型进行测试，验证矩阵乘法操作在自动微分中出现错误的情况
    @dtypes(torch.float, torch.complex64)
    def test_matmul_out_kernel_errors_with_autograd(self, device, dtype):
        # 创建具有梯度的空张量 a 和 b
        a = torch.empty((256, 512), device=device, dtype=dtype, requires_grad=True).unsqueeze(0)
        b = torch.empty((4, 128, 512), device=device, dtype=dtype, requires_grad=True).transpose(-1, -2)
        # 创建输出张量 c，移动维度以匹配乘法结果
        c = torch.empty((256, 4, 128), device=device, dtype=dtype).movedim(1, 0)
        
        # 使用 detach 方法执行矩阵乘法，避免计算梯度
        torch.matmul(a.detach(), b.detach(), out=c)
        
        # 使用 assertRaisesRegex 检测是否抛出预期的异常信息
        with self.assertRaisesRegex(RuntimeError, "functions with out=... arguments don't support automatic differentiation"):
            # 尝试对具有输出参数的乘法操作执行自动微分
            torch.matmul(a, b, out=c)
        
        # 使用 torch.no_grad 上下文管理器，禁用梯度计算
        with torch.no_grad():
            # 在无梯度计算的情况下执行乘法操作
            torch.matmul(a, b, out=c)

    # 使用大型张量进行测试，验证大规模矩阵乘法操作的反向传播
    # 由于测试会并行运行在 CI 中，因此分配大内存以防止内存不足
    @largeTensorTest('16GB', device='cuda')
    def test_large_bmm_mm_backward(self, device):
        # 创建随机张量 A，B 和 G，分别指定其设备和是否需要梯度
        A = torch.randn([1024, 2, 1024], device="cuda").mT.contiguous().mT
        B = torch.randn([1024, 65536], device="cuda", requires_grad=True)
        G = torch.randn([1024, 2, 65536], device="cuda")
        
        # 执行大规模矩阵乘法操作，并对结果执行反向传播
        (A @ B).backward(G)
    # 使用装饰器定义一个测试函数，标记为需要大量GPU内存（16GB），并指定设备为cuda
    @largeTensorTest('16GB', device='cuda')
    def test_large_bmm_backward(self, device):
        # 创建一个大小为 [1024, 2, 1024] 的张量 A，元素为随机数，存储在 CUDA 设备上，并进行转置操作
        A = torch.randn([1024, 2, 1024], device="cuda").mT.contiguous().mT
        # 创建一个大小为 [1, 1024, 65536] 的张量 B，元素为随机数，存储在 CUDA 设备上，并设置需要梯度
        B = torch.randn([1, 1024, 65536], device="cuda", requires_grad=True)
        # 创建一个大小为 [1024, 2, 65536] 的张量 G，元素为随机数，存储在 CUDA 设备上
        G = torch.randn([1024, 2, 65536], device="cuda")
    
        # 执行矩阵乘法 A @ B，并对结果进行反向传播，梯度为 G
        # 注意：这里的操作应避免创建大小为 [1024, 1024, 65536] 的中间张量（需要256GB内存），以避免内存溢出
        (A @ B).backward(G)
    
    # 定义一个测试函数，检验线性代数操作中的标量异常
    def test_linear_algebra_scalar_raises(self, device) -> None:
        # 创建一个大小为 [5, 5] 的随机张量 m，存储在指定设备上
        m = torch.randn(5, 5, device=device)
        # 创建一个大小为 [5] 的随机向量 v，存储在指定设备上
        v = torch.randn(5, device=device)
        # 创建一个值为 7 的标量张量 s，存储在指定设备上
        s = torch.tensor(7, device=device)
        # 检验 torch.mv 操作是否会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.mv(m, s))
        # 检验 torch.addmv 操作是否会引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.addmv(v, m, s))
    
    # 使用装饰器定义一个测试函数，指定数据类型为 torch.float32 和 torch.complex64
    @dtypes(torch.float32, torch.complex64)
    def test_cross(self, device, dtype):
        # 创建一个大小为 [100, 3, 100] 的随机张量 x，数据类型为指定类型，存储在指定设备上
        x = torch.rand(100, 3, 100, dtype=dtype, device=device)
        # 创建一个大小为 [100, 3, 100] 的随机张量 y，数据类型为指定类型，存储在指定设备上
        y = torch.rand(100, 3, 100, dtype=dtype, device=device)
        # 计算张量 x 和 y 的叉乘
        res1 = torch.cross(x, y)
        # 创建一个空张量 res2，数据类型为指定类型，存储在指定设备上
        res2 = torch.tensor((), dtype=dtype, device=device)
        # 将 x 和 y 的叉乘结果存储到 res2 中
        torch.cross(x, y, out=res2)
        # 断言 res1 和 res2 的结果是否相等
        self.assertEqual(res1, res2)
    
    # 使用装饰器定义一个测试函数，指定数据类型为 torch.float32 和 torch.complex64
    @dtypes(torch.float32, torch.complex64)
    def test_linalg_cross(self, device, dtype):
        # 创建一个大小为 [100, 3, 100] 的随机张量 x，数据类型为指定类型，存储在指定设备上
        x = torch.rand(100, 3, 100, dtype=dtype, device=device)
        # 创建一个大小为 [100, 3, 100] 的随机张量 y，数据类型为指定类型，存储在指定设备上
        y = torch.rand(100, 3, 100, dtype=dtype, device=device)
        # 计算张量 x 和 y 在维度 1 上的叉乘
        res1 = torch.linalg.cross(x, y, dim=1)
        # 创建一个空张量 res2，数据类型为指定类型，存储在指定设备上
        res2 = torch.tensor((), dtype=dtype, device=device)
        # 将 x 和 y 的维度 1 上的叉乘结果存储到 res2 中
        torch.linalg.cross(x, y, dim=1, out=res2)
        # 断言 res1 和 res2 的结果是否相等
        self.assertEqual(res1, res2)
    
        # 对可广播输入进行测试
        # 创建一个大小为 [1, 3, 2] 的随机张量 x，数据类型为指定类型，存储在指定设备上
        x = torch.rand(1, 3, 2, dtype=dtype, device=device)
        # 创建一个大小为 [4, 3, 1] 的随机张量 y，数据类型为指定类型，存储在指定设备上
        y = torch.rand(4, 3, 1, dtype=dtype, device=device)
        # 计算张量 x 和 y 在维度 1 上的叉乘
        res1 = torch.linalg.cross(x, y, dim=1)
        # 创建一个空张量 res2，数据类型为指定类型，存储在指定设备上
        res2 = torch.tensor((), dtype=dtype, device=device)
        # 将 x 和 y 的维度 1 上的叉乘结果存储到 res2 中
        torch.linalg.cross(x, y, dim=1, out=res2)
        # 断言 res1 和 res2 的结果是否相等
        self.assertEqual(res1, res2)
    
    # 使用装饰器定义一个测试函数，指定数据类型为 torch.float32 和 torch.complex64
    @dtypes(torch.float32, torch.complex64)
    def test_cross_with_and_without_dim(self, device, dtype):
        # 创建一个大小为 [100, 3] 的随机张量 x，数据类型为指定类型，存储在指定设备上
        x = torch.rand(100, 3, dtype=dtype, device=device)
        # 创建一个大小为 [100, 3] 的随机张量 y，数据类型为指定类型，存储在指定设备上
        y = torch.rand(100, 3, dtype=dtype, device=device)
        # 计算张量 x 和 y 在维度 1 上的叉乘
        res1 = torch.cross(x, y, dim=1)
        # 同样计算张量 x 和 y 在维度 -1 上的叉乘
        res2 = torch.cross(x, y, dim=-1)
        # 没有指定维度时，计算张量 x 和 y 的叉乘
        res3 = torch.cross(x, y)
        # 断言 res1 和 res2 的结果是否相等
        self.assertEqual(res1, res2)
        # 断言 res1 和 res3 的结果是否相等
        self.assertEqual(res1, res3)
    
    # 使用装饰器定义一个测试函数，指定数据类型为 torch.float32 和 torch.complex64
    @dtypes(torch.float32, torch.complex64)
    def test_linalg_cross_with_and_without_dim(self, device, dtype):
        # 创建一个大小为 [100, 3] 的随机张量 x，数据类型为指定类型，存储在指定设备上
        x = torch.rand(100, 3, dtype=dtype, device=device)
        # 创建一个大小为 [100, 3] 的随机张量 y，数据类型为指定类型，存储在指定设备上
        y = torch.rand(100, 3, dtype=dtype, device=device)
        # 计算张量 x 和 y 在维度 1 上的叉乘
        res1 = torch.linalg.cross(x, y, dim=1)
        # 同样计算张量 x 和 y 在维度 -1 上的叉乘
        res2 = torch.linalg.cross(x, y, dim=-1)
        # 没有指定维度时，计
    # 定义一个测试方法，用于测试 renorm 函数在给定设备上的行为
    def test_renorm(self, device):
        # 创建一个大小为 (20, 20) 的随机张量 m1，并指定设备
        m1 = torch.randn(20, 20, device=device)  # big enough to exercise vectorized path
        # 创建一个空张量 res1，并指定设备
        res1 = torch.tensor((), device=device)

        # 定义 renorm 函数，用于对给定矩阵进行重新归一化
        def renorm(matrix, value, dim, max_norm):
            # 将矩阵按指定维度 dim 进行转置，并确保内存连续性
            m1 = matrix.transpose(dim, 0).contiguous()
            # 将非 dim 维度压缩成一维
            m2 = m1.clone().resize_(m1.size(0), int(math.floor(m1.nelement() / m1.size(0))))
            # 计算 m2 指定维度上的范数 norms
            norms = m2.norm(value, 1, True)
            # 对 norms 进行截断处理
            new_norms = norms.clone()
            new_norms[torch.gt(norms, max_norm)] = max_norm
            new_norms.div_(norms.add_(1e-7))
            # 对 m1 进行重新归一化处理
            m1.mul_(new_norms.expand_as(m1))
            # 将重新归一化后的 m1 恢复原来的维度排列方式，并返回结果
            return m1.transpose(dim, 0)

        # 计算 m1 在第一维上的 L2 范数的均值作为 maxnorm
        maxnorm = m1.norm(2, 1).mean()
        # 使用 renorm 函数对 m1 进行重新归一化处理，得到 m2
        m2 = renorm(m1, 2, 1, maxnorm)
        # 直接在 m1 上原地进行 renorm 操作
        m1.renorm_(2, 1, maxnorm)
        # 使用 self.assertEqual 检查 m1 和 m2 在给定的误差范围内是否相等
        self.assertEqual(m1, m2, atol=1e-5, rtol=0)
        # 检查 m1 和 m2 在第零维上的 L2 范数是否相等
        self.assertEqual(m1.norm(2, 0), m2.norm(2, 0), atol=1e-5, rtol=0)

        # 创建一个大小为 (3, 4, 5) 的随机张量 m1，并指定设备
        m1 = torch.randn(3, 4, 5, device=device)
        # 将 m1 在第一维和第二维交换，并确保内存连续性，然后压缩成二维张量 m2
        m2 = m1.transpose(1, 2).contiguous().clone().resize_(15, 4)
        # 计算 m2 在第零维上的 L2 范数的均值作为 maxnorm
        maxnorm = m2.norm(2, 0).mean()
        # 使用 renorm 函数对 m2 进行重新归一化处理，得到 m2
        m2 = renorm(m2, 2, 1, maxnorm)
        # 在原地对 m1 进行 renorm 操作
        m1.renorm_(2, 1, maxnorm)
        # 将重新归一化后的 m1 按照第一维和第二维交换并确保内存连续性，然后压缩成二维张量 m3
        m3 = m1.transpose(1, 2).contiguous().clone().resize_(15, 4)
        # 使用 self.assertEqual 检查 m3 和 m2 是否相等
        self.assertEqual(m3, m2)
        # 检查 m3 在第零维上的 L2 范数和 m2 在第零维上的 L2 范数是否相等
        self.assertEqual(m3.norm(2, 0), m2.norm(2, 0))
    # 定义一个测试函数 test_ormqr，接受设备和数据类型作为参数
    def test_ormqr(self, device, dtype):

        # 定义内部函数 run_test，用于执行单个测试
        def run_test(batch, m, n, fortran_contiguous):
            # 创建一个大小为 (*batch, m, n) 的张量 A，并指定数据类型和设备
            A = make_tensor((*batch, m, n), dtype=dtype, device=device)
            # 对张量 A 进行 QR 分解，返回反射器和 tau 值
            reflectors, tau = torch.geqrf(A)
            
            # 如果不是 Fortran 连续存储，则要求 reflectors.mT 是连续的，并将其转换为连续的张量
            if not fortran_contiguous:
                self.assertTrue(reflectors.mT.is_contiguous())
                reflectors = reflectors.contiguous()

            # 计算完整的 QR 分解，得到 Q，其大小为 m x m
            Q, _ = torch.linalg.qr(A, mode='complete')

            # 创建大小为 (*batch, m, n) 的张量 C_right 和大小为 (*batch, n, m) 的张量 C_left
            C_right = make_tensor((*batch, m, n), dtype=dtype, device=device)
            C_left = make_tensor((*batch, n, m), dtype=dtype, device=device)

            # 计算预期结果 expected = Q @ C_right，并使用 torch.ormqr 计算实际结果 actual
            expected = Q @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=False)
            self.assertEqual(expected, actual)

            # 计算预期结果 expected = C_left @ Q，并使用 torch.ormqr 计算实际结果 actual
            expected = C_left @ Q
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=False)
            self.assertEqual(expected, actual)

            # 计算预期结果 expected = Q.mH @ C_right（Q 的共轭转置 @ C_right），并使用 torch.ormqr 计算实际结果 actual
            expected = Q.mH @ C_right
            actual = torch.ormqr(reflectors, tau, C_right, left=True, transpose=True)
            self.assertEqual(expected, actual)

            # 计算预期结果 expected = C_left @ Q.mH（C_left @ Q 的共轭转置），并使用 torch.ormqr 计算实际结果 actual
            expected = C_left @ Q.mH
            actual = torch.ormqr(reflectors, tau, C_left, left=False, transpose=True)
            self.assertEqual(expected, actual)

            # 如果 tau 全部为零，则隐式矩阵 Q 是单位矩阵，因此实际结果应为 C_right
            zero_tau = torch.zeros_like(tau)
            actual = torch.ormqr(reflectors, zero_tau, C_right, left=True, transpose=False)
            self.assertEqual(C_right, actual)

        # 定义不同的批次、大小组合以及是否 Fortran 连续存储的情况，使用 product 进行组合
        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        for batch, (m, n), fortran_contiguous in product(batches, product(ns, ns), [True, False]):
            run_test(batch, m, n, fortran_contiguous)

    # 使用装饰器 @skipCPUIfNoLapack，如果没有 Lapack 库则跳过测试
    @skipCPUIfNoLapack
    # 使用装饰器 @skipCUDAIfNoCusolver，如果没有 Cusolver 库则跳过 CUDA 测试
    @skipCUDAIfNoCusolver
    # 使用装饰器 @dtypes(*floating_and_complex_types())，指定浮点数和复数数据类型进行测试
    @dtypes(*floating_and_complex_types())
    # 测试 torch.ormqr 函数的错误和警告情况
    def test_ormqr_errors_and_warnings(self, device, dtype):
        # 定义测试用例列表，每个元素包含输入张量的大小、错误信息的正则表达式
        test_cases = [
            # input1 size, input2 size, input3 size, error regex
            ((10,), (2,), (2,), r"input must have at least 2 dimensions"),
            ((2, 2), (2,), (2,), r"other must have at least 2 dimensions"),
            ((10, 6), (20,), (10, 6), r"other.shape\[-2\] must be greater than or equal to tau.shape\[-1\]"),
            ((6, 6), (5,), (5, 5), r"other.shape\[-2\] must be equal to input.shape\[-2\]"),
            ((1, 2, 2), (2, 2), (1, 2, 2), r"batch dimensions of tau to be equal to input.shape\[:-2\]"),
            ((1, 2, 2), (1, 2), (2, 2, 2), r"batch dimensions of other to be equal to input.shape\[:-2\]"),
        ]
        # 对于每个测试用例，执行以下操作
        for a_size, tau_size, c_size, error_regex in test_cases:
            # 生成指定大小的张量 a, tau, c
            a = make_tensor(a_size, dtype=dtype, device=device)
            tau = make_tensor(tau_size, dtype=dtype, device=device)
            c = make_tensor(c_size, dtype=dtype, device=device)
            # 断言调用 torch.ormqr(a, tau, c) 时会抛出指定的异常消息
            with self.assertRaisesRegex(RuntimeError, error_regex):
                torch.ormqr(a, tau, c)

    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  torch.half,
                  *[torch.bfloat16] if SM53OrLater else []
                  ))
    @dtypes(*all_types_and_complex_and(torch.bfloat16))
    # 测试 cublasLtMatmul 函数的边界情况
    def test_corner_cases_of_cublasltmatmul(self, device, dtype):
        # 常见情况
        M = torch.randn(128, device=device).to(dtype)
        m1 = torch.randn(2048, 2400, device=device).to(dtype)
        m2 = torch.randn(128, 2400, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)
        
        # Ntrans_B 的 ld >> 行数
        m1 = torch.rand([128, 2400]).to(dtype).to(device).t()
        m2 = torch.rand([2048, 25272]).to(dtype).to(device).t()[21940:24340]
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2.t(), m1)
        
        # trans_A 的 ld >> 行数
        m1 = torch.rand([128, 25272]).to(dtype).to(device)[:, 21940:24340].t()
        m2 = torch.randn(2048, 2400, device=device).to(dtype)
        M = torch.rand([128]).to(dtype).to(device)
        torch.addmm(M, m2, m1)
        
        # 大型张量，维度 > 65535
        M = torch.randn(16, device=device).to(dtype)
        m1 = torch.randn(32, 131071 , device=device).to(dtype)
        m2 = torch.randn(16, 131071, device=device).to(dtype)
        torch.nn.functional.linear(m1, m2, M)

    @onlyCUDA
    @skipCUDAIfNotRocm
    @dtypes(*floating_types_and(torch.bfloat16, torch.half))
    # 定义一个测试函数，用于测试 hipBLASLt 在 ROCm 环境下的边界情况
    def test_hipblaslt_corner_cases_rocm(self, device, dtype):
        # 如果数据类型为双精度浮点数，跳过测试，因为 hipBLASLt 尚不支持双精度
        if dtype == torch.double:
            raise unittest.SkipTest("hipblasLt doesn't support doubles yet")

        # 启用 hipBLASLt 路径通过环境变量
        import os
        DISABLE_ADDMM_HIP_LT = "DISABLE_ADDMM_HIP_LT"
        # 获取之前的环境变量值以便恢复
        prev_val = os.getenv(DISABLE_ADDMM_HIP_LT)
        try:
            # 设置环境变量以启用 hipBLASLt 路径
            os.environ[DISABLE_ADDMM_HIP_LT] = "0"
            
            # 常见情况：生成设备上的随机张量 M、m1、m2
            M = torch.randn(128, device=device, dtype=dtype)
            m1 = torch.randn(2048, 2400, device=device, dtype=dtype)
            m2 = torch.randn(128, 2400, device=device, dtype=dtype)
            # 使用 functional.linear 计算线性操作并在设备上保存结果
            out1 = torch.nn.functional.linear(m1, m2, M)
            # 将结果张量转移到 CPU 上
            M_cpu = M.to('cpu')
            m1_cpu = m1.to('cpu')
            m2_cpu = m2.to('cpu')
            out1_cpu = torch.nn.functional.linear(m1_cpu, m2_cpu, M_cpu)
            # 断言两个结果张量在 CPU 上的接近性
            self.assertTrue(torch.allclose(out1_cpu, out1.cpu(), rtol=1e-2, atol=1e-2))

            # 常见情况，不使用偏置：生成设备上的随机张量 m1、m2
            m1 = torch.randn(2048, 2400, device=device, dtype=dtype)
            m2 = torch.randn(128, 2400, device=device, dtype=dtype)
            # 使用 functional.linear 计算线性操作并在设备上保存结果，不使用偏置
            out2 = torch.nn.functional.linear(m1, m2, bias=None)
            # 将结果张量转移到 CPU 上
            m1_cpu = m1.to('cpu')
            m2_cpu = m2.to('cpu')
            out2_cpu = torch.nn.functional.linear(m1_cpu, m2_cpu, bias=None)
            # 断言两个结果张量在 CPU 上的接近性
            self.assertTrue(torch.allclose(out2_cpu, out2.cpu(), rtol=1e-2, atol=1e-2))
        finally:
            # 恢复之前的环境变量值或删除设置的环境变量
            if prev_val is None:
                del os.environ[DISABLE_ADDMM_HIP_LT]
            else:
                os.environ[DISABLE_ADDMM_HIP_LT] = prev_val

    @dtypesIfCUDA(*floating_and_complex_types_and(
                  torch.half,
                  *[torch.bfloat16] if SM53OrLater else []
                  ))
    @dtypes(*all_types_and_complex_and(torch.bfloat16, torch.half))
    def test_blas_alpha_beta_empty(self, device, dtype):
        # This test is disabled on CUDA 9 due to:
        # See: https://github.com/pytorch/pytorch/issues/31006
        # 检查是否使用了 bfloat16 类型且设备类型为 'xla'，如果是，则跳过测试
        if dtype is torch.bfloat16 and self.device_type == 'xla':
            # TODO (@zasdfgbnm): this causes the following error on test
            # TestTorchDeviceTypeXLA.test_blas_alpha_beta_empty_xla_bfloat16:
            #
            #   RuntimeError: _th_equal not supported on CPUType for BFloat16
            # 如果条件满足，则直接返回，跳过测试
            return
        
        # ensure beta is respected
        # 确保 beta 参数被正确应用

        # Set up input tensors
        value = 11
        input = torch.full((2,), value, dtype=dtype, device=device)
        # 创建全为指定值的张量 input
        mat = torch.ones((2, 0), dtype=dtype, device=device)
        # 创建全为 1 的空矩阵 mat
        vec = torch.ones((0,), dtype=dtype, device=device)
        # 创建全为 1 的空向量 vec
        out = torch.empty((2,), dtype=dtype, device=device)
        # 创建空的输出张量 out

        # Determine alpha and beta based on dtype
        # 根据数据类型确定 alpha 和 beta 的值
        if dtype.is_complex:
            alpha = 6 + 7j
            beta = 3 + 4j
        else:
            alpha = 6
            beta = 3

        # Test torch.addmv function
        # 测试 torch.addmv 函数
        self.assertEqual(torch.full((2,), beta * value, dtype=dtype, device=device),
                         torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta))
        # 使用 torch.addmv 函数计算结果并进行断言验证

        self.assertEqual(torch.full((2,), beta * value, dtype=dtype, device=device),
                         torch.addmv(input=input, mat=mat, vec=vec, alpha=alpha, beta=beta, out=out))
        # 使用 torch.addmv 函数计算结果并进行断言验证，输出到预先定义的 out 张量

        # Test torch.addmm function
        # 测试 torch.addmm 函数
        input = torch.full((2, 3), value, dtype=dtype, device=device)
        # 重新设置 input 张量
        mat2 = torch.ones((0, 3), dtype=dtype, device=device)
        # 创建全为 1 的空矩阵 mat2
        out = torch.empty((2, 3), dtype=dtype, device=device)
        # 创建空的输出张量 out

        self.assertEqual(torch.full((2, 3), beta * value, dtype=dtype, device=device),
                         torch.addmm(input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta))
        # 使用 torch.addmm 函数计算结果并进行断言验证

        self.assertEqual(torch.full((2, 3), beta * value, dtype=dtype, device=device),
                         torch.addmm(input=input, mat1=mat, mat2=mat2, alpha=alpha, beta=beta, out=out))
        # 使用 torch.addmm 函数计算结果并进行断言验证，输出到预先定义的 out 张量
    # 定义测试函数 `test_blas_nan_out`，用于测试与 NaN 填充输出相关的功能
    def test_blas_nan_out(self, device, dtype):
        # 这些函数应正确处理 NaN 填充的输出，但需要特殊处理，参见 [NOTE: cpu_zero]

        # 设置变量 b, n, m, p 的值
        b = 3
        n = 5
        m = 7
        p = 11

        # 测试 torch.mv 函数
        nm = torch.randn((m, n), device=device).t()  # 创建随机张量 nm，大小为 (n, m)
        _m = torch.randn((), device=device).expand(m)  # 创建随机张量 _m，大小为 (m,)
        _m_out = torch.full((m,), float('nan'), device=device)  # 创建全为 NaN 的张量 _m_out，大小为 (m,)
        self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))  # 断言两次 torch.mv 的结果相等
        self.assertEqual(0, torch.isnan(torch.mv(nm, _m)).sum())  # 断言 torch.mv 结果中没有 NaN

        # 测试 torch.mm 函数
        mp = torch.randn((p, m), device=device).t()  # 创建随机张量 mp，大小为 (m, p)
        np_out = torch.full((n, p), float('nan'), device=device)  # 创建全为 NaN 的张量 np_out，大小为 (n, p)
        self.assertEqual(torch.mm(nm, mp), torch.mm(nm, mp, out=np_out))  # 断言两次 torch.mm 的结果相等

        # 测试 torch.bmm 函数
        bnm = torch.randn((b, m, n), device=device).transpose(1, 2)  # 创建随机张量 bnm，大小为 (b, n, m)
        bmp = torch.randn((b, p, m), device=device).transpose(1, 2)  # 创建随机张量 bmp，大小为 (b, m, p)
        bnp_out = torch.full((b, n, p), float('nan'), device=device)  # 创建全为 NaN 的张量 bnp_out，大小为 (b, n, p)
        self.assertEqual(torch.bmm(bnm, bmp), torch.bmm(bnm, bmp, out=bnp_out))  # 断言两次 torch.bmm 的结果相等

    @onlyCPU  # 标记测试函数仅适用于 CPU，不支持 CUBLAS
    def test_blas_mv_large_input(self, device):
        # 这些代码曾因为分配的输出中包含 NaN 而导致失败，参见：
        # https://github.com/pytorch/pytorch/issues/31663 和 [NOTE: cpu_zero]

        n = 3000  # 设置变量 n 的值
        m = 200  # 设置变量 m 的值

        nm = torch.randn((m, n), device=device).t()  # 创建随机张量 nm，大小为 (n, m)
        _m = torch.randn((), device=device).expand(m)  # 创建随机张量 _m，大小为 (m,)
        _m_out = torch.full((m,), 0., device=device)  # 创建全为 0 的张量 _m_out，大小为 (m,)

        self.assertEqual(torch.mv(nm, _m), torch.mv(nm, _m, out=_m_out))  # 断言两次 torch.mv 的结果相等

    @onlyCPU  # 标记测试函数仅适用于 CPU
    def test_renorm_ps(self, device):
        # 对完整的张量进行重新归一化

        x = torch.randn(5, 5)  # 创建大小为 (5, 5) 的随机张量 x
        xn = x.numpy()  # 将张量 x 转换为 NumPy 数组
        for p in [1, 2, 3, 4, inf]:  # 迭代计算不同的 p 范数
            res = x.renorm(p, 1, 1)  # 对张量 x 进行 p 范数的重新归一化处理
            expected = x / x.norm(p, 0, keepdim=True).clamp(min=1)  # 计算预期的重新归一化结果
            self.assertEqual(res, expected, msg=f"renorm failed for {p}-norm")  # 断言重新归一化结果与预期结果相等

    @skipCPUIfNoLapack  # 如果没有 Lapack 支持，则跳过测试
    @skipCUDAIfNoCusolver  # 如果没有 Cusolver 支持，则跳过测试
    @dtypes(*floating_and_complex_types())  # 根据浮点数和复数类型执行测试
    # 定义一个测试函数，用于测试 householder_product 方法，接受设备和数据类型作为参数
    def test_householder_product(self, device, dtype):
        # 定义一个内部函数，生成反射器和 tau 值
        def generate_reflectors_and_tau(A):
            """
            This function uses numpy.linalg.qr with mode "raw" to extract output of LAPACK's geqrf.
            There is torch.geqrf function but it doesn't work with complex-valued input.
            """
            # 如果 A 的元素数大于0，则处理
            if A.numel() > 0:
                # 将 A 转移到 CPU 上
                A_cpu = A.cpu()
                # 获取 A_cpu 的形状
                flattened_batch_shape = [-1, *A_cpu.shape[-2:]]
                # 创建一个与 A_cpu 形状相同的空张量 reflectors
                reflectors = torch.empty_like(A_cpu).view(*flattened_batch_shape)
                # 创建一个与 A_cpu 形状相同的空 tau 张量
                tau_shape = [*A_cpu.shape[:-2], A_cpu.shape[-1]]
                tau = torch.empty(tau_shape, dtype=dtype).view(-1, A_cpu.shape[-1])
                # 遍历 A_cpu，每次迭代中执行 numpy.linalg.qr 获取 reflectors_tmp 和 tau_i
                for A_i, reflectors_i, tau_i in zip(A_cpu.contiguous().view(*flattened_batch_shape), reflectors, tau):
                    reflectors_tmp, tau_i[:] = map(torch.from_numpy, np.linalg.qr(A_i, mode='raw'))
                    reflectors_i[:] = reflectors_tmp.T
                # 将 reflectors 和 tau 重塑为原始形状
                reflectors = reflectors.view(*A_cpu.shape)
                tau = tau.view(tau_shape)
                # 将结果移到指定设备上并返回
                return reflectors.to(A.device), tau.to(A.device)

            # 如果 A 的元素数为0，则返回与 A 形状相同的空 reflectors 和 tau 张量
            reflectors = torch.empty_like(A)
            tau = torch.empty(*A.shape[:-2], A.shape[-1], dtype=dtype, device=device)
            return reflectors, tau

        # 定义一个运行测试的函数，接受 shape 作为参数
        def run_test(shape):
            # 生成一个形状为 shape 的随机张量 A，指定设备和数据类型
            A = torch.randn(*shape, dtype=dtype, device=device)
            # 调用 generate_reflectors_and_tau 函数生成 reflectors 和 tau
            reflectors, tau = generate_reflectors_and_tau(A)
            # 使用 torch.linalg.qr 计算 A 的期望值和不使用的 _ 变量
            expected, _ = torch.linalg.qr(A)
            # 使用 torch.linalg.householder_product 计算 reflectors 和 tau 的实际值
            actual = torch.linalg.householder_product(reflectors, tau)
            # 如果 A 的元素数大于0，则使用断言验证 expected 和 actual 是否相等
            if (A.numel() > 0):
                self.assertEqual(expected, actual)
            else:
                # 否则验证 actual 的形状是否与 shape 相同
                self.assertTrue(actual.shape == shape)

            # 如果 A 的元素数大于0，则验证 tau 为空且 A 不是的情况下，actual 是否为对角线上为1的单位矩阵
            if (A.numel() > 0):
                tau_empty = torch.empty(*shape[:-2], 0, dtype=dtype, device=device)
                identity_mat = torch.zeros_like(reflectors)
                identity_mat.diagonal(dim1=-1, dim2=-2)[:] = 1
                actual = torch.linalg.householder_product(reflectors, tau_empty)
                self.assertEqual(actual, identity_mat)

            # 创建一个与 A 形状相同的空张量 out，并使用 torch.linalg.householder_product 计算结果 ans，并验证 ans 是否等于 out
            out = torch.empty_like(A)
            ans = torch.linalg.householder_product(reflectors, tau, out=out)
            self.assertEqual(ans, out)
            # 如果 A 的元素数大于0，则验证 expected 和 out 是否相等
            if (A.numel() > 0):
                self.assertEqual(expected, out)

        # 定义不同形状的测试用例列表 shapes
        shapes = [(0, 0), (5, 0),  # Empty matrix
                  (5, 5), (5, 3),  # Single matrix
                  (0, 0, 0), (0, 5, 5), (0, 5, 3),  # Zero batch dimension tensors
                  (2, 5, 5), (2, 5, 3),  # 3-dim tensors
                  (2, 1, 5, 5), (2, 1, 5, 3)]  # 4-dim tensors
        # 遍历 shapes 列表，并对每个 shape 运行 run_test 函数
        for shape in shapes:
            run_test(shape)

    # 设置装饰器，如果没有 Lapack 库则跳过测试
    @skipCPUIfNoLapack
    # 设置装饰器，如果没有 Cusolver 库则跳过测试
    @skipCUDAIfNoCusolver
    # 定义测试函数，用于测试 householder_product 函数的错误和警告情况
    def test_householder_product_errors_and_warnings(self, device):
        # 定义测试用例列表，每个元素包含输入张量的大小、tau 张量的大小以及预期的错误正则表达式
        test_cases = [
            ((10,), (2,), r"input must have at least 2 dimensions"),  # 输入张量至少需要 2 维
            ((10, 6), (20,), r"input.shape\[-1\] must be greater than or equal to tau.shape\[-1\]"),  # 输入张量的最后一维应大于等于 tau 张量的最后一维
            ((6, 10), (5,), r"input.shape\[-2\] must be greater than or equal to input.shape\[-1\]")  # 输入张量的倒数第二维应大于等于最后一维
        ]
        # 遍历测试用例
        for a_size, tau_size, error_regex in test_cases:
            # 创建随机张量 a 和 tau，根据设备指定设备
            a = torch.rand(*a_size, device=device)
            tau = torch.rand(*tau_size, device=device)
            # 使用 assertRaisesRegex 检查运行时错误，并验证错误信息符合预期的正则表达式
            with self.assertRaisesRegex(RuntimeError, error_regex):
                torch.linalg.householder_product(a, tau)

        # 如果传递了形状不正确的输出张量，会发出警告
        reflectors = torch.randn(3, 3, device=device)
        tau = torch.randn(3, device=device)
        out = torch.empty(2, 3, device=device)
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.linalg.householder_product(reflectors, tau, out=out)
            # 检查是否发出了警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 数据类型应能够安全转换
        out = torch.empty_like(reflectors).to(torch.int)
        # 使用 assertRaisesRegex 检查运行时错误，并验证错误信息符合预期的字符串
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.linalg.householder_product(reflectors, tau, out=out)

        # 检查 tau 张量数据类型与输入张量不匹配时的运行时错误
        with self.assertRaisesRegex(RuntimeError, "tau dtype Int does not match input dtype"):
            torch.linalg.householder_product(reflectors, tau.to(torch.int))

        # 如果 CUDA 可用，验证输出张量和输入张量应在相同设备上
        if torch.cuda.is_available():
            # 获取错误设备（与当前设备不同）
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty_like(reflectors).to(wrong_device)
            # 使用 assertRaisesRegex 检查运行时错误，并验证错误信息符合预期的字符串
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.linalg.householder_product(reflectors, tau, out=out)

            # 将 tau 张量转换到错误设备，验证是否抛出预期的运行时错误
            tau = tau.to(wrong_device)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.linalg.householder_product(reflectors, tau)
    # 定义一个测试函数，用于测试线性代数中的 LU 分解求解方程组功能，接受设备和数据类型参数
    def test_linalg_lu_solve(self, device, dtype):
        # 创建一个偏函数 make_arg，用于生成指定设备和数据类型的张量
        make_arg = partial(make_tensor, dtype=dtype, device=device)

        # 初始化支持的计算后端列表，至少包含默认后端
        backends = ["default"]

        # 如果设备是 CUDA，检查是否支持 MAGMA 后端，并添加到后端列表中
        if torch.device(device).type == 'cuda':
            if torch.cuda.has_magma:
                backends.append("magma")
            # 如果支持 CUSOLVER，同样添加到后端列表中
            if has_cusolver():
                backends.append("cusolver")

        # 定义生成测试矩阵对的生成器函数
        def gen_matrices():
            # 右手边向量的维度
            rhs = 3
            # 矩阵的维度组合，用于生成不同大小的矩阵对
            ns = (5, 2, 0)
            # 矩阵的批处理组合，用于生成不同批次的矩阵对
            batches = ((), (0,), (1,), (2,), (2, 1), (0, 2))
            for batch, n in product(batches, ns):
                yield make_arg(batch + (n, n)), make_arg(batch + (n, rhs))
            # 特定形状的矩阵对，用于覆盖所有可能的路径
            shapes = ((1, 64), (2, 128), (1025, 2))
            for b, n in shapes:
                yield make_arg((b, n, n)), make_arg((b, n, rhs))

        # 遍历所有生成的矩阵对
        for A, B in gen_matrices():
            # 对 A 进行 LU 分解，得到 LU 分解矩阵和置换向量
            LU, pivots = torch.linalg.lu_factor(A)
            # 遍历后端列表中的每个后端
            for backend in backends:
                # 设置 CUDA 首选的线性代数库为当前遍历的后端
                torch.backends.cuda.preferred_linalg_library(backend)

                # 遍历 left 和 adjoint 的所有组合
                for left, adjoint in product((True, False), repeat=2):
                    # 根据 left 的值选择 B 或其转置作为操作的右手边向量
                    B_left = B if left else B.mT
                    # 使用 LU 分解求解方程组，得到解 X
                    X = torch.linalg.lu_solve(LU, pivots, B_left, left=left, adjoint=adjoint)
                    # 根据 adjoint 的值选择 A 的共轭转置或转置作为操作的矩阵
                    A_adj = A.mH if adjoint else A
                    # 检查解 X 是否满足方程的要求
                    if left:
                        self.assertEqual(B_left, A_adj @ X)
                    else:
                        self.assertEqual(B_left, X @ A_adj)

    # 以下装饰器用于指定仅在 CPU 上运行的测试，并设置允许的数据类型
    @onlyCPU
    @dtypes(*floating_and_complex_types())
    # 定义一个测试函数，用于检查在 CPU 上执行 LU 分解时的异常情况
    def test_linalg_lu_cpu_errors(self, device, dtype):
        # Square tests
        # 创建一个随机张量 sample，形状为 (3, 2, 2)，指定设备和数据类型
        sample = torch.randn(3, 2, 2, device=device, dtype=dtype)
        # 创建一个随机张量 B，形状与 sample 相同，指定设备和数据类型
        B = torch.randn(3, 2, 2, device=device, dtype=dtype)
        # 对 sample 执行 LU 分解，LU 是分解后的矩阵，pivots 是置换向量
        LU, pivots = torch.linalg.lu_factor(sample)

        # This should run without issues
        # 运行 LU 解算，确保没有问题
        torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
        # 解包 LU 分解结果，返回原始矩阵 L、U 和置换向量 pivots
        torch.lu_unpack(LU, pivots)

        # 模拟错误情况：将第一个置换元素设置为 0
        pivots[0] = 0
        # 检查是否抛出 RuntimeError，并验证错误消息中包含 "greater or equal to 1"
        with self.assertRaisesRegex(RuntimeError, r"greater or equal to 1"):
            torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
        # 再次解包 LU 分解结果，应该抛出 RuntimeError，错误消息中包含 "between 1 and LU.size(-2)."
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)"):
            torch.lu_unpack(LU, pivots)

        # 模拟错误情况：将第一个置换元素设置为 3
        pivots[0] = 3
        # 检查是否抛出 RuntimeError，并验证错误消息中包含 "smaller or equal to LU.size(-2)"
        with self.assertRaisesRegex(RuntimeError, r"smaller or equal to LU.size\(-2\)"):
            torch.linalg.lu_solve(LU, pivots, B, adjoint=True)
        # 再次解包 LU 分解结果，应该抛出 RuntimeError，错误消息中包含 "between 1 and LU.size(-2)."
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)"):
            torch.lu_unpack(LU, pivots)

        # Rectangular tests
        # 创建一个随机张量 sample，形状为 (3, 4, 2)，指定设备和数据类型
        sample = torch.randn(3, 4, 2, device=device, dtype=dtype)
        # 创建一个随机张量 B，形状与 sample 相同，指定设备和数据类型
        B = torch.randn(3, 4, 2, device=device, dtype=dtype)
        # 对 sample 执行 LU 分解，LU 是分解后的矩阵，pivots 是置换向量
        LU, pivots = torch.linalg.lu_factor(sample)

        # This should run without issues
        # 解包 LU 分解结果，返回原始矩阵 L、U 和置换向量 pivots
        torch.lu_unpack(LU, pivots)

        # 模拟错误情况：将第一个置换元素设置为 0
        pivots[0] = 0
        # 再次解包 LU 分解结果，应该抛出 RuntimeError，错误消息中包含 "between 1 and LU.size(-2)."
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)"):
            torch.lu_unpack(LU, pivots)

        # 模拟错误情况：将第一个置换元素设置为 5
        pivots[0] = 5
        # 再次解包 LU 分解结果，应该抛出 RuntimeError，错误消息中包含 "between 1 and LU.size(-2)."
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)"):
            torch.lu_unpack(LU, pivots)


        # Rectangular tests
        # 创建一个随机张量 sample，形状为 (2, 3, 5)，指定设备和数据类型
        sample = torch.randn(2, 3, 5, device=device, dtype=dtype)
        # 创建一个随机张量 B，形状与 sample 相同，指定设备和数据类型
        B = torch.randn(2, 3, 5, device=device, dtype=dtype)
        # 对 sample 执行 LU 分解，LU 是分解后的矩阵，pivots 是置换向量
        LU, pivots = torch.linalg.lu_factor(sample)

        # This should run without issues
        # 解包 LU 分解结果，返回原始矩阵 L、U 和置换向量 pivots
        torch.lu_unpack(LU, pivots)

        # 模拟错误情况：将第一个置换元素设置为 0
        pivots[0] = 0
        # 再次解包 LU 分解结果，应该抛出 RuntimeError，错误消息中包含 "between 1 and LU.size(-2)."
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)"):
            torch.lu_unpack(LU, pivots)

        # 模拟错误情况：将第一个置换元素设置为 4
        pivots[0] = 4
        # 再次解包 LU 分解结果，应该抛出 RuntimeError，错误消息中包含 "between 1 and LU.size(-2)."
        with self.assertRaisesRegex(RuntimeError, r"between 1 and LU.size\(-2\)"):
            torch.lu_unpack(LU, pivots)


    @skipCPUIfNoLapack
    @skipCUDAIfNoMagma
    @dtypes(torch.double)
    # 定义一个测试函数，用于检查 lu_unpack 函数的输入参数是否符合预期
    def test_lu_unpack_check_input(self, device, dtype):
        # 创建一个随机张量 x，形状为 (5, 5, 5)，指定设备和数据类型
        x = torch.rand(5, 5, 5, device=device, dtype=dtype)
        # 对 x 执行 LU 分解，lu_data 是分解后的矩阵，lu_pivots 是置换向量
        lu_data, lu_pivots = torch.linalg.lu_factor(x)

        # 检查是否抛出 RuntimeError，并验证错误消息中包含 "torch.int32 dtype"
        with self.assertRaisesRegex(RuntimeError, "torch.int32 dtype"):
            torch.lu_unpack(lu_data, lu_pivots.long())

        # 检查当解包标志未设置时，返回的结果是否为 None
        p, l, u = torch.lu_unpack(lu_data, lu_pivots, unpack_data=False)
        # 验证是否返回的 l 和 u 张量的元素数为 0
        self.assertTrue(l.numel() == 0 and u.numel() == 0)
        p, l, u = torch.lu_unpack(lu_data, lu_pivots, unpack_pivots=False)
        # 验证是否返回的 p 张量的元素数为 0
        self.assertTrue(p.numel() == 0)
        p, l, u = torch.lu_unpack(lu_data, lu_pivots, unpack_data=False, unpack_pivots=False)
        # 验证是否返回的 p、l 和 u 张量的元素数均为 0
        self.assertTrue(p.numel() == 0 and l.numel() == 0 and u.numel() == 0)
    # 跳过没有Magma的CUDA设备的测试
    @skipCUDAIfNoMagma
    # 跳过没有Lapack的CPU设备的测试
    @skipCPUIfNoLapack
    # 指定测试使用的数据类型为双精度浮点数
    @dtypes(torch.double)
    # 测试 lobpcg_basic 方法
    def test_lobpcg_basic(self, device, dtype):
        # 调用 _test_lobpcg_method 方法，测试 basic 模式下的 lobpcg 方法
        self._test_lobpcg_method(device, dtype, 'basic')

    # 跳过没有Cusolver的CUDA设备的测试
    @skipCUDAIfNoCusolver
    # 跳过没有Lapack的CPU设备的测试
    @skipCPUIfNoLapack
    # 指定测试使用的数据类型为双精度浮点数
    @dtypes(torch.double)
    # 测试 lobpcg_ortho 方法
    def test_lobpcg_ortho(self, device, dtype):
        # 如果是 HIP 版本的 Torch，则设置 CUDA 预选线性代数库为 magma
        if torch.version.hip:
            torch.backends.cuda.preferred_linalg_library('magma')
        # 调用 _test_lobpcg_method 方法，测试 ortho 模式下的 lobpcg 方法
        self._test_lobpcg_method(device, dtype, 'ortho')
        # 如果是 HIP 版本的 Torch，则恢复 CUDA 默认的线性代数库设置
        if torch.version.hip:
            torch.backends.cuda.preferred_linalg_library('default')

    # 跳过没有Lapack的CPU设备的测试
    @skipCPUIfNoLapack
    # 仅在 CPU 上执行测试
    @onlyCPU
    # 指定测试使用的数据类型为双精度浮点数
    @dtypes(torch.double)
    # 测试 lobpcg_torchscript 方法
    def test_lobpcg_torchscript(self, device, dtype):
        # 导入随机生成稀疏正定矩阵的函数和矩阵乘法函数
        from torch.testing._internal.common_utils import random_sparse_pd_matrix
        from torch._linalg_utils import matmul as mm

        # 将 torch.lobpcg 方法编译为 TorchScript
        lobpcg = torch.jit.script(torch.lobpcg)

        # 设置矩阵 A1 的大小和稀疏度，生成随机稀疏正定矩阵
        m = 500
        k = 5
        A1 = random_sparse_pd_matrix(m, density=2.0 / m, device=device, dtype=dtype)
        # 生成随机的大小为 (m, k) 的矩阵 X1
        X1 = torch.randn((m, k), dtype=dtype, device=device)
        # 调用 lobpcg 方法求解特征值和特征向量
        E1, V1 = lobpcg(A1, X=X1)
        # 计算误差的相对二范数
        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()
        # 断言误差小于 1e-6
        self.assertLess(eq_err, 1e-6)

    # 如果没有安装 Scipy 或者安装的 Scipy 版本低于 1.4.1，则跳过测试
    @unittest.skipIf(not TEST_SCIPY or (TEST_SCIPY and scipy.__version__ < '1.4.1'), "Scipy not found or older than 1.4.1")
    # 跳过没有Lapack的CPU设备的测试
    @skipCPUIfNoLapack
    # 在 Torch Dynamo 中跳过测试，因为在追踪 scipy.sparse.lobpcg 时失败
    @skipIfTorchDynamo("fails in tracing scipy.sparse.lobpcg")
    # 仅在 CPU 上执行测试
    @onlyCPU
    # 指定测试使用的数据类型为双精度浮点数
    @dtypes(torch.double)
        # CPU timings: torch.lobpcg vs scipy.sparse.linalg.lobpcg
        # -------------------------------------------------------
        #               | standard    | generalized | method
        # torch.lobpcg  | {elapsed_ortho_ms:10.2f}  | {elapsed_ortho_general_ms:10.2f}  | ortho
        # scipy_lobpcg  | {elapsed_scipy_ms:10.2f}  | {elapsed_general_scipy_ms:10.2f}  | N/A
        # -(input size: {m:4}, eigenpairs:{k:2}, units: ms per call)-

        # Handling of very small tolerance
        tol = 1e-100

        # Initialize an empty list to store eigenvalues from torch.lobpcg
        lambdas1 = []

        # Define a function tracker to track eigenvalues during computation
        def tracker(worker):
            lambdas1.append(worker.E[:])

        # Perform computation using torch.lobpcg for standard eigenvalue problem
        E1, V1 = torch.lobpcg(A1, X=X1, niter=niter, largest=True, tracker=tracker, tol=tol)
        # Count the number of iterations performed
        iters1 = len(lambdas1)
        # Calculate the equivalence error for the solution
        eq_err = torch.norm((mm(A1, V1) - V1 * E1), 2) / E1.max()

        # Try performing computation using scipy.sparse.linalg.lobpcg for standard problem
        try:
            E2, V2, lambdas2 = scipy_lobpcg(A2, X2, maxiter=niter, largest=True, retLambdaHistory=True, tol=tol)
            # Count the number of iterations performed
            iters2 = len(lambdas2)
            # Calculate the equivalence error for the scipy solution
            eq_err_scipy = (abs(A2.dot(V2) - V2 * E2)**2).sum() ** 0.5 / E2.max()
        except Exception as msg:
            # Handle any exceptions raised during scipy computation
            print('Calling scipy_lobpcg failed [standard]:', msg)
            iters2 = -1
            eq_err_scipy = -1

        # Reset lambdas1 list for generalized eigenvalue computation
        lambdas1 = []

        # Define a function tracker to track eigenvalues during generalized eigenvalue computation
        def tracker(worker):
            lambdas1.append(worker.E[:])

        # Perform computation using torch.lobpcg for generalized eigenvalue problem
        E1, V1 = torch.lobpcg(A1, X=X1, B=B1, niter=niter, largest=True, tracker=tracker, tol=tol)
        # Count the number of iterations performed
        iters1_general = len(lambdas1)
        # Calculate the equivalence error for the generalized solution
        eq_err_general = torch.norm((mm(A1, V1) - mm(B1, V1) * E1), 2) / E1.max()

        # Try performing computation using scipy.sparse.linalg.lobpcg for generalized problem
        try:
            E2, V2, lambdas2 = scipy_lobpcg(A2, X2, B=B2, maxiter=niter, largest=True, retLambdaHistory=True, tol=tol)
            # Count the number of iterations performed
            iters2_general = len(lambdas2)
            # Calculate the equivalence error for the generalized scipy solution
            eq_err_general_scipy = (abs(A2.dot(V2) - B2.dot(V2) * E2)**2).sum() ** 0.5 / E2.max()
        except Exception as msg:
            # Handle any exceptions raised during scipy generalized computation
            print('Calling scipy_lobpcg failed [generalized]:', msg)
            iters2_general = -1
            eq_err_general_scipy = -1

        # Print a summary of errors and iterations for both methods
        print(f'''\
Handling of small tol={tol:6.0e}: torch.lobpcg vs scipy.sparse.linalg.lobpcg
----------------------------------------------------------------------------
              | standard    | generalized |  niter | method
torch.lobpcg  | {eq_err:10.2e}  | {eq_err_general:10.2e}  | {iters1:6} | ortho
scipy_lobpcg  | {eq_err_scipy:10.2e}  | {eq_err_general_scipy:10.2e}  | {iters2:6} | N/A
---(input size: {m:4}, eigenpairs:{k:2}, units: relative error, maxiter={niter:4})---
''')
    # 定义测试方法，用于测试矩阵乘法和向量乘法的组合
    def _test_addmm_addmv(self, f, t, m, v, *, alpha=None, beta=None, transpose_out=False, activation=None):
        # 获取张量 t 的数据类型
        dtype = t.dtype
        # 设置 numpy 数据类型与 torch 数据类型相同
        numpy_dtype = dtype
        # 如果数据类型为 torch.bfloat16 或 torch.half，则将 numpy 数据类型设置为 torch.float
        if dtype in {torch.bfloat16, torch.half}:
            numpy_dtype = torch.float
        # 如果数据类型为复数，则设置默认的 alpha 和 beta 值
        if dtype.is_complex:
            alpha = 0.9 + 0.3j if alpha is None else alpha
            beta = 0.5 + 0.6j if beta is None else beta
        else:
            # 否则设置默认的 alpha 和 beta 值
            alpha = 1.2 if alpha is None else alpha
            beta = 0.8 if beta is None else beta
        # 如果激活函数为 "gelu"，则使用带有 gelu 的特定参数调用函数 f
        if activation == "gelu":
            res1 = f(t, m, v, alpha=alpha, beta=beta, use_gelu=True)
        else:
            # 否则使用给定的 alpha 和 beta 值调用函数 f
            res1 = f(t, m, v, alpha=alpha, beta=beta)
        # 创建一个与 res1 具有相同大小的张量，并将其填充为 NaN
        res2 = torch.full_like(res1, math.nan)
        # 如果 transpose_out 为 True，则对 res2 进行转置操作
        if transpose_out:
            res2 = res2.t().clone(memory_format=torch.contiguous_format).t()
        # 如果激活函数为 "gelu"，则使用带有 gelu 的特定参数调用函数 f，并将结果存储在 res2 中
        if activation == "gelu":
            f(t, m, v, alpha=alpha, beta=beta, out=res2, use_gelu=True)
        else:
            # 否则使用给定的 alpha 和 beta 值调用函数 f，并将结果存储在 res2 中
            f(t, m, v, alpha=alpha, beta=beta, out=res2)
        # 计算 alpha * (m @ v) 的结果，并将其转换为 numpy 数据类型
        res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
        # 如果 beta 不等于 0，则将 beta * t 转换为 numpy 数据类型并加到 res3 上
        if beta != 0:
            res3 += (beta * t).to(numpy_dtype).cpu().numpy()
        # 如果激活函数为 "relu"，则将 res3 应用 ReLU 函数
        if activation == "relu":
            res3 = res3 * (res3 > 0)
        # 如果激活函数为 "gelu"，则将 res3 转换为张量并应用 GELU 激活函数
        elif activation == "gelu":
            res3_t = torch.from_numpy(res3).to(dtype)
            approximate = "tanh" if t.is_cuda else "none"
            res3_t = torch.nn.functional.gelu(res3_t, approximate=approximate)
            res3 = res3_t.to(numpy_dtype).cpu().numpy()
        else:
            # 否则断言激活函数为 None，否则抛出异常
            assert activation is None, f"unsupported activation {activation}"
        # 将 res3 转换为与张量 t 相同的数据类型
        res3 = torch.from_numpy(res3).to(dtype)
        # 断言 res1 与 res2 相等
        self.assertEqual(res1, res2)
        # 断言 res1 与 res3 相等
        self.assertEqual(res1, res3)
    # 定义名为 test_addmv 的测试方法，接受设备和数据类型作为参数
    def test_addmv(self, device, dtype):
        # 如果运行在 ARM64 架构且设备为 'cpu' 且数据类型为 torch.float16，则跳过测试并给出相关链接
        if IS_ARM64 and device == 'cpu' and dtype == torch.float16:
            raise unittest.SkipTest("Fails on ARM, see https://github.com/pytorch/pytorch/issues/125438")
        
        # 由于 torch.randn 不支持 bfloat16 类型，必须使用 torch.randn(...).to(bfloat16) 的方式初始化张量
        # "*0.2" 用于减小低精度下的误差
        ts = [
            0.2 * torch.randn(50, device=device).to(dtype),
            0.2 * torch.randn(1, device=device).to(dtype).expand(50),
        ]
        
        vs = [
            0.2 * torch.randn(100, device=device).to(dtype),
            0.2 * torch.ones(1, device=device).to(dtype).expand(100),  # 为了减小低精度下的误差
        ]
        
        ms = [
            # 0维张量
            0.2 * torch.ones((), device=device).to(dtype).expand(50, 100),  # 为了减小低精度下的误差
            # 1维张量
            0.2 * torch.randn((1, 100), device=device).to(dtype).expand(50, 100),
            # 这种初始化方式可以减小广播矩阵在低精度下的误差，确保中间值和结果值在低精度类型中完全可表示
            0.2 * torch.randint(3, (50, 1), dtype=torch.float, device=device).to(dtype).expand(50, 100),
            # 2维张量
            0.2 * torch.randn((50, 100), device=device).to(dtype),
            0.2 * torch.randn((100, 50), device=device).to(dtype).t(),
        ]
        
        # 使用 itertools.product 对 ms、vs、ts 中的张量进行组合
        for m, v, t in itertools.product(ms, vs, ts):
            # 调用自定义方法 _test_addmm_addmv，测试 torch.addmv 函数的功能
            self._test_addmm_addmv(torch.addmv, t, m, v)
        
        # 测试当 beta=0 且 t 为 nan 时的情况
        t = torch.full((50,), math.nan, device=device).to(dtype)
        for m, v in itertools.product(ms, vs):
            # 调用自定义方法 _test_addmm_addmv，测试 torch.addmv 函数在特定条件下的功能
            self._test_addmm_addmv(torch.addmv, t, m, v, beta=0)

    # 用于 CUDA 的浮点类型测试装饰器，支持 torch.bfloat16 类型（若使用 ROCm 或 SM53 或更新版本）
    # 否则，仅支持 torch.float 和 torch.double 类型
    @dtypesIfCUDA(*floating_types_and(*[torch.bfloat16] if TEST_WITH_ROCM or SM53OrLater else []))
    @dtypes(torch.float, torch.double)
    # 定义一个测试函数，用于测试 torch.addmv 的不同参数组合下的行优先和列优先情况
    def test_addmv_rowmajor_colmajor_incx_incy_lda(self, device, dtype):
        # 设置输出大小 o 和求和大小 s
        o = 5
        s = 3
        # 创建一个 o x s 的张量 a_data，其中填充了从 1 到 o*s 的连续数值
        a_data = torch.arange(1, o * s + 1, device=device, dtype=dtype).view(o, s)
        # 创建一个长度为 s 的张量 x_data，其中填充了从 1 到 s 的连续数值
        x_data = torch.arange(1, s + 1, 1, device=device, dtype=dtype)
        # 创建一个 o 长度的张量 y_data，其中所有元素初始化为 1
        y_data = torch.ones(o, device=device, dtype=dtype)
        # 创建一个控制用的张量 control，包含预期的输出结果
        control = torch.tensor([15., 33., 51., 69., 87.], device=device, dtype=dtype)

        # 定义内部测试函数 _test，用于测试不同的行优先、列优先、增量参数和 lda 尾部参数组合
        def _test(row_major, incx, incy, lda_tail):
            # 根据 row_major 参数选择合适的存储方式和形状来创建张量 a_storage
            if row_major:
                a_storage = torch.full((o, s + lda_tail), float('nan'), device=device, dtype=dtype)
            else:
                a_storage = torch.full((s, o + lda_tail), float('nan'), device=device, dtype=dtype).permute(1, 0)
            # 从 a_data 复制数据到张量 a，并根据不同的 row_major 参数进行设置
            a = a_storage[:o, :s].copy_(a_data)

            # 创建一个存储大小为 (s, incx) 的张量 x_storage，用 float('nan') 填充
            x_storage = torch.full((s, incx), float('nan'), device=device, dtype=dtype)
            # 从 x_data 复制数据到张量 x 的第一列，并根据 incx 参数进行设置
            x = x_storage[:, 0].copy_(x_data)

            # 创建一个存储大小为 (o, incy) 的张量 y_storage，用 float('nan') 填充
            y_storage = torch.full((o, incy), float('nan'), device=device, dtype=dtype)
            # 从 y_data 复制数据到张量 y 的第一列，并根据 incy 参数进行设置
            y = y_storage[:, 0].copy_(y_data)

            # 调用 self._test_addmm_addmv 方法进行 addmv 操作的测试
            self._test_addmm_addmv(torch.addmv, y, a, x)

        # 使用 itertools.product 生成 row_major、incx、incy 和 lda_tail 的所有组合，并逐一调用 _test 函数进行测试
        for row_major, incx, incy, lda_tail in itertools.product((False, True), (1, 2), (1, 2), (0, 1)):
            _test(row_major, incx, incy, lda_tail)
    # 定义一个测试方法，用于测试 torch.addmm 函数的实现
    def _test_addmm_impl(self, func, activation, device, dtype):
        # 创建大小为 10x25 的随机张量 M，存储在指定设备上，并使用指定数据类型
        M = torch.randn(10, 25, device=device).to(dtype)
        # 创建大小为 10x50 的随机张量 m1，存储在指定设备上，并使用指定数据类型
        m1 = torch.randn(10, 50, device=device).to(dtype)
        # 创建大小为 50x25 的随机张量 m2，存储在指定设备上，并使用指定数据类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 方法，测试 torch.addmm 函数，传入参数 M, m1, m2，并指定激活函数 activation
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # 创建大小为 25 的随机张量 V，存储在指定设备上，并使用指定数据类型
        V = torch.randn(25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 方法，测试 torch.addmm 函数，传入参数 V, m1, m2，同时设置 beta=1 和指定激活函数 activation
        # 这种设置会在 CUDA 中触发 epilogue 融合
        self._test_addmm_addmv(func, V, m1, m2, beta=1, activation=activation)

        # 测试 0-步长的情况
        # 创建大小为 10x25 的随机张量 M，初始大小为 10x1，存储在指定设备上，并使用指定数据类型，然后扩展为 10x25
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        # 创建大小为 10x50 的随机张量 m1，初始大小为 10x1，存储在指定设备上，并使用指定数据类型，然后扩展为 10x50
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        # 创建大小为 50x25 的随机张量 m2，存储在指定设备上，并使用指定数据类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 方法，测试 torch.addmm 函数，传入参数 M, m1, m2，并指定激活函数 activation
        self._test_addmm_addmv(func, M, m1, m2, activation=activation)

        # 测试 beta=0，M 包含 NaN 值的情况
        # 创建大小为 10x25 的张量 M，其所有元素均为 NaN，存储在指定设备上，并使用指定数据类型
        M = torch.full((10, 25), math.nan, device=device).to(dtype)
        # 创建大小为 10x50 的随机张量 m1，存储在指定设备上，并使用指定数据类型
        m1 = torch.randn(10, 50, device=device).to(dtype)
        # 创建大小为 50x25 的随机张量 m2，存储在指定设备上，并使用指定数据类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 方法，测试 torch.addmm 函数，传入参数 M, m1, m2，并指定 beta=0 和激活函数 activation
        self._test_addmm_addmv(func, M, m1, m2, beta=0, activation=activation)

        # 测试转置操作
        # 遍历四个布尔值组合，分别表示是否进行转置
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            # 定义一个可能执行转置的函数
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                # 如果条件为真，对输入张量 m 进行转置，并使用连续内存格式
                return m.t().clone(memory_format=torch.contiguous_format).t()

            # 根据 t1, t2, t3 的值，可能对 M, m1, m2 进行转置操作
            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            # 调用 _test_addmm_addmv 方法，测试 torch.addmm 函数，传入参数 M, m1, m2，并指定是否对输出进行转置和激活函数 activation
            self._test_addmm_addmv(func, M, m1, m2, transpose_out=t4, activation=activation)

            # 如果 t1 为真，使用向量 V 而不是矩阵 M，测试 CUDA 中的 epilogue 融合
            if t1:
                self._test_addmm_addmv(func, V, m1, m2, beta=1, transpose_out=t4, activation=activation,)
    def test_addmm_relu(self, device, dtype):
        self._test_addmm_impl(torch._addmm_activation, "relu", device, dtype)



    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 5e-2,
                        torch.half: 5e-2, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    @dtypesIfCUDA(*floating_types_and(
                  *[torch.bfloat16, torch.half] if TEST_WITH_ROCM or SM53OrLater else []))
    @dtypes(*floating_types_and(torch.bfloat16))
    @tf32_on_and_off(0.05)
    @bf32_on_and_off(0.05)
    def test_addmm_gelu(self, device, dtype):
        self._test_addmm_impl(torch._addmm_activation, "gelu", device, dtype)



    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(*floating_and_complex_types())
    @tf32_on_and_off(0.005)
    @bf32_on_and_off(0.005)
    def test_addmm_sizes(self, device, dtype):
        for m in [0, 1, 25]:
            for n in [0, 1, 10]:
                for k in [0, 1, 8]:
                    M = torch.randn(n, m, device=device).to(dtype)
                    m1 = torch.randn(n, k, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self._test_addmm_addmv(torch.addmm, M, m1, m2)

                    m1 = torch.randn(n, k + 1, device=device).to(dtype)
                    m2 = torch.randn(k, m, device=device).to(dtype)
                    self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.addmm(M, m1, m2))
                    self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.mm(m1, m2))



    @dtypes(torch.half)
    @onlyCUDA
    def test_addmm_baddbmm_overflow(self, device, dtype):
        orig = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        inp = torch.zeros(128, 128, dtype=torch.half, device=device)
        mat1 = torch.ones(128, 1000, dtype=torch.half, device=device) * 100
        mat2 = torch.ones(1000, 128, dtype=torch.half, device=device) * 100
        out = torch.addmm(inp, mat1, mat2, alpha=0.001, beta=0.)
        # just check for no overflow on ROCM
        if TEST_WITH_ROCM:
            self.assertFalse(out.isinf().any())
        else:
            self.assertTrue((out == 10000.).all())
        inp = torch.zeros(3, 128, 128, dtype=torch.half, device=device)
        mat1 = torch.ones(3, 128, 1000, dtype=torch.half, device=device) * 100
        mat2 = torch.ones(3, 1000, 128, dtype=torch.half, device=device) * 100
        out = torch.baddbmm(inp, mat1, mat2, alpha=0.001, beta=0.)
        if TEST_WITH_ROCM:
            self.assertFalse(out.isinf().any())
        else:
            self.assertTrue((out == 10000.).all())
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig



    @dtypes(torch.float)
    # 在给定设备和数据类型上，测试 torch.baddbmm 函数处理 NaN 输入时的行为
    def test_baddbmm_nan_input_with_zero_beta(self, device, dtype):
        # 遍历不同形状的张量
        for shape in [[3, 2, 2], [2, 20, 20]]:
            # 创建随机张量 mat1 和 mat2
            mat1, mat2 = (torch.randn(shape, dtype=dtype, device=device) for _ in range(2))
            # 创建包含 NaN 值的输入张量列表
            inputs = [torch.randn(shape, dtype=dtype, device=device),
                      torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan)]
            # 创建输出张量列表，其中部分也包含 NaN 值
            outs = [None, torch.randn(shape, dtype=dtype, device=device),
                    torch.randn(shape, dtype=dtype, device=device).fill_(torch.nan)]
            # 对输入和输出张量进行组合，生成所有可能的输入输出组合
            options = itertools.product(inputs, outs)
            for input, out in options:
                # 计算参考结果，使用 torch.bmm 计算矩阵乘积
                y_ref = torch.bmm(mat1, mat2)
                # 调用 torch.baddbmm 函数，处理 NaN 输入并使用 beta=0.0，输出到指定的 out 张量
                y = torch.baddbmm(input, mat1, mat2, beta=0.0, out=out)
                # 断言实际输出与参考结果相等
                self.assertEqual(y_ref, y)

    # 测试 torch.baddbmm 函数对不同输入数据类型的兼容性
    @dtypes(torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64)
    def test_baddbmm_input_dtypes_compatibility(self, device, dtype):
        # 创建随机张量 batch1 和 batch2，数据类型为 torch.float32
        batch1 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        batch2 = torch.rand((1, 2, 2), dtype=torch.float32, device=device)
        # 创建数据类型为 dtype 的随机输入张量 input_tensor
        input_tensor = torch.rand((1, 2, 2), device=device).to(dtype)
        # 如果 dtype 不是 torch.float32，则期望抛出 RuntimeError 异常
        if dtype != torch.float32:
            with self.assertRaisesRegex(RuntimeError, "Input dtypes must be the same"):
                y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0)
        else:
            # 创建指定形状的输出张量 out，并填充为 NaN
            out = torch.randn((1, 2, 2), dtype=dtype, device=device).fill_(torch.nan)
            # 计算参考结果，使用 torch.bmm 计算矩阵乘积
            y_ref = torch.bmm(batch1, batch2)
            # 调用 torch.baddbmm 函数，处理 NaN 输入并使用 beta=0.0，输出到指定的 out 张量
            y = torch.baddbmm(input_tensor, batch1, batch2, beta=0.0, out=out)
            # 断言实际输出与参考结果相等
            self.assertEqual(out, y_ref)

    # 测试特定情况下的 torch.matmul 函数
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    @onlyCUDA
    def test_matmul_45724(self, device):
        # 创建指定形状和数据类型的随机张量 a, b
        a = torch.rand(65537, 22, 64, device=device, dtype=torch.half)
        b = torch.rand(65537, 64, 22, device=device, dtype=torch.half)
        # 创建形状为 (65537, 22, 22) 的全为 NaN 的半精度张量 c
        c = torch.full((65537, 22, 22), math.nan, dtype=torch.half, device=device)
        # 在 CPU 上计算参考结果，然后转换为半精度，并移动到 GPU
        cpu_result = torch.matmul(a.cpu().float(), b.cpu().float()).cuda().half()
        # 使用 torch.matmul 计算 a 和 b 的矩阵乘积，结果存储到张量 c 中
        torch.matmul(a, b, out=c)
        # 断言张量 c 与 cpu_result 相等
        self.assertEqual(c, cpu_result)
    # 测试 _int_mm 函数在特定条件下是否会引发错误，针对不同的条件进行测试
    def test__int_mm_errors(self, device):
        # 如果使用 ROCM，则跳过测试，因为 _int_mm 在 ROCM 平台上未编译
        if TEST_WITH_ROCM:
            self.skipTest("_int_mm not compiled for ROCM")

        # 获取当前的 CUDA 版本
        version = _get_torch_cuda_version()
        # 如果 CUDA 版本小于 11.7，则跳过测试，因为 _int_mm 只在 CUDA 11.7 及以上版本编译
        if version < (11, 7):
            self.skipTest("_int_mm only compiled for CUDA 11.7")

        # 生成 torch.Tensor，数据类型为 torch.int8，用于测试
        def genf_int(x, y):
            return torch.empty((x, y), dtype=torch.int8, device=device)

        # 生成两个符合条件的 torch.Tensor 对象，用于 _int_mm 函数的参数
        def _gen_pair(m, k, n):
            return genf_int(m, k), genf_int(k, n)

        # 下面的代码块测试不同的输入条件是否会引发预期的 RuntimeError 异常，并捕获异常信息进行断言
        # 测试 self.size(0) 需要大于 16，但实际为 16 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"self.size\(0\) needs to be greater than 16, but got 16",
                               lambda: torch._int_mm(*_gen_pair(16, 8, 32)))
        # 测试 self.size(1) 需要大于 0 且为 8 的倍数，但实际为 7 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"self.size\(1\) needs to be greater than 0 and a multiple of 8, but got 7",
                               lambda: torch._int_mm(*_gen_pair(17, 7, 32)))
        # 测试 self.size(1) 需要与 mat2.size(0) 相匹配，但实际为不匹配的情况
        self.assertRaisesRegex(RuntimeError,
                               r"self.size\(1\) needs to match mat2.size\(0\) but got 8 and 7",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(7, 32)))
        # 测试 mat2.size(1) 需要大于 0 且为 8 的倍数，但实际为 31 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"mat2.size\(1\) needs to be greater than 0 and a multiple of 8, but got 31",
                               lambda: torch._int_mm(*_gen_pair(17, 8, 31)))
        # 测试期望输入数据类型为 torch.int8，但实际输入数据类型为 torch.float32 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"expected scalar type Char but found Float",
                               lambda: torch._int_mm(genf_int(17, 8).float(), genf_int(8, 32)))
        # 测试期望输入数据类型为 torch.int8，但实际输入数据类型为 torch.float32 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"expected scalar type Char but found Float",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32).float()))
        # 测试期望输出数据类型为 torch.int8，但实际输出数据类型为 torch.float32 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"Expected result dtype to be of type kInt but got float",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32), out=genf_int(16, 32).float()))
        # 测试期望输出结果的大小为 (17, *)，但实际输出结果大小的第一个维度为 15 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"Expected result.size\(0\) to be 17 but got 15",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32), out=genf_int(15, 32).int()))
        # 测试期望输出结果的大小为 (17, *)，但实际输出结果大小的第一个维度为 16 的情况
        self.assertRaisesRegex(RuntimeError,
                               r"Expected result.size\(0\) to be 17 but got 16",
                               lambda: torch._int_mm(genf_int(17, 8), genf_int(8, 32), out=genf_int(16, 31).int()))

    # 使用 onlyCPU 装饰器和 parametrize 装饰器为该测试方法添加参数化配置，进行更多的测试覆盖
    @onlyCPU
    @parametrize("m", [0, 8, 17])
    @parametrize("k", [0, 16, 32])
    @parametrize("n", [16, 32])
    @parametrize("use_transpose_a", [True, False])
    @parametrize("use_transpose_b", [True, False])
    @parametrize("non_contig_type", [0, 1, 2])
    # 定义测试函数，用于测试 torch._int_mm 函数在不同参数和条件下的表现
    def test__int_mm_cpu(self, device, m, k, n, use_transpose_a, use_transpose_b, non_contig_type):
        # non_contig_type:
        # 0: 整个数据缓冲区是连续的（可以转置）
        # 1: 至少一个维度的步长为1，但整个缓冲区不是连续的
        # 2: 两个维度的步长都不为1

        # 定义生成整型和浮点型输入数据的函数
        def genf_int_float(x, y, use_transpose, non_contig_type):
            # 如果需要转置，则交换 x 和 y
            if use_transpose:
                x, y = y, x
            # 根据 non_contig_type 调整 y 的大小
            if non_contig_type != 0:
                y = y * 2
            # 生成 torch.Tensor 类型的 x_int8，数据类型为 torch.int8
            x_int8 = torch.randint(-10, 10, (x, y), dtype=torch.int8, device=device)
            # 将 x_int8 转换为 torch.float32 类型的 x_float
            x_float = x_int8.to(torch.float32)
            # 根据 non_contig_type 进一步调整 x_int8 和 x_float
            if non_contig_type == 1:
                x_int8 = x_int8[:, : y // 2]
                x_float = x_float[:, : y // 2]
            elif non_contig_type == 2:
                x_int8 = x_int8[:, ::2]
                x_float = x_float[:, ::2]
            # 如果需要转置，则返回转置后的数据
            if use_transpose:
                return x_int8.t(), x_float.t()
            # 否则返回原始数据
            return x_int8, x_float

        # 如果 non_contig_type 不为0且 m 或 k 为0，则直接返回，跳过测试
        if non_contig_type != 0 and (m == 0 or k == 0):
            return

        # 生成 a_int8 和 a_float
        a_int8, a_float = genf_int_float(m, k, use_transpose_a, non_contig_type)
        # 生成 b_int8 和 b_float
        b_int8, b_float = genf_int_float(k, n, use_transpose_b, non_contig_type)
        
        # 调用 torch._int_mm 函数计算整型矩阵乘法，结果为 c_int32
        c_int32 = torch._int_mm(a_int8, b_int8)
        
        # 断言 c_int32 的数据类型为 torch.int32
        self.assertTrue(c_int32.dtype is torch.int32)
        # 断言 c_int32 的设备与指定的设备一致
        self.assertEqual(c_int32.device, torch.device(device))
        # 断言 c_int32 转换为浮点型后与 torch.mm(a_float, b_float) 的结果一致
        self.assertEqual(c_int32.float(), torch.mm(a_float, b_float))
        
        # 创建一个与 c_int32 相同大小的新张量 c_int32_result
        c_int32_result = c_int32.new_empty(c_int32.size())
        # 使用 torch._int_mm 函数计算整型矩阵乘法，并将结果存入 c_int32_result
        torch._int_mm(a_int8, b_int8, out=c_int32_result)
        # 断言 c_int32_result 转换为浮点型后与 torch.mm(a_float, b_float) 的结果一致
        self.assertEqual(c_int32_result.float(), torch.mm(a_float, b_float))

    # 跳过 Windows 平台下的测试
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    # 在 FBCODE 环境且远程 GPU 上时跳过测试
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    # 仅对本地设备类型进行测试
    @onlyNativeDeviceTypes
    # 参数化测试 m 取值为 32 或 64
    @parametrize("m", [32, 64])
    # 参数化测试 k 取值为 32 或 64
    @parametrize("k", [32, 64])
    # 参数化测试 n 取值为 48 或 64
    @parametrize("n", [48, 64])
    # 在测试函数中，针对整型4位矩阵乘法的性能进行单元测试
    def test__int4_mm(self, device, m, k, n):
        # 如果设备类型为cuda且不支持SM80或更高版本，则跳过测试
        if self.device_type == 'cuda' and not SM80OrLater:
            self.skipTest("requires SM80 or later")

        # 如果在ROCM平台上进行测试，则跳过测试，因为_int4_mm未在ROCM上编译
        if TEST_WITH_ROCM:
            self.skipTest("_int4_mm not compiled for ROCM")

        # 定义量化组数和内部k块数
        q_group = 32
        inner_k_tiles = 2

        # 设置随机种子，并生成设备上的随机bfloat16类型的矩阵a和b
        torch.manual_seed(1)
        a_bf16 = torch.rand((m, k), dtype=torch.bfloat16, device=device)
        b_bf16 = torch.rand((k, n), dtype=torch.bfloat16, device=device)

        # 定义将权重转换为int4pack格式的函数
        def convert_weight_to_int4pack(b):
            # 将b按照4位量化，并获取量化参数和零点值
            b_int32, b_scales_and_zeros = _group_quantize_tensor(
                b, n_bit=4, q_group_size=q_group
            )
            # 转换为int4pack格式的权重
            b_int4pack = torch._convert_weight_to_int4pack(
                b_int32, inner_k_tiles
            )
            return b_int4pack, b_scales_and_zeros

        # 定义使用int4pack格式权重进行矩阵乘法的函数
        def weight_int4pack_mm(a, b_int4pack, b_scales_and_zeros):
            return torch._weight_int4pack_mm(
                a, b_int4pack, q_group, b_scales_and_zeros
            )

        # 将bfloat16类型的b矩阵转换为int4pack格式，并获取其量化参数和零点值
        b_int4pack, b_scales_and_zeros_bf16 = convert_weight_to_int4pack(b_bf16)

        # 对每种数据类型（torch.bfloat16以及如果是CPU，则还有torch.float16和torch.float32）进行以下操作
        for dtype in [torch.bfloat16] + ([torch.float16, torch.float32] if device == "cpu" else []):
            # 将a转换为当前数据类型的张量
            a = a_bf16.to(dtype=dtype)
            # 将b和其量化参数和零点值转换为当前数据类型的张量
            b = b_bf16.to(dtype=dtype)
            b_scales_and_zeros = b_scales_and_zeros_bf16.to(dtype=dtype)
            # 计算参考结果
            ref = torch.mm(a, b)
            # 使用int4pack格式的权重进行矩阵乘法
            res = weight_int4pack_mm(a, b_int4pack, b_scales_and_zeros)

            # 计算平均误差并断言其小于0.05
            mean_err = ((res - ref).abs() / ref).mean()
            self.assertTrue(mean_err < 0.05)


    # 如果在Windows系统上则跳过此测试
    @unittest.skipIf(IS_WINDOWS, "Skipped on Windows!")
    # 如果在FBCODE且是远程GPU则跳过此测试
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "cublas runtime error")
    # 仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 参数化测试，测试多组不同的m、k、n值
    @parametrize("m", [32, 64])
    @parametrize("k", [32, 64])
    @parametrize("n", [48, 64])
    # 编译int4_mm函数的测试
    def test_compile_int4_mm(self, device, m, k, n):
        # 如果设备类型为cuda且不支持SM80或更高版本，则跳过测试
        if self.device_type == 'cuda' and not SM80OrLater:
            self.skipTest("requires SM80 or later")

        # 如果在ROCM平台上进行测试，则跳过测试，因为_int4_mm未在ROCM上编译
        if TEST_WITH_ROCM:
            self.skipTest("_int4_mm not compiled for ROCM")

        # 定义量化组数和内部k块数
        q_group = 32
        inner_k_tiles = 2

        # 设置随机种子，并生成设备上的随机bfloat16类型的矩阵a和b
        torch.manual_seed(1)
        a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
        b = torch.rand((k, n), dtype=torch.bfloat16, device=device)

        # 将b按照4位量化，并获取量化参数和零点值
        b_int32, b_scales_and_zeros = _group_quantize_tensor(
            b, n_bit=4, q_group_size=q_group
        )

        # 使用torch.compile装饰器编译int4_mm函数
        @torch.compile
        def int4_mm(a, b_int32, b_scales_and_zeros):
            # 将b_int32按照inner_k_tiles拆分为int4pack格式
            b_int4pack = torch._convert_weight_to_int4pack(
                b_int32, inner_k_tiles
            )
            # 使用int4pack格式的权重进行矩阵乘法
            return torch._weight_int4pack_mm(
                a, b_int4pack, q_group, b_scales_and_zeros
            )

        # 调用编译后的int4_mm函数进行矩阵乘法，并获取参考结果
        res = int4_mm(a, b_int32, b_scales_and_zeros)
        ref = torch.mm(a, b)

        # 计算平均误差并断言其小于0.05
        mean_err = ((res - ref).abs() / ref).mean()
        self.assertTrue(mean_err < 0.05)

    # 仅适用于CPU的测试
    @onlyCPU
    # 参数化测试，测试多组不同的m、k、n值
    @parametrize("m", [32, 64])
    @parametrize("k", [32, 64])
    @parametrize("n", [48, 64])
    # 定义一个测试函数，用于测试整型矩阵乘法的功能，采用指定设备和给定的矩阵维度
    def test__int8_mm(self, device, m, k, n):
        # 设置随机种子为1，保证结果可重复性
        torch.manual_seed(1)
        # 生成随机的 m × k 大小的张量 a，数据类型为 bfloat16，存储在指定设备上
        a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
        # 生成随机的 n × k 大小的张量 b，数据类型为 bfloat16，存储在指定设备上
        b = torch.rand((n, k), dtype=torch.bfloat16, device=device)

        # 定义内部函数，将权重张量 b 转换为 int8 格式的打包张量及其缩放系数
        def convert_weight_to_int8pack(b):
            # 调用 _dynamically_quantize_per_channel 函数进行通道量化，范围为 -128 到 127，数据类型为 torch.int8
            b_int8pack, b_scales, _ = _dynamically_quantize_per_channel(
                b, -128, 127, torch.int8
            )
            return b_int8pack, b_scales

        # 定义内部函数，执行整型矩阵乘法，输入参数为 a、b_int8pack、b_scales
        def weight_int8pack_mm(a, b_int8pack, b_scales):
            return torch._weight_int8pack_mm(
                a, b_int8pack, b_scales
            )

        # 调用 convert_weight_to_int8pack 函数，获取 b 的 int8 打包张量及其缩放系数
        b_int8pack, b_scales = convert_weight_to_int8pack(b)
        # 调用 weight_int8pack_mm 函数，执行整型矩阵乘法，得到结果张量 res
        res = weight_int8pack_mm(a, b_int8pack, b_scales)
        # 计算参考结果张量 ref，即使用 torch 自带的浮点数矩阵乘法函数 mm(a, b.transpose(0, 1))
        ref = torch.mm(a, b.transpose(0, 1))

        # 计算平均误差，mean_err 表示相对误差的平均值
        mean_err = ((res - ref).abs() / ref).mean()
        # 断言平均误差小于 0.05，用于验证整型矩阵乘法的准确性
        self.assertTrue(mean_err < 0.05)

    # 标记为仅 CPU 测试，参数化 m、k、n 的不同值进行测试
    @onlyCPU
    @parametrize("m", [32, 64])
    @parametrize("k", [32, 64])
    @parametrize("n", [48, 64])
    # 定义测试函数，用于验证整型矩阵乘法的编译执行，采用指定设备和给定的矩阵维度
    def test_compile_int8_mm(self, device, m, k, n):
        # 设置随机种子为1，保证结果可重复性
        torch.manual_seed(1)
        # 生成随机的 m × k 大小的张量 a，数据类型为 bfloat16，存储在指定设备上
        a = torch.rand((m, k), dtype=torch.bfloat16, device=device)
        # 生成随机的 n × k 大小的张量 b，数据类型为 bfloat16，存储在指定设备上
        b = torch.rand((n, k), dtype=torch.bfloat16, device=device)

        # 调用 _dynamically_quantize_per_channel 函数，将张量 b 量化为 int8 格式的打包张量及其缩放系数
        b_int8pack, b_scales, _ = _dynamically_quantize_per_channel(
            b, -128, 127, torch.int8
        )

        # 使用 torch.compile 装饰器定义编译后的整型矩阵乘法函数 int8_mm，输入参数为 a、b_int8pack、b_scales
        @torch.compile
        def int8_mm(a, b_int8pack, b_scales):
            return torch._weight_int8pack_mm(
                a, b_int8pack, b_scales
            )

        # 调用 int8_mm 函数，执行整型矩阵乘法，得到结果张量 res
        res = int8_mm(a, b_int8pack, b_scales)
        # 计算参考结果张量 ref，即使用 torch 自带的浮点数矩阵乘法函数 mm(a, b.transpose(0, 1))
        ref = torch.mm(a, b.transpose(0, 1))

        # 计算平均误差，mean_err 表示相对误差的平均值
        mean_err = ((res - ref).abs() / ref).mean()
        # 断言平均误差小于 0.05，用于验证整型矩阵乘法的准确性
        self.assertTrue(mean_err < 0.05)

    # 标记为仅 CPU 测试，参数化 m、k 的不同值进行测试
    @onlyCPU
    @parametrize("m", [32, 35, 36, 40, 64])
    @parametrize("k", [32, 35, 36, 40, 64])
    # 注释说明：该测试函数旨在覆盖 BlasKernel.cpp 中的 fp16_gemv_trans 函数。
    # 目前，被 32 整除，8 但不能被 32 整除，以及 4 但不能被 8 整除，这三种情况都很重要。
    def test_fp16_mv_transposed_first_argument_arm_cpu(self, device, m, k):
        # 设置随机种子为1，保证结果可重复性
        torch.manual_seed(1)
        # 生成随机的 m × k 大小的张量 a，数据类型为 half（即 float16），存储在指定设备上
        a = torch.rand((m, k), dtype=torch.half, device=device)
        # 生成随机的 1 × k 大小的张量 b，数据类型为 half（即 float16），存储在指定设备上
        b = torch.rand((1, k), dtype=torch.half, device=device)

        # 获取当前允许的 CPU FP16 精度缩减设置
        prev = torch._C._get_cpu_allow_fp16_reduced_precision_reduction()
        try:
            # 设置允许 CPU FP16 精度缩减为 False
            torch._C._set_cpu_allow_fp16_reduced_precision_reduction(False)
            # 计算参考结果张量 ref，即使用 torch 自带的浮点数矩阵乘法函数 mm(a, b.t())
            ref = torch.mm(a, b.t())
            try:
                # 尝试设置允许 CPU FP16 精度缩减为 True
                torch._C._set_cpu_allow_fp16_reduced_precision_reduction(True)
            except RuntimeError as e:
                # 如果设置失败，跳过当前测试
                raise unittest.SkipTest from e
            # 计算结果张量 res，即使用 torch 自带的浮点数矩阵乘法函数 mm(a, b.t())
            res = torch.mm(a, b.t())
            # 使用 torch.testing.assert_close 函数断言 res 和 ref 的接近程度，设定容忍的绝对误差和相对误差
            torch.testing.assert_close(res, ref, atol=1e-2, rtol=1e-2)
        finally:
            # 恢复之前的 CPU FP16 精度缩减设置
            torch._C._set_cpu_allow_fp16_reduced_precision_reduction(prev)

    # 标记为慢速测试，仅限本地设备类型
    @slowTest
    @onlyNativeDeviceTypes
    # 注释说明：bfloat16 精度不足以通过此测试
    # 参数化数据类型，包括 half、float32、float64、int32、int64、cfloat
    # 将装饰器应用于函数，指定在 CUDA 下使用的数据类型
    @dtypesIfCUDA(torch.float32, torch.float64, torch.cfloat, torch.cdouble)
    # 将装饰器应用于函数，开启或关闭 tf32 模式，设置概率为 0.01
    @tf32_on_and_off(0.01)
    # 将装饰器应用于函数，开启或关闭 bf32 模式，设置概率为 0.01
    @bf32_on_and_off(0.01)
    # 将装饰器应用于函数，仅允许在本地设备类型上运行
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试非内存稠密矩阵乘法
    def test_mm_bmm_non_memory_dense(self, device):
        # 定义内部函数 _slice，对张量进行切片操作
        def _slice(tensor, fn):
            return fn(tensor)[..., ::2]
        # 创建随机张量 A 和 B，数据类型为复数浮点数，位于指定设备上
        A = torch.randn(3, 6, dtype=torch.cfloat, device=device)
        B = torch.randn(3, 3, dtype=torch.cfloat, device=device)
        # 创建一个空的输出张量 out，数据类型为复数浮点数，位于指定设备上，并转置
        out = torch.empty(3, 3, device=device, dtype=torch.complex64).t()
        # 创建另一个空的输出张量 out1，数据类型为复数浮点数，位于指定设备上，并转置
        out1 = torch.empty(3, 3, device=device, dtype=torch.complex64).t()
        # 对张量 A 进行共轭操作后进行切片操作
        A_conj = _slice(A, torch.conj)
        # 对张量 A 进行物理共轭操作后进行切片操作
        A_conj_physical = _slice(A, torch.conj_physical)

        # 测试 torch.mm 函数，验证共轭操作后的矩阵乘法结果是否相等
        self.assertEqual(torch.mm(A_conj, B, out=out), torch.mm(A_conj_physical, B, out=out))
        self.assertEqual(torch.mm(A_conj.t(), B, out=out), torch.mm(A_conj_physical.t(), B, out=out))

        # 创建多维随机张量 Ab 和 Bb，数据类型为复数浮点数，位于指定设备上
        Ab = torch.randn(2, 3, 6, dtype=torch.cfloat, device=device)
        Bb = torch.randn(2, 3, 3, dtype=torch.cfloat, device=device)
        # 扩展 Bb 张量的第一维，使其与 Ab 张量的维度匹配
        Bb_ = torch.randn(1, 3, 3, dtype=torch.cfloat, device=device).expand(2, 3, 3)
        # 创建一个空的输出张量 out_b，数据类型为复数浮点数，位于指定设备上，并转置
        out_b = torch.empty(2, 3, 3, device=device, dtype=torch.complex64).mT

        # 对张量 Ab 进行共轭操作后进行切片操作
        Ab_conj = _slice(Ab, torch.conj)
        # 对张量 Ab 进行物理共轭操作后进行切片操作
        Ab_conj_physical = _slice(Ab, torch.conj_physical)

        # 定义内部函数 t_b，对张量进行转置操作
        def t_b(tensor):
            return tensor.mT

        # 测试 torch.bmm 函数，验证共轭操作后的批量矩阵乘法结果是否相等
        self.assertEqual(torch.bmm(Ab_conj, Bb, out=out_b), torch.bmm(Ab_conj_physical, Bb, out=out_b))
        self.assertEqual(torch.bmm(t_b(Ab_conj), Bb, out=out_b), torch.bmm(t_b(Ab_conj_physical), Bb, out=out_b))

        # 测试广播机制下的 torch.bmm 函数，验证共轭操作后的批量矩阵乘法结果是否相等
        self.assertEqual(torch.bmm(Ab_conj, Bb_, out=out_b), torch.bmm(Ab_conj_physical, Bb_, out=out_b))
        self.assertEqual(torch.bmm(t_b(Ab_conj), Bb_, out=out_b), torch.bmm(t_b(Ab_conj_physical), Bb_, out=out_b))

    # 将装饰器应用于函数，仅允许在本地设备类型上运行
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试共轭转置矩阵乘法
    def test_mm_conjtranspose(self, device):
        # 创建随机张量 A 和 B，数据类型为复数浮点数，位于指定设备上
        A = torch.randn(3, 3, dtype=torch.cfloat, device=device)
        B = torch.randn(3, 3, dtype=torch.cfloat, device=device)

        # 测试 A 的共轭转置矩阵乘法
        out1 = torch.mm(A.t().conj(), B)
        out1_ref = torch.mm(A.t().conj_physical(), B)
        self.assertEqual(out1, out1_ref)

        # 测试 B 的共轭转置矩阵乘法
        out1 = torch.mm(A, B.t().conj())
        out1_ref = torch.mm(A, B.t().conj_physical())
        self.assertEqual(out1, out1_ref)

        # 测试 A 和 B 的共轭转置矩阵乘法
        out1 = torch.mm(A.t().conj(), B.t().conj())
        out1_ref = torch.mm(A.t().conj_physical(), B.t().conj_physical())
        self.assertEqual(out1, out1_ref)

    # 将装饰器应用于函数，仅允许在本地设备类型上运行
    @onlyNativeDeviceTypes
    # 定义一个测试函数，用于测试空输入和混合数据类型错误的情况
    def test_mm_empty_inputs_mixed_dtype_errors(self, device):
        # 创建随机整数张量 a 和随机浮点数张量 b，数据类型不匹配，位于指定设备上
        a = torch.randint(0, 10, [1, 10], dtype=torch.int16, device=device)
        b = torch.randn(10, 20, dtype=torch.float32, device=device)
        # 断言在执行 torch.mm 函数时会抛出 RuntimeError，指出数据类型不匹配的错误
        with self.assertRaisesRegex(RuntimeError, "expected .* and .* to have the same dtype, but got:"):
            torch.mm(a, b)

    # 将装饰器应用于函数，仅允许在本地设备类型上运行，并指定允许的数据类型
    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    # 定义一个测试函数，测试带有不足对应维度大小的步幅的情况
    def test_strided_mm_bmm(self, device, dtype):
        # 创建一个包含浮点数的张量 x，形状为 2x3，指定设备和数据类型
        x = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=dtype, device=device)
        # 新的张量形状
        new_shape = [2, 2, 2]
        # 新的步幅
        new_stride = [3, 1, 1]
        # 根据指定的形状和步幅创建一个 strided view 张量 sx
        sx = torch.as_strided(x, size=new_shape, stride=new_stride)

        # 定义一个使用 torch.bmm 进行矩阵乘法的匿名函数
        torch_fn = lambda x: torch.bmm(x, x)  # noqa: E731
        # 定义一个使用 np.matmul 进行矩阵乘法的匿名函数
        np_fn = lambda x: np.matmul(x, x)  # noqa: E731
        # 调用方法，将 torch_fn 和 np_fn 应用于 sx，并进行比较
        self.compare_with_numpy(torch_fn, np_fn, sx)

        # 定义一个使用 torch.mm 进行矩阵乘法的匿名函数，应用于 sx 的第一个元素
        torch_fn = lambda x: torch.mm(x, x)  # noqa: E731
        # 调用方法，将 torch_fn 应用于 sx 的第一个元素，并与 np_fn 进行比较
        self.compare_with_numpy(torch_fn, np_fn, sx[0])

    # 装饰器和方法测试函数 _test_addbmm_baddbmm
    @precisionOverride({torch.half: 0.05, torch.bfloat16: 0.05})
    @onlyNativeDeviceTypes
    @dtypes(*floating_and_complex_types_and(torch.bfloat16, torch.half))
    @tf32_on_and_off(0.05)
    @bf32_on_and_off(0.05)
    @precisionOverride({torch.half: 0.1, torch.bfloat16: 0.5})
    @onlyNativeDeviceTypes
    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(torch.double, torch.cdouble)



    # 跳过不支持 LAPACK 的 CPU 环境下的测试
    # 在没有 MAGMA 和 CUSOLVER 的 CUDA 环境下跳过测试
    # 设置测试数据类型为双精度和复数双精度



    def test_matrix_power_non_negative(self, device, dtype):



    # 定义测试函数，用于测试矩阵幂的非负性



        def check(*size):



        # 定义内部函数 check，用于检查给定大小的张量



            t = make_tensor(size, dtype=dtype, device=device)



            # 创建指定大小、指定数据类型和设备的张量 t



            for n in range(8):



            # 对于 n 从 0 到 7 的循环



                res = torch.linalg.matrix_power(t, n)



                # 计算张量 t 的 n 次幂，结果存储在 res 中



                ref = np.linalg.matrix_power(t.cpu().numpy(), n)



                # 使用 NumPy 计算张量 t 的 n 次幂的参考结果，存储在 ref 中



                self.assertEqual(res.cpu(), torch.from_numpy(ref))



                # 断言：验证计算得到的 res（PyTorch 张量）与 ref（NumPy 数组）的值相等



        check(0, 0)
        check(1, 1)
        check(5, 5)
        check(0, 3, 3)
        check(2, 3, 3)



        # 对于不同大小的矩阵进行测试
        # 分别测试：0x0 矩阵，1x1 矩阵，5x5 矩阵，0x3x3 张量，2x3x3 张量



    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(torch.double, torch.cdouble)



    # 跳过不支持 LAPACK 的 CPU 环境下的测试
    # 在没有 MAGMA 和 CUSOLVER 的 CUDA 环境下跳过测试
    # 设置测试数据类型为双精度和复数双精度
    # 定义一个测试函数，用于测试负幂的矩阵幂运算
    def test_matrix_power_negative(self, device, dtype):
        # 从工厂函数中获取生成具有不同奇异值的全秩矩阵的函数
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        # 创建部分应用了设备和数据类型的工厂函数
        make_arg = partial(make_fullrank, device=device, dtype=dtype)

        # 定义内部函数用于检查不同大小的矩阵
        def check(*size):
            # 调用工厂函数生成具有给定大小的矩阵 t
            t = make_arg(*size)
            # 对于范围在 -7 到 -1 之间的每个整数 n
            for n in range(-7, 0):
                # 计算矩阵 t 的 n 次幂
                res = torch.linalg.matrix_power(t, n)
                # 使用 NumPy 计算矩阵 t 的 n 次幂作为参考结果
                ref = np.linalg.matrix_power(t.cpu().numpy(), n)
                # 断言 PyTorch 结果与 NumPy 结果相等
                self.assertEqual(res.cpu(), torch.from_numpy(ref))

        # 对不同大小的矩阵调用检查函数，测试其负幂矩阵幂运算
        check(0, 0)
        check(5, 5)
        check(2, 0, 0)
        check(0, 3, 3)
        check(2, 3, 3)
        check(2, 3, 5, 5)

    # 应用装饰器，指定条件下跳过 CUDA 测试
    @skipCUDAIfNoMagma
    # 应用装饰器，指定条件下跳过无 LAPACK 支持的 CPU 测试
    @skipCPUIfNoLapack
    # 指定数据类型为 torch.float 和 torch.complex64 的测试函数
    @dtypes(torch.float, torch.complex64)
    # 测试 linalg_matrix_exp_utils 功能的函数
    def test_linalg_matrix_exp_utils(self, device, dtype):
        # 测试线性组合功能
        def run_test(coeff_shape, data_shape):
            # 生成随机系数，形状为 coeff_shape，位于指定设备上
            coeffs = torch.rand(*coeff_shape, device=device, dtype=torch.float)
            # 生成随机数据 x，形状为 coeff_shape[1] 和 data_shape，并位于指定设备上，使用指定数据类型
            x = torch.rand(coeff_shape[1], *data_shape, device=device, dtype=dtype)

            # 计算线性组合的结果 res1
            res1 = torch._compute_linear_combination(x, coeffs)
            # 使用显式计算的方式计算线性组合的结果 res2
            res2 = (x.unsqueeze(0) * coeffs.view(*coeff_shape, *([1] * len(data_shape)))).sum(1)
            # 断言 res1 与 res2 相等，使用给定的绝对误差和相对误差容差
            self.assertEqual(res1, res2, atol=1e-5, rtol=0.0)

            # 检查带有 `out=` 参数的版本
            res3 = torch.zeros(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            # 将线性组合的结果存储到 res3 中
            torch._compute_linear_combination(x, coeffs, out=res3)
            # 断言 res1 与 res3 相等，使用给定的绝对误差和相对误差容差
            self.assertEqual(res1, res3, atol=1e-5, rtol=0.0)

            # 检查对 `out=` 参数应用 1.0 的版本
            res4 = torch.ones(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            # 将线性组合的结果存储到 res4 中，并将结果减去 1.0
            torch._compute_linear_combination(x, coeffs, out=res4)
            # 断言 res1 与 res4-1.0 相等，使用给定的绝对误差和相对误差容差
            self.assertEqual(res1, res4 - 1.0, atol=1e-5, rtol=0.0)

            # 检查对 `out=` 参数应用 1.0 的版本
            res5 = torch.ones(coeff_shape[0], *data_shape, device=device, dtype=dtype)
            res5_clone = res5.clone()
            # 将线性组合的结果存储到 res5 中，并将结果减去 res5_clone
            torch._compute_linear_combination(x, coeffs, out=res5)
            # 断言 res1 与 res5-res5_clone 相等，使用给定的绝对误差和相对误差容差
            self.assertEqual(res1, res5 - res5_clone, atol=1e-5, rtol=0.0)

        # 运行一系列测试，用不同的系数和数据形状来测试线性组合功能
        run_test([1, 3], [2, 2])
        run_test([3, 1], [2, 2])
        run_test([1, 10], [10, 10])
        run_test([10, 1], [10, 10])
        run_test([5, 3], [2, 2])
        run_test([5, 3], [100, 100])
        run_test([3, 4], [3, 3, 3])
        run_test([3, 4], [3, 3, 3, 3])

        # 对 GitHub 上的特定问题进行回归测试
        with self.assertRaises(RuntimeError):
            # 生成一个随机张量 x，形状为 []，位于指定设备上，使用指定数据类型
            x = torch.rand([], device=device, dtype=dtype)
            # 生成一个随机系数矩阵 coeffs，形状为 [2, 2]，位于指定设备上，使用指定数据类型
            coeffs = torch.rand([2, 2], device=device, dtype=dtype)
            # 调用线性组合函数，并断言会引发 RuntimeError 异常
            res = torch._compute_linear_combination(x, coeffs)

    # 应用装饰器，指定只在 CPU 上运行测试
    @onlyCPU
    # 应用装饰器，指定条件下跳过无 LAPACK 支持的 CPU 测试
    @skipCPUIfNoLapack
    # 指定数据类型为 torch.complex64 的测试函数
    @dtypes(torch.complex64)
    # 定义一个测试函数，用于测试 matrix_exp 函数在不产生警告的情况下的行为
    def test_linalg_matrix_exp_no_warnings(self, device, dtype):
        # 用于测试 https://github.com/pytorch/pytorch/issues/80948
        with freeze_rng_state():
            # 设置随机种子为 42
            torch.manual_seed(42)
            # 创建一个形状为 (10, 3, 3) 的张量，元素为 dtype 类型的随机数乘以 0.5，放置在指定的设备上
            tens = 0.5 * torch.randn(10, 3, 3, dtype=dtype, device=device)
            # 将张量的最后两个维度进行转置，并对应位置元素相加后乘以 0.5，更新原张量
            tens = (0.5 * (tens.transpose(-1, -2) + tens))
            # 捕获警告信息
            with warnings.catch_warnings(record=True) as w:
                # 将张量的虚部用 matrix_exp 函数处理
                tens.imag = torch.matrix_exp(tens.imag)
                # 断言警告信息列表长度为 0
                self.assertFalse(len(w))

    # 使用装饰器指定测试条件：当没有支持 CUDA 上的 MAGMA 或没有支持 LAPACK 的 CPU 时跳过测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    # 使用装饰器指定测试函数适用的数据类型
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    # 定义一个测试函数，测试 matrix_exp 函数在边界情况下的行为
    def test_linalg_matrix_exp_boundary_cases(self, device, dtype):
        # 将 torch.linalg.matrix_exp 赋值给 expm 变量
        expm = torch.linalg.matrix_exp

        # 断言运行时错误中包含特定字符串，检查是否传入整数类型的张量
        with self.assertRaisesRegex(RuntimeError, "Expected a floating point or complex tensor"):
            expm(torch.randn(3, 3).type(torch.int))

        # 断言运行时错误中包含特定字符串，检查是否传入至少二维的张量
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            expm(torch.randn(3))

        # 断言运行时错误中包含特定字符串，检查是否传入批次中不是方阵的张量
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            expm(torch.randn(3, 2, 1))

        # 检查 1x1 矩阵的情况
        x = torch.randn(3, 3, 1, 1)
        # 断言 expm(x) 与 x.exp() 结果相等
        self.assertEqual(expm(x), x.exp())

    # 使用装饰器指定测试条件：当没有支持 CUDA 上的 MAGMA 或没有支持 LAPACK 的 CPU 时跳过测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    # 使用装饰器指定测试函数适用的数据类型
    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    # 定义一个测试函数，测试 matrix_exp 函数在包含 NaN 值的特殊情况下的行为
    def test_linalg_matrix_exp_perverse_nan_values(self, device, dtype):
        # 将 torch.linalg.matrix_exp 赋值给 expm 变量
        expm = torch.linalg.matrix_exp

        # 定义一个函数，用于将张量中的第一个元素置为 NaN
        def with_nan(x):
            x[0, 0, 0] = torch.nan
            return x

        # 检查小批次的情况
        x = with_nan(torch.randn(1, 1, 1))
        # 断言 expm(x) 中是否存在 NaN 值
        self.assertTrue(torch.isnan(expm(x)).any())
        x = with_nan(torch.randn(1, 2, 2))
        # 对多个缩放因子进行测试，确保 expm(x / v) 中是否存在 NaN 值
        for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 100, 1000]:
            self.assertTrue(torch.isnan(expm(x / v)).any())

        # 检查大批次的情况
        x = with_nan(torch.randn(2, 2, 2))
        # 断言 expm(x) 中是否存在 NaN 值
        self.assertTrue(torch.isnan(expm(x)).any())
        x = with_nan(torch.randn(4096, 2, 2))
        # 断言 expm(x) 中是否存在 NaN 值
        self.assertTrue(torch.isnan(expm(x)).any())

    # 使用装饰器标记为一个耗时的测试
    @slowTest
    # 使用装饰器指定测试条件：当没有支持 CUDA 上的 MAGMA 或没有支持 LAPACK 的 CPU 时跳过测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    # 使用装饰器指定测试函数适用的数据类型
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 使用装饰器指定测试条件：当没有支持 CUDA 上的 MAGMA 或没有支持 LAPACK 的 CPU 时跳过测试
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    # 使用装饰器指定测试函数适用的数据类型
    @dtypes(torch.float, torch.double)
    # 定义一个测试方法，用于批量计算矩阵指数
    def test_linalg_matrix_exp_batch(self, device, dtype):

        # 定义内部函数，用于运行测试
        def run_test(*n):
            # 创建全零张量批次，并指定设备和数据类型
            tensors_batch = torch.zeros(n, dtype=dtype, device=device)
            # 重新视图化张量形状为 (num_matrices, n[-2], n[-1])
            tensors_batch = tensors_batch.view(-1, n[-2], n[-1])

            # 计算批次中的矩阵数量
            num_matrices = tensors_batch.size(0)
            tensors_list = []
            # 循环生成随机张量，并添加到列表中
            for i in range(num_matrices):
                tensors_list.append(torch.randn(n[-2], n[-1], dtype=dtype, device=device))

            # 将生成的随机张量分配给批次中的每个矩阵
            for i in range(num_matrices):
                tensors_batch[i, ...] = tensors_list[i]

            # 使用生成器表达式计算每个随机矩阵的指数映射
            tensors_exp_map = (torch.linalg.matrix_exp(x) for x in tensors_list)
            # 计算整个批次张量的指数映射
            tensors_exp_batch = torch.linalg.matrix_exp(tensors_batch)

            # 遍历生成的指数映射结果并进行断言比较
            for i, tensor_exp in enumerate(tensors_exp_map):
                self.assertEqual(tensors_exp_batch[i, ...], tensor_exp)

        # 对小批量矩阵进行测试
        run_test(3, 2, 2)
        run_test(3, 3, 3)
        run_test(3, 4, 4)
        run_test(3, 5, 5)

        # 对大批量矩阵进行测试
        run_test(3, 3, 2, 2)
        run_test(3, 3, 3, 3)
        run_test(3, 3, 4, 4)
        run_test(3, 3, 5, 5)

    # 跳过如果没有Magma加速库
    @skipCUDAIfNoMagma
    # 跳过如果没有LAPACK线性代数库
    @skipCPUIfNoLapack
    # 为torch.float, torch.double, torch.cfloat, torch.cdouble数据类型进行测试
    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    # 跳过如果没有Magma加速库
    @skipCUDAIfNoMagma
    # 跳过如果没有LAPACK线性代数库
    @skipCPUIfNoLapack
    # 为浮点数和复数类型的数据类型进行测试
    @dtypes(*floating_and_complex_types())
    # 设置精度覆盖，定义每种数据类型的比较精度阈值
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 跳过如果没有Magma加速库
    @skipCUDAIfNoMagma
    # 跳过如果没有LAPACK线性代数库
    @skipCPUIfNoLapack
    # 为浮点数和复数类型的数据类型进行测试
    @dtypes(*floating_and_complex_types())
    # 测试 slogdet 函数在不同情况下的错误和警告信息
    def test_slogdet_errors_and_warnings(self, device, dtype):
        # slogdet 要求输入是一个方阵或者一批方阵
        a = torch.randn(2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must be batches of square matrices'):
            torch.linalg.slogdet(a)

        # slogdet 要求输入至少是一个二维张量
        a = torch.randn(2, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, r'must have at least 2 dimensions'):
            torch.linalg.slogdet(a)

        # 不支持低精度的数据类型
        a = torch.randn(2, 2, device=device, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, r'Low precision dtypes not supported'):
            torch.linalg.slogdet(a)

        # 如果传入了形状不对的非空输出张量，会发出警告
        a = torch.randn(2, 3, 3, device=device, dtype=dtype)
        sign_out = torch.empty(1, device=device, dtype=dtype)
        real_dtype = a.real.dtype if dtype.is_complex else dtype
        logabsdet_out = torch.empty(1, device=device, dtype=real_dtype)
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.linalg.slogdet(a, out=(sign_out, logabsdet_out))
            # 检查是否发生了警告
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 设备应该匹配
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            sign_out = torch.empty(0, device=wrong_device, dtype=dtype)
            logabsdet_out = torch.empty(0, device=wrong_device, dtype=real_dtype)
            with self.assertRaisesRegex(RuntimeError, "tensors to be on the same device"):
                torch.linalg.slogdet(a, out=(sign_out, logabsdet_out))
    # 定义测试函数 test_cholesky_inverse，接受设备和数据类型作为参数
    def test_cholesky_inverse(self, device, dtype):
        # 导入需要的库函数
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix

        # 定义内部函数 run_test，用于执行单个测试
        def run_test(shape, batch, upper, contiguous):
            # 生成随机的 Hermitian 正定矩阵 A
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            # 如果 A 不是连续的，则转置并断言其不连续
            if A.numel() > 0 and not contiguous:
                A = A.mT
                self.assertFalse(A.is_contiguous())
            # 对 A 执行 Cholesky 分解得到下三角矩阵 L
            L = torch.linalg.cholesky(A)
            # 计算预期的逆矩阵
            expected_inverse = torch.inverse(A)
            # 根据 upper 参数选择 L 是否转置
            L = L.mH if upper else L
            # 计算实际的逆矩阵
            actual_inverse = torch.cholesky_inverse(L, upper)
            # 断言实际逆矩阵与预期逆矩阵相等
            self.assertEqual(actual_inverse, expected_inverse)

        # 定义不同的测试参数组合
        shapes = (0, 3, 5)
        batches = ((), (0,), (3, ), (2, 2))
        # 对所有可能的参数组合进行测试
        for shape, batch, upper, contiguous in list(itertools.product(shapes, batches, (True, False), (True, False))):
            run_test(shape, batch, upper, contiguous)

        # 检查 out= 变体
        # 生成一个随机的 Hermitian 正定矩阵 A
        A = random_hermitian_pd_matrix(3, 2, dtype=dtype, device=device)
        # 对 A 执行 Cholesky 分解得到下三角矩阵 L
        L = torch.linalg.cholesky(A)

        # 下面是对两种 out= 变体的测试路径的检查
        # 第一种代码路径的测试
        # 当 'out' 张量以 Fortran（列主内存）格式存在时，采用快速路径，直接重用存储在计算中
        out = torch.empty_like(A)
        out_t = out.mT.clone(memory_format=torch.contiguous_format)
        out = out_t.mT
        ans = torch.cholesky_inverse(L, out=out)
        # 断言计算得到的 ans 与 out 相等
        self.assertEqual(ans, out)
        # 计算预期的逆矩阵
        expected = torch.inverse(A)
        # 断言预期的逆矩阵与 out 相等
        self.assertEqual(expected, out)

        # 第二种代码路径的测试
        out = torch.empty_like(A)
        # 执行逆矩阵计算，并将结果存储到 out 中
        ans = torch.cholesky_inverse(L, out=out)
        # 断言计算得到的 ans 与 out 相等
        self.assertEqual(ans, out)
        # 计算预期的逆矩阵
        expected = torch.inverse(A)
        # 断言预期的逆矩阵与 out 相等
        self.assertEqual(expected, out)

    # 以下三个装饰器标记特定的测试条件
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(*floating_and_complex_types())
    # 定义测试函数 test_cholesky_inverse_errors_and_warnings，接受设备和数据类型参数
    def test_cholesky_inverse_errors_and_warnings(self, device, dtype):
        # cholesky_inverse 要求输入至少是二维张量
        a = torch.randn(2, device=device, dtype=dtype)
        # 使用断言检测是否引发 RuntimeError，提示至少需要两个维度
        with self.assertRaisesRegex(RuntimeError, "must have at least 2 dimensions"):
            torch.cholesky_inverse(a)

        # cholesky_inverse 要求输入是方阵
        a = torch.randn(2, 3, device=device, dtype=dtype)
        # 使用断言检测是否引发 RuntimeError，提示输入必须是批量的方阵
        with self.assertRaisesRegex(RuntimeError, "must be batches of square matrices"):
            torch.cholesky_inverse(a)

        # 如果传递了形状错误的非空输出张量，则会发出警告
        a = torch.randn(3, 3, device=device, dtype=dtype)
        out = torch.empty(2, 3, device=device, dtype=dtype)
        with warnings.catch_warnings(record=True) as w:
            # 触发警告
            torch.cholesky_inverse(a, out=out)
            # 检查是否有警告发生
            self.assertEqual(len(w), 1)
            self.assertTrue("An output with one or more elements was resized" in str(w[-1].message))

        # 数据类型应该是安全可转换的
        out = torch.empty(*a.shape, dtype=torch.int, device=device)
        with self.assertRaisesRegex(RuntimeError, "but got result with dtype Int"):
            torch.cholesky_inverse(a, out=out)

        # 设备应该匹配
        if torch.cuda.is_available():
            wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'
            out = torch.empty(0, device=wrong_device, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                torch.cholesky_inverse(a, out=out)

        # 在 CPU 上，cholesky_inverse 对无效输入会引发错误
        # 例如，如果至少一个对角元素为零
        a = torch.randn(3, 3, device=device, dtype=dtype)
        a[1, 1] = 0
        if self.device_type == 'cpu':
            with self.assertRaisesRegex(torch.linalg.LinAlgError, r"cholesky_inverse: The diagonal element 2 is zero"):
                torch.cholesky_inverse(a)
        # 在 GPU 上，cholesky_inverse 对这种情况不会引发错误
        elif self.device_type == 'cuda':
            out = torch.cholesky_inverse(a)
            self.assertTrue(out.isinf().any() or out.isnan().any())
    # 选择可广播维度的辅助函数，根据给定的dims_full参数（如果为None则随机生成），返回三个维度元组
    def _select_broadcastable_dims(self, dims_full=None):
        # 如果dims_full为None，则初始化为空列表，并随机生成维度数和每个维度的大小
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # 选择操作的实际维度：
        # 较大情况：保持完整的维度数，但各维度大小可能会减小
        # 较小情况：维度数可能减少，且各维度大小可能会减小
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # 没有减少的单一维度
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # 较大情况可能会有减少的单一维度
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # 较小情况可能会有减少的单一维度
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    # 对"fused_matmul"操作进行广播测试的方法，传入设备参数
    def test_broadcast_fused_matmul(self, device):
        # 所有可能的函数名列表
        fns = ["baddbmm", "addbmm", "addmm", "addmv", "addr"]

        # 对每个函数名循环进行测试
        for fn in fns:
            # 随机生成批次维度和各个维度大小
            batch_dim = random.randint(1, 8)
            n_dim = random.randint(1, 8)
            m_dim = random.randint(1, 8)
            p_dim = random.randint(1, 8)

            # 根据函数名返回适合该函数的完整维度
            def dims_full_for_fn():
                if fn == "baddbmm":
                    return ([batch_dim, n_dim, p_dim], [batch_dim, n_dim, m_dim], [batch_dim, m_dim, p_dim])
                elif fn == "addbmm":
                    return ([n_dim, p_dim], [batch_dim, n_dim, m_dim], [batch_dim, m_dim, p_dim])
                elif fn == "addmm":
                    return ([n_dim, p_dim], [n_dim, m_dim], [m_dim, p_dim])
                elif fn == "addmv":
                    return ([n_dim], [n_dim, m_dim], [m_dim])
                elif fn == "addr":
                    return ([n_dim, m_dim], [n_dim], [m_dim])
                else:
                    raise AssertionError("unknown function")

            # 获取函数适合的完整维度元组
            (t0_dims_full, t1_dims, t2_dims) = dims_full_for_fn()
            # 使用_select_broadcastable_dims方法获取可广播的小维度和完整维度
            (t0_dims_small, _, _) = self._select_broadcastable_dims(t0_dims_full)

            # 在指定设备上生成随机张量，数据类型为float
            t0_small = torch.randn(*t0_dims_small, device=device).float()
            t1 = torch.randn(*t1_dims, device=device).float()
            t2 = torch.randn(*t2_dims, device=device).float()

            # 将小维度张量扩展到完整维度，并转移到指定设备上
            t0_full = t0_small.expand(*t0_dims_full).to(device)

            # 获取torch库中对应函数名的函数对象
            fntorch = getattr(torch, fn)
            # 分别对小维度和完整维度的张量进行函数操作
            r0 = fntorch(t0_small, t1, t2)
            r1 = fntorch(t0_full, t1, t2)
            # 断言两种操作结果相等
            self.assertEqual(r0, r1)

    # 应用tf32_on_and_off和bf32_on_and_off装饰器，传入0.001作为参数
    @tf32_on_and_off(0.001)
    @bf32_on_and_off(0.001)
    # 定义一个帮助函数，用于测试 LU 分解求解器的功能
    def lu_solve_test_helper(self, A_dims, b_dims, pivot, device, dtype):
        # 使用带有独特奇异值的全秩矩阵生成器函数
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        # 部分应用生成器函数，固定设备和数据类型
        make_A = partial(make_fullrank, device=device, dtype=dtype)

        # 生成随机张量 b，设备为指定设备，数据类型为指定数据类型
        b = torch.randn(*b_dims, dtype=dtype, device=device)
        # 生成指定维度的全秩矩阵 A
        A = make_A(*A_dims)
        # 对矩阵 A 进行 LU 分解，返回 LU 因子、置换信息以及操作信息
        LU_data, LU_pivots, info = torch.linalg.lu_factor_ex(A)
        # 断言操作信息为零张量
        self.assertEqual(info, torch.zeros_like(info))
        # 返回生成的张量 b、矩阵 A、LU 分解数据以及置换信息
        return b, A, LU_data, LU_pivots

    # 标记为在没有 LAPACK 支持时跳过测试
    @skipCPUIfNoLapack
    # 标记为在没有 MAGMA 和 CUSOLVER 支持时跳过 CUDA 测试
    @skipCUDAIfNoMagmaAndNoCusolver
    # 应用于浮点数和复数类型的数据类型装饰器
    @dtypes(*floating_and_complex_types())
    # 精度覆盖装饰器，设置不同数据类型的精度
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # LU 解测试函数
    def test_lu_solve(self, device, dtype):
        # 定义 LU 解的子测试函数，接受置换标志作为参数
        def sub_test(pivot):
            # 对于 k 和 n 的每一对值，执行 LU 解测试助手函数
            for k, n in zip([2, 3, 5], [3, 5, 7]):
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper((n, n), (n, k), pivot, device, dtype)
                # 使用 LU 分解解算出 x
                x = torch.lu_solve(b, LU_data, LU_pivots)
                # 断言解 x 与 np.matmul(A.cpu(), x.cpu()) 的乘积等于 b
                self.assertEqual(b, np.matmul(A.cpu(), x.cpu()))

        # 对于置换为真的情况进行子测试
        sub_test(True)
        # 如果设备类型为 CUDA，则对置换为假的情况进行子测试
        if self.device_type == 'cuda':
            sub_test(False)

    # 标记为在没有 LAPACK 支持时跳过测试
    @skipCPUIfNoLapack
    # 标记为在没有 MAGMA 和 CUSOLVER 支持时跳过 CUDA 测试
    @skipCUDAIfNoMagmaAndNoCusolver
    # 应用于浮点数和复数类型的数据类型装饰器
    @dtypes(*floating_and_complex_types())
    # 精度覆盖装饰器，设置不同数据类型的精度
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 批量 LU 解测试函数
    def test_lu_solve_batched(self, device, dtype):
        # 定义批量 LU 解的子测试函数，接受置换标志作为参数
        def sub_test(pivot):
            # 定义 LU 解批量测试助手函数，接受 A 和 b 的维度以及置换标志作为参数
            def lu_solve_batch_test_helper(A_dims, b_dims, pivot):
                # 调用 LU 解测试助手函数，返回 b、A、LU 数据和置换信息
                b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, pivot, device, dtype)
                x_exp_list = []
                # 对于每个 batch 中的元素，使用 LU 解计算 x 并存储在列表中
                for i in range(b_dims[0]):
                    x_exp_list.append(torch.lu_solve(b[i], LU_data[i], LU_pivots[i]))
                # 将计算得到的 x 堆叠起来形成张量
                x_exp = torch.stack(x_exp_list)  # Stacked output
                # 计算真实的 x
                x_act = torch.lu_solve(b, LU_data, LU_pivots)  # Actual output
                # 断言预期的 x 等于真实的 x
                self.assertEqual(x_exp, x_act)  # Equality check
                # 计算 Ax，其中 A 为 CPU 上的矩阵
                Ax = np.matmul(A.cpu(), x_act.cpu())
                # 断言 b 等于 Ax
                self.assertEqual(b, Ax)

            # 对于批处理大小为 1、3 和 4 的每个情况，执行 LU 解批量测试助手函数
            for batchsize in [1, 3, 4]:
                lu_solve_batch_test_helper((batchsize, 5, 5), (batchsize, 5, 10), pivot)

        # 测试元素数为 0 的张量
        b = torch.randn(3, 0, 3, dtype=dtype, device=device)
        A = torch.randn(3, 0, 0, dtype=dtype, device=device)
        # 对 A 执行 LU 分解
        LU_data, LU_pivots = torch.linalg.lu_factor(A)
        # 断言空张量等于 b 的 LU 解
        self.assertEqual(torch.empty_like(b), b.lu_solve(LU_data, LU_pivots))

        # 对置换为真的情况进行子测试
        sub_test(True)
        # 如果设备类型为 CUDA，则对置换为假的情况进行子测试
        if self.device_type == 'cuda':
            sub_test(False)

    # 标记为慢速测试
    @slowTest
    # 标记为在没有 LAPACK 支持时跳过测试
    @skipCPUIfNoLapack
    # 标记为在没有 MAGMA 和 CUSOLVER 支持时跳过 CUDA 测试
    @skipCUDAIfNoMagmaAndNoCusolver
    # 应用于浮点数和复数类型的数据类型装饰器
    @dtypes(*floating_and_complex_types())
    # 测试批量解线性方程组的 LU 分解求解器在多个批次情况下的功能
    def test_lu_solve_batched_many_batches(self, device, dtype):
        # 定义内部函数，运行测试
        def run_test(A_dims, b_dims):
            # 使用辅助函数获取 LU 分解所需的矩阵 b、A、LU 数据和 LU 唯一标识
            b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, True, device, dtype)
            # 使用 Torch 提供的 lu_solve 函数解线性方程组
            x = torch.lu_solve(b, LU_data, LU_pivots)
            # 计算 Ax = A * x，要求其与 b 扩展后的形状相等
            Ax = torch.matmul(A, x)
            self.assertEqual(Ax, b.expand_as(Ax))

        # 运行测试，使用不同的维度来测试功能
        run_test((65536, 5, 5), (65536, 5, 10))
        run_test((262144, 5, 5), (262144, 5, 10))

    @skipCPUIfNoLapack
    @skipCUDAIfNoMagmaAndNoCusolver
    @dtypes(*floating_and_complex_types())
    # 测试批量解线性方程组的 LU 分解求解器支持广播的情况
    def test_lu_solve_batched_broadcasting(self, device, dtype):
        # 使用 make_fullrank 函数创建满秩矩阵
        make_fullrank = make_fullrank_matrices_with_distinct_singular_values
        make_A = partial(make_fullrank, device=device, dtype=dtype)

        # 定义内部函数，运行测试
        def run_test(A_dims, b_dims, pivot=True):
            # 获取 A 矩阵的尺寸信息
            A_matrix_size = A_dims[-1]
            # 获取 A 矩阵批处理维度
            A_batch_dims = A_dims[:-2]
            # 使用 make_A 函数创建矩阵 A 和 tensor b
            A = make_A(*A_batch_dims, A_matrix_size, A_matrix_size)
            b = make_tensor(b_dims, dtype=dtype, device=device)
            # 使用 numpy.linalg.solve 计算期望的解 x_exp
            x_exp = np.linalg.solve(A.cpu(), b.cpu())
            # 使用 Torch 提供的 lu_factor 函数进行 LU 分解
            LU_data, LU_pivots = torch.linalg.lu_factor(A)
            # 使用 Torch 提供的 lu_solve 函数解线性方程组
            x = torch.lu_solve(b, LU_data, LU_pivots)
            # 断言计算得到的解 x 与期望解 x_exp 相等
            self.assertEqual(x, x_exp)

        # 对不同的情况运行测试，测试不同的广播方式
        run_test((2, 1, 3, 4, 4), (2, 1, 3, 4, 6))  # no broadcasting
        run_test((2, 1, 3, 4, 4), (4, 6))  # broadcasting b
        run_test((4, 4), (2, 1, 3, 4, 2))  # broadcasting A
        run_test((1, 3, 1, 4, 4), (2, 1, 3, 4, 5))  # broadcasting A & b

    @onlyCUDA
    @skipCUDAIfNoMagma
    @dtypes(*floating_and_complex_types())
    # 测试在 CUDA 设备上解大尺寸矩阵线性方程组的 LU 分解求解器的功能
    # 这个测试处理 https://github.com/pytorch/pytorch/issues/36921
    def test_lu_solve_large_matrices(self, device, dtype):
        # 定义内部函数，运行测试
        def run_test(A_dims, b_dims):
            # 使用辅助函数获取 LU 分解所需的矩阵 b、A、LU 数据和 LU 唯一标识
            b, A, LU_data, LU_pivots = self.lu_solve_test_helper(A_dims, b_dims, True, device, dtype)
            # 使用 Torch 提供的 lu_solve 函数解线性方程组
            x = torch.lu_solve(b, LU_data, LU_pivots)
            # 计算 Ax = A * x，要求其与 b 扩展后的形状相等
            Ax = torch.matmul(A, x)
            self.assertEqual(Ax, b.expand_as(Ax))

        # 运行测试，测试解大尺寸矩阵的功能
        run_test((1, 1), (1, 1, 1025))

    @skipCUDAIfNoCusolver
    @skipCPUIfNoLapack
    # 确保 nuclear_norm 的 out 变体提供与非 out 变体相同的结果
    @onlyNativeDeviceTypes
    @skipCUDAIfNoMagma
    @skipCPUIfNoLapack
    @dtypes(torch.float32, torch.float64)
    # 定义测试函数 test_nuclear_norm_out，测试 nuclear_norm 函数的输出
    def test_nuclear_norm_out(self, device, dtype):
        # 定义测试用例，每个元素包含输入大小和指定的维度
        test_cases = [
            # input size, dim
            ((25, 25), None),
            ((25, 25), (0, 1)),
            ((25, 25), (1, 0)),
            ((25, 25, 25), (2, 0)),
            ((25, 25, 25), (0, 1)),
        ]
        # 遍历 keepdim 的两种可能取值
        for keepdim in [False, True]:
            # 遍历测试用例
            for input_size, dim in test_cases:
                # 生成描述信息
                msg = f'input_size: {input_size}, dim: {dim}, keepdim: {keepdim}'
                # 生成随机张量 x，指定设备和数据类型
                x = torch.randn(*input_size, device=device, dtype=dtype)
                # 创建一个空张量 result_out，指定设备和数据类型
                result_out = torch.empty(0, device=device, dtype=dtype)
                # 根据 dim 是否为 None，调用 torch.nuclear_norm 函数并将结果存入 result 或 result_out
                if dim is None:
                    result = torch.nuclear_norm(x, keepdim=keepdim)
                    torch.nuclear_norm(x, keepdim=keepdim, out=result_out)
                else:
                    result = torch.nuclear_norm(x, keepdim=keepdim, dim=dim)
                    torch.nuclear_norm(x, keepdim=keepdim, dim=dim, out=result_out)
                # 断言 result 与 result_out 相等，否则输出 msg
                self.assertEqual(result, result_out, msg=msg)

    # 使用装饰器 skipCUDAIfNoMagmaAndNoCusolver 跳过没有 Magma 和 Cusolver 的 CUDA 测试
    # 使用装饰器 skipCPUIfNoLapack 跳过没有 Lapack 的 CPU 测试
    @skipCUDAIfNoMagmaAndNoCusolver
    @skipCPUIfNoLapack
    # 使用装饰器 dtypes(*floating_and_complex_types())，测试涉及浮点数和复数类型的所有情况
    def test_geqrf(self, device, dtype):

        # 定义内部函数 run_test，运行针对给定形状的测试
        def run_test(shape):
            # numpy.linalg.qr 使用 mode='raw' 执行与 torch.geqrf 相同的操作
            # 因此此测试与该函数进行比较
            A = make_tensor(shape, dtype=dtype, device=device)

            # numpy.linalg.qr 不适用于批量输入
            m, n = A.shape[-2:]
            tau_size = "n" if m > n else "m"
            np_dtype = A.cpu().numpy().dtype
            ot = [np_dtype, np_dtype]
            # 创建 numpy_geqrf_batched 函数，使用 numpy.vectorize 包装 np.linalg.qr
            # 使用 otypes 参数指定输出类型
            numpy_geqrf_batched = np.vectorize(
                lambda x: np.linalg.qr(x, mode='raw'),
                otypes=ot,
                signature=f'(m,n)->(n,m),({tau_size})')

            # 获取 numpy_geqrf_batched 对 A 进行操作的期望输出
            expected = numpy_geqrf_batched(A.cpu())
            # 使用 torch.geqrf 对 A 进行操作，获取实际输出
            actual = torch.geqrf(A)

            # numpy.linalg.qr 返回转置的结果
            # 断言 torch.geqrf 的结果与 numpy.linalg.qr 的预期结果匹配
            self.assertEqual(expected[0].swapaxes(-2, -1), actual[0])
            self.assertEqual(expected[1], actual[1])

        # 定义测试的批次和大小
        batches = [(), (0, ), (2, ), (2, 1)]
        ns = [5, 2, 0]
        # 使用 product 函数生成 batches 和 ns 的组合，并对每个组合调用 run_test 函数进行测试
        for batch, (m, n) in product(batches, product(ns, ns)):
            run_test((*batch, m, n))

    # 使用装饰器 skipCUDAIfNoMagma 跳过没有 Magma 的 CUDA 测试
    # 使用装饰器 skipCPUIfNoLapack 跳过没有 Lapack 的 CPU 测试
    def test_lapack_empty(self, device):
        # FIXME: 这些是 LAPACK 函数的一个选择 —— 我们需要一个通用的策略。
        # LAPACK 函数本身通常不支持零尺寸的维度，尽管 numpy/sci 通常有直接的包装器（如 lu_factor）和一个“做正确事情”的包装器（如 lu）。
        # 我们经常将我们的函数命名为 LAPACK 函数的相同名称，因此需要工作来命名 / 迁移到更好的包装器。

        def fn(torchfn, *args):
            # 对于每个参数 shape，如果是元组，则使用设备生成随机数并传递给 torchfn；否则直接传递 shape。
            return torchfn(*tuple(torch.randn(shape, device=device) if isinstance(shape, tuple) else shape
                                  for shape in args))

        # inverse, pinverse
        # 测试 torch.inverse 和 torch.pinverse 函数的输出形状是否符合预期
        self.assertEqual((0, 0), fn(torch.inverse, (0, 0)).shape)
        self.assertEqual((5, 0), fn(torch.pinverse, (0, 5)).shape)
        self.assertEqual((0, 5), fn(torch.pinverse, (5, 0)).shape)
        self.assertEqual((0, 0), fn(torch.pinverse, (0, 0)).shape)

        # det, logdet, slogdet
        # 测试 torch.det, torch.logdet 和 torch.slogdet 函数的输出是否符合预期
        self.assertEqual(torch.tensor(1., device=device), fn(torch.det, (0, 0)))
        self.assertEqual(torch.tensor(0., device=device), fn(torch.logdet, (0, 0)))
        self.assertEqual((torch.tensor(1., device=device), torch.tensor(0., device=device)),
                         fn(torch.slogdet, (0, 0)))

    @tf32_on_and_off(0.005)
    @bf32_on_and_off(0.005)
    def test_tensordot(self, device):
        # 创建张量 a 和 b
        a = torch.arange(60., device=device).reshape(3, 4, 5)
        b = torch.arange(24., device=device).reshape(4, 3, 2)

        # 使用 torch.tensordot 计算张量 c，与 numpy 的 np.tensordot 进行比较
        c = torch.tensordot(a, b, dims=([1, 0], [0, 1])).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=([1, 0], [0, 1])))
        self.assertEqual(c, cn)

        # 使用指定的输出张量 cout 计算张量 c，并与预期结果进行比较
        cout = torch.zeros((5, 2), device=device)
        torch.tensordot(a, b, dims=([1, 0], [0, 1]), out=cout).cpu()
        self.assertEqual(c, cout)

        # 创建张量 a 和 b，并使用 torch.tensordot 计算张量 c，与 numpy 的 np.tensordot 进行比较
        a = torch.randn(2, 3, 4, 5, device=device)
        b = torch.randn(4, 5, 6, 7, device=device)
        c = torch.tensordot(a, b, dims=2).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy(),
                                           axes=2))

        # 测试传入负数维度时是否会引发 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "expects dims >= 0"):
            torch.tensordot(a, b, dims=-1)

        self.assertEqual(c, cn)

        # 计算张量 c，并与 numpy 的 np.tensordot 进行比较
        c = torch.tensordot(a, b).cpu()
        cn = torch.from_numpy(np.tensordot(a.cpu().numpy(), b.cpu().numpy()))
        self.assertEqual(c, cn)

        # 使用 torch.tensordot 计算标量张量的张量积，并与 numpy 的 np.tensordot 进行比较
        a = torch.tensordot(torch.tensor(0.), torch.tensor(0.), 0)
        an = torch.from_numpy(np.tensordot(np.zeros((), dtype=np.float32), np.zeros((), dtype=np.float32), 0))
        self.assertEqual(a, an)
    # 跳过测试如果 CUDA 版本低于 11.3.1
    @skipCUDAIf(_get_torch_cuda_version() < (11, 4), "not available before CUDA 11.3.1")
    # 使用所有浮点和复数数据类型进行数据类型参数化测试
    @dtypes(*floating_and_complex_types())
    def test_ldl_solve(self, device, dtype):
        # 导入生成随机 Hermitian 正定矩阵的函数
        from torch.testing._internal.common_utils import random_hermitian_pd_matrix
    
        def run_test(shape, batch, nrhs, hermitian):
            # 创建随机 Hermitian 正定矩阵 A 和对应的张量 B
            A = random_hermitian_pd_matrix(shape, *batch, dtype=dtype, device=device)
            B = make_tensor((*A.shape[:-1], nrhs), dtype=dtype, device=device)
            # 使用 torch.linalg.ldl_factor_ex 函数计算 LDL 分解
            factors, pivots, info = torch.linalg.ldl_factor_ex(A, hermitian=hermitian)
            # 使用 torch.linalg.ldl_solve 函数求解线性方程组
            X = torch.linalg.ldl_solve(factors, pivots, B, hermitian=hermitian)
    
            # 定义一个函数检验对称性
            def symmetric(A):
                return A.tril() + A.tril(-1).mT
    
            # 验证 A @ X == B 是否成立
            expected_B = symmetric(A) @ X if not hermitian else A @ X
            self.assertEqual(B, expected_B)
    
        # 如果数据类型是复数并且设备类型为 CPU，则 hermitian=True 不被支持
        hermitians = (True, False) if dtype.is_complex and self.device_type == 'cpu' else (False,)
    
        shapes = (5,)  # 测试的矩阵形状
        batches = ((), (4,), (2, 2))  # 批次大小
        nrhss = (1, 7)  # 右手边向量的数量
        # 对形状、批次、右手边向量数量和是否 Hermitian 的所有组合进行测试
        for shape, batch, nrhs, hermitian in itertools.product(shapes, batches, nrhss, hermitians):
            run_test(shape, batch, nrhs, hermitian)
    
    # 仅对 CUDA 可用
    @onlyCUDA
    # 如果没有安装 magma，则跳过测试
    @skipCUDAIfNoMagma
    # 如果没有安装 cusolver，则跳过测试
    @skipCUDAIfNoCusolver
    # 在测试完成后将线性代数后端设置为默认值
    @setLinalgBackendsToDefaultFinally
    def test_preferred_linalg_library(self):
        # 此测试的主要目的是确保这些“backend”调用正常工作，不会引发异常。
        x = torch.randint(2, 5, (2, 4, 4), device='cuda', dtype=torch.double)
    
        # 设置 CUDA 的首选线性代数库为 'cusolver'
        torch.backends.cuda.preferred_linalg_library('cusolver')
        out1 = torch.linalg.inv(x)
    
        # 设置 CUDA 的首选线性代数库为 'magma'
        torch.backends.cuda.preferred_linalg_library('magma')
        out2 = torch.linalg.inv(x)
    
        # 恢复 CUDA 的首选线性代数库为默认值
        torch.backends.cuda.preferred_linalg_library('default')
        # 尽管目前 linalg 首选标志不影响 CPU，
        # 我们设置这个以确保标志可以正常切换回默认值。
        out_ref = torch.linalg.inv(x.cpu())
    
        # 检验结果 out_ref 与 out1.cpu() 相等
        self.assertEqual(out_ref, out1.cpu())
        # 检验结果 out1 与 out2 相等
        self.assertEqual(out1, out2)
    
    # 仅对 CUDA 可用
    @onlyCUDA
    # 如果当前设备不支持 blasLt，则跳过测试
    @unittest.skipIf(not blaslt_supported_device(), "blasLt not supported on current device")
    # 在测试完成后将 BLAS 后端设置为默认值
    @setBlasBackendsToDefaultFinally
    def test_preferred_blas_library(self):
        # 测试首选的 BLAS 库设置功能

        # 创建在 CUDA 设备上的随机张量 m1 和 m2
        m1 = torch.randint(2, 5, (2048, 2400), device='cuda', dtype=torch.float)
        m2 = torch.randint(2, 5, (128, 2400), device='cuda', dtype=torch.float)

        # 设置首选的 BLAS 库为 'cublaslt' 并进行线性运算
        torch.backends.cuda.preferred_blas_library('cublaslt')
        out1 = torch.nn.functional.linear(m1, m2)

        # 再次设置首选的 BLAS 库为 'cublas' 并进行线性运算
        torch.backends.cuda.preferred_blas_library('cublas')
        out2 = torch.nn.functional.linear(m1, m2)

        # 尽管 BLAS 首选标志目前不影响 CPU，但我们设置此项以确保标志能够正常切换回默认设置

        # 使用 CPU 计算参考结果 out_ref，以验证 CUDA 计算的正确性
        out_ref = torch.nn.functional.linear(m1.cpu(), m2.cpu())

        # 断言两个 CUDA 计算的结果相等
        self.assertEqual(out1, out2)
        # 断言参考结果与 CUDA 计算结果在 CPU 上的对应结果相等
        self.assertEqual(out_ref, out2.cpu())

    def test_permute_matmul(self):
        # 测试张量 permute 和 matmul 操作的组合

        # 创建张量 a 和 b
        a = torch.ones([2, 5, 24, 24])
        b = torch.ones([3, 2, 5, 24, 24])

        # 对张量 a 进行维度置换操作，并与张量 b 进行矩阵乘法运算
        c = a.permute(0, 1, 3, 2).matmul(b)

        # 断言结果张量 c 的最小值、最大值和总和分别为 24、24 和 414720
        self.assertEqual([c.min(), c.max(), c.sum()], [24, 24, 414720])
    # 定义一个测试函数，用于验证在 gemm 参考路径中修复了 bf16 精度累积的问题
    def test_lower_precision_accumulation_with_ref_path(self):
        # 解决 https://github.com/pytorch/pytorch/issues/95125 和 https://github.com/pytorch/pytorch/issues/83863
        # 用于 gemm 参考路径中的 bf16 累积
        def check_correctness(fn, dtype, *args):
            # 生成预期的张量结果，将其转换为指定的数据类型
            expected = fn(*args).to(dtype=dtype)
            # 使用 mkldnn 标志禁用，执行测试函数以获取临时结果
            with torch.backends.mkldnn.flags(enabled=False):
                def test():
                    # 将所有参数转换为指定的数据类型
                    lower_args = (arg.to(dtype=dtype) for arg in args)
                    tmp_result = fn(*lower_args)
                    return tmp_result
                # 执行测试函数，检查是否与预期结果一致
                c = test()
                assert (torch.all(c == expected)), "Incorrect result with\n" \
                                                   f"expected: {expected}\n" \
                                                   f"got: {c}\n"
        
        # 测试 matmul 函数
        for dtype in [torch.bfloat16, torch.half]:
            for transa in [True, False]:
                for transb in [True, False]:
                    a = torch.ones(300, 300)
                    b = torch.ones(300, 300)
                    # 根据 transa 和 transb 的值，对矩阵进行转置和重排列
                    if transa:
                        a = a.transpose(0, 1).contiguous().transpose(0, 1)
                    if transb:
                        b = b.transpose(0, 1).contiguous().transpose(0, 1)
                    # 执行检查函数，验证 matmul 函数的正确性
                    check_correctness(torch.matmul, dtype, a, b)
        
        # 测试 bmm 函数
        a = torch.ones(1, 1, 300)
        b = torch.ones(1, 300, 1)
        # 执行检查函数，验证 bmm 函数的正确性
        check_correctness(torch.bmm, torch.bfloat16, a, b)
        check_correctness(torch.bmm, torch.half, a, b)
        
        # 测试 baddbmm 函数
        a = torch.ones(1, 1, 300)
        b = torch.ones(1, 300, 1)
        c = torch.ones(1, 1, 1)
        # 执行检查函数，验证 baddbmm 函数的正确性
        check_correctness(torch.baddbmm, torch.bfloat16, c, a, b)
        check_correctness(torch.baddbmm, torch.half, c, a, b)
        
        # 测试 mv/addmv 函数
        for dtype in [torch.bfloat16, torch.half]:
            for trans in [True, False]:
                c = torch.ones(300) * -300
                a = torch.ones(300, 300)
                # 根据 trans 的值，对矩阵进行转置和重排列
                if trans:
                    a = a.transpose(0, 1).contiguous().transpose(0, 1)
                b = torch.ones(300)
                # 执行检查函数，验证 mv 和 addmv 函数的正确性
                check_correctness(torch.mv, dtype, a, b)
                check_correctness(torch.addmv, dtype, c, a, b)
        
        # 测试 dot 函数
        a = torch.ones(300)
        b = torch.ones(300)
        # 执行检查函数，验证 dot 函数的正确性
        check_correctness(torch.dot, torch.bfloat16, a, b)
        check_correctness(torch.dot, torch.half, a, b)

    @dtypes(torch.float, torch.double)
    @precisionOverride({torch.float32: 1e-4})
    # 定义一个测试方法，用于测试尺寸为 (8, 1, 64) 的张量及其分步视图
    def test_1_sized_with_0_strided(self, device, dtype):
        # 创建一个大小为 (8, 1, 64) 的张量 a，指定设备和数据类型
        a = make_tensor((8, 1, 64), dtype=dtype, device=device)
        # 使用 torch.as_strided 函数创建张量 a 的分步视图，指定大小为 [8, 1, 64] 和步幅为 [64, 0, 1]
        a_strided = torch.as_strided(a, size=[8, 1, 64], stride=[64, 0, 1])
        # 创建一个大小为 (8, 64, 512) 的张量 b，指定设备和数据类型
        b = make_tensor((8, 64, 512), dtype=dtype, device=device)
        # 使用 torch.as_strided 函数创建张量 b 的分步视图，指定大小为 [8, 64, 512] 和步幅为 [64, 1, 512]
        b_strided = torch.as_strided(b, size=[8, 64, 512], stride=[64, 1, 512])
        # 使用 torch.bmm 函数对张量 a_strided 和 b_strided 进行批矩阵乘法
        res = torch.bmm(a_strided, b_strided)
        # 计算预期结果，将 a_strided 和 b_strided 转换为 CPU 上的 numpy 数组进行矩阵乘法，然后将结果转回指定设备和数据类型
        expect = torch.from_numpy(a_strided.cpu().numpy() @ b_strided.cpu().numpy()).to(device=device, dtype=dtype)
        # 使用 self.assertEqual 断言检查计算结果是否与预期结果相等
        self.assertEqual(expect, res)
# 在全局范围内实例化设备类型的测试，并将其绑定到 TestLinalg 类上
instantiate_device_type_tests(TestLinalg, globals())

# 如果当前脚本被直接执行（而非被导入到其他模块中）
if __name__ == '__main__':
    # 启用 TestCase 类的默认数据类型检查
    TestCase._default_dtype_check_enabled = True
    # 运行所有测试
    run_tests()
```