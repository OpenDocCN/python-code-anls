# `.\pytorch\test\test_sparse_csr.py`

```
# Owner(s): ["module: sparse"]

# 导入必要的库和模块
import torch  # 导入PyTorch库
import random  # 导入随机数模块
import itertools  # 导入迭代工具模块
import unittest  # 导入单元测试模块
import functools  # 导入函数工具模块
from torch.testing import make_tensor  # 导入用于创建测试张量的函数
from torch.testing._internal.common_cuda import SM53OrLater, SM80OrLater, TEST_CUSPARSE_GENERIC  # 导入CUDA相关测试和版本检查
from torch.testing._internal.common_utils import \
    (TEST_WITH_TORCHINDUCTOR, TEST_WITH_ROCM, TEST_SCIPY, TEST_NUMPY, TEST_MKL, IS_WINDOWS,  # 导入测试条件和实用函数
     TestCase, run_tests, load_tests, coalescedonoff, parametrize, subtest, skipIfTorchDynamo,
     skipIfRocm, IS_FBCODE, IS_REMOTE_GPU)
from torch.testing._internal.common_device_type import \
    (ops, instantiate_device_type_tests, dtypes, OpDTypes, dtypesIfCUDA, onlyCPU, onlyCUDA,  # 导入设备类型和相关测试函数
     skipCUDAIfNoSparseGeneric, precisionOverride, skipMeta, skipCUDAIf, skipCPUIfNoMklSparse,
     skipCUDAIfRocmVersionLessThan, largeTensorTest)
from torch.testing._internal.common_methods_invocations import \
    (op_db, sparse_csr_unary_ufuncs, ReductionOpInfo)  # 导入操作数据库和稀疏CSR矩阵的一元通用函数
from torch.testing._internal.common_cuda import _get_torch_cuda_version, TEST_CUDA  # 导入CUDA版本检查函数
from torch.testing._internal.common_dtype import (
    floating_types, all_types_and_complex_and, floating_and_complex_types, floating_types_and,  # 导入数据类型相关函数
    all_types_and_complex, floating_and_complex_types_and)
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse  # 导入稀疏操作的输入验证函数
from test_sparse import CUSPARSE_SPMM_COMPLEX128_SUPPORTED, HIPSPARSE_SPMM_COMPLEX128_SUPPORTED  # 导入稀疏矩阵乘法支持的测试标志
import operator  # 导入运算符模块

if TEST_SCIPY:
    import scipy.sparse as sp  # 如果需要测试SciPy，则导入SciPy稀疏矩阵模块

if TEST_NUMPY:
    import numpy as np  # 如果需要测试NumPy，则导入NumPy模块
# load_tests from torch.testing._internal.common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # 加载用于自动过滤测试的函数，以便在沙堡上进行分片测试

no_mkl_sparse = IS_WINDOWS or not TEST_MKL  # 根据操作系统和MKL测试标志设置no_mkl_sparse变量

def _check_cusparse_triangular_solve_available():
    version = _get_torch_cuda_version()
    # 检查CUDA版本是否支持cusparse的三角求解，需要CUDA版本不低于11.4
    min_supported_version = (11, 4)
    return version >= min_supported_version

def _check_cusparse_spgemm_available():
    # 检查CUDA版本是否支持cusparse的稀疏矩阵乘法，ROCm环境不支持
    return not TEST_WITH_ROCM

def _check_cusparse_sddmm_available():
    if TEST_WITH_ROCM:
        return True
    version = _get_torch_cuda_version()
    # 检查CUDA版本是否支持cusparse的稀疏矩阵对乘，需要CUDA版本不低于11.3
    min_supported_version = (11, 3)
    return version >= min_supported_version

# 过滤操作数据库，获取支持稀疏CSR格式的操作列表
_sparse_csr_ops = list(filter(lambda op: op.supports_sparse_csr, op_db))
# 过滤操作数据库，获取支持稀疏压缩格式的操作列表（CSR、CSC、BSR、BSC）
_sparse_compressed_ops = list(filter(lambda op: (op.supports_sparse_csr or op.supports_sparse_csc
                                                 or op.supports_sparse_bsr or op.supports_sparse_bsc), op_db))
# 稠密输出的二元函数列表
binary_functions_with_dense_output = ['mm', 'mv', ]
# 过滤操作数据库，获取输出结果为稠密张量的二元操作列表
binary_ops_with_dense_output = list(filter(lambda op: op.name in binary_functions_with_dense_output, op_db))

# CSR稀疏矩阵的逐元素一元操作允许自动求导的列表
UNARY_EWISE_CSR_ALLOW_AUTOGRAD = [
    'abs',  # 绝对值
    'conj_physical',  # 物理共轭
    'deg2rad',  # 角度转弧度
    'neg',  # 取负
    'positive',  # 正数
    'frac',  # 小数部分
    'nn.functional.relu',  # ReLU函数
    'log1p',  # log(1 + x)
    'rad2deg'


注释：


# 字符串字面量 'rad2deg'，可能用作变量名或者其他用途
# 这段代码定义了一个测试函数 _test_addmm_addmv，用于统一测试 torch.addmv 和 torch.addmm 函数的计算结果。
# 参数 test_case 是 TestCase 的一个实例，用于断言测试结果是否符合预期。
# 参数 f 是 torch.addmv 或 torch.addmm 函数的一个引用。
# 参数 t, m, v 是传递给函数 f 的张量参数。
# 参数 alpha 和 beta 是标量参数，控制线性组合的权重。
# 参数 transpose_out 控制输出张量是否以列优先顺序。
# 参数 layout 控制输入张量 m 的布局，可以是 torch.strided, torch.sparse_csr 或 torch.sparse_csc。
# 参数 mode 控制测试模式，可以是 "all_sparse", "dense_result" 或 None。
def _test_addmm_addmv(
    test_case,
    f,
    t,
    m,
    v,
    *,
    alpha=None,
    beta=None,
    transpose_out=False,
    layout=torch.strided,
    mode=None
):
    """
    Unified test for checking `f(t, m, v, alpha=alpha, beta=beta)` computation,
    where f is `torch.addmv` or `torch.addmm`.
    `transpose_out` controls whether the out argument is in column-major order.
    `layout` controls whether `m` is converted to specified layout or not.
    Custom behaviour is implemented only for torch.sparse_csr layout.
    """
    # 确定数据类型和对应的 numpy 数据类型以便后续计算
    dtype = t.dtype
    numpy_dtype = dtype
    if dtype in {torch.bfloat16}:
        numpy_dtype = torch.float
    # 如果数据类型是复数，设定默认的 alpha 和 beta 值
    if dtype.is_complex:
        alpha = 0.9 + 0.3j if alpha is None else alpha
        beta = 0.5 + 0.6j if beta is None else beta
    else:
        alpha = 1.2 if alpha is None else alpha
        beta = 0.8 if beta is None else beta

    # 转换输入矩阵 m 的布局，如果 layout 是 torch.sparse_csr，则转换为稀疏 CSR 格式，否则保持不变
    def convert_layout(mat):
        if layout == torch.sparse_csr:
            return mat.to_sparse_csr()
        elif layout == torch.sparse_csc:
            return mat.to_sparse_csc()
        else:
            assert mat.layout == layout
            return mat

    # 根据 mode 的不同选择不同的调用方式并获取计算结果 res1
    if mode == "all_sparse":
        res1 = f(*map(convert_layout, (t, m, v)), alpha=alpha, beta=beta)
        test_case.assertEqual(res1.layout, layout)
        res1 = res1.to_dense()
    elif mode == "dense_result":
        res1 = f(t, convert_layout(m), convert_layout(v), alpha=alpha, beta=beta)
    else:
        res1 = f(t, convert_layout(m), v, alpha=alpha, beta=beta)

    # 创建一个与 res1 相同形状的全为 NaN 的张量 res2，并根据 transpose_out 的值调整其内存布局
    res2 = torch.full_like(res1, float('nan'))
    if transpose_out:
        res2 = res2.t().clone(memory_format=torch.contiguous_format).t()

    # 使用函数 f 计算结果填充 res2
    f(t, convert_layout(m), v, alpha=alpha, beta=beta, out=res2)

    # 计算预期的数值结果 res3，并将其转换为对应的 PyTorch 张量
    res3 = alpha * (m.to(numpy_dtype).cpu().numpy() @ v.to(numpy_dtype).cpu().numpy())
    if beta != 0:
        res3 += (beta * t).to(numpy_dtype).cpu().numpy()
    res3 = torch.from_numpy(res3).to(dtype)

    # 断言 res1, res2 和 res3 相等，以验证计算结果的正确性
    test_case.assertEqual(res1, res2)
    test_case.assertEqual(res1, res3)


# 这是一个 TestCase 类的子类，用于测试稀疏 CSR 格式的采样器。
class TestSparseCSRSampler(TestCase):
    # 定义测试方法，用于测试 crow_indices 算法的正确性
    # 在 CPU 上和使用 int32 数据类型进行测试足够了
    def test_make_crow_indices(self):
        # 设定使用 CPU 设备
        device = torch.device('cpu')
        # 索引数据类型设定为 int32
        index_dtype = torch.int32
        # 循环遍历不同的行数
        for n_rows in range(1, 10):
            # 循环遍历不同的列数
            for n_cols in range(1, 10):
                # 循环遍历不同的非零元素个数
                for nnz in range(0, n_rows * n_cols + 1):
                    # 调用 _make_crow_indices 方法生成 crow_indices
                    crow_indices = self._make_crow_indices(
                        n_rows, n_cols, nnz,
                        device=device, dtype=index_dtype)
                    # 断言 crow_indices 的长度应为 n_rows + 1
                    self.assertEqual(len(crow_indices), n_rows + 1)
                    # 计算每行非零元素的数量
                    counts = crow_indices[1:] - crow_indices[:-1]
                    # 断言每行非零元素的总数应为 nnz
                    self.assertEqual(counts.sum(), nnz)
                    # 断言每行非零元素的数量不小于 0
                    self.assertGreaterEqual(counts.min(), 0)
                    # 断言每行非零元素的数量不大于 n_cols
                    self.assertLessEqual(counts.max(), n_cols)
def all_sparse_compressed_layouts(test_name='layout'):
    # 调用 parametrize 函数，为测试用例 'layout' 参数化不同的稀疏压缩布局
    return parametrize(test_name, [
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC'),
        subtest(torch.sparse_bsr, name='SparseBSR'),
        subtest(torch.sparse_bsc, name='SparseBSC')])


def sparse_compressed_nonblock_layouts(test_name='layout'):
    # 调用 parametrize 函数，为测试用例 'layout' 参数化不包含 BSR 和 BSC 的稀疏压缩布局
    return parametrize(test_name, [
        subtest(torch.sparse_csr, name='SparseCSR'),
        subtest(torch.sparse_csc, name='SparseCSC')])


sparse_compressed_indices_methods = {
    # 定义稀疏压缩张量布局对应的索引方法
    torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
    torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
    torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
    torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
}


def batched_nonbatched(test_name='batched'):
    # 调用 parametrize 函数，为测试用例 'batched' 参数化批处理和非批处理两种情况
    return parametrize(test_name, [
        subtest(True, name="Batched"),
        subtest(False, name="NonBatched")
    ])


def hybrid_nonhybrid(test_name='hybrid'):
    # 调用 parametrize 函数，为测试用例 'hybrid' 参数化混合和非混合两种情况
    return parametrize(test_name, [
        subtest(True, name="Hybrid"),
        subtest(False, name="NonHybrid")
    ])


class TestSparseCompressed(TestCase):
    """Testing sparse compressed (CSR, CSC, BSR, BSC) tensor generic features.
    """

    def genTensor(self, size, nnz, *, layout, device=None, dtype=torch.float, index_dtype=torch.int64):
        # 生成稀疏压缩张量，根据给定的大小、非零元素数、布局、设备类型和数据类型
        if device is None:
            device = self.device_type
        return self.genSparseCompressedTensor(size, nnz, device=device, dtype=dtype, index_dtype=index_dtype, layout=layout)

    @all_sparse_compressed_layouts()
    @onlyCPU
    def test_layout(self, layout):
        # 断言 layout 是 'torch.sparse_csr', 'torch.sparse_csc', 'torch.sparse_bsr', 'torch.sparse_bsc' 中的一种
        self.assertIn(str(layout), {'torch.sparse_csr', 'torch.sparse_csc', 'torch.sparse_bsr', 'torch.sparse_bsc'})
        # 断言 layout 的类型为 torch.layout
        self.assertEqual(type(layout), torch.layout)

    @parametrize('shape_and_device_inference', [subtest(False, name='_'), subtest(True, name='shape_and_device_inference')])
    @parametrize('use_factory_function', [subtest(False, name='_'), subtest(True, name='factory')])
    @parametrize('input_kind', [subtest('tensor', name='from_tensor'), subtest('list', name='from_list')])
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @skipMeta
    @sparse_compressed_nonblock_layouts()
    @dtypes(*all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half))
    # 定义一个测试方法，用于测试稀疏张量的空初始化行为
    def test_empty(self, layout, device, dtype):
        # 定义不同的维度列表
        ns = [5, 2, 0]
        # 定义不同的批次形状列表
        batch_shapes = [(), (2,), (2, 3)]
        # 根据布局选择压缩维度
        compressed_dim = {
            torch.sparse_csr: -2,
            torch.sparse_csc: -1,
        }[layout]
        # 获取稀疏张量的压缩索引方法和普通索引方法
        compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[layout]
        # 遍历所有可能的维度组合
        for m, n, b in itertools.product(ns, ns, batch_shapes):
            # 根据当前维度组合创建张量形状
            shape = (*b, m, n)
            # 在禁用稀疏张量不变性检查的情况下，使用torch.empty创建张量
            with torch.sparse.check_sparse_tensor_invariants(enable=False):
                # torch.empty 可能返回无效的稀疏压缩张量
                result = torch.empty(shape, dtype=dtype, device=device, layout=layout)
            # 断言张量的形状、数据类型、设备和布局是否符合预期
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, torch.device(device))
            self.assertEqual(result.layout, layout)
            # 断言压缩索引方法和普通索引方法的返回形状是否符合预期
            self.assertEqual(compressed_indices_mth(result).shape, (*b, shape[compressed_dim] + 1,))
            self.assertEqual(plain_indices_mth(result).shape, (*b, 0,))
            # 断言张量值的形状是否为空
            self.assertEqual(result.values().shape, (*b, 0,))
            # 断言张量的非零元素数量是否为0
            self.assertEqual(result._nnz(), 0)
            # 断言压缩索引方法和普通索引方法的设备是否与张量一致
            self.assertEqual(compressed_indices_mth(result).device, torch.device(device))
            self.assertEqual(plain_indices_mth(result).device, torch.device(device))
            # 断言张量值的设备是否与张量一致
            self.assertEqual(result.values().device, torch.device(device))
            # 断言压缩索引方法和普通索引方法的数据类型是否为torch.int64
            self.assertEqual(compressed_indices_mth(result).dtype, torch.int64)
            self.assertEqual(plain_indices_mth(result).dtype, torch.int64)
            # 断言张量值的数据类型是否与张量一致
            self.assertEqual(result.values().dtype, dtype)

    # 装饰器声明跳过元信息测试
    @skipMeta
    # 装饰器声明应用所有稀疏压缩非块布局
    @sparse_compressed_nonblock_layouts()
    # 装饰器声明适用于所有数据类型和复杂数据类型，并且包括torch.bool, torch.half, torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    # 定义一个测试方法，用于测试空初始化引发错误的情况
    def test_empty_errors(self, layout, device, dtype):
        # 断言引发RuntimeError，指出torch.empty仅支持批量稀疏压缩（非块）张量，但得到了指定大小的张量
        with self.assertRaisesRegex(RuntimeError,
                                    "torch.empty: Only batched sparse compressed \\(non-block\\) tensors are supported"
                                    ", but got size"):
            # 调用torch.empty创建指定形状的张量，预期引发错误
            torch.empty((5,), dtype=dtype, device=device, layout=layout)

    # 装饰器声明跳过元信息测试
    @skipMeta
    # 装饰器声明适用于所有稀疏压缩布局
    @all_sparse_compressed_layouts()
    # 装饰器声明适用于所有数据类型和复杂数据类型，并且包括torch.bool, torch.bfloat16, torch.half
    @dtypes(*all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half))
    # 定义测试方法，用于测试稀疏压缩张量的各种维度情况
    def test_sparse_compressed_tensor_with_dims(self, layout, device, dtype):

        # 定义函数，获取稀疏压缩张量的属性信息
        def get_sparse_compressed_tensor_properties(s):
            # 根据布局类型选择压缩后的索引和普通索引
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices, plain_indices = s.crow_indices(), s.col_indices()
            else:
                compressed_indices, plain_indices = s.ccol_indices(), s.row_indices()
            # 获取值
            values = s.values()
            # 返回包含张量属性的字典
            return dict(shape=s.shape, dtype=s.dtype, device=s.device, nnz=s._nnz(), layout=s.layout,
                        compressed_indices_shape=compressed_indices.shape,
                        compressed_indices_dtype=compressed_indices.dtype,
                        compressed_indices_device=compressed_indices.device,
                        plain_indices_shape=plain_indices.shape,
                        plain_indices_dtype=plain_indices.dtype,
                        plain_indices_device=plain_indices.device,
                        values_shape=values.shape,
                        values_dtype=values.dtype,
                        values_device=values.device)

        # 遍历索引数据类型，生成简单输入数据
        for index_dtype in [torch.int32, torch.int64]:
            for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
                # 获取稠密维度、稀疏维度和批处理维度
                dense_dim = t.dense_dim()
                sparse_dim = t.sparse_dim()
                batch_dim = t.ndim - sparse_dim - dense_dim
                nnz = t.values().shape[batch_dim]
                # 根据布局类型，获取块大小
                if layout in {torch.sparse_bsr, torch.sparse_bsc}:
                    blocksize = t.values().shape[batch_dim + 1: batch_dim + 1 + sparse_dim]
                else:
                    blocksize = ()

                # 调用底层函数创建稀疏压缩张量
                e = torch.ops.aten._sparse_compressed_tensor_with_dims(nnz, dense_dim, t.shape, blocksize, index_dtype,
                                                                       dtype=dtype, layout=layout, device=device)

                # 获取创建的稀疏压缩张量和原始张量的属性信息
                e_prop, t_prop = get_sparse_compressed_tensor_properties(e), get_sparse_compressed_tensor_properties(t)
                # 断言两个张量的属性相等
                for k, v in e_prop.items():
                    self.assertEqual(v, t_prop[k], lambda msg: f'{msg} when comparing {k}, expected {t_prop[k]}, got {v}')

    # 装饰器，标记测试方法为跳过元信息的
    @skipMeta
    # 装饰器，用于获取所有稀疏压缩布局类型
    @all_sparse_compressed_layouts()
    # 装饰器，用于获取所有数据类型和复杂类型，包括布尔型、半精度浮点型、bfloat16类型
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16))
    # 定义测试克隆方法
    def test_clone(self, layout, device, dtype):
        # 遍历生成的简单输入数据
        for sparse in self.generate_simple_inputs(
                layout, device=device, dtype=dtype, index_dtype=torch.int32):
            # 克隆稀疏张量
            cloned_sparse = sparse.clone()
            # 断言克隆的张量和原始张量相等
            self.assertEqual(sparse, cloned_sparse)

    # 装饰器，用于获取所有稀疏压缩布局类型
    @all_sparse_compressed_layouts()
    # 装饰器，标记测试方法为跳过元信息的
    @skipMeta
    # 装饰器，用于获取所有稀疏压缩布局类型
    @all_sparse_compressed_layouts()
    # 装饰器，用于获取所有数据类型和复杂类型，包括半精度浮点型、布尔型、bfloat16类型
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义测试方法，用于测试稀疏压缩张量的复制功能
    def test_copy(self, layout, device, dtype):

        # 定义测试函数，运行单个复制测试
        def run_test(shape, blocksize, nnz, index_type):
            # 生成稀疏压缩张量 a 和 b，用于复制操作
            a = self.genSparseCompressedTensor(shape, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_type, blocksize=blocksize)
            b = self.genSparseCompressedTensor(shape, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_type, blocksize=blocksize)

            # 执行复制操作
            a.copy_(b)

            # 断言复制后的张量 a 与 b 相等
            self.assertEqual(a, b)

        # 定义不同的维度数和块大小的组合
        ns = [(9, 3), (2, 1), (0, 0)]  # (维度数, 对应的块大小)
        batch_shapes = [(), (2,), (2, 3)]
        # 遍历维度和块大小的组合，同时迭代批处理形状和索引类型
        for ((m, bm), (n, bn), b), index_dtype in zip(itertools.product(ns, ns, batch_shapes), [torch.int32, torch.int64]):
            # 根据布局类型选择合适的块大小
            blocksize = (bm, bn) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
            # 运行两种不同的测试：无元素和满元素的情况
            run_test((*b, m, n), blocksize, 0, index_dtype)
            run_test((*b, m, n), blocksize, m * n, index_dtype)

    # 跳过元数据的装饰器
    @skipMeta
    # 应用所有稀疏压缩布局的装饰器
    @all_sparse_compressed_layouts()
    # 应用包含所有类型以及特定类型的装饰器：torch.half, torch.bool, torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 测试复制操作中可能引发的错误情况
    def test_copy_errors(self, layout, device, dtype):
        # 根据布局选择块大小，稀疏矩阵的布局包括稀疏 BSR 和 BSC
        blocksize = (2, 3) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        # 根据布局选择非零元素的数量
        nnz = 6 if layout in {torch.sparse_bsr, torch.sparse_bsc} else 1
        # 根据布局选择形状
        shape1 = (2 * 6, 3 * 6) if layout in {torch.sparse_bsr, torch.sparse_bsc} else (2, 3)
        
        # 遍历不同的索引数据类型
        for index_dtype in [torch.int32, torch.int64]:
            # 使用给定参数生成稀疏压缩张量 a
            a = self.genSparseCompressedTensor(shape1, 0, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)

            # 断言复制操作会引发 RuntimeError，错误消息指出不支持不同布局的稀疏压缩张量复制
            with self.assertRaisesRegex(RuntimeError,
                                        "copy of sparse compressed tensors having different layouts is not supported."):
                a.copy_(torch.empty(a.shape, dtype=dtype, device=device))

            # 使用给定参数生成稀疏压缩张量 b
            b = self.genSparseCompressedTensor(shape1, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            # 断言复制操作会引发 RuntimeError，错误消息指出只支持相同指定元素数量的稀疏压缩张量复制
            assert a._nnz() != b._nnz(), (a._nnz(), b._nnz())
            with self.assertRaisesRegex(RuntimeError,
                                        "only sparse compressed tensors with the same number of specified elements are supported."):
                a.copy_(b)

            # 创建形状与 b 相反的稀疏压缩张量 c
            shape2 = tuple(reversed(shape1))
            c = self.genSparseCompressedTensor(shape2, nnz, dtype=dtype, layout=layout, device=device,
                                               index_dtype=index_dtype, blocksize=blocksize)
            # 断言复制操作会引发 RuntimeError，错误消息指出期望 self 和 src 在维度上匹配
            with self.assertRaisesRegex(
                    RuntimeError,
                    "expected shapes of self and src to match along dimension"):
                b.copy_(c)

            # 如果存在块大小，则创建块大小与 b 不同的稀疏压缩张量 d
            if blocksize:
                blocksize1 = tuple(reversed(blocksize))
                d = self.genSparseCompressedTensor(shape1, nnz, dtype=dtype, layout=layout, device=device,
                                                   index_dtype=index_dtype, blocksize=blocksize1)
                # 断言复制操作会引发 RuntimeError，错误消息指出不支持具有不同块大小的稀疏压缩张量复制
                with self.assertRaisesRegex(RuntimeError,
                                            "copy of sparse compressed tensors having different block sizes is not supported"):
                    b.copy_(d)
    # 定义测试方法，用于测试空输入的情况
    def test_empty_like(self, layout, layout2, device, dtype):
        # 生成简单稀疏输入的迭代器，并遍历每个稀疏张量
        for sparse in self.generate_simple_inputs(layout):
            # 如果布局相同，创建一个与稀疏张量形状相同但布局为 layout2 的空张量
            if layout == layout2:
                result = torch.empty_like(sparse, layout=layout2)
                # 获取稀疏压缩索引方法和普通索引方法
                compressed_indices_mth, plain_indices_mth = sparse_compressed_indices_methods[result.layout]
                # 验证稀疏张量的压缩数据、普通索引、值以及形状
                torch._validate_sparse_compressed_tensor_args(compressed_indices_mth(result),
                                                              plain_indices_mth(result),
                                                              result.values(),
                                                              result.shape,
                                                              result.layout)
                # 断言稀疏张量的形状与原稀疏张量相同
                self.assertEqual(sparse.shape, result.shape)
            else:
                # 如果布局不同，断言调用 empty_like 抛出 RuntimeError 异常，异常信息为不支持不同布局的情况
                self.assertRaisesRegex(
                    RuntimeError,
                    "empty_like with different sparse layout is not supported",
                    lambda: torch.empty_like(sparse, layout=layout2)
                )

    # 装饰器声明，跳过元信息测试，应用于所有的稀疏压缩布局
    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义验证方法，用于测试稀疏张量的验证
    def test_validate(self, layout, device, dtype):
        # 定义创建零批次张量的内部函数
        def make_zero_batched(t):
            return torch.empty(*((0,) + t.shape), dtype=t.dtype, device=t.device)

        # 遍历稀疏输入生成器返回的简单输入组合及其参数
        for index_dtype in [torch.int32, torch.int64]:
            for (compressed_indices, plain_indices, values), kwargs in self.generate_simple_inputs(
                    layout, device=device, dtype=dtype, index_dtype=index_dtype, output_tensor=False):
                size = kwargs['size']
                # 验证稀疏压缩张量的压缩索引、普通索引、值以及形状
                torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, values, size, layout)

                # 检查空批次的情况
                torch._validate_sparse_compressed_tensor_args(
                    *(make_zero_batched(t) for t in (compressed_indices, plain_indices, values)),
                    (0,) + size,
                    layout
                )

            # 创建指定索引类型的零压缩索引和空普通索引的张量
            compressed_indices = torch.tensor([0, 0], dtype=index_dtype)
            plain_indices = torch.tensor([], dtype=index_dtype)
            # 验证压缩稀疏索引是否有效，根据布局确定是否支持 CSR 或 BSR 布局
            torch._validate_compressed_sparse_indices(layout in {torch.sparse_csr, torch.sparse_bsr},
                                                      compressed_indices, plain_indices, 1, 1, 0)

    # 装饰器声明，跳过元信息测试，应用于所有的稀疏压缩布局
    @skipMeta
    @all_sparse_compressed_layouts()
    @parametrize('target', [subtest('validate_sparse_compressed_tensor_args'),
                            subtest('sparse_compressed_tensor'),
                            subtest('sparse_compressed_tensor_no_size')])
    # 参数化测试，用于测试稀疏压缩张量的不同目标
    # 定义测试函数，用于测试不合法的输入情况
    def test_invalid_input(self, layout, device, target):
        # 使用 _generate_invalid_input 方法生成无效输入的测试数据，遍历每个测试数据
        for label, compressed_indices, plain_indices, values, size, errmsg in self._generate_invalid_input(layout, device):
            # 根据布局类型替换错误消息中的字符串，针对稀疏张量的不同布局方式进行特定替换
            if layout is torch.sparse_bsr:
                errmsg = errmsg.replace('compressed_indices_name', 'row block').replace('plain_indices_name', 'column block')
            elif layout is torch.sparse_bsc:
                errmsg = errmsg.replace('compressed_indices_name', 'column block').replace('plain_indices_name', 'row block')
            elif layout is torch.sparse_csr:
                errmsg = errmsg.replace('compressed_indices_name', 'row').replace('plain_indices_name', 'column')
            elif layout is torch.sparse_csc:
                errmsg = errmsg.replace('compressed_indices_name', 'column').replace('plain_indices_name', 'row')
            # 根据布局类型进一步替换错误消息中的字符串，针对不同的稀疏布局类型进行特定替换
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                errmsg = errmsg.replace('compressed_indices', 'crow_indices') \
                               .replace('plain_indices', 'col_indices') \
                               .replace('plain_dim', 'ncols') \
                               .replace('compressed_dim', 'nrows')
            else:
                errmsg = errmsg.replace('compressed_indices', 'ccol_indices') \
                               .replace('plain_indices', 'row_indices') \
                               .replace('plain_dim', 'nrows') \
                               .replace('compressed_dim', 'ncols')

            # 如果目标是 'sparse_compressed_tensor_no_size' 并且标签在特定集合内，则跳过该输入的测试
            if target == 'sparse_compressed_tensor_no_size' and label in {
                    'invalid size', 'invalid batchsize', 'invalid compressed_indices shape', 'invalid max(plain_indices)',
                    'invalid blocksize'}:
                continue

            # 使用 assertRaisesRegex 检查运行时异常是否符合预期的错误消息
            with self.assertRaisesRegex(RuntimeError, errmsg):
                # 根据目标类型调用不同的 Torch 函数，以验证或创建稀疏压缩张量，并检查是否抛出预期的异常
                if target == 'validate_sparse_compressed_tensor_args':
                    torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, values, size, layout)
                elif target == 'sparse_compressed_tensor':
                    torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, size, layout=layout)
                elif target == 'sparse_compressed_tensor_no_size':
                    torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, layout=layout)
                else:
                    raise NotImplementedError(target)

    # 装饰器，用于跳过元数据测试，仅在 CPU 环境下测试大型张量
    @skipMeta
    @onlyCPU
    @largeTensorTest("30GB", "cpu")
    # 测试大型无效输入的情况，验证 CSR 格式下的整数溢出问题

    # 设置行数为 2 的 31 次方，测试在行维度上的 32 位整数溢出情况
    rows = 2 ** 31
    with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in row dimension'):
        # 创建一个稀疏 CSR 张量，其中行索引使用 torch.int32 类型，
        # 当行数超过 2 的 31 次方时，预期引发运行时错误
        torch.sparse_csr_tensor(torch.arange(rows + 1, dtype=torch.int32) // rows,
                                torch.tensor([0], dtype=torch.int32),
                                torch.tensor([1]), (rows, 1))
    # 对于行数超过 2 的 31 次方的情况，使用 torch.int64 类型创建稀疏 CSR 张量
    torch.sparse_csr_tensor(torch.arange(rows + 1, dtype=torch.int64) // rows,
                            torch.tensor([0], dtype=torch.int64),
                            torch.tensor([1]), (rows, 1))

    # 设置列数为 2 的 31 次方，测试在列维度上的 32 位整数溢出情况
    cols = 2 ** 31
    with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in column dimension'):
        # 创建一个稀疏 CSR 张量，其中列索引使用 torch.int32 类型，
        # 当列数超过 2 的 31 次方时，预期引发运行时错误
        torch.sparse_csr_tensor(torch.arange(2, dtype=torch.int32),
                                torch.tensor([0], dtype=torch.int32),
                                torch.tensor([1]), (1, cols))
    # 对于列数超过 2 的 31 次方的情况，使用 torch.int64 类型创建稀疏 CSR 张量
    torch.sparse_csr_tensor(torch.arange(2, dtype=torch.int64),
                            torch.tensor([0], dtype=torch.int64),
                            torch.tensor([1]), (1, cols))

    # 设置非零元素数为 2 的 31 次方，测试在非零元素数上的 32 位整数溢出情况
    nnz = 2 ** 31
    with self.assertRaisesRegex(RuntimeError, '32-bit integer overflow in nnz'):
        # 创建一个稀疏 CSR 张量，其中行索引使用 torch.int32 类型，
        # 当非零元素数超过 2 的 31 次方时，预期引发运行时错误
        # 在构建 crow_indices 之前，确保使用 nnz - 1 来避免整数溢出
        torch.sparse_csr_tensor(torch.tensor([0, nnz // 2, nnz - 1], dtype=torch.int32),
                                torch.arange(nnz // 2, dtype=torch.int32).repeat(2),
                                torch.ones(nnz, dtype=torch.int8), (2, nnz // 2))
    # 对于非零元素数超过 2 的 31 次方的情况，使用 torch.int64 类型创建稀疏 CSR 张量
    torch.sparse_csr_tensor(torch.tensor([0, nnz // 2, nnz], dtype=torch.int64),
                            torch.arange(nnz // 2, dtype=torch.int64).repeat(2),
                            torch.ones(nnz, dtype=torch.int8), (2, nnz // 2))
    # 定义测试方法，测试稀疏张量转换到指定数据类型的功能
    def test_to_dtype(self, layout, device, dtype):
        # to_dense 方法不支持混合输入的情况
        # 使用 generate_simple_inputs 方法生成不支持混合输入的稀疏张量
        for sparse in self.generate_simple_inputs(layout, dtype=dtype, device=device, enable_hybrid=False):
            # 遍历所有数据类型以及 torch.bool、torch.half、torch.bfloat16
            for to_dtype in all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16):
                # 将稀疏张量 sparse 转换为目标数据类型 to_dtype
                sparse_to_dtype = sparse.to(to_dtype)
                # 将稀疏张量 sparse 转换为密集张量后再转换为目标数据类型 to_dtype
                dense_to_dtype = sparse.to_dense().to(to_dtype)
                # 断言稀疏张量转换为密集张量后再转换为目标数据类型的结果应与直接转换的结果相等
                self.assertEqual(sparse_to_dtype.to_dense(), dense_to_dtype)

    # 标记为跳过 Meta 的测试，并针对所有稀疏压缩布局执行测试
    @skipMeta
    @all_sparse_compressed_layouts()
    @dtypes(torch.double)
    # 测试将稀疏张量序列化为 pickle 格式并反序列化的功能
    def test_pickle(self, layout, dtype, device):
        import pickle

        # 使用 generate_simple_inputs 方法生成指定布局和数据类型的稀疏张量
        for sparse in self.generate_simple_inputs(layout, device=device, dtype=dtype):
            # 序列化稀疏张量 sparse
            serialized = pickle.dumps(sparse)
            # 反序列化 pickle 数据，得到稀疏张量 sparse_loaded
            sparse_loaded = pickle.loads(serialized)

            # 断言反序列化后的稀疏张量 sparse_loaded 应与原稀疏张量 sparse 相等
            self.assertEqual(sparse, sparse_loaded)

    # 对所有稀疏压缩布局执行测试，并参数化 index_dtype 为 torch.int32 和 torch.int64
    @all_sparse_compressed_layouts()
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    # 定义一个测试方法，用于测试 select_copy 方法在不同条件下的行为
    def test_select_copy(self, device, dtype, index_dtype, layout):

        # 定义一个内部函数，用于判断一个张量是否是另一个张量的视图
        def is_view_of(base, other):
            # 这段逻辑基本复制自 TestViewOps.is_view_of 方法
            # 检查 other 是否不是视图，或者 other 是否与 base 相同，或者 other 的基张量不是 base，或者设备类型不同
            if (
                not other._is_view() or
                other is base or
                other._base is not base or
                base.device != other.device
            ):
                return False
            # 如果设备类型是 'cpu' 或 'cuda'
            if base.device.type in ('cpu', 'cuda'):
                # 检查 base 和 other 的未类型化存储数据指针是否相同
                if base.untyped_storage().data_ptr() != other.untyped_storage().data_ptr():
                    return False
            return True

        # 准备测试参数的字典
        kwargs = dict(device=device, dtype=dtype, index_dtype=index_dtype)
        # 遍历生成的简单输入对，分别是稀疏张量和稠密张量
        for sparse, dense in zip(self.generate_simple_inputs(layout, **kwargs),
                                 self.generate_simple_inputs(torch.strided, **kwargs)):
            # 如果布局是稀疏 CSR 或 BSR
            if layout in {torch.sparse_csr, torch.sparse_bsr}:
                # 确定批次维度数
                n_batchdim = sparse.crow_indices().ndim - 1
            # 如果布局是稀疏 CSC 或 BSC
            elif layout in {torch.sparse_csc, torch.sparse_bsc}:
                # 确定批次维度数
                n_batchdim = sparse.ccol_indices().ndim - 1
            else:
                # 如果布局不在已知的稀疏类型中，引发断言错误，表示不可达的代码分支
                assert 0  # unreachable
            # 断言稀疏张量和稠密张量相等
            self.assertEqual(sparse, dense)
            # 遍历稀疏张量的各维度
            for dim in range(sparse.ndim):
                # 如果稀疏张量的某维度大小为 0
                if sparse.shape[dim] == 0:
                    # 使用断言检查 select_copy 方法对于索引 0 的行为
                    with self.assertRaisesRegex(IndexError, "index 0 out of range for tensor of size"):
                        torch.select_copy(sparse, dim, 0)
                    with self.assertRaisesRegex(IndexError, "index 0 out of range for tensor of size"):
                        torch.select_copy(dense, dim, 0)
                # 如果存在批次维度并且当前维度在批次维度范围内
                elif n_batchdim and dim >= n_batchdim and dim < n_batchdim + 2:
                    # 使用断言检查 select_copy 方法在选择稀疏维度时的行为
                    with self.assertRaisesRegex(
                            RuntimeError,
                            "selecting sparse dimensions is not supported for batched sparse compressed tensors"):
                        torch.select_copy(sparse, dim, 0)
                else:
                    # 对于稀疏和稠密张量的当前维度，遍历选定的索引集合
                    for index in {0, sparse.shape[dim] // 2, sparse.shape[dim] - 1}:
                        # 使用 select_copy 方法从稠密张量复制数据到新的张量 dense_select
                        dense_select = torch.select_copy(dense, dim, index)
                        # 使用 select_copy 方法从稀疏张量复制数据到新的张量 sparse_select
                        sparse_select = torch.select_copy(sparse, dim, index)
                        # 断言稀疏选择和稠密选择张量相等
                        self.assertEqual(sparse_select, dense_select)
                        # 使用 is_view_of 函数检查稀疏选择的值张量是否是稀疏值的视图
                        self.assertFalse(is_view_of(sparse_select.values(), sparse.values()))
    def _npref_block_addmm_addmv(c, a, b, alpha, beta):
        # 计算 alpha * (a @ b) + beta * c，其中 @ 表示矩阵乘法
        return alpha * (a @ b) + beta * c


    class TestSparseCSR(TestCase):

        def test_csr_stride(self):
            # 生成一个稀疏 CSR 张量 a
            a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

            # 断言在稀疏 CSR 张量上调用 stride() 方法会引发 RuntimeError，并且错误消息包含 "Sparse CSR tensors do not have strides"
            with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have strides"):
                a.stride()

            # 断言在稀疏 CSR 张量上调用 stride(-1) 方法会引发 RuntimeError，并且错误消息包含 "Sparse CSR tensors do not have strides"
            with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have strides"):
                a.stride(-1)

        def test_csr_storage(self):
            # 生成一个稀疏 CSR 张量 a
            a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

            # 断言在稀疏 CSR 张量上调用 storage() 方法会引发 RuntimeError，并且错误消息包含 "Cannot access storage of SparseCsrTensorImpl"
            with self.assertRaisesRegex(RuntimeError, "Cannot access storage of SparseCsrTensorImpl"):
                a.storage()

        def test_csr_is_contiguous(self):
            # 生成一个稀疏 CSR 张量 a
            a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)

            # 断言在稀疏 CSR 张量上调用 is_contiguous() 方法会引发 RuntimeError，并且错误消息包含 "Sparse CSR tensors do not have is_contiguous"
            with self.assertRaisesRegex(RuntimeError, "Sparse CSR tensors do not have is_contiguous"):
                a.is_contiguous()

        @onlyCPU
        @largeTensorTest("20GB", "cpu")
        def test_csr_nnz(self):
            # 测试 CSR 张量中指定元素数量的极限情况，参见 gh-102520
            for nnz in [0, 2**31]:
                rows, cols = 1, max(nnz, 1)
                crow_indices = torch.tensor([0, nnz], dtype=torch.int64)
                col_indices = torch.arange(nnz, dtype=torch.int64)
                values = torch.ones(nnz, dtype=torch.int8)
                # 创建一个稀疏 CSR 张量 a
                a = torch.sparse_csr_tensor(crow_indices, col_indices, values, (rows, cols))
                # 断言稀疏 CSR 张量的非零元素数量与 nnz 相等
                self.assertEqual(a._nnz(), nnz)

        def test_csr_double_to_sparse_csr(self):
            # 生成一个稀疏 CSR 张量 a
            a = self.genSparseCSRTensor((3, 3), 3, dtype=torch.float, device=self.device_type, index_dtype=torch.int64)
            # 连续两次调用 to_sparse_csr() 方法，不产生影响
            a.to_sparse_csr().to_sparse_csr()

        @all_sparse_compressed_layouts()
        @parametrize("index_dtype", [torch.int32, torch.int64])
        @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    # 定义测试方法，用于测试稀疏张量的选择操作，根据不同布局选择相应的索引方法和创建方法
    def test_select(self, device, dtype, index_dtype, layout):
        # 定义压缩索引方法字典，根据布局选择相应的稀疏 CSR 或 BSR 张量的压缩索引方法
        compressed_indices_mth = {
            torch.sparse_csr: torch.Tensor.crow_indices,
            torch.sparse_bsr: torch.Tensor.crow_indices,
            torch.sparse_csc: torch.Tensor.ccol_indices,
            torch.sparse_bsc: torch.Tensor.ccol_indices,
        }[layout]

        # 定义普通索引方法字典，根据布局选择相应的稀疏 CSR 或 BSR 张量的普通索引方法
        plain_indices_mth = {
            torch.sparse_csr: torch.Tensor.col_indices,
            torch.sparse_bsr: torch.Tensor.col_indices,
            torch.sparse_csc: torch.Tensor.row_indices,
            torch.sparse_bsc: torch.Tensor.row_indices,
        }[layout]
        
        # 定义创建稀疏张量方法字典，根据布局选择相应的稀疏 CSR 或 BSR 张量的创建方法
        create_tensor_mth = {
            torch.sparse_csr: torch.sparse_csr_tensor,
            torch.sparse_bsr: torch.sparse_bsr_tensor,
            torch.sparse_csc: torch.sparse_csc_tensor,
            torch.sparse_bsc: torch.sparse_bsc_tensor,
        }[layout]

        # 定义张量的形状、非零元素个数及块大小（如果是 BSR 或 BSC 布局）
        shape = (2, 3, 6, 10)
        nnz = 6
        blocksize = (2, 2) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        
        # 生成稀疏压缩张量
        sparse = self.genSparseCompressedTensor(
            shape, nnz, device=device, layout=layout, dtype=dtype, index_dtype=index_dtype, blocksize=blocksize)
        
        # 获取压缩索引、普通索引及值
        comp_indices = compressed_indices_mth(sparse)
        plain_indices = plain_indices_mth(sparse)
        values = sparse.values()

        # 从批处理维度选择稀疏张量
        sparse_selected12 = sparse.select(1, 2)
        
        # 期望的选择后的稀疏张量，使用相应的创建方法创建
        expected_sparse_selected12 = create_tensor_mth(comp_indices.select(1, 2).contiguous(),
                                                       plain_indices.select(1, 2).contiguous(),
                                                       values.select(1, 2).contiguous(),
                                                       size=(2, 6, 10),
                                                       dtype=dtype,
                                                       device=device)
        # 断言选择后的稀疏张量是否与期望相等
        self.assertEqual(expected_sparse_selected12, sparse_selected12)

        # 不允许从批处理维度选择行/列
        sparse_non_batched = sparse[0, 0]
        
        # 从稀疏维度选择操作的断言循环
        for select_args in [(0, 0), (1, 1)]:
            sparse_selected = sparse_non_batched.select(*select_args)
            dense_selected = sparse_non_batched.to_dense().select(*select_args)
            self.assertEqual(dense_selected, sparse_selected)

        # 验证稀疏张量的单个元素是否与密集张量中对应元素相等
        self.assertEqual(sparse[0, 0, 0, 0], sparse.to_dense()[0, 0, 0, 0])
        
        # 禁止通过索引分配给稀疏张量的断言
        with self.assertRaisesRegex(TypeError, "Cannot assign to a sparse tensor"):
            sparse[0, 0, 0, 0] = 99.0

        # 不支持保留批处理维度选择稀疏维度的断言
        msg = "selecting sparse dimensions is not supported for batched sparse compressed tensors."
        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-2, 0)

        with self.assertRaisesRegex(RuntimeError, msg):
            sparse.select(-1, 0)

    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_resize(self, device, dtype):
        # 定义一个函数用于计算张量的元素个数
        def numel(tensor):
            r = 1
            for s in tensor.shape:
                r *= s
            return r

        # 定义不同的批次形状
        batch_shapes = [(), (2,), (2, 3)]
        # 遍历不同的索引数据类型和批次形状组合
        for index_dtype, b in zip([torch.int32, torch.int64], batch_shapes):
            shape = (*b, 2, 3)  # 构造稀疏张量的形状
            nnz = 6  # 非零元素的数量
            # 生成稀疏 CSR 张量
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            # 断言稀疏张量的元素数量与预期一致
            self.assertEqual(a.numel(), numel(a))

            new_shape = (*b, 4, 5)
            # 调整稀疏张量的形状为新的形状
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # 调整到更大的形状不会增加指定的元素
            self.assertEqual(a._nnz(), nnz)
            self.assertEqual(a.numel(), numel(a))

            new_shape = (*b, 1, 5)
            # 再次调整稀疏张量的形状为新的形状
            a.resize_(new_shape)

            self.assertEqual(a.shape, new_shape)
            # 调整到更小的形状会裁剪指定的元素
            self.assertEqual(a._nnz(), 5)
            self.assertEqual(a.numel(), numel(a))

            # 裁剪批次维度
            a.resize_(new_shape[-2], new_shape[-1])
            self.assertEqual(a.shape, (new_shape[-2], new_shape[-1]))
            self.assertEqual(a._nnz(), 5)
            self.assertEqual(a.numel(), numel(a))

    @skipMeta
    @dtypes(torch.float, torch.bool)
    @all_sparse_compressed_layouts()
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_resize_errors(self, device, dtype):
        # 遍历不同的索引数据类型
        for index_dtype in [torch.int32, torch.int64]:
            shape = (2, 3)  # 稀疏张量的形状
            nnz = 6  # 非零元素的数量
            # 生成稀疏 CSR 张量
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)

            # 使用断言捕获预期的运行时错误信息
            with self.assertRaisesRegex(RuntimeError, "torch.resize_: Only batched sparse CSR matrices are supported"):
                new_shape = (4,)
                a.resize_(new_shape)

            # 调整稀疏 CSR 张量列的大小至较小的值未实现
            with self.assertRaisesRegex(
                RuntimeError,
                "torch.resize_: Resizing columns of sparse CSR tensors to a smaller value is not supported.",
            ):
                new_shape = (2, 2)
                a.resize_(new_shape)

    @skipIfTorchDynamo()
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义一个测试方法，用于测试从密集张量转换为稀疏 CSR 格式的功能
    def test_sparse_csr_from_dense(self, device, dtype):
        # 创建一个设备上的指定数据类型的密集张量
        dense = torch.tensor([[4, 5, 0], [0, 0, 0], [1, 0, 0]], dtype=dtype, device=device)
        # 将密集张量转换为稀疏 CSR 格式
        sparse = dense.to_sparse_csr()
        # 断言稀疏张量的行指针数组与预期结果一致
        self.assertEqual(torch.tensor([0, 2, 2, 3], dtype=torch.int64), sparse.crow_indices())
        # 断言稀疏张量的列索引数组与预期结果一致
        self.assertEqual(torch.tensor([0, 1, 0], dtype=torch.int64), sparse.col_indices())
        # 断言稀疏张量的数值数组与预期结果一致
        self.assertEqual(torch.tensor([4, 5, 1], dtype=dtype), sparse.values())

        # 创建另一个设备上的指定数据类型的密集张量
        dense = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=dtype, device=device)
        # 将密集张量转换为稀疏 CSR 格式
        sparse = dense.to_sparse_csr()
        # 断言稀疏张量的行指针数组与预期结果一致
        self.assertEqual(torch.tensor([0, 0, 1, 2], dtype=torch.int64), sparse.crow_indices())
        # 断言稀疏张量的列索引数组与预期结果一致
        self.assertEqual(torch.tensor([2, 0], dtype=torch.int64), sparse.col_indices())
        # 断言稀疏张量的数值数组与预期结果一致
        self.assertEqual(torch.tensor([1, 1], dtype=dtype), sparse.values())

        # 创建另一个设备上的指定数据类型的密集张量
        dense = torch.tensor([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=dtype, device=device)
        # 将密集张量转换为稀疏 CSR 格式
        sparse = dense.to_sparse_csr()
        # 断言稀疏张量的行指针数组与预期结果一致
        self.assertEqual(torch.tensor([0, 3, 6, 9], dtype=torch.int64), sparse.crow_indices())
        # 断言稀疏张量的列索引数组与预期结果一致
        self.assertEqual(torch.tensor([0, 1, 2] * 3, dtype=torch.int64), sparse.col_indices())
        # 断言稀疏张量的数值数组与预期结果一致
        self.assertEqual(torch.tensor([2] * 9, dtype=dtype), sparse.values())
    # 定义测试方法，将稀疏压缩格式转换为稠密格式
    def _test_sparse_compressed_to_dense(self, device, dtype, layout):
        # 根据布局获取压缩格式的字符串表示
        compressed_format_str = str(layout)[-3:]

        # 将稠密张量转换为指定的压缩格式稀疏张量
        def to_compressed(t):
            return getattr(t, f"to_sparse_{compressed_format_str}")()

        # 使用指定的压缩格式构造稀疏张量的构造器
        def compressed_constructor(*input, **kwargs):
            constructor = getattr(torch, f"sparse_{compressed_format_str}_tensor")
            return constructor(*input, **kwargs)

        # 根据布局获取稀疏张量的密集形状
        def get_dense_shape(shape, batch_ndim):
            if layout is torch.sparse_csc:
                compressed_dims_slice = slice(batch_ndim + 1, batch_ndim - 1, -1)
            else:
                compressed_dims_slice = slice(batch_ndim, batch_ndim + 2)
            return shape[:batch_ndim] + shape[compressed_dims_slice] + shape[batch_ndim + 2:]

        # 根据布局对稀疏张量进行转置
        def transpose(t, batch_ndim):
            if layout is torch.sparse_csc:
                return t.transpose(batch_ndim, batch_ndim + 1)
            return t

        # 定义测试用例的维度范围
        mn = [5, 2, 0]
        # 遍历所有的维度组合
        for (m, n) in itertools.product(mn, mn):
            size = (m, n)
            # 创建指定大小的稠密张量
            dense = make_tensor(size, dtype=dtype, device=device)
            # 将稠密张量转换为压缩格式的稀疏张量，并验证转换后是否与稠密张量相等
            sparse = to_compressed(dense)
            self.assertEqual(sparse.to_dense(), dense)

        # 定义批次形状
        batch_shape = (2, 3)
        # 创建压缩索引张量、普通索引张量和数值张量
        compressed_indices = torch.tensor([0, 3, 5], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        plain_indices = torch.tensor([0, 1, 2, 0, 1], device=device).repeat(6, 1).reshape(*batch_shape, -1)
        values = torch.tensor([1, 2, 1, 3, 4], device=device, dtype=dtype).repeat(6, 1).reshape(*batch_shape, -1)
        # 使用压缩构造器创建稀疏张量
        sparse = compressed_constructor(compressed_indices, plain_indices, values, dtype=dtype, device=device)
        # 获取稀疏张量的稠密形状
        dense_shape = get_dense_shape(sparse.shape, len(batch_shape))
        # 创建指定形状的稠密张量，并验证稀疏张量转换为稠密张量后是否与预期的稠密张量转置相等
        dense = torch.tensor([[1, 2, 1], [3, 4, 0]], dtype=dtype, device=device).repeat(6, 1).reshape(dense_shape)
        self.assertEqual(sparse.to_dense(), transpose(dense, len(batch_shape)))

    # 测试稀疏 CSR 格式转换为稠密格式
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_csr_to_dense(self, device, dtype):
        self._test_sparse_compressed_to_dense(device, dtype, torch.sparse_csr)

    # 测试稀疏 CSC 格式转换为稠密格式
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_csc_to_dense(self, device, dtype):
        self._test_sparse_compressed_to_dense(device, dtype, torch.sparse_csc)

    # 跳过特定的测试条件：无 MKL 稀疏库时跳过 CPU 测试
    @skipMeta
    @skipCPUIfNoMklSparse
    # 控制稀疏矩阵的稀疏化开关
    @coalescedonoff
    # 仅对双精度数据类型执行测试
    @dtypes(torch.double)
    # 测试将 COO 格式稀疏张量转换为 CSR 格式时的异常情况处理
    def test_coo_to_csr_convert(self, device, dtype, coalesced):
        # 使用断言验证输入张量是否为向量，若不是则抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "Input is supposed to be a vector"):
            torch._convert_indices_from_coo_to_csr(
                torch.randint(100, (5, 5), device=device),
                size=100)

        # 定义稀疏张量的大小、稀疏维度和非零元素数量
        size = (5, 5)
        sparse_dim = 2
        nnz = 10
        # 使用给定函数生成稀疏 COO 格式张量
        sparse_coo, _, _ = self.genSparseTensor(size, sparse_dim, nnz, coalesced, device, dtype)
        # 将稀疏 COO 格式张量转换为 CSR 格式
        sparse_csr = sparse_coo.to_sparse_csr()

        # 断言 CSR 格式张量的属性
        self.assertTrue(sparse_csr.is_sparse_csr)
        # 断言 CSR 格式张量与原始 COO 格式张量的密集表示是否一致
        self.assertEqual(sparse_csr.to_dense(), sparse_coo.to_dense())

        # 创建一个随机张量作为向量
        vec = torch.randn((5, 1), dtype=dtype, device=device)
        # 计算稀疏 COO 格式张量与向量的乘积
        coo_product = sparse_coo.matmul(vec)
        # 计算稀疏 CSR 格式张量与向量的乘积
        csr_product = sparse_csr.matmul(vec)

        # 断言两种格式乘积的结果是否相等
        self.assertEqual(coo_product, csr_product)

        # 创建一个更大的向量
        vec = torch.randn((100, 1), dtype=dtype, device=device)
        # 定义 COO 格式张量的索引和值
        index = torch.tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], dtype=torch.int32)
        values = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        # 创建稀疏 COO 格式张量
        coo = torch.sparse_coo_tensor(index, values, torch.Size([100, 100]), dtype=dtype, device=device)
        # 将稀疏 COO 格式张量转换为 CSR 格式
        csr = coo.to_sparse_csr()

        # 断言 COO 格式张量与 CSR 格式张量乘积的结果是否相等
        self.assertEqual(coo.matmul(vec), csr.matmul(vec))

        # 定义 CSR 格式张量的列索引
        col_indices = torch.tensor([
            31, 92, 65, 50, 34, 62, 22, 56, 74, 89
        ], dtype=torch.int64, device=device)
        # 断言 CSR 格式张量的列索引是否与预期相等
        self.assertEqual(csr.col_indices(), col_indices)

        # 定义 CSR 格式张量的值
        values = torch.tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7], dtype=dtype, device=device)
        # 断言 CSR 格式张量的值是否与预期相等
        self.assertEqual(csr.values(), values)

    # 参数化装饰器，测试 CSR 格式张量转换为块 CSR 格式张量时的各种情况
    @parametrize("blocksize", [2, 4])
    # 参数化装饰器，指定测试数据类型为双精度浮点和整型
    @dtypes((torch.double, torch.int32), (torch.double, torch.int64))
    # 如果未安装 SciPy 库，则跳过此测试
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    # 跳过元数据标记的测试用例
    @skipMeta
    def test_csr_to_block_csr(self, device, dtypes, blocksize):
        # 遍历不同形状的稀疏 CSR 格式张量
        for shape in [(24, 24), (12, 24)]:
            # 获取数据类型和索引类型
            dtype, index_dtype = dtypes
            m, k = shape
            # 随机生成稀疏 CSR 格式张量
            nnz = random.randint(0, m * k)
            t = self.genSparseCSRTensor((m * blocksize, k * blocksize), nnz, dtype=dtype,
                                        device=device, index_dtype=index_dtype)
            # 使用 SciPy 创建稀疏 CSR 格式矩阵
            st = sp.csr_matrix((t.values().cpu(), t.col_indices().cpu(), t.crow_indices().cpu()), shape=tuple(t.size()))
            # 将稀疏 CSR 格式张量转换为块 CSR 格式张量
            block_t = t.to_sparse_bsr((blocksize, blocksize))
            # 断言块 CSR 格式张量值的维度是否为 3
            self.assertEqual(block_t.values().dim(), 3)
            # 断言块 CSR 格式张量的布局是否为 torch.sparse_bsr
            self.assertTrue(block_t.layout == torch.sparse_bsr)
            # 使用 SciPy 将稀疏 CSR 格式矩阵转换为块 CSR 格式矩阵
            block_st = st.tobsr(blocksize=(blocksize, blocksize))
            block_st.sort_indices()
            # 断言块 CSR 格式张量的值与 SciPy 转换后的值是否相等
            self.assertEqual(block_t.values().cpu(), block_st.data)
            # 断言块 CSR 格式张量的列索引与 SciPy 转换后的列索引是否相等
            self.assertEqual(block_t.col_indices().cpu(), torch.tensor(block_st.indices).to(index_dtype))
            # 断言块 CSR 格式张量的行索引与 SciPy 转换后的行索引是否相等
            self.assertEqual(block_t.crow_indices().cpu(), torch.tensor(block_st.indptr).to(index_dtype))
    # 使用装饰器跳过测试，如果未安装 SciPy，则跳过此测试函数
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_csr_to_block_csr_errors(self, device, dtype):
        # 遍历索引数据类型的列表，测试函数多次
        for index_dtype in [torch.int32, torch.int64]:
            nnz = 15
            # 生成稀疏 CSR 张量，指定形状 (16, 16)，非零元素数为 nnz，数据类型为 dtype，
            # 存储设备为 device，索引数据类型为 index_dtype
            t = self.genSparseCSRTensor((16, 16), nnz, dtype=dtype,
                                        device=device, index_dtype=index_dtype)

            # 断言捕获运行时错误，并验证错误信息是否包含特定字符串
            with self.assertRaisesRegex(RuntimeError,
                                        r"tensor sparse size \(.*,.*\) must be divisible by given blocksize \(.*,.*\)"):
                # 尝试将 CSR 张量转换为块 CSR 张量，指定块大小为 (5, 5)
                block_t = t.to_sparse_bsr((5, 5))

    # TODO: 支持稀疏张量的设备自动生成检查
    # 参考：https://github.com/pytorch/pytorch/issues/59058
    @onlyCUDA
    @dtypes(torch.double)
    def test_matmul_device_mismatch(self, device, dtype):
        # 创建随机 CPU 张量和相应的 CUDA 张量
        cpu = torch.rand((10, 10))
        cuda = cpu.cuda()
        # 使用 itertools.product 遍历所有可能的组合
        for s, m1, m2 in itertools.product((cpu, cuda), repeat=3):
            # 将 m1 转换为稀疏 CSR 格式
            csr = m1.to_sparse()
            # 如果所有输入张量在同一设备上
            if s.device == csr.device == m2.device:
                # 执行张量乘法操作 s = s + csr * m2
                torch.addmm(s, csr, m2)
            else:
                # 断言捕获运行时错误，并验证错误信息是否包含特定字符串
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    # 尝试执行张量乘法操作，预期所有张量在同一设备上
                    torch.addmm(s, csr, m2)

    # 使用装饰器跳过测试，如果未安装 MKL Sparse，则跳过此测试函数
    @skipCPUIfNoMklSparse
    # 使用装饰器跳过测试，如果未安装通用的 CUDA 稀疏库，则跳过此测试函数
    @skipCUDAIfNoSparseGeneric
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater else [],
                  *[torch.bfloat16] if SM80OrLater else []))
    def test_csr_matvec(self, device, dtype):

        # 如果在 ROCm 平台下，并且数据类型为半精度或 bfloat16，则跳过此测试
        if TEST_WITH_ROCM and (dtype == torch.half or dtype == torch.bfloat16):
            self.skipTest("ROCm doesn't work with half dtypes correctly.")

        # 指定侧边长度为 100，遍历索引数据类型的列表
        side = 100
        for index_dtype in [torch.int32, torch.int64]:
            # 生成稀疏 CSR 张量，指定形状 (side, side)，非零元素数为 1000，
            # 存储设备为 device，数据类型为 dtype，索引数据类型为 index_dtype
            csr = self.genSparseCSRTensor((side, side), 1000, device=device, dtype=dtype, index_dtype=index_dtype)
            # 生成指定形状的随机张量，数据类型为 dtype，存储设备为 device
            vec = torch.randn(side, dtype=dtype, device=device)

            # 执行 CSR 矩阵与向量的乘法操作
            res = csr.matmul(vec)
            # 计算 CSR 矩阵的密集形式与向量的乘法结果（用于对比）
            expected = csr.to_dense().matmul(vec)

            # 断言验证两个结果张量相等
            self.assertEqual(res, expected)

            # 创建形状为 (side + 10) 的随机向量，数据类型为 dtype，存储设备为 device
            bad_vec = torch.randn(side + 10, dtype=dtype, device=device)
            # 断言捕获运行时错误，并验证错误信息是否包含特定字符串
            err_msg = "size mismatch, got"
            with self.assertRaisesRegex(RuntimeError, err_msg):
                # 尝试执行 CSR 矩阵与 bad_vec 的乘法操作，预期出现尺寸不匹配错误
                csr.matmul(bad_vec)

    # 使用装饰器跳过测试，如果 ROCm 版本低于 5.2，则跳过此测试函数
    @onlyCUDA
    @skipCUDAIfRocmVersionLessThan((5, 2))
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 定义一个测试方法 test_baddbmm，接受 device 和 dtype 作为参数
    def test_baddbmm(self, device, dtype):

        # 在 torch.baddbmm 中禁用不变性检查的装饰器，以避免处理非常规的 csr 张量，可能导致异常
        @torch.sparse.check_sparse_tensor_invariants(enable=False)
        def run_test(c, a, a_batched, b, op_b=False, op_out=False, *, dtype=None, device=None):
            # 如果 dtype 是复数类型，则 alpha 和 beta 是复数；否则为随机数
            alpha = complex(random.random(), random.random()) if dtype.is_complex else random.random()
            beta = complex(random.random(), random.random()) if dtype.is_complex else random.random()

            # 如果 op_b 为真且 a 的形状与 b 的形状相同，则使用 b 的共轭转置
            b = b.mH if (op_b and a.shape == b.shape) else b

            # 使用 torch.baddbmm 计算实际输出
            actual = torch.baddbmm(c, a_batched, b, alpha=alpha, beta=beta)

            # 创建一个与 c 形状相同的空张量 out，如果 op_out 为真且 a 的形状与 b 的形状相同，则使用 c 的共轭转置
            out = torch.empty_like(c.mH if op_out and a.shape == b.shape else c)

            # 使用 torch.baddbmm 计算，并将结果存储到 out 中
            torch.baddbmm(c, a_batched, b, alpha=alpha, beta=beta, out=out)

            # 期望的输出是对每个 batch 中的元素执行 torch.addmm 操作后得到的张量，组成一个张量数组
            expected = [torch.addmm(c[i], a, b[i], alpha=alpha, beta=beta) for i in range(c.shape[0])]
            expected = torch.stack(expected, 0)

            # 断言实际输出与 out 和 expected 相等
            self.assertEqual(actual, out)
            self.assertEqual(actual, expected)

        # 针对不同的 index_dtype 进行测试
        for index_dtype in [torch.int32, torch.int64]:
            # 使用 itertools 生成 m, n, k 的所有组合
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([2, 5], repeat=3), [1, 3], [True, False]):
                # 随机生成稀疏 CSR 张量 a
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)

                # 创建带有批处理维度的正常 CSR 张量 a_batched，形状为 (batch_size, m, k)
                a_batched = torch.sparse_csr_tensor(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k), check_invariants=False)

                # 创建形状为 (batch_size, k, n) 的张量 b 和形状为 (batch_size, m, n) 的张量 c
                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                c = make_tensor((batch_size, m, n), dtype=dtype, device=device, noncontiguous=noncontiguous)

                # 对 op_b 和 op_out 的所有可能组合运行测试
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    run_test(c, a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    # 使用 onlyCUDA 装饰器标记该测试仅在 CUDA 下运行
    @onlyCUDA
    # 如果在 ROCm 环境下，跳过此测试，因为仅支持 CUDA 11+
    @unittest.skipIf(TEST_WITH_ROCM, "Only CUDA 11+ is supported")
    # 如果没有通用稀疏 GPU 支持，跳过此测试
    @skipCUDAIfNoSparseGeneric
    # 指定测试的数据类型范围
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 定义测试方法 test_bmm，接受设备和数据类型作为参数
    def test_bmm(self, device, dtype):
        # 定义内部方法 run_test，用于执行测试
        def run_test(a, a_batched, b, op_b=False, op_out=False, *, dtype=None, device=None):
            # 如果 op_b 为真且 a 的形状与 b 的形状相同，则使用 b 的共轭转置（mH）
            b = b.mH if (op_b and a.shape == b.shape) else b

            # 计算实际结果，torch.bmm 是批量矩阵乘法的函数
            actual = torch.bmm(a_batched, b)

            # 根据 op_out 和 a 与 b 的形状，创建一个与 actual 相同形状的空张量 out
            out = torch.empty_like(actual.mH if op_out and a.shape == b.shape else actual)
            # 执行批量矩阵乘法，并将结果存储在 out 中
            torch.bmm(a_batched, b, out=out)

            # 计算预期结果，对每个 b 的批次执行矩阵乘法
            expected = [torch.mm(a, b[i]) for i in range(b.shape[0])]
            expected = torch.stack(expected, 0)

            # 使用断言函数检查 actual 和 out 是否相等
            self.assertEqual(actual, out)
            # 使用断言函数检查 actual 和 expected 是否相等
            self.assertEqual(actual, expected)

        # 遍历所有可能的 index_dtype，包括 torch.int32 和 torch.int64
        for index_dtype in [torch.int32, torch.int64]:
            # 使用 itertools.product 生成所有可能的参数组合
            for (m, n, k), batch_size, noncontiguous in zip(itertools.product([2, 5], repeat=3), [1, 3], [True, False]):
                # 随机生成稀疏 CSR 张量 a
                nnz = random.randint(0, m * k)
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)

                # 创建一个带有批次维度的常规 CSR 张量 a_batched
                # 在 PyTorch 中以这种方式表示批次稀疏张量是不正规的，
                # 因此在此处关闭了检查不变量的检查。
                a_batched = torch.sparse_csr_tensor(
                    a.crow_indices(), a.col_indices(), a.values(), (batch_size, m, k), check_invariants=False)

                # 创建张量 b，形状为 (batch_size, k, n)，支持设备和非连续存储
                b = make_tensor((batch_size, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                # 使用 itertools.product 生成 op_b 和 op_out 的所有可能组合
                for op_b, op_out in itertools.product([True, False], repeat=2):
                    # 执行测试
                    run_test(a, a_batched, b, op_b, op_out, dtype=dtype, device=device)

    # 定义测试方法 run_test_block_addmm_addmv，用于测试块矩阵加法和乘法
    def run_test_block_addmm_addmv(self,
                                   addmv_addmm,
                                   c,
                                   a,
                                   b,
                                   op_b=False,
                                   op_out=False,
                                   *,
                                   dtype=None,
                                   device=None,
                                   ref=_npref_block_addmm_addmv):
        # 根据数据类型是复数还是实数，随机生成 alpha 和 beta
        alpha = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        beta = complex(random.random(), random.random()) if dtype.is_complex else random.random()
        # 如果 op_b 为真且 a 的形状与 b 的形状相同，则使用 b 的共轭转置（mH）
        b = b.mH if (op_b and a.shape == b.shape) else b

        # 执行 addmv_addmm 函数，计算实际结果
        actual = addmv_addmm(c, a, b, alpha=alpha, beta=beta)

        # 根据 op_out 和 a 与 b 的形状，创建一个与 c 相同形状的空张量 out
        out = torch.empty_like(c.mH if op_out and a.shape == b.shape else c)
        # 执行 addmv_addmm 函数，并将结果存储在 out 中
        addmv_addmm(c, a, b, alpha=alpha, beta=beta, out=out)
        
        # 计算预期结果，使用 ref 函数计算
        expected = ref(c, a, b, alpha, beta)

        # 使用断言函数检查 actual 和 out 是否相等
        self.assertEqual(actual, out)
        # 使用断言函数检查 actual 和 expected 是否相等，如果不相等则输出详细信息
        self.assertEqual(actual, expected, lambda msg: f"{msg}\na={a}\nc={c}\nb={b}\nalpha={alpha} beta={beta}")

    # TODO: block_size 1 is broken
    # 使用参数化装饰器指定 block_size、index_dtype 和 noncontiguous 参数的测试用例
    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    # 跳过没有 MKL Sparse 支持的情况
    @skipCPUIfNoMklSparse
    # 如果未安装 SciPy，则跳过当前测试
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    # 如果在 Torch Dynamo 中，跳过这个测试
    @skipIfTorchDynamo("raises 'sparse matrix length is ambiguous; use getnnz()'")
    # 根据浮点数和复数类型动态设置数据类型
    @dtypes(*floating_and_complex_types())
    # 如果使用 CUDA，则根据 GPU 架构版本动态设置数据类型
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater else [],
                  *[torch.bfloat16] if SM80OrLater else []))
    # 设置特定精度覆盖值，针对不同数据类型
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-5, torch.complex128: 1e-5,
                        torch.float16: 1e-3, torch.bfloat16: 1e-3})
    # 使用不同的块大小参数化测试
    @parametrize("block_size", [2, 3])
    # 使用不同的索引数据类型参数化测试
    @parametrize("index_dtype", [torch.int32, torch.int64])
    # 使用是否连续参数化测试
    @parametrize("noncontiguous", [True, False])
    # 如果未安装 SciPy，则跳过当前测试
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    # 根据特定数据类型参数化测试
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 测试 block_addmv 方法的功能
    def test_block_addmv(self, device, dtype, index_dtype, block_size, noncontiguous):
        # TODO: 明确禁用块大小为 1 的支持
        # if (TEST_WITH_ROCM or not TEST_CUSPARSE_GENERIC) and block_size == 1:
        #     return
        
        # 定义参考的 block_addmv 函数
        def ref_block_addmv(c, a, b, alpha, beta):
            return _npref_block_addmm_addmv(c, a.to_dense(), b, alpha, beta)

        # 对给定的 m 和 k 参数进行排列组合测试
        for (m, k) in itertools.product([2, 5], repeat=2):
            # 随机生成非零元素个数 nnz
            nnz = random.randint(0, m * k)
            if not noncontiguous:
                # 生成稀疏 CSR 格式的张量 a，并转换为稀疏 BSR 格式
                a = self.genSparseCSRTensor((m * block_size, k * block_size), nnz,
                                            dtype=dtype, device=device, index_dtype=index_dtype)
                a = a.to_sparse_bsr((block_size, block_size))
            else:
                # 生成非连续的稀疏 CSR 格式的张量 a
                a = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                a_data = make_tensor((nnz, block_size, block_size), dtype=dtype, device=device)
                a_data = a_data.mT if noncontiguous else a_data   # 测试列主要块
                # 创建稀疏 BSR 格式的张量 a
                a = torch.sparse_bsr_tensor(a.crow_indices(), a.col_indices(),
                                            a_data, (m * block_size, k * block_size), check_invariants=False)
            
            # 生成张量 b，用于 block_addmv
            b = make_tensor((k * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            # 生成张量 c，用于 block_addmv
            c = make_tensor((m * block_size,), dtype=dtype, device=device, noncontiguous=noncontiguous)
            # 运行测试 block_addmv 方法
            self.run_test_block_addmm_addmv(torch.addmv, c, a, b, dtype=dtype, device=device, ref=ref_block_addmv)

    # 参数化测试不同的矩阵形状
    @parametrize("matrix_shape", [(3, 3), (5, 7), (11, 9)], name_fn=lambda x: "shape_{}x{}".format(*x))
    # 参数化测试不同的数据类型
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 仅在 CPU 上运行测试
    @onlyCPU
    # 定义一个测试方法，用于测试 torch.addmv 函数的功能
    def test_addmv(self, device, dtype, matrix_shape):
        # 创建一个指定形状的随机张量 mat，数据类型为 dtype，设备为 device
        mat = torch.randn(matrix_shape, dtype=dtype, device=device)
        # 将 mat 中实部小于 0 的元素置为 0
        mat[mat.real < 0] = 0
        # 将 mat 转换为稀疏 CSR 格式的张量 sparse_mat
        sparse_mat = mat.to_sparse_csr()
        # 创建一个随机向量 mvec，数据类型为 dtype，设备为 device
        mvec = torch.randn((mat.size(1),), dtype=dtype, device=device)
        # 创建一个随机向量 avec，数据类型为 torch.float64，设备为 device
        avec = torch.randn((mat.size(0),), dtype=torch.float64, device=device)
        # 使用 torch.addmv 计算 ref_output，作为参考输出
        ref_output = torch.addmv(avec, mat, mvec)
        # 使用 torch.addmv 计算 output，测试稀疏矩阵 sparse_mat 的加权向量乘法
        output = torch.addmv(avec, sparse_mat, mvec)
        # 断言 ref_output 与 output 相等
        self.assertEqual(ref_output, output)

    # 参数化测试方法，测试 torch.sparse.mm 函数的功能
    @parametrize("block_size", [2, 3])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @parametrize("noncontiguous", [True, False])
    @skipCPUIfNoMklSparse  # 如果没有 MKL Sparse，跳过 CPU 测试
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")  # 如果没有安装 SciPy，跳过测试
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)  # 参数化数据类型
    @skipCPUIfNoMklSparse  # 如果没有 MKL Sparse，跳过 CPU 测试
    @unittest.skipIf(TEST_WITH_ROCM, "Only CUDA 11+ is supported")  # 如果是 ROCm 环境，仅支持 CUDA 11+，跳过测试
    @dtypes(torch.double)  # 指定数据类型为 torch.double
    @skipCPUIfNoMklSparse  # 如果没有 MKL Sparse，跳过 CPU 测试
    @dtypes(*floating_and_complex_types())  # 参数化浮点数和复数类型数据
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater and TEST_CUSPARSE_GENERIC else [],  # 如果满足条件，包含 torch.half
                  *[torch.bfloat16] if SM80OrLater and TEST_CUSPARSE_GENERIC else []))  # 如果满足条件，包含 torch.bfloat16
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})  # 设置精度覆盖，torch.bfloat16 和 torch.float16 精度为 1e-2
    def test_sparse_mm(self, device, dtype):
        # 定义测试形状函数 test_shape，用于测试不同形状的稀疏矩阵乘法
        def test_shape(d1, d2, d3, nnz, transposed, index_dtype):
            # 如果 transposed 为 True，则创建随机张量 D，并转置
            if transposed:
                D = torch.randn(d3, d2, dtype=dtype, device=device).t_()
            else:
                # 否则，创建随机张量 D
                D = torch.randn(d2, d3, dtype=dtype, device=device)
            # 使用 self.genSparseCSRTensor 方法生成稀疏 CSR 格式的张量 S
            S = self.genSparseCSRTensor((d1, d2), nnz, device=device, dtype=dtype, index_dtype=index_dtype)
            # 将稀疏张量 S 转换为稠密张量 S_dense
            S_dense = S.to_dense()
            # 断言 torch.sparse.mm(S, D) 的结果与 torch.mm(S_dense, D) 相等
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

        # 遍历 index_dtype，分别测试指定参数下的稀疏矩阵形状
        for index_dtype in [torch.int32, torch.int64]:
            test_shape(7, 8, 9, 20, False, index_dtype)  # 测试非转置的形状
            test_shape(7, 8, 9, 20, True, index_dtype)   # 测试转置后的形状

    # 参数化测试方法，测试不同浮点数和复数类型的精度覆盖
    @dtypes(*floating_and_complex_types())
    @dtypesIfCUDA(*floating_and_complex_types_and(
                  *[torch.half] if SM53OrLater and TEST_CUSPARSE_GENERIC else [],  # 如果满足条件，包含 torch.half
                  *[torch.bfloat16] if SM80OrLater and TEST_CUSPARSE_GENERIC else []))  # 如果满足条件，包含 torch.bfloat16
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})  # 设置精度覆盖，torch.bfloat16 和 torch.float16 精度为 1e-2
    # 定义一个测试函数，用于测试稀疏矩阵的加法乘法操作
    def test_sparse_addmm(self, device, dtype):
        # 定义内部函数，测试不同形状的稀疏矩阵加法乘法操作
        def test_shape(m, n, p, nnz, broadcast, index_dtype, alpha_beta=None):
            # 如果未提供 alpha_beta 参数，则随机生成 alpha 和 beta
            if alpha_beta is None:
                alpha = random.random()
                beta = random.random()
            else:
                alpha, beta = alpha_beta
            # 根据 broadcast 参数创建 D1 张量，形状根据 broadcast 的值确定
            if broadcast:
                D1 = make_tensor((), dtype=dtype, device=device)
            else:
                D1 = make_tensor([n, p], dtype=dtype, device=device)
            # 创建形状为 [m, p] 的 D2 张量
            D2 = make_tensor([m, p], dtype=dtype, device=device)
            # 使用给定的参数生成一个稀疏 CSR 格式的张量 S
            S = self.genSparseCSRTensor([n, m], nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            # 将稀疏张量 S 转换为稠密张量 S_dense
            S_dense = S.to_dense()
            # 使用 torch.sparse.addmm 进行稀疏矩阵-稠密矩阵乘法加法运算
            Y = torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            # 使用 torch.addmm 进行稠密矩阵乘法加法运算
            Y_dense = torch.addmm(D1, S_dense, D2, beta=beta, alpha=alpha)
            # 断言稀疏矩阵加法乘法的结果与稠密矩阵加法乘法的结果相等
            self.assertEqual(Y, Y_dense)

        # 遍历不同的索引数据类型进行测试
        for index_dtype in [torch.int32, torch.int64]:
            # 不进行广播的测试
            test_shape(7, 8, 9, 20, False, index_dtype, None)
            # 进行广播的测试
            test_shape(7, 8, 9, 20, True, index_dtype, None)
            # 使用自定义的 alpha_beta 参数进行测试
            test_shape(7, 8, 9, 20, False, index_dtype, (1, 0))
            test_shape(7, 8, 9, 20, True, index_dtype, (1, 0))
            test_shape(7, 8, 9, 20, False, index_dtype, (1, 1))
            test_shape(7, 8, 9, 20, True, index_dtype, (1, 1))

    # 如果没有 MKL Sparse 支持，则跳过这个测试
    @skipCPUIfNoMklSparse
    # 指定测试的数据类型为浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 对于不同精度的数据类型，指定数值精度覆盖值
    @precisionOverride({torch.double: 1e-8, torch.float: 1e-4, torch.bfloat16: 0.6,
                        torch.half: 1e-1, torch.cfloat: 1e-4, torch.cdouble: 1e-8})
    # 如果在 CUDA 环境下，指定兼容的数据类型以及硬件特定条件
    @dtypesIfCUDA(*floating_types_and(torch.complex64,
                                      *[torch.bfloat16] if SM80OrLater else [],
                                      *[torch.half] if SM53OrLater else [],
                                      *[torch.complex128] if CUSPARSE_SPMM_COMPLEX128_SUPPORTED else []))
    # 使用稀疏压缩非块布局进行测试
    @sparse_compressed_nonblock_layouts()
    # 如果 cuSparse 通用 API SpGEMM 不可用，则跳过 CUDA 测试
    @skipCUDAIf(
        not _check_cusparse_spgemm_available(),
        "cuSparse Generic API SpGEMM is not available"
    )
    # 定义一个测试函数，用于测试稀疏 CSR 矩阵的 torch.addmm 函数
    def test_addmm_all_sparse_csr(self, device, dtype, layout):
        # 创建一个大小为 10x25 的随机张量 M，设备为指定设备，数据类型为指定类型
        M = torch.randn(10, 25, device=device).to(dtype)
        # 创建两个大小分别为 10x50 和 50x25 的随机张量 m1 和 m2，设备和数据类型均为指定
        m1 = torch.randn(10, 50, device=device).to(dtype)
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 函数进行测试，传入参数 M, m1, m2，指定布局和模式为 "all_sparse"
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="all_sparse")

        # Test 0-strided
        # 创建一个大小为 10x1 的随机张量 M，设备为指定设备，数据类型为指定类型，并扩展为 10x25
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        # 创建一个大小为 10x1 的随机张量 m1，设备为指定设备，数据类型为指定类型，并扩展为 10x50
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        # 创建一个大小为 50x25 的随机张量 m2，设备和数据类型均为指定
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 函数进行测试，传入参数 M, m1, m2，指定布局和模式为 "all_sparse"
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="all_sparse")

        # Test beta=0, M=nan
        # 创建一个大小为 10x25 的全为 NaN 的张量 M，设备为指定设备，数据类型为指定类型
        M = torch.full((10, 25), float('nan'), device=device).to(dtype)
        # 创建一个大小为 10x50 的随机张量 m1，设备为指定设备，数据类型为指定类型
        m1 = torch.randn(10, 50, device=device).to(dtype)
        # 创建一个大小为 50x25 的随机张量 m2，设备和数据类型均为指定
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 函数进行测试，传入参数 M, m1, m2，beta 设为 0，指定布局和模式为 "all_sparse"
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, beta=0, layout=layout, mode="all_sparse")

        # Test transpose
        # 使用 itertools.product 生成四个布尔值的组合进行迭代
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            # 定义一个函数 maybe_transpose，根据条件是否转置输入的张量 m
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                # 如果条件为真，对 m 进行转置并克隆为内存连续格式，然后再次转置
                return m.t().clone(memory_format=torch.contiguous_format).t()

            # 根据 t1, t2, t3, t4 条件，可能对随机张量 M, m1, m2 进行转置操作
            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            # 调用 _test_addmm_addmv 函数进行测试，传入参数 M, m1, m2，transpose_out 设为 t4，指定布局和模式为 "all_sparse"
            _test_addmm_addmv(self, torch.addmm, M, m1, m2, transpose_out=t4, layout=layout, mode="all_sparse")

    # 装饰器函数，要求测试仅在 CPU 上运行
    @onlyCPU
    # 装饰器函数，要求在 CPU 上运行时需要 MKL 支持稀疏计算
    @skipCPUIfNoMklSparse
    # 装饰器函数，指定接受浮点数和复数类型作为参数
    @dtypes(*floating_and_complex_types())
    # 装饰器函数，指定接受压缩稀疏非块布局作为参数
    @sparse_compressed_nonblock_layouts()
    # 定义一个测试方法，用于测试 torch.addmm 函数在不同条件下的行为
    def test_addmm_dense_result(self, device, dtype, layout):
        # 创建随机初始化的矩阵 M，形状为 10x25，设备为指定的 device，并转换为指定的 dtype 类型
        M = torch.randn(10, 25, device=device).to(dtype)
        # 创建随机初始化的矩阵 m1，形状为 10x50，设备为指定的 device，并转换为指定的 dtype 类型
        m1 = torch.randn(10, 50, device=device).to(dtype)
        # 创建随机初始化的矩阵 m2，形状为 50x25，设备为指定的 device，并转换为指定的 dtype 类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 方法来测试 torch.addmm 函数，传入 M, m1, m2 以及其他参数
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="dense_result")

        # 测试特殊情况：0-strided
        # 创建随机初始化的矩阵 M，形状为 10x1，设备为指定的 device，转换为指定的 dtype 类型，然后扩展为 10x25
        M = torch.randn(10, 1, device=device).to(dtype).expand(10, 25)
        # 创建随机初始化的矩阵 m1，形状为 10x1，设备为指定的 device，转换为指定的 dtype 类型，然后扩展为 10x50
        m1 = torch.randn(10, 1, device=device).to(dtype).expand(10, 50)
        # 创建随机初始化的矩阵 m2，形状为 50x25，设备为指定的 device，并转换为指定的 dtype 类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 方法来测试 torch.addmm 函数，传入 M, m1, m2 以及其他参数
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=layout, mode="dense_result")

        # 测试特殊情况：beta=0, M 元素为 NaN
        # 创建元素全为 NaN 的矩阵 M，形状为 10x25，设备为指定的 device，并转换为指定的 dtype 类型
        M = torch.full((10, 25), float('nan'), device=device).to(dtype)
        # 创建随机初始化的矩阵 m1，形状为 10x50，设备为指定的 device，并转换为指定的 dtype 类型
        m1 = torch.randn(10, 50, device=device).to(dtype)
        # 创建随机初始化的矩阵 m2，形状为 50x25，设备为指定的 device，并转换为指定的 dtype 类型
        m2 = torch.randn(50, 25, device=device).to(dtype)
        # 调用 _test_addmm_addmv 方法来测试 torch.addmm 函数，传入 M, m1, m2，beta=0 以及其他参数
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, beta=0, layout=layout, mode="dense_result")

        # 测试转置情况
        # 使用 itertools.product 生成 t1, t2, t3, t4 的所有 True 和 False 组合，并进行迭代
        for t1, t2, t3, t4 in itertools.product([True, False], repeat=4):
            # 定义一个函数 maybe_transpose，根据条件 cond 是否为 True 来转置矩阵 m
            def maybe_transpose(cond, m):
                if not cond:
                    return m
                # 如果 cond 为 True，则对 m 进行转置，并使用连续内存格式（contiguous format）进行克隆
                return m.t().clone(memory_format=torch.contiguous_format).t()

            # 根据 t1 的值，选择是否对随机初始化的矩阵 M 进行转置
            M = maybe_transpose(t1, torch.randn(10, 25, device=device).to(dtype))
            # 根据 t2 的值，选择是否对随机初始化的矩阵 m1 进行转置
            m1 = maybe_transpose(t2, torch.randn(10, 50, device=device).to(dtype))
            # 根据 t3 的值，选择是否对随机初始化的矩阵 m2 进行转置
            m2 = maybe_transpose(t3, torch.randn(50, 25, device=device).to(dtype))
            # 调用 _test_addmm_addmv 方法来测试 torch.addmm 函数，传入 M, m1, m2，transpose_out=t4 以及其他参数
            _test_addmm_addmv(self, torch.addmm, M, m1, m2, transpose_out=t4, layout=layout, mode="dense_result")
    def test_addmm_sizes_all_sparse_csr(self, device, dtype, m, n, k):
        # 如果在 ROCm 平台上且 k、n、m 均不为零，则跳过测试
        if (TEST_WITH_ROCM and k != 0 and n != 0 and m != 0):
            self.skipTest("Skipped on ROCm")
        
        # 创建随机的 dense tensor M、m1 和 m2
        M = torch.randn(n, m, device=device).to(dtype)
        m1 = torch.randn(n, k, device=device).to(dtype)
        m2 = torch.randn(k, m, device=device).to(dtype)
        
        # 调用 _test_addmm_addmv 函数测试 torch.addmm 方法，使用 sparse_csr 格式
        _test_addmm_addmv(self, torch.addmm, M, m1, m2, layout=torch.sparse_csr, mode="all_sparse")

        # 创建随机的 sparse_csr tensor M、m1 和 m2
        M = torch.randn(n, m, device=device).to(dtype).to_sparse_csr()
        m1 = torch.randn(n, k + 1, device=device).to(dtype).to_sparse_csr()
        m2 = torch.randn(k, m, device=device).to(dtype).to_sparse_csr()
        
        # 测试错误情况：期望引发 RuntimeError，并验证错误消息格式
        self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.addmm(M, m1, m2))
        self.assertRaisesRegex(RuntimeError, f"{n}x{k + 1}.*{k}x{m}", lambda: torch.mm(m1, m2))

    @skipCPUIfNoMklSparse
    @dtypes(torch.float)
    def test_addmm_errors(self, device, dtype):
        # 测试 dense 和 sparse 版本的错误应该相同
        import re

        def test1(*, is_sparse):
            # 矩阵乘法必须兼容形状
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.addmm(a, a_sparse, a)
            else:
                return torch.addmm(a, a, a)

        def test2(*, is_sparse):
            # mat2 必须是一个矩阵
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.addmm(a, a_sparse, a.unsqueeze(0))
            else:
                return torch.addmm(a, a, a.unsqueeze(0))

        def test3(*, is_sparse):
            # 第一个输入必须是 1D 或 2D
            a = make_tensor((3, 3), dtype=dtype, device=device)
            if is_sparse:
                a_sparse = a.to_sparse_csr()
                return torch.addmm(a.unsqueeze(0), a_sparse, a)
            else:
                return torch.addmm(a.unsqueeze(0), a, a)

        # 对 test1、test2 和 test3 函数进行迭代测试
        for test in (test1, test2, test3):
            try:
                test(is_sparse=False)
            except RuntimeError as msg:
                # 确保在 is_sparse=True 时引发相同的 RuntimeError，并验证错误消息
                with self.assertRaisesRegex(RuntimeError, re.escape(str(msg))):
                    test(is_sparse=True)

    @skipCPUIfNoMklSparse
    @dtypes(torch.float)
    def test_mm_errors(self, device, dtype):
        # 测试稠密和稀疏版本的矩阵乘法错误是否相同

        import re  # 导入正则表达式模块

        def test1(*, is_sparse):
            # 矩阵乘法需要保证形状兼容
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                # 如果是稀疏矩阵，将其转换为 CSR 格式后进行乘法运算
                a_sparse = a.to_sparse_csr()
                return torch.mm(a_sparse, a)
            else:
                return torch.mm(a, a)

        def test2(*, is_sparse):
            # mat2 必须是一个矩阵
            a = make_tensor((2, 3), dtype=dtype, device=device)
            if is_sparse:
                # 如果是稀疏矩阵，将其转换为 CSR 格式后进行乘法运算
                a_sparse = a.to_sparse_csr()
                return torch.mm(a_sparse, a.unsqueeze(0))
            else:
                return torch.mm(a, a.unsqueeze(0))

        for test in (test1, test2):
            try:
                test(is_sparse=False)
            except RuntimeError as msg:
                with self.assertRaisesRegex(RuntimeError, re.escape(str(msg))):
                    test(is_sparse=True)

    @sparse_compressed_nonblock_layouts()
    @dtypes(torch.float, torch.double)
    def test_add(self, device, layout, dtype):
        def _test_spadd_shape(nnz, shape):
            # sparse.to_dense() 在内部使用 torch.add，如果 torch.add 出错，
            # 密集张量将会错误，但此测试仍会通过
            # 还有一个单独的测试检查 .to_dense() 调用的正确性
            x = self.genSparseCompressedTensor(shape, nnz,
                                               dtype=dtype,
                                               device=device,
                                               index_dtype=torch.int32,
                                               layout=layout,
                                               blocksize=())
            y = torch.randn(*shape, dtype=dtype, device=device)
            r = random.random()

            # 执行 torch.add 操作
            res = torch.add(y, x, alpha=r)
            expected = y + r * x.to_dense()
            self.assertEqual(res, expected)

            # 交换顺序后的 torch.add 操作
            res_perm = torch.add(x, y, alpha=r)
            self.assertEqual(res_perm, expected)

            # 非连续的密集张量
            s = list(shape)
            s[0] = shape[-1]
            s[-1] = shape[0]
            y = torch.randn(*s, dtype=torch.double, device=device)
            y.transpose_(0, len(s) - 1)
            r = random.random()

            # 执行 torch.add 操作
            res = torch.add(y, x, alpha=r)
            expected = y + r * x.to_dense()

            # 交换顺序后的 torch.add 操作
            res_perm = torch.add(x, y, alpha=r)

            self.assertEqual(res, expected)
            self.assertEqual(res_perm, expected)


        ns = [2, 5]
        batch_shapes = [(), (2,), (2, 3)]
        for b, m, n in itertools.product(batch_shapes, ns, ns):
            _test_spadd_shape(0, (*b, m, n))
            _test_spadd_shape(m * n // 2, (*b, m, n))
            _test_spadd_shape(m * n, (*b, m, n))

    @dtypes(torch.float, torch.double)
    # TODO: enable hybrid once to_dense supports it
    # 设置参数化测试，参数为'enable_hybrid'，值为False
    @parametrize('enable_hybrid', [False])
    # 应用所有稀疏压缩布局的装饰器
    @all_sparse_compressed_layouts()
    # 定义数据类型参数，包括所有数据类型以及特殊类型（torch.bool, torch.bfloat16, torch.half）
    @dtypes(*all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half))
    # 测试矩阵乘以标量的方法
    def test_mul_scalar(self, layout, device, dtype, enable_hybrid):
        # 生成简单输入的稀疏张量，包括布局、设备、数据类型、索引数据类型以及是否启用混合模式
        for sparse in self.generate_simple_inputs(
                layout, device=device, dtype=dtype, index_dtype=torch.int32, enable_hybrid=enable_hybrid):
            # 遍历所有数据类型及其复数形式，跳过 torch.half 数据类型的复数形式（ComplexHalf 是实验性的）
            for scalar_dtype in all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half):
                if dtype is torch.half and scalar_dtype.is_complex:
                    continue

                # 创建标量张量
                scalar_t = torch.tensor(2, dtype=scalar_dtype)
                # 如果标量是张量，则取其标量值
                for scalar in (scalar_t, scalar_t.item()):
                    # 对稀疏张量进行标量乘法操作
                    res_out = sparse.mul(scalar)
                    # 验证乘法结果与预期相等
                    self.assertEqual(res_out, scalar * sparse)

                    # 将稀疏张量转换为密集张量，再进行标量乘法操作
                    res_dense_out = sparse.to_dense().mul(scalar)
                    # BUG: 调度程序忽略了 mul.Scalar(Tensor, Scalar) 操作
                    # 这个问题在 mul(Tensor, Tensor) 内核中绕过处理
                    self.assertEqual(res_out, res_dense_out)

                    # 如果数据类型与乘法结果的类型一致，则在原地进行标量乘法操作
                    if dtype == torch.result_type(sparse, scalar):
                        res_in_dense = sparse.to_dense().mul_(scalar)
                        # 在稀疏张量上执行原地乘法操作
                        res_in = sparse.clone().mul_(scalar)
                        # 验证原地操作后的结果与预期相等
                        self.assertEqual(res_in, res_in_dense)
                        self.assertEqual(res_out, res_in)

    # 如果没有 MKL 支持则跳过 CPU 测试
    @skipCPUIfNoMklSparse
    # 定义数据类型参数，包括 torch.float32, torch.float64, torch.complex64, torch.complex128
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 定义测试函数，用于测试稀疏张量的加法操作
    def test_sparse_add(self, device, dtype):
        # 定义内部函数run_test，用于执行单个测试
        def run_test(m, n, index_dtype):

            # 随机生成 alpha 值
            alpha = random.random()
            # 随机生成三个稀疏张量的非零元素数量
            nnz1 = random.randint(0, m * n)
            nnz2 = random.randint(0, m * n)
            nnz3 = random.randint(0, m * n)

            # 如果在 ROCm 平台下测试，确保 nnz 不为零，因为 ROCm 不支持 nnz = 0 的情况
            if TEST_WITH_ROCM:
                nnz1, nnz2, nnz3 = max(1, nnz1), max(1, nnz2), max(1, nnz3)

            # 使用给定参数生成三个稀疏 CSR 张量
            S1 = self.genSparseCSRTensor([m, n], nnz1, dtype=dtype, device=device, index_dtype=index_dtype)
            S2 = self.genSparseCSRTensor([m, n], nnz2, dtype=dtype, device=device, index_dtype=index_dtype)
            S3 = self.genSparseCSRTensor([m, n], nnz3, dtype=dtype, device=device, index_dtype=index_dtype)
            sparse_args = [S1, S2, S3]
            # 将稀疏张量转换为稠密张量，并组成列表
            dense_args = [t.to_dense() for t in sparse_args]
            # 创建参数索引的列表
            arg_idx = list(range(len(sparse_args)))
            # 输出索引为 None，表示输出为新创建的张量
            out_idx = arg_idx + [None]

            # 使用 itertools 生成参数组合
            for idx1, idx2, idx3 in itertools.product(arg_idx, arg_idx, out_idx):
                s1 = sparse_args[idx1]
                s2 = sparse_args[idx2]
                s3 = None if idx3 is None else sparse_args[idx3]
                d1 = dense_args[idx1]
                d2 = dense_args[idx2]
                d3 = None if idx3 is None else dense_args[idx3]

                # 预期的加法结果，使用 alpha 和给定的输出张量 d3
                expected = torch.add(d1, d2, alpha=alpha, out=d3)
                # 实际的稀疏张量加法结果，使用 alpha 和给定的输出张量 s3
                actual = torch.add(s1, s2, alpha=alpha, out=s3)
                # 断言稀疏张量的行索引和列索引的数据类型符合预期
                self.assertEqual(actual.crow_indices().dtype, index_dtype)
                self.assertEqual(actual.col_indices().dtype, index_dtype)
                # 断言实际结果等于预期结果
                self.assertEqual(actual, expected)
                # 断言稀疏输出张量 s3 等于稠密输出张量 d3
                self.assertEqual(s3, d3)
                # 如果 s3 不为 None，则断言其行索引和列索引的数据类型符合预期
                if s3 is not None:
                    self.assertEqual(s3.crow_indices().dtype, index_dtype)
                    self.assertEqual(s3.col_indices().dtype, index_dtype)

        # 对于指定的数据类型进行测试
        for index_dtype in [torch.int32, torch.int64]:
            # 对于给定的 m 和 n 值进行测试
            for m, n in itertools.product([3, 5], [3, 5]):
                run_test(m, n, index_dtype)

    # 使用装饰器指定多种数据类型进行测试的函数
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_sparse_add_errors(self, device, dtype):
        # 定义内部函数run_test，用于测试稀疏张量加法时的错误情况
        def run_test(index_type):
            # 生成两个形状不同的稀疏张量 a 和 b
            a = self.genSparseCSRTensor((2, 2), 3, dtype=dtype, device=device, index_dtype=index_dtype)
            b = self.genSparseCSRTensor((2, 1), 2, dtype=dtype, device=device, index_dtype=index_dtype)
            # 使用断言检测运行时错误，并确保错误消息包含特定文本
            with self.assertRaisesRegex(RuntimeError, "Expected input tensors to have the same shape"):
                torch.add(a, b)

        # 对于指定的索引数据类型进行测试
        for index_dtype in [torch.int32, torch.int64]:
            run_test(index_dtype)

    # 跳过没有 MKL Sparse 支持的 CPU 平台上的测试
    @skipCPUIfNoMklSparse
    # 如果 cuSparse 的三角解算 API 不可用，则跳过 CUDA 平台上的测试
    @skipCUDAIf(
        not _check_cusparse_triangular_solve_available(),
        "cuSparse Generic API SpSV is not available"
    )
    # 使用装饰器指定多种数据类型和精度进行测试的函数
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    # 跳过 CUDA 测试，如果 cuSparse 的 SDDMM 泛型 API 不可用
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    # 定义测试方法，支持多种数据类型
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 定义精度覆盖，每种数据类型对应不同的精度要求
    @precisionOverride({torch.float32: 1e-3, torch.complex64: 1e-3,
                        torch.float64: 1e-8, torch.complex128: 1e-8})
    def test_sampled_addmm(self, device, dtype):
        # 定义测试函数，接收输入的稀疏张量和操作数
        def run_test(c, a, b, op_a, op_b, *, alpha=None, beta=None):
            # 如果数据类型是复数，设置随机复数 alpha 和 beta
            if dtype.is_complex:
                alpha = random.random() + 0.3j if alpha is None else alpha
                beta = random.random() + 0.6j if beta is None else beta
            else:
                alpha = random.random() if alpha is None else alpha
                beta = random.random() if beta is None else beta

            # 如果 op_a 为真且 a 和 b 的形状相同，取 a 的共轭转置
            if op_a and a.shape == b.shape:
                a = a.mH
            # 如果 op_b 为真且 a 和 b 的形状相同，取 b 的共轭转置
            if op_b and a.shape == b.shape:
                b = b.mH

            # 调用稀疏矩阵-稠密矩阵乘加函数，返回实际结果
            actual = torch.sparse.sampled_addmm(c, a, b, alpha=alpha, beta=beta)

            # 创建稀疏 CSR 张量 out，尺寸和 actual 相同
            out = torch.sparse_csr_tensor(
                *map(torch.clone, (actual.crow_indices(), actual.col_indices())),
                torch.empty_like(actual.values()),
                size=actual.shape
            )
            # 调用稀疏矩阵-稠密矩阵乘加函数，将结果存入 out 中
            torch.sparse.sampled_addmm(c, a, b, alpha=alpha, beta=beta, out=out)

            # 创建稀疏 CSR 张量 spy_c，尺寸和 c 相同，值为全 1
            spy_c = torch.sparse_csr_tensor(c.crow_indices(), c.col_indices(), torch.ones_like(c.values()), size=c.shape)
            # 期望结果计算，包括 alpha*(a@b)*spy_c + beta*c 的稠密表示
            expected = alpha * (a @ b) * spy_c.to_dense() + beta * c.to_dense()
            # 断言实际结果和期望结果相等
            self.assertEqual(actual.to_dense(), out.to_dense())
            self.assertEqual(actual.to_dense(), expected)

        # 定义所有可能的 (m, n, k) 组合列表
        mnk = list(itertools.product([2, 5], repeat=3))

        # 添加一个大小为 0 的 a 和 b 张量的测试用例
        mnk = mnk + [(5, 5, 0)]

        # 定义批量形状列表和布尔值列表
        batch_shapes = [(), (2,), (2, 3)]
        tf = [True, False]
        # 遍历索引数据类型列表，例如 torch.int32 和 torch.int64
        for index_dtype in [torch.int32, torch.int64]:
            # 遍历所有 (m, n, k) 组合、批量形状、非连续布尔值和广播 c 布尔值的可能组合
            for (m, n, k), b, noncontiguous, bcast_c in itertools.product(mnk, batch_shapes, tf, tf):
                # 如果 bcast_c 为真且 b 的长度为 0，则跳过当前组合
                if bcast_c and len(b) == 0:
                    continue
                # 随机生成非零元素个数 nnz
                nnz = random.randint(0, m * n)
                # 如果 bcast_c 为真，则 c_batch 为空元组，否则为 b
                c_batch = () if bcast_c else b
                # 生成稀疏 CSR 张量 c，随机填充
                c = self.genSparseCSRTensor((*c_batch, m, n), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
                # 生成形状为 (*b, m, k) 的稠密张量 a，随机填充
                a = make_tensor((*b, m, k), dtype=dtype, device=device, noncontiguous=noncontiguous)
                # 生成形状为 (*b, k, n) 的稠密张量 b，随机填充
                b = make_tensor((*b, k, n), dtype=dtype, device=device, noncontiguous=noncontiguous)
                # 对所有 op_a 和 op_b 组合运行测试函数
                for op_a, op_b in itertools.product([True, False], repeat=2):
                    run_test(c, a, b, op_a, op_b)

    # 跳过 CUDA 测试，如果 cuSparse 的 SDDMM 泛型 API 不可用
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    # 定义测试方法，支持多种数据类型
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 定义一个测试函数，用于测试稀疏矩阵乘法的自动求导功能
    def test_sampled_addmm_autograd(self, device, dtype):
        # 导入测试用例生成函数
        from torch.testing._internal.common_methods_invocations import sample_inputs_sparse_sampled_addmm

        # 生成测试样本
        samples = list(sample_inputs_sparse_sampled_addmm(None, device, dtype, requires_grad=True))

        # 遍历每个样本和稠密矢量的标志
        for sample, dense_covector in zip(samples, [True, False]):
            # 获取输入张量
            c = sample.input
            a = sample.args[0]
            b = sample.args[1]

            # 计算稀疏矩阵乘法的结果
            output = torch.sparse.sampled_addmm(c, a, b, **sample.kwargs)

            # 根据需要，生成稠密矢量并进行反向传播
            covector = torch.randn_like(output).to_dense() if dense_covector else torch.randn_like(output)
            output.backward(covector)

            # 计算稠密矩阵乘法的结果并与稀疏矩阵乘法的结果进行比较
            c1, a1, b1 = (x.detach().to_dense().requires_grad_(True) for x in [c, a, b])
            dense_output = sample.kwargs['alpha'] * (a1 @ b1) * torch.ones_like(c).to_dense() + sample.kwargs['beta'] * c1
            self.assertEqual(output, dense_output)

            # 对稠密结果进行反向传播，并断言梯度相等
            dense_covector = covector.to_dense()
            dense_output.backward(dense_covector)
            self.assertEqual(c.grad, c1.grad)
            self.assertEqual(a.grad, a1.grad)
            self.assertEqual(b.grad, b1.grad)

    # 仅在CUDA环境下执行该测试函数
    @onlyCUDA
    # 当前在ROCm上工作，而且存在CUDA问题
    @skipCUDAIf(not TEST_WITH_ROCM, "Causes CUDA memory exception, see https://github.com/pytorch/pytorch/issues/72177")
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    # 指定测试数据类型和精度覆盖
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_sampled_addmm_zero_sized(self, device, dtype):
        # 定义运行测试的内部函数
        def run_test(c, a, b):
            # 执行稀疏矩阵乘法
            actual = torch.sparse.sampled_addmm(c, a, b)
            # 断言输出形状与输入张量形状相同
            self.assertEqual(actual.shape, c.shape)

        # 遍历各种尺寸的零大小组合
        for m, n, k in itertools.product([0, 5], repeat=3):
            # 创建稀疏矩阵和稠密矩阵a、b
            c = torch.empty(m, n, dtype=dtype, device=device, layout=torch.sparse_csr)
            a = make_tensor((m, k), dtype=dtype, device=device)
            b = make_tensor((k, n), dtype=dtype, device=device)
            # 运行测试
            run_test(c, a, b)

    # 仅在CUDA环境下执行该测试函数
    @onlyCUDA
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    # 指定测试数据类型
    @dtypes(torch.float32, torch.float64, torch.complex64, torch.complex128)
    # 定义一个测试方法，用于测试稀疏矩阵乘法函数的错误情况
    def test_sampled_addmm_errors(self, device, dtype):
        # 测试稠密和稀疏采样版本的错误是否相同
        # import re

        # 确保矩阵乘法的形状兼容性
        a = make_tensor((2, 3), dtype=dtype, device=device)
        a_sparse = a.to_sparse_csr()
        # 测试稀疏矩阵乘法函数对不兼容形状的处理
        with self.assertRaisesRegex(RuntimeError, r"cannot be multiplied"):
            torch.sparse.sampled_addmm(a_sparse, a, a)

        # mat1 必须是一个矩阵
        with self.assertRaisesRegex(RuntimeError, r"Expected mat1 to be a matrix"):
            torch.sparse.sampled_addmm(a_sparse, a[..., 0, :], a)

        # mat2 必须是一个矩阵
        with self.assertRaisesRegex(RuntimeError, r"Expected mat2 to be a matrix"):
            torch.sparse.sampled_addmm(a_sparse, a, a[..., 0, :])

        a = make_tensor((2, 2), dtype=dtype, device=device)
        b = make_tensor((3, 3), dtype=dtype, device=device)
        b_sparse = b.to_sparse_csr()
        # 测试稀疏矩阵乘法函数对不兼容形状的处理
        with self.assertRaisesRegex(RuntimeError, r"self.shape\[-2\] must match mat1.shape\[-2\]"):
            torch.sparse.sampled_addmm(b_sparse, a, a)

        b = make_tensor((2, 3), dtype=dtype, device=device)
        b_sparse = b.to_sparse_csr()
        # 测试稀疏矩阵乘法函数对不兼容形状的处理
        with self.assertRaisesRegex(RuntimeError, r"self.shape\[-1\] must match mat2.shape\[-1\]"):
            torch.sparse.sampled_addmm(b_sparse, a, a)

        a = make_tensor((2, 2), dtype=dtype, device=device)
        a_sparse = a.to_sparse_csr()
        # 测试稀疏矩阵乘法函数对布局要求的处理
        with self.assertRaisesRegex(RuntimeError, r"Expected mat1 to have strided layout"):
            torch.sparse.sampled_addmm(a_sparse, a_sparse, a_sparse)

        with self.assertRaisesRegex(RuntimeError, r"Expected mat2 to have strided layout"):
            torch.sparse.sampled_addmm(a_sparse, a, a_sparse)

    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 0.01})
    # 定义测试方法，用于测试稀疏矩阵稠密乘积的求和操作
    def test_sparse_mm_reduce_sum(self, device, dtype):
        # 定义内部方法，运行单个测试用例
        def run_test(m, n, k, nnz, train):
            # 生成稀疏的 CSR 格式的张量
            sparse = self.genSparseCSRTensor((m, k), nnz, dtype=dtype, device=device, index_dtype=torch.int64)
            # 将稀疏张量转换为密集张量
            dense = sparse.to_dense()

            # 生成随机的 k x n 的张量
            mat = torch.randn(k, n, dtype=dtype)
            # 克隆 mat 作为参考的密集张量
            ref_mat = mat.clone()

            # 如果处于训练模式，设置梯度信息
            if train:
                sparse.requires_grad_()
                mat.requires_grad_()
                dense.requires_grad_()
                ref_mat.requires_grad_()

            # 计算参考输出，即密集矩阵与其自身的乘积
            ref_out = torch.mm(dense, ref_mat)
            # 计算稀疏矩阵与 mat 的乘积，并进行求和操作
            out = torch.sparse.mm(sparse, mat, 'sum')

            # 断言稀疏乘积的结果与参考输出相等
            self.assertEqual(out, ref_out)

            # 如果处于训练模式，计算梯度并进行断言
            if train:
                ref_out.sum().backward()
                out.sum().backward()

                # 获取稀疏矩阵的梯度并与参考的 dense 的梯度进行比较
                grad_input = sparse.grad
                ref_grad_input = dense.grad
                grad_mat = mat.grad
                ref_grad_mat = ref_mat.grad

                # 断言稀疏矩阵的梯度转换为 dense 形式与参考的 dense 梯度相等
                self.assertEqual(grad_input.to_dense(), ref_grad_input)
                # 断言 mat 的梯度与参考的 mat 梯度相等
                self.assertEqual(grad_mat, ref_grad_mat)

        # 运行两个测试用例
        run_test(4, 5, 4, 10, False)
        run_test(4, 4, 4, 16, True)

    # 使用装饰器设置跳过条件，仅在不满足 Torch Dynamo 条件时执行测试
    @skipIfTorchDynamo()
    # 使用装饰器设置仅在 CPU 上执行测试
    @onlyCPU
    # 使用装饰器设置测试数据类型为 float32, float64, bfloat16
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    # 使用装饰器设置 bfloat16 精度覆盖为 0.01
    @precisionOverride({torch.bfloat16: 0.01})
    # 定义一个测试方法，用于测试稀疏矩阵乘法的减少操作，接受设备和数据类型参数
    def test_sparse_mm_reduce(self, device, dtype):
        # 定义内部方法 run_test，用于执行单个测试用例
        def run_test(m, n, k, nnz, reduce_type, index_dtype, train):
            # 生成稀疏 CSR 格式的张量 csr
            csr = self.genSparseCSRTensor((m, n), nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            # 随机生成大小为 (n, k) 的张量 mat
            mat = torch.randn(n, k, dtype=dtype)
            # 克隆 mat 和 csr 的值，作为参考值
            ref_mat = mat.clone()
            ref_values = csr.values().clone()

            # 判断 index_dtype 是否为 torch.int32，以决定输出是否为 int32
            out_int32 = index_dtype == torch.int32
            # 将 csr 格式的行和列索引转换为 COO 格式的索引
            coo_indices = torch._convert_indices_from_csr_to_coo(
                csr.crow_indices(),
                csr.col_indices(),
                out_int32=out_int32)
            row, col = coo_indices[0], coo_indices[1]

            # 定义参考方法 ref，用于计算稀疏矩阵乘法的预期输出
            def ref(row, col, val, mat):
                # 创建一个全零的大小为 (m, k) 的张量 out
                out = torch.zeros([m, k], dtype=dtype)
                # 从 mat 中选择列索引对应的行，作为权重
                weight = mat.index_select(0, col)
                # 对权重进行乘法运算，并按照索引将结果累加到 out 中
                src = weight.mul(val.view(-1, 1))
                index = row.view(-1, 1).expand_as(weight)
                index = index.to(dtype=torch.int64)
                # 使用 scatter_reduce_ 方法在指定维度上进行聚合操作
                out.scatter_reduce_(0, index, src, reduce=reduce_type, include_self=False)
                return out

            # 如果处于训练模式，设置 csr、mat、ref_values 和 ref_mat 为需要梯度计算的状态
            if train:
                csr.requires_grad_()
                mat.requires_grad_()
                ref_values.requires_grad_()
                ref_mat.requires_grad_()

            # 调用 ref 方法计算稀疏矩阵乘法的预期输出 ref_out
            ref_out = ref(row, col, ref_values, ref_mat)
            # 调用 torch.sparse.mm 方法进行稀疏矩阵乘法计算，指定聚合类型 reduce_type
            out = torch.sparse.mm(csr, mat, reduce_type)
            # 使用 self.assertEqual 进行预期输出 ref_out 与实际输出 out 的比较
            self.assertEqual(out, ref_out)

            # 如果处于训练模式且数据类型不为 torch.bfloat16，执行梯度反向传播
            if train and dtype is not torch.bfloat16:
                ref_out.sum().backward()  # 计算预期输出的梯度
                out.sum().backward()  # 计算实际输出的梯度

                # 获取 csr 和 mat 的梯度值
                grad_values = csr.grad.values()
                grad_weight = mat.grad
                # 获取参考值 ref_values 和 ref_mat 的梯度值
                ref_grad_values = ref_values.grad
                ref_grad_weight = ref_mat.grad
                # 使用 self.assertEqual 进行梯度值的比较
                self.assertEqual(grad_values, ref_grad_values)
                self.assertEqual(grad_weight, ref_grad_weight)

        # 循环测试不同的训练标志、索引数据类型和聚合类型组合
        for train in [False, True]:
            for index_dtype in [torch.int32, torch.int64]:
                for reduce_type in ["sum", "mean", "amax", "amin"]:
                    # 调用 run_test 方法执行一系列测试用例
                    # 通过设置 nnz < M，创建空行
                    run_test(3, 4, 11, 1, reduce_type, index_dtype, train)
                    run_test(3, 4, 11, 6, reduce_type, index_dtype, train)
                    run_test(3, 4, 11, 12, reduce_type, index_dtype, train)
                    # 进行大块化测试，当 K > 4x 向量长度时需要测试
                    run_test(4, 7, 33, 13, reduce_type, index_dtype, train)

    # 标记为跳过元测试的装饰器，用于指示该测试方法不需要元测试框架运行
    @skipMeta
    # 使用 dtypes 装饰器指定测试方法支持的数据类型，包括所有标准类型和 torch.half、torch.bool、torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义一个测试方法，用于验证稀疏 CSR 到稠密张量的转换是否正确
    def test_coo_csr_conversion(self, device, dtype):
        # 遍历不同的大小组合
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            # 创建指定设备和数据类型的稠密张量
            dense = make_tensor(size, dtype=dtype, device=device)
            # 将稠密张量转换为 COO 格式的稀疏张量
            coo_sparse = dense.to_sparse()
            # 将 COO 格式的稀疏张量转换为 CSR 格式的稀疏张量
            csr_sparse = coo_sparse.to_sparse_csr()

            # 断言 CSR 格式的稀疏张量转换为稠密张量后与原稠密张量相等
            self.assertEqual(csr_sparse.to_dense(), dense)

    # 装饰器指定跳过元数据的测试方法，并且支持所有数据类型以及复合数据类型
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 定义一个测试方法，用于验证 CSR 到 COO 格式的稀疏张量转换是否正确
    def test_csr_coo_conversion(self, device, dtype):
        # 遍历不同的大小组合
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            # 创建指定设备和数据类型的稠密张量
            dense = make_tensor(size, dtype=dtype, device=device)
            # 将稠密张量转换为 CSR 格式的稀疏张量
            csr_sparse = dense.to_sparse_csr()
            # 将 CSR 格式的稀疏张量转换为 COO 格式的稀疏张量
            coo_sparse = csr_sparse.to_sparse()

            # 断言 COO 格式的稀疏张量转换为稠密张量后与原稠密张量相等
            self.assertEqual(coo_sparse.to_dense(), dense)

    # 当前，PyTorch 没有关于在稀疏 CSR 张量上填充零值输出的规则
    # 因此，只支持那些有 0->0 对应的运算符，例如：sin(0) = 0, tan(0) = 0，但是
    # cos(0) = 1（因此不支持）。
    # 注意：这里，我们仅对一元操作符进行测试
    @ops(sparse_csr_unary_ufuncs)
    # 定义一个测试方法，用于验证稀疏 CSR 张量上的一元操作符是否保持了 0->0 的对应关系
    def test_zero_to_zero_correspondence_unary(self, device, dtype, op):
        # 创建零张量，指定设备和数据类型
        zero = torch.zeros((1, 2), dtype=dtype, device=device)
        # 创建显式零值稀疏 CSR 张量，指定设备和数据类型
        tensor_explicit_zeros = torch.sparse_csr_tensor([0, 1], [1], [0], dtype=dtype, device=device)

        # 对零张量应用操作符，期望输出与零张量相同
        output_zero = op(zero)
        expected_zero = zero.to(output_zero.dtype)

        # 对显式零值稀疏 CSR 张量应用操作符，转换为稠密张量后期望输出与稀疏张量转换的稠密张量相同
        output_explicit_zeros = op(tensor_explicit_zeros).to_dense()
        expected_explicit_zeros = tensor_explicit_zeros.to_dense().to(output_explicit_zeros.dtype)

        # 断言结果是否符合预期，如果操作符破坏了 0->0 的对应关系，则报错
        for (output, expected) in [
                (output_zero, expected_zero),
                (output_explicit_zeros, expected_explicit_zeros)
        ]:
            self.assertEqual(output, expected, f"This operator ({op.name}) should not be supported for "
                             "Sparse CSR as it breaks 0->0 correspondence.")

        # 验证操作符是否能保持稀疏模式的一致性
        for inp in [zero.to_sparse_csr(), tensor_explicit_zeros]:
            self.assertEqual(op(inp).values().numel(), inp.values().numel(),
                             f"{op.name} fails to preserve sparsity pattern.")

    # 装饰器指定支持稀疏 CSR 张量的一元操作符
    @ops(sparse_csr_unary_ufuncs)
    # 定义一个测试方法，用于测试稀疏 CSR 格式的张量的一元操作，包括输出情况
    def test_sparse_csr_unary_out(self, device, dtype, op):
        # 获取操作对象的样本输入
        samples = op.sample_inputs(device, dtype)

        # 如果操作不支持输出参数，跳过测试
        if not op.supports_out:
            self.skipTest("Skipped! Out not supported")

        # 遍历每个样本输入
        for sample in samples:
            assert torch.is_tensor(sample.input)
            # 稀疏 CSR 格式仅支持二维张量作为输入
            # 提前失败以防止此测试悄悄成功
            if sample.input.ndim != 2:
                raise ValueError(f"Expected 2D tensor but got tensor with dimension: {sample.input.ndim}.")

            # 将输入张量转换为稀疏 CSR 格式
            sample.input = sample.input.to_sparse_csr()
            # 执行操作，获取期望输出
            expect = op(sample.input, *sample.args, **sample.kwargs)

            # 生成一个稀疏 CSR 格式的输出张量
            out = self.genSparseCSRTensor(sample.input.size(), sample.input._nnz(),
                                          device=sample.input.device, dtype=expect.dtype,
                                          index_dtype=sample.input.crow_indices().dtype)
            # 执行操作，将输出存储到预定义的 out 参数中
            op(sample.input, *sample.args, **sample.kwargs, out=out)

            # 断言生成的输出与期望的输出相等
            self.assertEqual(out, expect)

    # 用于测试稀疏 CSR 格式的张量的一元原位操作
    @ops(sparse_csr_unary_ufuncs)
    def test_sparse_csr_unary_inplace(self, device, dtype, op):
        # 获取操作对象的样本输入
        samples = op.sample_inputs(device, dtype)

        # 如果操作没有原位变体，跳过测试
        if op.inplace_variant is None:
            self.skipTest("Skipped! Inplace variant not supported!")

        # 遍历每个样本输入
        for sample in samples:
            assert torch.is_tensor(sample.input)
            # 稀疏 CSR 格式仅支持二维张量作为输入
            # 提前失败以防止此测试悄悄成功
            if sample.input.ndim != 2:
                raise ValueError(f"Expected 2D tensor but got tensor with dimension: {sample.input.ndim}.")

            # 将输入张量转换为稀疏 CSR 格式
            sample.input = sample.input.to_sparse_csr()
            # 执行操作，获取期望输出
            expect = op(sample.input, *sample.args, **sample.kwargs)

            # 如果期望的数据类型无法转换为当前数据类型，验证引发异常
            if not torch.can_cast(expect.dtype, dtype):
                with self.assertRaisesRegex(RuntimeError, "result type"):
                    op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
                continue

            # 如果输入张量为复数且操作为 "abs"，验证引发异常
            if sample.input.is_complex() and op.name == "abs":
                with self.assertRaisesRegex(RuntimeError, "not supported"):
                    op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
                continue

            # 执行原位操作，并获取实际的返回值
            actual = op.inplace_variant(sample.input, *sample.args, **sample.kwargs)

            # 断言实际返回的对象与原输入对象相同
            self.assertIs(actual, sample.input)
            # 断言实际输出与期望输出相等
            self.assertEqual(actual, expect)

    # 根据 TorchDynamo 的状态决定是否跳过测试
    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    # 标记用于稀疏 CSR 格式的张量的一元操作的测试方法
    @ops(sparse_csr_unary_ufuncs, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, torch.cdouble])
    # 定义一个测试方法，用于测试稀疏 CSR 格式的自动求导功能
    def test_autograd_sparse_csr_unary(self, device, dtype, op):
        # 如果操作不在允许使用自动求导的 CSR 格式的操作列表中，则跳过测试并给出相应消息
        if op.name not in UNARY_EWISE_CSR_ALLOW_AUTOGRAD:
            self.skipTest(f"Skipped! Unary op {op.name} not supported with CSR input and autograd")

        # 获取操作的样本输入数据
        samples = list(op.sample_inputs(device, dtype))

        # 检查样本中是否至少包含一个二维张量，否则引发异常
        ndims_equals_2d = (s.input.ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples.")

        for sample in samples:
            # 对于低维度的样本输入，跳过处理，因为无法转换为稀疏压缩的格式
            if sample.input.ndim < 2:
                continue
            
            # 将样本输入转换为稀疏 CSR 格式，并设置需要进行梯度计算
            sparse_input = sample.input.to_sparse_csr().requires_grad_(True)

            def fn(input):
                # 调用操作的梯度检查封装函数，计算输出梯度
                output = op.gradcheck_wrapper(op.get_op(), input, *sample.args, **sample.kwargs)
                # 如果定义了处理输出梯度的函数，则对输出梯度进行处理
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            # 计算稀疏输入的结果
            output = fn(sparse_input)
            covector = torch.randn_like(output)
            # 对稀疏输入进行反向传播
            output.backward(covector)
            # 断言稀疏输入的梯度是一个张量并且是稀疏 CSR 格式
            self.assertTrue(torch.is_tensor(sparse_input.grad))
            self.assertTrue(sparse_input.grad.is_sparse_csr)

            # 计算稠密输入的结果，并与稀疏结果进行比较
            dense_input = sparse_input.detach().to_dense().requires_grad_(True)
            dense_output = fn(dense_input)
            dense_covector = covector.to_dense()
            # 对稠密输入进行反向传播
            dense_output.backward(dense_covector)
            # 断言稀疏输入和稠密输入的梯度相等
            self.assertEqual(sparse_input.grad, dense_input.grad)

    # 如果 cuSparse 通用 API 中的 SDDMM 不可用，则跳过 CUDA 测试
    @skipCUDAIf(
        not _check_cusparse_sddmm_available(),
        "cuSparse Generic API SDDMM is not available"
    )
    # 设置数据类型为 float64 的装饰器
    @dtypes(torch.float64)
    # 如果没有 MKL Sparse 库，则跳过 CPU 测试
    @skipCPUIfNoMklSparse
    # 再次设置数据类型为 float64 的装饰器
    @dtypes(torch.float64)
    def test_autograd_dense_output_addmv(self, device, dtype):
        from torch.testing._internal.common_methods_invocations import sample_inputs_addmv

        samples = list(sample_inputs_addmv(None, device, dtype, requires_grad=True))

        # Fail early to prevent silent success with this test
        # 检查是否存在至少一个输入是二维张量，否则抛出异常
        ndims_equals_2d = (s.args[0].ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples to convert to sparse.")

        for sample in samples:
            # TODO: Remove detach once we have autograd support for CSR input
            # 将输入的稀疏张量转换为 CSR 格式并分离梯度信息
            a = sample.args[0].to_sparse_csr().detach()

            def fn(c, b):
                # 执行 addmv 操作，其中 c 是输出张量，a 是稀疏张量，b 是输入张量，使用额外的参数
                output = torch.addmv(c, a, b, **sample.kwargs)
                if sample.output_process_fn_grad is not None:
                    # 如果定义了梯度后处理函数，则对输出应用该函数
                    return sample.output_process_fn_grad(output)
                return output

            # 使用 gradcheck 验证梯度是否正确计算
            self.assertTrue(torch.autograd.gradcheck(fn, [sample.input, sample.args[1]], fast_mode=True))

            # noncontiguous
            # 创建非连续的张量 c 和 b 进行梯度检查
            c = make_tensor(sample.input.shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True)
            b = make_tensor(sample.args[1].shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(fn, [c, b], fast_mode=True))

    @skipIfTorchDynamo("Not a TorchDynamo suitable test")
    @ops(binary_ops_with_dense_output, dtypes=OpDTypes.supported, allowed_dtypes=[torch.double, ])
    def test_autograd_dense_output(self, device, dtype, op):
        if op.name == "mv" and no_mkl_sparse and self.device_type == 'cpu':
            # 在 CPU 上且没有 MKL Sparse 支持时，跳过测试
            self.skipTest("MKL Sparse is not available")

        samples = list(op.sample_inputs(device, dtype, requires_grad=True))

        # Fail early to prevent silent success with this test
        # 检查是否存在至少一个输入是二维张量，否则抛出异常
        ndims_equals_2d = (s.input.ndim == 2 for s in samples)
        if not any(ndims_equals_2d):
            raise ValueError("Expected at least one 2D tensor in samples.")

        # Here we assume that the signature is op(sparse_input, dense_input) -> dense_output
        for sample in samples:
            # TODO: Remove detach once we have autograd support for CSR input
            # 将输入的稀疏张量转换为 CSR 格式并分离梯度信息
            sparse_input = sample.input.to_sparse_csr().detach()

            def fn(*args):
                # 调用 op 的 gradcheck_wrapper 方法，验证梯度计算的正确性
                output = op.gradcheck_wrapper(op.get_op(), sparse_input, *args, **sample.kwargs)
                if sample.output_process_fn_grad is not None:
                    # 如果定义了梯度后处理函数，则对输出应用该函数
                    return sample.output_process_fn_grad(output)
                return output

            # 使用 gradcheck 验证梯度是否正确计算
            self.assertTrue(torch.autograd.gradcheck(fn, sample.args, fast_mode=True))

            # noncontiguous
            # 创建非连续的输入参数进行梯度检查
            args = [make_tensor(a.shape, device=device, dtype=dtype, noncontiguous=True, requires_grad=True) for a in sample.args]
            self.assertTrue(torch.autograd.gradcheck(fn, args, fast_mode=True))

    @dtypes(*all_types_and_complex())
    # 测试稀疏张量的直接 COO 到 CSR 格式转换的功能
    def test_direct_coo_csr_conversion(self, device, dtype):
        # 遍历不同的矩阵大小组合
        for m, n in itertools.product([5, 2, 0], [5, 2, 0]):
            size = (m, n)
            # 创建指定设备和数据类型的稠密张量
            dense = make_tensor(size, dtype=dtype, device=device)
            # 将稠密张量转换为 COO 格式的稀疏张量
            coo_sparse = dense.to_sparse_coo()

            # 断言 COO 转 CSR 再转回 COO 后与原始 COO 张量相等
            self.assertEqual(coo_sparse.to_sparse_csr().to_sparse_coo(), coo_sparse)

    # 跳过元数据测试，对多种数据类型执行测试
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sum(self, device, dtype):
        # 定义运行测试的函数，包括形状、非零元素数量和索引类型
        def run_test(shape, nnz, index_type):
            # 生成指定设备和数据类型的稀疏 CSR 张量
            a = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_type)
            # 断言稀疏张量的总和与其值的总和相等
            self.assertEqual(a.sum(), a.values().sum())
            # 如果数据类型是浮点类型，执行梯度计算测试
            if dtype in floating_types():
                a.requires_grad_(True)
                a.sum().backward()
                # 断言梯度与全一张量相等
                self.assertEqual(a.grad, torch.ones(shape, dtype=dtype, device=device))
        
        # 遍历不同形状和索引类型的组合
        for shape, index_dtype in itertools.product(
                [(10, 5), (10, 10)],
                [torch.int32, torch.int64]):
            run_test(shape, 0, index_dtype)
            run_test(shape, max(shape), index_dtype)
            run_test(shape, shape[0] * shape[1], index_dtype)

    # 跳过 Torch Dynamo 环境下的测试，跳过元数据测试，对多种数据类型执行测试
    @skipIfTorchDynamo()
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @all_sparse_compressed_layouts()
    # TODO: This is a stopgap for a rigorous extension of our autograd tests
    # to test the functionality of detach
    @skipMeta
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_exercise_detach(self, device, dtype):
        # 设置稀疏张量的形状和非零元素数量
        shape = (3, 3)
        nnz = 4
        # 遍历索引类型的组合
        for index_dtype in [torch.int32, torch.int64]:
            # 生成指定设备和数据类型的稀疏 CSR 张量
            inp = self.genSparseCSRTensor(shape, nnz, dtype=dtype, device=device, index_dtype=index_dtype)
            # 分离输入张量的梯度信息
            detached_inp = inp.detach()
            # 断言输入张量与分离后的张量相等
            self.assertEqual(inp, detached_inp)
    # 定义一个私有方法，用于构建稀疏矩阵
    def _construct_sp_matrix(self, tensor, layout, blocksize=(2, 2)):
        # 检查输入的张量是否为稀疏格式之一，如果不是则抛出未实现的错误
        if tensor.layout in [torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.strided]:
            tensor = tensor.to_dense()
        else:
            raise NotImplementedError(repr(tensor))
        
        # 根据布局选择相应的稀疏矩阵类型并返回
        if layout is torch.sparse_csr:
            return sp.csr_matrix(tensor.cpu().numpy())
        if layout is torch.sparse_csc:
            return sp.csc_matrix(tensor.cpu().numpy())
        if layout is torch.sparse_bsr:
            # 使用块大小创建并返回BSR格式的稀疏矩阵，并排序索引
            return sp.bsr_matrix(tensor.cpu().numpy(), blocksize=blocksize).sorted_indices()
        if layout is torch.sparse_bsc:
            # SciPy不直接支持BSC格式，通过伪装使用转置后的BSR矩阵来模拟
            class FakeBscMatrix:
                def __init__(self, matrix):
                    self._matrix = matrix
                    self.shape = tuple(reversed(matrix.shape))
                    self.indptr = matrix.indptr
                    self.indices = matrix.indices
                    self.data = [x.transpose() for x in matrix.data]

                @staticmethod
                def from_matrix(matrix, blocksize):
                    blocksize = tuple(reversed(blocksize))
                    matrix = matrix.transpose()
                    return FakeBscMatrix(sp.bsr_matrix(matrix, blocksize=blocksize))

                def sorted_indices(self):
                    sub = self._matrix.sorted_indices()
                    return FakeBscMatrix(sub)

            # 使用给定的张量和块大小创建FakeBscMatrix，并返回排序后的索引
            return FakeBscMatrix.from_matrix(tensor.cpu().numpy(), blocksize=blocksize).sorted_indices()
        
        # 如果布局不在支持的稀疏格式之内，则抛出未实现的错误
        raise NotImplementedError(repr(tensor))
    # 定义测试函数，用于测试稀疏矩阵在不同布局下的压缩和非压缩格式之间的转换，同时与SciPy实现进行比较
    def test_sparse_to_sparse_compressed(self, device, dtype, coalesced, layout):
        """
        This test tests conversion from COO to CSR and CSC and CSC to CSR and CSC
        by comparing to SciPy's implementation.

        Here we test only those conversion combinations that SciPy
        supports to ensure that PyTorch conversions are in the same
        page with SciPy.  Independent from SciPy, all conversion
        combinations are tested in TestSparseAny.test_to_sparse.
        """

        blocksize_kw = {}
        # 如果布局为 torch.sparse_bsc 或 torch.sparse_bsr，则设置块大小为 (2, 2)
        if layout in (torch.sparse_bsc, torch.sparse_bsr):
            blocksize_kw['blocksize'] = (2, 2)
            # 块模式不支持宽度或高度为0的情况
            shapes = [(6, 10)]
        # 如果布局为 torch.sparse_csc 或 torch.sparse_csr，则设置不同形状的测试用例
        elif layout in (torch.sparse_csc, torch.sparse_csr):
            shapes = [(0, 10), (6, 0), (6, 10), (0, 0)]
        else:
            # 抛出未实现错误，处理未知的布局类型
            raise NotImplementedError("unhandled layout")

        # 根据布局类型选择压缩索引和普通索引的方法
        if layout in (torch.sparse_bsc, torch.sparse_csc):
            compressed_indices_mth = torch.Tensor.ccol_indices
            plain_indices_mth = torch.Tensor.row_indices
        elif layout in (torch.sparse_bsr, torch.sparse_csr):
            compressed_indices_mth = torch.Tensor.crow_indices
            plain_indices_mth = torch.Tensor.col_indices
        else:
            raise NotImplementedError("unhandled layout")

        # 遍历不同形状的稀疏矩阵测试用例
        for shape in shapes:
            sparse_dim = 2
            nnz = shape[0] * shape[1] // 2
            # 生成稀疏张量用于测试
            sparse, _, _ = self.genSparseTensor(shape, sparse_dim, nnz, coalesced, device, dtype)
            # 根据给定布局构造稀疏矩阵
            sp_matrix = self._construct_sp_matrix(sparse, layout)
            # 将稀疏张量转换为指定布局的稀疏矩阵，传入块大小参数
            pt_matrix = sparse.to_sparse(layout=layout, **blocksize_kw)

            # 断言检查：布局类型相同
            self.assertEqual(layout, pt_matrix.layout)
            # 断言检查：形状相同
            self.assertEqual(sp_matrix.shape, pt_matrix.shape)
            # 断言检查：压缩索引与预期相符
            self.assertEqual(torch.tensor(sp_matrix.indptr, dtype=torch.int64), compressed_indices_mth(pt_matrix))
            # 断言检查：普通索引与预期相符
            self.assertEqual(torch.tensor(sp_matrix.indices, dtype=torch.int64), plain_indices_mth(pt_matrix))
            # 断言检查：数值与预期相符
            self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values())

            # 将稀疏张量转换为 CSC 格式，用于进一步测试
            sparse_csc = sparse.to_sparse_csc()
            sp_matrix = self._construct_sp_matrix(sparse_csc, layout)
            # 将稀疏张量转换为指定布局的稀疏矩阵，传入块大小参数
            pt_matrix = sparse_csc.to_sparse(layout=layout, **blocksize_kw)

            # 断言检查：布局类型相同
            self.assertEqual(layout, pt_matrix.layout)
            # 断言检查：形状相同
            self.assertEqual(sp_matrix.shape, pt_matrix.shape)
            # 断言检查：压缩索引与预期相符
            self.assertEqual(torch.tensor(sp_matrix.indptr, dtype=torch.int64), compressed_indices_mth(pt_matrix))
            # 断言检查：普通索引与预期相符
            self.assertEqual(torch.tensor(sp_matrix.indices, dtype=torch.int64), plain_indices_mth(pt_matrix))
            # 断言检查：数值与预期相符
            self.assertEqual(torch.tensor(sp_matrix.data), pt_matrix.values())
# 如果 Triton 模块可用，则无操作，直接返回原类
def skipIfNoTriton(cls):
    from torch.utils._triton import has_triton

    # 检查是否已加载 Triton 模块
    if has_triton():
        return cls
    else:
        # Triton 模块不可用时，创建一个新的类 skipped_cls，继承自原始的测试类 cls
        @functools.wraps(cls, updated=())
        class skipped_cls(cls):
            # 在 setUp 方法中跳过测试，抛出跳过测试的异常
            def setUp(self):
                self.skipTest("Triton is not available.")

        # 返回新创建的 skipped_cls 类，用于替代原始的测试类 cls
        return skipped_cls

# 使用 skipIfNoTriton 装饰器，如果 Triton 不可用，则跳过测试类
@skipIfNoTriton
class TestSparseCompressedTritonKernels(TestCase):

    # 将输入张量 d 修改为原地 (upper/lower) block-triangular 格式的函数
    def _to_block_triangular_inplace(self, d, row_block, col_block):
        """
        This function modifies `d` to become (upper/lower) block-triangular in-place.
        It is assumed that `d.shape[-2]` is divisible by `row_block` and
        `d.shape[-1]` is divisible by `col_block`.
        """

        from torch.sparse._triton_ops import tile_to_blocksize

        m, n = d.shape[-2:]
        # 使用 Triton 提供的 tile_to_blocksize 函数将 d 转换为指定块大小的张量
        d_tiled = tile_to_blocksize(d, (row_block, col_block))
        d_tiled = d_tiled.moveaxis(-4, -1).moveaxis(-4, -1)
        # 根据 m 和 n 的大小关系选择操作，将 d_tiled 转换为 block-triangular 格式
        if m // row_block > n // col_block:
            d_tiled.tril_()
        else:
            d_tiled.triu_()

        return d

    # 在 CUDA 环境下执行的测试方法，测试 Triton 加速的 BSR softmax 实现
    @onlyCUDA
    # 在 ROCm 环境下跳过测试，因为测试在 ROCm 上运行过慢
    @skipIfRocm(msg="test is too slow on ROCm stack")
    # 指定测试方法支持的数据类型为 torch.half、torch.bfloat16、torch.float
    @dtypes(torch.half, torch.bfloat16, torch.float)
    # 如果在 CUDA 环境下，根据硬件支持情况选择数据类型
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    # 如果在 FBCode 并且是远程 GPU 上运行，则跳过测试，要求 Triton 模块可用
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    # 测试 Triton 加速的 BSR softmax 实现
    def test_triton_bsr_softmax(self, device, dtype):
        from functools import partial
        from torch.sparse._triton_ops import bsr_softmax

        # 创建一个 partial 函数 tensor，用于生成指定设备和数据类型的张量
        tensor = partial(make_tensor, device=device, dtype=dtype, low=1.0, high=3.0)

        # 不支持带有零尺寸的 batch 维度在 to_sparse_bsr 中的测试提示
        batches = [(), (2,), (2, 2)]
        size = [6, 12, 0]
        block_size = [2, 3]

        # 对于给定的 block_size、batches 和 size 组合，执行一般正确性测试
        for row_block, col_block, b, m, n in itertools.product(block_size, block_size, batches, size, size):
            # 生成指定尺寸的输入张量
            input = tensor(b + (m, n))
            # 设置对角线元素为 m * n
            input.diagonal(dim1=-2, dim2=-1).fill_(m * n)
            # 使用 _to_block_triangular_inplace 方法将 input 转换为 block-triangular 格式
            input = self._to_block_triangular_inplace(input, row_block, col_block)

            # 将 input 转换为 BSR 格式的稀疏张量
            bsr = input.to_sparse_bsr((row_block, col_block))
            # 将 input 转换为 COO 格式的稀疏张量，并转换为 torch.float 类型
            coo = input.to_sparse().to(torch.float)

            # 使用 Triton 加速的 bsr_softmax 函数对 bsr 进行 softmax 计算
            res_tri = bsr_softmax(bsr)
            # 使用 torch 自带的 sparse.softmax 对 coo 进行 softmax 计算
            res_coo = torch.sparse.softmax(coo, -1)
            # 断言 Triton 加速的结果与 torch 自带的结果一致
            self.assertEqual(res_tri, res_coo.to(input.dtype))

        # 测试超出 Triton 最大元素数限制（2 ** 17）的长行
        input = tensor(b + (1, 150000))
        # 将 input 转换为 BSR 格式的稀疏张量，并使用 softmax 计算
        bsr = input.to_sparse_bsr(1)
        self.assertEqual(input.softmax(-1), bsr_softmax(bsr))

    # 使用 parametrize 装饰器，为 block_size 和 index_dtype 参数进行参数化测试
    @parametrize("block_size", [16, 32, 64])
    @parametrize("index_dtype", [torch.int32, torch.int64])
    # 仅在 CUDA 环境下执行测试
    @onlyCUDA
    # 指定测试方法支持的数据类型为 torch.half、torch.bfloat16、torch.float
    @dtypes(torch.half, torch.bfloat16, torch.float)
    # 如果在 CUDA 环境下，根据硬件支持情况选择数据类型
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    # 标记为单元测试函数，根据条件跳过部分情形（不使用 TorchInductor、FBCODE且远程 GPU、或者运行于部署环境）
    @unittest.skipIf((not TEST_WITH_TORCHINDUCTOR) or (IS_FBCODE and IS_REMOTE_GPU) or torch._running_with_deploy(),
                     "Skipped for deploy and internal with remote GPUs")
    # 仅在 CUDA 下运行测试
    @onlyCUDA
    # 设置数据类型为 torch.half
    @dtypes(torch.half)
    # 根据条件跳过测试（FBCODE且远程 GPU或者运行于部署环境）
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU or torch._running_with_deploy(),
                     "Skipped for deploy and internal with remote GPUs")
    # 定义测试函数，测试 Triton 的 BSR 稠密乘法的错误消息处理
    def test_triton_bsr_dense_bmm_error_messages(self, device, dtype):
        # 导入 Triton 的 BSR 稠密乘法操作
        from torch.sparse._triton_ops import bsr_dense_mm

        # 创建随机张量作为右手边的稠密矩阵，并指定设备和数据类型
        rhs = torch.rand(32, 32, dtype=dtype, device=device)
        # 将右手边的稠密矩阵转换为 BSR 格式
        lhs = rhs.to_sparse_bsr(16)
        
        # 检查异常是否抛出，期望抛出 "only BSR sparse format is supported"
        with self.assertRaisesRegex(ValueError, "only BSR sparse format is supported"):
            bsr_dense_mm(lhs.to_sparse_bsc(16), rhs)
        
        # 检查异常是否抛出，期望抛出 "on the same GPU device"
        with self.assertRaisesRegex(ValueError, "on the same GPU device"):
            bsr_dense_mm(lhs, rhs.cpu())
        
        # 如果有多个 CUDA 设备，检查异常是否抛出，期望抛出 "on the same GPU device"
        if torch.cuda.device_count() > 1:
            with self.assertRaisesRegex(ValueError, "on the same GPU device"):
                bsr_dense_mm(lhs.to("cuda:0"), rhs.to("cuda:1"))
        
        # 检查异常是否抛出，期望抛出 "all inputs are expected to be of the same dtype"
        with self.assertRaisesRegex(ValueError, "all inputs are expected to be of the same dtype"):
            bsr_dense_mm(lhs, rhs.to(torch.float))
        
        # 检查异常是否抛出，期望抛出 "and one of (half, bfloat16, float32)"
        with self.assertRaisesRegex(ValueError, r"and one of \(half, bfloat16, float32\)"):
            bsr_dense_mm(lhs.to(torch.double), rhs.to(torch.double))
        
        # 检查异常是否抛出，期望抛出 "all inputs involved in the matrix product are expected to be at least 2D"
        with self.assertRaisesRegex(ValueError, "all inputs involved in the matrix product are expected to be at least 2D"):
            bsr_dense_mm(lhs, torch.rand(1, dtype=dtype, device=device))
        
        # 检查异常是否抛出，期望抛出 "sizes involved in the matrix product are not compatible for matrix multiplication"
        with self.assertRaisesRegex(ValueError, "sizes involved in the matrix product are not compatible for matrix multiplication"):
            bsr_dense_mm(lhs, torch.rand(1, 1, dtype=dtype, device=device))
        
        # 检查异常是否抛出，期望抛出 "dense.size(-1) == 15 should be divisible by 16"
        with self.assertRaisesRegex(ValueError, r"dense.size\(-1\) == 15 should be divisible by 16"):
            bsr_dense_mm(lhs, torch.rand(32, 15, dtype=dtype, device=device))
        
        # 检查块大小是否满足条件
        for blocksize in (15, 30):
            n = blocksize * 2
            rhs = torch.rand(n, n, dtype=dtype, device=device)
            lhs = rhs.to_sparse_bsr(blocksize)
            # 检查异常是否抛出，期望抛出 "should be at least 16 and a power of 2"
            with self.assertRaisesRegex(ValueError, "should be at least 16 and a power of 2"):
                bsr_dense_mm(lhs, rhs)
        
        # 检查输出张量的形状是否正确
        rhs = torch.rand(2, 32, 32, dtype=dtype, device=device)
        lhs = rhs.to_sparse_bsr(16)
        # 检查异常是否抛出，期望抛出 "`out` argument has wrong shape"
        with self.assertRaisesRegex(ValueError, r"`out` argument has wrong shape"):
            out = torch.rand(2, 30, 30, dtype=dtype, device=device)
            bsr_dense_mm(lhs, rhs, out=out)
        
        # 检查异常是否抛出，期望抛出 "only row-major/col-major `out`"
        with self.assertRaisesRegex(ValueError, r"only row-major/col-major `out`"):
            out = torch.rand(32, 32, 2, dtype=dtype, device=device).transpose(0, -1)
            bsr_dense_mm(lhs, rhs, out=out)
    @dtypes(torch.half, torch.bfloat16, torch.float)
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    @precisionOverride({torch.float16: 1e-3})


    # 定义测试函数，用于测试 Triton 的 scaled dot product attention
    def test_triton_scaled_dot_product_attention(self, device, dtype, block_size):
        from functools import partial
        from torch.sparse._triton_ops import _scaled_dot_product_attention

        # 部分函数，用于生成特定设备和数据类型的张量
        tensor = partial(make_tensor, device=device, dtype=dtype, low=0.3, high=1.2)

        def broadcast_input(*ts):
            # 计算批次维度，确保张量形状一致
            batch_dims = torch.broadcast_shapes(*(t.shape[:-2] for t in ts))
            yield from (torch.broadcast_to(t, batch_dims + t.shape[-2:]) for t in ts)

        # NOTE: batch dims with zero sizes are not supported in `to_sparse_bsr`.
        # 定义不同的批次大小和尺寸
        batches = [(), (2,), (2, 2)]
        size = [128, 256, 0]

        # 使用 itertools 生成各种批次组合
        for bam, bq, bk, bv, m, n, k in itertools.product(batches, batches, batches, batches, size, size, size):
            # 生成查询、键和值张量
            query = tensor(bq + (m, k))
            key = tensor(bk + (n, k))
            value = tensor(bv + (n, k))

            # 创建注意力掩码，并转换为稀疏 BSR 格式
            attn_mask = torch.ones(bam + (m, n), device=device, dtype=torch.bool)
            attn_mask = self._to_block_triangular_inplace(attn_mask, block_size, block_size)
            attn_mask_bsr = attn_mask.to_sparse_bsr(block_size)

            # NOTE: only boolean mask is directly compatible with the Strided version
            # without any pre-/post-processing. Hence we test against a boolean mask.
            # 测试不同的缩放因子和掩码数据类型
            for scale in (None, 1. / 16):
                if scale is None and query.size(-1) == 0:
                    scale = 1
                expected = torch.nn.functional.scaled_dot_product_attention(
                    *broadcast_input(query, key, value, attn_mask), scale=scale
                )

                for mask_dtype in (torch.bool, dtype):
                    # 调用 Triton 实现的 scaled dot product attention 函数
                    res = _scaled_dot_product_attention(query, key, value, attn_mask_bsr.to(mask_dtype), scale=scale)
                    # 断言结果与预期相等
                    self.assertEqual(res, expected)


    @parametrize("block_size", [16, 32, 64])
    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.half, torch.bfloat16, torch.float)
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    @onlyCUDA
    @skipIfRocm
    @dtypes(torch.half, torch.bfloat16, torch.float)
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    # 定义一个测试方法，用于测试 Triton 的 scatter_mm 函数
    def test_triton_scatter_mm(self, device, dtype):
        # 导入 Triton 的 scatter_mm 函数
        from torch.sparse._triton_ops import scatter_mm
        # 导入 functools 库的 partial 函数，用于创建部分应用的 make_tensor 函数
        from functools import partial
        # 创建 tensor 函数的部分应用，指定设备和数据类型，并设置数据范围
        tensor = partial(make_tensor, device=device, dtype=dtype, low=0.5, high=1.5)
        
        # 定义测试尺寸
        sizes = [8, 16]
        # 使用 itertools 的 product 函数，生成 m, k, n 的所有组合
        for m, k, n in itertools.product(sizes, sizes, sizes):
            # 创建两个大小为 m x k 的张量块，堆叠成 blocks 张量
            blocks = torch.stack([tensor(m, k), tensor(m, k)])
            # 创建两个大小为 k x n 的张量块，堆叠成 others 张量
            others = torch.stack([tensor(k, n), tensor(k, n)])

            # 计算期望的结果张量，使用矩阵乘法操作
            expected = torch.stack([blocks[0] @ others[0] + blocks[1] @ others[0],
                                    blocks[0] @ others[1],
                                    blocks[1] @ others[1]])

            # 定义稀疏矩阵乘法的索引数据
            indices_data = (
                'scatter_mm',  # 索引数据的类型标识符
                torch.tensor([0, 2, 3, 4], dtype=torch.int32, device=device),  # 行偏移数组
                torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.int32, device=device)  # 列偏移数组
            )

            # 调用 scatter_mm 函数进行计算
            result = scatter_mm(blocks, others, indices_data=indices_data)

            # 使用 unittest 的 assertEqual 方法断言计算结果与期望结果相等
            self.assertEqual(result, expected)

            # 定义另一种稀疏矩阵乘法的索引数据
            indices_data = (
                'bsr_strided_mm',  # 索引数据的类型标识符
                torch.tensor([0, 2, 4, 5, 6], dtype=torch.int32, device=device),  # 行偏移数组
                torch.tensor([0, n, 2 * n * m, 2 * n * m + n], dtype=torch.int32, device=device),  # 列偏移数组
                torch.tensor([1, 0, 1, 0, 1, 1], dtype=torch.int32, device=device),  # 块行偏移数组
                torch.tensor([0, 2 * k * n, n, 2 * k * n + n, 2 * k * n, 2 * k * n + n],  # 块列偏移数组
                             dtype=torch.int32, device=device),
                dict(SPLIT_N=2, is_compressed=False, TILE_M=m, TILE_N=n, GROUP_SIZE=1)  # 其他参数字典
            )

            # 对于不同的块大小进行循环测试
            for bsize in [(), (2,), (3, 4)]:
                # 创建指定大小的张量 other
                other = tensor(*bsize, 2 * k, 2 * n)
                # 计算期望的结果张量，使用矩阵乘法操作
                expected = torch.cat([
                    torch.cat([blocks[1], blocks[0]], dim=1),
                    torch.cat([torch.zeros_like(blocks[0]), blocks[1]], dim=1)], dim=0) @ other
                # 调用 scatter_mm 函数进行计算
                result = scatter_mm(blocks, other, indices_data=indices_data)
                # 使用 unittest 的 assertEqual 方法断言计算结果与期望结果相等
                self.assertEqual(result, expected)
    
    # 对 test_triton_scatter_mm 方法进行参数化测试，blocksize 参数为列表中的各个值
    @parametrize("blocksize", [2, '2x3', 16, '16x32', 32, 64])
    # 仅在 CUDA 设备上执行该测试
    @onlyCUDA
    # 设置测试使用的数据类型为 torch.half, torch.bfloat16, torch.float
    @dtypes(torch.half, torch.bfloat16, torch.float)
    # 如果在 CUDA 设备上，根据条件添加额外的数据类型 torch.bfloat16
    @dtypesIfCUDA(torch.half, *[torch.bfloat16] if SM80OrLater else [], torch.float)
    # 如果运行环境为 FBCODE 并且使用远程 GPU，则跳过该测试
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    # 定义一个测试函数，用于测试 Triton 的 BSR 稀疏矩阵乘法
    def test_triton_bsr_scatter_mm(self, device, dtype, blocksize):
        # 导入 Triton 相关模块
        import triton
        from torch.sparse._triton_ops import bsr_scatter_mm, bsr_scatter_mm_indices_data
        from functools import partial
        
        # 如果 blocksize 是字符串，则将其解析为元组
        if isinstance(blocksize, str):
            blocksize = tuple(map(int, blocksize.split('x')))
        else:
            blocksize = (blocksize,) * 2
        
        # 定义一个生成张量的部分函数，设定设备、数据类型、数值范围
        tensor = partial(make_tensor, device=device, dtype=dtype, low=0.5, high=1.5)

        # 定义不同的批次维度和大小
        batches = [(), (2,), (2, 2)]
        sizes = [blocksize[0], 2 * blocksize[0], 4 * blocksize[0]]
        sizes_K = [blocksize[1], 2 * blocksize[1]]

        # 使用 itertools 生成各种批次组合的迭代器
        for bd, bs, M, K, N, has_zero_row_block in itertools.product(batches, batches[:1], sizes, sizes_K, sizes, (False, True)):
            # 生成 BSR 稠密矩阵张量
            bsr_dense = tensor(bs + (M, K))
            
            # 如果设置了零行块标志并且 M 大于块大小的第一个维度
            if has_zero_row_block:
                if M > blocksize[0]:
                    bsr_dense[:blocksize[0]].zero_()
                else:
                    continue
            
            # 将稠密矩阵转换为 BSR 稀疏矩阵
            bsr = bsr_dense.to_sparse_bsr(blocksize)
            
            # 生成稠密矩阵张量
            dense = tensor(bd + (K, N))
            
            # 计算期望的结果
            expected = bsr.to_dense() @ dense

            # 遍历不同的索引格式
            for indices_format in ('bsr_strided_mm', 'bsr_strided_mm_compressed', 'scatter_mm'):
                # 对于 bsr_strided_mm 和 bsr_strided_mm_compressed，生成不同的分割尺寸列表
                if indices_format in {'bsr_strided_mm', 'bsr_strided_mm_compressed'}:
                    SPLIT_N_list = [N]
                    while SPLIT_N_list[-1] > 1:
                        SPLIT_N_list.append(max(1, SPLIT_N_list[-1] // 2))
                else:
                    SPLIT_N_list = [1]
                
                # 遍历分割尺寸列表
                for SPLIT_N in SPLIT_N_list:
                    # 生成索引数据
                    indices_data = bsr_scatter_mm_indices_data(
                        bsr, dense, indices_format=indices_format, SPLIT_N=SPLIT_N)
                    
                    try:
                        # 调用 Triton 的稀疏矩阵乘法函数
                        result = bsr_scatter_mm(bsr, dense, indices_data=indices_data)
                    except triton.compiler.OutOfResources:
                        # 如果超出资源，则断言 SPLIT_N 小于列表中的第一个 SPLIT_N
                        assert SPLIT_N < SPLIT_N_list[0]
                        break
                    
                    # 断言结果与期望值相等
                    self.assertEqual(result, expected)
        
        # 清除 Triton 的缓存
        torch.sparse._triton_ops._bsr_scatter_mm_indices_data.cache_clear()
    # 使用unittest.skipIf装饰器，如果IS_FBCODE为真且IS_REMOTE_GPU为真，则跳过此测试（需要Triton）
    @unittest.skipIf(IS_FBCODE and IS_REMOTE_GPU, "Test requires Triton")
    # 定义测试函数test_triton_tune，接受参数op、device、dtype
    def test_triton_tune(self, op, device, dtype):
        # 导入Triton操作bsr_dense_addmm和相关元数据操作函数
        from torch.sparse._triton_ops import bsr_dense_addmm
        from torch.sparse._triton_ops_meta import (create_blocked_tensor, tune_bsr_dense_addmm, get_meta)

        # 从字典中根据op选择对应的操作函数和调谐器函数
        operation = dict(bsr_dense_addmm=bsr_dense_addmm)[op]
        tuner = dict(bsr_dense_addmm=tune_bsr_dense_addmm)[op]

        # 定义矩阵维度和稀疏度
        M, K, N = 16, 16, 32
        sparsity = 1.0
        blocksize = (16, 16)
        # 创建稀疏块张量BSR，使用create_blocked_tensor函数生成，然后转换为稀疏BSR格式
        bsr = create_blocked_tensor(0, M, K, blocksize, sparsity, dtype, device).to_sparse_bsr(blocksize)
        # 计算实际稀疏度
        sparsity = 1 - bsr._nnz() * blocksize[0] * blocksize[1] / (M * K)
        # 生成输入张量和密集张量
        input = make_tensor(K, N, dtype=dtype, device=device)
        dense = make_tensor(K, N, dtype=dtype, device=device)

        # 如果op为'bsr_dense_addmm'，则将参数设定为(input, bsr, dense)
        if op == 'bsr_dense_addmm':
            args = (input, bsr, dense)

            # 定义获取当前元数据的函数get_current_meta
            def get_current_meta():
                version = (0, dtype, sparsity)
                meta_key = (M, K, N, *blocksize, False, True, True)
                return get_meta(op, meta_key, version=version, exact=True)
        else:
            # 如果op不是'bsr_dense_addmm'，则抛出NotImplementedError异常
            raise NotImplementedError(op)

        # 断言当前元数据为None
        self.assertEqual(get_current_meta(), None)

        # 调用调谐器函数获取元数据meta，并断言当前元数据等于meta
        meta = tuner(*args, **dict(store=True, verbose=False))
        self.assertEqual(get_current_meta(), meta)

        # 执行操作函数operation，预期得到expected结果
        expected = operation(*args)
        # 使用元数据meta执行操作函数operation，得到结果result
        result = operation(*args, **dict(meta=meta))
        # 断言结果result等于预期结果expected
        self.assertEqual(result, expected)
# 调用函数实例化特定类型的测试用例，针对不同设备类型（CPU和CUDA）
instantiate_device_type_tests(TestSparseCSR, globals())
# 调用函数实例化特定类型的测试用例，针对不同设备类型（CPU和CUDA）
instantiate_device_type_tests(TestSparseCompressed, globals())
# 调用函数实例化特定类型的测试用例，针对不同设备类型（CPU和CUDA）
instantiate_device_type_tests(TestSparseCompressedTritonKernels, globals())

# 检查脚本是否作为主程序运行
if __name__ == '__main__':
    # 运行所有的测试用例
    run_tests()
```