# `.\pytorch\test\test_sparse.py`

```py
# Owner(s): ["module: sparse"]

import torch  # 导入 PyTorch 库
import itertools  # 导入 itertools 库，用于迭代操作
import functools  # 导入 functools 库，用于高阶函数操作
import operator  # 导入 operator 模块，提供了一系列对应 Python 操作符的函数
import random  # 导入 random 模块，用于生成随机数
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from torch.testing import make_tensor  # 从 torch.testing 导入 make_tensor 函数
from torch.testing._internal.common_utils import (
    TestCase,  # 导入 TestCase 类，用于编写测试用例
    run_tests,  # 导入 run_tests 函数，用于运行测试
    skipIfRocm,  # 导入 skipIfRocm 装饰器，用于在 ROCm 环境下跳过测试
    do_test_dtypes,  # 导入 do_test_dtypes 函数，用于测试数据类型
    load_tests,  # 导入 load_tests 函数，用于加载测试
    TEST_NUMPY,  # 导入 TEST_NUMPY 常量，用于检查是否测试 NumPy 相关功能
    TEST_SCIPY,  # 导入 TEST_SCIPY 常量，用于检查是否测试 SciPy 相关功能
    IS_WINDOWS,  # 导入 IS_WINDOWS 常量，用于检查是否在 Windows 环境下
    gradcheck,  # 导入 gradcheck 函数，用于梯度检查
    coalescedonoff,  # 导入 coalescedonoff 常量，用于测试索引是否压缩
    DeterministicGuard,  # 导入 DeterministicGuard 类，用于测试时设置随机数种子
    first_sample,  # 导入 first_sample 函数，用于选择第一个样本
    TEST_WITH_CROSSREF,  # 导入 TEST_WITH_CROSSREF 常量，用于检查是否测试交叉引用
    TEST_WITH_ROCM,  # 导入 TEST_WITH_ROCM 常量，用于检查是否在 ROCm 环境下测试
    skipIfTorchDynamo,  # 导入 skipIfTorchDynamo 装饰器，用于在 Torch Dynamo 下跳过测试
    parametrize,  # 导入 parametrize 装饰器，用于参数化测试
    subtest,  # 导入 subtest 函数，用于运行子测试
    is_coalesced_indices,  # 导入 is_coalesced_indices 函数，用于检查索引是否压缩
    suppress_warnings,  # 导入 suppress_warnings 函数，用于抑制警告
    instantiate_parametrized_tests,  # 导入 instantiate_parametrized_tests 函数，用于实例化参数化测试
    skipIfCrossRef  # 导入 skipIfCrossRef 装饰器，用于在跨引用情况下跳过测试
)
from torch.testing._internal.common_cuda import TEST_CUDA  # 从 torch.testing._internal.common_cuda 导入 TEST_CUDA 常量
from numbers import Number  # 导入 Number 类型，用于数值检查
from typing import Dict, Any  # 导入 Dict 和 Any 类型提示
from packaging import version  # 导入 version 模块，用于版本比较
from torch.testing._internal.common_cuda import (
    SM53OrLater, SM80OrLater, TEST_MULTIGPU  # 导入 GPU 相关常量和函数
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, ops, dtypes, dtypesIfCUDA, onlyCPU, onlyCUDA,
    precisionOverride, deviceCountAtLeast, OpDTypes, onlyNativeDeviceTypes
)  # 导入设备类型相关的测试函数和常量
from torch.testing._internal.common_methods_invocations import (
    op_db, reduction_ops, sparse_unary_ufuncs, sparse_masked_reduction_ops, binary_ufuncs
)  # 导入操作数据库和相关函数
from torch.testing._internal.common_dtype import (
    all_types, all_types_and_complex, all_types_and_complex_and, floating_and_complex_types,
    floating_and_complex_types_and, integral_types, floating_types_and
)  # 导入数据类型相关的测试常量和函数
from torch.testing._internal.opinfo.definitions.sparse import validate_sample_input_sparse  # 导入稀疏操作定义
from torch.testing._internal.opinfo.refs import (
    ElementwiseBinaryPythonRefInfo, ReductionPythonRefInfo
)  # 导入稀疏操作参考实现

def _op_supports_any_sparse(op):
    return (op.supports_sparse
            or op.supports_sparse_csr
            or op.supports_sparse_csc
            or op.supports_sparse_bsr
            or op.supports_sparse_bsc)  # 检查操作是否支持稀疏格式的输入

# 筛选具有稀疏支持的带约简操作
reduction_ops_with_sparse_support = [
    op for op in reduction_ops if 'masked.' not in op.name and
    _op_supports_any_sparse(op) and not isinstance(op, ReductionPythonRefInfo)
]

# 筛选具有稀疏支持的二元操作
binary_ufuncs_with_sparse_support = [
    op for op in binary_ufuncs if _op_supports_any_sparse(op) and
    not isinstance(op, ElementwiseBinaryPythonRefInfo)
]

# 筛选具有稀疏支持的像操作
like_fns_with_sparse_support = [op for op in op_db if _op_supports_any_sparse(op) and '_like' in op.name]

if TEST_SCIPY:
    import scipy.sparse  # 如果测试需要，导入 scipy.sparse 库

# load_tests 函数用于在 sandcastle 上自动过滤测试用例以进行分片，以下代码行用于消除 flake 警告
load_tests = load_tests

# 部分梯度检查不支持稀疏操作，因此使用 functools.partial 禁用批处理梯度检查
gradcheck = functools.partial(gradcheck, check_batched_grad=False)

# 检查是否支持复数稀疏矩阵乘法（CUSPARSE_SPMM_COMPLEX128_SUPPORTED 和 HIPSPARSE_SPMM_COMPLEX128_SUPPORTED 用于版本和平台判断）
CUSPARSE_SPMM_COMPLEX128_SUPPORTED = (
    IS_WINDOWS and torch.version.cuda and version.parse(torch.version.cuda) > version.parse("11.2")
) or (not IS_WINDOWS and not TEST_WITH_ROCM)

HIPSPARSE_SPMM_COMPLEX128_SUPPORTED = torch.version.hip and version.parse(torch.version.hip.split("-")[0]) >= version.parse("6.0")

def all_sparse_layouts(test_name='layout', include_strided=False):
    # 使用 parametrize 函数对 test_name 进行参数化，并返回结果
    return parametrize(test_name, [
        # 调用 subtest 函数生成包含 torch.strided 的子测试对象，命名为 'Strided'
        subtest(torch.strided, name='Strided'),
        # 调用 subtest 函数生成包含 torch.sparse_coo 的子测试对象，命名为 'SparseCOO'
        subtest(torch.sparse_coo, name='SparseCOO'),
        # 调用 subtest 函数生成包含 torch.sparse_csr 的子测试对象，命名为 'SparseCSR'
        subtest(torch.sparse_csr, name='SparseCSR'),
        # 调用 subtest 函数生成包含 torch.sparse_csc 的子测试对象，命名为 'SparseCSC'
        subtest(torch.sparse_csc, name='SparseCSC'),
        # 调用 subtest 函数生成包含 torch.sparse_bsr 的子测试对象，命名为 'SparseBSR'
        subtest(torch.sparse_bsr, name='SparseBSR'),
        # 调用 subtest 函数生成包含 torch.sparse_bsc 的子测试对象，命名为 'SparseBSC'
        subtest(torch.sparse_bsc, name='SparseBSC'),
    # 根据 include_strided 决定使用的子测试对象的范围，0 表示包含，1 表示不包含
    ][
        (0 if include_strided else 1):  # 切片操作，根据 include_strided 来选择切片起始位置
    ])
def gradcheck_semantics(test_name='gradcheck'):
    # 创建两个偏函数，用于分别进行梯度检查，一个不使用掩码，一个使用掩码
    gradcheck_sparse = functools.partial(gradcheck, masked=False)
    gradcheck_masked = functools.partial(gradcheck, masked=True)
    
    # 设置gradcheck_sparse和gradcheck_masked的masked属性
    gradcheck_sparse.masked = False
    gradcheck_masked.masked = True
    
    # 使用parametrize函数对测试名称和两个梯度检查函数进行参数化
    return parametrize(test_name, [
        # 创建两个子测试，分别对应稀疏模式和掩码模式的梯度检查
        subtest(gradcheck_sparse, name='sparse'),
        subtest(gradcheck_masked, name='masked')])


class CrossRefSparseFakeMode(torch._subclasses.CrossRefFakeMode):
    def __init__(self):
        # 调用父类构造函数初始化
        super().__init__(
            self.ignore_op, check_strides=False,
            check_aliasing=False,
        )  # TODO: enable stride/alias checking

    # 排除empty_like函数，因为稀疏复杂性而被排除
    # aten._to_dense.default这个函数在csc模式下被调用
    @staticmethod
    def ignore_op(func):
        # 定义一个静态方法，判断给定的函数是否在忽略列表中
        return func in (
            torch.ops.aten.empty_like.default,
            torch.ops.aten.set_.source_Storage_storage_offset,
            torch.ops.aten.sspaddmm.out,
            torch.ops.aten._spdiags.default,
            torch.ops.aten._to_dense.default,
            torch.ops.aten.indices.default,
            torch.ops.aten._indices.default,
            torch.ops.aten.values.default,
            torch.ops.aten._values.default,
        )

class TestSparseLegacyAndDeprecation(TestCase):

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_legacy_warnings(self):
        # 定义函数 f1，测试对应的警告信息和替代方案
        def f1():
            "torch.sparse.SparseTensor() is deprecated."\
                "  Please use torch.sparse_coo_tensor((0,), dtype=)"
            # 创建稀疏张量 x_ref，推荐使用的新方法
            x_ref = torch.sparse_coo_tensor((0,), dtype=torch.float64)
            # 创建稀疏双精度张量 x
            x = torch.sparse.DoubleTensor()
            # 断言 x 等于 x_ref
            self.assertEqual(x, x_ref)

        # 定义函数 f2，测试对应的警告信息和替代方案
        def f2():
            "torch.sparse.SparseTensor(cdata=x._cdata) is deprecated."\
                "  Please use torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)"
            # 创建稀疏张量 x_ref，推荐使用的新方法
            x_ref = torch.tensor([[1, 2], [3, 4]], dtype=torch.float64).to_sparse()
            # 创建稀疏双精度张量 x，使用推荐的参数形式
            x = torch.sparse.DoubleTensor(cdata=x_ref._cdata)
            # 创建稀疏 COO 张量 y，使用推荐的参数形式
            y = torch.sparse_coo_tensor(x._indices(), x._values(), x.shape)
            # 断言 x 等于 x_ref，y 也等于 x_ref
            self.assertEqual(x, x_ref)
            self.assertEqual(y, x_ref)

        # 定义函数 f3，测试对应的警告信息和替代方案
        def f3():
            "torch.sparse.SparseTensor(indices, values, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, dtype=, device=)"
            # 创建稀疏 COO 张量 x_ref，推荐使用的新方法
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], dtype=torch.float64)
            # 创建稀疏双精度张量 x，使用推荐的参数形式
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64))
            # 断言 x 等于 x_ref
            self.assertEqual(x, x_ref)

        # 定义函数 f4，测试对应的警告信息和替代方案
        def f4():
            "torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=)"
            # 创建稀疏 COO 张量 x_ref，推荐使用的新方法
            x_ref = torch.sparse_coo_tensor([[0, 0, 1, 1], [0, 1, 0, 1]], [1, 2, 3, 4], (2, 3), dtype=torch.float64)
            # 创建稀疏双精度张量 x，使用推荐的参数形式
            x = torch.sparse.DoubleTensor(torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]]),
                                          torch.tensor([1, 2, 3, 4], dtype=torch.float64), (2, 3))
            # 断言 x 等于 x_ref
            self.assertEqual(x, x_ref)

        # 定义函数 f5，测试对应的警告信息和替代方案
        def f5():
            "torch.sparse.SparseTensor(shape, *, device=) is deprecated."\
                "  Please use torch.sparse_coo_tensor(shape, dtype=, device=)"
            # 创建稀疏 COO 张量 x_ref，推荐使用的新方法
            x_ref = torch.sparse_coo_tensor((2, 3), dtype=torch.float64)
            # 创建稀疏双精度张量 x，使用推荐的参数形式
            x = torch.sparse.DoubleTensor(2, 3)
            # 断言 x 等于 x_ref
            self.assertEqual(x, x_ref)

        # 遍历测试函数列表，检查警告是否发出且仅发出一次
        for test_f in [f1, f2, f3, f4, f5]:
            with self.assertWarns(UserWarning, msg=test_f.__doc__) as cm:
                test_f()
                test_f()

            # 检查警告是否仅发出一次
            self.assertEqual(len(cm.warnings), 1)
    # 定义 TestSparseBase 类，继承自 TestCase
    class TestSparseBase(TestCase):
        
        # 重写 run 方法，处理交叉引用测试模式开关
        def run(self, result=None):
            # 如果 TEST_WITH_CROSSREF 为真
            if TEST_WITH_CROSSREF:
                # 使用交叉引用的虚拟模式
                with CrossRefSparseFakeMode():
                    # 调用父类的 run 方法并返回结果
                    return super().run(result)
            else:
                # 否则直接调用父类的 run 方法并返回结果
                return super().run(result)

    # 定义 TestSparse 类，继承自 TestSparseBase
    class TestSparse(TestSparseBase):

        # 设置测试的初始化方法
        def setUp(self):
            # 调用父类的 setUp 方法
            TestCase.setUp(self)

            # 定义一个创建索引张量的 lambda 函数
            self.index_tensor = lambda *args, **kwargs: torch.tensor(*args, **kwargs, dtype=torch.int64)

            # 定义一个稀疏张量为空的工厂函数
            def sparse_empty_factory(*args, **kwargs):
                kwargs['layout'] = kwargs.get('layout', torch.sparse_coo)
                return torch.empty(*args, **kwargs)
            self.sparse_empty = sparse_empty_factory

            # 定义一个创建稀疏张量的工厂函数
            def sparse_tensor_factory(*args, **kwargs):
                return torch.sparse_coo_tensor(*args, **kwargs)
            self.sparse_tensor = sparse_tensor_factory

        # 生成稀疏张量的方法
        def _gen_sparse(self, sparse_dim, nnz, with_size, dtype, device, coalesced):
            # 如果 with_size 是数字，则重复 sparse_dim 次
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dim

            # 调用 genSparseTensor 方法生成稀疏张量 x, i, v
            x, i, v = self.genSparseTensor(with_size, sparse_dim, nnz, not coalesced, dtype=dtype, device=device)

            # 如果未压缩，则断言 x 未压缩
            if not coalesced:
                self.assert_uncoalesced(x)

            # 返回稀疏张量 x, 索引 i, 值 v
            return x, i, v

        # 断言张量未压缩的方法
        def assert_uncoalesced(self, x):
            """
            Test if a CPU tensor is uncoalesced.  This is used to ensure
            correctness of the uncoalesced tensor generation algorithm.
            """
            # 断言 x 未压缩
            assert not x.is_coalesced()
            existing_indices = set()
            indices = x._indices()
            # 遍历所有索引
            for i in range(x._nnz()):
                index = str(indices[:, i])
                # 如果索引已存在于集合中，则返回真
                if index in existing_indices:
                    return True
                else:
                    existing_indices.add(index)

        # 生成随机张量的方法，支持 TEST_CUDA 情况
        def randn(self, *args, **kwargs):
            """
            Variant of torch.randn that also works in the TEST_CUDA case.
            """
            # TODO: Put this in torch.cuda.randn
            # 返回一个使用正态分布填充的空张量
            return torch.empty(*args, **kwargs).normal_()

        # 测试打印压缩和未压缩状态的方法，使用指定的设备和数据类型
        @dtypes(torch.double)
        def test_print_coalesced(self, device, dtype):
            self._test_print(device, dtype, True)

        @dtypes(torch.double)
        def test_print_uncoalesced(self, device, dtype):
            self._test_print(device, dtype, False)
    # 定义测试方法，用于打印稀疏张量相关信息
    def _test_print(self, device, dtype, coalesced):
        # 定义不同形状、稀疏维度、非零元素个数的测试数据集合
        shape_sparse_dim_nnz = [
            ((), 0, 2),              # 空形状，0 稀疏维度，2 非零元素个数
            ((0,), 0, 10),           # 形状 (0,)，0 稀疏维度，10 非零元素个数
            ((2,), 0, 3),            # 形状 (2,)，0 稀疏维度，3 非零元素个数
            ((100, 3), 1, 3),        # 形状 (100, 3)，1 稀疏维度，3 非零元素个数
            ((100, 20, 3), 2, 0),    # 形状 (100, 20, 3)，2 稀疏维度，0 非零元素个数
            ((10, 0, 3), 0, 3),      # 形状 (10, 0, 3)，0 稀疏维度，3 非零元素个数
            ((10, 0, 3), 0, 0),      # 形状 (10, 0, 3)，0 稀疏维度，0 非零元素个数
        ]
        # 初始化打印内容列表
        printed = []
        # 遍历测试数据集合
        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            # 计算 indices 的形状
            indices_shape = torch.Size((sparse_dim, nnz))
            # 计算 values 的形状
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            # 添加形状相关信息到打印内容列表
            printed.append(f"# shape: {torch.Size(shape)}")
            printed.append(f"# nnz: {nnz}")
            printed.append(f"# sparse_dim: {sparse_dim}")
            printed.append(f"# indices shape: {indices_shape}")
            printed.append(f"# values shape: {values_shape}")

            # 生成 indices 张量，使用连续的索引值
            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=device).view(indices_shape)
            # 对每个稀疏维度进行 clamp 操作，确保索引值在有效范围内
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))
            # 如果未压缩并且 indices 元素个数大于 0，则使其不压缩
            if not coalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]
            # 计算 values 的元素个数
            values_numel = values_shape.numel()
            # 生成 values 张量，使用连续的值，然后进行归一化处理
            values = torch.arange(values_numel, dtype=dtype,
                                  device=device).view(values_shape).div_(values_numel / 2.)
            # 创建稀疏张量 sp_tensor
            sp_tensor = self.sparse_tensor(indices, values, shape, dtype=dtype, device=device)

            # 确定需要测试的数据类型列表
            dtypes = [torch.int32]
            if values.dtype == torch.double:
                dtypes.append(torch.float)
            else:
                dtypes.append(torch.double)
            # 遍历数据类型列表
            for dtype in dtypes:
                # 添加当前数据类型标记到打印内容列表
                printed.append(f"########## {dtype} ##########")
                # 将 sp_tensor 转换为当前数据类型的张量 x
                x = sp_tensor.detach().to(dtype)
                # 添加稀疏张量信息到打印内容列表
                printed.append("# sparse tensor")
                printed.append(str(x))
                # 若 x 的数据类型为浮点数类型
                if x.dtype.is_floating_point:
                    # 添加 requires_grad_ 后的信息到打印内容列表
                    printed.append("# after requires_grad_")
                    printed.append(str(x.requires_grad_()))
                    # 添加加法操作后的信息到打印内容列表
                    printed.append("# after addition")
                    printed.append(str(x + x))
                # 添加 _indices 方法返回的信息到打印内容列表
                printed.append("# _indices")
                printed.append(str(x._indices()))
                # 添加 _values 方法返回的信息到打印内容列表
                printed.append("# _values")
                printed.append(str(x._values()))
            # 添加空行到打印内容列表
            printed.append('')
        # 断言打印内容与预期输出一致
        self.assertExpected('\n'.join(printed))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义测试方法，测试稀疏张量的基本功能
    def test_basic(self, device, dtype, coalesced):
        # 定义测试形状的辅助函数
        def test_shape(sparse_dims, nnz, with_size):
            # 如果 with_size 是数字，则转换为一个包含相同数字的列表
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            # 生成稀疏张量 x、索引 i 和值 v
            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            # 断言索引 i 等于 x 的内部索引
            self.assertEqual(i, x._indices())
            # 断言值 v 等于 x 的内部值
            self.assertEqual(v, x._values())
            # 断言 x 的维度数量等于 with_size 的长度
            self.assertEqual(x.ndimension(), len(with_size))
            # 断言 x 的压缩后非零元素数量等于 nnz（如果 x 已经压缩）或 nnz 的一半（如果 x 未压缩）
            self.assertEqual(x.coalesce()._nnz(), nnz if x.is_coalesced() else nnz // 2)
            # 断言 x 的尺寸列表等于 with_size
            self.assertEqual(list(x.size()), with_size)

            # 测试 .indices() 和 .values() 方法
            if not coalesced:
                # 如果未压缩，则断言调用 .indices() 和 .values() 会引发 RuntimeError
                with self.assertRaisesRegex(RuntimeError, "Cannot get indices on an uncoalesced tensor"):
                    x.indices()
                with self.assertRaisesRegex(RuntimeError, "Cannot get values on an uncoalesced tensor"):
                    x.values()
            else:
                # 如果已压缩，则断言 .indices() 返回与 x._indices() 相同的结果
                self.assertEqual(x.indices(), x._indices())
                # 断言 .values() 返回与 x._values() 相同的结果
                self.assertEqual(x.values(), x._values())

        # 测试不同形状的稀疏张量
        test_shape(3, 10, 100)
        test_shape(3, 10, [100, 100, 100])
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

        # 确保 coalesce 方法能正确处理重复索引
        # 创建包含重复索引的索引张量 i
        i = self.index_tensor([[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]], device=device)
        # 创建值张量 v，每个值是其索引的平方和索引本身
        v = torch.tensor([[idx**2, idx] for idx in range(i.size(1))], dtype=dtype, device=device)
        # 使用索引 i 和值 v 创建稀疏张量 x
        x = self.sparse_tensor(i, v, torch.Size([10, 2]), dtype=dtype, device=device)
        # 断言 x 压缩后的非零元素数量为 9
        self.assertEqual(x.coalesce()._nnz(), 9)

    # 使用装饰器设置测试的数据类型、精度和是否压缩
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 1e-2})
    # 如果遇到 TorchDynamo 问题 #1991，则跳过这个测试
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    # 测试稀疏张量的合并操作的正确性
    def test_coalesce(self, device, dtype, coalesced):
        
        # 内部函数，用于测试稀疏张量的合并操作
        def _test_coalesce(t):
            # 对输入的张量 t 进行合并操作
            tc = t.coalesce()
            # 断言合并后稀疏张量与原始张量在密集形式上的相等性
            self.assertEqual(tc.to_dense(), t.to_dense())
            # 断言合并后张量是否为稀疏格式
            self.assertTrue(tc.is_coalesced())
            
            # 当输入张量 t 的非零元素数为 0 时，以下代码块将不起作用，因为此时 t 是0维张量，而不是2维张量
            if t._nnz() == 0:
                # 断言稀疏张量的索引与值与原始张量相同
                self.assertEqual(t._indices(), tc._indices())
                self.assertEqual(t._values(), tc._values())
                return tc

            # 创建一个字典，用于存储张量 t 的索引和对应值的映射关系
            value_map: Dict[Any, Any] = {}
            for idx, val in zip(t._indices().t(), t._values()):
                idx_tup = tuple(idx.tolist())
                if idx_tup in value_map:
                    value_map[idx_tup] += val
                else:
                    # 如果值是张量，则进行克隆操作，否则直接赋值
                    value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

            # 对索引进行排序
            new_indices = sorted(value_map.keys())
            # 根据排序后的索引获取对应的值
            _new_values = [value_map[idx] for idx in new_indices]
            
            # 根据原始值的维度情况创建新的值张量
            if t._values().ndimension() < 2:
                new_values = t._values().new(_new_values)
            else:
                new_values = torch.stack(_new_values)

            # 创建新的稀疏张量 tg
            new_indices = t._indices().new(new_indices).t()
            tg = t.new(new_indices, new_values, t.size())

            # 断言合并后的稀疏张量与新创建的张量 tg 在索引和值上的一致性
            self.assertEqual(tc._indices(), tg._indices())
            self.assertEqual(tc._values(), tg._values())

            # 如果输入张量 t 已经是稀疏的，则断言合并后的稀疏张量与原始张量在索引和值上的一致性
            if t.is_coalesced():
                self.assertEqual(tc._indices(), t._indices())
                self.assertEqual(tc._values(), t._values())

        # 遍历所有可能的空值情况，生成稀疏张量进行合并测试
        for empty_i, empty_v, empty_nnz in itertools.product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5

            # 生成稀疏张量 t，并调用内部函数 _test_coalesce 进行测试
            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size + dense_size, dtype, device, coalesced)
            _test_coalesce(t)  # 这里测试合并操作的正确性

    # 使用双精度类型进行测试
    @dtypes(torch.double)
    # 如果存在 Torch Dynamo 问题，跳过此测试
    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/89395")
    def test_coalesce_reference_cycle(self, device, dtype):
        # 测试合并操作是否会创建自动求导图的循环引用（gh-52253）

        # 检查辅助类是否按预期工作
        t = torch.rand(2)
        t_ref = torch._C._WeakTensorRef(t)
        self.assertFalse(t_ref.expired())

        # 删除张量 t，并检查其引用是否已过期
        del t
        self.assertTrue(t_ref.expired())

        # 内部函数，测试稀疏张量的求和操作
        def test_sparse_sum():
            i = torch.tensor([[0], [4]], dtype=torch.long, device=device)
            v = torch.tensor([[[-0.4567, -1.8797, 0.0380, 1.4316]]],
                             dtype=dtype, device=device)
            S = torch.sparse_coo_tensor(i, v)
            S = S.coalesce()
            S.requires_grad_(True)
            S2 = S.coalesce()
            # 断言合并后的稀疏张量 S2 是否为合并格式
            self.assertTrue(S2.is_coalesced())
            return torch._C._WeakTensorRef(S2)

        # 调用内部函数 test_sparse_sum，测试稀疏张量求和操作
        ref = test_sparse_sum()
        # 断言引用 ref 是否已过期
        self.assertTrue(ref.expired())

    # 使用双精度类型进行测试
    @dtypes(torch.double)
    # 测试在使用大尺寸稀疏张量（gh-57416）时是否检测到整数溢出问题。
    # 注意，构建张量时内部计算元素数量（numel），因此溢出可能发生在张量构建阶段。
    def test_ctor_large_sizes(self, device, dtype):
        N = 100000
        # 创建包含大尺寸索引的张量，使用设备上的整型64位数据类型
        indices = torch.tensor([[N, N - 1]] * 4, dtype=torch.int64, device=device)
        # 创建值张量，使用给定的数据类型和设备
        values = torch.tensor([1, 2], dtype=dtype, device=device)
        # 断言是否引发 RuntimeError 异常，当构建稀疏 COO 张量时，尺寸维度溢出
        self.assertRaises(RuntimeError,
                          lambda: torch.sparse_coo_tensor(
                              indices, values, (N + 1,) * 4, device=device))

    @dtypes(torch.double, torch.cdouble)
    def test_ctor_size_checks(self, device, dtype):
        # 创建索引张量，这些索引与设备一致
        indices = self.index_tensor([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
            [0, 0, 0],
        ], device=device)
        # 创建值张量，使用给定的数据类型和设备
        values = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)

        # 断言是否引发 RuntimeError 异常，当索引与给定尺寸不一致时
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 1, 1])))

        # 更新值张量，使用给定的数据类型和设备
        values = torch.tensor([
            [2, 1, 2, 1],
            [1, 0, 5, 2],
        ], dtype=dtype, device=device)
        # 断言是否引发 RuntimeError 异常，当值张量与给定尺寸不一致时
        self.assertRaises(
            RuntimeError,
            lambda: self.sparse_tensor(indices, values, torch.Size([2, 4, 2, 1])))

    @coalescedonoff
    @dtypes(torch.double)
    def test_ctor_is_coalesced_with_gradcheck(self, device, dtype, coalesced):
        # 对于每个稀疏尺寸和非零元素数组合
        for sparse_size, nnz in (((3, 3), 5), ((2, 3, 1, 5), 11)):
            # 生成稀疏张量及其相关信息
            t, _, _ = self._gen_sparse(len(sparse_size), nnz, sparse_size, dtype, device, coalesced)
            # 断言生成的稀疏张量是否为给定的是否聚合状态
            self.assertEqual(t.is_coalesced(), coalesced)

            def func(indices, values, shape, is_coalesced):
                # 创建稀疏 COO 张量，并进行不变性检查，设定是否聚合状态
                s = torch.sparse_coo_tensor(indices, values, shape, check_invariants=True, is_coalesced=is_coalesced)
                # 断言创建的稀疏张量是否为给定的是否聚合状态
                self.assertEqual(s.is_coalesced(), is_coalesced)
                return s.to_dense(masked_grad=False)

            # 使用梯度检查，测试函数 func 的梯度，参数为稀疏张量的索引、值、形状和是否聚合状态
            if coalesced:
                torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), t.shape, False))
                torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), t.shape, True))
            else:
                torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), t.shape, False))
                # 断言是否引发 RuntimeError 异常，当尝试将索引对应于未聚合的 COO 张量时
                with self.assertRaisesRegex(RuntimeError,
                                            "cannot set is_coalesced to true if indices correspond to uncoalesced COO tensor"):
                    torch.autograd.gradcheck(func, (t._indices(), t._values().requires_grad_(True), t.shape, True))

    @dtypes(*floating_and_complex_types_and(torch.float16, torch.bfloat16))
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    # 使用装饰器 gradcheck_semantics() 对 test_to_dense_with_gradcheck 方法进行语义检查
    @gradcheck_semantics()
    # 定义 test_to_dense_with_gradcheck 方法，参数包括 device（设备）、dtype（数据类型）、gradcheck（梯度检查）
    def test_to_dense_with_gradcheck(self, device, dtype, gradcheck):
    
        # 定义内部函数 test_tensor，用于测试稀疏张量转换为密集张量的操作
        def test_tensor(x, res):
            x.to_dense()  # 测试三次 to_dense 操作，以检测内存损坏
            x.to_dense()
            x.to_dense()
            dense_x = x.to_dense()  # 将稀疏张量转换为密集张量
            safe_dense_x = self.safeToDense(x)  # 调用 self.safeToDense 方法，将稀疏张量转换为密集张量
            dense_x = dense_x.to(res.dtype)  # 将密集张量转换为指定的数据类型
            safe_dense_x = safe_dense_x.to(res.dtype)  # 将安全的密集张量转换为指定的数据类型
            self.assertEqual(res, dense_x)  # 断言密集张量 dense_x 与预期结果 res 相等
            self.assertEqual(res, safe_dense_x)  # 断言安全密集张量 safe_dense_x 与预期结果 res 相等
    
            # 仅当张量 x 的数据类型为 torch.float64 时才运行自动梯度测试
            if x.dtype != torch.float64:
                return
    
            # 定义函数 fn，接收稀疏张量 x 作为输入，返回其转换为密集张量的结果，支持掩码梯度
            def fn(x):
                return x.to_dense(masked_grad=gradcheck.masked)
            x.requires_grad_(True)  # 设置张量 x 需要计算梯度
            gradcheck(fn, (x,))  # 使用 gradcheck 对 fn 函数进行梯度检查
    
        # 遍历数据类型列表，对每种数据类型进行测试
        for value_type in [torch.double, torch.cdouble]:
            # 创建索引张量 i，指定设备为 device
            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            # 对于在 CPU 上的半精度数据类型，不需要进行 to_dense 操作，因为它实现了较慢的 add_ 操作
            v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
            # 创建稀疏张量 x，指定形状和数据类型，并使用指定的设备
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]), dtype=value_type, device=device)
            # 创建预期的密集张量 res，指定形状、数据类型和设备
            res = torch.tensor([
                [[2, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]],
                [[0, 3, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 4]],
            ], dtype=dtype, device=device)
    
            # 调用 test_tensor 函数，对稀疏张量 x 进行测试
            test_tensor(x, res)
            # 再次调用 test_tensor 函数，对密集张量 res 进行测试
            test_tensor(res, res)
    
            # 创建索引张量 i，指定设备为 device
            i = self.index_tensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ], device=device)
            # 创建空的张量 v，指定形状和数据类型，使用指定的设备
            v = torch.empty(4, 0, dtype=dtype, device=device)
            # 创建稀疏张量 x，指定形状、数据类型和设备
            x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]), dtype=value_type, device=device)
            # 创建空的密集张量 res，指定形状、数据类型和设备
            res = torch.empty((3, 4, 5, 0), dtype=dtype, device=device)
            # 调用 test_tensor 函数，对稀疏张量 x 进行测试
            test_tensor(x, res)
    
    # 使用装饰器 coalescedonoff 对下一行代码进行修饰
    @coalescedonoff
    # 使用 dtypes 方法，传入多种数据类型参数，对下一行代码进行修饰
    @dtypes(torch.float16, torch.bfloat16, torch.float64, torch.int, torch.cfloat, torch.cdouble)
    # 测试稀疏张量转换为稀疏格式的功能，使用给定的设备和数据类型
    def test_to_sparse(self, device, dtype, coalesced):
        # 定义张量的形状
        shape = [5, 2, 10, 4]
        # 初始化最大非零元素数量
        max_nnz = 1
        # 遍历数据类型列表，包括 torch.double 和 torch.cdouble
        for value_type in [torch.double, torch.cdouble]:
            # 遍历形状的每个维度
            for dim, dim_sz in enumerate(shape, 1):
                # 计算当前维度下的最大非零元素数量
                max_nnz *= dim_sz
                # 随机生成当前维度下的非零元素数量
                rnnz = torch.randint(2, max_nnz, (1,)).item()
                # 遍历不同的非零元素数量情况：0个、1个、随机生成的数量
                for nnz in [0, 1, rnnz]:
                    # 生成稀疏张量及其期望值和索引
                    expected, _, _ = self._gen_sparse(dim, nnz, shape, dtype=value_type, device=device,
                                                      coalesced=coalesced)
                    # 将期望值张量转换为指定数据类型
                    expected = expected.to(dtype)

                    # 将期望值张量转换为稠密张量
                    d = expected.to_dense()
                    # 将稠密张量转换为指定维度的稀疏张量
                    result = d.to_sparse(dim)
                    # 断言稠密张量和转换后的稠密张量相等
                    self.assertEqual(d, result.to_dense())
                    # 断言期望值张量和转换后的稀疏张量具有相同的尺寸
                    self.assertEqual(expected.size(), result.size())
                    # 断言稀疏张量的稀疏维度与指定维度相等
                    self.assertEqual(dim, result.sparse_dim())

    @dtypes(torch.double, torch.cdouble)
    # 测试布尔类型的稀疏张量转换和断言
    def test_sparse_bool(self, device, dtype):
        # 创建包含 True 和 False 的张量，并转换为稀疏张量后再转换回稠密张量
        a = torch.tensor([True, False], dtype=dtype, device=device).to(torch.bool)
        b = a.to_sparse().to_dense()
        self.assertEqual(a, b)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/108667")
    @dtypes(torch.double, torch.cdouble)
    # 测试包含标量值的稀疏张量的各种操作
    def test_scalar(self, device, dtype):
        # 创建具有给定值的稀疏张量，并执行多种断言操作
        # 情况1: 张量只包含一个值
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1), 12.3, [], dtype=dtype, device=device)
        self.assertEqual(1, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

        # 情况2: 张量包含多个相同的值
        a = self.sparse_tensor(self.index_tensor([], device=device).unsqueeze(1).expand(0, 2),
                               [12.3, 12.3], [], dtype=dtype, device=device)
        self.assertEqual(2, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(12.3 * 2, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a.coalesce(), a.coalesce().to_dense().to_sparse())

        # 情况3: 张量不包含值
        a = self.sparse_empty((), dtype=dtype, device=device)
        self.assertEqual(0, a._values().numel())
        self.assertEqual(a, a.clone())
        a_coalesced = a.coalesce()
        self.assertTrue(a_coalesced.is_coalesced())
        self.assertEqual(torch.tensor(0, dtype=dtype, device=device), a.to_dense())
        self.assertEqual(a, a.to_dense().to_sparse())

    @dtypes(torch.double, torch.cdouble)
    # 定义一个测试方法，用于测试稀疏张量到稠密张量的转换
    def test_shared(self, device, dtype):
        # 创建索引张量 i，包含一个值为 2 的索引
        i = self.index_tensor([[2]], device=device)
        # 创建值张量 v，包含一个值为 5 的张量，指定数据类型和设备
        v = torch.tensor([5], dtype=dtype, device=device)
        # 使用稀疏张量构造函数创建稀疏张量 x，形状为 [3]
        x = self.sparse_tensor(i, v, torch.Size([3]))
        # 修改值张量 v 中的第一个元素为 6
        v[0] = 6
        # 断言稠密张量与安全转换后的 x 的结果一致
        self.assertEqual(torch.tensor([0, 0, 6], dtype=dtype, device=device), self.safeToDense(x))
        # 修改索引张量 i 中的第一个元素为 0
        i[0][0] = 0
        # 断言稠密张量与安全转换后的 x 的结果一致
        self.assertEqual(torch.tensor([6, 0, 0], dtype=dtype, device=device), self.safeToDense(x))

        # 重新创建索引张量 i，包含一个值为 2 的索引
        i = self.index_tensor([[2]], device=device)
        # 创建一个空的值张量 v，形状为 (1, 0)，指定数据类型和设备
        v = torch.empty((1, 0), dtype=dtype, device=device)
        # 使用稀疏张量构造函数创建稀疏张量 x，形状为 [3, 0]
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        # 修改索引张量 i 中的第一个元素为 0
        i[0][0] = 0
        # 断言稠密张量与安全转换后的 x 的结果一致
        self.assertEqual(torch.empty((3, 0), dtype=dtype, device=device), self.safeToDense(x))

    # 定义一个用于测试稀疏张量转换为稠密张量的混合类型的方法
    @dtypes(torch.double, torch.cdouble)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    @gradcheck_semantics()
    def test_to_dense_hybrid(self, device, dtype, gradcheck):

        # 定义一个内部方法 test_tensor，用于测试张量 x 和其预期结果 res 的转换
        def test_tensor(x, res):
            # 调用 x 的 to_dense 方法多次，用于测试内存损坏
            x.to_dense()
            x.to_dense()
            x.to_dense()
            # 断言稠密张量与安全转换后的 x 的结果与预期结果 res 一致
            self.assertEqual(res, x.to_dense())
            # 断言稠密张量与安全转换后的 x 的结果与预期结果 res 一致
            self.assertEqual(res, self.safeToDense(x))

            # 定义一个函数 fn，接受参数 x，并返回其稠密版本的张量，支持梯度检查
            def fn(x):
                return x.to_dense(masked_grad=gradcheck.masked)
            # 将 x 设置为需要计算梯度
            x.requires_grad_(True)
            # 使用 gradcheck 检查函数 fn 的梯度
            gradcheck(fn, (x,))

        # 创建索引张量 i，包含多个索引值
        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        # 创建值张量 v，包含多个值，指定数据类型和设备
        v = torch.tensor([[2, 3], [1, 2], [3, 4], [4, 5]], dtype=dtype, device=device)
        # 使用稀疏张量构造函数创建稀疏张量 x，形状为 [3, 4, 2]
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2]))
        # 创建预期结果 res，形状与 x 相同，包含特定的数值
        res = torch.tensor([
            [[2, 3],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[1, 2],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[3, 4],
             [0, 0],
             [0, 0],
             [4, 5]],
        ], dtype=dtype, device=device)
        # 调用 test_tensor 方法，测试稀疏张量 x 与预期结果 res
        test_tensor(x, res)

        # 重新创建索引张量 i，包含多个索引值
        i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ], device=device)
        # 创建一个空的值张量 v，形状为 (4, 2, 0)，指定数据类型和设备
        v = torch.empty((4, 2, 0), dtype=dtype, device=device)
        # 使用稀疏张量构造函数创建稀疏张量 x，形状为 [3, 4, 2, 0]
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 2, 0]))
        # 创建一个空的预期结果 res，形状与 x 相同
        res = torch.empty((3, 4, 2, 0), dtype=dtype, device=device)
        # 调用 test_tensor 方法，测试稀疏张量 x 与预期结果 res
        test_tensor(x, res)
    # 定义测试函数 test_contig，接受设备和数据类型作为参数
    def test_contig(self, device, dtype):
        # 定义内部函数 test_tensor，用于测试稀疏张量的索引和值
        def test_tensor(x, exp_i, exp_v):
            # 对稀疏张量进行稀疏化操作
            x = x.coalesce()
            # 断言稀疏张量的索引与期望索引相等
            self.assertEqual(exp_i, x._indices())
            # 断言稀疏张量的值与期望值相等
            self.assertEqual(exp_v, x._values())

        # 创建第一个稀疏张量 x
        i = self.index_tensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ], device=device)
        v = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([100, 100]))
        exp_i = self.index_tensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ], device=device)
        exp_v = torch.tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7], dtype=dtype, device=device)
        # 调用 test_tensor 函数进行测试
        test_tensor(x, exp_i, exp_v)

        # 创建第二个稀疏张量 x
        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.tensor([2, 1, 3, 4], dtype=dtype, device=device)
        # 调用 test_tensor 函数进行测试
        test_tensor(x, exp_i, exp_v)

        # 创建第三个稀疏张量 x
        i = self.index_tensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ], device=device)
        exp_v = torch.empty([4, 0], dtype=dtype, device=device)
        # 调用 test_tensor 函数进行测试
        test_tensor(x, exp_i, exp_v)

        # 创建具有重复索引的稀疏张量 x
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.tensor([3, 2, 4, 1], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.tensor([6, 4], dtype=dtype, device=device)
        # 调用 test_tensor 函数进行测试
        test_tensor(x, exp_i, exp_v)

        # 创建具有重复索引的稀疏张量 x
        i = self.index_tensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ], device=device)
        v = torch.empty([4, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.index_tensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ], device=device)
        exp_v = torch.empty([2, 0], dtype=dtype, device=device)
        # 调用 test_tensor 函数进行测试
        test_tensor(x, exp_i, exp_v)

    # 使用装饰器指定参数类型为 torch.double 或 torch.cdouble
    @dtypes(torch.double, torch.cdouble)
    # 使用装饰器启用或禁用稀疏张量的稀疏化操作
    @coalescedonoff
    # 使用装饰器指定参数类型为 torch.double 或 torch.cdouble
    @dtypes(torch.double, torch.cdouble)
    def test_clone(self, device, dtype, coalesced):
        # 定义内部测试函数，用于测试不同稀疏度、非零元素数量和大小的稀疏张量克隆操作
        def test_shape(sparse_dims, nnz, with_size):
            # 生成稀疏张量 x，并返回生成的稀疏张量及其属性
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            # 如果不是紧凑的，则断言 x 不是紧凑的
            if not coalesced:
                self.assertFalse(x.is_coalesced())
                # 对 x 进行克隆操作，断言克隆结果 y 不是紧凑的
                y = x.clone()
                self.assertFalse(y.is_coalesced())
            # 将 x 进行紧凑化处理
            x = x.coalesce()
            # 断言 x 已经是紧凑的
            self.assertTrue(x.is_coalesced())
            # 对 x 进行克隆操作，断言克隆结果 y 也是紧凑的
            y = x.clone()
            self.assertTrue(y.is_coalesced())

        # 使用不同的参数调用 test_shape 函数进行测试
        test_shape(4, 20, 5)
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble, torch.bfloat16)
    @precisionOverride({torch.bfloat16: 2e-2})
    def test_Sparse_to_Sparse_copy_(self, device, dtype, coalesced):
        # 用于测试 torch.copy_(SparseTensor, SparseTensor)
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # 混合稀疏
        # 生成两个稀疏张量 x1 和 x2，并返回生成的稀疏张量及其属性
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)

        # 测试复制操作
        x2_dense = x2.to_dense()
        x1.copy_(x2)
        # 断言 x1 的密集形式与 x2 的相同
        self.assertEqual(x2_dense, x1.to_dense())

        # 测试类型转换（当 x1.copy_(x2) 时，x1 的数据类型应该保持不变）
        x1 = x1.to(torch.float32)

        x2 = x2.to(torch.float16)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        # 断言 x1 的数据类型不变
        self.assertEqual(x1_dtype, x1.dtype)

        x2 = x2.to(torch.float64)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        # 断言 x1 的数据类型不变
        self.assertEqual(x1_dtype, x1.dtype)

        # 测试无广播情况
        self.assertRaises(RuntimeError, lambda: x1.copy_(x2.narrow_copy(0, 0, 1)))

        # 测试在密集张量和稀疏张量之间复制操作会引发错误
        self.assertRaises(RuntimeError, lambda: x1.copy_(torch.randn(5, 5)))

        # 测试自动求导
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone()
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        # 断言梯度的期望值与 x2 的梯度的密集形式相同
        self.assertEqual(expected_grad.to_dense(), x2.grad.to_dense())
        self.assertEqual(None, x1.grad)

    @coalescedonoff
    @unittest.skipIf(not TEST_MULTIGPU, "multi-GPU not supported")
    @dtypes(torch.double, torch.cdouble)
    def test_Sparse_to_Sparse_copy_multi_gpu(self, device, dtype, coalesced):
        # This function tests copying between SparseTensor objects across multiple GPU devices.
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # Define sizes for hybrid sparse tensor
        # Generate two sparse tensors with different nnz values
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes, dtype, device, coalesced)
        x1 = x1.to('cuda:0')  # Move x1 tensor to 'cuda:0' device

        def test_cross_device(x1, x2):
            # Inner function to test copy_() method across different devices
            x1_device = x1.device  # Get current device of x1
            x1.copy_(x2)  # Copy data from x2 to x1
            # Assert that the dense representation of x1 matches x2 after the copy
            self.assertEqual(x2.to('cuda:0').to_dense(), x1.to_dense())
            self.assertEqual(x1_device, x1.device)  # Assert device of x1 remains unchanged

        test_cross_device(x1, x2.to('cuda:1'))  # Test across different GPU devices
        test_cross_device(x1, x2.to('cpu'))  # Test between CPU and GPU

        # Test autograd capabilities
        x2 = x2.to('cuda:1')  # Move x2 to 'cuda:1'
        x2.requires_grad_(True)  # Enable gradient tracking for x2
        x1.copy_(x2)  # Copy x2 data to x1
        y = x1 * 2  # Perform a multiplication operation on x1
        x2_clone = x2.clone().to('cuda:0')  # Clone x2 and move it to 'cuda:0'
        y.backward(x2_clone)  # Perform backward pass with respect to x2_clone
        expected_grad = x2_clone * 2  # Compute expected gradient
        # Assert that gradients are correctly calculated and match expected values
        self.assertEqual(expected_grad.to_dense(), x2.grad.to('cuda:0').to_dense())
        self.assertEqual(None, x1.grad)  # Assert no gradient for x1

    @onlyCUDA
    def test_cuda_empty(self, device):
        # Test various properties of sparse tensors when moved across different devices
        def test_tensor(x):
            y = x.to(device)  # Move tensor x to specified device
            self.assertEqual(x.sparse_dim(), y.sparse_dim())  # Assert sparse dimensions are preserved
            self.assertEqual(x.dense_dim(), y.dense_dim())  # Assert dense dimensions are preserved
            x = y.cpu()  # Move tensor back to CPU
            self.assertEqual(y.sparse_dim(), x.sparse_dim())  # Assert sparse dimensions are preserved
            self.assertEqual(y.dense_dim(), x.dense_dim())  # Assert dense dimensions are preserved

        # Test different sparse tensors with specified coordinates and data type
        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float32)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4), dtype=torch.float16)
        test_tensor(x)

        x = torch.sparse_coo_tensor((2, 3, 4, 0), dtype=torch.float32)
        test_tensor(x)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_transpose(self, device, dtype, coalesced):
        # Test transpose operations on generated sparse tensors
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            y = self.safeToDense(x)  # Convert sparse tensor x to dense tensor y for comparison

            # Test transpose operations between all pairs of dimensions (i, j)
            for i, j in itertools.combinations(range(4), 2):
                x = x.transpose_(i, j)  # In-place transpose operation on x
                y = y.transpose(i, j)  # Transpose operation on y for comparison
                self.assertEqual(self.safeToDense(x), y)  # Assert equality after transpose

                x = x.transpose(i, j)  # Non-in-place transpose operation on x
                y = y.transpose(i, j)  # Transpose operation on y for comparison
                self.assertEqual(self.safeToDense(x), y)  # Assert equality after transpose

        test_shape(4, 6, 3)  # Test with specific dimensions and nnz values
        test_shape(4, 3, [7, 7, 7, 3, 3, 3, 0])  # Test with varying sizes
        test_shape(4, 0, [0, 0, 7, 3, 3, 3, 0])  # Test with zero nnz and varying sizes
    # 定义一个测试方法，用于测试稀疏张量的 permute 操作
    def test_permute(self, device, dtype, coalesced, gradcheck):
        # 创建一个随机的稀疏张量 s，形状为 (3, 3, 3)，在指定设备上和指定数据类型上进行操作，并转换为稀疏格式
        s = torch.rand(3, 3, 3, device=device, dtype=dtype).to_sparse()
        # 断言当指定的维度顺序不匹配时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "does not match the length"):
            s.permute(dims=(1, 0))
        # 断言当指定的维度包含重复时会抛出 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, "duplicate dims"):
            s.permute(dims=(1, 1, 1))
        
        # 在稀疏张量 x 上调用 permute 方法，之前使用空元组调用会导致崩溃，参考 https://github.com/pytorch/pytorch/issues/116325
        x = torch.rand((), device=device, dtype=dtype).to_sparse()
        x.permute(())
        # 断言调用 permute 方法后稀疏张量 x 的值的长度为 1
        self.assertEqual(len(x.values()), 1)

        # 定义一个测试形状的函数，测试不同的稀疏维度、非零元素个数和尺寸组合
        def test_shape(sparse_dims, nnz, with_size):
            ndim = len(with_size)
            # 获取有效的稀疏维度和密集维度的范围
            valid_sparse_dims = torch.arange(-ndim, -ndim + sparse_dims)
            valid_dense_dims = torch.arange(-ndim + sparse_dims, 0)

            # 遍历所有维度的排列组合
            for dims in itertools.permutations(range(-ndim, 0)):
                # 生成稀疏张量 s 和对应的密集张量 d
                s = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
                d = self.safeToDense(s)

                # 对稀疏维度和密集维度进行排序
                dims_sparse, _ = torch.tensor(dims[:sparse_dims]).sort()
                dims_dense, _ = torch.tensor(dims[sparse_dims:]).sort()

                # 如果当前排列是有效的，测试其正确性
                if (valid_sparse_dims == dims_sparse).all() and (valid_dense_dims == dims_dense).all():
                    # 对稀疏张量进行 permute 操作，并断言结果与对应的密集张量 permute 后的结果相等
                    s_permuted = s.permute(dims)
                    self.assertEqual(s_permuted, d.permute(dims))

                    # 如果原始稀疏张量是 coalesced，并且 permute 操作不涉及到第一个维度，则结果也必须是 coalesced 的
                    if dims[0] == 0:
                        self.assertEqual(s_permuted.is_coalesced(), s.is_coalesced())
                    else:
                        self.assertFalse(s_permuted.is_coalesced())

                    # 使用 gradcheck 方法验证 permute 操作对梯度的影响
                    gradcheck(lambda t: t.permute(dims).to_dense(masked_grad=gradcheck.masked), s.requires_grad_())
                else:
                    # 否则检查是否抛出了预期的 RuntimeError 异常
                    fail_message = "transpositions between sparse and dense dimensions are not allowed"
                    with self.assertRaisesRegex(RuntimeError, fail_message):
                        s.permute(dims)

        # 分别使用不同的参数调用 test_shape 函数进行测试
        test_shape(2, 3, [2, 3, 4, 5])
        test_shape(2, 3, [2, 2, 0])
        # 当 nnz=0 时，稀疏张量 t 不等于 t.to_dense().to_sparse()，除非 t.sparse_dim == t.dim（即 t 不是混合张量）
        test_shape(3, 0, [0, 0, 2])
    # 定义测试函数 `test_coalesce_transpose_mm`，接受设备、数据类型和是否压缩参数
    def test_coalesce_transpose_mm(self, device, dtype, coalesced):
        # 定义测试形状的内部函数 `test_shape`，接受非零元素数、行数、列数和非零元素数
        def test_shape(di, dj, dk, nnz):
            # 生成稀疏张量 `x` 和随机张量 `y`
            x, _, _ = self._gen_sparse(2, nnz, [dj, di], dtype, device, coalesced)
            y = torch.randn(dj, dk, dtype=dtype, device=device)

            # 对 `x` 进行压缩操作，验证其已被压缩
            x_coalesced = x.coalesce()
            self.assertTrue(x_coalesced.is_coalesced())

            # 对 `x` 的转置进行操作，并验证转置是否保持了压缩性质（如果索引张量为空则是）
            x_coalesced_t = x_coalesced.t()
            # Transpose is `colasced`-preserving if the indices tensor is empty.
            self.assertEqual(x_coalesced_t.is_coalesced(), di * nnz == 0)

            # 执行稀疏矩阵与稠密矩阵的乘法运算
            res = torch.mm(x_coalesced_t, y)
            expected = torch.mm(self.safeToDense(x_coalesced_t), y)
            self.assertEqual(res, expected)

        # 调用 `test_shape` 函数，测试不同的形状和非零元素数量组合
        test_shape(10, 20, 30, 20)
        test_shape(0, 20, 30, 0)
        test_shape(10, 0, 30, 0)
        test_shape(10, 20, 0, 0)
        test_shape(10, 20, 0, 20)

    # 使用 TorchDynamo 时跳过当前测试，链接到相应的 GitHub 问题页面
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1166")
    # 指定数据类型为双精度或复数双精度
    @dtypes(torch.double, torch.cdouble)
    # 定义测试函数 `test_t_empty`，接受设备和数据类型参数
    def test_t_empty(self, device, dtype):
        # 定义对原地转置操作的测试函数 `test_in_place`，接受稀疏张量 `x` 作为参数
        def test_in_place(x):
            # 记录原始形状
            shape_original = x.shape
            # 执行原地转置操作
            x.t_()
            # 验证转置后的形状是否符合预期
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), x.size())
            # 验证转置后的索引张量是否为空
            self.assertEqual(0, x._indices().numel())
            # 验证转置后的值张量是否为空
            self.assertEqual(0, x._values().numel())
            # 验证转置后的稀疏维度是否为2
            self.assertEqual(x.sparse_dim(), 2)
            # 验证转置后的稠密维度是否为0
            self.assertEqual(x.dense_dim(), 0)

        # 定义对非原地转置操作的测试函数 `test_not_in_place`，接受稀疏张量 `x` 作为参数
        def test_not_in_place(x):
            # 记录原始形状
            shape_original = x.shape
            # 执行非原地转置操作
            y = x.t()
            # 验证转置后的形状是否符合预期
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), y.size())
            # 验证转置后的索引张量是否为空
            self.assertEqual(0, y._indices().numel())
            # 验证转置后的值张量是否为空
            self.assertEqual(0, y._values().numel())
            # 验证原始张量 `x` 的稀疏维度是否为2
            self.assertEqual(x.sparse_dim(), 2)
            # 验证原始张量 `x` 的稠密维度是否为0
            self.assertEqual(x.dense_dim(), 0)

        # 使用 `sparse_empty` 生成稀疏张量 `x` 进行测试
        x = self.sparse_empty(2, 3, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

        # 使用 `sparse_empty` 生成稀疏张量 `x` 进行测试
        x = self.sparse_empty(2, 0, dtype=dtype, device=device)
        test_in_place(x)
        test_not_in_place(x)

    # 定义测试函数 `test_add_zeros`，接受设备、数据类型和是否压缩参数
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_add_zeros(self, device, dtype, coalesced):
        # 定义测试形状的内部函数 `test_shape`，接受稀疏维度、非零元素数和张量尺寸作为参数
        def test_shape(sparse_dims, nnz, sizes):
            # 生成稀疏张量 `x` 和全零稀疏张量 `zeros`
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            zeros = torch.sparse_coo_tensor(sizes, device=x.device)
            # 执行零稀疏张量与 `x` 的加法操作
            r1 = zeros + x
            r2 = x + zeros
            # 验证加法结果与 `x` 是否相等
            self.assertEqual(r1, x)
            self.assertEqual(r2, x)

        # 测试不同稀疏维度、非零元素数和张量尺寸的组合
        test_shape(1, 20, [1])
        test_shape(4, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 0])

    # 指定数据类型为双精度或复数双精度
    @dtypes(torch.double, torch.cdouble)
    # 测试稀疏张量的加法和减法，验证非零元素数量不会无限增长 (gh-34964)
    def test_add_sub_nnz(self, device, dtype):
        # 创建一个在设备上指定类型的随机稀疏张量 x
        x = torch.randn(10, dtype=dtype, device=device).to_sparse()
        # 将 x 自身加到 x 上
        x.add_(x)
        # 再次将 x 自身加到 x 上
        x.add_(x)
        # 断言稀疏张量 x 的非零元素数量不超过 10
        self.assertLessEqual(x._nnz(), 10)

        # 将 2 倍的 x 减去 x
        x.sub_(2 * x)
        # 再次将 2 倍的 x 减去 x
        x.sub_(2 * x)
        # 断言稀疏张量 x 的非零元素数量不超过 10
        self.assertLessEqual(x._nnz(), 10)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 测试 torch.cat 方法的不同输入情况
    def test_cat(self, device, dtype, coalesced):
        # 定义一个函数，用于测试不同形状的输入张量的拼接操作
        def test_shapes(shapes, dim, fail_message=None):
            # 生成稀疏张量列表，根据提供的形状信息
            inputs = [self._gen_sparse(shape[0], shape[1], shape[2], dtype, device, coalesced)[0]
                      for shape in shapes]
            # 如果有指定错误消息，则测试应该抛出特定的运行时错误
            if fail_message:
                with self.assertRaisesRegex(RuntimeError, fail_message):
                    torch.cat(inputs, dim)
            else:
                # 否则，进行拼接操作，并验证结果与稠密张量拼接的结果一致
                result = torch.cat(inputs, dim)
                dense_result = torch.cat([t.to_dense() for t in inputs], dim)
                self.assertEqual(dense_result, result.to_dense())

        # 测试在第 1 维度上拼接不同形状的稀疏张量
        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], 1)

        # 测试形状不匹配的情况，期望抛出特定的运行时错误
        test_shapes([(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4])], 0,
                    "All tensors must have the same shape: \\[2, 3, 4].*\\[2, 1, 4]")

        # 测试混合稀疏和稠密张量的拼接
        test_shapes(
            [(2, 10, [2, 3, 4]), (2, 10, [2, 1, 4]), (2, 10, [2, 4, 4])], 1)

        # 测试在稠密维度上的拼接操作
        test_shapes([(2, 10, [2, 3, 4]), (2, 10, [2, 3, 7])], 2)

        # 其他形状相同的稀疏张量拼接操作
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 1)
        test_shapes([(1, 10, [2, 3, 4]), (1, 10, [2, 3, 4])], 2)

        # 测试形状不匹配的情况，期望抛出特定的运行时错误
        test_shapes([(2, 10, [2, 3, 4]), (3, 10, [2, 3, 4])], 0,
                    "All tensors must have the same.*2, 1, but tensor at position 1 has 3, 0.")

        # 在负数索引的维度上进行拼接
        test_shapes(
            [(3, 10, [2, 3, 4]), (3, 10, [2, 1, 4]), (3, 10, [2, 4, 4])], -2)

        # 测试稀疏张量与稠密张量混合拼接的情况，期望抛出特定的运行时错误
        sp = self._gen_sparse(3, 10, [2, 3, 4], dtype, device, coalesced)[0]
        dn = sp.to_dense()
        with self.assertRaisesRegex(RuntimeError,
                                    "Concatenating sparse tensors, but a dense tensor was found at position 1."):
            torch.cat((sp, dn))

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义测试函数 `test_unsqueeze`，接受参数 `self`, `device`, `dtype`, `coalesced`
    def test_unsqueeze(self, device, dtype, coalesced):
        # 定义内部测试形状的函数 `test_shape`，接受参数 `sparse_dims`, `nnz`, `sizes`, `unsqueeze_dim`, `fail_message=None`
        def test_shape(sparse_dims, nnz, sizes, unsqueeze_dim, fail_message=None):
            # 使用 `_gen_sparse` 方法生成稀疏张量 `x`，并解构返回的元组中的其他值
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            # 如果提供了 `fail_message`，验证在期望情况下是否抛出 `IndexError` 异常
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.unsqueeze(x, unsqueeze_dim)
            else:
                # 否则，对稀疏张量 `x` 在指定维度 `unsqueeze_dim` 上进行 unsqueeze 操作，并比较结果
                result = torch.unsqueeze(x, unsqueeze_dim)
                # 将稀疏张量转为密集张量，然后再在相同维度上进行 unsqueeze 操作
                dense_result = torch.unsqueeze(x.to_dense(), unsqueeze_dim)
                # 断言两个密集张量结果是否相等
                self.assertEqual(dense_result, result.to_dense())

        # 基本情况：在维度 0 上 unsqueeze
        test_shape(3, 10, [5, 7, 11], 0)

        # 混合稀疏/密集张量，在稀疏维度上 unsqueeze
        test_shape(3, 10, [5, 7, 11, 13, 17], 0)
        test_shape(3, 10, [5, 7, 11, 13, 17], 3)

        # 在密集维度上 unsqueeze
        test_shape(3, 10, [5, 7, 11, 13, 17], 4)
        test_shape(3, 10, [5, 7, 11, 13, 17], 5)

        # 包装的负数维度
        test_shape(3, 10, [5, 7, 11, 13, 17], -1)
        test_shape(3, 10, [5, 7, 11, 13, 17], -6)

        # 超出边界的维度测试
        test_shape(3, 10, [5, 7, 11, 13, 17], -7, "Dimension out of range")
        test_shape(3, 10, [5, 7, 11, 13, 17], 6, "Dimension out of range")

    # 使用装饰器 `coalescedonoff` 和 `dtypes`，测试函数 `test_select`，接受参数 `self`, `device`, `dtype`, `coalesced`
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_select(self, device, dtype, coalesced):
        # 定义内部测试形状的函数 `test_shape`，接受参数 `sparse_dims`, `nnz`, `sizes`, `select_dim`, `select_index`, `fail_message=None`
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            # 使用 `_gen_sparse` 方法生成稀疏张量 `x`，并解构返回的元组中的其他值
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            # 如果提供了 `fail_message`，验证在期望情况下是否抛出 `IndexError` 异常
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.select(x, select_dim, select_index)
            else:
                # 否则，对稀疏张量 `x` 在指定维度 `select_dim` 和索引 `select_index` 上进行 select 操作，并比较结果
                result = torch.select(x, select_dim, select_index)
                # 如果结果是稀疏张量，将其转换为密集张量
                if result.is_sparse:
                    result = result.to_dense()
                # 对稀疏张量 `x` 的密集形式在相同维度 `select_dim` 和索引 `select_index` 上进行 select 操作
                dense_result = torch.select(x.to_dense(), select_dim, select_index)
                # 断言两个密集张量结果是否相等
                self.assertEqual(dense_result, result)

        sizes = [5, 7, 11, 13, 17]
        
        # 混合稀疏/密集张量，在稀疏维度上 select，并期望结果为密集张量
        for i in range(sizes[0]):
            test_shape(1, 10, sizes, 0, i)
        # 在稀疏维度上 select，超出索引范围，期望抛出异常并验证异常消息
        test_shape(1, 10, sizes, 0, sizes[0] + 1, r'select[(][)][:] index \d out of range.*')

        # 混合稀疏/密集张量，在稀疏维度上 select，结果为稀疏张量
        for d in range(3):
            for i in range(sizes[d]):
                test_shape(3, 10, sizes, d, i)

        # 混合稀疏/密集张量，在密集维度上 select，结果为稀疏张量
        for d in range(1, 3):
            for i in range(sizes[d]):
                test_shape(1, 10, sizes, d, i)

    # 使用装饰器 `dtypes`，测试函数 `test_select`，接受参数为整数类型
    @dtypes(*integral_types())
    # 定义一个测试方法，用于测试不进行类型提升的情况下的选择操作
    def test_select_no_type_promotion(self, device, dtype):
        # 引用了 GitHub 上的 issue 链接，解释为什么需要进行这个测试
        # 创建一个稀疏张量的索引
        idx = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
        # 创建值为1的张量，指定数据类型为dtype
        val = torch.ones(6, dtype=dtype)
        # 使用索引和值创建稀疏 COO 格式的张量
        s = torch.sparse_coo_tensor(idx, val, size=(3, 3))

        # 对于每个张量 t，执行以下操作
        for t in (s, s * torch.tensor(0, dtype=dtype)):
            # 进行空值检查
            # 断言张量 t 的数据类型与其某个索引位置（这里是 t[2]）的数据类型相同
            self.assertEqual(t.dtype, t[2].dtype)
            # 断言张量 t 的数据类型与其某个具体元素（这里是 t[0, 1]）的数据类型相同
            self.assertEqual(t.dtype, t[0, 1].dtype)
            # 断言张量 t 的数据类型与其某个元素（这里是 t[0, 0]）的数据类型相同
            # 预期此处的数据类型不会因为求和操作而发生类型提升
            self.assertEqual(t.dtype, t[0, 0].dtype)
            # 断言张量 t 的数据类型与其某个元素（这里是 t[1, 1]）的数据类型相同
            self.assertEqual(t.dtype, t[1, 1].dtype)

    # 使用装饰器指定了稀疏张量的结合方式和数据类型，用于测试索引选择的方法
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select(self, device, dtype, coalesced):
        # 定义测试形状选择的方法
        def test_shape(sparse_dims, nnz, sizes, select_dim, select_index, fail_message=None):
            # 如果 select_index 是整数，则转换为列表
            if isinstance(select_index, int):
                select_index = [select_index]
            # 如果 select_index 是列表，则转换为 torch 的长整型张量
            if isinstance(select_index, list):
                select_index = torch.tensor(select_index, device=device, dtype=torch.long)
            # 使用 _gen_sparse 方法生成稀疏张量 x，以及其他返回值
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes, dtype, device, coalesced)
            # 如果有失败信息，则预期在索引选择时会抛出 IndexError 异常
            if fail_message:
                with self.assertRaisesRegex(IndexError, fail_message):
                    torch.index_select(x, select_dim, select_index)
            else:
                # 否则，执行索引选择操作，并比较结果
                result = torch.index_select(x, select_dim, select_index)
                # 如果结果是稀疏的，则转换为稠密张量
                if result.is_sparse:
                    result = result.to_dense()
                # 将稀疏张量 x 转换为稠密张量，并进行相同的索引选择操作
                dense_result = torch.index_select(x.to_dense(), select_dim, select_index)
                # 断言稠密结果与稀疏结果相等
                self.assertEqual(dense_result, result)

        # 指定测试的张量尺寸
        sizes = [5, 7, 11, 13, 17]
        # 对于每个尺寸的维度 d
        for d in range(len(sizes)):
            # 对于每个索引选择方式，执行测试形状选择方法
            for index in [0, sizes[d] - 1, [0, sizes[d] // 2, sizes[d] - 1]]:
                test_shape(1, 10, sizes, d, index)
                test_shape(len(sizes) // 2, 10, sizes, d, index)
                test_shape(len(sizes), 10, sizes, d, index)
    # 定义一个测试方法，用于测试索引选择的详尽情况
    def _test_index_select_exhaustive_index(self, sizes, dims, device, dtype, coalesced):
        # 创建一个指定大小和设备的张量
        t = make_tensor(sizes, dtype=dtype, device=device)
        # 将稠密张量转换为稀疏张量并且合并重复索引（如果需要）
        t_sparse = t.to_sparse().coalesce() if coalesced else t.to_sparse()
        # 生成一个小稀疏张量，以及其对应的稠密张量
        t_small_sparse, _, _ = self._gen_sparse(len(sizes), 2, sizes, dtype, device, coalesced)
        t_small = t_small_sparse.to_dense()
        # 对每个指定的维度进行迭代
        for d in dims:
            # NOTE: indices are negative
            # 生成一个范围为负的索引列表
            idx_dim_d_range = list(range(-sizes[d], 0))
            # 对于每个可能的索引长度进行迭代
            for idx_len in range(sizes[d], sizes[d] + 1):
                # creates all possible valid indices into dim d of lenght idx_len
                # 生成维度 d 中所有可能的有效索引组合
                for idx in itertools.product(*itertools.repeat(idx_dim_d_range, idx_len)):
                    # 创建一个张量索引，使用指定的数据类型和设备
                    t_idx = torch.tensor(idx, dtype=torch.long, device=device)

                    # NOTE: index_select for dense does not support negative indices,
                    # hence + sizes[d]. See https://github.com/pytorch/pytorch/issues/76347

                    # tests the nnz > sizes[d] branch
                    # 测试 nnz 大于 sizes[d] 的分支
                    dense_result = t.index_select(d, t_idx + sizes[d])
                    sparse_result = t_sparse.index_select(d, t_idx)
                    self.assertEqual(dense_result, sparse_result)

                    # tests the nnz <= sizes[d] branch
                    # 测试 nnz 小于等于 sizes[d] 的分支
                    small_dense_result = t_small.index_select(d, t_idx + sizes[d])
                    small_sparse_result = t_small_sparse.index_select(d, t_idx)
                    self.assertEqual(small_dense_result, small_sparse_result)

    # 使用装饰器设置稀疏张量的合并策略以及数据类型
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select_exhaustive_index_small(self, device, dtype, coalesced):
        # will trigger brute-force algo
        # 触发暴力算法
        self._test_index_select_exhaustive_index((3, 3, 4), range(3), device, dtype, coalesced)

    # 使用装饰器设置稀疏张量的合并策略以及数据类型
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_index_select_exhaustive_index_large(self, device, dtype, coalesced):
        # will trigger more sophisticated algos
        # 触发更复杂的算法
        self._test_index_select_exhaustive_index((100, 50, 3, 3), (2, 3), device, dtype, coalesced)

    # 使用装饰器设置数据类型
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义测试方法，用于测试空索引和非连续索引的情况
    def test_index_select_empty_and_non_contiguous_index(self, device, dtype, coalesced):
        # 创建一个空的索引张量
        idx_empty = torch.tensor([], dtype=torch.long, device=device)
        # 创建一个形状为 (5, 5) 的张量
        t = make_tensor((5, 5), dtype=dtype, device=device)
        # 使用空索引从张量中选择元素，得到稠密张量的结果
        res_dense = t.index_select(0, idx_empty)
        # 将张量转换为稀疏张量，再使用空索引选择元素，得到稀疏张量的结果
        res_sparse = t.to_sparse().index_select(0, idx_empty)
        # 断言稠密张量的结果与稀疏张量的结果相等
        self.assertEqual(res_dense, res_sparse)

        # 创建一个非连续的索引张量
        idx = torch.randint(low=0, high=5, size=(10, 2), device=device)[:, 0]

        def run_test(sizes):
            # 根据给定的尺寸创建张量
            t = make_tensor(sizes, dtype=dtype, device=device)
            # 使用非连续的索引从张量中选择元素，得到稠密张量的结果
            res_dense = t.index_select(0, idx)
            # 将张量转换为稀疏张量，再使用非连续的索引选择元素，得到稀疏张量的结果
            res_sparse = t.to_sparse().index_select(0, idx)
            # 断言稠密张量的结果与稀疏张量的结果相等
            self.assertEqual(res_dense, res_sparse)

            # 当 nnz <= size[d] 时
            # 生成一个稀疏张量，并使用非连续的索引选择元素，得到稀疏张量的结果
            t_small_sparse, _, _ = self._gen_sparse(len(sizes), 2, sizes, dtype, device, coalesced)
            res_sparse = t_small_sparse.index_select(0, idx)
            # 将稀疏张量转换为稠密张量，再使用非连续的索引选择元素，得到稠密张量的结果
            res_dense = t_small_sparse.to_dense().index_select(0, idx)
            # 断言稠密张量的结果与稀疏张量的结果相等
            self.assertEqual(res_dense, res_sparse)

        # brute-force 算法测试
        run_test((10, 10))
        # 更复杂的算法测试
        run_test((10, 100, 100))

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 测试索引选择的并行化功能
    def test_index_select_parallelization(self, device, dtype, coalesced):
        """
        Test with sizes that will trigger parallelization (i.e. with sizes
        that are >= at::internal::GRAIN_SIZE)
        """
        def run_test(nnz, size):
            # 生成一个稀疏张量，并转换为稠密张量
            t_sparse, _, _ = self._gen_sparse(1, nnz, (size,), dtype, device, coalesced)
            t_dense = t_sparse.to_dense()

            # 生成一个小型非连续索引，用于在稀疏张量和稠密张量中选择元素
            idx_small = torch.randint(size, (nnz // 2,), device=device)
            # 生成一个大型非连续索引，用于在稀疏张量和稠密张量中选择元素
            idx_large = torch.randint(size, (nnz * 2,), device=device)
            # 对于每一个索引，分别从稠密张量和稀疏张量中选择元素，并断言它们相等
            for idx in (idx_small, idx_large):
                res_dense = t_dense.index_select(0, idx)
                res_sparse = t_sparse.index_select(0, idx)
                self.assertEqual(res_dense, res_sparse)

        # GRAIN_SIZE = 32768 注意事项
        # 当 nnz <= size[d] 时的测试用例
        tlen = 70000  # > 2 * GRAIN_SIZE
        run_test(tlen, tlen)

        # 当 nnz > size[d] 时的测试用例
        run_test(tlen, tlen // 2)

    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义测试方法，用于测试稀疏矩阵与稠密矩阵之间的乘法操作
    def test_mm(self, device, dtype, coalesced):
        # 定义内部函数test_shape，用于测试不同形状的稀疏矩阵与稠密矩阵的乘法
        def test_shape(di, dj, dk, nnz):
            # 生成稀疏矩阵x，其形状为[di, dj]，包含nnz个非零元素，数据类型为dtype，存储在device上，根据coalesced指定是否压缩
            x, _, _ = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)
            # 生成形状为[di, dk]的稠密矩阵t，数据类型为dtype，存储在device上
            t = torch.randn(di, dk, dtype=dtype, device=device)
            # 生成形状为[dj, dk]的稠密矩阵y，数据类型为dtype，存储在device上
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            # 随机生成alpha和beta作为乘法的参数
            alpha = random.random()
            beta = random.random()

            # 使用torch.addmm进行矩阵乘法：t + alpha * x @ y + beta * t
            res = torch.addmm(t, x, y, beta=beta, alpha=alpha)
            # 使用稀疏矩阵x的稠密表示进行同样的乘法操作，作为预期结果
            expected = torch.addmm(t, self.safeToDense(x), y, beta=beta, alpha=alpha)
            # 断言结果res与预期expected相等
            self.assertEqual(res, expected)

            # 使用torch.addmm进行简单矩阵乘法：t + x @ y
            res = torch.addmm(t, x, y)
            # 使用稀疏矩阵x的稠密表示进行同样的乘法操作，作为预期结果
            expected = torch.addmm(t, self.safeToDense(x), y)
            # 断言结果res与预期expected相等
            self.assertEqual(res, expected)

            # 使用torch.mm进行矩阵乘法：x @ y
            res = torch.mm(x, y)
            # 使用稀疏矩阵x的稠密表示进行同样的乘法操作，作为预期结果
            expected = torch.mm(self.safeToDense(x), y)
            # 断言结果res与预期expected相等
            self.assertEqual(res, expected)

        # 分别以不同的参数调用test_shape函数进行测试
        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 200, 20)
        test_shape(64, 10000, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 100, 0)
        test_shape(10, 100, 0, 0)
        test_shape(10, 100, 0, 20)

    # 在Windows环境且使用CUDA时，跳过测试，因为不支持稀疏矩阵与稠密矩阵之间的乘法
    @unittest.skipIf(
        IS_WINDOWS and TEST_CUDA,
        "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1"
    )
    # 应用coalescedonoff装饰器，指定是否压缩稀疏矩阵
    @coalescedonoff
    # 设置数据类型为双精度浮点数
    @dtypes(torch.double)
    # 定义一个测试函数，用于测试稀疏矩阵与稠密矩阵的批量矩阵乘法（Batch Matrix Multiplication）
    def test_bmm(self, device, dtype, coalesced):
        # 定义一个内部函数，用于测试不同形状的矩阵乘法
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            # 初始化两个空列表，用于存储生成的稀疏矩阵和随机生成的稠密矩阵
            a_list = []
            b_list = []
            # 循环生成指定数量的稀疏矩阵和随机稠密矩阵
            for mat_idx in range(num_mats):
                # 调用自定义方法生成稀疏矩阵，并添加到列表中
                a_mat = self._gen_sparse(2, nnz, [dim_i, dim_j], dtype, device, coalesced)[0]
                # 生成随机稠密矩阵，并添加到列表中
                b_mat = torch.randn([dim_j, dim_k], dtype=dtype, device=device)
                a_list.append(a_mat)
                b_list.append(b_mat)

            # 将列表中的稀疏矩阵和稠密矩阵堆叠成一个新的张量
            a = torch.stack(a_list)
            b = torch.stack(b_list)
            # 使用批量矩阵乘法计算结果
            ab = a.bmm(b)

            # 逐个比较每个矩阵的结果与使用 mm() 方法计算的结果
            for mat_idx in range(num_mats):
                a_mat = a_list[mat_idx]
                b_mat = b_list[mat_idx]
                ab_mat_bmm = ab[mat_idx]
                ab_mat_mm = a_mat.mm(b_mat)
                # 使用断言检查 bmm() 计算的结果与 mm() 计算的结果是否一致
                self.assertEqual(ab_mat_bmm, ab_mat_mm)

        # 分别测试不同形状的矩阵乘法
        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

        # 创建一个具有随机数据的张量，并将其中某些矩阵设置为全零，然后转换为稀疏张量
        a = torch.rand([10, 23, 32], dtype=dtype, device=device)
        a[3] = torch.zeros(23, 32, dtype=dtype, device=device)
        a[6] = torch.zeros(23, 32, dtype=dtype, device=device)
        a = a.to_sparse()
        # 创建另一个具有随机数据的张量，并将其中某些矩阵设置为全零
        b = torch.rand([10, 32, 10], dtype=dtype, device=device)
        b[4] = torch.zeros(32, 10, dtype=dtype, device=device)
        b[6] = torch.zeros(32, 10, dtype=dtype, device=device)
        # 使用批量矩阵乘法计算结果
        ab = a.bmm(b)
        # 逐个比较每个矩阵的结果与使用 mm() 方法计算的结果
        for mat_idx in range(ab.size(0)):
            ab_mat = ab[mat_idx]
            ab_mat_check = a[mat_idx].mm(b[mat_idx])
            # 使用断言检查 bmm() 计算的结果与 mm() 计算的结果是否一致
            self.assertEqual(ab_mat, ab_mat_check)

        # 计算转置后的稀疏张量与转置后的稠密张量之间的批量矩阵乘法
        ab_traspose_check = b.transpose(1, 2).to_sparse().bmm(
            a.transpose(1, 2).to_dense()
        ).transpose(1, 2)
        # 使用断言检查 bmm() 计算的结果与预期结果是否一致
        self.assertEqual(ab, ab_traspose_check)

    # 标记该测试方法只在 CUDA 设备上运行
    @onlyCUDA
    # 标记该测试方法根据装饰器的参数在是否支持混合内存布局下运行
    @coalescedonoff
    # 标记该测试方法只在 torch.double 数据类型下运行
    @dtypes(torch.double)
    # 如果运行环境是 Windows，则跳过该测试方法
    @unittest.skipIf(
        IS_WINDOWS,
        "bmm sparse-dense CUDA is not yet supported in Windows, at least up to CUDA 10.1"
    )
    # 定义一个测试方法，用于测试确定性的 torch.bmm 操作在不同条件下的行为
    def test_bmm_deterministic(self, device, dtype, coalesced):
        # 定义一个内部方法，用于测试给定形状的稀疏矩阵乘法
        def test_shape(num_mats, dim_i, dim_j, dim_k, nnz):
            # 初始化空列表，用于存储生成的稀疏矩阵和随机矩阵
            a_list = []
            b_list = []
            # 对于每个矩阵索引，生成稀疏矩阵并添加到列表中
            for mat_idx in range(num_mats):
                # 使用 _gen_sparse 方法生成稀疏矩阵，并将其添加到 a_list
                a_list.append(self._gen_sparse(2, nnz, [dim_i, dim_j], dtype, device, coalesced)[0])
                # 生成随机矩阵并添加到 b_list
                b_list.append(torch.randn([dim_j, dim_k], dtype=dtype, device=device))

            # 将列表中的所有矩阵堆叠为一个 CUDA 张量 a
            a = torch.stack(a_list).cuda()
            # 将列表中的所有矩阵堆叠为一个 CUDA 张量 b
            b = torch.stack(b_list).cuda()
            
            # 使用 DeterministicGuard 确保当前的算法是确定性的
            with DeterministicGuard(torch.are_deterministic_algorithms_enabled()):
                # 关闭确定性算法，执行非确定性的 torch.bmm 操作
                torch.use_deterministic_algorithms(False)
                ab_nondeterministic = torch.bmm(a, b)
                # 打开确定性算法，执行确定性的 torch.bmm 操作
                torch.use_deterministic_algorithms(True)
                ab_deterministic = torch.bmm(a, b)
            
            # 计算两次结果之间的绝对差异和相对差异
            diff_abs = (ab_deterministic - ab_nondeterministic).abs()
            diff_rel = diff_abs / ab_deterministic.abs()
            diff_rel[torch.isnan(diff_rel)] = 0

            # 断言确定性和非确定性的结果要么完全相等，要么在相对差异小于0.001的范围内
            equal_abs_or_rel = diff_abs.eq(0).logical_or(diff_rel.lt(0.001))
            self.assertTrue(equal_abs_or_rel.all())

        # 分别测试多组不同形状和稀疏性的矩阵乘法
        test_shape(10, 10, 100, 99, 20)
        test_shape(10, 100, 1000, 200, 20)
        test_shape(10, 64, 10000, 300, 20)
        test_shape(10, 0, 100, 99, 0)
        test_shape(10, 10, 0, 100, 0)
        test_shape(10, 10, 100, 0, 0)
        test_shape(10, 10, 100, 0, 20)
        test_shape(10, 10, 100, 0, 20)

    # 仅在 CUDA 环境下执行的测试，用于验证在特定条件下的稀疏矩阵乘法是否引发错误
    @onlyCUDA
    @unittest.skipIf(
        not IS_WINDOWS or not TEST_WITH_ROCM,
        "this test ensures bmm sparse-dense CUDA gives an error when run on Windows with CUDA < 11.0"
    )
    @dtypes(torch.double)
    def test_bmm_windows_error(self, device, dtype):
        # 创建一个随机的稀疏张量 a，并将其移动到 CUDA 设备
        a = torch.rand(2, 2, 2, dtype=dtype).to_sparse().cuda()
        # 创建一个随机的密集张量 b，并将其移动到 CUDA 设备
        b = torch.rand(2, 2, 2, dtype=dtype).cuda()
        # 使用断言验证在 Windows 环境下、CUDA 版本小于 11.0 时，执行稀疏-密集乘法会抛出特定错误信息
        with self.assertRaisesRegex(
                RuntimeError,
                "bmm sparse-dense CUDA is not supported on Windows with cuda before 11.0"):
            ab = a.bmm(b)

    # 仅在 CPU 环境下执行的测试，用于测试在不同条件下的稀疏矩阵乘法
    @onlyCPU
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义测试函数 test_saddmm，接受设备、数据类型和稀疏矩阵是否合并作为参数
    def test_saddmm(self, device, dtype, coalesced):
        # 定义内部函数 test_shape，用于测试指定形状的稀疏矩阵乘法
        def test_shape(di, dj, dk, nnz):
            # 生成稀疏张量 x，其形状为 [di, dj]，非零元素个数为 nnz
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            # 生成稀疏张量 t，其形状为 [di, dk]，非零元素个数为 nnz
            t = self._gen_sparse(2, nnz, [di, dk], dtype, device, coalesced)[0]
            # 生成随机张量 y，形状为 [dj, dk]，数据类型为 dtype，设备为 device
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            # 生成随机数 alpha 和 beta
            alpha = random.random()
            beta = random.random()

            # 使用 torch.saddmm 执行稀疏矩阵乘法，带有权重参数 alpha 和 beta
            res = torch.saddmm(t, x, y, beta=beta, alpha=alpha)
            # 期望结果，使用 torch.addmm 计算密集矩阵乘法的结果
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            # 断言结果与期望值相等
            self.assertEqual(self.safeToDense(res), expected)

            # 使用默认的权重参数执行 torch.saddmm 稀疏矩阵乘法
            res = torch.saddmm(t, x, y)
            # 期望结果，使用 torch.addmm 计算密集矩阵乘法的结果
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            # 断言结果与期望值相等
            self.assertEqual(self.safeToDense(res), expected)

            # 使用 torch.smm 执行稀疏矩阵乘法，不带权重参数
            res = torch.smm(x, y)
            # 期望结果，使用 torch.mm 计算密集矩阵乘法的结果
            expected = torch.mm(self.safeToDense(x), y)
            # 断言结果与期望值相等
            self.assertEqual(self.safeToDense(res), expected)

        # 分别测试不同的稀疏矩阵形状和非零元素个数
        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

    # 设置测试仅在 CPU 下运行
    @onlyCPU
    # 设置测试根据 coalescedonoff 装饰器决定是否执行
    @coalescedonoff
    # 设置测试在 Torch Dynamo 环境下跳过执行
    @skipIfTorchDynamo("skip")
    # 设置测试使用双精度浮点数和复数双精度浮点数作为数据类型
    @dtypes(torch.double, torch.cdouble)
    # 定义测试函数，测试稀疏张量的 sspaddmm 方法
    def test_sspaddmm(self, device, dtype, coalesced):

        # 定义内部函数 test_shape，用于测试不同形状的稀疏张量
        def test_shape(di, dj, dk, nnz):
            # 生成稀疏张量 x，其形状为 [di, dj]，稀疏度为 nnz
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            # 生成稀疏张量 t，其形状为 [di, dk]，稀疏度为 nnz
            t = self._gen_sparse(2, nnz, [di, dk], dtype, device, coalesced)[0]
            # 生成随机张量 y，形状为 [dj, dk]
            y = torch.randn(dj, dk, dtype=dtype, device=device)
            # 随机生成 alpha 和 beta 值
            alpha = random.random()
            beta = random.random()

            # 使用稀疏张量 t 的 sspaddmm 方法计算结果 res
            res = t.sspaddmm(x, y, beta=beta, alpha=alpha)
            # 使用 torch.addmm 计算期望结果
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y, beta=beta, alpha=alpha)
            # 断言结果 res 与期望结果 expected 相等
            self.assertEqual(self.safeToDense(res), expected)

            # 测试不带 alpha 和 beta 参数的 sspaddmm 方法
            res = t.sspaddmm(x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        # 使用不同的形状和稀疏度测试 test_shape 函数
        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

        # 测试来自 GitHub 问题 https://github.com/pytorch/pytorch/issues/45113 的代码
        batch_size, input_size, hidden_size = 5, 3, 7

        # 创建非连续索引的稀疏张量 weight
        weight = torch.randn(hidden_size, input_size, dtype=dtype, device=device).to_sparse()
        self.assertTrue(weight.is_coalesced())
        non_contig_indices = weight.indices().mT.contiguous().mT
        weight = torch.sparse_coo_tensor(
            indices=non_contig_indices, values=weight.values(), size=weight.shape)
        weight._coalesced_(True)
        self.assertFalse(weight._indices().is_contiguous())

        # 创建稀疏张量 bias
        bias = torch.randn((hidden_size, 1), dtype=dtype, device=device).to_sparse()
        bias = torch.cat([bias] * batch_size, dim=1)

        # 如果 coalesced 为 True，则对 bias 进行合并操作
        if coalesced:
            bias = bias.coalesce()

        # 创建输入张量 x，形状为 [input_size, batch_size]
        x = torch.randn(input_size, batch_size, dtype=dtype, device=device)
        # 使用 sspaddmm 方法计算结果 res
        res = bias.sspaddmm(weight, x)

        # 计算真实结果 true_result，等同于 (bias.to_dense() + torch.matmul(weight.to_dense(), x)).to_sparse()
        true_result = (bias.to_dense() + torch.matmul(weight.to_dense(), x)).to_sparse()
        # 断言结果 res 与真实结果 true_result 相等
        self.assertEqual(self.safeToDense(res), self.safeToDense(true_result))

    # 使用装饰器设置函数的特定属性
    @coalescedonoff
    @precisionOverride({torch.bfloat16: 5e-2, torch.float16: 5e-2})
    @dtypes(torch.double, torch.cdouble, torch.bfloat16, torch.float16)
    def test_sparse_addmm(self, device, dtype, coalesced):
        # 如果数据类型为 torch.bfloat16 或 torch.float16 并且设备以 "cuda" 开头，
        # 则跳过测试，因为 addmm_sparse_cuda 尚未实现对 BFloat16 和 Half 数据类型的支持
        if (dtype is torch.bfloat16 or dtype is torch.float16) and device.startswith("cuda"):
            self.skipTest('addmm_sparse_cuda is not implemented for BFloat16 and Half')


        def test_shape(m, n, p, nnz, broadcast, alpha_beta=None):
            # 如果未提供 alpha_beta，则随机生成 alpha 和 beta
            if alpha_beta is None:
                alpha = random.random()
                beta = random.random()
            else:
                alpha, beta = alpha_beta
            # 根据 broadcast 标志创建 tensor D1
            if broadcast:
                D1 = make_tensor((), dtype=dtype, device=device, requires_grad=True)
            else:
                D1 = make_tensor([n, p], dtype=dtype, device=device, requires_grad=True)
            # 创建 tensor D2 和稀疏 tensor S
            D2 = make_tensor([m, p], dtype=dtype, device=device, requires_grad=True)
            S = self._gen_sparse(2, nnz, [n, m], dtype, device, coalesced)[0]
            # 将稀疏 tensor S 转换为 dense tensor，并设置 requires_grad=True
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            # 使用 torch.sparse.addmm 计算 Y
            Y = torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            # 使用 torch.addmm 计算 Y_dense
            Y_dense = torch.addmm(D1, S_dense, D2, beta=beta, alpha=alpha)
            # 断言稀疏和密集版本的计算结果 Y 和 Y_dense 相等
            self.assertEqual(Y, Y_dense)

            # 对于非 torch.double 和 torch.cdouble 数据类型，gradcheck 可能会失败
            if dtype not in {torch.double, torch.cdouble}:
                return

            # 定义用于 gradcheck 的函数 fn
            def fn(S, D1, D2, beta=beta, alpha=alpha):
                return torch.sparse.addmm(D1, S, D2, beta=beta, alpha=alpha)
            # 执行 gradcheck
            gradcheck(fn, (S, D1, D2), masked=True)

        # 分别测试不同形状和参数的情况
        test_shape(7, 8, 9, 20, False, None)
        test_shape(7, 8, 9, 20, True, None)
        test_shape(7, 8, 9, 20, False, (1, 0))
        test_shape(7, 8, 9, 20, True, (1, 0))
        test_shape(7, 8, 9, 20, False, (1, 1))
        test_shape(7, 8, 9, 20, True, (1, 1))

    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    def test_sparse_mm(self, device, dtype, coalesced):
        def test_shape(d1, d2, d3, nnz, transposed):
            # 如果 transposed 为 True，则创建转置后的 tensor D
            if transposed:
                D = torch.randn(d3, d2, dtype=dtype,
                                device=device).t_().requires_grad_(True)
            else:
                D = torch.randn(d2, d3, dtype=dtype, device=device).requires_grad_(True)
            # 创建稀疏 tensor S
            S = self._gen_sparse(2, nnz, [d1, d2], dtype, device, coalesced)[0]
            # 将稀疏 tensor S 转换为 dense tensor，并设置 requires_grad=True
            S_dense = S.to_dense().requires_grad_(True)
            S.requires_grad_(True)
            # 断言稀疏和密集版本的计算结果相等
            self.assertEqual(torch.sparse.mm(S, D), torch.mm(S_dense, D))

            # 定义用于 gradcheck 的函数 fn
            def fn(S, D):
                return torch.sparse.mm(S, D)
            # 执行 gradcheck
            gradcheck(fn, (S, D), masked=True)

        # 分别测试不同形状和参数的情况
        test_shape(7, 8, 9, 20, False)
        test_shape(7, 8, 9, 20, True)

    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    @gradcheck_semantics()
    def test_sparse_mul(self, device, dtype, coalesced, gradcheck):
        # 测试稀疏矩阵乘法，用于检查梯度
        a = torch.tensor([[0., 1]], dtype=dtype, device=device).to_sparse().requires_grad_(True)
        b = torch.tensor([[0., 1]], dtype=dtype, device=device).to_sparse().requires_grad_(True)
        gradcheck(lambda x, y: torch.sparse.sum(x * y).to_dense(masked_grad=gradcheck.masked), [a, b])

        def test_shape(sparse_dims, nnz, with_shape):
            # 生成稀疏矩阵并测试形状、梯度
            a = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[0].requires_grad_(True)
            b = self._gen_sparse(sparse_dims, nnz, with_shape, dtype, device, coalesced)[0].requires_grad_(True)

            self.assertEqual((a * b).to_dense(), a.to_dense() * b.to_dense(), masked=True)
            gradcheck(lambda x, y: (x * y).to_dense(), [a, b])
            # 处理 0 维度索引/值的问题
            gradcheck(lambda x, y: torch.sparse.sum(x * y).to_dense(), [a, b], masked=True)

        # TODO: Re-enable these
        # test_shape(2, 3, [2, 3, 4, 5])
        # test_shape(2, 3, [2, 2, 0])

    @coalescedonoff
    @dtypes(torch.double)
    def test_dsmm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            # 测试稀疏矩阵与密集矩阵乘法
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            y = self.randn(dj, dk, dtype=dtype, device=device)

            res = torch.dsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    @coalescedonoff
    @dtypes(torch.double)
    def test_hsmm(self, device, dtype, coalesced):
        def test_shape(di, dj, dk, nnz):
            # 测试分块稀疏矩阵与密集矩阵乘法
            x = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)[0]
            y = self.randn(dj, dk, dtype=dtype, device=device)

            res = torch.hsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    @coalescedonoff
    @dtypes(torch.double)
    # 定义测试方法 test_spadd，接受参数 self, device, dtype, coalesced
    def test_spadd(self, device, dtype, coalesced):

        # 定义内部方法 _test_spadd_shape，用于测试稀疏张量加法的形状
        def _test_spadd_shape(nnz, shape_i, shape_v=None):
            # 将输入形状组合成一个列表 shape
            shape = shape_i + (shape_v or [])
            # 使用 self._gen_sparse 方法生成稀疏张量 x，并返回 x, _, _
            x, _, _ = self._gen_sparse(len(shape_i), nnz, shape, dtype, device, coalesced)
            # 生成形状为 shape 的随机张量 y
            y = self.randn(*shape, dtype=dtype, device=device)
            # 生成一个随机标量 r
            r = random.random()

            # 使用 torch.add 计算 y + r * x 的结果
            res = torch.add(y, x, alpha=r)
            # 计算预期结果 y + r * self.safeToDense(x)
            expected = y + r * self.safeToDense(x)

            # 断言 res 和 expected 相等
            self.assertEqual(res, expected)

            # 创建一个非连续的密集张量 s，与 shape 有相同的维度和元素类型
            s = list(shape)
            s[0] = shape[-1]
            s[-1] = shape[0]
            # 在 s 上进行转置操作
            y = self.randn(*s, dtype=dtype, device=device)
            y.transpose_(0, len(s) - 1)
            # 生成一个新的随机标量 r
            r = random.random()

            # 再次使用 torch.add 计算 y + r * x 的结果
            res = torch.add(y, x, alpha=r)
            # 计算预期结果 y + r * self.safeToDense(x)
            expected = y + r * self.safeToDense(x)

            # 断言 res 和 expected 相等
            self.assertEqual(res, expected)

            # 重新生成稀疏张量 x, i, v，以及其非零元素数量 nnz
            x, i, v = self._gen_sparse(len(shape_i), nnz, shape, dtype, device, coalesced)
            nnz = i.size(1)

            # 创建一个非连续的稀疏索引张量 x_
            x_ = self.sparse_tensor(i[:, ::2], v[:(nnz + 1) // 2], x.shape, dtype=dtype, device=device)
            # 使用 torch.add 计算 y + r * x_ 的结果
            res = torch.add(y, x_, alpha=r)
            # 计算预期结果 y + r * self.safeToDense(x_)
            expected = y + r * self.safeToDense(x_)
            # 断言 res 和 expected 相等
            self.assertEqual(res, expected)

            # 创建一个非连续的稀疏值张量 x_
            x_ = self.sparse_tensor(i[:, :(nnz + 1) // 2], v[::2], x.shape, dtype=dtype, device=device)
            # 使用 torch.add 计算 y + r * x_ 的结果
            res = torch.add(y, x_, alpha=r)
            # 计算预期结果 y + r * self.safeToDense(x_)
            expected = y + r * self.safeToDense(x_)
            # 断言 res 和 expected 相等
            self.assertEqual(res, expected)

            # 创建一个同时具有非连续稀疏索引和值张量 x_
            x_ = self.sparse_tensor(i[:, 1::2], v[1::2], x.shape, dtype=dtype, device=device)
            # 使用 torch.add 计算 y + r * x_ 的结果
            res = torch.add(y, x_, alpha=r)
            # 计算预期结果 y + r * self.safeToDense(x_)
            expected = y + r * self.safeToDense(x_)
            # 断言 res 和 expected 相等
            self.assertEqual(res, expected)

        # 定义 _test_spadd 方法，用于测试不同形状的稀疏张量加法
        def _test_spadd():
            _test_spadd_shape(10, [5, 6])
            _test_spadd_shape(10, [10, 10, 10])
            _test_spadd_shape(10, [50, 30, 20])
            _test_spadd_shape(10, [5, 5, 5, 5, 5, 5])
            _test_spadd_shape(0, [0, 30, 20])
            _test_spadd_shape(0, [50, 0, 20])
            _test_spadd_shape(0, [50, 30, 0])

        # 定义 _test_spadd_hybrid 方法，用于测试混合形状的稀疏张量加法
        def _test_spadd_hybrid():
            _test_spadd_shape(10, [5, 6], [2, 3])
            _test_spadd_shape(10, [10, 10, 10], [3])
            _test_spadd_shape(10, [50, 30, 20], [2])
            _test_spadd_shape(10, [5, 5, 5, 5, 5, 5], [2])
            _test_spadd_shape(0, [0, 30, 20], [2, 0])
            _test_spadd_shape(0, [50, 0, 20], [2, 0])
            _test_spadd_shape(0, [50, 30, 0], [2, 0])
            _test_spadd_shape(10, [50, 30, 20], [2, 0])

        # 调用 _test_spadd 方法
        _test_spadd()
        # 调用 _test_spadd_hybrid 方法
        _test_spadd_hybrid()

    # 应用 coalescedonoff 和 dtypes 装饰器到当前测试方法
    @coalescedonoff
    @dtypes(torch.float)
    # 使用特定设备和数据类型生成稀疏张量，并返回其与另一个相同参数的稀疏张量的和（fp32精度）
    def test_sparse_add_out_bfloat16(self, device, dtype, coalesced):
        # 生成稀疏张量 x 和 y，每个张量包含3行5列的非零元素，总元素数为10个
        x, _, _ = self._gen_sparse(3, 5, 10, dtype, device, coalesced)
        y, _, _ = self._gen_sparse(3, 5, 10, dtype, device, coalesced)
        # 计算稀疏张量 x 和 y 的和，结果存储在 res_fp32 中（fp32精度）
        res_fp32 = torch.add(x, y)

        # 将稀疏张量 x 和 y 转换为 bfloat16 精度
        x = x.bfloat16()
        y = y.bfloat16()
        # 计算 bfloat16 精度下的稀疏张量 x 和 y 的和，存储在 res_bf16 中
        res_bf16 = torch.add(x, y)
        # 将结果 res_bf16 转换回 float32 精度，以便与参考结果进行比较
        res_bf16 = res_bf16.float()
        # 使用断言确保 fp32 精度的结果与 bfloat16 精度转换回 fp32 精度后的结果在给定的误差范围内相等
        self.assertEqual(res_fp32, res_bf16, atol=1e-2, rtol=0)

    # 根据装饰器设置，测试稀疏张量的范数计算
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    def test_norm(self, device, dtype, coalesced):
        # 定义测试函数，用于不同稀疏张量形状和非零元素数量的范数测试
        def test_shape(sparse_dims, nnz, with_size):
            # 生成稀疏张量 x，指定稀疏维度、非零元素数量和张量大小
            x, _, _ = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            # 对稀疏张量 x 进行合并操作，使其压缩存储
            y = x.coalesce()
            # 使用断言确保稀疏张量 x 的范数与其合并后的值的范数相等
            self.assertEqual(x.norm(), y._values().norm())

        # 不同测试用例，测试不同稀疏张量形状和非零元素数量的范数
        test_shape(3, 10, 100)
        test_shape(4, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(4, 0, [0, 0, 100, 5, 5, 5, 0])

        # 测试不支持的参数应该引发错误
        kwarg_error_pairs = [
            ({'keepdim': True},
             RuntimeError, r'norm_sparse currently does not support keepdim=True'),
            ({'dim': 0},
             RuntimeError, r'norm_sparse currently only supports full reductions'),
            ({'dtype': torch.double, 'p': 'fro'},
             ValueError, r'dtype argument is not supported in frobenius norm'),
            ({'dtype': torch.double, 'p': 0},
             RuntimeError, r"norm_sparse currently does not support 'dtype' argument")
        ]
        # 生成稀疏张量 x 以用于参数错误测试
        x = self._gen_sparse(3, 10, 100, dtype, device, coalesced)[0]
        # 遍历参数错误测试用例，确保使用相应参数调用 x.norm() 会引发预期的异常和消息
        for kwargs, err, msg in kwarg_error_pairs:
            with self.assertRaisesRegex(err, msg):
                x.norm(**kwargs)

    # 根据装饰器设置，跳过测试，如果测试与交叉引用有关则触发 cuda 设备错误
    @coalescedonoff
    @dtypes(torch.double)
    @unittest.skipIf(TEST_WITH_CROSSREF, "fallback triggers cuda device error")
    # 定义一个测试方法，用于测试稀疏张量的求和操作
    def test_sparse_sum(self, device, dtype, coalesced):

        # 定义内部函数，用于运行稀疏张量的求和测试
        def run_tests(S, td=None):
            # 将稀疏张量转化为密集张量，再将其梯度设置为可求导
            D = S.coalesce().to_dense().detach().requires_grad_(True)
            # 如果未指定维度 td，则分别计算稀疏张量 S 和密集张量 D 的总和，并进行断言比较
            if td is None:
                S_sum = torch.sparse.sum(S)
                D_sum = D.sum()
                self.assertEqual(S_sum.item(), D_sum.item())

                # 定义一个函数 fn，返回稀疏张量 S 的总和
                def fn(S):
                    return torch.sparse.sum(S)
                # 对函数 fn 进行梯度检查，传入参数 S，使用掩码计算梯度
                gradcheck(fn, (S,), masked=True)
            else:
                # 使用指定的维度 td 计算稀疏张量 S 和密集张量 D 的总和，并进行断言比较
                S_sum = torch.sparse.sum(S, td)
                D_sum = D.sum(td)
                self.assertEqual(S_sum.to_dense() if S_sum.is_sparse else S_sum, D_sum)

                # 定义一个函数 fn，返回在指定维度 td 上的稀疏张量 S 的总和，将结果转化为密集张量并使用掩码计算梯度
                def fn(S):
                    res = torch.sparse.sum(S, td)
                    return res.to_dense(masked_grad=True)
                # 对函数 fn 进行梯度检查，传入参数 S，使用掩码计算梯度
                gradcheck(fn, (S,), masked=True)

        # 设置非零元素个数和稀疏张量的维度
        nnz = 10
        sparse_dims = 2
        # 设置具有一个密集维度的测试尺寸
        with_size = [5, 5, 1, 4]  # use a dense dim = 1 to test for squeeze
        # 生成测试维度的组合列表
        test_dims = []
        for i in range(1, 5):
            test_dims += itertools.combinations(range(len(with_size)), i)

        # 创建稀疏张量 x，设置为稀疏张量后进行断言比较
        x = torch.tensor([[1., 0., 0., 1.],
                          [0., 1., 0., 0.],
                          [0., 1., 1., 0.],
                          [0., 1., 0., 2.]], dtype=dtype, device=device).to_sparse()
        self.assertEqual(torch.sparse.sum(x, dim=0), torch.sparse.sum(x, dim=-2))
        # 使用全局求和比较稀疏张量 x 的密集形式与指定维度的求和结果
        self.assertEqual(torch.sum(x.to_dense(), dim=0), torch.sparse.sum(x, dim=0).to_dense())

        # 生成稀疏张量 S，使用给定的稀疏维度、非零元素个数、尺寸、数据类型、设备和合并标志
        S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]

        # 测试超出范围的维度引发 IndexError 异常
        self.assertRaises(IndexError, lambda: torch.sparse.sum(S, 5))

        # 测试维度列表中包含重复的维度引发 RuntimeError 异常
        self.assertRaises(RuntimeError, lambda: torch.sparse.sum(S, [0, 0]))

        # 对空张量进行求和操作，比较稀疏形式和密集形式的结果
        empty_S = torch.sparse_coo_tensor(size=with_size, dtype=dtype, device=device)
        self.assertEqual(torch.sparse.sum(empty_S, [0]).to_dense(), torch.sum(empty_S.to_dense(), [0]))
        # 比较对空张量进行全局求和操作的结果与零张量的比较结果
        self.assertEqual(torch.sparse.sum(empty_S), torch.tensor(0, dtype=dtype, device=device))
        # 将空张量设置为可求导，对其进行求和并反向传播，比较梯度结果
        empty_S.requires_grad_(True)
        empty_S_sum = torch.sparse.sum(empty_S)
        empty_S_sum.backward()
        self.assertEqual(empty_S.grad.to_dense(), empty_S.clone().detach().to_dense())

        # 测试 values().sum() 方法
        S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
        run_tests(S.requires_grad_(True))

        # 遍历测试维度的各种组合，并对每种组合的稀疏张量进行求和测试
        for test_dim in test_dims:
            S = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            run_tests(S.requires_grad_(True), test_dim)
    # 定义测试函数，用于测试稀疏张量的基本操作和形状
    def _test_basic_ops_shape(self, nnz_x1, nnz_x2, shape_i, shape_v, dtype, device, coalesced):
        # 计算张量的完整形状
        shape = shape_i + (shape_v)
        # 生成两个稀疏张量 x1 和 x2，使用 _gen_sparse 方法
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape, dtype, device, coalesced)

        # 执行张量加法操作
        y1 = x1 + x2
        # 克隆 x1 并执行原地加法操作
        y2 = x1.clone()
        y2.add_(x2)
        # 计算期望的稠密张量加法结果
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        # 断言 y1 和 y2 的稠密表示与期望值相等
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # 执行张量减法操作，同上类似的模式
        y1 = x1 - x2
        y2 = x1.clone()
        y2.sub_(x2)
        expected = self.safeToDense(x1) - self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # 执行张量乘法操作，同上类似的模式
        y1 = x1 * x2
        y2 = x1.clone()
        y2.mul_(x2)
        expected = self.safeToDense(x1) * self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # 执行张量乘以标量操作，同上类似的模式
        y1 = x1 * 37.5
        y2 = x1.clone()
        y2.mul_(37.5)
        expected = self.safeToDense(x1) * 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # 执行张量除以标量操作，同上类似的模式
        y1 = x1 / 37.5
        y2 = x1.clone()
        y2.div_(37.5)
        expected = self.safeToDense(x1) / 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # 执行张量整除操作，同上类似的模式
        y1 = x1 // 37.5
        y2 = x1.clone()
        y2.floor_divide_(37.5)
        expected = self.safeToDense(x1) // 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # TODO: 添加支持原地操作的功能
        y1 = x1 ** 2
        y2 = x1.clone()
        y2 = y2.pow(2)
        expected = self.safeToDense(x1) ** 2
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # 将张量 y 清零，生成一个期望的全零张量
        y = x1.clone()
        y.zero_()
        expected = torch.zeros(x1.size(), dtype=dtype, device=device)
        self.assertEqual(self.safeToDense(y), expected)

        # 断言原张量是否被合并，然后执行合并操作并再次断言
        self.assertEqual(x1.is_coalesced(), coalesced)
        y = x1.coalesce()
        z = x1.coalesce()
        self.assertEqual(x1.is_coalesced(), coalesced)
        self.assertTrue(y.is_coalesced())
        y._values().add_(1)
        if not x1.is_coalesced():
            # 如果原张量未被合并，检查合并操作是否为非原地的
            self.assertEqual(z._values() + 1, y._values())
        else:
            # 如果原张量已被合并，检查合并操作是否为原地的
            self.assertEqual(z._values(), y._values())

    # 用装饰器设置 coalescedonoff 和 dtypes(torch.double)
    @coalescedonoff
    @dtypes(torch.double)
    # 定义一个测试函数，用于测试基本操作
    def test_basic_ops(self, device, dtype, coalesced):

        # 定义内部函数，用于执行基本操作测试
        def _test_basic_ops():
            # 调用 _test_basic_ops_shape 方法进行形状为 [5, 6] 的基本操作测试
            self._test_basic_ops_shape(9, 12, [5, 6], [], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10] 的基本操作测试
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [50, 30, 20] 的基本操作测试
            self._test_basic_ops_shape(9, 12, [50, 30, 20], [], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [5, 5, 5, 5, 5, 5] 的基本操作测试
            self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10]，但自身为空的基本操作测试
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [9, 0, 10, 10, 10] 的基本操作测试
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10]，但自身为空的基本操作测试
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 0]，但自身为空的基本操作测试
            self._test_basic_ops_shape(0, 0, [10, 10, 0], [], dtype, device, coalesced)

        # 定义内部函数，用于执行混合模式的基本操作测试
        def _test_basic_ops_hybrid():
            # 调用 _test_basic_ops_shape 方法进行形状为 [5, 6] 的混合模式基本操作测试
            self._test_basic_ops_shape(9, 12, [5, 6], [2, 3], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10] 的混合模式基本操作测试
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [3], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [50, 30, 20] 的混合模式基本操作测试
            self._test_basic_ops_shape(9, 12, [50, 30, 20], [2], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [5, 5, 5, 5, 5, 5] 的混合模式基本操作测试
            self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [2], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10]，但自身为空的混合模式基本操作测试
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [2], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [9, 0, 10, 10, 10] 的混合模式基本操作测试
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [2], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10]，但自身为空的混合模式基本操作测试
            self._test_basic_ops_shape(0, 0, [10, 10, 10], [2], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10]，但自身为空，稀疏维度为 [2, 0] 的混合模式基本操作测试
            self._test_basic_ops_shape(9, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 10]，但自身为空，稀疏维度为 [2, 0] 的混合模式基本操作测试
            self._test_basic_ops_shape(0, 12, [10, 10, 10], [2, 0], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [9, 0, 10, 10, 10]，但自身为空，稀疏维度为 [2, 0] 的混合模式基本操作测试
            self._test_basic_ops_shape(9, 0, [10, 10, 10], [2, 0], dtype, device, coalesced)
            # 调用 _test_basic_ops_shape 方法进行形状为 [10, 10, 0]，但自身为空，稀疏维度为 [2, 0] 的混合模式基本操作测试
            self._test_basic_ops_shape(0, 0, [10, 10, 0], [2, 0], dtype, device, coalesced)

        # 执行基本操作测试
        _test_basic_ops()
        # 执行混合模式基本操作测试
        _test_basic_ops_hybrid()

    # 使用 @dtypes 装饰器定义测试函数，用于测试稠密与稀疏不匹配情况
    @dtypes(torch.double, torch.cdouble)
    def test_add_dense_sparse_mismatch(self, device, dtype):
        # 定义测试形状函数，测试稠密大小、稀疏维度形状、稠密维度形状、稀疏大小
        def test_shape(dense_size, sparse_dims_shape, dense_dims_shape, sparse_size):
            # 创建全零稠密张量 x
            x = torch.zeros(dense_size, dtype=dtype, device=device)
            # 创建稀疏张量 sparse_y，稀疏维度形状为 sparse_dims_shape，稠密维度形状为 dense_dims_shape，稀疏大小为 sparse_size
            sparse_y = self.sparse_tensor(torch.zeros(sparse_dims_shape, dtype=torch.int64, device=device),
                                          torch.randn(dense_dims_shape, dtype=dtype, device=device),
                                          torch.Size(sparse_size))
            # 使用断言检查是否引发 RuntimeError，错误信息为 "add: expected 'self' and 'other' to have same size"
            with self.assertRaisesRegex(
                    RuntimeError,
                    "add: expected 'self' and 'other' to have same size"):
                # 进行稠密张量 x 与稀疏张量 sparse_y 的加法操作
                x + sparse_y

        # 调用 test_shape 函数进行测试，稠密大小为 [3, 4]，稀疏维度形状为 [1, 4]，稠密维度形状为 [4, 4, 4]，稀疏大小为 [3, 4, 4]
        test_shape([3, 4], [1, 4], [4, 4, 4], [3, 4, 4])
        # 调用 test_shape 函数进行测试，稠密大小为 [3, 4,
    # 使用装饰器定义一个测试函数，指定输入数据类型为双精度和复数双精度
    @dtypes(torch.double, torch.cdouble)
    # 定义一个测试函数，测试稀疏张量的非连续加法
    def test_add_noncontiguous(self, device, dtype):
        # 创建索引张量，指定设备
        indices = self.index_tensor([[1, 2], [0, 2]], device=device)
        # 创建值张量，使用指定的数据类型和设备，扩展为指定形状
        values = torch.tensor([1.], dtype=dtype, device=device).expand(2, 3, 4, 5)
        # 使用索引和值创建稀疏张量 x
        x = self.sparse_tensor(indices, values, dtype=dtype, device=device)
        # 断言稀疏张量 x 的值不是连续存储的
        assert not x._values().is_contiguous()
        # 对稀疏张量 x 进行加法操作，得到稀疏张量 y
        y = x + x
        # 计算预期的稠密张量结果，即稀疏张量 x 的稠密表示加上自身的稠密表示
        expected = self.safeToDense(x) + self.safeToDense(x)
        # 断言稀疏张量 y 的稠密表示与预期结果相等
        self.assertEqual(self.safeToDense(y), expected)

    # 定义一个测试函数，测试稀疏掩码的形状
    def _test_sparse_mask_shape(self, nnz_x1, nnz_x2, shape_i, shape_v, dtype, device, coalesced):
        # 根据输入参数生成稀疏张量 x1 和 x2
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape, dtype, device, coalesced)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape, dtype, device, coalesced)

        # 对 x1 和 x2 执行加法操作，得到稀疏张量 y1 和 y2
        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        # 计算预期的稠密张量结果，即稀疏张量 x1 和 x2 的稠密表示之和
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        # 断言稀疏张量 y1 和 y2 的稠密表示与预期结果相等
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

    # 使用装饰器定义一个测试函数，指定输入数据类型为双精度和复数双精度
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 使用装饰器定义一个测试函数，指定输入数据类型为双精度和复数双精度
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    @dtypes(torch.double, torch.cdouble)
    # 装饰器，跳过交叉引用的测试
    @skipIfCrossRef
    # 定义一个测试函数，测试稀疏掩码的反向传播
    def test_sparse_mask_backward(self, device, dtype):
        # 导入 product 和 repeat 函数
        from itertools import product, repeat

        # 定义张量的形状和稀疏维度
        shape = (5, 5)
        sparse_dims = len(shape)
        # 定义非零元素数量的范围
        nnzs = (0, 5, 15, 25)

        # 创建左右手边的稀疏数据张量
        lhs_data = torch.arange(1, 26, device=device).reshape(shape).to(dtype).to_sparse(sparse_dims)
        rhs_data = lhs_data.clone()

        # 遍历非零元素数量的范围
        for nnz in nnzs:
            # 遍历左右稀疏张量是否紧凑的组合
            for lhs_is_coalesced, rhs_is_coalesced in product(*repeat((True, False), 2)):
                # 克隆左手边稀疏张量，并设定其梯度需求为真
                lhs = torch.sparse_coo_tensor(
                    lhs_data._indices()[:, :nnz],
                    lhs_data._values()[:nnz],
                    lhs_data.shape
                ).clone()._coalesced_(lhs_is_coalesced).requires_grad_(True)

                # 克隆右手边稀疏张量
                rhs = torch.sparse_coo_tensor(
                    lhs_data._indices()[:, -nnz:],
                    lhs_data._values()[-nnz:],
                    lhs_data.shape
                ).clone()._coalesced_(rhs_is_coalesced)

                # 为测试掩码语义，确保 sparsity_pattern(lhs) == sparsity_pattern(lhs.grad)
                # 使用 lhs.sparse_mask(lhs_mask) 实现这一点
                lhs_mask = lhs.detach().clone()
                # 梯度检查，验证稀疏掩码对稠密张量的影响，masked_grad=True 表示使用掩码梯度
                gradcheck(lambda x: x.sparse_mask(lhs_mask).sparse_mask(rhs).to_dense(masked_grad=True), (lhs,), masked=True)
                # 梯度检查，验证稀疏掩码对稠密张量的影响，masked_grad=False 表示不使用掩码梯度
                gradcheck(lambda x: x.sparse_mask(rhs).to_dense(masked_grad=False), (lhs,), masked=False)

    # 使用装饰器定义一个测试函数，指定输入数据类型为双精度和复数双精度
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义测试函数 test_zeros，用于测试稀疏张量的生成与属性检查
    def test_zeros(self, device, dtype, coalesced):
        # 定义内部测试函数 _test_zeros，用于测试给定非零元素数量的稀疏张量
        def _test_zeros(nnzs, shape, out_shape_i, out_shape_v=None):
            # 根据输入参数组合确定输出形状
            out_shape = out_shape_i + (out_shape_v or [])
            # 遍历指定的非零元素数量
            for nnz in nnzs:
                # 使用自定义函数 _gen_sparse 生成稀疏张量 out
                out, _, _ = self._gen_sparse(len(out_shape_i), nnz, out_shape, dtype, device, coalesced)
                # 在给定形状下生成全零张量，使用指定的数据类型和设备
                torch.zeros(*shape, out=out, dtype=dtype, device=device)
                # 断言生成的张量 out 的大小与期望的形状相同
                self.assertEqual(tuple(out.size()), tuple(shape))
                # 断言生成的张量 out 的非零元素索引和值均为零
                self.assertTrue(out._indices().numel() == out._values().numel() == 0)
                # 断言生成的张量 out 的非零元素数量为零
                self.assertEqual(out._nnz(), 0)
                # 断言生成的张量 out 的稀疏维度等于指定的形状维度
                self.assertEqual(out.sparse_dim(), len(shape))
                # 断言生成的张量 out 的稠密维度为零
                self.assertEqual(out.dense_dim(), 0)

        # 定义形状测试函数 test_shape，用于测试不同输入维度组合下的稀疏张量生成
        def test_shape(i_shapes, v_shapes, shape, nnzs):
            # 遍历输入维度的可能组合
            for i_dim in range(1, len(i_shapes) + 1):
                # 遍历值维度的可能组合
                for v_dim in range(len(v_shapes) + 1):
                    # 调用 _test_zeros 测试函数，传入指定的形状和非零元素数量
                    _test_zeros(nnzs, shape, i_shapes[:i_dim], v_shapes[:v_dim])

        # 调用 test_shape 函数，测试多种形状和非零元素数量组合
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 4], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 0], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 0], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 0], [9, 12])

    # 应用装饰器 @coalescedonoff，指定测试函数在合并和非合并模式下执行
    @coalescedonoff
    # 应用装饰器 @dtypes，指定测试函数在不同数据类型下执行，包括双精度和复数双精度
    @dtypes(torch.double, torch.cdouble)
    def test_zeros_like(self, device, dtype, coalesced):
        # 定义测试函数，用于测试 torch.zeros_like 方法
        def _test_zeros_like(nnzs, template_shape_i, template_shape_v=None):
            # 内部函数，测试给定稀疏模板形状和非零元素个数的情况
            template_shape_v = template_shape_v or []  # 如果模板稀疏形状 v 未提供，则默认为空列表
            template_shape = template_shape_i + template_shape_v  # 合并模板的稀疏和密集形状
            for nnz in nnzs:
                # 生成稀疏张量 t，返回的 t 是一个稀疏张量对象
                t, _, _ = self._gen_sparse(len(template_shape_i), nnz, template_shape, dtype, device, coalesced)
                # 使用 torch.zeros_like 方法生成与 t 形状相同的零张量 res
                res = torch.zeros_like(t)
                # 断言 res 的形状与模板形状相同
                self.assertEqual(tuple(res.size()), tuple(template_shape))
                # 断言 res 的指标和值的元素数都为 0
                self.assertTrue(res._indices().numel() == res._values().numel() == 0)
                # 断言 res 的非零元素数为 0
                self.assertEqual(res._nnz(), 0)
                # 断言 res 的稀疏维度等于模板的稀疏形状维度
                self.assertEqual(res.sparse_dim(), len(template_shape_i))
                # 断言 res 的密集维度等于模板的密集形状维度
                self.assertEqual(res.dense_dim(), len(template_shape_v))

        # 定义测试形状函数，用于组合各种形状的测试用例
        def test_shape(i_shapes, v_shapes, nnzs):
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    _test_zeros_like(nnzs, i_shapes[:i_dim], v_shapes[:v_dim])

        # 执行测试用例
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])

        # 生成稀疏张量，并测试不同内存格式的情况
        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced)
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):
            # 断言在不支持的内存格式上会抛出 RuntimeError
            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.zeros_like(x, memory_format=mem_format)

            # 使用指定的内存布局生成零张量，并断言其布局正确
            result = torch.zeros_like(x, layout=torch.strided, memory_format=mem_format)
            self.assertTrue(result.layout == torch.strided)

        # 将稀疏张量转换为密集张量，生成相同形状的零张量，并断言布局为稀疏 COO 格式
        dense_tensor = sparse_tensor.to_dense()
        result = torch.zeros_like(dense_tensor, layout=torch.sparse_coo)
        self.assertEqual(dense_tensor.shape, result.shape)
        self.assertEqual(result.layout, torch.sparse_coo)

        # 生成稀疏 COO 张量，并断言其指标和值的形状与 result 相同
        sparse_zeros = torch.sparse_coo_tensor(dense_tensor.shape)
        self.assertEqual(result._indices().shape, sparse_zeros._indices().shape)
        self.assertEqual(result._values().shape, sparse_zeros._values().shape)
    def _assert_sparse_invars(self, t):
        # SparseTensor has the following invariants:
        # - sparse_dim + dense_dim = len(SparseTensor.shape)
        # - SparseTensor._indices().shape = (sparse_dim, nnz)
        # - SparseTensor._values().shape = (nnz, SparseTensor.shape[sparse_dim:])
        # 断言稀疏张量的不变性条件：
        # - 稀疏维度 + 密集维度 = SparseTensor.shape 的长度
        # - SparseTensor._indices().shape 应为 (sparse_dim, nnz)
        # - SparseTensor._values().shape 应为 (nnz, SparseTensor.shape[sparse_dim:])
        self.assertEqual(t.sparse_dim() + t.dense_dim(), len(t.shape))
        self.assertEqual(tuple(t._indices().shape), (t.sparse_dim(), t._nnz()))
        self.assertEqual(tuple(t._values().shape), (t._nnz(), ) + t.shape[t.sparse_dim():])

    def _test_empty_like(self, sparse_tensor, dtype, device, coalesced):
        # 测试 torch.empty_like 函数
        result = torch.empty_like(sparse_tensor)
        # 断言结果是稀疏的
        self.assertTrue(result.is_sparse)
        # 断言稀疏张量的不变性条件
        self._assert_sparse_invars(result)
        # 断言结果的形状与输入稀疏张量相同
        self.assertEqual(result.shape, sparse_tensor.shape)
        # 断言结果的数据类型与输入稀疏张量相同
        self.assertEqual(result.dtype, sparse_tensor.dtype)
        # 断言结果的设备与输入稀疏张量相同
        self.assertEqual(result.device, sparse_tensor.device)
        # 断言结果的稀疏维度与输入稀疏张量相同
        self.assertEqual(result.sparse_dim(), sparse_tensor.sparse_dim())
        # 断言结果的密集维度与输入稀疏张量相同
        self.assertEqual(result.dense_dim(), sparse_tensor.dense_dim())

        # 生成一个稀疏张量及相关数据
        sparse_tensor, _, _ = self._gen_sparse(len([2, 3]), 9, [2, 3] + [5, 6], dtype, device, coalesced)
        # 数据数组
        data = (sparse_tensor, sparse_tensor, sparse_tensor, sparse_tensor.unsqueeze(0))
        # 内存格式数组
        mem_formats = [torch.channels_last, torch.contiguous_format, torch.preserve_format, torch.channels_last_3d]
        for x, mem_format in zip(data, mem_formats):
            # 使用不支持的内存格式进行测试时应抛出异常
            with self.assertRaisesRegex(RuntimeError, "memory format option is only supported by strided tensors"):
                result = torch.empty_like(x, memory_format=mem_format)

            # 使用指定布局和内存格式创建结果张量
            result = torch.empty_like(x, layout=torch.strided, memory_format=mem_format)
            # 断言结果张量的布局为 torch.strided
            self.assertTrue(result.layout == torch.strided)

        # 尝试在稀疏张量上使用不支持的布局时应抛出异常
        with self.assertRaisesRegex(
            RuntimeError, r"Could not run 'aten::empty_strided' with arguments from the 'Sparse(CPU|CUDA)' backend"
        ):
            # 将稀疏张量转换为密集张量，然后使用指定的布局
            dense_tensor = sparse_tensor.to_dense()
            result = torch.empty_like(dense_tensor, layout=torch.sparse_coo)

    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义一个测试方法，用于测试空的稀疏张量的行为
    def test_empty_like(self, device, dtype, coalesced):
        # 测试用例来源于 https://github.com/pytorch/pytorch/issues/43699

        # 如果需要对齐稀疏张量进行测试
        if coalesced:
            # 创建一个稀疏的 COO 张量，然后进行压缩操作
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),  # 定义索引
                values=torch.tensor([3.0, -4.0, 5.0]),  # 定义值
                size=[3, ],  # 定义张量的大小
                dtype=dtype,  # 定义数据类型
                device=device  # 指定设备
            ).coalesce()  # 执行压缩操作
            # 调用内部方法以测试空稀疏张量的行为
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

            # 对混合稀疏输入进行测试
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),  # 定义索引
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),  # 定义值
                size=[4, 5, 2],  # 定义张量的大小
                dtype=dtype,  # 定义数据类型
                device=device  # 指定设备
            ).coalesce()  # 执行压缩操作
            # 调用内部方法以测试空稀疏张量的行为
            self._test_empty_like(input_coalesced, dtype, device, coalesced)

        # 如果不需要对齐稀疏张量进行测试
        if not coalesced:
            # 测试未对齐的输入稀疏张量
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),  # 定义索引
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),  # 定义值
                size=[3, ],  # 定义张量的大小
                dtype=dtype,  # 定义数据类型
                device=device  # 指定设备
            )
            # 调用内部方法以测试空稀疏张量的行为
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)

            # 对空的稀疏张量进行测试
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),  # 定义索引，创建空的张量
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),  # 定义值，创建空的张量
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],  # 定义张量的大小，创建空的张量
                dtype=dtype,  # 定义数据类型
                device=device  # 指定设备
            )
            # 调用内部方法以测试空稀疏张量的行为
            self._test_empty_like(input_uncoalesced, dtype, device, coalesced)
    # 测试稀疏张量的窄化操作
    def test_narrow(self, device, dtype, coalesced):
        # 定义张量的形状
        shape = [3, 3, 4, 2]
        # 使用辅助函数生成稀疏张量及其相关的稠密张量和索引
        input, _, _ = self._gen_sparse(4, 19, shape, dtype, device, coalesced)
        # 对所有可能的窄化参数组合进行测试
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(input, narrow_args)

        # 测试异常情况：维度小于0时的窄化操作
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(-1, 0, 3))  # dim < 0
        # 测试异常情况：维度大于输入张量的维度时的窄化操作
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(10, 0, 3))  # dim > input.dim()
        # 测试异常情况：起始位置大于维度的大小时的窄化操作
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, shape[0] + 1, 3))  # start > size of dim
        # 测试异常情况：起始位置加长度大于维度的大小时的窄化操作
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, 2, shape[0]))  # start+length > size of dim

        # 使用辅助函数生成另一个带稠密部分的稀疏张量及其相关的稠密张量和索引
        with_dense, _, _ = self._gen_sparse(2, 7, shape, dtype, device, coalesced)
        # 对所有可能的窄化参数组合进行测试
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(with_dense, narrow_args)

        # 测试异常情况：维度大于稀疏部分加稠密部分的维度时的窄化操作
        self.assertRaises(RuntimeError, lambda: with_dense.narrow_copy(10, 0, 3))  # dim > sparseDim + denseDim

    # 测试稀疏张量的 log1p() 方法及相关操作
    def _test_log1p_tensor(self, sparse_tensor, coalesced):
        # 判断数据类型是否为整数类型
        def is_integral(dtype):
            return dtype in integral_types()

        # 将稀疏张量转换为稠密张量
        dense_tensor = sparse_tensor.to_dense()
        # 计算期望输出：稠密张量的 log1p() 结果
        expected_output = dense_tensor.log1p()
        # 判断当前数据类型是否为整数类型
        is_integral_dtype = is_integral(sparse_tensor.dtype)
        # 验证稀疏张量的 log1p() 方法是否得到了期望的稠密张量输出
        self.assertEqual(expected_output, sparse_tensor.log1p().to_dense())
        # 如果数据类型为整数类型，验证稀疏张量在压缩后调用 log1p_() 方法是否会引发错误
        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                sparse_tensor.coalesce().log1p_()
        else:
            # 否则，验证稀疏张量在压缩后调用 log1p_() 方法是否得到了期望的稠密张量输出
            self.assertEqual(expected_output, sparse_tensor.coalesce().log1p_().to_dense())

        # 如果稀疏张量未压缩，则测试在未压缩输入上执行原地操作是否会引发错误
        if not coalesced:
            with self.assertRaisesRegex(RuntimeError, "log1p_ requires coalesced input"):
                sparse_tensor.log1p_()

        # 如果数据类型为整数类型，则测试在此类张量上调用 requires_grad_() 方法是否会引发错误
        if is_integral_dtype:
            with self.assertRaisesRegex(RuntimeError, "only Tensors of floating point dtype can require gradients"):
                sparse_tensor.requires_grad_()

    # 应用装饰器设置是否启用稀疏张量压缩，并使用所有数据类型装饰器
    @coalescedonoff
    @dtypes(*all_types())
    # 定义测试函数，用于测试稀疏张量的 log1p 函数
    def test_log1p(self, device, dtype, coalesced):
        # 如果输入是已经合并的稀疏张量
        if coalesced:
            # 创建稀疏的 COO 张量，使用指定的索引和数值，进行合并操作
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([3.0, 4.0, 5.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            ).coalesce()
            # 调用内部函数，测试 log1p 函数在合并的稀疏张量上的表现
            self._test_log1p_tensor(input_coalesced, coalesced)

            # 对于混合稀疏输入的情况
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
                size=[4, 5, 2],
                device=device,
                dtype=dtype
            ).coalesce()
            # 再次调用内部函数，测试 log1p 函数在混合稀疏张量上的表现
            self._test_log1p_tensor(input_coalesced, coalesced)

        # 如果输入不是合并的稀疏张量
        if not coalesced:
            # 测试未合并输入的情况
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, 3.0, 4.0, 1.0, 1.0, 1.0]),
                size=[3, ],
                device=device,
                dtype=dtype
            )
            # 调用内部函数，测试 log1p 函数在未合并的稀疏张量上的表现
            self._test_log1p_tensor(input_uncoalesced, coalesced)

            # 测试空稀疏张量的情况
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                device=device,
                dtype=dtype
            )
            # 强制设定稀疏张量处于未合并状态
            input_uncoalesced._coalesced_(False)
            # 再次调用内部函数，测试 log1p 函数在空稀疏张量上的表现
            self._test_log1p_tensor(input_uncoalesced, coalesced)

    # 测试稀疏张量的负数操作
    def _test_neg_negative(self, sparse_tensor):
        # 将稀疏张量转换为密集张量
        dense_tensor = sparse_tensor.to_dense()
        # 计算期望的输出，即密集张量的负数
        expected_output = dense_tensor.neg()

        # 定义操作集合，包括 torch.neg, torch.Tensor.neg, torch.Tensor.neg_, torch.negative 等
        ops = (
            torch.neg, torch.Tensor.neg, torch.Tensor.neg_,
            torch.negative, torch.Tensor.negative, torch.Tensor.negative_,
            operator.neg
        )
        # 遍历操作集合，逐个进行测试
        for op in ops:
            # 复制稀疏张量，因为每次操作后张量状态可能改变
            sparse_tensor_copy = sparse_tensor.clone()
            # 断言操作后的结果与期望输出一致
            self.assertEqual(expected_output, op(sparse_tensor_copy).to_dense())

            # 对于 torch.neg 和 torch.negative 操作，还需要检查输出张量是否正确
            if op in (torch.neg, torch.negative):
                # 创建一个与稀疏张量形状相同的零张量
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                # 在指定的输出张量上执行操作
                op(sparse_tensor, out=sparse_tensor_out)
                # 断言操作后的输出与期望输出一致
                self.assertEqual(expected_output, sparse_tensor_out.to_dense())

    # 装饰器函数，用于控制稀疏张量的合并状态和数据类型
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义测试方法，测试在稀疏张量上执行负值负化操作
    def test_neg_negative(self, device, dtype, coalesced):

        # 如果输入已经合并
        if coalesced:
            # 创建稀疏 COO 张量，指定索引和数值，然后进行合并操作
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2]]),
                values=torch.tensor([3.0, -4.0, 5.0]),
                size=[3, ],
                dtype=dtype,
                device=device
            ).coalesce()
            # 在合并后的稀疏张量上执行负值负化测试
            self._test_neg_negative(input_coalesced)

            # 混合稀疏输入
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-1.0, 3.0], [-5.0, 7.0]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            # 在混合稀疏输入上执行负值负化测试
            self._test_neg_negative(input_coalesced)

        # 如果输入未合并
        if not coalesced:
            # 测试未合并的输入
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([2.0, -3.0, -4.0, 1.0, -1.0, 1.5]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            # 在未合并的稀疏张量上执行负值负化测试
            self._test_neg_negative(input_uncoalesced)

            # 测试空的稀疏张量
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            # 在空的稀疏张量上执行负值负化测试
            self._test_neg_negative(input_uncoalesced)
    # 定义一个测试函数，用于测试稀疏张量 sparse_tensor 的 arcsin 操作
    def _test_asin_arcsin(self, sparse_tensor, coalesced):
        # 内部函数，判断给定的数据类型是否为整数类型
        def is_integral(dtype):
            return dtype in integral_types()
        
        # 判断 sparse_tensor 的数据类型是否为整数类型
        is_integral_dtype = is_integral(sparse_tensor.dtype)

        # 将稀疏张量 sparse_tensor 转换为密集张量 dense_tensor
        dense_tensor = sparse_tensor.to_dense()
        # 计算密集张量 dense_tensor 的 arcsin 函数值，作为期望输出
        expected_output = dense_tensor.asin()

        # 定义四个操作函数列表 ops，分别为 torch.asin, torch.Tensor.asin,
        # torch.arcsin, torch.Tensor.arcsin
        ops = (
            torch.asin, torch.Tensor.asin,
            torch.arcsin, torch.Tensor.arcsin,
        )
        # 遍历操作函数列表 ops
        for op in ops:
            # 断言 sparse_tensor 经过操作 op 的结果与期望输出相同，并且转换为密集张量后也相同
            self.assertEqual(expected_output, op(sparse_tensor).to_dense())
            # 若当前操作函数为 torch.asin 或 torch.arcsin
            if op in (torch.asin, torch.arcsin):
                # 创建一个与 sparse_tensor 同形状的零张量 sparse_tensor_out
                sparse_tensor_out = torch.zeros_like(sparse_tensor)
                # 如果数据类型不是整数类型，则进行就地操作，将结果存入 sparse_tensor_out
                if not is_integral_dtype:
                    op(sparse_tensor, out=sparse_tensor_out)
                    # 断言就地操作后的 sparse_tensor_out 的结果与期望输出相同，转换为密集张量后也相同
                    self.assertEqual(expected_output, sparse_tensor_out.to_dense())
                else:
                    # 如果数据类型是整数类型，则断言执行操作会引发运行时错误，指明结果类型无法转换
                    with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                        op(sparse_tensor, out=sparse_tensor_out)

        # 遍历操作函数列表 (torch.Tensor.asin_, torch.Tensor.arcsin_)
        for op in (torch.Tensor.asin_, torch.Tensor.arcsin_):
            # 如果数据类型是整数类型
            if is_integral_dtype:
                # 断言执行操作会引发运行时错误，指明结果类型无法转换
                with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                    op(sparse_tensor.clone().coalesce()).to_dense()
            else:
                # 断言操作的结果与期望输出相同，将稀疏张量的克隆进行 coalesce 操作后再转换为密集张量
                self.assertEqual(expected_output, op(sparse_tensor.clone().coalesce()).to_dense())

            # 如果输入稀疏张量未经过 coalesce 操作
            if not coalesced:
                # 断言执行就地操作会引发运行时错误，指明 asin_ 要求输入必须是 coalesced（已合并）的
                with self.assertRaisesRegex(RuntimeError, "asin_ requires coalesced input"):
                    op(sparse_tensor)

    # 装饰器：用于设置是否进行 coalesce 操作，并设置数据类型为所有类型
    @coalescedonoff
    @dtypes(*all_types())
    # 测试asin和arcsin函数，使用稀疏张量作为输入
    def test_asin_arcsin(self, device, dtype, coalesced):
        # 如果输入已经被合并
        if coalesced:
            # 创建一个稀疏的COO张量，其值为0.5, -0.5, 0.7, -0.7，形状为[4,]，指定设备和数据类型
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1, 2, 3]]),
                values=torch.tensor([0.5, -0.5, 0.7, -0.7]),
                size=[4, ],
                dtype=dtype,
                device=device
            ).coalesce()  # 合并稀疏张量
            # 调用测试函数，传入合并后的稀疏张量和合并标志
            self._test_asin_arcsin(input_coalesced, coalesced)

            # 混合稀疏输入
            input_coalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[1, 3], [2, 4]]),
                values=torch.tensor([[-0.1, 0.24], [-0.44, 0.1]]),
                size=[4, 5, 2],
                dtype=dtype,
                device=device
            ).coalesce()
            # 调用测试函数，传入合并后的混合稀疏张量和合并标志
            self._test_asin_arcsin(input_coalesced, coalesced)

        # 如果输入未被合并
        if not coalesced:
            # 测试未合并的稀疏输入
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.tensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0),
                values=torch.tensor([0.3, -0.3, -0.4, 0.3, -0.5, 0.15]),
                size=[3, ],
                dtype=dtype,
                device=device
            )
            # 调用测试函数，传入未合并的稀疏张量和合并标志
            self._test_asin_arcsin(input_uncoalesced, coalesced)

            # 测试空稀疏张量
            input_uncoalesced = torch.sparse_coo_tensor(
                indices=torch.zeros([2, 0]),
                values=torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
                size=[0, 0, 5, 5, 5, 5, 5, 5, 0],
                dtype=dtype,
                device=device
            )
            # 强制设定稀疏张量为未合并状态
            input_uncoalesced._coalesced_(False)
            # 调用测试函数，传入强制未合并的空稀疏张量和合并标志
            self._test_asin_arcsin(input_uncoalesced, coalesced)

    # 装饰器，控制coalesced参数为on或off
    @coalescedonoff
    # 装饰器，指定数据类型为torch.double
    @dtypes(torch.double)
    # 测试矩阵向量乘法函数，验证结果是否正确
    def test_mv(self, device, dtype, coalesced):
        # 定义形状测试函数
        def test_shape(di, dj, dk, nnz):
            # 生成稀疏张量，其形状为[di, dj]，非零元素数量为nnz，指定数据类型和设备，以及是否合并
            x, _, _ = self._gen_sparse(2, nnz, [di, dj], dtype, device, coalesced)
            # 生成随机张量t，数据类型为dtype，指定设备
            t = torch.randn(dk, dtype=dtype, device=device)

            # 执行矩阵向量乘法操作
            res = x.matmul(t)
            # 计算期望结果（将稀疏张量转换为密集张量后进行矩阵乘法）
            expected = self.safeToDense(x).matmul(t)
            # 断言结果与期望相等
            self.assertEqual(res, expected)

        # 不同形状和非零元素数量的测试用例
        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 1000, 20)
        test_shape(64, 10000, 10000, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 0, 0)
        test_shape(10, 100, 100, 0)
        test_shape(10, 100, 100, 20)

        # 预期引发运行时错误，检查异常消息是否符合预期
        with self.assertRaisesRegex(RuntimeError, r"mv: expected self\.size\(-1\) == vec\.size\(-1\)"):
            test_shape(10, 100, 10, 20)

        # 预期引发运行时错误，检查异常消息是否符合预期
        with self.assertRaisesRegex(RuntimeError, "mv: two tensor dim should be 2 and 1"):
            # 生成两个稀疏张量，形状为[10, 100]，非零元素数量为20，指定数据类型和设备，以及是否合并
            x, _, _ = self._gen_sparse(2, 20, [10, 100], dtype, device, coalesced)
            y, _, _ = self._gen_sparse(2, 20, [10, 100], dtype, device, coalesced)
            # 执行向量乘法操作
            res = x.mv(y)

    # 装饰器，指定数据类型为所有浮点数和复数类型的组合
    @dtypes(*floating_and_complex_types())
    # 在稀疏张量上测试加法并验证是否已稀疏化
    def test_sparse_add_coalesce(self, device, dtype):
        # 创建索引张量
        i = self.index_tensor([[1, 2, 1]], device=device)
        # 创建值张量
        v = torch.tensor([3, 4, 5], dtype=dtype, device=device)
        # 创建稀疏张量x
        x = self.sparse_tensor(i, v, torch.Size([3]))
        # 创建另一个相同的稀疏张量y
        y = self.sparse_tensor(i, v, torch.Size([3]))
        # 对x和y进行加法操作，得到稀疏张量z
        z = x + y

        # 断言：z的索引元素数量不等于2且已经稀疏化
        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

        # 创建具有空值张量的稀疏张量x和y
        i = self.index_tensor([[1, 2, 1]], device=device)
        v = torch.empty([3, 0], dtype=dtype, device=device)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]))
        y = self.sparse_tensor(i, v, torch.Size([3, 0]))
        z = x + y

        # 断言：z的索引元素数量不等于2且已经稀疏化
        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

    # 只在CUDA设备上运行的测试
    @onlyCUDA
    def test_storage_not_null(self, device):
        # 创建一个空的COO格式稀疏张量x，数据类型为torch.float32
        x = torch.sparse_coo_tensor((2,), dtype=torch.float32, device=device)
        # 断言：稀疏张量x的设备ID不等于-1（即存在有效的设备）
        self.assertNotEqual(x.get_device(), -1)

        # 创建一个形状为(2, 0)的空COO格式稀疏张量x，数据类型为torch.float32
        x = torch.sparse_coo_tensor((2, 0), dtype=torch.float32, device=device)
        # 断言：稀疏张量x的设备ID不等于-1（即存在有效的设备）
        self.assertNotEqual(x.get_device(), -1)

    # 只在CUDA设备上且设备数至少为2时运行的测试
    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_same_gpu(self, devices):
        # 检查稀疏张量x是否在指定的设备ID上
        def check_device(x, device_id):
            self.assertEqual(x.get_device(), device_id)
            self.assertEqual(x._values().get_device(), device_id)
            self.assertEqual(x._indices().get_device(), device_id)

        # 获取前两个设备的ID
        dev1, dev2 = devices[0], devices[1]

        # 创建一个指定设备dev2上的稀疏张量x
        i = self.index_tensor([[2]], device=dev2)
        v = torch.tensor([5], device=dev2)
        x = self.sparse_tensor(i, v, torch.Size([3]), device=1)
        check_device(x, 1)

        # 创建一个指定设备dev2上的空值稀疏张量x
        i = self.index_tensor([[2]], device=dev2)
        v = torch.empty(1, 0, device=dev2)
        x = self.sparse_tensor(i, v, torch.Size([3, 0]), device=1)
        check_device(x, 1)

        # 创建一个在设备1上的空COO格式稀疏张量x
        x = self.sparse_empty(3, device=1)
        check_device(x, 1)

        # 创建一个在设备1上形状为(3, 0)的空COO格式稀疏张量x
        x = self.sparse_empty(3, 0, device=1)
        check_device(x, 1)

    # 测试在新设备上创建稀疏张量
    def _test_new_device(self, size, device=torch.cuda):
        # 使用指定设备创建一个新的COO格式稀疏张量x
        with torch.cuda.device(device):
            x = torch.sparse_coo_tensor(size, device='cuda', dtype=torch.float64)
        # 断言：稀疏张量x的设备ID等于指定的设备
        self.assertEqual(x.get_device(), device)
        # 创建x的新实例x1和x2
        x1 = x.new()
        x2 = x.new(2, 3)
        # 断言：新实例x1和x2的设备ID等于指定的设备
        self.assertEqual(x1.get_device(), device)
        self.assertEqual(x2.get_device(), device)

    # 只在CUDA设备上运行的测试，测试在单个GPU上创建新设备
    @onlyCUDA
    def test_new_device_single_gpu(self):
        self._test_new_device((), 0)
        self._test_new_device((30, 20), 0)
        self._test_new_device((30, 20, 10), 0)
        self._test_new_device((30, 20, 10, 0), 0)

    # 只在CUDA设备上运行的测试，测试在多个GPU上创建新设备
    @onlyCUDA
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_new_device_multi_gpu(self):
        self._test_new_device((), 1)
        self._test_new_device((30, 20), 1)
        self._test_new_device((30, 20, 10), 1)
        self._test_new_device((30, 20, 10, 0), 1)

    # 用于指定是否在稀疏张量上启用或禁用稀疏化的装饰器，数据类型为torch.double或torch.cdouble
    @coalescedonoff
    @dtypes(torch.double, torch.cdouble)
    # 定义一个测试函数，用于测试稀疏张量的形状和内容
    def test_new(self, device, dtype, coalesced):
        # 定义内部函数，用于测试特定稀疏张量形状的情况
        def test_shape(sparse_dims, nnz, with_size):
            # 生成稀疏张量的数据：稀疏张量 x、索引 indices、值 values
            x, indices, values = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)
            
            # 如果 x 不在 CUDA 设备上，需要指定大小来创建新的稀疏张量 out
            if not x.is_cuda:
                # 使用 x 的类型创建新的稀疏张量 out，然后进行合并操作
                out = x.new(indices, values).coalesce()
                # 对原始稀疏张量 x 进行合并操作
                x_c = x.coalesce()
                # 断言新创建的稀疏张量 out 的索引和值与原始稀疏张量 x_c 相同
                self.assertEqual((out.indices(), out.values()), (x_c.indices(), x_c.values()))
            
            # 断言基于 x 创建的新稀疏张量的内容和大小与 x 相同
            self.assertEqual(x.new(indices, values, x.size()), x)

        # 使用不同的参数调用 test_shape 函数进行测试
        test_shape(3, 10, 100)
        test_shape(3, 0, [100, 100, 0])

    # 标记只在 CPU 上运行的测试用例装饰器，指定多个数据类型进行测试
    @onlyCPU  # not really, but we only really want to run this once
    @dtypes(torch.float64, torch.float32, torch.float16, torch.cfloat, torch.cdouble)
    # 定义测试工厂方法，用于生成各种条件下的稀疏张量并进行断言验证
    def test_factory(self, device, dtype):
        # 对于测试空张量的情况，分别进行测试
        for test_empty_tensor in [True, False]:
            if test_empty_tensor:
                # 设置空张量的默认大小和当前大小
                default_size = torch.Size([1, 3, 0])
                size = torch.Size([3, 3, 0])
            else:
                # 设置非空张量的默认大小和当前大小
                default_size = torch.Size([1, 3])
                size = torch.Size([3, 3])
            # 遍历是否包含大小信息、使用张量索引和值的情况、是否使用CUDA的情况
            for include_size in [True, False]:
                for use_tensor_idx in [True, False]:
                    for use_tensor_val in [True, False]:
                        for use_cuda in ([False] if not torch.cuda.is_available() else [True, False]):
                            # 如果使用CUDA，则必须包含大小信息
                            include_size = include_size or use_cuda
                            # 设置长整型数据类型为 int64
                            long_dtype = torch.int64
                            # 如果不使用CUDA，则设备为CPU；否则，设备为最后一个CUDA设备
                            device = torch.device('cpu') if not use_cuda else \
                                torch.device(torch.cuda.device_count() - 1)
                            # 根据是否使用张量索引，设置索引值
                            indices = torch.tensor(([0], [2]), dtype=long_dtype) if use_tensor_idx else ([0], [2])
                            # 根据测试空张量和是否使用张量值，设置值的张量
                            if test_empty_tensor:
                                values = torch.empty(1, 0).to(dtype)
                            else:
                                if use_tensor_val:
                                    values = torch.tensor([1.], dtype=dtype)
                                else:
                                    values = 1.
                            # 创建稀疏 COO 张量，根据是否包含大小信息设置不同参数
                            if include_size:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, size, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            else:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            # 断言稀疏张量的索引与设置的索引相等
                            self.assertEqual(indices, sparse_tensor._indices())
                            # 断言稀疏张量的值与设置的值相等
                            self.assertEqual(values, sparse_tensor._values())
                            # 断言稀疏张量的大小与设置的大小相等（根据是否包含大小信息选择不同的比较对象）
                            self.assertEqual(size if include_size else default_size, sparse_tensor.size())
                            # 断言稀疏张量的数据类型与设置的数据类型相等
                            self.assertEqual(dtype, sparse_tensor.dtype)
                            # 如果使用CUDA，断言稀疏张量的值所在设备与设置的设备相等
                            if use_cuda:
                                self.assertEqual(device, sparse_tensor._values().device)
                            # 断言稀疏张量的 requires_grad 属性为 True
                            self.assertEqual(True, sparse_tensor.requires_grad)

    @dtypes(torch.double, torch.cdouble)
    # 定义测试方法 test_factory_size_check，用于检查稀疏张量工厂的尺寸相关错误
    def test_factory_size_check(self, device, dtype):
        # 创建索引张量，包含两个稀疏张量的索引
        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        # 创建值张量，包含两个值为 0.5 的张量，指定数据类型和设备
        values = torch.tensor([.5, .5], dtype=dtype, device=device)
        # 创建尺寸张量，表示稀疏张量的形状为 [2, 3]
        sizes = torch.Size([2, 3])
        # 使用断言检测是否引发 RuntimeError 异常，异常信息包含 "size is inconsistent with indices"
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            # 创建稀疏 COO 张量，指定索引、值、尺寸、数据类型和设备
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        # 将索引张量填充为 -1
        indices.fill_(-1)
        # 使用断言检测是否引发 RuntimeError 异常，异常信息包含 "found negative index"
        with self.assertRaisesRegex(RuntimeError, "found negative index"):
            # 创建稀疏 COO 张量，指定填充后的索引、值、尺寸、数据类型和设备
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        # 重新创建索引张量，包含两个稀疏张量的索引
        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        # 创建空的值张量，指定形状为 [2, 1, 0]，数据类型和设备
        values = torch.empty([2, 1, 0], dtype=dtype, device=device)
        # 创建尺寸张量，表示稀疏张量的形状为 [2, 3, 1, 0]
        sizes = torch.Size([2, 3, 1, 0])
        # 使用断言检测是否引发 RuntimeError 异常，异常信息包含 "size is inconsistent with indices"
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            # 创建稀疏 COO 张量，指定索引、值、尺寸、数据类型和设备
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        # 重新创建索引张量，包含两个稀疏张量的索引
        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        # 创建空的值张量，指定形状为 [2, 2, 2]，数据类型和设备
        values = torch.empty([2, 2, 2], dtype=dtype, device=device)
        # 创建尺寸张量，表示稀疏张量的形状为 [0, 0, 2, 2]
        sizes = torch.Size([0, 0, 2, 2])
        # 使用断言检测是否引发 RuntimeError 异常，异常信息包含 "size is inconsistent with indices"
        with self.assertRaisesRegex(RuntimeError, "size is inconsistent with indices"):
            # 创建稀疏 COO 张量，指定索引、值、尺寸、数据类型和设备
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        # 重新创建索引张量，包含两个稀疏张量的索引
        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        # 创建值张量，包含值为 1 的张量，形状为 [2, 3, 2]，数据类型和设备
        values = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=dtype, device=device)
        # 创建尺寸张量，表示稀疏张量的形状为 [3, 3, 2]
        sizes = torch.Size([3, 3, 2])
        # 使用断言检测是否引发 RuntimeError 异常，异常信息包含 "values has incorrect size"
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            # 创建稀疏 COO 张量，指定索引、值、尺寸、数据类型和设备
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        # 重新创建索引张量，包含两个稀疏张量的索引
        indices = self.index_tensor([[1, 2],
                                    [0, 2]], device=device)
        # 创建空的值张量，指定形状为 [2, 1, 0]，数据类型和设备
        values = torch.empty([2, 1, 0], dtype=dtype, device=device)
        # 创建尺寸张量，表示稀疏张量的形状为 [3, 3, 2, 0]
        sizes = torch.Size([3, 3, 2, 0])
        # 使用断言检测是否引发 RuntimeError 异常，异常信息包含 "values has incorrect size"
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            # 创建稀疏 COO 张量，指定索引、值、尺寸、数据类型和设备
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)
    @dtypes(torch.double, torch.cdouble)
    # 声明一个测试函数，用于测试具有指定设备和数据类型的稀疏张量工厂方法，支持双精度和复双精度
    def test_factory_nnz(self, device, dtype):
        indices = self.index_tensor([[0]], device=device)  # (sparse_dim, nnz): (1, 1)
        # 创建稀疏张量的索引张量，表示稀疏度为1的情况，使用给定的设备
        values = torch.tensor([[1, 1], [1, 1]], dtype=dtype, device=device)  # (nnz, ...): (2, 2)
        # 创建稀疏张量的值张量，形状为(2, 2)，使用指定的数据类型和设备
        sizes = torch.Size([2, 2])
        # 定义稀疏张量的大小

        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            # 使用断言检查运行时错误，确保索引和值具有相同的非零元素数目
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

        indices = self.index_tensor([[0]], device=device)  # (sparse_dim, nnz): (1, 1)
        # 再次创建稀疏张量的索引张量，形状为(1, 1)，使用指定的设备
        values = torch.empty([2, 0], dtype=dtype, device=device)  # (nnz, ...): (2, 0)
        # 创建一个空的稀疏张量值张量，形状为(2, 0)，使用指定的数据类型和设备
        sizes = torch.Size([2, 0])
        # 定义稀疏张量的大小

        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            # 使用断言检查运行时错误，确保索引和值具有相同的非零元素数目
            torch.sparse_coo_tensor(indices, values, sizes, dtype=dtype, device=device)

    @dtypes(torch.double, torch.cdouble)
    # 声明一个测试函数，用于测试具有指定设备和数据类型的稀疏张量工厂方法，支持双精度和复双精度
    def test_factory_nnz_zero(self, device, dtype):
        def test_shape(i_shape, v_shape, size, expected_size):
            # 定义一个内部函数，用于测试不同形状参数的稀疏张量的形状和大小

            if size:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), torch.Size(size),
                                            dtype=dtype, device=device)
            else:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), dtype=dtype, device=device)
            # 创建稀疏 COO 张量，根据给定的索引和值张量形状，使用指定的数据类型和设备

            expected_indices = torch.empty(i_shape, device=device, dtype=torch.int64)
            # 创建期望的稀疏张量索引张量，形状与给定的索引形状相同，使用指定的设备和数据类型
            expected_values = torch.empty(v_shape, device=device, dtype=dtype)
            # 创建期望的稀疏张量值张量，形状与给定的值形状相同，使用指定的设备和数据类型
            expected_size = torch.Size(expected_size)
            # 定义期望的稀疏张量大小

            self.assertEqual(t._indices(), expected_indices)
            # 使用断言检查实际稀疏张量的索引张量与期望的索引张量是否相等
            self.assertEqual(t._values(), expected_values)
            # 使用断言检查实际稀疏张量的值张量与期望的值张量是否相等
            self.assertEqual(t.size(), expected_size)
            # 使用断言检查实际稀疏张量的大小与期望的大小是否相等

        test_shape([1, 0], [0, 2, 4, 0], None, [0, 2, 4, 0])
        # 调用内部函数，测试稀疏张量形状，索引形状为[1, 0]，值形状为[0, 2, 4, 0]，大小为None，期望大小为[0, 2, 4, 0]
        test_shape([3, 0], [0, 2, 4, 0], None, [0, 0, 0, 2, 4, 0])
        # 调用内部函数，测试稀疏张量形状，索引形状为[3, 0]，值形状为[0, 2, 4, 0]，大小为None，期望大小为[0, 0, 0, 2, 4, 0]
        test_shape([1, 0], [0, 2, 4, 0], [0, 2, 4, 0], [0, 2, 4, 0])
        # 调用内部函数，测试稀疏张量形状，索引形状为[1, 0]，值形状为[0, 2, 4, 0]，大小为[0, 2, 4, 0]，期望大小为[0, 2, 4, 0]
        test_shape([3, 0], [0, 2, 4, 0], [0, 0, 0, 2, 4, 0], [0, 0, 0, 2, 4, 0])
        # 调用内部函数，测试稀疏张量形状，索引形状为[3, 0]，值形状为[0, 2, 4, 0]，大小为[0, 0, 0, 2, 4, 0]，期望大小为[0, 0, 0, 2, 4, 0]
        test_shape([3, 0], [0, 2, 4, 0], [1, 2, 3, 2, 4, 0], [1, 2, 3, 2, 4, 0])
        # 调用内部函数，测试稀疏张量形状，索引形状为[3, 0]，值形状为[0, 2, 4, 0]，大小为[1, 2, 3, 2, 4, 0]，期望大小为[1, 2, 3, 2, 4, 0]

    @dtypes(torch.double, torch.cdouble)
    # 声明一个测试函数，用于测试具有指定设备和数据类型的稀疏张量工厂方法，支持双精度和复双精度
    def test_factory_dense_dim(self, device, dtype):
        indices = self.index_tensor([[0]], device=device)
        # 创建稀疏张量的索引张量，表示稠密维度为1的情况，使用指定的设备
        values = torch.tensor([[[1, 1, 1], [1, 1, 1]]], dtype=dtype, device=device)
        # 创建稀疏张量的值张量，形状为[1, 2, 3]，使用指定的数据类型和设备
        sizes = torch.Size([1, 3, 4])
        # 定义稀疏张量的大小

        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            # 使用断言检查运行时错误，确保值张量大小不正确
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.index_tensor([[0]], device=device)
        # 再次创建稀疏张量的索引张量，形状为[1, 1]，使用指定的设备
        values = torch.empty([1, 2, 3, 0], dtype=dtype, device=device)
        # 创建一个空的稀
    # 测试工厂类型推断功能，使用不同的设备和数据类型进行测试
    def test_factory_type_inference(self, device, dtype):
        # 创建稀疏 COO 张量，使用给定的索引和数据，验证数据类型是否与指定的 dtype 一致
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=dtype))
        self.assertEqual(dtype, t.dtype)
        # 创建稀疏 COO 张量，使用给定的索引和数据，验证数据类型是否为 torch.int64
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1]))
        self.assertEqual(torch.int64, t.dtype)

        # 使用稀疏 COO 张量构造函数，指定半精度浮点数数据类型 torch.float16
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.HalfTensor(1, 0))
        self.assertEqual(torch.float16, t.dtype)
        # 使用稀疏 COO 张量构造函数，指定单精度浮点数数据类型 torch.float32
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.FloatTensor(1, 0))
        self.assertEqual(torch.float32, t.dtype)
        # 使用稀疏 COO 张量构造函数，指定双精度浮点数数据类型 torch.float64
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.DoubleTensor(1, 0))
        self.assertEqual(torch.float64, t.dtype)
        # 使用稀疏 COO 张量构造函数，指定 64 位整型数据类型 torch.int64
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.LongTensor(1, 0))
        self.assertEqual(torch.int64, t.dtype)

    @onlyCUDA
    def test_factory_device_type_inference(self, device):
        # 只有当 indices 和 values 都在 CUDA 设备上时才会执行以下操作

        # 定义设备选择范围：'cpu', 'cuda' 及其组合包括 None
        cpu_cuda = ('cpu', 'cuda')
        cpu_cuda_none = cpu_cuda + (None,)
        
        # 使用 itertools.product 迭代所有 indices_device, values_device 和 device 的组合
        for indices_device, values_device, device in itertools.product(cpu_cuda,
                                                                       cpu_cuda,
                                                                       cpu_cuda_none):
            # 创建在指定设备上的索引张量
            indices = torch.tensor(([0], [2]), device=indices_device)
            # 创建在指定设备上的值张量
            values = torch.tensor([1.], device=values_device)
            # 创建在指定设备上的空值张量
            empty_values = torch.empty(1, 0).to(values_device)
            # 定义形状
            shape = (1, 3)
            # 定义空形状
            empty_shape = (1, 3, 0)
            
            # 如果 device 为 None 且 indices_device 与 values_device 不同，则应抛出 RuntimeError
            if device is None and indices_device != values_device:
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, values, shape, device=device)
                with self.assertRaises(RuntimeError):
                    torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
            else:
                # 使用稀疏 COO 张量构造函数，在指定设备上创建稀疏张量 t
                t = torch.sparse_coo_tensor(indices, values, shape, device=device)
                # 使用稀疏 COO 张量构造函数，在指定设备上创建稀疏张量 t_empty
                t_empty = torch.sparse_coo_tensor(indices, empty_values, empty_shape, device=device)
                # 判断 t 是否在 CUDA 设备上
                should_be_cuda = (device == 'cuda' or (device is None and values_device == 'cuda'))
                self.assertEqual(should_be_cuda, t.is_cuda)
                # 验证 t 和 t_empty 是否在相同的设备上
                self.assertEqual(t.is_cuda, t_empty.is_cuda)

    @onlyCPU
    # 定义一个测试工厂函数，用于复制测试用例，接受设备参数
    def test_factory_copy(self, device):
        # 定义一个测试张量的内部函数，接受稀疏张量的指标、值以及指标和值是否相等的标志
        def test_tensor(indices, values, indices_equal, values_equal):
            # 创建稀疏 COO 张量，使用给定的指标和值，并指定数据类型和设备
            sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=torch.float64, device=device)
            # 如果指标相等，断言稀疏张量的指标数据指针与给定指标数据指针相等
            if indices_equal:
                self.assertEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            else:
                # 否则断言它们不相等
                self.assertNotEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            # 如果值相等，断言稀疏张量的值数据指针与给定值数据指针相等
            if values_equal:
                self.assertEqual(values.data_ptr(), sparse_tensor._values().data_ptr())
            else:
                # 否则断言它们不相等
                self.assertNotEqual(values.data_ptr(), sparse_tensor._values().data_ptr())

        # 测试用例：指标和值均正确
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, True, True)

        # 测试用例：只有指标正确
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, True, True)

        # 测试用例：只有指标正确，但值的数据类型不匹配
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float16)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, True, True)  # 空张量的 data_ptr 总是等于 0

        # 测试用例：只有值正确
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, False, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, False, True)

        # 测试用例：指标和值均不正确
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, False, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, False, True)  # 空张量的 data_ptr 总是等于 0

        # 复杂支持情况的测试用例
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = make_tensor([1, ], dtype=torch.cdouble, device=device)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = make_tensor([1, 1], dtype=torch.cdouble, device=device)
        test_tensor(indices, values, False, False)

    @onlyCPU  # 仅运行一次，我们测试 CPU 和 CUDA 两种情况
    # 测试在旧设备上创建新的稀疏张量
    def test_legacy_new_device(self, device):
        # 定义稀疏张量的索引
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        # 定义稀疏张量的值
        v = torch.tensor([3., 4., 5.])
        # 定义稀疏张量的大小
        size = torch.Size([2, 3])

        # 在CPU上创建稀疏COO张量
        x = torch.sparse_coo_tensor(i, v, size, device='cpu')
        # 测试在CPU上尝试将张量转移到CUDA设备是否引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        # 测试在CPU上尝试将张量转移到CUDA设备并指定索引和值是否引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cuda'))
        # 测试在CPU上尝试将张量转移到CUDA设备并指定索引、值和大小是否引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cuda'))
        # 测试在CPU上尝试将张量转移到CUDA设备并指定新的大小是否引发 RuntimeError
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))

        # 如果CUDA可用，重复上述测试步骤，但在CUDA设备上创建稀疏COO张量
        if torch.cuda.is_available():
            x = torch.sparse_coo_tensor(i, v, size, device='cuda')
            # 测试在CUDA设备上尝试将张量转移到CPU是否引发 RuntimeError
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            # 测试在CUDA设备上尝试将张量转移到CPU并指定索引和值是否引发 RuntimeError
            self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cpu'))
            # 测试在CUDA设备上尝试将张量转移到CPU并指定索引、值和大小是否引发 RuntimeError
            self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cpu'))
            # 测试在CUDA设备上尝试将张量转移到CPU并指定新的大小是否引发 RuntimeError
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))

    # 测试在旧设备上创建新的稀疏张量
    def test_legacy_new(self, device):
        # 定义稀疏张量的索引
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        # 定义稀疏张量的值
        v = torch.tensor([3., 4., 5.])
        # 定义稀疏张量的大小
        size = torch.Size([2, 3])
        # 创建稀疏COO张量
        s = torch.sparse_coo_tensor(i, v, size)

        # 测试在CPU上创建新的稀疏COO张量后其布局是否为 sparse_coo
        self.assertEqual(torch.sparse_coo, s.new(device='cpu').layout)
        # 测试尝试使用未指定存储类型创建新的稀疏张量是否引发 TypeError
        self.assertRaises(TypeError, lambda: s.new(v.untyped_storage()))
        # 测试尝试使用未指定值创建新的稀疏张量是否引发 TypeError
        self.assertRaises(TypeError, lambda: s.new(v))
        # 测试在CPU上创建新的稀疏COO张量并指定新的大小后其布局是否为 sparse_coo
        self.assertEqual(torch.sparse_coo, s.new(torch.Size([2, 3])).layout)
        # 测试尝试使用单个元素的列表创建新的稀疏张量是否引发 TypeError
        self.assertRaises(TypeError, lambda: s.new([6]))

    # 仅在CPU上运行此测试函数，用于测试数据类型
    @onlyCPU  # not really, but we only really want to run this once
    def test_dtypes(self, device):
        # 获取所有稀疏张量支持的数据类型
        all_sparse_dtypes = all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16)
        # 在CPU上运行数据类型测试函数
        do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cpu'))
        # 如果CUDA可用，在CUDA设备上运行数据类型测试函数
        if torch.cuda.is_available():
            do_test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cuda:0'))
    # 定义一个测试方法，用于测试稀疏张量的空和满情况，接受设备、数据类型和梯度需求作为参数
    def _test_empty_full(self, device, dtype, requires_grad):
        # 设置张量的形状
        shape = (2, 3)
        # 设置张量的布局为稀疏 COO 格式
        layout = torch.sparse_coo

        # 定义一个内部函数，用于检查张量的值和属性
        def check_value(tensor, value=None, dtype=dtype, requires_grad=requires_grad):
            # 断言张量的形状与预期相符
            self.assertEqual(shape, tensor.shape)
            # 断言张量的数据类型与预期相符
            self.assertIs(dtype, tensor.dtype)
            # 断言张量的布局与预期相符
            self.assertIs(layout, tensor.layout)
            # 断言张量的梯度需求与预期相符
            self.assertEqual(tensor.requires_grad, requires_grad)
            # 如果张量在 GPU 上且设备不为空，则断言张量的设备与预期相符
            if tensor.is_cuda and device is not None:
                self.assertEqual(device, tensor.device)
            # 如果给定了值，则创建一个空的张量并填充该值，然后断言两个张量相等
            if value is not None:
                fill = tensor.empty(shape, dtype=dtype).fill_(value)
                self.assertEqual(tensor, fill)

        # 创建一个稀疏 COO 张量 v，用于测试
        v = torch.sparse_coo_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        # 调用 check_value 函数，检查张量 v 的值和属性
        check_value(v)

        # 创建一个与 v 相同形状的全零张量 out，并使用其初始化一个新的稀疏 COO 张量，再次调用 check_value 函数检查其值和属性
        out = v.new()
        check_value(torch.zeros(shape, out=out, device=device, requires_grad=requires_grad))

        # 定义一个 int64 类型的数据类型
        int64_dtype = torch.int64
        # 使用 v 的方法创建一个空张量，不需要梯度，再次调用 check_value 函数检查其值和属性
        check_value(v.new_empty(shape), requires_grad=False)
        # 使用 v 的方法创建一个空张量，指定数据类型为 int64，并且可能的话指定设备，再次调用 check_value 函数检查其值和属性
        check_value(v.new_empty(shape, dtype=int64_dtype, device=device, requires_grad=False),
                    dtype=int64_dtype, requires_grad=False)
        # 使用 v 的方法创建一个与 v 相同的空张量，不需要梯度，再次调用 check_value 函数检查其值和属性
        check_value(torch.empty_like(v), requires_grad=False)
        # 使用 v 的方法创建一个与 v 相同形状的空张量，指定数据类型为 int64、布局为稀疏 COO，并且可能的话指定设备，再次调用 check_value 函数检查其值和属性
        check_value(torch.empty_like(v, dtype=int64_dtype, layout=layout, device=device, requires_grad=False),
                    dtype=int64_dtype, requires_grad=False)

    # 标记只在 CPU 上运行的测试方法，但实际上我们只需要运行一次这个方法
    @onlyCPU  # not really, but we only really want to run this once
    # 参数化测试数据类型，包括所有类型和复杂类型以及 torch.half、torch.bool、torch.bfloat16
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    # 参数化梯度需求，True 或 False
    @parametrize('requires_grad', (True, False))
    # 测试方法，用于测试稀疏张量的空和满情况，接受设备、数据类型和梯度需求作为参数
    def test_empty_full(self, device, dtype, requires_grad):
        # 如果需要梯度，并且数据类型不是浮点型或复数型，则跳过测试并给出相应信息
        if requires_grad and not (dtype.is_floating_point or dtype.is_complex):
            self.skipTest(f'requires_grad==True requires float or complex dtype, got {dtype}')

        # 调用 _test_empty_full 方法，测试稀疏张量的空和满情况
        self._test_empty_full(device, dtype, requires_grad)
        # 如果 CUDA 可用，再次调用 _test_empty_full 方法，测试稀疏张量的空和满情况，设备为 CUDA
        if torch.cuda.is_available():
            self._test_empty_full(None, dtype, requires_grad)
            self._test_empty_full(torch.device('cuda:0'), dtype, requires_grad)

    # 测试方法，用于测试稀疏张量的稀疏性
    def test_is_sparse(self, device):
        # 创建一个形状为 (3, 3) 的普通张量 x，断言其不是稀疏张量
        x = torch.randn(3, 3)
        self.assertFalse(x.is_sparse)

        # 创建一个形状为 (3, 3, 0) 的普通张量 x，断言其不是稀疏张量
        x = torch.randn(3, 3, 0)
        self.assertFalse(x.is_sparse)

        # 使用 sparse_empty 方法创建一个形状为 (1, 0) 的稀疏张量 x，断言其是稀疏张量
        x = self.sparse_empty(1, 0, device=device)
        self.assertTrue(x.is_sparse)

    # 测试方法，用于测试稀疏张量的尺寸调整
    def test_resize_as(self, device):
        # 定义一个内部函数，用于执行测试
        def do_test(t):
            # 创建一个新的张量 y，其形状与 t 相同，并且所有元素设置为零
            y = t.new().resize_as_(t).zero_()
            # 断言张量 y 的形状与 t 相同
            self.assertEqual(y.shape, t.shape)
            # 检查能否将 y 加到 t 上，这要求稀疏维度和密集维度匹配
            self.assertEqual(t, t + y)

        # 执行测试，使用形状为 [3, 0] 的稀疏张量作为输入
        do_test(self.sparse_empty([3, 0], device=device))
        # 执行测试，使用形状为 [3, 3] 的稀疏张量作为输入
        do_test(self.sparse_empty([3, 3], device=device))
    # 定义一个测试方法，用于验证稀疏张量的形状调整操作
    def _test_resize_shape(self, x_i, x_v, x_size, y_i, y_v, y_size, dtype, device):
        # 计算稀疏张量 x_v 的元素数量
        x_v_numel = torch.zeros(x_v).numel()
        # 计算稀疏张量 y_v 的元素数量
        y_v_numel = torch.zeros(y_v).numel()
        # 使用给定的索引、值和大小创建稀疏 COO 张量 x
        x = torch.sparse_coo_tensor(torch.zeros(x_i),
                                    torch.arange(x_v_numel).resize_(x_v).to(torch.float),
                                    torch.Size(x_size), dtype=dtype, device=device)
        # 将稀疏张量 x 转换为密集张量
        x_dense = x.to_dense()
        # 使用给定的索引、值和大小创建稀疏 COO 张量 y
        y = torch.sparse_coo_tensor(torch.zeros(y_i),
                                    torch.ones(y_v).to(torch.float),
                                    torch.Size(y_size), dtype=dtype, device=device)
        # 将稀疏张量 y 转换为密集张量
        y_dense = y.to_dense()
        # 调整稀疏张量 x 的形状以匹配张量 y
        x.resize_as_(y)
        # 调整稠密张量 x_dense 的形状以匹配张量 y_dense
        x_dense.resize_as_(y_dense)
        # 使用断言方法验证 x 和 y 的形状是否相等
        self.assertEqual(x.shape, y.shape)
        # 使用断言方法验证 x 和 y 的稀疏维度是否相等
        self.assertEqual(x.sparse_dim(), y.sparse_dim())
        # 使用断言方法验证 x 和 y 的密集维度是否相等
        self.assertEqual(x.dense_dim(), y.dense_dim())
        # 使用断言方法验证 x_dense 和 y_dense 的形状是否相等
        self.assertEqual(x.shape, x_dense.shape)
        # 使用断言方法验证 x_dense 和 y_dense 的形状是否相等
        self.assertEqual(y.shape, y_dense.shape)
        # 在调整形状后，确保原始数据在稀疏张量和其对应的密集张量中得到保留
        self.assertEqual(x.to_dense().view(-1)[0:x_v_numel].view(x_v),
                         x_dense.view(-1)[0:x_v_numel].view(x_v))

    @dtypes(torch.double, torch.cdouble)
    # 定义一个装饰器，用于指定测试方法的数据类型
    def test_is_nonzero(self, device):
        # 验证包含单个非零元素的稀疏张量是否被正确识别为非零
        self.assertTrue(torch.sparse_coo_tensor(([0],), 1., (1,), device=device).is_nonzero())
        # 验证包含单个零元素的稀疏张量是否被正确识别为零
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0., (1,), device=device).is_nonzero())
        # 验证包含多个零元素的稀疏张量是否被正确识别为零
        self.assertFalse(torch.sparse_coo_tensor(([0], [0]), 0., (1, 1), device=device).is_nonzero())
        # 验证包含多个零元素的复数稀疏张量是否被正确识别为零
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (0., 0.), (1,), device=device).is_nonzero())
        # 验证包含复数元素的稀疏张量是否被正确识别为非零
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (-1., 1.), (1,), device=device).is_nonzero())

        # 验证标量稀疏张量是否被正确识别为非零
        self.assertTrue(torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, [], device=device).is_nonzero())
        # 验证对于不包含值的稀疏张量是否会引发错误
        with self.assertRaisesRegex(RuntimeError, "Boolean value of Tensor with no values is ambiguous"):
            torch.sparse_coo_tensor(([0, 1],), torch.empty(2, 0), (4, 0), device=device).is_nonzero()
        # 验证包含复数元素的稀疏张量是否被正确识别为非零
        self.assertTrue(torch.sparse_coo_tensor(([0],), 2.3 - 4.5j, (1,), dtype=torch.cfloat, device=device)
                        .is_nonzero())
        # 验证包含复数元素的稀疏张量是否被正确识别为非零
        self.assertTrue(torch.sparse_coo_tensor(([0],), 2.3 - 4.5j, (1,), dtype=torch.cdouble, device=device)
                        .is_nonzero())
        # 验证包含复数元素的稀疏张量是否被正确识别为零
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0. + 0j, (1,), dtype=torch.cfloat, device=device)
                         .is_nonzero())
        # 验证包含复数元素的稀疏张量是否被正确识别为零
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0. + 0j, (1,), dtype=torch.cdouble, device=device)
                         .is_nonzero())
    # 测试改变稀疏张量的元数据
    def test_change_tensor_metadata(self, device, dtype):
        # 创建索引张量 i，包含 [[0], [1]]，并指定设备和数据类型
        i = self.index_tensor([[0], [1]], device=device)
        # 创建值张量 v，包含 [[3, 4, 5]]，并指定数据类型和设备
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        # 创建稀疏 COO 张量 t，使用 i 和 v，形状为 [1, 2, 3]，并指定数据类型和设备
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]), dtype=dtype, device=device)
        # 改变索引张量 i 的尺寸为 (2, 3)
        i.resize_(2, 3)
        # 改变值张量 v 的尺寸为 (4, 5)
        v.resize_(4, 5)
        # 断言稀疏张量 t 的 coalesce 后的索引尺寸为 [2, 1]
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        # 断言稀疏张量 t 的 coalesce 后的值尺寸为 [1, 3]
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        # 重新创建索引张量 i 和值张量 v，然后创建稀疏 COO 张量 t，形状为 [1, 2, 3]
        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        # 将索引张量 i 的尺寸调整为和 index_tensor([0, 1]) 一致
        i.resize_as_(self.index_tensor([0, 1], device=device))
        # 将值张量 v 的尺寸调整为和 torch.tensor([3, 4, 5]) 一致
        v.resize_as_(torch.tensor([3, 4, 5], dtype=dtype, device=device))
        # 断言稀疏张量 t 的 coalesce 后的索引尺寸为 [2, 1]
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        # 断言稀疏张量 t 的 coalesce 后的值尺寸为 [1, 3]
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        # 重新创建索引张量 i 和值张量 v，然后创建稀疏 COO 张量 t，形状为 [1, 2, 3]
        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        # 使用 as_strided_ 方法改变索引张量 i 的形状为 (2, 1)，步长为 (1, 1)
        i.as_strided_((2, 1), (1, 1))
        # 使用 as_strided_ 方法改变值张量 v 的形状为 (1, 3)，步长为 (1, 1)
        v.as_strided_((1, 3), (1, 1))
        # 断言稀疏张量 t 的 coalesce 后的索引尺寸为 [2, 1]
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        # 断言稀疏张量 t 的 coalesce 后的值尺寸为 [1, 3]
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        # 重新创建索引张量 i 和值张量 v，然后创建稀疏 COO 张量 t，形状为 [1, 2, 3]
        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        # 使用 set_ 方法将索引张量 i 设置为 index_tensor([0, 1]) 的值
        i.set_(self.index_tensor([0, 1], device=device))
        # 使用 set_ 方法将值张量 v 设置为 torch.tensor([3, 4, 5]) 的值
        v.set_(torch.tensor([3, 4, 5], dtype=dtype, device=device))
        # 断言稀疏张量 t 的 coalesce 后的索引尺寸为 [2, 1]
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        # 断言稀疏张量 t 的 coalesce 后的值尺寸为 [1, 3]
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

        # 重新创建索引张量 i 和值张量 v，然后创建稀疏 COO 张量 t，形状为 [1, 2, 3]
        i = self.index_tensor([[0], [1]], device=device)
        v = torch.tensor([[3, 4, 5]], dtype=dtype, device=device)
        t = torch.sparse_coo_tensor(i, v, torch.Size([1, 2, 3]))
        # 使用 transpose_ 方法交换索引张量 i 的维度 0 和 1
        i.transpose_(0, 1)
        # 使用 transpose_ 方法交换值张量 v 的维度 0 和 1
        v.transpose_(0, 1)
        # 断言稀疏张量 t 的 coalesce 后的索引尺寸为 [2, 1]
        self.assertEqual(list(t.coalesce().indices().size()), [2, 1])
        # 断言稀疏张量 t 的 coalesce 后的值尺寸为 [1, 3]
        self.assertEqual(list(t.coalesce().values().size()), [1, 3])

    @coalescedonoff
    @dtypes(torch.double)
    # 定义一个测试函数，用于测试序列化与反序列化稀疏张量
    def test_pickle(self, device, dtype, coalesced):
        # 导入 pickle 库，用于对象的序列化与反序列化操作
        import pickle

        # 定义多组稀疏张量的形状、稀疏维度及非零元素个数的组合
        shape_sparse_dim_nnz = [
            ((), 0, 2),         # 0维稀疏张量，稀疏维度为0，非零元素个数为2
            ((0,), 0, 10),      # 1维稀疏张量，稀疏维度为0，非零元素个数为10
            ((2,), 0, 3),       # 1维稀疏张量，稀疏维度为0，非零元素个数为3
            ((100, 3), 1, 3),   # 2维稀疏张量，稀疏维度为1，非零元素个数为3
            ((100, 20, 3), 2, 0),   # 3维稀疏张量，稀疏维度为2，非零元素个数为0
            ((10, 0, 3), 0, 3),     # 3维稀疏张量，稀疏维度为0，非零元素个数为3
            ((10, 0, 3), 0, 0),     # 3维稀疏张量，稀疏维度为0，非零元素个数为0
        ]

        # 遍历每组稀疏张量的形状、稀疏维度及非零元素个数的组合
        for shape, sparse_dim, nnz in shape_sparse_dim_nnz:
            # 计算稀疏索引的形状
            indices_shape = torch.Size((sparse_dim, nnz))
            # 计算稀疏值的形状
            values_shape = torch.Size((nnz,) + shape[sparse_dim:])
            # 生成稀疏索引，使用设备上的数据类型，并视图为指定形状
            indices = torch.arange(indices_shape.numel(), dtype=self.index_tensor(0).dtype,
                                   device=device).view(indices_shape)
            # 对每个稀疏维度的索引进行裁剪，确保它们是有效索引
            for d in range(sparse_dim):
                indices[d].clamp_(max=(shape[d] - 1))
            # 如果未压缩且索引数大于0，则使最后一个索引与第一个索引不合并
            if not coalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]
            # 计算稀疏值的元素数量
            values_numel = values_shape.numel()
            # 生成稀疏值，使用指定数据类型，并视图为指定形状，再除以元素数量的一半
            values = torch.arange(values_numel, dtype=dtype,
                                  device=device).view(values_shape).div_(values_numel / 2.)
            # 创建稀疏张量
            sp_tensor = self.sparse_tensor(indices, values, shape)
            # 序列化稀疏张量
            serialized = pickle.dumps(sp_tensor)
            # 反序列化稀疏张量
            sp_tensor_loaded = pickle.loads(serialized)
            # 断言序列化前后稀疏张量是否相等
            self.assertEqual(sp_tensor, sp_tensor_loaded)

    # 定义一个测试函数，用于测试稀疏张量中是否有任意真值
    def test_any(self, device):
        # 创建稀疏张量 t，指定非零元素索引及对应的布尔值，使用指定设备
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([False, False]), device=device)
        # 创建布尔值张量 t_any，用于断言是否存在任意真值
        t_any = torch.tensor(False)
        # 断言稀疏张量中是否有任意真值
        self.assertEqual(torch.any(t), t_any)
        # 更新稀疏张量 t 的非零元素索引及对应的布尔值
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([True, False]), device=device)
        # 更新布尔值张量 t_any，用于断言是否存在任意真值
        t_any = torch.tensor(True)
        # 断言稀疏张量中是否有任意真值
        self.assertEqual(torch.any(t), t_any)

    # 定义一个测试函数，用于测试稀疏张量中是否存在 NaN
    def test_isnan(self, device):
        # 创建稀疏张量 t，指定非零元素索引及对应的数值，使用指定设备
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([1, 4]), device=device)
        # 创建布尔值稀疏张量 t_nan，用于断言是否存在 NaN
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([False, False]), device=device)
        # 断言稀疏张量中 NaN 的情况
        self.assertEqual(torch.isnan(t).int(), t_nan.int())
        # 更新稀疏张量 t 的非零元素索引及对应的数值，其中包含 NaN
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([1, float("nan")]), device=device)
        # 更新布尔值稀疏张量 t_nan，用于断言是否存在 NaN
        t_nan = torch.sparse_coo_tensor(torch.tensor(([0, 0], [0, 2])), torch.tensor([False, True]), device=device)
        # 断言稀疏张量中 NaN 的情况
        self.assertEqual(torch.isnan(t).int(), t_nan.int())
    # 定义一个测试方法，用于测试稀疏张量的除法和舍入模式
    def test_div_rounding_mode(self, device, dtype, coalesced):
        # 生成一个稀疏张量，以及相应的稠密张量
        sparse, _, _ = self._gen_sparse(2, 10, (10, 10), dtype, device, coalesced)
        dense = self.safeToDense(sparse)

        # 遍历不同的舍入模式：无，'floor'，'trunc'
        for mode in (None, 'floor', 'trunc'):
            # 对稀疏张量进行除法操作，指定舍入模式，得到实际结果
            actual = sparse.div(-2, rounding_mode=mode)
            # 对稠密张量进行除法操作，指定舍入模式，得到期望结果
            expect = dense.div(-2, rounding_mode=mode)
            # 断言实际结果与期望结果相同
            self.assertEqual(self.safeToDense(actual), expect)

            # 测试原地操作
            actual = sparse.clone().div_(-2, rounding_mode=mode)
            self.assertEqual(self.safeToDense(actual), expect)

            # 测试使用 out 参数指定输出张量
            actual.zero_()
            torch.div(sparse, -2, rounding_mode=mode, out=actual)
            self.assertEqual(self.safeToDense(actual), expect)

    # 定义一个测试方法，测试稀疏张量之间的除法，预期触发运行时错误
    def test_div_by_sparse_error(self, device):
        self.assertRaisesRegex(RuntimeError, 'Sparse division requires',
                               lambda: torch.tensor(1., device=device).to_sparse()
                               / torch.tensor(1., device=device).to_sparse())

    # 定义一个测试方法，测试稀疏张量之间的整数除法，预期触发运行时错误
    def test_floor_divide_by_sparse_error(self, device):
        self.assertRaisesRegex(RuntimeError, 'Sparse floor division requires',
                               lambda: torch.tensor(1., device=device).to_sparse()
                               // torch.tensor(1., device=device).to_sparse())

    # 装饰器：当未安装 NumPy 时跳过测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    @onlyCPU
    # 定义一个测试方法，测试稀疏张量转换为 NumPy 数组时的类型错误，预期触发类型错误
    def test_sparse_to_numpy(self, device):
        # 创建一个稀疏 COO 张量
        t = torch.sparse_coo_tensor(torch.tensor(([0, 0], [2, 0])), torch.tensor([1, 4]))
        # 断言调用 t.numpy() 会触发 TypeError
        self.assertRaises(TypeError, lambda: t.numpy())

    # 装饰器：设置张量类型和设备，检查零非零的 softmax 操作
    @coalescedonoff
    @dtypes(torch.double)
    # 定义一个内部方法，检查零非零的 softmax 操作，验证输出为零张量
    def _check_zero_nnz_softmax_op(self, func, ndim, device, dtype):
        # 创建一个零非零稀疏张量
        t = torch.sparse_coo_tensor([[] for _ in range(ndim)], [], (0,) * (ndim - 1) + (3,), device=device, dtype=dtype)
        # 执行指定的 softmax 操作，期望输出为与输入形状相同的零张量
        out = func(t, 0)
        self.assertEqual(out, torch.zeros_like(t))

        # 梯度检查
        t = t.requires_grad_()
        gradcheck(lambda x: func(x, 0).to_dense(), (t,), masked=True)

    # 装饰器：设置张量类型和设备，测试零非零的 softmax 操作
    @dtypes(torch.double, torch.float)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    # 定义一个测试方法，测试零非零的 softmax 操作
    def test_softmax_zero_nnz(self, device, dtype):
        self._check_zero_nnz_softmax_op(torch.sparse.softmax, 1, device, dtype)
        self._check_zero_nnz_softmax_op(torch.sparse.softmax, 10, device, dtype)

    # 装饰器：设置张量类型和设备，测试零非零的 log softmax 操作
    @dtypes(torch.double, torch.float)
    @unittest.skipIf(TEST_WITH_CROSSREF, "generator unsupport triggers assertion error")
    # 定义一个测试方法，测试零非零的 log softmax 操作
    def test_log_softmax_zero_nnz(self, device, dtype):
        self._check_zero_nnz_softmax_op(torch.sparse.log_softmax, 1, device, dtype)
        self._check_zero_nnz_softmax_op(torch.sparse.log_softmax, 10, device, dtype)

    # 待办事项：在确认后检查，为什么 ROCm 的 cusparseXcsrgemm2Nnz 函数与 CUDA 返回的 nnz 值不同
    @skipIfRocm
    @coalescedonoff
    # 使用装饰器设置测试函数的数据类型为浮点数和复数类型
    @dtypes(*floating_and_complex_types())
    # 根据CUDA是否可用，设置特定数据类型的测试，如torch.half和torch.bfloat16等
    @dtypesIfCUDA(*floating_types_and(*[torch.half] if SM53OrLater else [],
                                      *[torch.bfloat16] if SM80OrLater else [],
                                      torch.complex64,
                                      *[torch.complex128] if CUSPARSE_SPMM_COMPLEX128_SUPPORTED else []))
    # 在跨参考测试时跳过当前测试，因为它无法与假张量一起工作
    @unittest.skipIf(TEST_WITH_CROSSREF, "not working with fake tensor")
    # 设置特定精度的覆盖值，适用于torch.bfloat16、torch.float16、torch.complex64和torch.float32
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2, torch.complex64: 1e-2, torch.float32: 1e-2})
    # 控制是否进行数据协调的装饰器
    @coalescedonoff
    # 设置测试函数仅接受torch.double类型的输入
    @dtypes(torch.double)
    # 定义测试函数test_assign，接受设备、数据类型和协调参数作为输入
    def test_assign(self, device, dtype, coalesced):
        # 定义内部函数assign_to，用于生成稀疏张量并试图对其进行赋值操作
        def assign_to():
            # 生成稀疏张量a，包括索引i_a和值v_a，使用给定的数据类型和设备
            a, i_a, v_a = self._gen_sparse(2, 5, [2, 3], dtype, device, coalesced)
            # 尝试将稀疏张量a的第一个元素赋值为100，预期会引发TypeError异常
            a[0] = 100

        # 验证assign_to函数确实会引发TypeError异常
        self.assertRaises(TypeError, assign_to)

    # 设置测试函数的数据类型为torch.double和torch.cdouble
    @dtypes(torch.double, torch.cdouble)
    # 定义测试函数test_full_broadcast_to，接受设备和数据类型作为输入
    def test_full_broadcast_to(self, device, dtype):
        # 定义内部函数can_broadcast，用于判断两个形状是否可广播
        def can_broadcast(s0, s1):
            # 反转形状s0和s1，因为broadcast_to需要反向匹配维度
            s0 = tuple(reversed(s0))
            s1 = tuple(reversed(s1))
            # 检查每个维度是否可广播
            for i in range(len(s0)):
                if s0[i] != 1 and s0[i] != s1[i]:
                    return False
            return True

        # 定义不同的形状组合列表sizes，用于测试广播操作
        sizes = (
            (), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)
        )
        # 对sizes中所有形状组合进行两两组合的迭代
        for s0, s1 in itertools.combinations(sizes, r=2):
            # 生成具有给定形状和数据类型的张量t
            t = make_tensor(s0, dtype=dtype, device=device, low=-9, high=9)
            # 将张量t稀疏化为sparse_dims维度的稀疏张量s
            for sparse_dims in range(1, len(s0) + 1):
                s = t.to_sparse(sparse_dims)
                # 如果s0可以广播到s1，则进行广播操作并验证结果
                if can_broadcast(s0, s1):
                    # 使用torch.broadcast_to对张量t进行广播
                    t_res = torch.broadcast_to(t, s1)
                    # 使用torch._sparse_broadcast_to对稀疏张量s进行广播
                    s_res = torch._sparse_broadcast_to(s, s1)
                    # 验证稀疏张量的参数是否有效
                    torch._validate_sparse_coo_tensor_args(s_res._indices(), s_res._values(), s_res.shape)
                    # 如果稀疏张量是协调的，则确保is_coalesced方法的正确性
                    if s_res.is_coalesced():
                        self.assertEqual(s_res, torch.sparse_coo_tensor(s_res._indices(), s_res._values(), s_res.shape).coalesce())
                    # 验证稀疏张量转为密集张量后是否等于广播后的张量t_res
                    self.assertEqual(s_res.to_dense(), t_res)
                else:
                    # 如果s0不能广播到s1，则验证torch._sparse_broadcast_to是否引发预期的异常
                    with self.assertRaisesRegex(RuntimeError,
                                                r"The expanded size of the tensor \(\d\) "
                                                r"must match the existing size \(\d\)"):
                        torch._sparse_broadcast_to(s, s1)

    # 控制是否进行数据协调的装饰器
    @coalescedonoff
    # 设置测试函数的数据类型为torch.double和torch.cdouble
    @dtypes(torch.double, torch.cdouble)
    # 定义测试方法，用于测试稀疏张量的广播操作
    def test_sparse_broadcast_to(self, device, dtype, coalesced):
        # 定义测试函数，生成稀疏张量并转换为稠密张量，进行广播操作后比较结果
        def test(sparse_dims, nnz, with_size, new_size):
            # 生成稀疏张量并转换为稠密张量
            x = self._gen_sparse(sparse_dims, nnz, with_size, dtype, device, coalesced)[0]
            y = self.safeToDense(x)
            # 使用 PyTorch 的 _sparse_broadcast_to 方法进行张量广播
            x1 = torch._sparse_broadcast_to(x, new_size)
            # 使用稠密张量的 broadcast_to 方法进行广播
            y1 = y.broadcast_to(new_size)
            # 断言广播后的稀疏张量和稠密张量的结果是否一致
            self.assertEqual(self.safeToDense(x1), y1)

        # 不同的测试用例调用 test 函数
        test(4, 6, [7, 3, 1, 3, 0], [7, 3, 4, 3, 0])
        test(4, 6, [7, 3, 1, 3, 0], [2, 7, 3, 1, 3, 0])
        test(4, 6, [7, 3, 1, 3, 1, 3], [7, 3, 1, 3, 2, 3])
        test(4, 6, [7, 3, 1, 3, 2, 1], [7, 3, 1, 3, 2, 3])

    # 跳过测试用例的装饰器，根据条件决定是否跳过测试
    def _test_mul_skips(self, device, dtype, coalesced):
        # 默认不跳过测试
        skipTestIfUncoalesced = False
        # 如果输入未合并且数据类型为 float16 或 bfloat16，则跳过测试
        if not coalesced and dtype in {torch.float16, torch.bfloat16}:
            skipTestIfUncoalesced = True
        # 对于布尔类型且在 CUDA 设备上且未合并的张量，跳过测试
        if not coalesced and dtype == torch.bool and torch.device(device).type == "cuda":
            skipTestIfUncoalesced = True

        # 如果需要跳过测试，则输出相应的跳过信息
        if skipTestIfUncoalesced:
            self.skipTest(f"Test with dtype={dtype}, device={device} runs only with coalesced inputs")

    # 设置装饰器，指定函数的输入类型和精度覆盖
    @coalescedonoff
    # 注意：addcmul_out 方法在布尔类型中未实现
    @dtypes(*all_types_and_complex_and(torch.bfloat16, torch.float16))
    @precisionOverride({torch.bfloat16: 1e-2, torch.float16: 1e-2})
    # 定义一个测试方法，用于测试稀疏张量的乘法操作
    def test_sparse_sparse_mul(self, device, dtype, coalesced):
        # 调用内部方法，测试跳过操作
        self._test_mul_skips(device, dtype, coalesced)

        # 定义张量的形状和非零元素数量
        shape = (2, 3, 4, 10)
        nnz = 10

        # 定义内部方法，用于检查稀疏张量的乘法操作结果
        def check(self, x, y):
            # 计算稀疏张量的乘法结果
            res_sparse = x * y
            # 计算稠密张量的乘法结果
            res_dense = x.to_dense() * y.to_dense()
            # 断言稀疏张量乘法结果与稠密张量乘法结果相等
            self.assertEqual(res_sparse.to_dense(), res_dense)

        # 定义内部方法，用于检查空稀疏张量的乘法操作
        def check_empty(sparse_shape, nnz, dense_shape, coalesce):
            from itertools import product
            # 遍历非零元素数量和形状的组合
            for nnz_val, shape_suffix in product((nnz, 0), ((), (0,))):
                # 计算空稀疏张量和对应的稠密张量
                empty_sparse_shape = sparse_shape + shape_suffix
                empty_dense_shape = dense_shape + shape_suffix
                x = self._gen_sparse(sparse_dim, nnz_val, empty_sparse_shape, dtype, device, coalesce)[0]
                # 调用检查方法，检查稀疏张量与自身的乘法结果
                check(self, x, x)

        # 循环遍历张量形状的维度
        for dim in range(len(shape) + 1):
            sub_shape = shape[dim:]
            sparse_dim = len(sub_shape) // 2

            # 调用检查空稀疏张量的方法
            check_empty(sub_shape, nnz, shape, coalesced)

            # 生成稀疏张量并调用检查方法，检查稀疏张量与自身及其他稀疏张量的乘法结果
            x = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            y = self._gen_sparse(sparse_dim, nnz, sub_shape, dtype, device, coalesced)[0]
            check(self, x, y)

            # 检查在稠密维度中的广播操作
            for d in range(sparse_dim, len(sub_shape)):
                new_shape = sub_shape[:d] + (1,) + sub_shape[d + 1:]
                y = self._gen_sparse(sparse_dim, nnz, new_shape, dtype, device, coalesced)[0]
                check(self, x, y)
    # 定义测试方法，验证稀疏 COO 张量的稀疏性质

    # 创建一个 nnz == 0 的 COO 张量，始终是稀疏的
    self.assertTrue(torch.sparse_coo_tensor([[], []], [], (2, 2)).is_coalesced())

    # 创建只有一个 nnz 的 COO 张量，同样是稀疏的
    self.assertTrue(torch.sparse_coo_tensor([[0], [0]], [1], (2, 2)).is_coalesced())

    # 创建两个或更多 nnz 的 COO 张量，由于没有进行昂贵的检查，无法保证其稀疏性
    self.assertFalse(torch.sparse_coo_tensor([[0, 0], [0, 0]], [1, 2], (2, 2)).is_coalesced())

    # 即使没有重复元素，两个或更多 nnz 的 COO 张量同样不是稀疏的
    self.assertFalse(torch.sparse_coo_tensor([[0, 1], [0, 1]], [1, 2], (2, 2)).is_coalesced())

@coalescedonoff
@dtypes(*all_types_and_complex_and(torch.bool))
def test_sum(self, device, dtype, coalesced):
    # 定义测试求和方法，验证稀疏张量的求和功能

    def run_test(shape, nnz):
        # 生成稀疏张量，并运行测试
        a = self._gen_sparse(2, nnz, shape, dtype, device, coalesced)[0]

        # 验证张量的总和
        self.assertEqual(a.sum(), a._values().sum())

        # 对于浮点数或复数类型，进行额外的梯度检查
        if dtype.is_floating_point or dtype.is_complex:
            a.requires_grad_(True)
            a_inter = a.sum()
            a_inter.abs().backward()
            with torch.no_grad():
                self.assertEqual(a.grad, torch.ones(shape, dtype=dtype, device=device) * torch.sgn(a_inter))

    # 针对不同的形状进行测试
    for shape in [(10, 5), (10, 10)]:
        run_test(shape, 0)  # 测试零 nnz 的情况
        run_test(shape, max(shape))  # 测试最大可能 nnz 的情况
        run_test(shape, shape[0] * shape[1])  # 测试达到最大容量的 nnz 的情况
class TestSparseOneOff(TestCase):
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_from_cpu(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"):
            # 尝试创建一个稀疏张量，使用 CUDA 设备上的数据和 CPU 上的索引，预期抛出运行时错误
            torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                    torch.randn(4, 4, 4),
                                    [3, 4, 4])

        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"):
            # 尝试创建一个稀疏张量，使用 CUDA 设备上的数据和 CPU 上的索引，预期抛出运行时错误
            torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                    torch.randn(4, 4, 4, 0),
                                    [3, 4, 4, 0])

        with self.assertRaisesRegex(
                RuntimeError,
                "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!"):
            # 尝试创建一个稀疏张量，使用 CUDA 设备上的数据和 CPU 上的索引，预期抛出运行时错误
            torch.sparse_coo_tensor(torch.empty(1, 0).long().cuda(),
                                    torch.randn(0, 4, 4, 0),
                                    [0, 4, 4, 0])

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_sparse_cpu_dense_add(self):
        x = torch.zeros(3, 4, 4)
        sparse_y = torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                           torch.randn(4, 4, 4).cuda(),
                                           [3, 4, 4])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            # 尝试在 CPU 张量 x 上加法操作稀疏张量 sparse_y，预期抛出运行时错误
            x + sparse_y

        x = torch.zeros(3, 4, 4, 0)
        sparse_y = torch.sparse_coo_tensor(torch.zeros(1, 4).long().cuda(),
                                           torch.randn(4, 4, 4, 0).cuda(),
                                           [3, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            # 尝试在 CPU 张量 x 上加法操作稀疏张量 sparse_y，预期抛出运行时错误
            x + sparse_y

        x = torch.zeros(0, 4, 4, 0)
        sparse_y = torch.sparse_coo_tensor(torch.empty(1, 0).long().cuda(),
                                           torch.randn(0, 4, 4, 0).cuda(),
                                           [0, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'self' to be a CUDA tensor, but got a CPU tensor"):
            # 尝试在 CPU 张量 x 上加法操作稀疏张量 sparse_y，预期抛出运行时错误
            x + sparse_y


def _sparse_to_dense(tensor):
    if tensor.dtype != torch.bool:
        # 如果张量数据类型不是布尔型，将其转换为密集形式，保留梯度信息
        return tensor.to_dense(masked_grad=True)

    # 对于布尔型张量，由于 coalesce 方法不适用，先转换为 int8 类型，再转换为密集形式，最后再转换回布尔型
    return tensor.to(torch.int8).to_dense().to(torch.bool)


_sparse_unary_ops = ops(sparse_unary_ufuncs, dtypes=OpDTypes.supported,
                        allowed_dtypes=all_types_and_complex())
class TestSparseUnaryUfuncs(TestCase):
    exact_dtype = True


    @_sparse_unary_ops
    # 定义一个测试方法，用于检验稀疏张量操作的一致性
    def test_sparse_consistency(self, device, dtype, op):
        # 从操作的样本输入中获取第一个样本
        sample = first_sample(self, op.sample_inputs(device, dtype))
        # 断言输入的样本是一个张量
        assert isinstance(sample.input, torch.Tensor)

        # 计算预期输出，使用操作对输入进行处理
        expected = op(sample.input, *sample.args, **sample.kwargs)
        # 断言预期输出是一个张量
        assert torch.is_tensor(expected)

        # 使用稀疏化后的输入进行操作，计算输出
        output = op(sample.input.to_sparse(), *sample.args, **sample.kwargs)
        # 断言输出是一个张量
        assert torch.is_tensor(output)

        # 断言稀疏输出转为密集张量后与预期输出相等
        self.assertEqual(_sparse_to_dense(output), expected)

    # 用于测试稀疏张量操作中的 `out` 参数的方法修饰器
    @_sparse_unary_ops
    def test_out(self, device, dtype, op):
        # 如果操作不支持 `out` 参数，则跳过测试
        if not op.supports_out:
            self.skipTest("Skipped! Out not supported")

        # 从操作的样本输入中获取第一个样本
        sample = first_sample(self, op.sample_inputs(device, dtype))
        # 将输入样本转换为稀疏张量
        sample.input = sample.input.to_sparse()
        # 计算期望输出
        expect = op(sample.input, *sample.args, **sample.kwargs)

        # 创建一个与期望输出相同形状的稀疏张量 `out`
        out = torch.sparse_coo_tensor(sample.input.shape, device=device,
                                      dtype=expect.dtype)
        # 使用 `out` 参数进行操作
        op(sample.input, *sample.args, **sample.kwargs, out=out)
        # 断言 `out` 的结果与期望输出相等
        self.assertEqual(out, expect)

    # 用于测试稀疏张量操作中的原地操作的方法修饰器
    @_sparse_unary_ops
    def test_inplace(self, device, dtype, op):
        # 如果操作没有原地变体，则跳过测试
        if op.inplace_variant is None:
            self.skipTest("Skipped! Out not supported")

        # 从操作的样本输入中获取第一个样本
        sample = first_sample(self, op.sample_inputs(device, dtype))
        # 将输入样本转换为稀疏张量并压缩
        sample.input = sample.input.to_sparse().coalesce()
        # 计算期望输出
        expect = op(sample.input, *sample.args, **sample.kwargs)

        # 如果无法将期望输出的数据类型转换为指定数据类型，则期望引发运行时错误
        if not torch.can_cast(expect.dtype, dtype):
            with self.assertRaisesRegex(RuntimeError, "result type .* can't be cast to"):
                op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
            return

        # 执行原地操作，获取实际输出
        actual = op.inplace_variant(sample.input, *sample.args, **sample.kwargs)
        # 断言实际输出与期望输出相等
        self.assertIs(actual, sample.input)
        self.assertEqual(actual, expect)

    # 用于测试稀疏张量中零维情况的方法修饰器
    @_sparse_unary_ops
    def test_sparse_zero_dims(self, device, dtype, op):
        # 测试 0x0 的稀疏张量情况
        indices = torch.empty(2, 0, dtype=torch.int64)
        values = torch.empty(0, dtype=dtype)
        sparse_0x0 = torch.sparse_coo_tensor(indices, values, (0, 0))
        # 计算期望输出
        expected = torch.sparse_coo_tensor(indices, op(values), (0, 0))
        # 计算实际输出
        actual = op(sparse_0x0)
        # 断言期望输出与实际输出相等
        self.assertEqual(expected, actual)

    # 用于测试稀疏张量中全零输入的方法修饰器
    @_sparse_unary_ops
    def test_sparse_zeros(self, device, dtype, op):
        # 从操作的样本输入中获取所有样本
        samples = op.sample_inputs(device, dtype)

        # 创建一个零维密集张量和一个零维稀疏张量作为输入
        zero_input = torch.zeros((), device=device, dtype=dtype)
        sparse_input = torch.sparse_coo_tensor((), dtype=dtype, device=device)

        # 计算期望输出
        expect = op(zero_input)
        # 计算实际输出
        actual = op(sparse_input)
        # 断言期望输出与实际输出转为密集张量后相等
        self.assertEqual(expect, _sparse_to_dense(actual))

    # 用于操作修饰器的参数设置，指定支持的操作和数据类型
    @ops(sparse_unary_ufuncs, dtypes=OpDTypes.supported,
         allowed_dtypes=[torch.double, torch.cdouble])
    # 测试稀疏张量函数的梯度计算
    def test_sparse_fn_grad(self, device, dtype, op):
        # 如果操作不支持自动求导，则跳过测试
        if not op.supports_autograd:
            self.skipTest("Skipped! Op doesn't support autograd")

        # 遍历操作的样本输入
        for sample in op.sample_inputs(device, dtype):
            # 将样本输入转换为稀疏张量，并设置为需要梯度计算
            sparse_input = sample.input.to_sparse().detach().requires_grad_(True)

            # 定义函数 fn，将稀疏张量输入转换为密集张量
            def fn(x):
                return _sparse_to_dense(
                    op(x, *sample.args, **sample.kwargs))

            # 执行梯度检查，确保计算梯度正确性
            self.assertTrue(gradcheck(
                fn,
                (sparse_input,),
                check_batched_grad=False,
                check_grad_dtypes=True,
                nondet_tol=op.gradcheck_nondet_tol,
                fast_mode=op.gradcheck_fast_mode,
                masked=True))
# 定义 TestCase 类 TestSparseMaskedReductions，用于测试稀疏掩码减少操作
class TestSparseMaskedReductions(TestCase):
    # 设置确切的数据类型匹配标志为 True
    exact_dtype = True

    # 定义 fp16_low_precision_list 列表，包含需要低精度处理的操作名
    fp16_low_precision_list = {
        'masked.prod',
    }

    # ops 装饰器，应用于 test_future_empty_dim 方法，用于测试稀疏掩码减少操作
    @ops(sparse_masked_reduction_ops)
    def test_future_empty_dim(self, device, dtype, op):
        """Currently, `dim=()` in reductions operations means "reduce over
        all dimensions" while in future, it will read "no reduce". See
        https://github.com/pytorch/pytorch/issues/29137

        For sparse masked reductions, we'll implement the current behavior.

        For testing, we'll use samples with `dim=0` and map it to
        `dim=()` until
        torch.testing._internal.common_methods_invocations._generate_reduction_kwargs
        is made to generate samples with `dim=()` for non-scalar
        inputs. With this and after gh-29137 is resolved, this test
        can be deleted. See also `torch.masked._canonical_dim`
        implementation about changing the `dim=()` behavior.
        """

        # 使用 op 的 sample_inputs_func 方法生成样本输入数据
        samples = op.sample_inputs_func(op, device, dtype, requires_grad=False)
        # 获取操作名称，并去除 'masked.' 前缀
        op_name = op.name.replace('masked.', '')
        
        # 遍历样本输入数据
        for sample_input in samples:
            # 如果样本输入数据的关键字参数 dim 不为 0，则继续下一次循环
            if sample_input.kwargs.get('dim') != 0:
                continue
            
            # 复制样本输入数据的关键字参数并将 dim 设置为 ()，表示在所有维度上进行减少
            sample_input_kwargs = dict(sample_input.kwargs)
            sample_input_kwargs['dim'] = ()    # reduce over all dimensions

            # 获取输入张量 t
            t = sample_input.input
            # 获取样本输入数据的 mask 参数
            mask = sample_input_kwargs.get('mask')
            
            # 如果 mask 为 None，并且操作名在 {'prod', 'amax', 'amin'} 中
            if mask is None and op_name in {'prod', 'amax', 'amin'}:
                # FIXME: 目前对于稀疏 COO 张量，不支持具有非零减少身份和未指定 mask 的减少操作
                # 详见 torch.masked.prod 的实现细节。
                continue
            
            # 复制稀疏操作的关键字参数
            sparse_op_kwargs = dict(sample_input_kwargs)
            # 使用 op 对稀疏版本的输入张量进行操作，获取实际结果
            actual = op(t.to_sparse(), *sample_input.args, **sample_input_kwargs)
            # 断言实际结果的布局为稀疏 COO 格式
            self.assertEqual(actual.layout, torch.sparse_coo)

            # 使用 op 对原始输入张量进行操作，获取预期的稀疏结果
            expected = op(t, *sample_input.args, **sample_input_kwargs).to_sparse()
            atol = None
            rtol = None
            # 如果操作名在 fp16_low_precision_list 中，并且数据类型为 torch.half
            if op.name in self.fp16_low_precision_list and dtype == torch.half:
                # 设置允许的绝对误差和相对误差
                atol = 1e-5
                rtol = 2e-3
            # 断言实际结果等于预期结果，使用设置的绝对误差和相对误差
            self.assertEqual(actual, expected, atol=atol, rtol=rtol)


# 定义 TestCase 类 TestSparseMeta，用于测试稀疏元数据
class TestSparseMeta(TestCase):
    # 设置确切的数据类型匹配标志为 True
    exact_dtype = True
    # 定义一个测试方法，用于测试稀疏 COO 格式的张量操作，接受数据类型作为参数
    def _test_meta_sparse_coo(self, dtype):
        # 创建一个空的 4x4 稀疏 COO 格式张量，布局为 torch.sparse_coo，设备为 'meta'，指定数据类型为 dtype
        r = torch.empty(4, 4, layout=torch.sparse_coo, device='meta', dtype=dtype)
        # 断言张量 r 是 meta 设备的元数据张量
        self.assertTrue(r.is_meta)
        # 断言张量 r 的设备类型为 "meta"
        self.assertEqual(r.device.type, "meta")
        # 创建一个与 r 相同大小的空张量 r2
        r2 = torch.empty_like(r)
        # 断言张量 r2 是 meta 设备的元数据张量
        self.assertTrue(r2.is_meta)
        # 断言张量 r 和 r2 在值上完全相等
        self.assertEqual(r, r2)
        # 创建一个空的 4x4 稀疏 COO 格式张量 r3，设备为 'meta'，指定数据类型为 dtype
        r3 = torch.sparse_coo_tensor(size=(4, 4), device='meta', dtype=dtype)
        # 断言张量 r3 是 meta 设备的元数据张量
        self.assertTrue(r3.is_meta)
        # 断言张量 r 和 r3 在值上完全相等
        self.assertEqual(r, r3)
        # 将张量 r 的稀疏尺寸调整为 (4, 4)，并设置其稀疏维度和稠密维度为 1
        r.sparse_resize_((4, 4), 1, 1)
        # 将张量 r 的稀疏尺寸调整为 (4, 4, 4)，并设置其稀疏维度和稠密维度为 2和1
        r.sparse_resize_and_clear_((4, 4, 4), 2, 1)
        # 断言张量 r 的稀疏维度为 2
        self.assertEqual(r.sparse_dim(), 2)
        # 断言张量 r 的稠密维度为 1
        self.assertEqual(r.dense_dim(), 1)
        # 断言张量 r 的 _dimV() 返回 1
        self.assertEqual(r._dimV(), 1)
        # 断言张量 r 的 _nnz() 返回 0，即非零元素个数为 0
        self.assertEqual(r._nnz(), 0)
        # 断言张量 r 是否被压缩，预期为 True，因为零元素稀疏张量在创建时总是被压缩的
        self.assertEqual(r.is_coalesced(), True)
        # 强制张量 r 进入未压缩状态
        r._coalesced_(False)
        # 断言张量 r 是否未被压缩，预期为 False
        self.assertEqual(r.is_coalesced(), False)
        # 将张量 r 恢复为压缩状态，以便进行索引/值的访问
        r._coalesced_(True)
        # TODO: 这种类型的别名操作可能需要通过功能化处理
        # 断言张量 r 的 _indices() 返回一个空的大小为 (2, 0) 的 int64 类型张量，设备为 'meta'
        self.assertEqual(r._indices(), torch.empty(2, 0, device='meta', dtype=torch.int64))
        # 断言张量 r 的 _values() 返回一个空的大小为 (0, 4) 的 dtype 类型张量，设备为 'meta'
        self.assertEqual(r._values(), torch.empty(0, 4, device='meta', dtype=dtype))
        # 断言张量 r 的 indices() 返回一个空的大小为 (2, 0) 的 int64 类型张量，设备为 'meta'
        self.assertEqual(r.indices(), torch.empty(2, 0, device='meta', dtype=torch.int64))
        # 断言张量 r 的 values() 返回一个空的大小为 (0, 4) 的 dtype 类型张量，设备为 'meta'
        self.assertEqual(r.values(), torch.empty(0, 4, device='meta', dtype=dtype))
    # 定义测试方法，用于测试稀疏压缩张量的元信息
    def _test_meta_sparse_compressed(self, dtype, index_dtype, layout, batchsize, densesize):
        # 强制将索引数据类型设置为 torch.int64
        index_dtype = torch.int64
        # 根据布局选择块大小，若布局为 torch.sparse_bsr 或 torch.sparse_bsc，则设置为 (2, 3)，否则为空元组
        blocksize = (2, 3) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        # 设置稀疏维度的大小为 (4, 6)
        sparsesize = (4, 6)
        # 非零元素数量设为 0
        nnz = 0

        # 计算张量的形状，包括批次大小、稀疏大小和密集大小
        shape = (*batchsize, *sparsesize, *densesize)
        # 根据布局确定压缩的维度，若布局为 torch.sparse_csr 或 torch.sparse_bsr，则为 0，否则为 1
        compressed_dim = 0 if layout in {torch.sparse_csr, torch.sparse_bsr} else 1
        # 计算压缩索引的数量
        nof_compressed_indices = (sparsesize[compressed_dim] // blocksize[compressed_dim] + 1 if blocksize
                                  else sparsesize[compressed_dim] + 1)
        # 创建一个空的压缩索引张量，设备为 'meta'，数据类型为 index_dtype
        compressed_indices = torch.empty((*batchsize, nof_compressed_indices), device='meta', dtype=index_dtype)
        # 创建一个空的普通索引张量，设备为 'meta'，数据类型为 index_dtype
        plain_indices = torch.empty((*batchsize, nnz), device='meta', dtype=index_dtype)

        # 创建一个空的值张量，设备为 'meta'，数据类型为 dtype
        values = torch.empty((*batchsize, nnz, *blocksize, *densesize), device='meta', dtype=dtype)
        # 创建稀疏压缩张量 r
        r = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            shape,
            layout=layout
        )

        # 断言 r 是一个元信息张量
        self.assertTrue(r.is_meta)
        # 断言 r 的设备类型为 "meta"
        self.assertEqual(r.device.type, "meta")

        # 断言 r 的稀疏维度为 2
        self.assertEqual(r.sparse_dim(), 2)
        # 断言 r 的密集维度等于 densesize 的长度
        self.assertEqual(r.dense_dim(), len(densesize))
        # 断言 r 的非零元素数量为 nnz
        self.assertEqual(r._nnz(), nnz)
        
        # 计算批次维度数
        batch_dims = r.ndim - r.sparse_dim() - r.dense_dim()
        # 获取 r 的块大小
        r_blocksize = r.values().shape[batch_dims + 1: batch_dims + 1 + len(blocksize)]
        # 断言 r 的块大小与预期的 blocksize 相等
        self.assertEqual(r_blocksize, blocksize)

        # 根据布局选择 r 的压缩索引
        r_compressed_indices = r.crow_indices() if layout in {torch.sparse_csr, torch.sparse_bsr} else r.ccol_indices()
        # 根据布局选择 r 的普通索引
        r_plain_indices = r.col_indices() if layout in {torch.sparse_csr, torch.sparse_bsr} else r.row_indices()

        # 断言 r 的压缩索引为空张量，形状和设备与预期相符
        self.assertEqual(r_compressed_indices,
                         torch.empty((*batchsize, nof_compressed_indices), device='meta', dtype=index_dtype))
        # 断言 r 的普通索引为空张量，形状和设备与预期相符
        self.assertEqual(r_plain_indices, torch.empty((*batchsize, nnz), device='meta', dtype=index_dtype))
        # 断言 r 的值张量为空张量，形状和设备与预期相符
        self.assertEqual(r.values(), torch.empty((*batchsize, nnz, *blocksize, *densesize), device='meta', dtype=dtype))

        # 创建一个与 r 形状相同的空张量 r2
        r2 = torch.empty_like(r)
        # 断言 r2 是一个元信息张量
        self.assertTrue(r2.is_meta)
        # 断言 r2 与 r 相等
        self.assertEqual(r2, r)

        # 如果布局为 torch.sparse_csr 或 torch.sparse_csc
        if layout in {torch.sparse_csr, torch.sparse_csc}:
            # 创建一个空的稀疏张量 r3，形状为 (*batchsize, *sparsesize)，数据类型为 dtype，布局为 layout，设备为 "meta"
            r3 = torch.empty((*batchsize, *sparsesize), dtype=dtype, layout=layout, device="meta")
            # 断言 r3 是一个元信息张量
            self.assertTrue(r3.is_meta)
            # 如果 densesize 为空，断言 r3 与 r 相等
            if not densesize:
                # dense dimensions cannot be specified for torch.empty
                self.assertEqual(r3, r)
    # 定义测试函数，测试稀疏张量的元数据打印
    def test_meta(self, dtype, layout):
        # 如果布局是稀疏 COO 格式
        if layout is torch.sparse_coo:
            # 调用稀疏 COO 格式的元数据测试函数
            self._test_meta_sparse_coo(dtype)
        else:
            # 默认使用 torch.int64 类型的索引数据类型
            index_dtype = torch.int64
            # 使用 itertools.product 迭代所有组合，包括空元组和 (2,) 组合
            for batchsize, densesize in itertools.product([(), (2,)], [(), (3,)]):
                # 调用稀疏压缩格式的元数据测试函数
                self._test_meta_sparse_compressed(dtype, index_dtype, layout, batchsize, densesize)

    # 定义内部函数，用于打印稀疏张量的元数据
    def _test_print_meta_data(self, dtype, layout, batchsize, sparsesize, densesize):
        # 默认使用 torch.int64 类型的索引数据类型
        index_dtype = torch.int64
        # 初始化非零元素数目
        nnz = 0
        # 如果布局为稀疏 BSR 或 BSC，则设置块大小为 (2, 3)，否则为空元组
        blocksize = (2, 3) if layout in {torch.sparse_bsr, torch.sparse_bsc} else ()
        # 设置张量形状，包括批处理大小、稀疏尺寸、密集尺寸
        shape = (*batchsize, *sparsesize, *densesize)
        # 创建空的稀疏值张量，设备为 'meta'，指定数据类型和形状
        values = torch.empty((*batchsize, nnz, *blocksize, *densesize), device='meta', dtype=dtype)
        # 如果布局为稀疏 COO 格式
        if layout is torch.sparse_coo:
            # 创建空的稀疏 COO 张量的索引张量，设备为 'meta'，数据类型为 index_dtype
            indices = torch.empty((len(sparsesize), nnz), device='meta', dtype=index_dtype)
            # 创建稀疏 COO 张量
            x = torch.sparse_coo_tensor(indices, values, shape)
        else:
            # 确定压缩维度：如果布局为 CSR 或 BSR，则压缩维度为 0，否则为 1
            compressed_dim = 0 if layout in {torch.sparse_csr, torch.sparse_bsr} else 1
            # 计算压缩索引数目
            nof_compressed_indices = (sparsesize[compressed_dim] // blocksize[compressed_dim] + 1 if blocksize
                                      else sparsesize[compressed_dim] + 1)
            # 创建压缩索引张量和普通索引张量
            compressed_indices = torch.empty((*batchsize, nof_compressed_indices), device='meta', dtype=index_dtype)
            plain_indices = torch.empty((*batchsize, nnz), device='meta', dtype=index_dtype)
            # 创建稀疏压缩张量
            x = torch.sparse_compressed_tensor(
                compressed_indices,
                plain_indices,
                values,
                shape,
                layout=layout
            )

        # 初始化打印内容列表
        printed = []
        # 添加标题行到打印内容列表
        printed.append(f"########## {dtype}/{index_dtype}/size={batchsize}+{sparsesize}+{blocksize}+{densesize} ##########")
        # 添加稀疏元数据张量的注释行
        printed.append("# sparse meta tensor")
        # 将张量 x 的字符串表示添加到打印内容列表
        printed.append(str(x))

        # 返回打印内容列表
        return printed

    # 使用所有稀疏布局进行参数化的测试函数，测试稀疏张量的元数据打印
    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_print_meta(self, dtype, layout):
        # 初始化打印内容列表
        printed = []
        # 使用 itertools.product 迭代所有组合，包括空元组和 (2,) 组合
        for batchsize, sparsesize, densesize in itertools.product(
                [(), (2,)], [(4, 6), (3, 5, 7)], [(), (3,)]
        ):
            # 如果布局为稀疏 COO 并且批处理大小不为空
            if layout is torch.sparse_coo and batchsize:
                # 跳过，因为 COO 张量不支持批处理维度
                continue
            # 如果布局不为稀疏 COO 并且稀疏尺寸的长度不为 2
            if layout is not torch.sparse_coo and len(sparsesize) != 2:
                # 跳过，因为 CSR/CSC/BSR/BSC 张量必须有 2 个稀疏维度
                continue
            # 调用内部函数，获取打印内容列表，并添加到 printed 列表中
            printed += self._test_print_meta_data(dtype, layout, batchsize, sparsesize, densesize)

        # 保存原始的最大差异值
        orig_maxDiff = self.maxDiff
        # 设置最大差异值为 None，以便比较输出时忽略差异
        self.maxDiff = None
        try:
            # 断言打印内容与预期输出相符
            self.assertExpected('\n'.join(printed))
            # 恢复原始的最大差异值
            self.maxDiff = orig_maxDiff
        except Exception:
            # 如果断言失败，恢复原始的最大差异值并抛出异常
            self.maxDiff = orig_maxDiff
            raise
    def assertEqualMeta(self, x, y, expected_nnz):
        # 断言两个张量的布局（稀疏/密集）、形状、数据类型、稀疏维度和密集维度是否相等
        self.assertEqual(x.layout, y.layout)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x.dtype, y.dtype)
        self.assertEqual(x.sparse_dim(), y.sparse_dim())
        self.assertEqual(x.dense_dim(), y.dense_dim())

        def assertEqualAttrs(x, y, expected_shape):
            # 断言两个张量属性的形状、数据类型、布局是否相等，如果不是元数据，则断言设备是否相等
            self.assertEqual(x.shape, expected_shape)
            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x.layout, y.layout)
            if not x.is_meta:
                self.assertEqual(x.device, y.device)

        if x.layout is torch.sparse_coo:
            # 如果布局是稀疏 COO，则断言稀疏索引和值的属性是否相等，期望非零元素个数为 expected_nnz
            assertEqualAttrs(x._indices(), y._indices(), (*y._indices().shape[:-1], expected_nnz))
            assertEqualAttrs(x._values(), y._values(), (expected_nnz, *y._values().shape[1:]))
        elif x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            # 如果布局是稀疏 CSR 或 BSR，则断言行索引和列索引的属性是否相等，以及值的形状是否符合预期
            assertEqualAttrs(x.crow_indices(), y.crow_indices(), y.crow_indices().shape)
            assertEqualAttrs(x.col_indices(), y.col_indices(), (*y.col_indices().shape[:-1], expected_nnz))
            batch_dim = x.col_indices().ndim - 1
            values_shape = (*y.values().shape[:batch_dim], expected_nnz, *y.values().shape[batch_dim + 1:])
            self.assertEqual(x.values().layout, y.values().layout)
            self.assertEqual(x.values().dtype, y.values().dtype)
            self.assertEqual(x.values().shape, values_shape)
        elif x.layout in {torch.sparse_csc, torch.sparse_bsc}:
            # 如果布局是稀疏 CSC 或 BSC，则断言列索引和行索引的属性是否相等，以及值的形状是否符合预期
            assertEqualAttrs(x.ccol_indices(), y.ccol_indices(), y.ccol_indices().shape)
            assertEqualAttrs(x.row_indices(), y.row_indices(), (*y.row_indices().shape[:-1], expected_nnz))
            batch_dim = x.row_indices().ndim - 1
            values_shape = (*y.values().shape[:batch_dim], expected_nnz, *y.values().shape[batch_dim + 1:])
            self.assertEqual(x.values().layout, y.values().layout)
            self.assertEqual(x.values().dtype, y.values().dtype)
            self.assertEqual(x.values().shape, values_shape)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_to_meta(self, dtype, layout):
        # 测试将输入张量转换为元数据类型时的断言
        index_dtype = torch.int64
        device = 'cpu'
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            m = t.to(device="meta")
            self.assertEqual(m.device.type, "meta")
            self.assertEqualMeta(m, t, 0)

    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_zeros_like_meta(self, dtype, layout):
        # 测试生成与输入张量相同形状的元数据类型零张量时的断言
        index_dtype = torch.int64
        device = 'cpu'
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            m = torch.zeros_like(t, device="meta")
            self.assertEqual(m.device.type, "meta")
            self.assertEqualMeta(m, t, 0)

    @all_sparse_layouts('layout', include_strided=False)
    # 使用参数化测试，测试特定数据类型的假张量
    @parametrize("dtype", [torch.float64])
    def test_fake(self, dtype, layout):
        # 导入假张量相关的类和函数
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
        # 创建假张量模式对象
        fake_mode = FakeTensorMode()
        # 索引数据类型为 torch.int64
        index_dtype = torch.int64
        # 设备为 CPU
        device = 'cpu'
        # 遍历生成简单输入数据的生成器
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            # 将真实张量 t 转换为假张量 f
            f = FakeTensor.from_tensor(t, fake_mode)
            # 断言 f 是 FakeTensor 类的实例
            self.assertIsInstance(f, FakeTensor)
            # 断言 f 和真实张量 t 在元数据上相等
            self.assertEqualMeta(f, t, 0)

            # 分离假张量 f 的副本 d
            d = f.detach()
            # 断言 d 是 FakeTensor 类的实例
            self.assertIsInstance(d, FakeTensor)
            # 断言 d 和真实张量 t 在元数据上相等
            self.assertEqualMeta(d, t, 0)

    # 使用参数化测试，测试 torch.zeros_like 生成的假张量
    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_zeros_like_fake(self, dtype, layout):
        # 导入假张量相关的类和函数
        from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
        # 导入模式工具函数
        from torch.utils._mode_utils import no_dispatch
        # 创建假张量模式对象
        fake_mode = FakeTensorMode()
        # 索引数据类型为 torch.int64
        index_dtype = torch.int64
        # 设备为 CPU
        device = 'cpu'
        # 遍历生成简单输入数据的生成器
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            # 将真实张量 t 转换为假张量 f
            f = FakeTensor.from_tensor(t, fake_mode)
            # 生成与真实张量 t 形状相同的零张量 expected
            expected = torch.zeros_like(t)
            # 在没有调度的情况下，生成假张量的零张量 result
            with no_dispatch():
                result = torch.zeros_like(f, device=f.fake_device)
            # 断言 result 和 expected 在值上相等
            self.assertEqual(result, expected)
            # 断言 result 和 expected 在元数据上相等
            self.assertEqualMeta(result, expected, 0)

    # 使用参数化测试，测试元数据相关的求和操作
    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_sum_meta(self, dtype, layout):
        # 设备为 CPU
        device = 'cpu'
        # 索引数据类型为 torch.int64
        index_dtype = torch.int64
        # 遍历生成简单输入数据的生成器
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            # 将真实张量 t 转换为元数据张量 m
            m = t.to(device='meta')
            # 计算元数据张量 m 的求和 r
            r = torch.sum(m)
            # 计算真实张量 t 的求和，并转换为元数据形式，作为期望值 expected
            expected = torch.sum(t).to(device="meta")
            # 断言 r 是元数据张量
            self.assertTrue(r.is_meta)
            # 断言 r 和 expected 在元数据上相等
            self.assertEqualMeta(r, expected, 0)

    # 使用参数化测试，测试元数据相关的加法操作
    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("dtype", [torch.float64])
    def test_add_meta(self, dtype, layout):
        # 设备为 CPU
        device = 'cpu'
        # 索引数据类型为 torch.int64
        index_dtype = torch.int64
        # 遍历生成简单输入数据的生成器
        for t in self.generate_simple_inputs(layout, device=device, dtype=dtype, index_dtype=index_dtype):
            # 计算真实张量 t 的加法操作的期望值，并转换为元数据形式
            expected = torch.add(t, t).to(device='meta')
            # 将真实张量 t 转换为元数据张量 m
            m = t.to(device='meta')
            # 计算元数据张量 m 的加法操作结果 r
            r = torch.add(m, m)
            # 断言 r 和 expected 在元数据上相等
            self.assertEqualMeta(r, expected, 0)
class _SparseDataset(torch.utils.data.Dataset):
    # An utility class used in TestSparseAny.test_dataloader method.

    def __init__(self, sparse_tensors):
        # 初始化方法，接收稀疏张量列表作为参数
        self.sparse_tensors = sparse_tensors

    def __len__(self):
        # 返回稀疏张量列表的长度作为数据集的长度
        return len(self.sparse_tensors)

    def __getitem__(self, index):
        # 根据索引返回对应位置的稀疏张量
        return self.sparse_tensors[index]


class TestSparseAny(TestCase):

    @onlyCPU
    @all_sparse_layouts('layout', include_strided=False)
    @torch.sparse.check_sparse_tensor_invariants(enable=False)
    @all_sparse_layouts('layout', include_strided=False)
    @all_sparse_layouts('from_layout', include_strided=False)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize("index_dtype", [torch.int32, torch.int64])
    def test_to_dense(self, from_layout, device, dtype, index_dtype):
        """
        This test tests conversion from any layout to strided layout.
        """
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype):
            # 将稀疏张量 t 转换为密集张量 r
            r = t.to_dense()
            # 断言转换后的密集张量布局为 strided
            self.assertEqual(r.layout, torch.strided)
            # 断言转换后的密集张量与原始稀疏张量 t 相等
            self.assertEqual(r, t)

    @all_sparse_layouts('from_layout', include_strided=False)
    @dtypes(torch.float64, torch.complex128)
    @parametrize("index_dtype", [torch.int64])
    @gradcheck_semantics()
    def test_gradcheck_to_dense(self, from_layout, device, dtype, index_dtype, gradcheck):
        for t in self.generate_simple_inputs(
                from_layout, device=device, dtype=dtype, index_dtype=index_dtype):
            batch_dim = t.dim() - t.dense_dim() - t.sparse_dim()
            if batch_dim > 0:
                # 如果批处理维度大于0，则跳过此测试
                # TODO: implement batch support in _convert_indices_from_csr_to_coo
                continue
            # 克隆张量 t 并设置为需要梯度计算
            t = t.clone().detach().requires_grad_(True)
            # 使用 gradcheck 方法进行梯度检查，验证转换为密集张量的梯度计算
            r = gradcheck(lambda x: torch.Tensor.to_dense(x, masked_grad=gradcheck.masked), t)
            # 断言梯度检查通过
            self.assertTrue(r)

    @all_sparse_layouts('from_layout', include_strided=True)
    @all_sparse_layouts('to_layout', include_strided=False)
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    @parametrize("index_dtype", [torch.int32, torch.int64])
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(reduction_ops_with_sparse_support)
    @precisionOverride({torch.bfloat16: 5e-4, torch.float16: 5e-3})
    @all_sparse_layouts('layout', include_strided=False)
    # 定义测试函数，用于测试稀疏张量操作的减少(reduction)
    def test_reductions(self, layout, device, dtype, op):
        # 初始化计数器
        count = 0
        # 遍历操作对象op生成的稀疏输入样本
        for sample in op.sample_inputs_sparse(layout, device, dtype):
            # 计数器加一
            count += 1

            # 获取样本的输入、参数和关键字参数
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            # 对输入执行操作op，并保存结果
            result = op.op(t_inp, *t_args, **t_kwargs)

            # 检查不变性：rop(inp, ...).to_dense() == rop(inp.to_dense(), ...)
            dense = op.op(t_inp.to_dense(), *t_args, **t_kwargs)
            self.assertEqual(result, dense)

        # 如果没有样本输入，则跳过测试并输出消息
        if count == 0:
            self.skipTest('no sample inputs')

    # 标记为仅适用于本地设备类型的测试
    # 压制警告
    # 使用具有稀疏支持的减少操作和指定的数据类型进行操作
    # 对所有稀疏布局进行测试，排除步进的布局
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(reduction_ops_with_sparse_support, allowed_dtypes=(torch.float32, torch.float64, torch.complex64, torch.complex128))
    @all_sparse_layouts('layout', include_strided=False)
    def test_reductions_backward(self, layout, device, dtype, op):
        # 初始化计数器
        count = 0
        # 遍历操作对象op生成的需要梯度的稀疏输入样本
        for sample in op.sample_inputs_sparse(layout, device, dtype, requires_grad=True):
            # 获取样本的输入、参数和关键字参数
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            # 对输入执行操作op，并保存结果
            r = op.op(t_inp, *t_args, **t_kwargs)
            # 如果结果的元素数不为零，则对结果进行求和
            if r.numel() != 0:
                r = r.sum()

            # 如果操作名为'sum'，则执行以下操作
            if op.name == 'sum':
                count += 1
                # 计算结果的绝对值的梯度
                r.abs().backward()
                # 检查输入的梯度是否等于全1向量乘以结果的符号值
                self.assertEqual(t_inp.grad, torch.ones(t_inp.shape, dtype=dtype, device=device) * torch.sgn(r))
            else:
                # 如果操作不是'sum'，则跳过测试并输出消息
                self.skipTest('NOT IMPL')

        # 如果没有样本输入，则跳过测试并输出消息
        if count == 0:
            self.skipTest('no sample inputs')

    # 标记为仅适用于本地设备类型的测试
    # 压制警告
    # 参数化测试方法，测试包括子测试对象和指定的方法列表
    # 对所有稀疏布局进行测试，包括步进的布局
    @onlyNativeDeviceTypes
    @suppress_warnings
    @parametrize("mth", [subtest(mth, name=mth.__name__)
                         for mth in [torch.Tensor.is_coalesced,
                                     torch.Tensor.coalesce,
                                     torch.Tensor.indices,
                                     torch.Tensor.values,
                                     torch.Tensor.crow_indices,
                                     torch.Tensor.col_indices,
                                     torch.Tensor.ccol_indices,
                                     torch.Tensor.row_indices,
                                     ]])
    @all_sparse_layouts('layout', include_strided=True)
    # 定义测试方法，验证不支持的后端错误消息
    def test_unsupported_backend_error_message(self, mth, layout, device):
        # 创建稀疏张量 inp，根据布局设置不同的块大小
        inp = torch.tensor([[1, 2], [3, 4]], device=device).to_sparse(
            layout=layout,
            blocksize=(1, 1) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None)
        # 断言稀疏张量的布局与给定布局相同
        assert inp.layout is layout

        # 定义期望的行为字典，根据方法名称选择相应的布局和异常消息
        expected_behaviour = dict(
            is_coalesced=({torch.sparse_coo},
                          "is_coalesced expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            coalesce=({torch.sparse_coo},
                      "coalesce expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            indices=({torch.sparse_coo},
                     "indices expected sparse coordinate tensor layout but got (Sparse(Csr|Csc|Bsr|Bsc)|Strided)"),
            values=({torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc},
                    "values expected sparse tensor layout but got Strided"),
            crow_indices=({torch.sparse_csr, torch.sparse_bsr},
                          "crow_indices expected sparse row compressed tensor layout but got (Sparse(Csc|Bsc|)|Strided)"),
            col_indices=({torch.sparse_csr, torch.sparse_bsr},
                         "col_indices expected sparse row compressed tensor layout but got (Sparse(Csc|Bsc|)|Strided)"),
            ccol_indices=({torch.sparse_csc, torch.sparse_bsc},
                          "ccol_indices expected sparse column compressed tensor layout but got (Sparse(Csr|Bsr|)|Strided)"),
            row_indices=({torch.sparse_csc, torch.sparse_bsc},
                         "row_indices expected sparse column compressed tensor layout but got (Sparse(Csr|Bsr|)|Strided)"),
        )[mth.__name__]

        # 根据当前布局选择期望行为的布局集合
        if layout in expected_behaviour[0]:
            # 如果当前布局在支持的布局集合中，则调用被测试的方法 mth(inp)
            mth(inp)
        else:
            # 如果当前布局不在支持的布局集合中，则期望抛出 RuntimeError 异常，并验证异常消息是否符合预期
            with self.assertRaisesRegex(RuntimeError, expected_behaviour[1]):
                mth(inp)

    # 应用测试装饰器，指定只在原生设备类型上运行测试
    @onlyNativeDeviceTypes
    # 应用测试装饰器，指定所有稀疏布局，不包括 Strided 布局
    @all_sparse_layouts('layout', include_strided=not True)
    # 应用参数化装饰器，指定 masked 参数为 False 和 True 的两种子测试
    @parametrize("masked", [subtest(False, name='sparse'), subtest(True, name='masked')])
    # 应用参数化装饰器，指定 fast_mode 参数为 False 和 True 的两种子测试
    @parametrize("fast_mode", [subtest(False, name='slow'), subtest(True, name='fast')])
    # 定义一个测试函数，用于梯度检查矩阵乘法操作
    def test_gradcheck_mm(self, layout, dtype, device, masked, fast_mode):
        # 不检查以下情况:
        # - 批量或混合张量，因为 addmm 尚不支持这些输入
        # - 当 check_forward_ad=True 时，由于在 aten::view_as_real、torch._VF._make_dual 等函数中不支持稀疏张量
        # 创建参考输入张量 ref_x 和 ref_y
        ref_x = torch.tensor([[1, 2, 0, 0],
                              [0, 6, 0, 0],
                              [0, 0, 0, 0],
                              [13, 14, 0, 15]], dtype=dtype, device=device)
        ref_y = torch.tensor([[11, 12, 13, 14],
                              [21, 22, 23, 24],
                              [31, 32, 33, 34],
                              [41, 42, 43, 44]],
                             dtype=dtype, device=device)

        # 根据是否需要遮罩选择稀疏矩阵乘法函数
        mm = torch.sparse.mm if masked else torch.mm

        # 根据布局选择块大小，如果布局是 torch.sparse_bsr 或 torch.sparse_bsc，则设置为 (2, 2)，否则为 None
        blocksize = (2, 2) if layout in {torch.sparse_bsr, torch.sparse_bsc} else None
        # 将 ref_x 转换为稀疏张量，并设置 requires_grad=True
        x = ref_x.to_sparse(layout=layout, blocksize=blocksize).requires_grad_(True)
        # 设置 ref_y 的 requires_grad=True
        y = ref_y.requires_grad_(True)

        # 处理特定布局和遮罩的情况
        if layout is torch.sparse_bsr and not masked or layout is torch.sparse_bsc:
            # 断言在运行时会抛出 RuntimeError，并包含特定错误信息
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"addmm: computation on (CPU|CUDA) is not implemented for Strided \+ Sparse(Bsr|Bsc) @ Strided"):
                # 执行梯度检查
                torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
            # 跳过当前测试，标记为未实现
            self.skipTest('NOT IMPL')
        elif layout in {torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc} and masked:
            # 断言在运行时会抛出 RuntimeError，并包含特定错误信息
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"(sparse_addmm_sparse_backward: unsupported combination of layouts,"
                    r" grad: Strided, mat1: Sparse(Csc|Bsr|Bsc), mat2: Strided"
                    r"|addmm: computation on (CPU|CUDA) is not implemented for "
                    r"Strided \+ Sparse(Csc|Bsr|Bsc) @ Strided without MKL)"):
                # 执行梯度检查
                torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)
            # 跳过当前测试，标记为未实现
            self.skipTest('NOT IMPL')
        else:
            # 普通情况下执行梯度检查
            torch.autograd.gradcheck(mm, (x, y), fast_mode=fast_mode, masked=masked)

    # 使用装饰器指定仅适用于本地设备类型的测试
    @onlyNativeDeviceTypes
    # 使用装饰器抑制警告
    @suppress_warnings
    # 使用装饰器将函数注册为具有稀疏支持的二元 ufunc 操作
    @ops(binary_ufuncs_with_sparse_support)
    # 使用装饰器指定适用于所有稀疏布局的操作，不包括 Strided 布局
    @all_sparse_layouts('layout', include_strided=False)
    # 定义测试函数，用于测试稀疏操作
    def test_binary_operation(self, layout, device, dtype, op):
        # 检查操作是否支持给定的稀疏布局，如果不支持则跳过测试
        if not op.supports_sparse_layout(layout):
            self.skipTest(f'{layout} is not supported in `{op.name}` OpInfo definition. Skipping!')

        # 遍历操作生成的稀疏输入样本
        for sample in op.sample_inputs_sparse(layout, device, dtype):
            # 如果验证稀疏输入样本失败，则跳过该样本
            if validate_sample_input_sparse(op, sample, check_validate=False) is not sample:
                # 如果验证返回了包含错误输入实例的稀疏样本，则继续下一个循环
                continue

            # 从样本中获取输入张量及其参数
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            # 计算操作的结果
            result = op.op(t_inp, *t_args, **t_kwargs)

            # 检查结果张量的形状是否与输入张量相同
            self.assertEqual(result.shape, t_inp.shape)

            # 检查结果张量的稀疏维度是否与输入张量相同
            self.assertEqual(result.sparse_dim(), t_inp.sparse_dim())

            # 检查结果张量的稠密维度是否与输入张量相同
            self.assertEqual(result.dense_dim(), t_inp.dense_dim())

            # 检查不变性条件：rop(inp, ...).to_dense() == rop(inp.to_dense(), ...)
            try:
                # 尝试使用稠密张量进行操作，并比较结果
                dense = op.op(t_inp.to_dense(), *(t_args[0].to_dense(), *t_args[1:]), **t_kwargs)
            except Exception as msg:
                # 处理特定异常情况：跳过样本的处理
                if "\"cpublas_axpy_impl\" not implemented for 'ComplexHalf'" in str(msg):
                    # 如果遇到特定未实现的操作异常，则静默跳过该样本处理
                    continue
                # 其他异常情况则抛出异常
                raise
            # 比较稀疏操作和稠密操作的结果是否相等
            self.assertEqual(result, dense)

    # 定义只在 CPU 上运行的测试函数，测试稀疏张量转换为自身的操作
    @onlyCPU
    @all_sparse_layouts('layout', include_strided=True)
    @dtypes(torch.double)
    def test_to_sparse_identity(self, device, layout, dtype):
        # 遍历不同稠密维度的情况
        for dense_dim in range(4):
            # 创建单位对角矩阵作为稠密张量输入
            x_dense = torch.eye(dense_dim, dtype=dtype, device=device)
            # 遍历不同的稀疏维度输入
            for sparse_dim_in in range(1, dense_dim):
                # 将稠密张量转换为指定的稀疏维度输入
                x_sparse = x_dense.to_sparse(sparse_dim_in)
                # 遍历不同的稀疏维度输出
                for sparse_dim_out in range(0, dense_dim):
                    if sparse_dim_out == sparse_dim_in:
                        # 如果输入输出的稀疏维度相同，检查转换后的稀疏维度是否正确
                        self.assertTrue(x_sparse.to_sparse(sparse_dim_out).sparse_dim() == sparse_dim_out)
                    else:
                        # 如果输入输出的稀疏维度不同，检查转换时是否抛出预期的异常
                        with self.assertRaisesRegex(
                                RuntimeError,
                                r"to_sparse: conversion from Sparse to Sparse with sparse_dim argument !=self.sparse_dim\(\)"
                                " is not supported"):
                            x_sparse.to_sparse(sparse_dim_out)

    # 定义仅在原生设备类型上运行的测试函数，用于测试具有稀疏支持的函数
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(like_fns_with_sparse_support)
    @all_sparse_layouts('layout', include_strided=False)
    # 定义测试函数，用于测试稀疏张量操作
    def test_like_fns(self, layout, device, dtype, op):

        # 遍历稀疏张量操作的样本输入
        for sample in op.sample_inputs_sparse(layout, device, dtype):
            # 从样本中获取输入张量及其参数
            t_inp, t_args, t_kwargs = sample.input, sample.args, sample.kwargs
            # 计算批次维度
            batch_dim = t_inp.dim() - t_inp.dense_dim() - t_inp.sparse_dim()

            # 如果输入张量的布局是稀疏 BSR 或 BSC
            if t_inp.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                # 预期的块大小是输入值的形状的特定切片
                expected_blocksize = t_inp.values().shape[batch_dim + 1:batch_dim + 3]
            else:
                expected_blocksize = None

            # 预期的数据类型、设备和布局
            expected_dtype = t_kwargs.get('dtype', dtype)
            expected_device = torch.device(t_kwargs.get('device', device))
            expected_layout = t_kwargs.get('layout', layout)

            # 执行稀疏张量操作
            result = op.op(t_inp, *t_args, **t_kwargs)

            # 断言结果的数据类型、设备和布局与预期相符
            self.assertEqual(result.dtype, expected_dtype)
            self.assertEqual(result.device.type, expected_device.type)
            self.assertEqual(result.layout, expected_layout)

            # 如果结果的布局是稀疏 BSR 或 BSC
            if result.layout in {torch.sparse_bsr, torch.sparse_bsc}:
                # 计算结果的批次维度
                result_batch_dim = result.dim() - result.dense_dim() - result.sparse_dim()
                # 获取结果值的形状的特定切片，作为块大小
                blocksize = result.values().shape[result_batch_dim + 1:result_batch_dim + 3]
                # 断言结果的块大小与预期相符
                self.assertEqual(blocksize, expected_blocksize)

            # 检查操作后的形状与输入形状是否相同
            self.assertEqual(result.shape, t_inp.shape)

            # 如果预期布局是 strided
            if expected_layout is torch.strided:
                # 断言结果的稀疏维度为 0
                self.assertEqual(result.sparse_dim(), 0)
                # 断言操作后的密集维度与输入的维度相同
                self.assertEqual(result.dense_dim(), t_inp.dim())
            # 如果预期布局是稀疏 COO
            elif expected_layout is torch.sparse_coo:
                # 断言结果的稀疏维度与输入的批次维度加上稀疏维度相同
                self.assertEqual(result.sparse_dim(), batch_dim + t_inp.sparse_dim())
                # 断言操作后的密集维度与输入的密集维度相同
                self.assertEqual(result.dense_dim(), t_inp.dense_dim())

                # 验证稀疏 COO 张量的参数
                torch._validate_sparse_coo_tensor_args(result._indices(), result._values(), result.shape)
            else:
                # 断言操作后的稀疏维度与输入的稀疏维度相同
                self.assertEqual(result.sparse_dim(), t_inp.sparse_dim())
                # 断言操作后的密集维度与输入的密集维度相同
                self.assertEqual(result.dense_dim(), t_inp.dense_dim())

                # 如果结果的布局是稀疏 CSR 或 BSR
                if result.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    compressed_indices, plain_indices = result.crow_indices(), result.col_indices()
                else:
                    compressed_indices, plain_indices = result.ccol_indices(), result.row_indices()

                # 验证压缩稀疏张量的参数
                torch._validate_sparse_compressed_tensor_args(compressed_indices, plain_indices, result.values(),
                                                              result.shape, result.layout)

    # 使用所有的稀疏布局进行测试，排除 strided 布局
    @all_sparse_layouts('mask_layout', include_strided=False)
    # 应用装饰器 `onlyNativeDeviceTypes` 和 `dtypes` 到以下测试方法，用于限制适用的设备类型和数据类型
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
    def test_sparse_mask(self, mask_layout, device, dtype):
        # 设置输入张量的布局为 strided
        input_layout = torch.strided
        # 设置掩码的数据类型为布尔型
        mask_dtype = torch.bool
        # 针对生成的简单输入掩码调用生成器函数，并且禁用混合和批处理
        for mask in self.generate_simple_inputs(mask_layout, dtype=mask_dtype, device=device,
                                                enable_hybrid=False, enable_batch=False):
            # 创建稀疏张量 `x`，并设置布局为指定的 `input_layout`
            x = make_tensor(mask.shape, dtype=dtype, device=device).to_sparse(layout=input_layout)
    
            # 使用稀疏掩码操作 `sparse_mask` 对稀疏张量 `x` 进行掩码
            result = x.sparse_mask(mask)
    
            # 检查不变式 `x.sparse_mask(mask).<indices> == mask.<indices>`
            if mask_layout is torch.sparse_coo:
                self.assertEqual(result._indices(), mask._indices())
                ones = torch.sparse_coo_tensor(mask._indices(),
                                               torch.ones_like(mask._values(), dtype=x.dtype),
                                               mask.shape,
                                               is_coalesced=mask.is_coalesced())
            elif mask_layout in {torch.sparse_csr, torch.sparse_bsr}:
                self.assertEqual(result.crow_indices(), mask.crow_indices())
                self.assertEqual(result.col_indices(), mask.col_indices())
                ones = torch.sparse_compressed_tensor(mask.crow_indices(), mask.col_indices(),
                                                      torch.ones_like(mask.values(), dtype=x.dtype),
                                                      mask.shape, layout=mask.layout)
            else:
                self.assertEqual(result.ccol_indices(), mask.ccol_indices())
                self.assertEqual(result.row_indices(), mask.row_indices())
                ones = torch.sparse_compressed_tensor(mask.ccol_indices(), mask.row_indices(),
                                                      torch.ones_like(mask.values(), dtype=x.dtype),
                                                      mask.shape, layout=mask.layout)
    
            # 检查不变式:
            # x.sparse_mask(mask).to_dense() == x.mul(sparse_xyz_tensor(<mask indices>,
            #                                         ones_like(<mask values>)).to_dense())
            expected = x.mul(ones.to_dense())
            self.assertEqual(result.to_dense(), expected)
    
            # 检查不变式 `mask.to_dense().sparse_mask(mask) == mask`
            result = mask.to_dense().sparse_mask(mask)
            self.assertEqual(result, mask)
    
    # 应用装饰器 `all_sparse_layouts` 和 `parametrize` 到以下测试方法，用于参数化稀疏布局和额外的参数组合
    @all_sparse_layouts('layout', include_strided=False)
    @parametrize("masked", [subtest(False, name='nonmasked'), subtest(True, name='masked')])
    @parametrize("fast_mode", [subtest(False, name='slow'), subtest(True, name='fast')])
    # 定义一个测试方法，用于检查稀疏梯度计算的正确性
    def test_as_sparse_gradcheck(self, layout, device, masked, fast_mode):
        # 获取稀疏梯度检查的函数
        gradcheck = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)
        # 定义稀疏压缩布局的集合
        sparse_compressed_layouts = {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}

        # 定义一个简单的恒等函数
        def identity(x):
            return x

        # 遍历需要进行梯度检查的函数
        for func in (torch.Tensor.to_dense,
                     torch.Tensor.sum,
                     identity,
                     torch.Tensor.to_sparse,
                     torch.Tensor.values,
                     ):
            # 生成简单的输入数据集，考虑布局、设备和数据类型
            for x in self.generate_simple_inputs(
                    layout,
                    device=device,
                    dtype=torch.float64,
                    # TODO: fix gh-104868  to enable batched samples:
                    enable_batch=layout not in sparse_compressed_layouts,
                    enable_hybrid=not (
                        layout in sparse_compressed_layouts and (
                            # FIXME: RuntimeError: sparse_mask(): the
                            # number of sparse dimensions in `self`
                            # should match that of the `mask`. Got
                            # `self.sparse_dim() == 3` !=
                            # `mask.sparse_dim() == 2
                            func.__name__ == 'sum'
                            # FIXME: RuntimeError: expected
                            # col_indices to be a contiguous tensor
                            # per batch
                            or func.__name__ == 'to_sparse'
                        ))):
                # 如果布局为稀疏 COO，并且函数名为 'values'，则调用 coalesce 方法
                if layout is torch.sparse_coo and func.__name__ == 'values':
                    x = x.coalesce()

                # 对每个函数和输入数据进行梯度检查
                gradcheck(func, x.requires_grad_(True), masked=masked, fast_mode=fast_mode)

    # 用于测试数据加载器的方法，仅适用于 CPU
    @onlyCPU
    # 应用于所有稀疏布局的装饰器，排除分块布局
    @all_sparse_layouts('layout', include_strided=False)
    # 数据类型限定为双精度浮点数
    @dtypes(torch.double)
    def test_dataloader(self, device, layout, dtype):

        # 生成简单输入数据列表
        data = list(self.generate_simple_inputs(layout, device=device, dtype=dtype))

        # 创建 _SparseDataset 对象
        dataset = _SparseDataset(data)
        # 创建数据加载器，不设置批量大小，设置两个工作进程
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)

        # 加载数据到列表
        loaded_data = list(loader)
        # 断言加载的数据与原始数据一致
        self.assertEqual(data, loaded_data)

    # 仅适用于 CPU 的方法装饰器
    @onlyCPU
    # 定义一个测试方法，用于测试不合法的块大小设置
    def test_invalid_blocksize(self):
        # 当块大小应为包含两个值的元组、列表或torch.Size时，检查是否引发运行时错误，错误信息应包含“blocksize”，但实际得到了1
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 1"):
            # 调用torch.randn(1).to_sparse方法，传入不合法的块大小作为参数
            torch.randn(1).to_sparse(blocksize=(1,))
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 1"):
            # 同上，传入列表形式的不合法块大小参数
            torch.randn(1).to_sparse(blocksize=[1])
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 1"):
            # 同上，传入torch.Size对象形式的不合法块大小参数
            torch.randn(1).to_sparse(blocksize=torch.Size((1,)))
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 3"):
            # 检查传入包含三个值的元组形式的不合法块大小参数
            torch.randn(1).to_sparse(blocksize=(1, 1, 1))
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 3"):
            # 同上，传入列表形式的不合法块大小参数
            torch.randn(1).to_sparse(blocksize=[1, 1, 1])
        with self.assertRaisesRegex(RuntimeError, ".*blocksize.*, but got 3"):
            # 同上，传入torch.Size对象形式的不合法块大小参数
            torch.randn(1).to_sparse(blocksize=torch.Size((1, 1, 1)))
# 根据给定的模板类和全局变量实例化设备类型的测试用例，例如 TestSparseUnaryUfuncsCPU 和 TestSparseUnaryUfuncsCUDA
instantiate_device_type_tests(TestSparseUnaryUfuncs, globals(), except_for='meta')

# 根据给定的模板类和全局变量实例化设备类型的测试用例，例如 TestSparseMaskedReductions
instantiate_device_type_tests(TestSparseMaskedReductions, globals(), except_for='meta')

# 根据给定的模板类和全局变量实例化设备类型的测试用例，例如 TestSparseCPU 和 TestSparseCUDA
instantiate_device_type_tests(TestSparse, globals(), except_for='meta')

# 根据给定的模板类和全局变量实例化设备类型的测试用例，例如 TestSparseAny
instantiate_device_type_tests(TestSparseAny, globals(), except_for='meta')

# 实例化参数化的测试用例，例如 TestSparseMeta
instantiate_parametrized_tests(TestSparseMeta)

# 实例化参数化的测试用例，例如 TestSparseLegacyAndDeprecation
instantiate_parametrized_tests(TestSparseLegacyAndDeprecation)

# 如果当前脚本作为主程序运行，则执行所有测试
if __name__ == '__main__':
    run_tests()
```