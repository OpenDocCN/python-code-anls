# `.\pytorch\test\test_ops.py`

```py
# Owner(s): ["module: unknown"]

import contextlib  # 导入上下文管理模块，用于管理上下文中的资源
import copy  # 导入复制模块，用于创建对象的深拷贝
import inspect  # 导入检查模块，用于获取对象信息
import itertools  # 导入迭代工具模块，用于生成迭代器
import os  # 导入操作系统模块，提供与操作系统相关的功能
import re  # 导入正则表达式模块，用于处理正则表达式操作
import unittest  # 导入单元测试模块，用于编写和运行测试用例
import warnings  # 导入警告模块，用于管理警告信息的显示

from collections import defaultdict  # 导入默认字典集合，提供默认值的字典
from collections.abc import Sequence  # 导入抽象基类模块，提供序列抽象基类
from functools import partial  # 导入偏函数模块，用于部分应用函数
from importlib import import_module  # 导入模块导入模块，用于动态导入模块
from typing import Dict, List  # 导入类型提示模块，指定函数参数和返回值的类型

import torch  # 导入PyTorch模块，提供深度学习库的功能

import torch._prims as prims  # 导入PyTorch私有模块，提供底层操作的原始函数
import torch.utils._pytree as pytree  # 导入PyTorch工具私有模块，提供树操作的工具函数
from torch._prims.context import TorchRefsMode  # 从PyTorch私有模块中导入上下文模式
from torch._prims_common.wrappers import _maybe_remove_out_wrapper  # 导入PyTorch私有模块中的包装器函数
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # 导入PyTorch子类私有模块中的伪张量类和模式
from torch._subclasses.fake_utils import outputs_alias_inputs  # 导入PyTorch子类私有模块中的函数
from torch.testing import make_tensor  # 从PyTorch测试模块中导入创建张量的函数

from torch.testing._internal import composite_compliance, opinfo  # 从PyTorch内部测试模块中导入组合兼容性和操作信息
from torch.testing._internal.common_device_type import (  # 从PyTorch内部测试模块中导入通用设备类型函数
    deviceCountAtLeast,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    OpDTypes,
    ops,
    skipMeta,
)
from torch.testing._internal.common_dtype import (  # 从PyTorch内部测试模块中导入通用数据类型函数
    all_types_and_complex_and,
    floating_and_complex_types_and,
    integral_types_and,
)
from torch.testing._internal.common_methods_invocations import (  # 从PyTorch内部测试模块中导入通用方法调用函数
    BinaryUfuncInfo,
    op_db,
    ops_and_refs,
    python_ref_db,
    ReductionOpInfo,
    ReductionPythonRefInfo,
    skip,
    skipOps,
    SpectralFuncInfo,
    UnaryUfuncInfo,
    xfail,
)

from torch.testing._internal.common_utils import (  # 从PyTorch内部测试模块中导入通用工具函数
    clone_input_helper,
    first_sample,
    IS_CI,
    IS_FBCODE,
    is_iterable_of_tensors,
    IS_SANDCASTLE,
    IS_WINDOWS,
    noncontiguous_like,
    parametrize,
    run_tests,
    set_default_dtype,
    skipIfTorchInductor,
    slowTest,
    suppress_warnings,
    TEST_WITH_ASAN,
    TEST_WITH_ROCM,
    TEST_WITH_TORCHDYNAMO,
    TEST_WITH_TORCHINDUCTOR,
    TEST_WITH_UBSAN,
    TestCase,
    unMarkDynamoStrictTest,
)
from torch.utils._python_dispatch import TorchDispatchMode  # 从PyTorch工具私有模块中导入分发模式
from torch.utils._pytree import tree_map  # 从PyTorch工具私有模块中导入树映射函数

assert torch.get_default_dtype() == torch.float32  # 断言默认张量类型为32位浮点型

# variant testing is only done with torch.float and torch.cfloat to avoid
#   excessive test times and maximize signal to noise ratio
_variant_ops = partial(
    ops, dtypes=OpDTypes.supported, allowed_dtypes=(torch.float, torch.cfloat)
)  # 部分应用ops函数，限定支持的操作数据类型为浮点型和复数浮点型

# Get names of all the operators which have ref in their entry in OpInfo (testing infra)
#   except for elementwise unary operators (separately implemented in test/test_unary_ufuncs.py),
#   elementwise binary operators (separately implemented in test_binary_ufuncs.py),
#   reduction operations (separately impelemented in test_reductions.py),
#   and Spectral Functions (separately implemented for only 1D as of now, in test/test_spectral_ops.py)
_ref_test_ops = tuple(
    filter(
        lambda op: not isinstance(
            op, (UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo, BinaryUfuncInfo)
        )
        and op.ref is not None,
        op_db,
    )
)


def reduction_dtype_filter(op):
    # 检查条件：如果 op 不是 ReductionPythonRefInfo 类型，或者不支持输出（supports_out），或者不支持 torch.int16 数据类型，则返回 False
    if (
        not isinstance(op, ReductionPythonRefInfo)
        or not op.supports_out
        or torch.int16 not in op.dtypes
    ):
        # 如果以上条件任一不满足，返回 False
        return False
    
    # 使用 inspect 模块获取 op.op 函数的完整参数规范
    argspec = inspect.getfullargspec(op.op)
    
    # 检查关键字参数列表中是否包含 'dtype'
    if "dtype" not in argspec.kwonlyargs:
        # 如果 'dtype' 不在关键字参数列表中，返回 False
        return False
    
    # 如果所有条件满足，返回 True
    return True
# 创建一个操作符列表，这些操作符是 _ref_test_ops 的子集，但没有与之对应的 numpy 引用。
# 如果 CPU 和 CUDA 都与 numpy 进行比较，则它们不需要相互比较。
_ops_and_refs_with_no_numpy_ref = [op for op in ops_and_refs if op.ref is None]

# 导入 torch 的 aten 操作符命名空间
aten = torch.ops.aten

# 适用于所有操作符的测试，与任何特定系统无关
@unMarkDynamoStrictTest
class TestCommon(TestCase):
    exact_dtype = True

    # 在测试类销毁时，验证没有 OpInfo 在 CI 中仍然使用动态数据类型
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_CI:
            # 如果在 CI 环境中，则输出警告信息
            err_msg = (
                "The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                "This is OK for testing, but be sure to set the dtypes manually before landing your PR!"
            )
            # 筛选出具有动态数据类型设置的 OpInfo 条目
            filtered_ops = list(filter(opinfo.utils.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                # 格式化 OpInfo 条目的动态数据类型信息为字符串
                fmt_str = opinfo.utils.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            # 断言没有 OpInfo 条目具有动态数据类型设置
            assert len(filtered_ops) == 0, err_msg

    # 验证每个 OpInfo 在不同 CUDA 设备上的正确运行
    @onlyCUDA
    @deviceCountAtLeast(2)
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long))
    def test_multiple_devices(self, devices, dtype, op):
        for cuda_device_str in devices:
            cuda_device = torch.device(cuda_device_str)
            # 注意：仅对第一个样本进行测试
            samples = op.sample_inputs(cuda_device, dtype)
            sample = first_sample(self, samples)
            result = op(sample.input, *sample.args, **sample.kwargs)

            if isinstance(result, torch.Tensor):
                self.assertTrue(result.device == cuda_device)
            elif is_iterable_of_tensors(result):
                self.assertTrue(all(t.device == cuda_device for t in result))
            else:
                self.skipTest(
                    "Skipped! Only supports single tensor or iterable of tensor outputs."
                )

    # 测试函数及其（接受 ndarray 的）参考是否在相应 op 的 sample_inputs 函数的张量上产生相同的值。
    # 该测试以双精度和复数双精度运行，因为 NumPy 对许多函数内部使用双精度进行计算，
    # 可能导致相等性检查失败。
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(_ref_test_ops, allowed_dtypes=(torch.float64, torch.long, torch.complex128))
    # 定义一个测试方法，用于测试与 NumPy 引用相关的操作
    def test_numpy_ref(self, device, dtype, op):
        # 如果测试使用 Torch Inductor，并且操作的格式化名称在指定列表中，
        # 并且数据类型为 torch.float64，并且设备包含 "cuda"
        if (
            TEST_WITH_TORCHINDUCTOR
            and op.formatted_name
            in ("signal_windows_exponential", "signal_windows_bartlett")
            and dtype == torch.float64
            and "cuda" in device
        ):  # noqa: E121
            # 抛出跳过测试的异常，说明不支持张量类数据近似比较
            raise unittest.SkipTest("XXX: raises tensor-likes are not close.")

        # 设置默认数据类型为 NumPy 的双精度类型
        with set_default_dtype(torch.double):
            # 对每个操作的参考输入进行比较
            for sample_input in op.reference_inputs(device, dtype):
                # 调用比较方法，比较操作、参考输出和样本输入，
                # 确保数据类型与 torch.long 不完全相等
                self.compare_with_reference(
                    op, op.ref, sample_input, exact_dtype=(dtype is not torch.long)
                )

    # 测试 CPU 和 GPU 的结果是否一致
    @onlyCUDA
    @suppress_warnings
    @slowTest
    @ops(_ops_and_refs_with_no_numpy_ref, dtypes=OpDTypes.any_common_cpu_cuda_one)
    def test_compare_cpu(self, device, dtype, op):
        # 将输入参数转换到 CPU 上的方法
        def to_cpu(arg):
            if isinstance(arg, torch.Tensor):
                return arg.to(device="cpu")
            return arg

        # 获取操作的参考输入样本
        samples = op.reference_inputs(device, dtype)

        # 遍历每个样本
        for sample in samples:
            # 将样本数据转换为 CPU 上的数据
            cpu_sample = sample.transform(to_cpu)
            # 调用操作，获取 CUDA 上的结果
            cuda_results = op(sample.input, *sample.args, **sample.kwargs)
            # 在 CPU 上调用操作，获取 CPU 上的结果
            cpu_results = op(cpu_sample.input, *cpu_sample.args, **cpu_sample.kwargs)

            # 使用特定的函数处理输出梯度
            cuda_results = sample.output_process_fn_grad(cuda_results)
            cpu_results = cpu_sample.output_process_fn_grad(cpu_results)

            # 由于运行速度较慢，降低容差以避免频繁测试失败
            # 使用绝对误差和相对误差进行结果比较
            self.assertEqual(cuda_results, cpu_results, atol=1e-3, rtol=1e-3)

    # 测试 Python 引用是否正确传播形状、数据类型和设备元数据
    # 参考：https://github.com/pytorch/pytorch/issues/78050 讨论步幅传播问题
    @onlyNativeDeviceTypes
    @ops(python_ref_db)
    @skipIfTorchInductor("Takes too long for inductor")
    # 定义测试方法，用于检查 Python 引用的元信息
    def test_python_ref_meta(self, device, dtype, op):
        # 定义跳过的操作集合，包含 torch._refs.linalg.svd
        CHECK_CONJ_SKIPS = {
            torch._refs.linalg.svd,
        }

        # 进入 FakeTensorMode 上下文
        with FakeTensorMode() as mode:
            pass

        # 定义函数，将输入转换为 FakeTensorMeta 对象
        def _to_tensormeta(x):
            # 如果输入是 torch.Tensor，则使用 FakeTensor.from_tensor 方法进行转换
            if isinstance(x, torch.Tensor):
                out = FakeTensor.from_tensor(x, mode)
                return out
            # 否则直接返回输入
            return x

        # 遍历操作 op 的参考输入，设定 requires_grad 为 False
        # op.reference_inputs 返回一个生成器，每次生成一个包含输入和参数的样本
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            # 调用 op 进行计算
            result = op(sample.input, *sample.args, **sample.kwargs)

            # 对样本应用 _to_tensormeta 函数转换
            meta_sample = sample.transform(_to_tensormeta)
            try:
                # 在 mode 上下文中调用 op 进行计算
                with mode:
                    meta_result = op(
                        meta_sample.input, *meta_sample.args, **meta_sample.kwargs
                    )
            # 捕获可能出现的异常并继续执行
            except torch._subclasses.fake_tensor.UnsupportedFakeTensorException:
                continue
            except torch._subclasses.fake_tensor.DataDependentOutputException:
                continue
            except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                continue

            # 检查 result 是否为 torch.Tensor 类型
            if isinstance(result, torch.Tensor):
                # 断言 meta_result 是 FakeTensor 类型
                self.assertTrue(isinstance(meta_result, FakeTensor))
                # 比较 result 和 meta_result 的元信息
                prims.utils.compare_tensor_meta(
                    result, meta_result, check_conj=op.op not in CHECK_CONJ_SKIPS
                )
            # 如果 result 是序列类型
            elif isinstance(result, Sequence):
                # 遍历 result 和 meta_result 的每对元素
                for a, b in zip(result, meta_result):
                    # 如果 a 或 b 是 torch.Tensor 类型
                    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                        # 断言 b 是 FakeTensor 类型
                        self.assertTrue(isinstance(b, FakeTensor))
                        # 比较 a 和 b 的元信息
                        prims.utils.compare_tensor_meta(
                            a, b, check_conj=op.op not in CHECK_CONJ_SKIPS
                        )

    # 辅助方法，执行引用测试
    def _ref_test_helper(
        self,
        ctx,
        device,
        dtype,
        op,
        skip_zero_numel=False,
        skip_zero_dim=False,
        skip_bfloat=False,
        skip_view_consistency=False,
    ):
        # 测试实验性 Python 引用是否与操作本身执行相同计算
        # 当 torch 命名空间中的操作调用被重定向到 refs 命名空间时（torch.foo 变成 refs.foo）
        @onlyNativeDeviceTypes
        @ops(python_ref_db)
        @skipIfTorchInductor("Takes too long for inductor")
        def test_python_ref(self, device, dtype, op):
            # 在此测试中，primTorch 引用调用进入 refs 命名空间
            # 例如，带有 torch.foo 的引用将调用 refs.foo
            # 直接调用 refs 和 prims 不受影响
            if (
                TEST_WITH_ROCM
                and (op.name == "_refs.fft.ihfftn" or op.name == "_refs.fft.ihfft2")
                and dtype == torch.float16
            ):
                self.skipTest("Skipped on ROCm")
            # 调用 _ref_test_helper 方法进行引用测试辅助
            self._ref_test_helper(lambda: TorchRefsMode(strict=True), device, dtype, op)

    # 测试实验性 Python 引用是否执行相同计算
    # 将修饰符应用于测试方法，限制只在本地设备类型上执行
    @onlyNativeDeviceTypes
    # 将修饰符应用于测试方法，使用python_ref_db中的操作作为操作的参数
    @ops(python_ref_db)
    # 如果在Torch Inductor上运行测试会花费太长时间，则跳过此测试
    @skipIfTorchInductor("Takes too long for inductor")
    # 定义一个测试方法，用于测试在Torch命名空间中使用Python引用回退的情况
    def test_python_ref_torch_fallback(self, device, dtype, op):
        # 如果在ROCm环境下，并且操作名称为"_refs.fft.ihfftn"且数据类型为torch.float16，则跳过测试
        if TEST_WITH_ROCM and op.name == "_refs.fft.ihfftn" and dtype == torch.float16:
            self.skipTest("Skipped on ROCm")
        # 调用_ref_test_helper方法，用于辅助测试
        self._ref_test_helper(contextlib.nullcontext, device, dtype, op)

    # 如果在ASAN环境下，则跳过此测试
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    # 只在CUDA设备上执行测试
    @onlyCUDA
    # 将修饰符应用于测试方法，使用python_ref_db中的操作作为操作的参数
    @ops(python_ref_db)
    # 参数化测试方法，executor参数为"aten"
    @parametrize(
        "executor",
        [
            "aten",
        ],
    )
    # 如果在Torch Inductor上运行测试会花费太长时间，则跳过此测试
    @skipIfTorchInductor("Takes too long for inductor")
    # 定义一个测试方法，用于测试在执行器环境中的Python引用
    def test_python_ref_executor(self, device, dtype, op, executor):
        # 如果在ROCm环境下，并且操作名称为"_refs.fft.ihfftn"或"_refs.fft.ihfft2"且数据类型为torch.float16，则跳过测试
        if (
            TEST_WITH_ROCM
            and (op.name == "_refs.fft.ihfftn" or op.name == "_refs.fft.ihfft2")
            and dtype == torch.float16
        ):
            self.skipTest("Skipped on ROCm")
        # 创建op的副本，并将其操作部分替换为经过跟踪的操作，executor参数为executor
        op = copy(op)
        op.op = partial(make_traced(op.op), executor=executor)
        # 调用_ref_test_helper方法，用于辅助测试
        self._ref_test_helper(
            contextlib.nullcontext,
            device,
            dtype,
            op,
        )

    # 跳过元测试
    @skipMeta
    # 只在本地设备类型上执行测试
    @onlyNativeDeviceTypes
    # 将修饰符应用于测试方法，操作来自op_db并且具有非None的error_inputs_func
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    # 定义一个测试方法，用于测试操作中的错误情况
    def test_errors(self, device, op):
        # 获取操作op的错误输入
        error_inputs = op.error_inputs(device)
        # 遍历错误输入
        for ei in error_inputs:
            # 获取样本输入si
            si = ei.sample_input
            # 使用断言检查是否引发了预期的错误类型和错误正则表达式
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                out = op(si.input, *si.args, **si.kwargs)
                # 确保输出不是NotImplemented类型的实例
                self.assertFalse(isinstance(out, type(NotImplemented)))

    # 跳过元测试
    @skipMeta
    # 只在本地设备类型上执行测试
    @onlyNativeDeviceTypes
    # 将修饰符应用于测试方法，操作来自op_db并且具有非None的error_inputs_sparse_func
    @ops(
        [op for op in op_db if op.error_inputs_sparse_func is not None],
        dtypes=OpDTypes.none,
    )
    # 参数化测试方法，layout参数为稀疏张量的不同布局类型
    @parametrize(
        "layout",
        (
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
            torch.sparse_coo,
        ),
    )
    # 定义一个测试函数，用于测试稀疏输入情况下的错误处理
    def test_errors_sparse(self, device, op, layout):
        # 遍历操作(op)生成的稀疏错误输入
        for ei in op.error_inputs_sparse(device, layout):
            # 获取样本输入
            si = ei.sample_input
            # 断言操作(op)执行时会抛出特定类型和正则表达式匹配的错误
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                # 执行操作(op)，传入样本输入的参数
                out = op(si.input, *si.args, **si.kwargs)
                # 断言输出(out)不是NotImplemented类型的实例
                self.assertFalse(isinstance(out, type(NotImplemented)))

    # 使用装饰器配置，跳过一些特定的测试
    @skipMeta
    @onlyNativeDeviceTypes
    @ops(
        # 遍历Python参考数据库中具有错误输入函数的操作(op)
        [op for op in python_ref_db if op.error_inputs_func is not None],
        # 设置操作的数据类型为OpDTypes.none
        dtypes=OpDTypes.none,
    )
    # 在Torch Inductor中跳过测试，因为运行时间太长
    @skipIfTorchInductor("Takes too long for inductor")
    # 测试Python参考实现中的错误处理
    def test_python_ref_errors(self, device, op):
        # 创建一个FakeTensorMode上下文
        mode = FakeTensorMode()
        with mode:
            pass

        # 定义一个内部函数，用于将输入转换为FakeTensorMeta
        def _to_tensormeta(x):
            if isinstance(x, torch.Tensor):
                return FakeTensor.from_tensor(x, mode)
            return x

        # 获取操作(op)的错误输入列表
        error_inputs = op.error_inputs(device)
        # 遍历错误输入列表
        for ei in error_inputs:
            # 获取样本输入
            si = ei.sample_input
            # 将样本输入转换为FakeTensorMeta
            meta_sample = si.transform(_to_tensormeta)
            # 断言操作(op)执行时会抛出特定类型和正则表达式匹配的错误
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                # 执行操作(op)，传入FakeTensorMeta的参数
                op(meta_sample.input, *meta_sample.args, **meta_sample.kwargs)

    # 测试函数的装饰器，用于跳过在Windows下的运行
    # 只在本机设备类型下运行测试
    # 抑制警告
    # 仅对指定的操作数据库和允许的数据类型进行操作
    # 以下操作分离一个用例，因为许多操作尚未正确实现out参数的警告
    #
    # 对以下测试的分析：
    #   - 使用正确的dtype和device，但形状错误的out=
    #   - 验证操作实现了正确的out=行为
    #   - 验证以下情况：
    #     - Case 0: out具有正确的形状、dtype和device，但是包含极端值
    #     - Case 1: out具有正确的形状、dtype和device，但是非连续
    #     - Case 2: out具有正确的dtype和device，但元素数量为零
    #     - Case 3: out具有正确的形状和dtype，但在不同的设备类型上
    #     - Case 4: out具有正确的形状和device，但是不能安全地转换为的dtype
    #
    # 当op是工厂函数时，Case 3和Case 4稍有不同：
    #   - 如果未传递device、dtype，则任何dtype/device的组合对out都应该是OK的
    #   - 如果传递了device、dtype，则device和dtype应该匹配
    @unittest.skipIf(IS_WINDOWS, "Skipped under Windows")
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long, torch.complex64))
    @ops(ops_and_refs, dtypes=OpDTypes.none)
    @ops(ops_and_refs, dtypes=OpDTypes.any_one)
    @ops(
        [
            op
            for op in op_db
            # 从操作数据库中筛选出支持输出且支持自动求导或是工厂函数的操作
            if op.supports_out and (op.supports_autograd or op.is_factory_function)
        ],
        dtypes=OpDTypes.supported,
        allowed_dtypes=[torch.float, torch.cfloat],
    )
    # 使用ops装饰器声明测试方法，设置操作类型、数据类型和允许的数据类型
    def test_out_requires_grad_error(self, device, dtype, op):
        # 获取样本输入数据
        sample = first_sample(self, op.sample_inputs(device, dtype))

        # 调用操作以获取输出参数的原型
        expect = op(sample.input, *sample.args, **sample.kwargs)
        any_requires_grad = False

        # 定义函数设置张量的requires_grad属性
        def set_requires_grad(x):
            nonlocal any_requires_grad
            # 如果是浮点型或复数型张量，则设置requires_grad为True
            if isinstance(x, torch.Tensor) and (
                x.is_floating_point() or x.is_complex()
            ):
                any_requires_grad = True
                x.requires_grad_(True)
            return x

        # 对操作的输出参数应用set_requires_grad函数
        out = pytree.tree_map_(set_requires_grad, expect)
        if not any_requires_grad:
            # 跳过没有浮点数输出的操作，例如isnan
            return

        # 出错时的异常消息
        msg = (
            "functions with out=... arguments don't support automatic "
            "differentiation, but one of the arguments requires grad."
        )
        # 断言操作会引发RuntimeError异常，并输出指定的消息
        with self.assertRaises(RuntimeError, msg=msg):
            op(sample.input, *sample.args, **sample.kwargs, out=out)

    @ops(filter(reduction_dtype_filter, ops_and_refs), dtypes=(torch.int16,))
    # 测试函数，用于测试在给定设备、数据类型和操作的情况下是否符合整数数据类型的要求
    def test_out_integral_dtype(self, device, dtype, op):
        # 嵌套函数，辅助执行测试，并检查是否符合预期失败的条件
        def helper(with_out, expectFail, op_to_test, inputs, *args, **kwargs):
            out = None
            try:
                # 如果需要输出张量
                if with_out:
                    # 创建一个空的张量用于输出，数据类型为 torch.int32，存储于指定设备上
                    out = torch.empty(0, dtype=torch.int32, device=device)
                    # 执行测试操作，将结果存储到预先创建的 out 张量中
                    op_to_test(inputs, *args, out=out, **kwargs)
                else:
                    # 直接执行测试操作，不需要输出张量
                    out = op_to_test(inputs, *args, **kwargs)
                # 断言操作不会产生异常，即不会预期失败
                self.assertFalse(expectFail)
            except RuntimeError as err:
                # 捕获 RuntimeError 异常，检查错误信息是否与指定信息相匹配
                self.assertEqual(
                    str(err), "dtype argument and out dtype must match in reduction"
                )
                # 断言预期失败条件为真
                self.assertTrue(expectFail)
            return out

        # 获取操作的样本输入
        samples = op.sample_inputs(device, dtype)
        # 遍历样本输入进行测试
        for sample in samples:
            # 如果样本中未指定数据类型
            if "dtype" not in sample.kwargs:
                # 调用 helper 函数进行测试，测试不使用输出张量，预期不失败
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                # 调用 helper 函数进行测试，测试使用输出张量，预期不失败
                helper(True, False, op, sample.input, *sample.args, **sample.kwargs)
                # 设置样本数据类型为 torch.int16
                sample.kwargs["dtype"] = torch.int16
                # 调用 helper 函数进行测试，测试不使用输出张量，预期不失败
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                # 调用 helper 函数进行测试，测试使用输出张量，预期失败
                helper(True, True, op, sample.input, *sample.args, **sample.kwargs)
                # 设置样本数据类型为 torch.int32
                sample.kwargs["dtype"] = torch.int32
                # 调用 helper 函数进行测试，测试不使用输出张量，预期不失败
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                # 调用 helper 函数进行测试，测试使用输出张量，预期不失败
                helper(True, False, op, sample.input, *sample.args, **sample.kwargs)
            else:
                # 如果样本中已指定数据类型
                # 调用 helper 函数进行测试，测试不使用输出张量，预期不失败
                helper(False, False, op, sample.input, *sample.args, **sample.kwargs)
                # 调用 helper 函数进行测试，测试使用输出张量，预期失败条件为数据类型不为 torch.int32
                helper(
                    True,
                    sample.kwargs["dtype"] != torch.int32,
                    op,
                    sample.input,
                    *sample.args,
                    **sample.kwargs,
                )

    # 用于测试操作的前向和反向传播是否对方法和就地变体产生相同的值
    #   这是针对操作在 eager 模式下的标准函数变体进行交叉测试
    @_variant_ops(op_db)
    # 在 complex32 上进行操作的参考测试，与 complex64 进行比较
    # 注意：我们对 complex64 进行测试，因为 NumPy 没有 complex32 等效的数据类型
    @ops(op_db, allowed_dtypes=(torch.complex32,))
    # 测试复杂半参考测试
    def test_complex_half_reference_testing(self, device, dtype, op):
        # 如果操作不支持 torch.complex32 类型，则跳过测试
        if not op.supports_dtype(torch.complex32, device):
            unittest.skip("Does not support complex32")

        # 对于 op.sample_inputs 返回的每个样本
        for sample in op.sample_inputs(device, dtype):
            # 调用操作 op，获取实际输出
            actual = op(sample.input, *sample.args, **sample.kwargs)
            
            # sample.transform 应用 lambda 函数到 torch.Tensor 和 torch.dtype
            # 我们只想将其应用于 dtype 是 torch.complex32 的张量
            transformed_sample = sample.transform(
                lambda x: x.to(torch.complex64)
                if isinstance(x, torch.Tensor) and x.dtype is torch.complex32
                else x
            )
            
            # 使用转换后的样本调用操作 op，获取预期输出
            expected = op(
                transformed_sample.input,
                *transformed_sample.args,
                **transformed_sample.kwargs,
            )
            
            # 由于 chalf 的范围比 cfloat 小得多，
            # 在操作如 pow、exp 等中容易得到无穷大，因此将 cfloat 转换回 chalf
            expected = tree_map(
                lambda x: x.to(torch.complex32)
                if isinstance(x, torch.Tensor) and x.dtype is torch.complex64
                else x,
                expected,
            )

            # 对比实际输出和预期输出是否相等，忽略精确的数据类型匹配
            self.assertEqual(actual, expected, exact_dtype=False)

    # 使用 op_db 中的操作，允许的数据类型为 torch.bool
    # 跳过 UBSAN 测试
    @ops(op_db, allowed_dtypes=(torch.bool,))
    @unittest.skipIf(TEST_WITH_UBSAN, "Test uses undefined behavior")
    def test_non_standard_bool_values(self, device, dtype, op):
        # 测试除 0x00 和 0x01 之外的布尔值情况 (gh-54789)
        def convert_boolean_tensors(x):
            # 如果 x 不是 torch.Tensor 或者不是 torch.bool 类型，则直接返回
            if not isinstance(x, torch.Tensor) or x.dtype != torch.bool:
                return x

            # 将 False 映射为 0，True 映射为随机值在 [2, 255] 之间
            true_vals = torch.randint(
                2, 255, x.shape, dtype=torch.uint8, device=x.device
            )
            false_vals = torch.zeros((), dtype=torch.uint8, device=x.device)
            x_int = torch.where(x, true_vals, false_vals)

            # 将整数张量重新视为布尔型张量
            ret = x_int.view(torch.bool)
            # 断言转换后的张量与原始张量 x 相等
            self.assertEqual(ret, x)
            return ret

        # 对于 op.sample_inputs 返回的每个样本
        for sample in op.sample_inputs(device, dtype):
            # 调用操作 op，获取期望输出
            expect = op(sample.input, *sample.args, **sample.kwargs)

            # 使用 convert_boolean_tensors 转换样本
            transformed = sample.transform(convert_boolean_tensors)
            # 调用操作 op，获取实际输出
            actual = op(transformed.input, *transformed.args, **transformed.kwargs)

            # 对比期望输出和实际输出是否相等
            self.assertEqual(expect, actual)

    # 验证每个 OpInfo 是否正确指定了其 CPU 和 CUDA 设备上的前向和反向数据类型
    @skipMeta
    @onlyNativeDeviceTypes
    @ops(ops_and_refs, dtypes=OpDTypes.none)
    # 验证每个设置 promotes_int_to_float=True 的 OpInfo 是否如其所说
    # 跳过元信息的测试装饰器
    @skipMeta
    # 仅在本地设备类型上运行的测试装饰器
    @onlyNativeDeviceTypes
    # 对于操作数据库中所有提升整数到浮点数的操作进行测试
    @ops(
        (op for op in op_db if op.promotes_int_to_float),
        # 允许的数据类型为整数类型和 torch.bool 类型
        allowed_dtypes=integral_types_and(torch.bool),
    )
    # 测试整数到浮点数提升的功能
    def test_promotes_int_to_float(self, device, dtype, op):
        # 对于每个操作的样本输入，从操作中获取
        for sample in op.sample_inputs(device, dtype):
            # 调用操作并获取输出
            output = op(sample.input, *sample.args, **sample.kwargs)
            # 如果输出的数据类型不是浮点数类型，则测试失败
            if not output.dtype.is_floating_point:
                self.fail(
                    f"The OpInfo sets `promotes_int_to_float=True`, but {dtype} was promoted to {output.dtype}."
                )
# 标记为不严格的 Dynamo 测试，表示此测试可能不适用于所有情况
@unMarkDynamoStrictTest
# 定义 TestCompositeCompliance 类，用于测试复合操作符的兼容性
class TestCompositeCompliance(TestCase):
    
    # 测试操作符的兼容性，验证其是否支持大多数后端和张量子类
    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ 在 fbcode 中不起作用"
    )
    # 应用操作符数据库中的操作符，并限制数据类型为 torch.float
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_operator(self, device, dtype, op):
        # 生成操作符的样本输入数据，不要求梯度
        samples = op.sample_inputs(device, dtype, requires_grad=False)

        # 遍历样本数据
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 检查操作符在给定模式下的兼容性
            composite_compliance.check_with_mode(op, args, kwargs, self.assertEqual)
            # 检查所有排列组合的兼容性
            composite_compliance.check_all_permutations(
                op, args, kwargs, self.assertEqual
            )

    # 测试反向传播的兼容性
    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ 在 fbcode 中不起作用"
    )
    # 仅选择支持自动求导的操作符，并限制数据类型为 torch.float
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    def test_backward(self, device, dtype, op):
        # 生成操作符的样本输入数据，要求梯度
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 遍历样本数据
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 检查反向传播公式的兼容性
            composite_compliance.check_backward_formula(
                op.get_op(),
                args,
                kwargs,
                sample.output_process_fn_grad,
                op.gradcheck_wrapper,
                self.assertEqual,
            )

    # 测试自动求导的前向传播的兼容性
    @unittest.skipIf(
        IS_FBCODE or IS_SANDCASTLE, "__torch_dispatch__ 在 fbcode 中不起作用"
    )
    # 应用操作符数据库中的操作符，并限制数据类型为 torch.float
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_forward_ad(self, device, dtype, op):
        # 如果操作符不支持在给定设备上的 torch.float 类型的自动求导，则跳过测试
        if torch.float not in op.supported_backward_dtypes(device):
            raise unittest.SkipTest("不支持自动求导")

        # 如果操作符不支持前向自动求导，则跳过测试
        if not op.supports_forward_ad:
            raise unittest.SkipTest("不支持前向自动求导")

        # 生成操作符的样本输入数据，要求梯度
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 遍历样本数据
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            # 检查前向自动求导公式的兼容性
            composite_compliance.check_forward_ad_formula(
                op.get_op(), args, kwargs, op.gradcheck_wrapper, self.assertEqual
            )

    # 应用操作符数据库中的操作符，并限制数据类型为 torch.float
    @ops(op_db, allowed_dtypes=(torch.float,))
    # 应用操作符数据库中的操作符，并限制数据类型为 torch.float
    @ops(op_db, allowed_dtypes=(torch.float,))
    # 定义一个测试方法，用于验证视图重播功能
    def test_view_replay(self, device, dtype, op):
        
        # 定义内部函数，用于断言两个张量的元数据匹配
        def _assert_match_metadata(a, b):
            self.assertEqual(a.size(), b.size())  # 断言张量大小相同
            self.assertEqual(a.stride(), b.stride())  # 断言张量步长相同
            self.assertEqual(a.storage_offset(), b.storage_offset())  # 断言张量存储偏移相同
            self.assertEqual(a.device, b.device)  # 断言张量设备相同
            self.assertEqual(a.dtype, b.dtype)  # 断言张量数据类型相同

        # 确保视图重播功能被启用
        with torch.autograd._force_original_view_tracking(True):
            # 对操作的样本输入进行迭代
            for sample in op.sample_inputs(device, dtype, requires_grad=False):
                inp = sample.input  # 获取样本输入
                outs = op(inp, *sample.args, **sample.kwargs)  # 执行操作得到输出
                if not isinstance(outs, (tuple, List)):
                    outs = [outs]

                # 对于所有作为输入视图的输出，应能够通过视图函数和反向视图函数进行正向和反向重播
                for out in outs:
                    if not (
                        isinstance(out, torch.Tensor)  # 输出必须是张量
                        and out._is_view()  # 输出必须是视图
                        and out._base is inp  # 输出的基张量必须是输入张量
                    ):
                        continue

                    # 正向视图函数
                    new_inp = inp.clone()  # 克隆输入张量
                    _assert_match_metadata(new_inp, inp)  # 断言克隆后的元数据与原始输入相同
                    new_out = out._view_func_unsafe(new_inp)  # 使用视图函数创建新的输出
                    _assert_match_metadata(new_out, out)  # 断言新输出的元数据与原输出相同
                    self.assertEqual(new_out, out)  # 断言新输出与原输出相等

                    # 反向视图函数
                    new_out = out.detach()  # 分离原输出
                    new_inp = out._rev_view_func_unsafe(new_out)  # 使用反向视图函数创建新的输入
                    _assert_match_metadata(new_inp, inp)  # 断言新输入的元数据与原始输入相同
                    self.assertTrue(new_inp._is_view())  # 断言新输入是视图
                    self.assertTrue(new_inp._base is new_out)  # 断言新输入的基张量是新输出
# 定义一个测试类 TestMathBits，用于测试数学运算相关的功能
@unMarkDynamoStrictTest
class TestMathBits(TestCase):
    # 测试以下几个方面：
    # 1. 物理共轭/负数张量及其视图的运算结果是否相同
    # 2. 在上述情况下梯度是否相同
    # 3. 如果操作支持原地变体，则测试在共轭/负数视图张量上调用 inplace 操作的正确性，
    #    并确保输出的 conj/neg 位设置为 true
    # 此测试仅适用于 C -> R 和 C -> C 的函数
    # TODO: 添加对 `R->C` 函数的测试
    # 注意：此测试适用于接受张量和张量列表作为输入的函数。
    def _test_math_view(
        self,
        device,
        dtype,
        op,
        samples,
        math_op_physical,
        math_op_view,
        is_bit_set,
        out_type,
    ):
        # 检查操作是否支持复数类型（torch.cfloat）
        @ops(ops_and_refs, allowed_dtypes=(torch.cfloat,))
        def test_conj_view(self, device, dtype, op):
            # 如果操作不支持共轭输入，则跳过该测试
            if not op.test_conjugated_samples:
                self.skipTest("Operation doesn't support conjugated inputs.")
            # 设置物理共轭操作
            math_op_physical = torch.conj_physical
            # 设置视图共轭操作
            math_op_view = torch.conj
            # 检查是否需要计算梯度
            _requires_grad = torch.cfloat in op.supported_backward_dtypes(
                torch.device(device).type
            )
            # 设置是否为共轭位
            is_bit_set = torch.is_conj
            # 获取样本输入
            samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
            # 执行具体的数学视图测试
            self._test_math_view(
                device,
                dtype,
                op,
                samples,
                math_op_physical,
                math_op_view,
                is_bit_set,
                torch.is_complex,
            )

        # 检查操作是否支持双精度类型（torch.double）
        @ops(ops_and_refs, allowed_dtypes=(torch.double,))
        def test_neg_view(self, device, dtype, op):
            # 如果操作不支持负数视图，则跳过该测试
            if not op.test_neg_view:
                self.skipTest("Operation not tested with tensors with negative bit.")
            # 设置物理负数操作
            math_op_physical = torch.neg
            # 设置视图负数操作
            math_op_view = torch._neg_view
            # 设置是否为负数位
            is_bit_set = torch.is_neg
            # 获取样本输入
            samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)
            # 执行具体的数学视图测试
            self._test_math_view(
                device,
                dtype,
                op,
                samples,
                math_op_physical,
                math_op_view,
                is_bit_set,
                lambda x: True,
            )

        # 检查操作是否支持双复数类型（torch.cdouble）
    # 定义一个测试方法，用于测试具有负视图和共轭样本的操作
    def test_neg_conj_view(self, device, dtype, op):
        # 如果操作不支持负视图，则跳过测试
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        # 如果操作不支持共轭输入，则跳过测试
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")

        # 定义物理操作函数，对输入取负共轭
        def math_op_physical(x):
            return -x.conj_physical()

        # 定义视图操作函数，对输入取负共轭
        def math_op_view(x):
            return torch._neg_view(x).conj()

        # 定义函数，判断输入是否同时具有负和共轭标志位
        def is_bit_set(x):
            return torch.is_neg(x) and torch.is_conj(x)

        # 检查当前数据类型是否在操作支持的反向传播数据类型中
        _requires_grad = dtype in op.supported_backward_dtypes(
            torch.device(device).type
        )
        # 生成操作的样本输入，仅选择第一个样本进行测试
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        samples = itertools.islice(samples, 1)
        # 调用内部方法测试数学视图
        self._test_math_view(
            device,
            dtype,
            op,
            samples,
            math_op_physical,
            math_op_view,
            is_bit_set,
            torch.is_complex,
        )
# 当前函数用于检查 inplace 操作的影响，由于原地操作的结果可能会改变输入的步幅和大小
def check_inplace_view(func, input, rs, input_size, input_strides):
    if func is None:
        return
    
    # TODO: 将此测试扩展到测试具有多个输出和像 native_batch_norm(_legit).out 这样的操作，
    # 它们可能改变的不一定是第一个输入。
    if isinstance(rs, torch.Tensor) and rs is input:
        unequal_size = rs.size() != input_size
        unequal_strides = rs.stride() != input_strides
        
        # resize_ 应该可能具有 inplace_view 标签。由于它会破坏某些代码生成逻辑，因此没有添加该标签。
        if unequal_size or unequal_strides:
            if isinstance(func, torch._ops.OpOverloadPacket):
                func = func.default
            
            # 引用：https://github.com/pytorch/pytorch/issues/78759
            if func is not torch.ops.aten.resize_.default:
                # TODO: 当我们有单独的测试用例来测试每个标签时，使用 self.assertIn
                assert torch.Tag.inplace_view in func.tags


# 一种模式，启用时会运行正确性检查，以确保操作符根据其输入和输出张量属性具有预期的标签
class TestTagsMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if isinstance(args[0], torch.Tensor):
            old_size = args[0].size()
            old_stride = args[0].stride()
            rs = func(*args, **kwargs)
            check_inplace_view(func, args[0], rs, old_size, old_stride)
        else:
            rs = func(*args, **kwargs)
        return rs


# 用于验证 `tags.yaml` 中标签正确性的测试，也可以通过 `torch.Tags` 访问
@unMarkDynamoStrictTest
class TestTags(TestCase):
    @onlyCPU
    @ops(ops_and_refs, dtypes=OpDTypes.any_one)
    def test_tags(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample in samples:
            # TODO: 测试返回张量列表的操作的标签
            input = sample.input
            if isinstance(input, torch.Tensor):
                old_size = input.size()
                old_stride = input.stride()
                with TestTagsMode():
                    rs = op(input, *sample.args, **sample.kwargs)
                # TODO: 添加关于别名的测试：https://github.com/pytorch/pytorch/issues/78761
                aten_name = op.aten_name if op.aten_name is not None else op.name
                opoverloadpacket = getattr(torch.ops.aten, aten_name, None)
                check_inplace_view(opoverloadpacket, input, rs, old_size, old_stride)


# 用于测试验证即使参数名称为 "self"，也可以使用所有 kwargs 调用 aten 操作的测试类
class TestSelfKwarg(TestCase):
    def test_self_kwargs(self):
        """验证即使参数名称为 "self"，也可以使用所有 kwargs 调用 aten 操作"""
        torch.ops.aten.reshape.default(self=torch.rand(1, 2), shape=[2])
        torch.ops.aten.min.default(self=torch.rand(100))


@unMarkDynamoStrictTest
class TestRefsOpsInfo(TestCase):
    # 需要导入的模块路径列表
    import_paths = [
        "_refs",
        "_refs.special",
        "_refs.nn.functional",
        "_refs.fft",
        "_refs._conversions",
    ]
    # 每个模块路径及其 __all__ 列表的元组列表
    module_alls = [
        (path, import_module(f"torch.{path}").__all__) for path in import_paths
    ]
    # 所有引用操作的名称，通过迭代器链连接所有模块的操作名称
    ref_ops_names = tuple(
        itertools.chain.from_iterable(
            [f"{path}.{op}" for op in module_all] for path, module_all in module_alls
        )
    )
    # Python 参考数据库中存在的引用操作名称集合
    ref_db_names = {ref_op.name for ref_op in python_ref_db}

    # TODO: References that do not have an entry in python_ref_db
    # 不在 python_ref_db 中的引用操作集合
    skip_ref_ops = {
        "_refs.alias",
        "_refs.bitwise_right_shift",
        "_refs.copy_to",
        "_refs.empty_permuted",
        "_refs.empty_strided",
        "_refs.equal",
        "_refs.full",
        "_refs.full_like",
        "_refs.is_complex",
        "_refs.to",
        "_refs.mvlgamma",
        "_refs.ones",
        "_refs.ones_like",
        "_refs.special.expit",
        "_refs.std_var",
        "_refs.swap_axes",
        "_refs.uniform",
        "_refs.scalar_tensor",
        "_refs.trunc_divide",
        "_refs.zero",
        "_refs.zeros",
        "_refs.zeros_like",
        "_refs.rfloordiv",
        "_refs.rtruediv",
        "_refs.rpow",
        # These should be tested with their out-of-place counterparts
        "_refs.index_add_",
        "_refs.index_copy_",
        "_refs.index_fill_",
        "_refs.native_group_norm",
    }

    # 测试方法：确保所有引用操作都在 python_ref_db 中
    @parametrize("op", ref_ops_names)
    def test_refs_are_in_python_ref_db(self, op):
        # 检查是否为原地操作
        inplace = op[-1] == "_"
        if op in self.skip_ref_ops:
            # 如果操作在 skip_ref_ops 中，跳过测试
            raise unittest.SkipTest(f"{op} does not have an entry in python_ref_db")
        elif inplace:
            # 对于原地操作，确保不在 ref_db_names 中
            self.assertNotIn(
                op,
                self.ref_db_names,
                msg=f"{op} is an in-place operation and should not have an OpInfo",
            )
        else:
            # 对于非原地操作，确保在 ref_db_names 中
            # 意图是避免使用 assertIn，以避免打印非常大的容器
            self.assertTrue(op in self.ref_db_names, msg=f"{op} not in ref_db_names")

    # 测试方法：确保所有引用操作都在 decomposition_table 中
    @parametrize("op", ref_ops_names)
    def test_refs_are_in_decomp_table(self, op):
        # 将操作名分解为模块路径和操作名
        path = op.split(".")
        module_path = ".".join(path[:-1])
        op_name = path[-1]
        # 获取操作的实现对象
        op_impl = getattr(import_module(f"torch.{module_path}"), op_name)

        if op in self.not_in_decomp_table:
            # 如果操作不在 not_in_decomp_table 中，则确保不在 decomposition_table.values() 中
            self.assertNotIn(
                op_impl,
                torch._decomp.decomposition_table.values(),
                f"Unexpectedly found {op} in torch._decomp.decomposition_table.values()",
            )
        else:
            # 否则，确保在 decomposition_table.values() 中
            self.assertIn(
                op_impl,
                torch._decomp.decomposition_table.values(),
                f"Did not find {op} in torch._decomp.decomposition_table.values()",
            )


fake_skips = (
    "aminmax",  # failing input
    "cov",  # aweights cannot be negtaive
)
    "istft",  # 在执行 istft 操作时，窗口重叠添加最小值为 0
    "linalg.eigvals",  # 张量具有非零元素数，但其数据尚未分配
    "linalg.eigvalsh",  # 使用 'Meta' 后端从 'aten::linalg_eigvalsh.out' 运行
    "linalg.matrix_power",  # 无法运行 'aten::eye.m_out'，使用 'Meta' 后端
    # "linalg.pinv",  # 无法运行 'aten::pinv.out'，使用 'Meta' 后端
    "linalg.matrix_rank.hermitian",  # 无法运行 'aten::linalg_eigvalsh.out'，使用 'Meta' 后端
    "linalg.pinv.hermitian",  # tensor.mH 仅支持矩阵或矩阵批次。输入是 1-D 张量
    "linalg.solve",  # 无法运行 'aten::linalg_solve'，使用 'Meta' 后端
    "linalg.tensorsolve",  # 无法运行 'aten::linalg_solve'，使用 'Meta' 后端
    "lu_solve",  # 内存分配错误：调试中
    "multinomial",  # 无法运行 'aten::multinomial'，使用 'Meta' 后端
    "mvlgamma.mvlgamma_p_1",  # 无法运行 'aten::_local_scalar_dense'，使用 'Meta' 后端
    "mvlgamma.mvlgamma_p_3",  # 无法运行 'aten::_local_scalar_dense'，使用 'Meta' 后端
    "mvlgamma.mvlgamma_p_5",  # 无法运行 'aten::_local_scalar_dense'，使用 'Meta' 后端
    "nanmean",  # logical_not() 函数意外地获得了 'out' 关键字参数
    "quantile",  # quantile() 函数的 q 值必须在 [0, 1] 范围内
    "nanquantile",  # quantile() 函数的 q 值必须在 [0, 1] 范围内
    "nn.functional.ctc_loss",  # 张量具有非零元素数，但其数据尚未分配
    "nn.functional.embedding_bag",  # 有时会出现错误
    "nn.functional.nll_loss",  # 有时会出现错误
    "nn.functional.max_pool1d",  # 张量具有非零元素数
    "to_sparse",  # 无法运行 'aten::_to_sparse'，使用 'Meta' 后端
    "tensor_split",  # 张量具有非零元素数，但其数据尚未分配
    "repeat_interleave",  # 无法重复 interleave 一个元数据张量，需要 output_size
    "sparse.sampled.addmm",  # 不支持稀疏性
    # 当前无法从元数据中推断出总类数，因此无法抛出 DynamicOutputShapeException
    "nn.functional.one_hot",
    "narrow",  # 仅对一个 DataDependentOutputException 过载失败（因此跳过）。
# 创建一个空的默认字典，用于存储键为字符串、值为字典的数据结构
fake_autocast_device_skips = defaultdict(dict)

# TODO: investigate/fix
# 在 fake_autocast_device_skips 字典中添加一个键为 "cpu" 的条目，对应的值是包含 "linalg.pinv" 字符串的集合
fake_autocast_device_skips["cpu"] = {"linalg.pinv"}

# 定义一个包含多个字符串元素的元组，每个字符串代表一个动态输出操作的测试名称
dynamic_output_op_tests = (
    "argwhere",
    "bincount",
    "combinations",
    "linalg.lstsq",
    "masked_select",
    "nonzero",
    "unique_consecutive",
    "unique",
    "linalg.lstsq.grad_oriented",
)

# 在 allow_dynamic_shape_ops 为 True 的情况下，我们能够处理的具有动态输出形状的操作的测试名称
supported_dynamic_output_op_tests = (
    "nonzero",
    "unique",
    "repeat_interleave",
    "masked_select",
)

# 一些输入调用动态输出形状运算符，而其他一些则不调用
sometimes_dynamic_output_op_test = (
    "__getitem__",
    "index_select",
)

# 包含一组数据相关操作的测试名称
data_dependent_op_tests = (
    "equal",
    "corrcoef",
    "nn.functional.gaussian_nll_loss",
    "allclose",
)

# 包含会导致别名问题的操作的测试名称
aliasing_failures = ("histogramdd",)

# 创建一个集合，其中包含需要跳过反向传播的操作的名称
fake_backward_skips = {
    "linalg.cond",
    "linalg.matrix_norm",
    "linalg.norm",
    "linalg.svd",
    "linalg.svdvals",
    "pca_lowrank",
    "roll",
    "svd_lowrank",
    "sgn",
}

# 创建一个集合，其中包含需要标记为反向传播失败的操作的名称
fake_backward_xfails = {skip(s) for s in fake_backward_skips} | {
    xfail("fft.ihfftn"),  # 在 aten._conj_physical.default 中存在不匹配
    xfail("fft.ihfft2"),  # 在 aten._conj_physical.default 中存在不匹配
    skip("nn.functional.ctc_loss"),
}

# 创建一个集合，其中包含自动转换后向传播失败的操作的名称
fake_autocast_backward_xfails = {
    skip("nn.functional.binary_cross_entropy"),
    skip("sparse.sampled_addmm"),
    skip("linalg.pinv"),
    skip("linalg.pinv", "hermitian"),
    skip("linalg.pinv", "singular"),
    skip("pinverse"),
}

# 用于标记未受 Dynamo 严格测试装饰器保护的 TestFakeTensor 测试类
@unMarkDynamoStrictTest
class TestFakeTensor(TestCase):
    def setUp(self):
        # 开启 FakeTensor 的缓存和交叉检查功能以用于这些测试：
        cache_enabled = unittest.mock.patch(
            "torch._dynamo.config.fake_tensor_cache_enabled", True
        )
        cache_enabled.start()
        self.addCleanup(cache_enabled.stop)

        cache_crosscheck = unittest.mock.patch(
            "torch._dynamo.config.fake_tensor_cache_crosscheck_enabled", True
        )
        cache_crosscheck.start()
        self.addCleanup(cache_crosscheck.stop)

    # 使用 op_db 数据库和任意一种数据类型执行操作的测试
    @ops(op_db, dtypes=OpDTypes.any_one)
    # 定义一个测试点对点运算的方法，接受设备、数据类型和操作作为参数
    def test_pointwise_ops(self, device, dtype, op):
        # 获取操作的名称
        name = op.name
        # 如果操作有变体测试名称，则添加到名称中
        if op.variant_test_name:
            name += "." + op.variant_test_name
        # 如果名称在假跳过列表中，或者名称中包含"sparse"或"jiterator"，则跳过此测试
        if name in fake_skips or "sparse" in name or "jiterator" in name:
            self.skipTest("Skip failing test")

        # 将当前测试对象赋值给test_self变量
        test_self = self

        # 定义一个测试点对点模式类，继承自TorchDispatchMode
        class TestPointwiseMode(TorchDispatchMode):
            # 定义torch_dispatch方法，接受func、types、args和kwargs作为参数
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}

                # 调用func函数，传入args和kwargs，获取输出out
                out = func(*args, **kwargs)

                # 如果func被标记为torch.Tag.pointwise
                if torch.Tag.pointwise in func.tags:
                    # 初始化一个空列表用于存储输入的形状
                    shapes = []
                    # 遍历args和kwargs中的所有叶子节点
                    for inp in pytree.arg_tree_leaves(*args, **kwargs):
                        if isinstance(inp, torch.Tensor):
                            shapes.append(inp.shape)

                    # 计算输出的形状，调用torch._refs._broadcast_shapes函数
                    out_shape = torch._refs._broadcast_shapes(*shapes)

                    # 遍历输出out中的所有叶子节点
                    for out_elem in pytree.tree_leaves(out):
                        if isinstance(out_elem, torch.Tensor):
                            # 断言每个张量的形状与计算得到的out_shape相等
                            test_self.assertEqual(out_elem.shape, out_shape)

                # 返回函数的输出out
                return out

        # 使用操作的sample_inputs方法获取样本数据集
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        # 遍历样本数据集中的每一个样本
        for sample in samples:
            # 创建一个FakeTensorMode实例mode
            mode = FakeTensorMode()

            # 定义一个映射函数map_to_fake，将输入的张量映射到FakeTensorMode实例中的形式
            def map_to_fake(e):
                if isinstance(e, torch.Tensor):
                    return mode.from_tensor(e)
                else:
                    return e

            # 使用tree_map函数，将样本的输入、args和kwargs映射到FakeTensorMode实例中的形式
            input = tree_map(map_to_fake, sample.input)
            args = tree_map(map_to_fake, sample.args)
            kwargs = tree_map(map_to_fake, sample.kwargs)

            try:
                # 调用操作op，传入映射后的输入、args和kwargs
                op(input, *args, **kwargs)
            except Exception as e:
                # 如果出现异常，继续下一个样本的测试
                continue

            # 使用TestPointwiseMode上下文管理器
            with TestPointwiseMode():
                # 使用mode上下文管理器
                with mode:
                    # 再次调用操作op，传入映射后的输入、args和kwargs
                    op(input, *args, **kwargs)

    # 使用ops装饰器，为op_db中的每一个操作定义测试方法
    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        # 调用_test_fake_helper方法，传入设备、数据类型、操作和nullcontext上下文
        self._test_fake_helper(device, dtype, op, contextlib.nullcontext)

    # 使用ops装饰器，为op_db中的每一个操作定义自动转换测试方法
    @ops(op_db, dtypes=OpDTypes.any_one)
    def test_fake_autocast(self, device, dtype, op):
        # 如果操作名称在当前设备的fake_autocast_device_skips中，则跳过测试
        if op.name in fake_autocast_device_skips[device]:
            self.skipTest("Skip failing test")
        # 根据设备选择不同的autocast上下文
        context = (
            torch.cuda.amp.autocast if device == "cuda" else torch.cpu.amp.autocast
        )
        # 调用_test_fake_helper方法，传入设备、数据类型、操作和选择的autocast上下文
        self._test_fake_helper(device, dtype, op, context)
    # 定义一个辅助测试方法，用于测试伪造的交叉引用（fake crossref）
    def _test_fake_crossref_helper(self, device, dtype, op, context):
        # 使用操作对象的方法获取样本数据，要求梯度为True
        samples = op.sample_inputs(device, dtype, requires_grad=True)

        # 遍历样本数据
        for iter, sample in enumerate(samples):
            # 构建参数列表，包括输入数据和其他参数
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs

            # 定义一组常见的操作集合，用于加速测试（跳过这些操作）
            common_skip_ops = (
                aten.detach.default,
                aten.empty_strided.default,
                aten.copy_.default,
                aten.is_same_size.default,
            )

            # TODO: 启用检查别名，但批量标准化操作会失败
            try:
                # 使用伪造的交叉引用（CrossRefFakeMode）执行操作
                with torch._subclasses.CrossRefFakeMode(
                    ignore_op_fn=lambda fn: fn in common_skip_ops, check_aliasing=True
                ):
                    # 使用警告捕获、指定的上下文和关闭多线程功能，计算预期的梯度
                    with warnings.catch_warnings(), context(), torch.autograd.set_multithreading_enabled(
                        False
                    ):
                        composite_compliance.compute_expected_grads(
                            op.get_op(),
                            args,
                            kwargs,
                            sample.output_process_fn_grad,
                            op.gradcheck_wrapper,
                        )
            # 捕获不支持的操作异常
            except torch._subclasses.fake_tensor.UnsupportedOperatorException:
                pass

    # 使用CUDA时运行的测试方法，仅测试支持自动求导的操作
    @onlyCUDA
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps(
        "TestFakeTensor", "test_fake_crossref_backward_no_amp", fake_backward_xfails
    )
    def test_fake_crossref_backward_no_amp(self, device, dtype, op):
        # 调用辅助方法，传入设备、数据类型、操作对象以及空上下文
        self._test_fake_crossref_helper(device, dtype, op, contextlib.nullcontext)

    # 使用CUDA时运行的测试方法，仅测试支持自动求导的操作，同时支持自动混合精度
    @onlyCUDA
    @ops([op for op in op_db if op.supports_autograd], allowed_dtypes=(torch.float,))
    @skipOps(
        "TestFakeTensor",
        "test_fake_crossref_backward_amp",
        fake_backward_xfails | fake_autocast_backward_xfails,
    )
    def test_fake_crossref_backward_amp(self, device, dtype, op):
        # 调用辅助方法，传入设备、数据类型、操作对象以及CUDA自动混合精度上下文
        self._test_fake_crossref_helper(device, dtype, op, torch.cuda.amp.autocast)

    # 测试 strided 布局的方法
    @ops([op for op in ops_and_refs if op.is_factory_function])
    def test_strided_layout(self, device, dtype, op):
        # 使用操作对象的方法获取样本数据
        samples = op.sample_inputs(device, dtype)
        # 遍历样本数据
        for sample in samples:
            kwargs = sample.kwargs.copy()
            kwargs["layout"] = torch.strided
            # 执行操作，并断言结果的布局是否为 strided
            strided_result = op(sample.input, *sample.args, **kwargs)
            self.assertEqual(strided_result.layout, torch.strided)
# 调用函数实例化测试用例，针对 TestCommon 类的测试，使用全局命名空间
instantiate_device_type_tests(TestCommon, globals())

# 调用函数实例化测试用例，针对 TestCompositeCompliance 类的测试，使用全局命名空间
instantiate_device_type_tests(TestCompositeCompliance, globals())

# 调用函数实例化测试用例，针对 TestMathBits 类的测试，使用全局命名空间
instantiate_device_type_tests(TestMathBits, globals())

# 调用函数实例化测试用例，针对 TestRefsOpsInfo 类的测试，只在 "cpu" 环境下使用全局命名空间
instantiate_device_type_tests(TestRefsOpsInfo, globals(), only_for="cpu")

# 调用函数实例化测试用例，针对 TestFakeTensor 类的测试，使用全局命名空间
instantiate_device_type_tests(TestFakeTensor, globals())

# 调用函数实例化测试用例，针对 TestTags 类的测试，使用全局命名空间
instantiate_device_type_tests(TestTags, globals())

# 如果该脚本作为主程序运行，则设置 TestCase 类的默认数据类型检查为启用状态
if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    
    # 运行所有的测试用例
    run_tests()
```