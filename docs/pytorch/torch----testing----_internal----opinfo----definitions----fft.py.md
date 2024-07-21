# `.\pytorch\torch\testing\_internal\opinfo\definitions\fft.py`

```py
# 忽略 mypy 类型检查的错误
# 导入单元测试模块
import unittest
# 导入 functools 模块的 partial 函数
from functools import partial
# 导入 List 类型提示
from typing import List

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 torch 库
import torch

# 从 torch.testing 模块导入 make_tensor 函数
from torch.testing import make_tensor
# 从 torch.testing._internal.common_cuda 模块导入 SM53OrLater 类
from torch.testing._internal.common_cuda import SM53OrLater
# 从 torch.testing._internal.common_device_type 模块导入 precisionOverride 函数
from torch.testing._internal.common_device_type import precisionOverride
# 从 torch.testing._internal.common_dtype 模块导入各种类型的常量
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
)
# 从 torch.testing._internal.common_utils 模块导入 TEST_SCIPY 和 TEST_WITH_ROCM 常量
from torch.testing._internal.common_utils import TEST_SCIPY, TEST_WITH_ROCM
# 从 torch.testing._internal.opinfo.core 模块导入多个类和函数
from torch.testing._internal.opinfo.core import (
    DecorateInfo,
    ErrorInput,
    OpInfo,
    sample_inputs_spectral_ops,
    SampleInput,
    SpectralFuncInfo,
    SpectralFuncType,
)
# 从 torch.testing._internal.opinfo.refs 模块导入多个函数
from torch.testing._internal.opinfo.refs import (
    _find_referenced_opinfo,
    _inherit_constructor_args,
    PythonRefInfo,
)

# 初始化一个标志，表示是否导入了 scipy.fft
has_scipy_fft = False
# 如果测试标志 TEST_SCIPY 为真，则尝试导入 scipy.fft
if TEST_SCIPY:
    try:
        import scipy.fft
        # 若成功导入，则设置标志为真
        has_scipy_fft = True
    except ModuleNotFoundError:
        # 若导入失败，则忽略异常
        pass


class SpectralFuncPythonRefInfo(SpectralFuncInfo):
    """
    用于元素级一元操作的 Python 参考的 OpInfo 类。
    """

    def __init__(
        self,
        name,  # 可调用 Python 参考的字符串名称
        *,
        op=None,  # 操作的函数变体，如果为 None，则填充为 torch.<name>
        torch_opinfo_name,  # 对应 torch opinfo 的字符串名称
        torch_opinfo_variant="",  # torch opinfo 的变体字符串
        **kwargs,
    ):  # 额外的 kwargs 覆盖从 torch opinfo 继承的 kwargs
        self.torch_opinfo_name = torch_opinfo_name
        # 查找引用的 torch opinfo 对象
        self.torch_opinfo = _find_referenced_opinfo(
            torch_opinfo_name, torch_opinfo_variant, op_db=op_db
        )
        # 断言确保找到的是 SpectralFuncInfo 类的实例
        assert isinstance(self.torch_opinfo, SpectralFuncInfo)

        # 继承自 torch opinfo 的原始谱函数参数
        inherited = self.torch_opinfo._original_spectral_func_args
        # 继承构造函数参数，并使用 kwargs 覆盖
        ukwargs = _inherit_constructor_args(name, op, inherited, kwargs)

        # 调用父类的构造函数
        super().__init__(**ukwargs)


def error_inputs_fft(op_info, device, **kwargs):
    # partial 函数创建 make_tensor 函数的一个包装，设定设备和数据类型
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    # 生成一个错误输入，表示零维张量没有维度可以进行 FFT 操作
    yield ErrorInput(
        SampleInput(make_arg()),
        error_type=IndexError,
        error_regex="Dimension specified as -1 but tensor has no dimensions",
    )


def error_inputs_fftn(op_info, device, **kwargs):
    # partial 函数创建 make_tensor 函数的一个包装，设定设备和数据类型
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    # 生成一个错误输入，表示在零维张量上指定维度进行 FFTN 操作
    yield ErrorInput(
        SampleInput(make_arg(), dim=(0,)),
        error_type=IndexError,
        error_regex="Dimension specified as 0 but tensor has no dimensions",
    )


def sample_inputs_fft_with_min(
    op_info, device, dtype, requires_grad=False, *, min_size, **kwargs
):
    # 生成一系列谱操作的输入样本
    yield from sample_inputs_spectral_ops(
        op_info, device, dtype, requires_grad, **kwargs
    )
    # 如果在 ROCm 平台上进行测试，则返回
    if TEST_WITH_ROCM:
        # FIXME: 在 ROCm 平台上会导致浮点异常
        return
    # 检查“无效的数据点数”错误是否太严格
    # https://github.com/pytorch/pytorch/pull/109083

    # 使用 make_tensor 函数创建张量 a，指定最小尺寸、数据类型、设备和梯度需求
    a = make_tensor(min_size, dtype=dtype, device=device, requires_grad=requires_grad)

    # 生成一个 SampleInput 对象，包含张量 a 作为输入
    yield SampleInput(a)
# 定义一个函数用于生成包含样本输入的迭代器，每个样本是一个张量
# 参数 op_info: 操作信息对象
# 参数 device: 设备类型（CPU 或 CUDA）
# 参数 dtype: 数据类型
# 参数 requires_grad: 是否需要梯度
# 参数 kwargs: 其他关键字参数
def sample_inputs_fftshift(op_info, device, dtype, requires_grad, **kwargs):
    # 定义内部函数 mt，用于创建张量
    def mt(shape, **kwargs):
        return make_tensor(
            shape, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs
        )

    # 生成并返回一个包含样本输入的迭代器
    yield SampleInput(mt((9, 10)))
    yield SampleInput(mt((50,)), kwargs=dict(dim=0))
    yield SampleInput(mt((5, 11)), kwargs=dict(dim=(1,)))
    yield SampleInput(mt((5, 6)), kwargs=dict(dim=(0, 1)))
    yield SampleInput(mt((5, 6, 2)), kwargs=dict(dim=(0, 2)))


# Operator database
# 定义操作信息对象列表 op_db
op_db: List[OpInfo] = [
    SpectralFuncInfo(
        "fft.fft",  # 操作名称
        aten_name="fft_fft",  # ATen 函数名称
        decomp_aten_name="_fft_c2c",  # 复杂 FFT 的 ATen 函数名称
        ref=np.fft.fft,  # 参考实现（NumPy 中的 FFT 函数）
        ndimensional=SpectralFuncType.OneD,  # 操作的维度类型（一维）
        dtypes=all_types_and_complex_and(torch.bool),  # 所支持的数据类型（包括布尔型）
        # CUDA 只支持半精度和复杂半精度 FFT 在 SM53 或更高的架构上
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=1),  # 生成样本输入的函数
        error_inputs_func=error_inputs_fft,  # 生成错误输入的函数
        # 启用快速梯度检查模式
        gradcheck_fast_mode=True,
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        # 参考 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
    ),
    SpectralFuncInfo(
        "fft.fft2",  # 操作名称
        aten_name="fft_fft2",  # ATen 函数名称
        ref=np.fft.fft2,  # 参考实现（NumPy 中的 2D FFT 函数）
        decomp_aten_name="_fft_c2c",  # 复杂 FFT 的 ATen 函数名称
        ndimensional=SpectralFuncType.TwoD,  # 操作的维度类型（二维）
        dtypes=all_types_and_complex_and(torch.bool),  # 所支持的数据类型（包括布尔型）
        # CUDA 只支持半精度和复杂半精度 FFT 在 SM53 或更高的架构上
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),  # 生成样本输入的函数
        error_inputs_func=error_inputs_fftn,  # 生成错误输入的函数
        # 启用快速梯度检查模式
        gradcheck_fast_mode=True,
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        # 参考 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})],  # 添加修饰器，设置精度
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),  # 跳过测试的信息
                "TestCommon",  # 测试的类别
                "test_complex_half_reference_testing",  # 测试方法名称
                device_type="cuda",  # 设备类型为 CUDA
                dtypes=[torch.complex32],  # 测试的数据类型
                active_if=TEST_WITH_ROCM,  # 如果在 ROCM 下激活
            ),
        ),
    ),
    # 创建 SpectralFuncInfo 对象，描述 FFT 函数 fft.fftn
    SpectralFuncInfo(
        "fft.fftn",
        aten_name="fft_fftn",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.fftn,
        ndimensional=SpectralFuncType.ND,
        dtypes=all_types_and_complex_and(torch.bool),
        # 仅当 CUDA 支持半精度/复数半精度 FFT 且在 SM53 或更新的架构上时才包括
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        # 用于生成 FFT 函数的输入样本的函数，保证最小尺寸为 (1, 1)
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),
        # 用于 FFT 函数的错误输入样本的函数
        error_inputs_func=error_inputs_fftn,
        # 控制是否使用快速模式进行梯度检查
        gradcheck_fast_mode=True,
        # 支持正向自动求导
        supports_forward_ad=True,
        # 支持正向-反向梯度双向传播
        supports_fwgrad_bwgrad=True,
        # 查看批处理正向梯度的问题，详见 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        # 对于特定精度的覆盖修饰器，控制误差容差
        decorators=[precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4})],
    ),
    # 创建 SpectralFuncInfo 对象，描述 Hermite FFT 函数 fft.hfft
    SpectralFuncInfo(
        "fft.hfft",
        aten_name="fft_hfft",
        decomp_aten_name="_fft_c2r",
        ref=np.fft.hfft,
        ndimensional=SpectralFuncType.OneD,
        dtypes=all_types_and_complex_and(torch.bool),
        # 仅当 CUDA 支持半精度/复数半精度 FFT 且在 SM53 或更新的架构上时才包括
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        # 用于生成 Hermite FFT 函数的输入样本的函数，保证最小尺寸为 2
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=2),
        # 用于 Hermite FFT 函数的错误输入样本的函数
        error_inputs_func=error_inputs_fft,
        # 控制是否使用快速模式进行梯度检查
        gradcheck_fast_mode=True,
        # 支持正向自动求导
        supports_forward_ad=True,
        # 支持正向-反向梯度双向传播
        supports_fwgrad_bwgrad=True,
        # 查看批处理正向和反向梯度的问题，详见 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        # 查看批处理二阶梯度的问题
        check_batched_gradgrad=False,
        # 跳过的测试信息，详见 https://github.com/pytorch/pytorch/issues/82479
        skips=(
            # 与 conj 和 torch 分派有关的问题，详见 https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
        ),
    ),
    SpectralFuncInfo(
        "fft.hfft2",  # 定义一个 SpectralFuncInfo 对象，表示 2D 傅里叶变换
        aten_name="fft_hfft2",  # 对应的 PyTorch 原生函数名
        decomp_aten_name="_fft_c2r",  # 相关的分解函数名
        ref=scipy.fft.hfft2 if has_scipy_fft else None,  # 参考实现，如果有 SciPy FFT 库的话
        ndimensional=SpectralFuncType.TwoD,  # 表示是二维的频谱函数
        dtypes=all_types_and_complex_and(torch.bool),  # 支持的数据类型，包括布尔类型
        # CUDA 支持 Half/ComplexHalf 精度的 FFT，但仅适用于 SM53 或更新的架构
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(2, 2)),  # 获取示例输入的函数
        error_inputs_func=error_inputs_fftn,  # 获取错误输入的函数
        # 开启快速模式的梯度检查
        gradcheck_fast_mode=True,
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        check_batched_gradgrad=False,  # 不检查批处理的二阶梯度
        # 参考 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,  # 不检查批处理的前向梯度
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),  # 设置精度覆盖
                "TestFFT",  # 测试用例名称
                "test_reference_nd",  # 测试的参考名称
            )
        ],
        skips=(
            # 存在 conj 和 torch 分发的问题，参见 https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过
                "TestSchemaCheckModeOpInfo",  # 测试用例名称
                "test_schema_correctness",  # 测试模式正确性
            ),
        ),
    ),
    SpectralFuncInfo(
        "fft.hfftn",  # 定义一个 SpectralFuncInfo 对象，表示 N 维傅里叶变换
        aten_name="fft_hfftn",  # 对应的 PyTorch 原生函数名
        decomp_aten_name="_fft_c2r",  # 相关的分解函数名
        ref=scipy.fft.hfftn if has_scipy_fft else None,  # 参考实现，如果有 SciPy FFT 库的话
        ndimensional=SpectralFuncType.ND,  # 表示是 N 维的频谱函数
        dtypes=all_types_and_complex_and(torch.bool),  # 支持的数据类型，包括布尔类型
        # CUDA 支持 Half/ComplexHalf 精度的 FFT，但仅适用于 SM53 或更新的架构
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(2, 2)),  # 获取示例输入的函数
        error_inputs_func=error_inputs_fftn,  # 获取错误输入的函数
        # 开启快速模式的梯度检查
        gradcheck_fast_mode=True,
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向梯度和反向梯度
        check_batched_gradgrad=False,  # 不检查批处理的二阶梯度
        # 参考 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,  # 不检查批处理的前向梯度
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),  # 设置精度覆盖
                "TestFFT",  # 测试用例名称
                "test_reference_nd",  # 测试的参考名称
            ),
        ],
        skips=(
            # 存在 conj 和 torch 分发的问题，参见 https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),  # 标记为跳过
                "TestSchemaCheckModeOpInfo",  # 测试用例名称
                "test_schema_correctness",  # 测试模式正确性
            ),
        ),
    ),
    SpectralFuncInfo(
        "fft.rfft",  # 函数名称为 fft.rfft
        aten_name="fft_rfft",  # ATen 函数名称为 fft_rfft
        decomp_aten_name="_fft_r2c",  # 分解 ATen 函数名称为 _fft_r2c
        ref=np.fft.rfft,  # 参考实现来自 numpy 中的 np.fft.rfft 函数
        ndimensional=SpectralFuncType.OneD,  # 函数处理的数据维度为一维
        dtypes=all_types_and(torch.bool),  # 支持的数据类型包括所有类型和布尔类型
        # CUDA 只在 SM53 或更新的架构上支持 Half/ComplexHalf 精度的 FFT
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (not SM53OrLater) else (torch.half,))
        ),
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=1),  # 用于生成样本输入的函数，最小尺寸为1
        error_inputs_func=error_inputs_fft,  # 用于生成错误输入的函数为 error_inputs_fft
        # 支持前向自动求导
        supports_forward_ad=True,
        # 支持前向-反向梯度
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,  # 不检查批处理的梯度
        skips=(),  # 没有跳过项
        check_batched_gradgrad=False,  # 不检查批处理的梯度二阶导数
    ),
    SpectralFuncInfo(
        "fft.rfft2",  # 函数名称为 fft.rfft2
        aten_name="fft_rfft2",  # ATen 函数名称为 fft_rfft2
        decomp_aten_name="_fft_r2c",  # 分解 ATen 函数名称为 _fft_r2c
        ref=np.fft.rfft2,  # 参考实现来自 numpy 中的 np.fft.rfft2 函数
        ndimensional=SpectralFuncType.TwoD,  # 函数处理的数据维度为二维
        dtypes=all_types_and(torch.bool),  # 支持的数据类型包括所有类型和布尔类型
        # CUDA 只在 SM53 或更新的架构上支持 Half/ComplexHalf 精度的 FFT
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (not SM53OrLater) else (torch.half,))
        ),
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),  # 用于生成样本输入的函数，最小尺寸为 (1, 1)
        error_inputs_func=error_inputs_fftn,  # 用于生成错误输入的函数为 error_inputs_fftn
        # 支持前向自动求导
        supports_forward_ad=True,
        # 支持前向-反向梯度
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,  # 不检查批处理的梯度
        check_batched_gradgrad=False,  # 不检查批处理的梯度二阶导数
        decorators=[
            precisionOverride({torch.float: 1e-4}),  # 使用 precisionOverride 装饰器设置 torch.float 类型的精度为 1e-4
        ],
    ),
    SpectralFuncInfo(
        "fft.rfftn",  # 函数名称为 fft.rfftn
        aten_name="fft_rfftn",  # ATen 函数名称为 fft_rfftn
        decomp_aten_name="_fft_r2c",  # 分解 ATen 函数名称为 _fft_r2c
        ref=np.fft.rfftn,  # 参考实现来自 numpy 中的 np.fft.rfftn 函数
        ndimensional=SpectralFuncType.ND,  # 函数处理的数据维度为多维
        dtypes=all_types_and(torch.bool),  # 支持的数据类型包括所有类型和布尔类型
        # CUDA 只在 SM53 或更新的架构上支持 Half/ComplexHalf 精度的 FFT
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (not SM53OrLater) else (torch.half,))
        ),
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),  # 用于生成样本输入的函数，最小尺寸为 (1, 1)
        error_inputs_func=error_inputs_fftn,  # 用于生成错误输入的函数为 error_inputs_fftn
        # 支持前向自动求导
        supports_forward_ad=True,
        # 支持前向-反向梯度
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,  # 不检查批处理的梯度
        check_batched_gradgrad=False,  # 不检查批处理的梯度二阶导数
        decorators=[
            precisionOverride({torch.float: 1e-4}),  # 使用 precisionOverride 装饰器设置 torch.float 类型的精度为 1e-4
        ],
    ),
    SpectralFuncInfo(
        "fft.ifft",
        aten_name="fft_ifft",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.ifft,
        ndimensional=SpectralFuncType.OneD,
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=1),
        error_inputs_func=error_inputs_fft,
        # gradcheck_fast_mode 设置为 True，加快梯度检查速度
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # check_batched_forward_grad 设置为 False，禁用批处理前向梯度检查
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # 如果是 CUDA，支持 Half/ComplexHalf 精度的 FFT，要求 SM53 或更新的架构
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
    ),
    SpectralFuncInfo(
        "fft.ifft2",
        aten_name="fft_ifft2",
        decomp_aten_name="_fft_c2c",
        ref=np.fft.ifft2,
        ndimensional=SpectralFuncType.TwoD,
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),
        error_inputs_func=error_inputs_fftn,
        # gradcheck_fast_mode 设置为 True，加快梯度检查速度
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # check_batched_forward_grad 设置为 False，禁用批处理前向梯度检查
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),
        # 如果是 CUDA，支持 Half/ComplexHalf 精度的 FFT，要求 SM53 或更新的架构
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        decorators=[
            # 设置装饰器，精度覆盖为 {torch.float: 1e-4, torch.cfloat: 1e-4}
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    SpectralFuncInfo(
        "fft.ifftn",  # 函数名为 fft.ifftn
        aten_name="fft_ifftn",  # 在 PyTorch 中的命名为 fft_ifftn
        decomp_aten_name="_fft_c2c",  # 分解后的 PyTorch 函数名为 _fft_c2c
        ref=np.fft.ifftn,  # 参考实现来自 NumPy 的 ifftn 函数
        ndimensional=SpectralFuncType.ND,  # 处理的数据维度类型为多维
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),  # 部分应用的函数，用于生成最小大小的输入样本
        error_inputs_func=error_inputs_fftn,  # 用于错误输入的函数
        gradcheck_fast_mode=True,  # 使用快速模式进行梯度检查
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向和双向梯度
        check_batched_forward_grad=False,  # 不检查批处理前向梯度
        dtypes=all_types_and_complex_and(torch.bool),  # 支持所有类型和复杂类型以及布尔类型的数据类型
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),  # 如果是 CUDA，则还支持半精度和复杂半精度类型
        ),
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),  # 重写精度设置，浮点数和复杂浮点数的精度为 1e-4
                "TestFFT",  # 测试用例的名称
                "test_reference_nd",  # 参考的多维测试
            )
        ],
    ),
    SpectralFuncInfo(
        "fft.ihfft",  # 函数名为 fft.ihfft
        aten_name="fft_ihfft",  # 在 PyTorch 中的命名为 fft_ihfft
        decomp_aten_name="_fft_r2c",  # 分解后的 PyTorch 函数名为 _fft_r2c
        ref=np.fft.ihfft,  # 参考实现来自 NumPy 的 ihfft 函数
        ndimensional=SpectralFuncType.OneD,  # 处理的数据维度类型为一维
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),  # 部分应用的函数，用于生成最小大小的输入样本
        error_inputs_func=error_inputs_fft,  # 用于错误输入的函数
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向和双向梯度
        check_batched_forward_grad=False,  # 不检查批处理前向梯度
        dtypes=all_types_and(torch.bool),  # 支持所有类型和布尔类型的数据类型
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (not SM53OrLater) else (torch.half,))  # 如果是 CUDA，则还支持半精度类型
        ),
        skips=(),  # 不跳过任何测试
        check_batched_grad=False,  # 不检查批处理梯度
    ),
    SpectralFuncInfo(
        "fft.ihfft2",  # 定义一个名为 fft.ihfft2 的 SpectralFuncInfo 对象
        aten_name="fft_ihfft2",  # 设置属性 aten_name 为 "fft_ihfft2"
        decomp_aten_name="_fft_r2c",  # 设置属性 decomp_aten_name 为 "_fft_r2c"
        ref=scipy.fft.ihfftn if has_scipy_fft else None,  # 设置属性 ref 为 scipy.fft.ihfftn 或 None（根据条件判断）
        ndimensional=SpectralFuncType.TwoD,  # 设置属性 ndimensional 为 SpectralFuncType.TwoD
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),  # 设置属性 sample_inputs_func 为 sample_inputs_fft_with_min 的部分应用
        error_inputs_func=error_inputs_fftn,  # 设置属性 error_inputs_func 为 error_inputs_fftn
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,  # 设置属性 gradcheck_fast_mode 为 True
        supports_forward_ad=True,  # 设置属性 supports_forward_ad 为 True
        supports_fwgrad_bwgrad=True,  # 设置属性 supports_fwgrad_bwgrad 为 True
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,  # 设置属性 check_batched_forward_grad 为 False
        dtypes=all_types_and(torch.bool),  # 设置属性 dtypes 为 all_types_and(torch.bool)
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archs
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (not SM53OrLater) else (torch.half,))
        ),  # 设置属性 dtypesIfCUDA 根据条件设置不同的类型组合
        check_batched_grad=False,  # 设置属性 check_batched_grad 为 False
        check_batched_gradgrad=False,  # 设置属性 check_batched_gradgrad 为 False
        decorators=(  # 设置属性 decorators 为装饰器的元组
            # The values for attribute 'shape' do not match: torch.Size([5, 6, 5]) != torch.Size([5, 6, 6]).
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warning"),  # 设置装饰器信息
            DecorateInfo(  # 设置装饰器信息
                precisionOverride({torch.float: 2e-4}), "TestFFT", "test_reference_nd"
            ),
            # Mismatched elements!
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out"),  # 设置装饰器信息
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warnings"),  # 设置装饰器信息
        ),
    ),
    SpectralFuncInfo(
        "fft.ihfftn",  # 定义一个名为 fft.ihfftn 的 SpectralFuncInfo 对象
        aten_name="fft_ihfftn",  # 设置属性 aten_name 为 "fft_ihfftn"
        decomp_aten_name="_fft_r2c",  # 设置属性 decomp_aten_name 为 "_fft_r2c"
        ref=scipy.fft.ihfftn if has_scipy_fft else None,  # 设置属性 ref 为 scipy.fft.ihfftn 或 None（根据条件判断）
        ndimensional=SpectralFuncType.ND,  # 设置属性 ndimensional 为 SpectralFuncType.ND
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 1)),  # 设置属性 sample_inputs_func 为 sample_inputs_fft_with_min 的部分应用
        error_inputs_func=error_inputs_fftn,  # 设置属性 error_inputs_func 为 error_inputs_fftn
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,  # 设置属性 gradcheck_fast_mode 为 True
        supports_forward_ad=True,  # 设置属性 supports_forward_ad 为 True
        supports_fwgrad_bwgrad=True,  # 设置属性 supports_fwgrad_bwgrad 为 True
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,  # 设置属性 check_batched_forward_grad 为 False
        dtypes=all_types_and(torch.bool),  # 设置属性 dtypes 为 all_types_and(torch.bool)
        # CUDA supports Half/ComplexHalf Precision FFT only on SM53 or later archss
        dtypesIfCUDA=all_types_and(
            torch.bool, *(() if (not SM53OrLater) else (torch.half,))
        ),  # 设置属性 dtypesIfCUDA 根据条件设置不同的类型组合
        check_batched_grad=False,  # 设置属性 check_batched_grad 为 False
        check_batched_gradgrad=False,  # 设置属性 check_batched_gradgrad 为 False
        decorators=[  # 设置属性 decorators 为装饰器的列表
            # The values for attribute 'shape' do not match: torch.Size([5, 6, 5]) != torch.Size([5, 6, 6]).
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out_warning"),  # 设置装饰器信息
            # Mismatched elements!
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_out"),  # 设置装饰器信息
            DecorateInfo(  # 设置装饰器信息
                precisionOverride({torch.float: 2e-4}), "TestFFT", "test_reference_nd"
            ),
        ],
    ),
    SpectralFuncInfo(
        "fft.irfft",  # 定义一个 SpectralFuncInfo 对象，表示反向实部 FFT 函数 irfft
        aten_name="fft_irfft",  # 对应的 ATen 函数名
        decomp_aten_name="_fft_c2r",  # 对应的 ATen 函数名（用于复杂到实数的 FFT 变换）
        ref=np.fft.irfft,  # 参考的 NumPy 函数
        ndimensional=SpectralFuncType.OneD,  # 函数操作的维度，这里是一维
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 2)),  # 生成输入样本的函数，带有最小尺寸要求
        error_inputs_func=error_inputs_fft,  # 生成错误输入的函数
        # 设置快速梯度检查模式，详见 https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向和后向梯度
        # 不检查批处理的前向梯度，详见 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),  # 支持的数据类型，包括布尔类型
        # 如果是 CUDA，只支持 SM53 或更新的架构上的 Half/ComplexHalf 精度 FFT
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        check_batched_gradgrad=False,  # 不检查批处理的二阶梯度
    ),
    SpectralFuncInfo(
        "fft.irfft2",  # 定义一个 SpectralFuncInfo 对象，表示二维反向实部 FFT 函数 irfft2
        aten_name="fft_irfft2",  # 对应的 ATen 函数名
        decomp_aten_name="_fft_c2r",  # 对应的 ATen 函数名（用于复杂到实数的 FFT 变换）
        ref=np.fft.irfft2,  # 参考的 NumPy 函数
        ndimensional=SpectralFuncType.TwoD,  # 函数操作的维度，这里是二维
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 2)),  # 生成输入样本的函数，带有最小尺寸要求
        error_inputs_func=error_inputs_fftn,  # 生成错误输入的函数
        # 设置快速梯度检查模式，详见 https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,  # 支持前向自动求导
        supports_fwgrad_bwgrad=True,  # 支持前向和后向梯度
        # 不检查批处理的前向梯度，详见 https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        dtypes=all_types_and_complex_and(torch.bool),  # 支持的数据类型，包括布尔类型
        # 如果是 CUDA，只支持 SM53 或更新的架构上的 Half/ComplexHalf 精度 FFT
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),
        ),
        check_batched_gradgrad=False,  # 不检查批处理的二阶梯度
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),  # 设置精度修正
                "TestFFT",  # 测试用例的名称
                "test_reference_nd",  # 测试的参考点
            )
        ],  # 设置装饰器信息
    ),
    SpectralFuncInfo(
        "fft.irfftn",  # 定义了 FFT 的逆实数多维傅里叶变换函数
        aten_name="fft_irfftn",  # 在 PyTorch ATen 中的函数名
        decomp_aten_name="_fft_c2r",  # 用于分解的 ATen 函数名
        ref=np.fft.irfftn,  # 参考实现来自 NumPy 中的 irfftn 函数
        ndimensional=SpectralFuncType.ND,  # 函数支持的维度类型
        sample_inputs_func=partial(sample_inputs_fft_with_min, min_size=(1, 2)),  # 用于生成样本输入的函数
        error_inputs_func=error_inputs_fftn,  # 用于生成错误输入的函数
        gradcheck_fast_mode=True,  # 开启快速梯度检查模式
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向后向梯度传播
        check_batched_forward_grad=False,  # 不检查批处理的前向梯度
        dtypes=all_types_and_complex_and(torch.bool),  # 支持的数据类型，包括布尔型
        dtypesIfCUDA=all_types_and_complex_and(
            torch.bool,
            *(() if (not SM53OrLater) else (torch.half, torch.complex32)),  # 如果在 CUDA 下且是 SM53 架构及更高，则支持半精度和复数半精度
        ),
        check_batched_gradgrad=False,  # 不检查批处理的梯度二阶导数
        decorators=[  # 修饰器信息列表开始
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),  # 设置精度覆盖值
                "TestFFT",  # 修饰器名称
                "test_reference_nd",  # 修饰的测试函数名称
            )
        ],  # 修饰器信息列表结束
    ),  # SpectralFuncInfo 对象结束
    OpInfo(
        "fft.fftshift",  # 定义了 FFT 的移位函数
        dtypes=all_types_and_complex_and(  # 支持的数据类型，包括布尔型、bfloat16、半精度和复数半精度
            torch.bool, torch.bfloat16, torch.half, torch.chalf
        ),
        sample_inputs_func=sample_inputs_fftshift,  # 用于生成样本输入的函数
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向后向梯度传播
    ),  # OpInfo 对象结束
    OpInfo(
        "fft.ifftshift",  # 定义了 FFT 的反移位函数
        dtypes=all_types_and_complex_and(  # 支持的数据类型，包括布尔型、bfloat16、半精度和复数半精度
            torch.bool, torch.bfloat16, torch.half, torch.chalf
        ),
        sample_inputs_func=sample_inputs_fftshift,  # 用于生成样本输入的函数
        supports_out=False,  # 不支持输出参数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向后向梯度传播
    ),  # OpInfo 对象结束
# 定义一个名为 python_ref_db 的列表，其元素类型为 OpInfo 类型
python_ref_db: List[OpInfo] = [
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.fft 作为函数引用，
    # torch_opinfo_name 参数为 "fft.fft"
    SpectralFuncPythonRefInfo(
        "_refs.fft.fft",
        torch_opinfo_name="fft.fft",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.ifft 作为函数引用，
    # torch_opinfo_name 参数为 "fft.ifft"
    SpectralFuncPythonRefInfo(
        "_refs.fft.ifft",
        torch_opinfo_name="fft.ifft",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.rfft 作为函数引用，
    # torch_opinfo_name 参数为 "fft.rfft"
    SpectralFuncPythonRefInfo(
        "_refs.fft.rfft",
        torch_opinfo_name="fft.rfft",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.irfft 作为函数引用，
    # torch_opinfo_name 参数为 "fft.irfft"
    SpectralFuncPythonRefInfo(
        "_refs.fft.irfft",
        torch_opinfo_name="fft.irfft",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.hfft 作为函数引用，
    # torch_opinfo_name 参数为 "fft.hfft"
    SpectralFuncPythonRefInfo(
        "_refs.fft.hfft",
        torch_opinfo_name="fft.hfft",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.ihfft 作为函数引用，
    # torch_opinfo_name 参数为 "fft.ihfft"
    SpectralFuncPythonRefInfo(
        "_refs.fft.ihfft",
        torch_opinfo_name="fft.ihfft",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.fftn 作为函数引用，
    # torch_opinfo_name 参数为 "fft.fftn"，并添加一个 DecorateInfo 装饰器对象
    SpectralFuncPythonRefInfo(
        "_refs.fft.fftn",
        torch_opinfo_name="fft.fftn",
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.ifftn 作为函数引用，
    # torch_opinfo_name 参数为 "fft.ifftn"，并添加一个 DecorateInfo 装饰器对象
    SpectralFuncPythonRefInfo(
        "_refs.fft.ifftn",
        torch_opinfo_name="fft.ifftn",
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.rfftn 作为函数引用，
    # torch_opinfo_name 参数为 "fft.rfftn"
    SpectralFuncPythonRefInfo(
        "_refs.fft.rfftn",
        torch_opinfo_name="fft.rfftn",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.irfftn 作为函数引用，
    # torch_opinfo_name 参数为 "fft.irfftn"，并添加一个 DecorateInfo 装饰器对象
    SpectralFuncPythonRefInfo(
        "_refs.fft.irfftn",
        torch_opinfo_name="fft.irfftn",
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.hfftn 作为函数引用，
    # torch_opinfo_name 参数为 "fft.hfftn"，并添加一个 DecorateInfo 装饰器对象
    SpectralFuncPythonRefInfo(
        "_refs.fft.hfftn",
        torch_opinfo_name="fft.hfftn",
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.ihfftn 作为函数引用，
    # torch_opinfo_name 参数为 "fft.ihfftn"，并添加一个 DecorateInfo 装饰器对象
    SpectralFuncPythonRefInfo(
        "_refs.fft.ihfftn",
        torch_opinfo_name="fft.ihfftn",
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 2e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.fft2 作为函数引用，
    # torch_opinfo_name 参数为 "fft.fft2"
    SpectralFuncPythonRefInfo(
        "_refs.fft.fft2",
        torch_opinfo_name="fft.fft2",
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.ifft2 作为函数引用，
    # torch_opinfo_name 参数为 "fft.ifft2"，并添加一个 DecorateInfo 装饰器对象
    SpectralFuncPythonRefInfo(
        "_refs.fft.ifft2",
        torch_opinfo_name="fft.ifft2",
        decorators=[
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),
                "TestFFT",
                "test_reference_nd",
            )
        ],
    ),
    # 创建 SpectralFuncPythonRefInfo 对象，并指定 _refs.fft.rfft2 作为函数引用，
    # torch_opinfo_name 参数为 "fft.rfft2"
    SpectralFuncPythonRefInfo(
        "_refs.fft.rfft2",
        torch_opinfo_name="fft.rfft2",
    ),
    SpectralFuncPythonRefInfo(
        "_refs.fft.irfft2",  # 创建 SpectralFuncPythonRefInfo 对象，代表反 2 维实数快速傅里叶逆变换
        torch_opinfo_name="fft.irfft2",  # 设置 Torch 操作信息的名称为 'fft.irfft2'
        decorators=[  # 添加装饰器列表
            DecorateInfo(
                precisionOverride({torch.float: 1e-4, torch.cfloat: 1e-4}),  # 设置精度覆盖，浮点数和复数浮点数精度为 1e-4
                "TestFFT",  # 指定测试名称为 'TestFFT'
                "test_reference_nd",  # 指定测试函数为 'test_reference_nd'
            )
        ],
    ),
    SpectralFuncPythonRefInfo(
        "_refs.fft.hfft2",  # 创建 SpectralFuncPythonRefInfo 对象，代表 2 维埃尔米特快速傅里叶变换
        torch_opinfo_name="fft.hfft2",  # 设置 Torch 操作信息的名称为 'fft.hfft2'
        decorators=[  # 添加装饰器列表
            DecorateInfo(
                precisionOverride({torch.float: 2e-4, torch.cfloat: 2e-4}),  # 设置精度覆盖，浮点数和复数浮点数精度为 2e-4
                "TestFFT",  # 指定测试名称为 'TestFFT'
                "test_reference_nd",  # 指定测试函数为 'test_reference_nd'
            )
        ],
    ),
    SpectralFuncPythonRefInfo(
        "_refs.fft.ihfft2",  # 创建 SpectralFuncPythonRefInfo 对象，代表 2 维埃尔米特快速傅里叶逆变换
        torch_opinfo_name="fft.ihfft2",  # 设置 Torch 操作信息的名称为 'fft.ihfft2'
        decorators=[  # 添加装饰器列表
            DecorateInfo(
                precisionOverride({torch.float: 2e-4}),  # 设置浮点数精度覆盖为 2e-4
                "TestFFT",  # 指定测试名称为 'TestFFT'
                "test_reference_nd",  # 指定测试函数为 'test_reference_nd'
            )
        ],
    ),
    PythonRefInfo(
        "_refs.fft.fftshift",  # 创建 PythonRefInfo 对象，代表快速傅里叶频移
        op_db=op_db,  # 设置操作数据库为 op_db
        torch_opinfo_name="fft.fftshift",  # 设置 Torch 操作信息的名称为 'fft.fftshift'
    ),
    PythonRefInfo(
        "_refs.fft.ifftshift",  # 创建 PythonRefInfo 对象，代表快速傅里叶逆频移
        op_db=op_db,  # 设置操作数据库为 op_db
        torch_opinfo_name="fft.ifftshift",  # 设置 Torch 操作信息的名称为 'fft.ifftshift'
    ),
]



# 定义一个空列表，用于存储结果
result = []



# 遍历从1到10的每一个数字（不包括10）
for i in range(1, 10):



# 如果当前数字可以被2整除
if i % 2 == 0:



# 将当前数字的平方添加到结果列表中
result.append(i * i)



# 输出最终的结果列表
print(result)
```