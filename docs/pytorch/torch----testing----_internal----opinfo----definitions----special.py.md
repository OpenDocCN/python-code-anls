# `.\pytorch\torch\testing\_internal\opinfo\definitions\special.py`

```
# 忽略类型检查错误，通常用于静态类型检查工具
# 如 mypy，此处忽略特定的错误类型
# 详细信息：https://github.com/python/mypy/issues/6810
# 每个单元测试文件都需要导入 unittest 模块
import unittest
# 导入 functools 模块中的 partial 函数，用于创建 partial 函数应用
from functools import partial
# 导入 itertools 模块中的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product
# 导入 typing 模块中的 List 类型，用于声明列表类型的参数或变量
from typing import List

# 导入 numpy 库，使用 np 别名
import numpy as np

# 导入 torch 库
import torch
# 从 torch.testing 模块中导入 make_tensor 函数
from torch.testing import make_tensor
# 从 torch.testing._internal.common_device_type 模块中导入一系列函数和变量
from torch.testing._internal.common_device_type import (
    precisionOverride,
    tol,
    toleranceOverride,
)
# 从 torch.testing._internal.common_dtype 模块中导入 all_types_and 和 floating_types 函数
from torch.testing._internal.common_dtype import all_types_and, floating_types
# 从 torch.testing._internal.common_utils 模块中导入一系列常量和函数
from torch.testing._internal.common_utils import (
    TEST_SCIPY,
    TEST_WITH_ROCM,
    torch_to_numpy_dtype_dict,
)
# 从 torch.testing._internal.opinfo.core 模块中导入一系列类
from torch.testing._internal.opinfo.core import (
    BinaryUfuncInfo,
    DecorateInfo,
    L,
    NumericsFilter,
    OpInfo,
    S,
    SampleInput,
    UnaryUfuncInfo,
)
# 从 torch.testing._internal.opinfo.refs 模块中导入两个类
from torch.testing._internal.opinfo.refs import (
    ElementwiseBinaryPythonRefInfo,
    ElementwiseUnaryPythonRefInfo,
)
# 从 torch.testing._internal.opinfo.utils 模块中导入 np_unary_ufunc_integer_promotion_wrapper 函数
from torch.testing._internal.opinfo.utils import (
    np_unary_ufunc_integer_promotion_wrapper,
)

# 如果 TEST_SCIPY 为真，则导入 scipy.special 模块
if TEST_SCIPY:
    import scipy.special


# TODO: 当 `make_tensor` 函数支持 `exclude` 参数时，将 `i0e` 与 `sample_inputs_unary` 合并。
#       更多信息请参见：https://github.com/pytorch/pytorch/pull/56352#discussion_r633277617
# 定义一个生成器函数 sample_inputs_i0_i1，用于生成输入样本
def sample_inputs_i0_i1(op_info, device, dtype, requires_grad, **kwargs):
    # 根据是否需要梯度和操作是否为 torch.special.i0e 决定是否排除零值
    exclude_zero = requires_grad and op_info.op == torch.special.i0e
    # 创建 make_arg 函数，部分应用 make_tensor 函数，并传入参数
    make_arg = partial(
        make_tensor,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        exclude_zero=exclude_zero,
    )
    # 生成包含 make_arg 返回值的 SampleInput 对象，传入 (S,) 形状
    yield SampleInput(make_arg((S,)))
    # 生成不包含 exclude_zero 的 SampleInput 对象，传入空形状
    yield SampleInput(make_arg(()))

    # 如果需要梯度且不排除零值
    if requires_grad and not exclude_zero:
        # 特殊情况，用于梯度
        # 创建一个包含零值的 make_arg 返回值
        t = make_arg((S,))
        t[0] = 0
        yield SampleInput(t)


# 定义一个生成器函数 sample_inputs_polygamma，用于生成输入样本
def sample_inputs_polygamma(op_info, device, dtype, requires_grad, **kwargs):
    # 创建 make_arg 函数，部分应用 make_tensor 函数，并传入参数
    make_arg = partial(
        make_tensor,
        device=device,
        # TODO: 在修复 gh-106692 后消除低限制
        low=(1 if dtype in {torch.int32, torch.int64} else None),
        dtype=dtype,
        requires_grad=requires_grad,
    )
    # 定义张量形状和整数 ns
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    # 遍历 tensor_shapes 和 ns 的笛卡尔积
    for shape, n in product(tensor_shapes, ns):
        yield SampleInput(make_arg(shape), args=(n,))


# 定义 reference_polygamma 函数，用于计算 polygamma 函数的参考实现
def reference_polygamma(x, n):
    # 处理 scipy.special.polygamma 的奇怪行为，将输出类型转换为默认的 torch dtype 或保留 double
    result_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
    if x.dtype == np.double:
        result_dtype = np.double
    return scipy.special.polygamma(n, x).astype(result_dtype)


# 定义 sample_inputs_entr 函数，用于生成 entr 函数的输入样本
def sample_inputs_entr(op_info, device, dtype, requires_grad, **kwargs):
    # 获取 op_info 的 domain 元组中的低限制
    low, _ = op_info.domain

    # 如果需要梯度，则根据 op_info 的 _domain_eps 调整低限制
    if requires_grad:
        low = 0 + op_info._domain_eps
    # 使用 functools.partial 函数创建 make_arg 函数的部分应用，固定了部分参数
    make_arg = partial(
        make_tensor, dtype=dtype, device=device, low=low, requires_grad=requires_grad
    )
    # 生成并返回一个 SampleInput 对象，调用 make_arg 函数传入参数 (L,)，生成长度为 L 的张量
    yield SampleInput(make_arg((L,)))
    # 生成并返回一个 SampleInput 对象，调用 make_arg 函数传入参数 ()，生成一个标量张量
    yield SampleInput(make_arg(()))
def sample_inputs_erfcx(op_info, device, dtype, requires_grad, **kwargs):
    # 生成器函数，用于生成不同形状的样本输入
    for shape in ((L,), (1, 0, 3), ()):
        yield SampleInput(
            make_tensor(
                shape,
                device=device,
                dtype=dtype,
                low=-5,
                requires_grad=requires_grad,
            ),
        )


op_db: List[OpInfo] = [
    UnaryUfuncInfo(
        "special.i0e",
        aten_name="special_i0e",
        ref=scipy.special.i0e if TEST_SCIPY else None,
        decorators=(precisionOverride({torch.bfloat16: 3e-1, torch.float16: 3e-1}),),
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        backward_dtypes=floating_types(),
        sample_inputs_func=sample_inputs_i0_i1,  # 设置输入样本生成函数为 sample_inputs_i0_i1
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    UnaryUfuncInfo(
        "special.i1",
        aten_name="special_i1",
        ref=np_unary_ufunc_integer_promotion_wrapper(scipy.special.i1)
        if TEST_SCIPY
        else None,
        dtypes=all_types_and(torch.bool),
        dtypesIfCUDA=all_types_and(torch.bool),
        sample_inputs_func=sample_inputs_i0_i1,  # 设置输入样本生成函数为 sample_inputs_i0_i1
        decorators=(
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float32: tol(atol=1e-4, rtol=0),  # 设置浮点数类型的容忍度
                        torch.bool: tol(atol=1e-4, rtol=0),  # 设置布尔类型的容忍度
                    }
                )
            ),
        ),
        skips=(
            DecorateInfo(
                unittest.skip("Incorrect result!"),  # 跳过测试用例，给出跳过原因
                "TestUnaryUfuncs",
                "test_reference_numerics_large",
                dtypes=(torch.int8,),  # 仅适用于 torch.int8 类型的数据
            ),
        ),
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
    ),
    UnaryUfuncInfo(
        "special.i1e",
        aten_name="special_i1e",
        ref=scipy.special.i1e if TEST_SCIPY else None,
        dtypes=all_types_and(torch.bool),
        dtypesIfCUDA=all_types_and(torch.bool),
        sample_inputs_func=sample_inputs_i0_i1,  # 设置输入样本生成函数为 sample_inputs_i0_i1
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
    ),
    UnaryUfuncInfo(
        "special.ndtr",
        aten_name="special_ndtr",
        decorators=(precisionOverride({torch.bfloat16: 5e-3, torch.float16: 5e-4}),),
        ref=scipy.special.ndtr if TEST_SCIPY else None,
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # Dispatch stub: unsupported device typemeta
            DecorateInfo(
                unittest.expectedFailure,  # 预期该测试用例失败
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="meta",  # 指定设备类型为 "meta"
            ),
        ),
    ),
    # A separate OpInfo entry for special.polygamma is needed to reorder the arguments
    # for the alias. See the discussion here: https://github.com/pytorch/pytorch/pull/59691#discussion_r650261939
    UnaryUfuncInfo(
        "special.polygamma",  # 定义一个一元ufunc信息对象，处理特殊的polygamma函数
        op=lambda x, n, **kwargs: torch.special.polygamma(n, x, **kwargs),  # 操作为调用torch中的polygamma函数
        variant_test_name="special_polygamma_n_0",  # 变体测试名称为special_polygamma_n_0
        ref=reference_polygamma if TEST_SCIPY else None,  # 如果TEST_SCIPY为真，则使用reference_polygamma作为参考
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),  # 支持的数据类型包括所有类型和torch.bool, torch.half, torch.bfloat16
        dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),  # 在CUDA环境下支持的数据类型与CPU相同
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向和后向梯度计算
        sample_inputs_func=sample_inputs_polygamma,  # 用于生成polygamma函数的示例输入的函数
        skips=(
            # 下面是需要跳过的测试用例
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestNormalizeOperators",
                "test_normalize_operator_exhaustive",
            ),
        ),
        sample_kwargs=lambda device, dtype, input: ({"n": 0}, {"n": 0}),  # 示例输入的关键字参数
        reference_numerics_filter=NumericsFilter(
            condition=lambda x: (x < 0.1) & ((x - x.round()).abs() < 1e-4), safe_val=1  # 对polygamma函数的数值特性进行过滤，保证数值稳定性
        ),
    ),
    BinaryUfuncInfo(
        "special.xlog1py",  # 定义一个二元ufunc信息对象，处理特殊的xlog1py函数
        aten_name="special_xlog1py",  # 在ATen中的名称为special_xlog1py
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),  # 支持的数据类型包括所有类型和torch.bool, torch.half, torch.bfloat16
        promotes_int_to_float=True,  # 支持整数提升为浮点数
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向和后向梯度计算
        supports_one_python_scalar=True,  # 支持Python标量作为输入的一种情况
        rhs_make_tensor_kwargs=dict(low=-0.99),  # 用于生成右操作数张量的关键字参数，设置下限为-0.99
    ),
    BinaryUfuncInfo(
        "special.zeta",  # 定义一个二元ufunc信息对象，处理特殊的zeta函数
        aten_name="special_zeta",  # 在ATen中的名称为special_zeta
        dtypes=all_types_and(torch.bool),  # 支持的数据类型包括所有类型和torch.bool
        promotes_int_to_float=True,  # 支持整数提升为浮点数
        supports_autograd=False,  # 不支持自动微分
        supports_one_python_scalar=True,  # 支持Python标量作为输入的一种情况
        skips=(
            # 下面是需要跳过的测试用例
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    # TODO: FIXME
    # OpInfo entry to verify the gradient formula of `other`/`q`
    # BinaryUfuncInfo('special.zeta',
    #                 op=lambda q, x, **kwargs: torch.special.zeta(x, q, **kwargs),
    #                 aten_name='special_zeta',
    #                 variant_test_name='grad',
    #                 dtypes=all_types_and(torch.bool),
    #                 promotes_int_to_float=True,
    #                 supports_autograd=True,
    #                 supports_rhs_python_scalar=False,
    #                 decorators=[
    #                     # Derivative wrt first tensor not implemented
    #                     DecorateInfo(unittest.expectedFailure, "TestCommon",
    #                                  "test_floating_inputs_are_differentiable")
    #                 ],
    #                 skips=(
    #                     # Lambda doesn't work in JIT test
    # 创建一个UnaryUfuncInfo对象，用于描述一元ufunc函数的测试信息
    UnaryUfuncInfo(
        "special.entr",  # 函数名为'special.entr'
        ref=scipy.special.entr if TEST_SCIPY else None,  # 如果TEST_SCIPY为真，则引用scipy.special.entr，否则为None
        aten_name="special_entr",  # ATen中的函数名为'special_entr'
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向后向梯度
        decorators=(precisionOverride({torch.float16: 1e-1, torch.bfloat16: 1e-1}),),  # 设置精度修饰器，适用于torch.float16和torch.bfloat16
        dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),  # 数据类型包括所有类型以及torch.bool、torch.half、torch.bfloat16
        skips=(  # 设置跳过信息的元组
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用unittest.skip函数创建一个跳过测试的装饰器
                "TestUnaryUfuncs",  # 测试类名称为'TestUnaryUfuncs'
                "test_reference_numerics_large",  # 测试方法名称为'test_reference_numerics_large'
                dtypes=[torch.bfloat16, torch.float16],  # 要跳过的数据类型为torch.bfloat16和torch.float16
            ),
        ),
        supports_inplace_autograd=False,  # 不支持原地自动微分
        sample_inputs_func=sample_inputs_entr,  # 样本输入函数为sample_inputs_entr
    ),

    # 创建一个UnaryUfuncInfo对象，用于描述一元ufunc函数的测试信息
    UnaryUfuncInfo(
        "special.ndtri",  # 函数名为'special.ndtri'
        ref=scipy.special.ndtri if TEST_SCIPY else None,  # 如果TEST_SCIPY为真，则引用scipy.special.ndtri，否则为None
        domain=(0, 1),  # 函数定义域为(0, 1)
        aten_name="special_ndtri",  # ATen中的函数名为'special_ndtri'
        dtypes=all_types_and(torch.bool),  # 数据类型包括所有类型以及torch.bool
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向后向梯度
    ),

    # 创建一个UnaryUfuncInfo对象，用于描述一元ufunc函数的测试信息
    UnaryUfuncInfo(
        "special.log_ndtr",  # 函数名为'special.log_ndtr'
        aten_name="special_log_ndtr",  # ATen中的函数名为'special_log_ndtr'
        ref=scipy.special.log_ndtr if TEST_SCIPY else None,  # 如果TEST_SCIPY为真，则引用scipy.special.log_ndtr，否则为None
        dtypes=all_types_and(torch.bool),  # 数据类型包括所有类型以及torch.bool
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向后向梯度
    ),

    # 创建一个UnaryUfuncInfo对象，用于描述一元ufunc函数的测试信息
    UnaryUfuncInfo(
        "special.erfcx",  # 函数名为'special.erfcx'
        ref=scipy.special.erfcx if TEST_SCIPY else None,  # 如果TEST_SCIPY为真，则引用scipy.special.erfcx，否则为None
        aten_name="special_erfcx",  # ATen中的函数名为'special_erfcx'
        decorators=(  # 设置修饰器元组
            toleranceOverride(
                {
                    torch.float32: tol(atol=0, rtol=4e-6),  # 对于torch.float32，设置tolerance为tol(atol=0, rtol=4e-6)
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 数据类型包括所有类型以及torch.bool
        supports_forward_ad=True,  # 支持前向自动微分
        supports_fwgrad_bwgrad=True,  # 支持前向后向梯度
        sample_inputs_func=sample_inputs_erfcx,  # 样本输入函数为sample_inputs_erfcx
    ),

    # 创建一个UnaryUfuncInfo对象，用于描述一元ufunc函数的测试信息
    UnaryUfuncInfo(
        "special.airy_ai",  # 函数名为'special.airy_ai'
        decorators=(  # 设置修饰器元组
            precisionOverride(
                {
                    torch.float32: 1e-03,  # 对于torch.float32，设置精度为1e-03
                    torch.float64: 1e-05,  # 对于torch.float64，设置精度为1e-05
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 数据类型包括所有类型以及torch.bool
        ref=lambda x: scipy.special.airy(x)[0] if TEST_SCIPY else None,  # 引用函数为lambda x: scipy.special.airy(x)[0]（如果TEST_SCIPY为真）
        skips=(  # 设置跳过信息的元组
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用unittest.skip函数创建一个跳过测试的装饰器
                "TestUnaryUfuncs",  # 测试类名称为'TestUnaryUfuncs'
                "test_reference_numerics_large",  # 测试方法名称为'test_reference_numerics_large'
            ),
        ),
        supports_autograd=False,  # 不支持自动微分
    ),

    # 创建一个UnaryUfuncInfo对象，用于描述一元ufunc函数的测试信息
    UnaryUfuncInfo(
        "special.bessel_j0",  # 函数名为'special.bessel_j0'
        decorators=(  # 设置修饰器元组
            precisionOverride(
                {
                    torch.float32: 1e-04,  # 对于torch.float32，设置精度为1e-04
                    torch.float64: 1e-05,  # 对于torch.float64，设置精度为1e-05
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 数据类型包括所有类型以及torch.bool
        ref=scipy.special.j0 if TEST_SCIPY else None,  # 如果TEST_SCIPY为真，则引用scipy.special.j0，否则为None
        supports_autograd=False,  # 不支持自动微分
    ),
    UnaryUfuncInfo(
        "special.bessel_j1",  # 定义一元ufunc函数信息，对应于scipy.special.bessel_j1
        decorators=(  # 使用修饰器来设置精度覆盖，适用于torch.float32和torch.float64类型
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 定义接受的数据类型，包括所有类型和torch.bool
        ref=scipy.special.j1 if TEST_SCIPY else None,  # 设置参考实现为scipy.special.j1（如果测试为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.bessel_y0",  # 定义一元ufunc函数信息，对应于scipy.special.bessel_y0
        decorators=(  # 使用修饰器来设置精度覆盖，适用于torch.float32和torch.float64类型
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 定义接受的数据类型，包括所有类型和torch.bool
        ref=scipy.special.y0 if TEST_SCIPY else None,  # 设置参考实现为scipy.special.y0（如果测试为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.bessel_y1",  # 定义一元ufunc函数信息，对应于scipy.special.bessel_y1
        decorators=(  # 使用修饰器来设置精度覆盖，适用于torch.float32和torch.float64类型
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 定义接受的数据类型，包括所有类型和torch.bool
        ref=scipy.special.y1 if TEST_SCIPY else None,  # 设置参考实现为scipy.special.y1（如果测试为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_t",  # 定义二元ufunc函数信息，对应于scipy.special.chebyshev_polynomial_t
        dtypes=all_types_and(torch.bool),  # 定义接受的数据类型，包括所有类型和torch.bool
        promotes_int_to_float=True,  # 推广整数转换为浮点数
        skips=(  # 定义跳过的测试装饰器信息
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,  # 支持单一Python标量参数
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_u",  # 定义二元ufunc函数信息，对应于scipy.special.chebyshev_polynomial_u
        dtypes=all_types_and(torch.bool),  # 定义接受的数据类型，包括所有类型和torch.bool
        promotes_int_to_float=True,  # 推广整数转换为浮点数
        skips=(  # 定义跳过的测试装饰器信息
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,  # 支持单一Python标量参数
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_v",  # 定义二元ufunc函数信息，对应于scipy.special.chebyshev_polynomial_v
        dtypes=all_types_and(torch.bool),  # 定义接受的数据类型，包括所有类型和torch.bool
        promotes_int_to_float=True,  # 推广整数转换为浮点数
        skips=(  # 定义跳过的测试装饰器信息
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
        ),
        supports_one_python_scalar=True,  # 支持单一Python标量参数
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.chebyshev_polynomial_w",  # 定义一个二元ufunc的信息，用于计算切比雪夫多项式
        dtypes=all_types_and(torch.bool),  # 支持所有数据类型以及布尔类型
        promotes_int_to_float=True,  # 自动将整数提升为浮点数
        skips=(
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )  # 标记为跳过测试，因为测试时间过长
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),  # 标记为跳过CUDA融合操作的测试
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),  # 标记为跳过NNC操作的测试
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.hermite_polynomial_h",  # 定义一个二元ufunc的信息，用于计算厄米多项式
        dtypes=all_types_and(torch.bool),  # 支持所有数据类型以及布尔类型
        promotes_int_to_float=True,  # 自动将整数提升为浮点数
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),  # 标记为跳过CUDA融合操作的测试
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),  # 标记为跳过NNC操作的测试
            # 最大的绝对差异：无穷大
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),  # 标记为预期的测试失败，用于CPU比较测试
            DecorateInfo(unittest.skip("Hangs on ROCm 6.1"), active_if=TEST_WITH_ROCM),  # 标记为跳过测试，因为在ROCm 6.1上卡住
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.hermite_polynomial_he",  # 定义一个二元ufunc的信息，用于计算厄米多项式
        dtypes=all_types_and(torch.bool),  # 支持所有数据类型以及布尔类型
        promotes_int_to_float=True,  # 自动将整数提升为浮点数
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),  # 标记为跳过CUDA融合操作的测试
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),  # 标记为跳过NNC操作的测试
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),  # 标记为跳过测试，因为测试时间过长
                "TestCommon",  # 测试模块名称
                "test_compare_cpu",  # 测试方法名称
            ),
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.laguerre_polynomial_l",  # 定义一个二元ufunc的信息，用于计算拉盖尔多项式
        dtypes=all_types_and(torch.bool),  # 支持所有数据类型以及布尔类型
        promotes_int_to_float=True,  # 自动将整数提升为浮点数
        skips=(
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),  # 标记为跳过CUDA融合操作的测试
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),  # 标记为跳过NNC操作的测试
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),  # 标记为跳过测试，因为测试时间过长
                "TestCommon",  # 测试模块名称
                "test_compare_cpu",  # 测试方法名称
            ),
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.legendre_polynomial_p",  # 定义了二元通用函数的信息，用于计算Legendre多项式
        dtypes=all_types_and(torch.bool),  # 适用的数据类型包括所有类型和布尔类型
        promotes_int_to_float=True,  # 支持将整数提升为浮点数
        skips=(  # 跳过以下测试的装饰信息列表
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )  # 使用unittest的skip装饰器跳过测试，原因是测试时间过长
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),  # 跳过CUDA相关操作的测试
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),  # 跳过NNC操作的测试
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),  # 再次说明跳过测试的原因
                "TestCommon",
                "test_compare_cpu",  # 跳过测试通用函数中CPU比较测试
            ),
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量作为参数
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_i0",  # 定义了一元通用函数的信息，用于计算修改的Bessel函数I0
        decorators=(  # 装饰器列表，用于设置精度覆盖
            precisionOverride(
                {
                    torch.float32: 1e-03,  # 对于float32类型，设置精度为1e-03
                    torch.float64: 1e-05,  # 对于float64类型，设置精度为1e-05
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 适用的数据类型包括所有类型和布尔类型
        ref=scipy.special.i0 if TEST_SCIPY else None,  # 参考实现为SciPy中的I0函数（如果测试标志TEST_SCIPY为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_i1",  # 定义了一元通用函数的信息，用于计算修改的Bessel函数I1
        decorators=(  # 装饰器列表，用于设置精度覆盖
            precisionOverride(
                {
                    torch.float32: 1e-03,  # 对于float32类型，设置精度为1e-03
                    torch.float64: 1e-05,  # 对于float64类型，设置精度为1e-05
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 适用的数据类型包括所有类型和布尔类型
        ref=scipy.special.i1 if TEST_SCIPY else None,  # 参考实现为SciPy中的I1函数（如果测试标志TEST_SCIPY为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_k0",  # 定义了一元通用函数的信息，用于计算修改的Bessel函数K0
        decorators=(  # 装饰器列表，用于设置精度覆盖
            precisionOverride(
                {
                    torch.float32: 1e-03,  # 对于float32类型，设置精度为1e-03
                    torch.float64: 1e-05,  # 对于float64类型，设置精度为1e-05
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 适用的数据类型包括所有类型和布尔类型
        ref=scipy.special.k0 if TEST_SCIPY else None,  # 参考实现为SciPy中的K0函数（如果测试标志TEST_SCIPY为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.modified_bessel_k1",  # 定义了一元通用函数的信息，用于计算修改的Bessel函数K1
        decorators=(  # 装饰器列表，用于设置精度覆盖
            precisionOverride(
                {
                    torch.float32: 1e-03,  # 对于float32类型，设置精度为1e-03
                    torch.float64: 1e-05,  # 对于float64类型，设置精度为1e-05
                },
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 适用的数据类型包括所有类型和布尔类型
        ref=scipy.special.k1 if TEST_SCIPY else None,  # 参考实现为SciPy中的K1函数（如果测试标志TEST_SCIPY为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.scaled_modified_bessel_k0",  # 定义了一元通用函数的信息，用于计算缩放修改的Bessel函数K0
        decorators=(  # 装饰器列表，用于设置容差覆盖
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),  # 对于float32类型，设置容差
                    torch.float64: tol(atol=1e-05, rtol=1e-03),  # 对于float64类型，设置容差
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 适用的数据类型包括所有类型和布尔类型
        ref=scipy.special.k0e if TEST_SCIPY else None,  # 参考实现为SciPy中的K0e函数（如果测试标志TEST_SCIPY为真）
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.scaled_modified_bessel_k1",  # 定义一元ufunc函数名为special.scaled_modified_bessel_k1
        decorators=(  # 设置装饰器
            toleranceOverride(  # 调用toleranceOverride装饰器
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),  # 对于torch.float32类型设置容差为1e-03和1e-03
                    torch.float64: tol(atol=1e-05, rtol=1e-03),  # 对于torch.float64类型设置容差为1e-05和1e-03
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 设置函数适用的数据类型为所有类型和torch.bool类型
        ref=scipy.special.k1e if TEST_SCIPY else None,  # 如果TEST_SCIPY为真，则引用scipy.special.k1e，否则为None
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_t",  # 定义二元ufunc函数名为special.shifted_chebyshev_polynomial_t
        dtypes=all_types_and(torch.bool),  # 设置函数适用的数据类型为所有类型和torch.bool类型
        promotes_int_to_float=True,  # 提升整数到浮点数
        skips=(  # 设置跳过的测试
            DecorateInfo(
                unittest.skip(  # 使用unittest.skip函数，说明跳过原因
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),  # 跳过TestCudaFuserOpInfo测试
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),  # 跳过TestNNCOpInfo测试
            DecorateInfo(
                unittest.skip(  # 使用unittest.skip函数，说明跳过原因
                    "testing takes an unreasonably long time, #79528"
                ),
                "TestCommon",
                "test_compare_cpu",  # 在TestCommon类中跳过test_compare_cpu测试
            ),
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量参数
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_u",  # 定义二元ufunc函数名为special.shifted_chebyshev_polynomial_u
        dtypes=all_types_and(torch.bool),  # 设置函数适用的数据类型为所有类型和torch.bool类型
        promotes_int_to_float=True,  # 提升整数到浮点数
        skips=(  # 设置跳过的测试，与上述类似
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip(
                    "testing takes an unreasonably long time, #79528"
                ),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量参数
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_v",  # 定义二元ufunc函数名为special.shifted_chebyshev_polynomial_v
        dtypes=all_types_and(torch.bool),  # 设置函数适用的数据类型为所有类型和torch.bool类型
        promotes_int_to_float=True,  # 提升整数到浮点数
        skips=(  # 设置跳过的测试，与上述类似
            DecorateInfo(
                unittest.skip(
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),
            DecorateInfo(
                unittest.skip(
                    "testing takes an unreasonably long time, #79528"
                ),
                "TestCommon",
                "test_compare_cpu",
            ),
        ),
        supports_one_python_scalar=True,  # 支持单个Python标量参数
        supports_autograd=False,  # 不支持自动求导
    ),
    BinaryUfuncInfo(
        "special.shifted_chebyshev_polynomial_w",  # 定义一个二元ufunc的信息对象，表示特定的函数
        dtypes=all_types_and(torch.bool),  # 支持所有数据类型和布尔类型
        promotes_int_to_float=True,  # 支持整数提升为浮点数
        skips=(  # 跳过以下测试装饰信息
            DecorateInfo(
                unittest.skip(  # 使用unittest模块跳过测试
                    "Skipping - testing takes an unreasonably long time, #79528"
                )
            ),
            DecorateInfo(unittest.skip("Skipped!"), "TestCudaFuserOpInfo"),  # 跳过名为TestCudaFuserOpInfo的测试
            DecorateInfo(unittest.skip("Skipped!"), "TestNNCOpInfo"),  # 跳过名为TestNNCOpInfo的测试
            DecorateInfo(
                unittest.skip("testing takes an unreasonably long time, #79528"),  # 再次跳过长时间测试
                "TestCommon",  # 跳过名为TestCommon的测试
                "test_compare_cpu",  # 跳过名为test_compare_cpu的测试
            ),
        ),
        supports_one_python_scalar=True,  # 支持Python标量
        supports_autograd=False,  # 不支持自动求导
    ),
    UnaryUfuncInfo(
        "special.spherical_bessel_j0",  # 定义一个一元ufunc的信息对象，表示特定的函数
        decorators=(  # 装饰器列表
            toleranceOverride(  # 覆盖默认容差值
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),  # 对于float32类型设置容差
                    torch.float64: tol(atol=1e-05, rtol=1e-03),  # 对于float64类型设置容差
                }
            ),
        ),
        dtypes=all_types_and(torch.bool),  # 支持所有数据类型和布尔类型
        ref=lambda x: scipy.special.spherical_jn(0, x) if TEST_SCIPY else None,  # 参考实现函数，如果TEST_SCIPY为True，则使用SciPy的函数
        supports_autograd=False,  # 不支持自动求导
    ),
python_ref_db: List[OpInfo] = [
    # 定义一个列表 python_ref_db，存储 OpInfo 类型的对象
    #
    # Elementwise Unary Special OpInfos
    #
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.bessel_j0",  # OpInfo 对象的唯一标识符
        torch_opinfo_name="special.bessel_j0",  # Torch 操作的名称
        op_db=op_db,  # OpInfo 对象的数据库
        decorators=(  # 装饰器列表，用于设置精度覆盖
            precisionOverride(
                {
                    torch.float32: 1e-04,  # 对于 torch.float32 类型的精度覆盖
                    torch.float64: 1e-05,  # 对于 torch.float64 类型的精度覆盖
                },
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.bessel_j1",
        torch_opinfo_name="special.bessel_j1",
        op_db=op_db,
        decorators=(
            precisionOverride(
                {
                    torch.float32: 1e-04,
                    torch.float64: 1e-05,
                },
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.entr",
        torch_opinfo_name="special.entr",
        op_db=op_db,
        decorators=(  # 装饰器列表，用于设置精度覆盖和跳过测试
            precisionOverride({torch.float16: 1e-1, torch.bfloat16: 1e-1}),
        ),
        skips=(  # 跳过测试的信息列表
            DecorateInfo(
                unittest.skip("Skipped!"),  # 使用 unittest.skip 跳过测试
                "TestUnaryUfuncs",  # 测试类名称
                "test_reference_numerics_large",  # 测试方法名称
                dtypes=[torch.bfloat16, torch.float16],  # 需要跳过的数据类型列表
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.erfcx",
        torch_opinfo_name="special.erfcx",
        op_db=op_db,
        decorators=(  # 装饰器列表，用于设置容差覆盖
            toleranceOverride(
                {
                    torch.float32: tol(atol=0, rtol=4e-6),  # 对于 torch.float32 类型的容差设置
                }
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.i0e",
        torch_opinfo_name="special.i0e",
        op_db=op_db,
        decorators=(  # 装饰器列表，用于设置精度覆盖
            precisionOverride({torch.bfloat16: 3e-1, torch.float16: 3e-1}),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.i1",
        torch_opinfo_name="special.i1",
        op_db=op_db,
        decorators=(  # 装饰器列表，用于设置容差覆盖
            DecorateInfo(
                toleranceOverride(
                    {
                        torch.float32: tol(atol=1e-4, rtol=0),  # 对于 torch.float32 类型的容差设置
                        torch.bool: tol(atol=1e-4, rtol=0),    # 对于 torch.bool 类型的容差设置
                    }
                )
            ),
        ),
        skips=(  # 跳过测试的信息列表
            DecorateInfo(
                unittest.skip("Incorrect result!"),  # 使用 unittest.skip 跳过测试
                "TestUnaryUfuncs",  # 测试类名称
                "test_reference_numerics_large",  # 测试方法名称
                dtypes=(torch.int8,),  # 需要跳过的数据类型列表
            ),
        ),
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.i1e",
        torch_opinfo_name="special.i1e",
        op_db=op_db,
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.log_ndtr",
        torch_opinfo_name="special.log_ndtr",
        op_db=op_db,
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.ndtr",
        torch_opinfo_name="special.ndtr",
        op_db=op_db,
    ),
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.ndtri",
        torch_opinfo_name="special.ndtri",
        op_db=op_db,
    ),
    # 创建 ElementwiseUnaryPythonRefInfo 对象，指定特定的函数 "_refs.special.spherical_bessel_j0"
    # 并关联到 Torch 操作 "special.spherical_bessel_j0"
    ElementwiseUnaryPythonRefInfo(
        "_refs.special.spherical_bessel_j0",
        torch_opinfo_name="special.spherical_bessel_j0",
        op_db=op_db,
        decorators=(
            # 设置特定数据类型的容差覆盖，指定浮点数精度和容差
            toleranceOverride(
                {
                    torch.float32: tol(atol=1e-03, rtol=1e-03),
                    torch.float64: tol(atol=1e-05, rtol=1e-03),
                }
            ),
        ),
    ),
    
    #
    # Elementwise Binary Special OpInfos
    #
    
    # 创建 ElementwiseBinaryPythonRefInfo 对象，指定特定的函数 "_refs.special.zeta"
    # 并关联到 Torch 操作 "special.zeta"
    ElementwiseBinaryPythonRefInfo(
        "_refs.special.zeta",
        torch_opinfo_name="special.zeta",
        supports_one_python_scalar=True,
        op_db=op_db,
        skips=(
            # 标记为跳过的测试用例，这些测试用例会在特定条件下预期失败
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
]



# 访问列表或字典中的某个元素，此处是代码片段的结束标记
```