# `.\pytorch\torch\_inductor\kernel\mm.py`

```
# 引入mypy: allow-untyped-defs，允许在未类型化定义中使用mypy
import functools  # 导入functools模块，提供对高阶函数的支持
import logging  # 导入logging模块，用于记录日志信息
from typing import Any, Dict, List, Optional  # 导入类型提示，用于类型检查

import torch  # 导入PyTorch库
from torch._inductor.codegen.cpp_gemm_template import CppPackedGemmTemplate  # 导入C++ GEMM模板
from torch._inductor.virtualized import V  # 导入虚拟化相关模块
from .. import config as inductor_config  # 导入上级目录的config模块作为inductor_config
from ..codegen.common import BackendFeature  # 导入codegen.common模块的BackendFeature
from ..codegen.cuda.gemm_template import CUTLASSGemmTemplate  # 导入CUDA GEMM模板
from ..codegen.rocm.ck_universal_gemm_template import CKGemmTemplate  # 导入ROCM通用GEMM模板
from ..codegen.wrapper import WrapperCodeGen  # 导入wrapper模块的WrapperCodeGen
from ..ir import FlexibleLayout  # 导入ir模块的FlexibleLayout
from ..lowering import register_lowering  # 导入lowering模块的register_lowering
from ..select_algorithm import (  # 导入select_algorithm模块的以下内容：
    autotune_select_algorithm,  # 自动调优选择算法
    ExternKernelChoice,  # 外部内核选择
    NoValidChoicesError,  # 无有效选择错误
    TritonTemplate,  # Triton模板
)
from ..utils import (  # 导入utils模块的以下内容：
    get_gpu_shared_memory,  # 获取GPU共享内存函数
    use_aten_gemm_kernels,  # 使用ATen GEMM内核函数
    use_ck_template,  # 使用CK模板函数
    use_cpp_packed_gemm_template,  # 使用C++打包GEMM模板函数
    use_cutlass_template,  # 使用Cutlass模板函数
    use_max_autotune,  # 使用最大自动调优函数
    use_triton_template,  # 使用Triton模板函数
)
from .mm_common import (  # 从当前目录导入mm_common模块的以下内容：
    addmm_epilogue,  # addmm函数的尾声
    int8_mm_configs,  # int8矩阵乘法配置
    mixed_mm_configs,  # 混合矩阵乘法配置
    mm_args,  # 矩阵乘法参数
    mm_configs,  # 矩阵乘法配置
    mm_grid,  # 矩阵乘法网格
    mm_options,  # 矩阵乘法选项
    triton_config,  # Triton配置
)

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
aten = torch.ops.aten  # 获取PyTorch的ATen操作对象

mm_template = TritonTemplate(
    name="mm",  # 设置模板名称为"mm"
    grid=mm_grid,  # 使用导入的mm_grid作为网格参数
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}  # 从A张量获取维度0的大小，并赋值给M
    N = {{size("B", 1)}}  # 从B张量获取维度1的大小，并赋值给N
    K = {{size("A", 1)}}  # 从A张量获取维度1的大小，并赋值给K
    if M * N == 0:  # 如果M和N的乘积为0
        # 由于输入大小为零，提前退出
        return  # 返回空值
    stride_am = {{stride("A", 0)}}  # 从A张量获取维度0的步长，并赋值给stride_am
    stride_ak = {{stride("A", 1)}}  # 从A张量获取维度1的步长，并赋值给stride_ak
    stride_bk = {{stride("B", 0)}}  # 从B张量获取维度0的步长，并赋值给stride_bk
    stride_bn = {{stride("B", 1)}}  # 从B张量获取维度1的步长，并赋值给stride_bn

    # 基于triton.ops.matmul操作
    pid = tl.program_id(0)  # 获取程序ID为0的线程ID
    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # 计算M维度的网格数
    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # 计算N维度的网格数

    # 重新排序程序ID以获得更好的L2缓存性能
    width = GROUP_M * grid_n  # 定义宽度为GROUP_M乘以grid_n
    group_id = pid // width  # 计算分组ID
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)  # 计算分组大小
    pid_m = group_id * GROUP_M + (pid % group_size)  # 计算M维度的线程ID
    pid_n = (pid % width) // (group_size)  # 计算N维度的线程ID

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # 定义rm为pid_m乘以BLOCK_M加上BLOCK_M范围内的数组
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 定义rn为pid_n乘以BLOCK_N加上BLOCK_N范围内的数组
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)  # 计算连续内存的最大值
    else:
        ram = rm % M  # 否则，ram为rm对M取余
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)  # 计算连续内存的最大值
    else:
        rbn = rn % N  # 否则，rbn为rn对N取余
    rk = tl.arange(0, BLOCK_K)  # 定义rk为0到BLOCK_K范围的数组
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)  # 计算A张量
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)  # 计算B张量

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)  # 创建全零张量acc，形状为(BLOCK_M, BLOCK_N)，数据类型为ACC_TYPE
"""
)
    # 使用逆序迭代从 K 到 1，步长为 -BLOCK_K
    for k in range(K, 0, -BLOCK_K):
        # 如果 EVEN_K 为真，则加载张量 A 和 B
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            # 否则，加载张量 A 和 B，只保留 rk 与 k 比较为真的部分，其他置为 0
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        # 如果 B_PROLOGUE_CAST_TYPE 不为空，则将张量 b 转换为指定类型
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        # 计算张量 a 和 b 的乘积，并累加到 acc 中，允许 tf32 类型计算
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        # 更新张量 A 和 B 的位置指针
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # 重新生成 rm 和 rn 以节省寄存器
    # rm 包含 pid_m * BLOCK_M 到 pid_m * BLOCK_M + BLOCK_M 的范围
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # rn 包含 pid_n * BLOCK_N 到 pid_n * BLOCK_N + BLOCK_N 的范围
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # 创建索引张量 idx_m 和 idx_n，用于后续操作
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    # 创建布尔掩码，用于检查 idx_m 和 idx_n 是否在 M 和 N 的有效范围内
    mask = (idx_m < M) & (idx_n < N)

    # 生成一个后缀，用于存储输出
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
# 定义一个类 ExternKernelChoice，用于选择外部内核实现，并指定相应的操作名称和重载选项
aten_mm = ExternKernelChoice(torch.mm, "at::mm_out")

# 定义一个类 ExternKernelChoice，用于选择外部内核实现，并指定相应的操作名称、重载选项为 torch.addmm 的默认值
aten_addmm = ExternKernelChoice(
    torch.addmm, "at::addmm_out", op_overload=aten.addmm.default
)

# 定义一个类 ExternKernelChoice，用于选择外部内核实现，并指定相应的操作名称为 at::_int_mm
aten__int_mm = ExternKernelChoice(torch._int_mm, "at::_int_mm")

# 定义一个函数 _is_int8_mat，判断输入的矩阵是否为 torch.int8 或 torch.uint8 类型
def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)

# 定义一个函数 bias_addmm，用于对输入进行偏置加权矩阵乘法操作，根据条件调用不同的 torch.addmm 实现
def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if inp.stride(0) == 0 or inp.size(0) == 1:
        # 若 inp 的 stride(0) 为 0 或 size(0) 为 1，则调用 torch.addmm 的优化版本进行计算
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    # 否则调用普通版本的 torch.addmm 进行计算
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)

# 定义一个类 ExternKernelChoice，用于选择外部内核实现，函数为 bias_addmm
aten_bias_addmm = ExternKernelChoice(bias_addmm, None)

# 注册一个函数 lowering，用于降低 aten.mm 操作的类型提升
@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    # 解析 mm_args 返回的参数，设置矩阵的维度和布局
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

    # 用于 ATen 矩阵乘法的布局选择
    aten_layout = layout
    if not use_max_autotune():
        # 如果不使用最大自动调整，则使用灵活的布局
        aten_layout = FlexibleLayout(
            device=layout.device, dtype=layout.dtype, size=layout.size
        )

    # 初始化选择列表
    choices = (
        [aten_mm.bind((mat1, mat2), aten_layout)] if use_aten_gemm_kernels() else []
    )

    # 检查是否是静态问题，并确定是否使用 Triton 模板
    static_shape, is_nonzero = _is_static_problem([mat1, mat2], layout)
    if is_nonzero and use_triton_template(layout):
        for config in mm_configs(m, n, k):
            # 将 Triton 模板的选择追加到选择列表中
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

    # 如果是静态问题且非零，并且使用 Cutlass 模板
    if static_shape and is_nonzero and use_cutlass_template(layout, m, n, k):
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2])

    # 如果使用 CK 模板
    if use_ck_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, [mat1, mat2])

    # 如果使用 CPP 打包 GEMM 模板
    if use_cpp_packed_gemm_template(layout, mat1, mat2):
        CppPackedGemmTemplate.add_choices(
            choices,
            layout,
            [mat1, mat2],
        )

    # 如果没有选择，且不使用 ATen GEMM 内核，并且自动回退到 ATen
    if (
        len(choices) == 0
        and not use_aten_gemm_kernels()
        and inductor_config.autotune_fallback_to_aten
    ):
        # 记录警告信息，没有 GEMM 的选择，回退到 ATen 后端
        log.warning("No choices for GEMM, using ATen backend as fallback")
        return aten_mm.bind((mat1, mat2), aten_layout).output_node()

    try:
        # 自动调整选择算法
        return autotune_select_algorithm("mm", choices, [mat1, mat2], layout)
    except NoValidChoicesError:
        # 如果没有有效的选择，则根据配置决定是否回退到 ATen
        if not inductor_config.autotune_fallback_to_aten:
            raise
        log.warning("All choices for GEMM were invalid, using ATen backend as fallback")
        return aten_mm.bind((mat1, mat2), aten_layout).output_node()

# 定义一个函数 _is_static_problem，用于检查所有输入张量和输出布局是否具有静态形状
def _is_static_problem(inputs_tensors, layout):
    # 通过尝试将维度转换为 int 来检查输入张量和输出布局是否具有静态形状
    static_shape = True
    # 获取静态尺寸，这是通过 WrapperCodeGen 类的 statically_known_list_of_ints_or_none 方法得到的结果
    static_size = WrapperCodeGen.statically_known_list_of_ints_or_none(layout.size)
    
    # 如果 static_size 是 None，则进行进一步的检查
    if static_size is None:
        # 初始化 nonzero 标志为 True
        nonzero = True
        # 遍历 layout.size 中的每一个尺寸 s
        for s in layout.size:
            # 获取 s 的静态整数值，使用 WrapperCodeGen 类的 statically_known_int_or_none 方法
            sz = WrapperCodeGen.statically_known_int_or_none(s)
            # 如果 sz 不为 None 并且 sz 等于 0，则将 nonzero 设为 False，并跳出循环
            if sz is not None and sz == 0:
                nonzero = False
                break
        # 返回结果为 False 和 nonzero 的值
        return False, nonzero
    
    # 计算元素的总数 numel，初始值为 1
    numel = 1
    # 遍历 static_size 中的每一个维度 dim
    for dim in static_size:
        # 将 numel 乘以 dim，计算总元素数
        numel *= dim
    
    # 检查元素总数 numel 是否大于 0，将结果存储在 nonzero 中
    nonzero = numel > 0
    # 返回 static_shape 和 nonzero 的值作为结果
    return static_shape, nonzero
@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    # 调用 mm_args 函数获取矩阵的维度信息，并指定输出数据类型为 torch.int32
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    # 检查矩阵是否具有静态形状和非零元素
    static_shape, is_nonzero = _is_static_problem([mat1, mat2], layout)
    # 根据静态形状和非零元素的情况决定是否使用 Cutlass 模板
    use_cutlass = static_shape and is_nonzero and use_cutlass_template(layout, m, n, k)

    # 如果使用 ATen GEMM 内核，将其加入选择列表中
    choices = (
        [aten__int_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    )

    # TODO: 一旦 cuBLAS 问题得到修复，重新启用急切模式的实现
    # 如果使用 Cutlass 模板或 Triton 模板且启用了 int32，清空选择列表
    if use_cutlass or use_triton_template(layout, enable_int32=True):
        choices = []

    # 如果使用 Cutlass 模板，向选择列表中添加 Cutlass GEMM 选项
    if use_cutlass:
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True
        )

    # 如果矩阵具有非零元素且使用 Triton 模板，为不同配置生成 int8 GEMM 选项
    if is_nonzero and use_triton_template(layout, enable_int32=True):
        for config in int8_mm_configs(m, n, k):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

    # 如果选择列表为空，记录警告并使用 ATen 后端作为回退
    if len(choices) == 0:
        log.warning(
            "No choices for integer GEMM avaialbe using configured backends, using ATen backend as fallback"
        )
        choices = [aten__int_mm.bind((mat1, mat2), layout)]

    try:
        # 根据选择的算法自动调优并返回结果
        return autotune_select_algorithm("int_mm", choices, [mat1, mat2], layout)
    except NoValidChoicesError:
        # 如果没有有效的选择且不允许回退到 ATen 后端，则抛出异常
        if not inductor_config.autotune_fallback_to_aten:
            raise
        log.warning("All choices for GEMM were invalid, using ATen backend as fallback")
        choices = [aten__int_mm.bind((mat1, mat2), layout)]
        # 使用 ATen 后端作为回退
        return autotune_select_algorithm("int_mm", choices, [mat1, mat2], layout)


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    # 确定 CPP 内核的参数顺序
    ordered_kwargs_for_cpp_kernel = ("beta", "alpha")
    # 获取矩阵的维度信息，同时将输入矩阵扩展以匹配输出布局
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    # 检查输入矩阵、mat1 和 mat2 是否具有静态形状和非零元素
    static_shape, is_nonzero = _is_static_problem([inp, mat1, mat2], layout)

    # 如果不是静态形状或不使用最大自动调优，则使用 FlexibleLayout
    if (not is_nonzero) or (not use_max_autotune()):
        # 如果布局是 FixedLayout 类型，则转换为 FlexibleLayout
        from torch._inductor.ir import FixedLayout, FlexibleLayout

        if isinstance(layout, FixedLayout):
            layout = FlexibleLayout(
                device=layout.device, dtype=layout.dtype, size=layout.size
            )
        
        # 如果使用 ATen GEMM 内核，将其加入选择列表中
        choices = (
            [
                aten_addmm.bind(
                    (inp, mat1, mat2),
                    layout,
                    alpha=alpha,
                    beta=beta,
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        
        # 根据选择的算法自动调优并返回结果
        return autotune_select_algorithm("addmm", choices, [inp, mat1, mat2], layout)
    choices = (
        [
            aten_addmm.bind(
                (inp_expanded, mat1, mat2),
                layout,
                alpha=alpha,
                beta=beta,
            )
        ]
        if use_aten_gemm_kernels()  # 如果使用 ATen GEMM 内核
        else []  # 否则为空列表
    )

    if (
        use_aten_gemm_kernels()  # 如果使用 ATen GEMM 内核
        and inp_expanded.get_stride()[0] == 0  # 并且扩展输入的第一个步长为0
        and inp_expanded.get_device().type == "cuda"  # 并且扩展输入的设备类型为 "cuda"
        and inductor_config.triton.autotune_cublasLt  # 并且 Triton 的自动调优 cublasLt 是启用状态
    ):
        # 确保使用 cublasLt 的融合 addmm
        choices.insert(
            0,
            aten_bias_addmm.bind(
                (inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta
            ),
        )

    if is_nonzero and use_triton_template(layout):  # 如果非零且使用 Triton 模板
        for config in mm_configs(m, n, k):  # 对于每个 mm 配置
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(inp_expanded, mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),  # 使用给定的 mm 配置生成选项
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),  # 添加后续操作函数
            )

    if static_shape and is_nonzero and use_cutlass_template(layout, m, n, k):
        # 过滤已知引起 CUDA 非法内存访问错误的情况
        # 在 addmm 使用的线性 GEMM 后处理中，最后一个维度的偏置广播似乎无法工作
        if (
            WrapperCodeGen.statically_known_int_or_none(inp_expanded.layout.stride[-1])
            != 0
        ):
            # 添加 Cutlass GEMM 选项
            CUTLASSGemmTemplate.add_cutlass_gemm_choices(
                choices,
                layout,
                [mat1, mat2, inp_expanded],
                alpha=alpha,
                beta=beta,
            )

    if use_cpp_packed_gemm_template(layout, mat1, mat2):
        # 添加 C++ 打包 GEMM 模板选项
        CppPackedGemmTemplate.add_choices(
            choices,
            layout,
            [inp_expanded, mat1, mat2],
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    add_aten_fallback = False
    if len(choices) == 0:  # 如果选择列表为空
        log.warning("No choices for GEMM, using ATen backend as fallback")  # 记录警告信息
        add_aten_fallback = True  # 设置添加 ATen 回退选项为真

    if add_aten_fallback:
        # 添加 ATen 回退选项
        choices.append(
            aten_addmm.bind(
                (inp_expanded, mat1, mat2),
                layout,
                ordered_kwargs_for_cpp_kernel,  # 按照 C++ 内核的有序关键字参数
                alpha=alpha,
                beta=beta,
            )
        )

        if (
            inp_expanded.get_stride()[0] == 0  # 如果扩展输入的第一个步长为0
            and inp_expanded.get_device().type == "cuda"  # 并且扩展输入的设备类型为 "cuda"
            and inductor_config.triton.autotune_cublasLt  # 并且 Triton 的自动调优 cublasLt 是启用状态
        ):
            # 确保使用 cublasLt 的融合 addmm
            choices.insert(
                0,
                aten_bias_addmm.bind(
                    (inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta
                ),
            )
    try:
        # 尝试选择最优的算法执行 "addmm" 操作，根据给定的选择和参数
        return autotune_select_algorithm(
            "addmm", choices, [inp_expanded, mat1, mat2], layout
        )
    except NoValidChoicesError:
        # 如果没有有效的选择可用
        if not inductor_config.autotune_fallback_to_aten:
            # 如果不允许回退到 ATen 后端，则抛出异常
            raise
        # 记录警告日志，说明所有的 GEMM 选择都无效，将使用 ATen 后端作为回退方案
        log.warning("All choices for GEMM were invalid, using ATen backend as fallback")
        # 使用 ATen 的 addmm 绑定作为回退选择，并传递必要的参数
        fallback_choice = aten_addmm.bind(
            (inp, mat1, mat2),
            layout,
            ordered_kwargs_for_cpp_kernel,
            alpha=alpha,
            beta=beta,
        )
        # 返回 ATen 后端的执行结果节点
        return fallback_choice.output_node()
def fallback_mixed_mm(mat1, mat2, *, out):
    # 使用 torch.mm 执行矩阵乘法，将 mat2 转换为与 mat1 相同的数据类型，并将结果写入 out
    return torch.mm(mat1, mat2.to(mat1.dtype), out=out)


aten_fallback_mixed_mm = ExternKernelChoice(fallback_mixed_mm, None)


@functools.lru_cache(None)
def _is_sm7x_or_older_gpu(index: Optional[int]) -> bool:
    # 获取指定索引（或默认为 0 的索引）的 CUDA 设备属性
    props = torch.cuda.get_device_properties(index or 0)
    # 检查设备的 CUDA 主版本是否小于等于 7，确定是否是 SM7x 或更早的 GPU
    return props.major <= 7


def try_heuristic(m, n, k, choices, mat1, mat2, mat2_dtype, layout):
    if mat1.dtype != torch.float16:
        return None

    # 只有在运行在 A100 GPU 上时才使用启发式方法
    # torch.cuda.get_device_capability() >= (8, 0) 返回 true 表示 A10G，不支持某些配置所需的足够共享内存
    # get_gpu_shared_memory() != 166912 返回 true 表示设备没有 166912 的共享内存
    if (
        not torch.cuda.get_device_capability() >= (8, 0)
    ) or get_gpu_shared_memory() != 166912:
        return None

    if m == 1 and (n % 16 != 0 or k % 16 != 0):
        return None

    if m <= 16 and n >= 4096 and k >= 4096:
        # 针对特定的 m, n, k 值返回 Triton 配置
        return triton_config(
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=128,
            num_stages=5,
            num_warps=4,
        )
    elif m > 16 and m <= 32 and n >= 4096 and k >= 4096:
        return triton_config(
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=128,
            num_stages=5,
            num_warps=4,
        )
    elif m > 32 and m <= 64 and n >= 4096 and k >= 4096:
        return triton_config(
            BLOCK_M=64,
            BLOCK_N=32,
            BLOCK_K=128,
            num_stages=5,
            num_warps=4,
        )
    return None


def tuned_mixed_mm(mat1, mat2, mat2_dtype):
    # 获取矩阵乘法的参数，并确保 layout 为 None
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=None)
    # 检查是否是静态问题，返回静态形状和非零性
    static_shape, is_nonzero = _is_static_problem([mat1, mat2], layout)

    # 绑定 aten_fallback_mixed_mm 作为备选的混合矩阵乘法内核
    fallback = aten_fallback_mixed_mm.bind((mat1, mat2), layout)

    # 初始化选择列表，包含 fallback 内核
    choices = [fallback]

    # 判断是否应跳过 Triton 内核的使用条件
    skip_triton = (
        (
            mat1.layout.dtype != torch.float32
            and not (mat2.layout.is_contiguous() or mat2.layout.is_transposed())
        )
        or _is_sm7x_or_older_gpu(layout.device.index)
        or inductor_config.mixed_mm_choice == "aten"
        or not V.graph.has_feature(layout.device, BackendFeature.TRITON_TEMPLATES)
    )

    if inductor_config.mixed_mm_choice == "triton":
        # 如果配置要求使用 Triton 内核，则清空选择列表
        choices = []
    # 如果不跳过 Triton 加速：
    if not skip_triton:
        # 根据 mat2_dtype 构造用于指定 prologue 类型的字符串，去除开头的 "torch."
        b_prologue_cast_type = f"tl.{mat2_dtype}".replace("torch.", "")
        
        # 如果选择混合矩阵乘法的策略是 "heuristic"：
        if inductor_config.mixed_mm_choice == "heuristic":
            # 初始化空列表用于存放选择
            choices = []
            # 尝试通过启发式方法生成配置
            config = try_heuristic(m, n, k, choices, mat1, mat2, mat2_dtype, layout)
            # 如果成功生成了配置：
            if config is not None:
                # 将选择添加到 mm_template 中，包括输入节点、布局和其他配置选项
                mm_template.maybe_append_choice(
                    choices,
                    input_nodes=(mat1, mat2),
                    layout=layout,
                    **mm_options(config, m, n, k, layout, b_prologue_cast_type),
                )
            # 将 fallback 选项添加到选择列表
            choices.append(fallback)

        # 检查 mat1 或 mat2 是否为 int8 张量
        has_int8_tensor = _is_int8_mat(mat1) or _is_int8_mat(mat2)
        # 遍历混合矩阵乘法的所有配置
        for config in mixed_mm_configs(m, n, k, has_int8_tensor=has_int8_tensor):
            # 将选择添加到 mm_template 中，包括输入节点、布局和其他配置选项
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout, b_prologue_cast_type),
            )

    # 如果静态形状为真且非零，且布局、m、n、k 支持使用 Cutlass 模板：
    if static_shape and is_nonzero and use_cutlass_template(layout, m, n, k):
        # 添加 Cutlass GEMM 模板的选择
        CUTLASSGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True
        )

    # 如果跳过 Triton 加速且没有任何选择，则将 fallback 添加到选择列表
    if skip_triton and not choices:
        choices = [fallback]
    
    # 返回经过自动调优选择算法后的最佳算法名称
    return autotune_select_algorithm("mixed_mm", choices, [mat1, mat2], layout)
# 这个函数是针对特定情况下的 int_mm 操作的调优版本，基于模式 _int_mm -> mul（定义在../fx_passes/post_grad.py），
# 通过强制与 mul 操作融合，防止 int32 _int_mm 输出的实现。
# 仅在 config.force_fuse_int_mm_with_mul = True 时使用此功能。
def tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype, *, layout=None):
    # 如果未指定输出数据类型，则根据 mat3 的数据类型提升为 torch.int32
    out_dtype = (
        torch.promote_types(mat3.get_dtype(), torch.int32)
        if out_dtype is None
        else out_dtype
    )
    # 根据输入参数调整矩阵乘法的参数和布局
    m, n, k, layout, mat1, mat2, mat3 = mm_args(
        mat1, mat2, mat3, layout=layout, out_dtype=out_dtype
    )
    # 创建一个空列表来存储选择的配置项
    choices: List[Dict[Any, Any]] = []
    # 对于给定的 m, n, k，生成 int8 矩阵乘法的配置
    for config in int8_mm_configs(m, n, k):
        # 将可能的选择添加到 choices 列表中，设置输入节点和其他参数
        mm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2, mat3),
            layout=layout,
            **dict(mm_options(config, m, n, k, layout), ACC_TYPE="tl.int32"),
            suffix_args=1,
            epilogue_fn=V.ops.mul,
        )
    # 自动选择算法来执行 "int_mm"，并返回结果
    return autotune_select_algorithm("int_mm", choices, [mat1, mat2, mat3], layout)
```