# `.\pytorch\torch\_inductor\kernel\bmm.py`

```
# 导入必要的模块和库
# mypy: allow-untyped-defs
import logging  # 导入日志模块

import torch  # 导入PyTorch库

from .. import ir, lowering as L  # 导入自定义模块 ir 和 lowering as L
from ..select_algorithm import (  # 从自定义模块导入选择算法相关函数和类
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import (  # 从自定义模块导入工具函数
    ceildiv as cdiv,
    use_aten_gemm_kernels,
    use_cutlass_template,
    use_triton_template,
)
from ..virtualized import V  # 从虚拟化模块导入 V 类

from .mm import _is_static_problem  # 导入 mm 模块中的 _is_static_problem 函数

from .mm_common import addmm_epilogue, mm_args, mm_configs, mm_options  # 从 mm_common 模块导入多个函数和变量

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
aten = torch.ops.aten  # 设置 aten 变量为 torch 的 aten 操作符


def bmm_grid(b, m, n, meta):
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), b, 1)
    # 返回 BMM 矩阵乘法的网格维度，根据元数据中的 BLOCK_M 和 BLOCK_N 参数


bmm_template = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # 基于 triton.ops.matmul 的实现
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # 重新排序程序ID以提高L2缓存性能
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1)  # BMM 的批次维度
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q * stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q * stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # 重新生成 rm 和 rn 以节省寄存器
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1)  # BMM 的批次维度
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # 生成输出的后缀
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask")}}
""",
)
# 定义 Triton 模板对象 bmm_template，用于生成 BMM 矩阵乘法的代码
# 定义对应于 torch.bmm 的外部核选择对象，使用 "at::bmm_out" 作为标识符
aten_bmm = ExternKernelChoice(torch.bmm, "at::bmm_out")

# 定义对应于 torch.baddbmm 的外部核选择对象，使用 "at::baddbmm_out" 作为标识符
aten_baddbmm = ExternKernelChoice(torch.baddbmm, "at::baddbmm_out")

# 注册一个降低操作，将 torch.bmm 降低到 tuned_bmm 函数
@L.register_lowering(aten.bmm)
def tuned_bmm(mat1, mat2, *, layout=None):
    # 如果 mat1 和 mat2 都在 CPU 上
    if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
        # 当内存受限时，将操作分解为小操作
        if mat1.get_size()[1] == 1 or mat2.get_size()[2] == 1:
            # 如果 mat1 的第二维或 mat2 的第三维为 1，添加维度
            mat1 = L.unsqueeze(mat1, -1)
            mat2 = L.unsqueeze(mat2, 1)
            # 返回经过乘法和求和操作后的结果
            return L.sum_(L.mul(mat1, mat2), axis=2)

        # 判断是否可以要求张量在内存中连续存储
        def is_valid_to_require_contiguous(t):
            if not ir.is_storage_and_layout(t):
                return True
            _, layout = ir.as_storage_and_layout(t, freeze=False)
            return isinstance(layout, ir.FlexibleLayout)

        # 判断是否作为 bmm 输入的首选布局
        def is_preferred_layout_as_bmm_input(sizes, strides):
            # 在最后两个维度中有一个是连续的
            return (
                strides[-1] == 1 and (sizes[-2] == 1 or strides[-2] >= sizes[-1])
            ) or (strides[-2] == 1 and (sizes[-1] == 1 or strides[-1] >= sizes[-2]))

        # 如果输入不在最后两个维度上连续，则可能需要将其变为连续
        def may_require_contiguous(t, meta_t):
            sizes = meta_t.meta["val"].size()
            strides = meta_t.meta["val"].stride()
            if not is_preferred_layout_as_bmm_input(sizes, strides):
                t = ir.ExternKernel.require_contiguous(t)
            return t

        # 如果 mat1 可以要求连续，则将其标记为当前节点的第一个参数
        if is_valid_to_require_contiguous(mat1):
            meta_mat1 = V.graph.current_node.args[0]
            mat1 = may_require_contiguous(mat1, meta_mat1)
        # 如果 mat2 可以要求连续，则将其标记为当前节点的第二个参数
        if is_valid_to_require_contiguous(mat2):
            meta_mat2 = V.graph.current_node.args[1]
            mat2 = may_require_contiguous(mat2, meta_mat2)

    # 获取矩阵乘法的参数
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)

    # 选择进行优化的选项
    choices = [aten_bmm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []

    # 如果使用 Triton 模板，生成对应的选择
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

    # 检查矩阵是否为静态形状且非零，并且使用 Cutlass 模板
    static_shape, is_nonzero = _is_static_problem([mat1, mat2], layout)
    if static_shape and is_nonzero and use_cutlass_template(layout, m, n, k):
        # 添加 Cutlass 模板的选择
        from ..codegen.cuda.gemm_template import CUTLASSGemmTemplate

        CUTLASSGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2])

    # 如果没有选择，则记录警告信息并使用 ATen 后端作为后备方案
    if len(choices) == 0:
        log.warning("No choices for GEMM, using ATen backend as fallback")
        choices.append(aten_bmm.bind((mat1, mat2), layout))

    # 使用自动调优算法选择最佳的算法
    return autotune_select_algorithm("bmm", choices, [mat1, mat2], layout)

# 不要注册此函数，因为它比分解后的操作更慢
# 注册函数tuned_baddbmm到特定的下降操作列表L.register_lowering中
# @L.register_lowering(aten.baddbmm)
def tuned_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    # 调用mm_args函数，返回经过调整的m、n、k、layout、mat1、mat2、inp
    m, n, k, layout, mat1, mat2, inp = mm_args(mat1, mat2, inp, layout=layout)

    # 从不同的选项中进行选择以进行调优
    choices = (
        # 如果使用aten_gemm内核，则绑定aten_baddbmm操作
        [aten_baddbmm.bind((inp, mat1, mat2), layout, alpha=alpha, beta=beta)]
        if use_aten_gemm_kernels()
        else []  # 否则选项为空列表
    )

    # 如果使用triton模板，通过遍历mm_configs(m, n, k)配置生成器中的配置
    if use_triton_template(layout):
        for config in mm_configs(m, n, k):
            # 将可能的选项附加到choices列表中，传递输入节点和其他选项
            bmm_template.maybe_append_choice(
                choices,
                input_nodes=(inp, mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
            )

    # 自动选择算法以优化baddbmm操作，并返回结果
    return autotune_select_algorithm("baddbmm", choices, [inp, mat1, mat2], layout)
```