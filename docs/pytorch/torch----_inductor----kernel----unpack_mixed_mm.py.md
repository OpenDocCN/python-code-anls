# `.\pytorch\torch\_inductor\kernel\unpack_mixed_mm.py`

```
# mypy: allow-untyped-defs
# 导入日志模块
import logging
# 导入类型提示模块
from typing import List, TYPE_CHECKING

# 导入自定义模块
from ..select_algorithm import autotune_select_algorithm, TritonTemplate
from .mm_common import mm_args, mm_configs, mm_grid, mm_options

# 如果类型检查开启，导入特定模块
if TYPE_CHECKING:
    from ..ir import ChoiceCaller

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 定义一个特定的 Triton 模板，用于描述一个特定的矩阵乘法操作
uint4x2_mixed_mm_template = TritonTemplate(
    name="uint4x2_mixed_mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # 基于 Triton 的矩阵乘法操作
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # 重新排序程序 ID 以获得更好的 L2 缓存性能
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None]//2 * stride_bk + rbn[None, :] * stride_bn)
    b_shifts = 4*(rk%2)
    b_subs = 8*(1-(rk%2))

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        b = ((b >> b_shifts[:, None]) & 0xF) - 8
        b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K//2 * stride_bk

    # 重新生成 rm 和 rn 以节省寄存器
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # 生成后缀以存储输出结果
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)

# 定义一个函数，用于执行经过调优的 uint4x2_mixed_mm 矩阵乘法操作
def tuned_uint4x2_mixed_mm(mat1, mat2, mat2_mm_shape, mat2_dtype):
    # 解析矩阵的维度和布局信息
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=None, use_4x2_dim=True)
    # 初始化选择列表
    choices: List[ChoiceCaller] = []
    # 根据不同的配置，可能添加选择项到选择列表中
    b_prologue_cast_type = f"tl.{mat2_dtype}".replace("torch.", "")
    for config in mm_configs(m, n, k):
        uint4x2_mixed_mm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2),
            layout=layout,
            **mm_options(config, m, n, k, layout, b_prologue_cast_type),
        )
    # 自动选择最优的算法并返回结果
    return autotune_select_algorithm("uint4x2_mixed_mm", choices, [mat1, mat2], layout)
```