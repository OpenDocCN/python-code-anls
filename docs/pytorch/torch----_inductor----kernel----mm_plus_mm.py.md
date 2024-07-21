# `.\pytorch\torch\_inductor\kernel\mm_plus_mm.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 模块
import functools

# 导入 torch 库
import torch

# 导入 lowering 模块中的 lowerings 对象
from ..lowering import lowerings

# 导入 select_algorithm 模块中的 autotune_select_algorithm、ExternKernelChoice 和 TritonTemplate 对象
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)

# 导入 utils 模块中的 use_aten_gemm_kernels 和 use_triton_template 函数
from ..utils import use_aten_gemm_kernels, use_triton_template

# 导入 virtualized 模块中的 V 对象
from ..virtualized import V

# 导入当前目录下的 mm_common 模块中的 mm_args、mm_grid 和 mm_options 对象
from .mm_common import mm_args, mm_grid, mm_options

# 从 torch.ops.aten 中导入 aten 对象
aten = torch.ops.aten

# 定义 ExternKernelChoice 对象 aten_mm_plus_mm，其调用了 torch.ops.inductor._mm_plus_mm 函数
aten_mm_plus_mm = ExternKernelChoice(
    torch.ops.inductor._mm_plus_mm, "torch::inductor::_mm_plus_mm"
)

# 定义 TritonTemplate 对象 mm_plus_mm_template，用于描述 mm_plus_mm 算法模板
mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",  # 模板名称为 mm_plus_mm
    grid=mm_grid,       # 使用 mm_grid 来定义计算的网格结构
    debug=False,        # 调试模式设置为 False
    source=r"""
{{def_kernel("A", "B", "C", "D")}}
    M = {{size("A", 0)}}         # 从张量 A 中获取维度 0 的大小 M
    N = {{size("B", 1)}}         # 从张量 B 中获取维度 1 的大小 N
    K1 = {{size("A", 1)}}        # 从张量 A 中获取维度 1 的大小 K1
    if M * N == 0:
        # 当输入的尺寸为零时，提前退出
        return
    # K2 = {{size("C", 1)}}
    stride_am = {{stride("A", 0)}}   # 获取张量 A 在维度 0 上的步长 stride_am
    stride_ak = {{stride("A", 1)}}   # 获取张量 A 在维度 1 上的步长 stride_ak
    stride_bk = {{stride("B", 0)}}   # 获取张量 B 在维度 0 上的步长 stride_bk
    stride_bn = {{stride("B", 1)}}   # 获取张量 B 在维度 1 上的步长 stride_bn
    stride_cm = {{stride("C", 0)}}   # 获取张量 C 在维度 0 上的步长 stride_cm
    stride_ck = {{stride("C", 1)}}   # 获取张量 C 在维度 1 上的步长 stride_ck
    stride_dk = {{stride("D", 0)}}   # 获取张量 D 在维度 0 上的步长 stride_dk
    stride_dn = {{stride("D", 1)}}   # 获取张量 D 在维度 1 上的步长 stride_dn

    # 基于 Triton 操作的矩阵乘法
    pid = tl.program_id(0)        # 获取当前程序的 ID
    grid_m = (M + BLOCK_M - 1) // BLOCK_M   # 计算 M 维度上的网格数量
    grid_n = (N + BLOCK_N - 1) // BLOCK_N   # 计算 N 维度上的网格数量

    # 重新排列程序 ID 以提高 L2 缓存性能
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if (((stride_am == 1 and stride_ak == M) or (stride_am == K1 and stride_ak == 1))
        and ((stride_cm == 1 and stride_ck == M) or (stride_cm == K1 and stride_ck == 1))):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M

    if (((stride_bk == 1 and stride_bn == K1) or (stride_bk == N and stride_bn == 1))
        and ((stride_dk == 1 and stride_dn == K1) or (stride_dk == N and stride_dn == 1))):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    C = C + (ram[:, None] * stride_cm + rk[None, :] * stride_ck)
    D = D + (rk[:, None] * stride_dk + rbn[None, :] * stride_dn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k1 in range(K1, 0, -BLOCK_K):
        # 首次矩阵乘法 A @ B
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k1, other=0.)
            b = tl.load(B, mask=rk[:, None] < k1, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
""",
)
    # 对于第二次矩阵乘法，计算 C @ D
    if EVEN_K:
        # 如果 K1 是偶数，则直接加载 C 和 D
        c = tl.load(C)
        d = tl.load(D)
    else:
        # 如果 K1 是奇数，则根据条件加载 C 和 D，未加载部分填充为 0
        c = tl.load(C, mask=rk[None, :] < k2, other=0.)
        d = tl.load(D, mask=rk[:, None] < k2, other=0.)
    # 累加结果到 acc 中
    acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)
    # 更新 C 和 D 的位置，移动到下一个块的起始位置
    C += BLOCK_K * stride_ck
    D += BLOCK_K * stride_dk

# 创建用于索引的 idx_m 和 idx_n
idx_m = rm[:, None]
idx_n = rn[None, :]
# 创建一个布尔掩码，用于过滤超出边界的索引
mask = (idx_m < M) & (idx_n < N)

# 将结果存储到后缀中
{{store_output(("idx_m", "idx_n"), "acc", "mask")}}
# 装饰器，用于缓存函数的返回值，避免重复计算
@functools.lru_cache(None)
# 返回包含针对不同硬件配置的矩阵乘法参数的列表
def mm_configs():
    import triton

    # 存储各种内核配置的字典列表。只有条件为真的配置才会在目标平台上使用
    mm_triton_configs = [
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 3,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 16,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
            "num_stages": 4,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            "num_stages": 1,
            "num_warps": 8,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 128},
            "num_stages": 1,
            "num_warps": 8,
            "cond": torch.version.hip is None,
        },
        {
            "config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 16},
            "num_stages": 2,
            "num_warps": 4,
            "cond": True,
        },
        {
            "config": {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 16},
            "num_stages": 1,
            "num_warps": 2,
            "cond": True,
        },
    ]

    # 根据当前的运行环境，过滤出条件为真的配置
    if torch.version.hip:
        # 在 ROCm 平台上，将 num_stages 设置为 1，因为流水线化不提供任何好处
        filtered_configs = [
            triton.Config(c["config"], num_stages=1, num_warps=c["num_warps"])
            for c in mm_triton_configs
            if c["cond"]
        ]
    else:
        # 在其他平台上，保留满足条件的配置
        filtered_configs = [
            triton.Config(
                c["config"], num_stages=c["num_stages"], num_warps=c["num_warps"]
            )
            for c in mm_triton_configs
            if c["cond"]
        ]

    # 返回过滤后的配置列表
    return filtered_configs


def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    """
    计算 mm(mat1, mat2) + mm(mat3, mat4)
    """
    # 提取矩阵乘法的参数，并进行布局优化（如果有的话）
    m1, n1, k1, layout1, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    m2, n2, _, layout2, mat3, mat4 = mm_args(mat3, mat4, layout=layout)
    # 优化是可选的，因为我们可以选择不进行融合
    # 检查矩阵乘法的维度条件是否满足，或者是否尺寸静态已知的列表相等
    if (
        m1 * n1 == 0                                 # 检查第一个矩阵的行列乘积是否为零
        or m2 * n2 == 0                              # 检查第二个矩阵的行列乘积是否为零
        or not V.graph.sizevars.statically_known_list_equals(
            mat1.get_size(), mat3.get_size()         # 检查第一个矩阵和第三个矩阵的大小是否静态已知并相等
        )
        or not V.graph.sizevars.statically_known_list_equals(
            mat2.get_size(), mat4.get_size()         # 检查第二个矩阵和第四个矩阵的大小是否静态已知并相等
        )
    ):
        # 当问题解决后，支持不同的 K 值：https://github.com/openai/triton/issues/967
        # 返回加法操作的结果，使用低级下降函数计算矩阵乘积
        return lowerings[aten.add](
            lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4)
        )

    assert layout1 == layout2                       # 断言确认布局是否一致
    # 选择调整的选项
    choices = (
        [aten_mm_plus_mm.bind((mat1, mat2, mat3, mat4), layout1)]   # 使用 ATen GEMM 内核时的选项
        if use_aten_gemm_kernels()                                 # 检查是否使用 ATen GEMM 内核
        else []                                                    # 否则为空列表
    )
    if use_triton_template(layout1):                             # 检查是否使用 Triton 模板
        for config in mm_configs():                              # 遍历所有的矩阵乘法配置
            # 查看 https://github.com/openai/triton/issues/1298
            # 如果配置中的 BLOCK_K 值小于 k1，则追加选择
            if config.kwargs["BLOCK_K"] < k1:
                mm_plus_mm_template.maybe_append_choice(
                    choices,
                    input_nodes=(mat1, mat2, mat3, mat4),         # 添加输入节点
                    layout=layout1,
                    **mm_options(config, m1, n1, k1, layout1),   # 使用矩阵乘法的选项
                )

    # 选择最佳算法进行自动调优，以执行矩阵乘法加法操作
    return autotune_select_algorithm(
        "mm_plus_mm", choices, [mat1, mat2, mat3, mat4], layout1
    )
```