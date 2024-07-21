# `.\pytorch\torch\_inductor\kernel\mm_common.py`

```
# mypy: allow-untyped-defs
# 导入必要的模块和库
import functools
import itertools
import logging
from typing import cast, List, Tuple  # 导入类型提示相关模块

import sympy  # 导入sympy库，用于符号计算

import torch  # 导入PyTorch库
from torch._inductor.select_algorithm import realize_inputs  # 导入select_algorithm模块中的realize_inputs函数
from torch._inductor.virtualized import V  # 导入virtualized模块中的V类

from .. import config as inductor_config  # 导入当前目录上级的config模块，重命名为inductor_config
from ..runtime.runtime_utils import next_power_of_2  # 从runtime_utils模块中导入next_power_of_2函数
from ..utils import ceildiv as cdiv  # 从utils模块中导入ceildiv函数，重命名为cdiv

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


def triton_config(num_stages, num_warps, **kwargs):
    from triton import Config  # 从triton库中导入Config类

    return Config(kwargs, num_stages=num_stages, num_warps=num_warps)  # 创建Config类的实例并返回


def filtered_configs(
    m: int,
    n: int,
    k: int,
    configs: List[Tuple[int, int, int, int, int]],
    has_int8_tensor=False,
):
    """Heuristic to shrink configs when they are bigger than the input size"""

    min_block_size = 16  # 定义最小块大小为16
    # block_k=16 seems to be causing issues
    # see: https://github.com/triton-lang/triton/issues/2156#issuecomment-1695897424
    min_block_size_k = 32 if has_int8_tensor else 16  # 如果has_int8_tensor为True，设置min_block_size_k为32，否则为16
    m = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                m, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size,
    )  # 计算m的下一个最接近的2的幂，取该值与最小块大小的较大者作为新的m值
    n = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                n, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size,
    )  # 计算n的下一个最接近的2的幂，取该值与最小块大小的较大者作为新的n值
    k = max(
        next_power_of_2(
            V.graph.sizevars.size_hint(
                k, fallback=torch._inductor.config.unbacked_symint_fallback  # type: ignore[arg-type]
            )
        ),
        min_block_size_k,
    )  # 计算k的下一个最接近的2的幂，取该值与min_block_size_k的较大者作为新的k值
    used = set()  # 创建一个空集合used，用于记录已使用的配置
    for block_m, block_n, block_k, num_stages, num_warps in configs:
        # 对于给定的每组配置参数，依次进行处理

        # 缩小配置以适应小尺寸
        block_m = max(min(block_m, m), min_block_size)
        block_n = max(min(block_n, n), min_block_size)
        block_k = max(min(block_k, k), min_block_size_k)

        # 每个线程束计算一个 16x16 的块 = 256
        num_warps = min(num_warps, block_m * block_n // 256)

        # 如果是 HIP 平台
        if torch.version.hip:
            # 遍历 matrix_instr_nonkdim 的可能值 [0, 16]
            for matrix_instr_nonkdim in [0, 16]:
                # 如果 matrix_instr_nonkdim 不是 0 并且 block_m 或 block_n 不是 matrix_instr_nonkdim 的倍数，则跳过
                if matrix_instr_nonkdim != 0 and (
                    block_m % matrix_instr_nonkdim != 0
                    or block_n % matrix_instr_nonkdim != 0
                ):
                    # block_m 和 block_n 必须是 matrix_instr_nonkdim 的倍数
                    continue
                
                # 如果当前配置参数组合未被使用过，则加入到 used 集合中，并生成配置
                if (
                    block_m,
                    block_n,
                    block_k,
                    num_stages,
                    num_warps,
                    matrix_instr_nonkdim,
                ) not in used:
                    used.add(
                        (
                            block_m,
                            block_n,
                            block_k,
                            num_stages,
                            num_warps,
                            matrix_instr_nonkdim,
                        )
                    )
                    # 生成 Triton 配置并 yield 返回
                    yield triton_config(
                        BLOCK_M=block_m,
                        BLOCK_N=block_n,
                        BLOCK_K=block_k,
                        num_stages=num_stages,
                        num_warps=num_warps,
                        matrix_instr_nonkdim=matrix_instr_nonkdim,
                    )
        # 如果不是 HIP 平台
        else:
            # 如果当前配置参数组合未被使用过，则加入到 used 集合中，并生成配置
            if (block_m, block_n, block_k, num_stages, num_warps, 0) not in used:
                used.add((block_m, block_n, block_k, num_stages, num_warps, 0))
                # 生成 Triton 配置并 yield 返回
                yield triton_config(
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_K=block_k,
                    num_stages=num_stages,
                    num_warps=num_warps,
                )
# 定义了一个列表，用于存储内核配置的字典。这些配置会根据条件在目标平台上使用。
# 每个字典包含两个键："config" 表示内核配置元组，"cond" 是一个布尔条件。
mm_kernel_configs = (
    [
        {"config": (32, 32, 16, 1, 2), "cond": True},
        {"config": (32, 32, 128, 2, 4), "cond": torch.version.hip is None},
        {"config": (32, 64, 32, 5, 8), "cond": True},
        {"config": (64, 32, 32, 5, 8), "cond": True},
        {"config": (64, 32, 128, 5, 4), "cond": True},
        {"config": (64, 64, 16, 2, 4), "cond": True},
        {"config": (64, 64, 32, 2, 4), "cond": True},
        {"config": (64, 64, 64, 3, 8), "cond": True},
        {"config": (64, 64, 128, 5, 4), "cond": True},
        {"config": (64, 128, 32, 3, 4), "cond": True},
        {"config": (64, 128, 32, 4, 8), "cond": True},
        {"config": (64, 128, 64, 3, 4), "cond": True},
        {"config": (64, 128, 128, 4, 4), "cond": True},
        {"config": (128, 64, 32, 3, 4), "cond": True},
        {"config": (128, 64, 32, 4, 8), "cond": True},
        {"config": (128, 128, 32, 2, 8), "cond": True},
        {"config": (128, 128, 32, 3, 4), "cond": True},
        {"config": (128, 128, 64, 3, 4), "cond": True},
        {"config": (128, 128, 64, 5, 8), "cond": True},
    ]
    # 根据不同的条件生成不同的内核配置列表
    if inductor_config.max_autotune_gemm_search_space != "EXHAUSTIVE"
    # 如果搜索空间不是完全穷举，则使用上述硬编码的内核配置
    else [
        {"config": (BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps), "cond": True}
        # 通过迭代计算产生可能的内核配置
        for BLOCK_M, BLOCK_N, BLOCK_K in itertools.product(
            [16, 32, 64, 128, 256], repeat=3
        )
        for num_stages in [1, 2, 3, 4, 5]
        for num_warps in [2, 4, 8]
    ]
)

# 用于存储 int8 混合精度内核配置的列表
int8_mm_kernel_configs = [
    {"config": (64, 64, 32, 2, 4), "cond": True},
    {"config": (64, 128, 32, 3, 4), "cond": True},
    {"config": (128, 64, 32, 3, 4), "cond": True},
    {"config": (64, 128, 32, 4, 8), "cond": True},
    {"config": (128, 64, 32, 4, 8), "cond": True},
    {"config": (64, 32, 32, 5, 8), "cond": True},
    {"config": (32, 64, 32, 5, 8), "cond": True},
    {"config": (128, 128, 32, 2, 8), "cond": True},
    {"config": (64, 64, 64, 3, 8), "cond": True},
    # 下面的配置基于条件判断是否包含在内
    # {"config": (32, 32, 128, 2, 4), "cond": True},
    # {"config": (64, 64, 16, 2, 4), "cond": True},
    # {"config": (32, 32, 16, 1, 2), "cond": True},
    {"config": (128, 256, 128, 3, 8), "cond": torch.version.hip is None},
    {"config": (256, 128, 128, 3, 8), "cond": torch.version.hip is None},
]

# 用于存储小 m 值的 mm 的混合精度内核配置列表
mixed_mm_kernel_configs_small_m = [
    {"config": (16, 128, 256, 3, 4), "cond": True},
    {"config": (16, 128, 256, 5, 8), "cond": True},
]

# 如果不是完全穷举搜索空间，将小 m 值的混合精度内核配置添加到标准内核配置中
mixed_mm_kernel_configs = (
    mm_kernel_configs + mixed_mm_kernel_configs_small_m
    if inductor_config.max_autotune_gemm_search_space != "EXHAUSTIVE"
    else mm_kernel_configs
)

# mm 平台配置的元组
mm_platform_configs = tuple(
    # 对于 mm_kernel_configs 列表中满足条件 config["cond"] 的每一个 config 配置项：
    #   - 从 config["config"] 中取出一个元组，元组包含五个整数值。
    cast(Tuple[int, int, int, int, int], config["config"])
    # 将这个元组强制类型转换为 Tuple[int, int, int, int, int] 类型。
    for config in mm_kernel_configs
    if config["cond"]
)
# 将 int8_mm_kernel_configs 中每个满足条件的 config["config"] 转换为 Tuple[int, int, int, int, int] 类型的元组
int8_platform_configs = tuple(
    cast(Tuple[int, int, int, int, int], config["config"])
    for config in int8_mm_kernel_configs
    if config["cond"]
)

# 将 mixed_mm_kernel_configs 中每个满足条件的 config["config"] 转换为 Tuple[int, int, int, int, int] 类型的元组
mixed_mm_platform_configs = tuple(
    cast(Tuple[int, int, int, int, int], config["config"])
    for config in mixed_mm_kernel_configs
    if config["cond"]
)

# 当在 ROCm 上时，将 mm_platform_configs 中的每个配置的第三个参数修改为 0，以启用软件流水线
if torch.version.hip:
    mm_platform_configs = tuple(
        (config[0], config[1], config[2], 0, config[4])
        for config in mm_platform_configs
    )
    int8_platform_configs = tuple(
        (config[0], config[1], config[2], 0, config[4])
        for config in mm_platform_configs
    )
    mixed_mm_platform_configs = tuple(
        (config[0], config[1], config[2], 0, config[4])
        for config in mixed_mm_platform_configs
    )

# 定义 mm_configs 函数，使用 functools.partial 部分应用 filtered_configs 函数，并传入 mm_platform_configs 作为参数
mm_configs = functools.partial(
    filtered_configs,
    configs=mm_platform_configs,
)

# 定义 int8_mm_configs 函数，使用 functools.partial 部分应用 filtered_configs 函数，并传入 int8_platform_configs 作为参数
int8_mm_configs = functools.partial(
    filtered_configs,
    configs=int8_platform_configs,
)

# 定义 mixed_mm_configs 函数，使用 functools.partial 部分应用 filtered_configs 函数，并传入 mixed_mm_platform_configs 作为参数
mixed_mm_configs = functools.partial(
    filtered_configs,
    configs=mixed_mm_platform_configs,
)


def mm_grid(m, n, meta):
    """
    The CUDA grid size for matmul triton templates.
    """
    # 返回 matmul triton 模板的 CUDA 网格大小
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


def acc_type(dtype):
    # 根据 dtype 返回相应的精度类型字符串
    if dtype in (torch.float16, torch.bfloat16):
        return "tl.float32"
    return f"tl.{dtype}".replace("torch.", "")


def mm_options(config, sym_m, sym_n, sym_k, layout, b_prologue_cast_type=None):
    """
    Common options to matmul triton templates.
    """
    # 计算 even_k_symbolic，判断是否允许 TF32，返回选项字典
    even_k_symbolic = (
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"])
        == config.kwargs["BLOCK_K"]
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
        not inductor_config.force_same_precision
        or ((sym_m % 16) == 0 and (sym_n % 16) == 0 and (sym_k % 8) == 0)
    )
    return dict(
        GROUP_M=8,
        EVEN_K=even_k_symbolic,
        ALLOW_TF32=allow_tf32,
        ACC_TYPE=acc_type(layout.dtype),
        B_PROLOGUE_CAST_TYPE=b_prologue_cast_type,
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        **config.kwargs,
    )


def mm_args(mat1, mat2, *others, layout=None, out_dtype=None, use_4x2_dim=False):
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    # 实现 mm,bmm,addmm 等的常见参数处理
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.guard_equals(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()
        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."
    # 从较低级别模块导入 expand 函数
    from ..lowering import expand
    
    # 使用列表推导式对 others 列表中的每个元素进行处理
    # 对于列表 others 中的每个元素 x，调用 expand 函数并传入 x 和 layout.size 作为参数，
    # realize_inputs 函数用于处理 expand 函数的返回值，将处理结果组成新的列表
    others = [realize_inputs(expand(x, layout.size)) for x in others]
    
    # 返回包含 m, n, k, layout, mat1, mat2 以及 others 列表中所有元素的列表
    return [m, n, k, layout, mat1, mat2, *others]
# 定义一个函数 addmm_epilogue，接受三个参数：dtype（数据类型）、alpha（乘数）、beta（乘数）
def addmm_epilogue(dtype, alpha, beta):
    # 定义内部函数 epilogue，接受两个参数：acc（累加器）、bias（偏置）
    def epilogue(acc, bias):
        # 如果 alpha 不等于 1，则将 acc 乘以 alpha，并返回结果
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        # 如果 beta 不等于 1，则将 bias 乘以 beta，并返回结果
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        # 返回 acc 和 bias 相加的结果
        return V.ops.add(acc, bias)

    # 返回内部函数 epilogue 的引用
    return epilogue
```