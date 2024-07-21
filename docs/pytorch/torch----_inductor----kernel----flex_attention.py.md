# `.\pytorch\torch\_inductor\kernel\flex_attention.py`

```py
# mypy: allow-untyped-defs
""" Triton Implementation of the flex_attention Kernel"""

# 引入日志模块
import logging
# 引入枚举类型的支持
from enum import auto, Enum
# 引入类型提示相关的模块
from typing import Any, List, Tuple

# 引入PyTorch库
import torch
# 引入项目的配置模块
from .. import config
# 引入项目中的IR相关模块
from ..ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    InputBuffer,
    IRNode,
    StorageBox,
    Subgraph,
    TensorBox,
)
# 引入降低（lowering）相关的模块和函数
from ..lowering import empty_strided, lowerings, register_lowering
# 引入算法选择相关模块
from ..select_algorithm import autotune_select_algorithm, TritonTemplate

# 设置日志记录器
log = logging.getLogger(__name__)
# 设置PyTorch的操作命名空间别名
aten = torch.ops.aten


class SubgraphType(Enum):
    """The type of subgraph for which we want to generate an output buffer."""

    # 子图的类型枚举定义
    FWD = auto()  # 前向传播
    JOINT_FWD = auto()  # 反向传播的重计算步骤
    JOINT_BWD = auto()  # 联合反向传播的反向传播步骤


def flex_attention_grid(batch_size, num_heads, num_queries, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, ceil_div(n_queries, query_block_size), 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    # 导入Triton库
    import triton

    # 返回并行化网格的维度
    return (triton.cdiv(num_queries, meta["BLOCK_M"]), batch_size * num_heads, 1)


def create_placeholder(
    name: str, dtype: torch.dtype, device: torch.device
) -> TensorBox:
    """Creates a placeholder input buffers for producing subgraph_output."""
    # 创建用于生成子图输出的占位符输入缓冲区
    input_buffer = InputBuffer(name, FixedLayout(device, dtype, [1], [1]))
    # 使用输入缓冲区创建张量盒子
    return TensorBox.create(input_buffer)


def index_to_other_buffers(cnt: int, graph_type: SubgraphType) -> int:
    """This function needs to be aware of the signatures for flex_attention_forward
    and flex_attention_backward. If new args are added, or the signature changes
    be sure to update the indexing math

    Args:
        cnt (int): The current index of the placeholder node
        is_joint_graph (bool): Whether or not this subgraph represents the joint graph
    """
    # 当前的前向传播参数列表
    # fwd_args = [
    #   query,
    #   key,
    #   value,
    #   score_mod,
    #   sparse_kv_num_blocks,
    #   sparse_kv_indices,
    #   sparse_q_num_blocks,
    #   sparse_q_indices,
    #   SPARSE_KV_BLOCK_SIZE,
    #   SPARSE_Q_BLOCK_SIZE,
    #   *other_buffers
    # ]
    # 对于前向图，有5个虚拟值，因此当第一个被提升的参数出现时，cnt = 5，
    # 并且索引缓冲区的起始位置是args[10]，因此我们从当前的cnt中加上5
    if graph_type == SubgraphType.FWD:
        return cnt + 5

    # 当前的反向传播参数列表
    # bwd_args = [
    #   q,
    #   k,
    #   v,
    #   out,
    #   lse,
    #   grad_out,
    #   fw_graph,
    #   joint_graph,
    #   sparse_kv_num_blocks,
    #   sparse_kv_indices,
    #   sparse_q_num_blocks,
    #   sparse_q_indices,
    #   SPARSE_KV_BLOCK_SIZE,
    #   SPARSE_Q_BLOCK_SIZE,
    #   *other_buffers
    # ]
    # 虽然有5个虚拟值，但other_buffers的起始位置是在索引14处
    # 因此，对于反向图，我们需要将当前的cnt值加上14
    return cnt + 14
    # 如果图类型为 SubgraphType.JOINT_FWD，则返回 cnt + 9
    if graph_type == SubgraphType.JOINT_FWD:
        return cnt + 9

    # 如果图类型为 SubgraphType.JOINT_BWD，则返回 cnt + 8
    # 这里使用了 6 个虚拟值作为 bwd 的参数，同时 other_buffers 仍然从索引 14 开始
    if graph_type == SubgraphType.JOINT_BWD:
        return cnt + 8
# 初始化计数器，用于计数环境变量的数量
cnt = 0
# 初始化空字典，用于存储环境变量
env = {}
    # 遍历子图的节点
    for node in subgraph.graph_module.graph.nodes:
        # 对于占位符节点"placeholder"有两类需要特别处理
        # 对于前 n_scalar_inps 个输入，我们期望这些占位符是在 flex Attention HOP 中的 make_fx 调用中生成的
        # 因此，我们需要为每个输入创建一个新的 TensorBox 占位符
        # 对于其余的输入，我们期望这些是填充 '*other_buffers' 元组的提升输入，已经作为参数传递了相应的 TensorBoxes
        if node.op == "placeholder":
            # 判断是否为提升的输入
            is_lifted_input = cnt >= len(placeholder_inps)
            # 获取提升输入的索引
            lifted_input_index = index_to_other_buffers(cnt, graph_type)
            # 根据条件选择对应的 TensorBox，并将其存储在环境字典中
            env[node] = (
                args[lifted_input_index] if is_lifted_input else placeholder_inps[cnt]
            )
            cnt += 1
        elif node.op == "call_function":
            # 对于 call_function 我们使用默认的 lowerings 并将已创建的 TensorBoxes 作为参数传递
            from torch.utils._pytree import tree_map

            # 使用环境字典映射节点的参数和关键字参数，如果节点在环境中存在则使用其值，否则使用节点自身
            args, kwargs = tree_map(
                lambda x: env[x] if x in env else x, (node.args, node.kwargs)
            )
            # 使用 lowerings 执行函数调用，并将结果存储在环境字典中
            env[node] = lowerings[node.target](*args, **kwargs)
        elif node.op == "output":
            # 对于输出节点，我们需要创建一个 ComputedBuffer
            # 该节点代表实际的分数修改
            if graph_type == SubgraphType.FWD or graph_type == SubgraphType.JOINT_FWD:
                output_node = node.args[0]
            else:
                output_node = node.args[0][0]
            # 获取输出节点对应的 TensorBox
            output_buffer = env[output_node]
            # 断言确保输出节点的数据类型为 TensorBox
            assert isinstance(output_buffer, TensorBox), (
                "The output node for flex attention's subgraph must be a TensorBox, but got: ",
                type(output_buffer),
            )
            # 断言确保输出节点的数据类型为 StorageBox
            assert isinstance(output_buffer.data, StorageBox), (
                "The output node for the flex attention subgraph must be a StorageBox, but got: ",
                type(output_buffer),
            )
            # 直接创建并返回将被内联到修改块中的 ComputedBuffer
            subgraph_buffer = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=output_buffer.data.get_device(),
                    dtype=output_buffer.data.get_dtype(),
                    size=output_buffer.data.get_size(),
                ),
                data=output_buffer.data.data,  # type: ignore[arg-type]
            )
            return subgraph_buffer

    # 如果没有找到输出节点，则抛出异常
    raise ValueError("TemplatedAttention was passed a subgraph with no output node!")
flex_attention_template = TritonTemplate(
    name="flex_attention",
    grid=flex_attention_grid,
    source=r"""
{{def_kernel("Q", "K", "V", "LSE", "SPARSE_KV_NUM_BLKS", "SPARSE_KV_IDX")}}
    # 定义灵活注意力机制模板的源代码字符串，包含模板函数和配置说明

    # 静态断言：确保稀疏查询块大小大于等于 BLOCK_M，并且可以被 BLOCK_M 整除
    tl.static_assert(SPARSE_Q_BLOCK_SIZE >= BLOCK_M and SPARSE_Q_BLOCK_SIZE % BLOCK_M == 0)
    # 静态断言：确保稀疏键/值块大小大于等于 BLOCK_N，并且可以被 BLOCK_N 整除
    tl.static_assert(SPARSE_KV_BLOCK_SIZE >= BLOCK_N and SPARSE_KV_BLOCK_SIZE % BLOCK_N == 0)

    # 定义查询 Q 的步长
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qk = {{stride("Q", 3)}}

    # 定义键 K 的步长
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_kn = {{stride("K", 2)}}
    stride_kk = {{stride("K", 3)}}

    # 定义值 V 的步长
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    stride_vk = {{stride("V", 2)}}
    stride_vn = {{stride("V", 3)}}

    # 获取 Q 的维度信息
    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    Q_LEN = {{size("Q", 2)}}
    KV_LEN = {{size("K", 2)}}

    # 定义 QK 缩放系数
    qk_scale = 1.0
    # 获取矩阵乘法的精度类型
    MATMUL_PRECISION = Q.dtype.element_ty

    # 获取线程的程序 ID，用于计算开始索引和偏移量
    start_m = tl.program_id(0)
    off_z = tl.program_id(1) // H
    off_h = tl.program_id(1) % H

    # 计算 Q 的偏移量
    q_offset = off_z * stride_qz + off_h * stride_qh
    # 计算 K 的偏移量
    k_offset = off_z * stride_kz + off_h * stride_kh
    # 计算 V 的偏移量
    v_offset = off_z * stride_vz + off_h * stride_vh

    # 获取稀疏查询和键/值的维度信息
    SPARSE_Z = {{size("SPARSE_KV_NUM_BLKS", 0)}}
    SPARSE_H = {{size("SPARSE_KV_NUM_BLKS", 1)}}

    # 计算稀疏查询的索引
    sparse_idx_z = off_z % SPARSE_Z
    sparse_idx_h = off_h % SPARSE_H

    # 定义常量：稀疏查询块大小与 BLOCK_M 的比率
    SPARSE_Q_MULTIPLE: tl.constexpr = (SPARSE_Q_BLOCK_SIZE // BLOCK_M)
    # 定义常量：稀疏键/值块大小与 BLOCK_N 的比率
    SPARSE_KV_MULTIPLE: tl.constexpr = (SPARSE_KV_BLOCK_SIZE // BLOCK_N)

    # 定义常量：查询 Q 的块数
    SPARSE_Q_BLOCK_CNT: tl.constexpr = Q_LEN // SPARSE_Q_BLOCK_SIZE
    # 定义常量：键/值 K/V 的块数
    SPARSE_KV_BLOCK_CNT: tl.constexpr = KV_LEN // SPARSE_KV_BLOCK_SIZE

    # 计算稀疏键/值的偏移量
    sparse_hz_offset = sparse_idx_z * SPARSE_H + sparse_idx_h
    # 计算稀疏矩阵操作的块偏移量
    sparse_kv_num_blks_offset = sparse_hz_offset * SPARSE_Q_BLOCK_CNT + start_m // SPARSE_Q_MULTIPLE
    # 计算稀疏矩阵索引的偏移量
    sparse_kv_idx_offset = sparse_hz_offset * SPARSE_Q_BLOCK_CNT * SPARSE_KV_BLOCK_CNT + (start_m // SPARSE_Q_MULTIPLE) * SPARSE_KV_BLOCK_CNT  # noqa: B950
    # 计算稀疏矩阵的索引起始位置
    kv_indices = SPARSE_KV_IDX + sparse_kv_idx_offset
    # 计算稀疏矩阵中第一个 KV 块的起始位置
    kv_start = tl.load(kv_indices) * SPARSE_KV_BLOCK_SIZE # first kv block we're loading
    # 加载稀疏矩阵的块数量
    sparse_kv_num_blocks = tl.load(SPARSE_KV_NUM_BLKS + sparse_kv_num_blks_offset)

    # 创建指向 Q 矩阵块的指针
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(Q_LEN, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # 创建指向 K 矩阵块的指针
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, KV_LEN),
        strides=(stride_kk, stride_kn),
        offsets=(0, kv_start),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    # 创建指向 V 矩阵块的指针
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(KV_LEN, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # 初始化偏移量
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = kv_start + tl.arange(0, BLOCK_N)
    # 初始化指向 m 和 l 的指针
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 加载 Q 矩阵块并按需进行缩放
    q = tl.load(Q_block_ptr)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = (q * qk_scale).to(MATMUL_PRECISION)

    # 循环处理 k 和 v 并更新累加器
    lo = 0
    hi = sparse_kv_num_blocks * SPARSE_KV_MULTIPLE
    for start_n in range(0, hi):
        # -- load k --
        # 从指针 K_block_ptr 处加载数据到 k
        k = tl.load(K_block_ptr)
        
        # -- compute qk ---
        # 计算 q 和 k 的点积，得到 qk
        qk = tl.dot(q, k)
        
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        # 应用分数修正操作，根据指定的参数调整分数
        m = offs_m[:, None]
        n = offs_n[None, :]
        {{ modification(
            subgraph_number=0,
            output_name="post_mod_scores",
            score="qk",
            b="off_z",
            h="off_h",
            m="m",
            n="n",
            out="qk"
        ) | indent_except_first(2) }}
        
        # TODO: In the case that score_mod is linear, this can be LICMed
        # 如果 score_mod 是线性的，则可以进行线性整数最小覆盖处理（LICMed）

        if not SCORE_MOD_IS_LINEAR:
            # 如果 score_mod 不是线性的，则乘以对数的常数以调整分数
            post_mod_scores *= 1.44269504
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        # 计算缩放常数 m_ij
        m_ij = tl.maximum(m_i, tl.max(post_mod_scores, 1))
        
        alpha = tl.math.exp2(m_i - m_ij)
        p = tl.math.exp2(post_mod_scores - m_ij[:, None])

        if not ROWS_GUARANTEED_SAFE:
            # 如果行不保证安全，则进行行掩码操作
            masked_out_rows = (m_ij == float("-inf"))
            alpha = tl.where(masked_out_rows, 0, alpha)
            p = tl.where(masked_out_rows[:, None], 0, p)

        # NB: l_i update is pulled up here since it's a bit faster
        # 更新 l_i 的值，这里将其提前以提高执行效率
        l_i = l_i * alpha + tl.sum(p, 1)
        
        # -- scale and update acc --
        # 缩放和更新 acc
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(MATMUL_PRECISION), v, acc)

        # -- update m_i
        # 更新 m_i 的值为 m_ij
        m_i = m_ij

        # update pointers
        # 更新指针位置
        indices_idx = start_n // SPARSE_KV_MULTIPLE
        
        cur_block = tl.load(kv_indices + indices_idx)
        next_block = tl.load(kv_indices + indices_idx + 1)
        needs_jump = (start_n + 1) % SPARSE_KV_MULTIPLE == 0
        jump_to_block = (next_block - cur_block ) * SPARSE_KV_BLOCK_SIZE - (SPARSE_KV_MULTIPLE - 1) * BLOCK_N
        
        offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK_N
        
        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, offset))
        
        offs_n = offs_n + offset

    # Store output and logsumexp
    # 存储输出和对数求和
    acc = acc / l_i[:, None]
    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]

    mask = idx_m < Q_LEN
    # TODO generalize and add proper mask support
    # TODO: 泛化和添加适当的掩码支持
    {{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc", "mask")}}

    # TODO dont want to write this if we dont require grad
    # 如果不需要梯度，则不执行此操作
    if OUTPUT_LOGSUMEXP:
        off_hz = tl.program_id(1)
        l_ptrs = LSE + off_hz * Q_LEN + offs_m
        lse = m_i + tl.math.log2(l_i)
        tl.store(l_ptrs, lse)
# TODO: 我们可能也需要一个布局约束？
@register_lowering(torch.ops.higher_order.flex_attention, type_promotion_kind=None)
# 注册一个降低（lowering）函数，将其与 torch.ops.higher_order.flex_attention 关联起来，并且禁用类型提升
def flex_attention(*args, **kwargs):
    # 解包函数参数
    (
        query,  # 查询张量
        key,  # 键张量
        value,  # 值张量
        subgraph,  # 子图数据结构
        sparse_kv_num_blocks,  # 稀疏键值对块的数量
        sparse_kv_indices,  # 稀疏键值对的索引
        sparse_q_num_blocks,  # 稀疏查询块的数量
        sparse_q_indices,  # 稀疏查询的索引
        SPARSE_KV_BLOCK_SIZE,  # 稀疏键值对块的大小
        SPARSE_Q_BLOCK_SIZE,  # 稀疏查询块的大小
        *other_buffers,  # 其他缓冲区
    ) = args
    # 对于每个缓冲区，调用其 realize 方法以确保数据已经加载
    for buf in [
        query,
        key,
        value,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
    ]:
        buf.realize()

    # 创建占位符输入列表，用于存储创建的占位符张量
    placeholder_inps = [
        create_placeholder(name, dtype, query.get_device())
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]

    # 构建子图缓冲区，用于存储子图相关信息
    subgraph_buffer = build_subgraph_buffer(
        args, placeholder_inps, subgraph, graph_type=SubgraphType.FWD
    )

    # 创建固定布局对象，指定张量的设备、数据类型、大小和步幅
    layout = FixedLayout(
        query.get_device(),
        query.get_dtype(),
        query.get_size(),
        query.get_stride(),
    )

    # 创建用于存储 logsumexp 的张量，根据 query 的尺寸定义形状
    # 注意：logsumexp 始终以 fp32 存储，不受输入数据类型的影响
    logsumexp_shape = query.get_size()[:-1]  # [B, H, M]
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=torch.float32,
        device=query.get_device(),
    )

    # 初始化空列表和配置元组列表，用于存储选择和配置信息
    choices: List[Any] = []
    configs: List[Tuple[int, int, int, int]] = []

    # 添加默认配置信息到配置列表中
    configs.append(_get_default_config_fwd(query))

    # 如果启用自动调优，添加额外的配置信息到配置列表中
    if config.max_autotune:
        configs += [
            (128, 64, 4, 3),
            (128, 128, 4, 3),
            (128, 128, 8, 2),
            (64, 128, 4, 3),
            (64, 64, 4, 3),
        ]

    # 遍历配置列表，为每个配置尝试添加选择项
    for BLOCK_M, BLOCK_N, num_warps, num_stages in configs:
        # 检查当前配置是否与稀疏矩阵块大小兼容，否则跳过该配置
        if SPARSE_KV_BLOCK_SIZE % BLOCK_N != 0 or SPARSE_Q_BLOCK_SIZE % BLOCK_M != 0:
            continue

        # 添加选择项到选择列表中，配置各种参数和输入
        flex_attention_template.maybe_append_choice(
            choices=choices,
            input_nodes=[
                query,
                key,
                value,
                logsumexp,
                sparse_kv_num_blocks,
                sparse_kv_indices,
            ],
            layout=layout,
            subgraphs=[
                subgraph_buffer,
            ],
            mutated_inputs=[
                logsumexp,
            ],
            num_stages=num_stages,
            num_warps=num_warps,
            call_sizes=query.get_size(),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=query.get_size()[-1],
            SCORE_MOD_IS_LINEAR=False,  # 默认为非线性模式
            ROWS_GUARANTEED_SAFE=False,  # 不保证行安全性
            OUTPUT_LOGSUMEXP=True,  # 输出 logsumexp
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        )

    # 准备用于自动调优的输入列表，包括所有输入缓冲区和其他缓冲区
    inputs_for_autotuning = [
        query,
        key,
        value,
        logsumexp,
        sparse_kv_num_blocks,
        sparse_kv_indices,
    ] + list(other_buffers)
    return (
        autotune_select_algorithm(
            "flex_attention", choices, inputs_for_autotuning, layout
        ),
        logsumexp,
    )



    # 返回一个元组，包含以下两个元素：
    # 1. 调用 autotune_select_algorithm 函数，选择 "flex_attention" 算法，传入 choices, inputs_for_autotuning, layout 参数
    # 2. 返回 logsumexp 变量的当前值
# ---------------------------- Backward HOP Implementation ----------------------------

# 定义一个灵活注意力机制的反向传播模板函数，用于计算并行化的网格配置
def flex_attention_backward_grid(
    batch_size, num_heads, num_queries, d_model, num_key_value, meta
):
    """How is this kernel parallelized?
    Currently this is only parallelizing over batch * num_heads, but we can, and want to
    parallelize over ceil_div(num_key_value, key_value_block_size). To do this will either require
    atomic updates to some grad values or to have a two pass kernel design.
    """
    import triton

    # 返回计算网格的尺寸，主要是通过 ceil_div(num_key_value, meta["BLOCK_N1"]) 和 triton.cdiv(num_queries, meta["BLOCK_M2"]) 来决定
    return (
        triton.cdiv(num_queries, meta["BLOCK_M2"])
        + triton.cdiv(num_key_value, meta["BLOCK_N1"]),
        1,
        batch_size * num_heads,
    )


# 定义一个 Triton 模板对象，用于生成灵活注意力机制的反向传播 CUDA 源代码
flex_attention_backward_template = TritonTemplate(
    name="flex_attention_backward",
    grid=flex_attention_backward_grid,
    source=r"""
{{def_kernel("Q", "K", "V", "OUT", "LSE", "DELTA", "DO", "DQ", "DV", "SPARSE_KV_NUM_BLKS", "SPARSE_KV_IDX", "SPARSE_Q_NUM_BLKS", "SPARSE_Q_IDX")}}
    # Sub notation for this kernel:
    #
    # Q: Query, K: Key, V: Value
    # OUT: Forward output, LSE: logsumexp (logsumexp is always stored in fp32 regardless of the input dtype)
    # DELTA: Precomputed sum(OUT* DO, axis=1)
    # DO: Derivative of Output, DQ: Derivative of Query, DV: Derivative of Value
    # DK: Derivative of Key, is the written to via the store_output call due to some limitations with
    # inductor codegen
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries or keys/values, d: Head dim
    # (Modifiable) Config options:
    # BLOCK_M1: when calculating DK & DV, iterate over BLOCK_M1 across the seqlen dim of Q in each thread block.
    # BLOCK_N1: when calculating DK & DV, the thread block size across the seqlen dim of K/V.
    # BLOCK_M2: when calculating DQ, the thread block size across the seqlen dim of Q.
    # BLOCK_N2: when calculating DQ, iterate over BLOCK_N2 across the seqlen dim of K/V in each thread block.
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    #
    # The following SPARSE_* is defined in the block sparse mask grid, rather than the thread block grid.
    # SPARSE_KV_NUM_BLKS: The number of unmasked K/V blocks for each query.
    # SPARSE_KV_IDX: The indices of unmasked K/V blocks for each query.
    # SPARSE_Q_NUM_BLKS: The number of unmasked Q blocks for each key/value.
    # SPARSE_Q_IDX: The indices of unmasked Q blocks for each key/value.

    # 定义 Q 的步长
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qd = {{stride("Q", 3)}}
    # 定义 K 的步长
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_km = {{stride("K", 2)}}
    stride_kd = {{stride("K", 3)}}
    # 定义 V 的步长
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    # 定义 stride_vm 作为 'V' 维度上的步幅为 2 的张量
    stride_vm = {{stride("V", 2)}}
    
    # 定义 stride_vd 作为 'V' 维度上的步幅为 3 的张量
    stride_vd = {{stride("V", 3)}}

    # 定义 Z 为第 0 维度上的大小
    Z = {{size("Q", 0)}}
    
    # 定义 H 为第 1 维度上的大小
    H = {{size("Q", 1)}}
    
    # 定义 Q_LEN 为第 2 维度上的大小
    Q_LEN = {{size("Q", 2)}}
    
    # 定义 KV_LEN 为 'K' 维度上的第 2 维度大小
    KV_LEN = {{size("K", 2)}}

    # 定义 MATMUL_PRECISION 为 Q 张量元素类型
    MATMUL_PRECISION = Q.dtype.element_ty

    # 获取程序的 ID
    pid = tl.program_id(0)
    
    # 计算 KV 块的数量
    NUM_KV_BLOCKS = KV_LEN // BLOCK_N1

    # 计算 off_hz 作为程序 ID 为 2 时的偏移量
    off_hz = tl.program_id(2)
    
    # 计算 off_z 作为 off_hz 除以 H 得到的批次索引
    off_z = off_hz // H # batch idx
    
    # 计算 off_h 作为 off_hz 对 H 求余得到的头索引
    off_h = off_hz % H # head idx

    # 获取 SPARSE_KV_NUM_BLKS 的第 0 维度大小
    SM_Z = {{size("SPARSE_KV_NUM_BLKS", 0)}}
    
    # 获取 SPARSE_KV_NUM_BLKS 的第 1 维度大小
    SM_H = {{size("SPARSE_KV_NUM_BLKS", 1)}}

    # 计算 sparse_idx_z 作为 off_z 对 SM_Z 求余
    sparse_idx_z = off_z % SM_Z
    
    # 计算 sparse_idx_h 作为 off_h 对 SM_H 求余
    sparse_idx_h = off_h % SM_H

    # 计算 off_chz 作为 off_hz 乘以 Q_LEN 转换为 int64 类型
    off_chz = (off_hz * Q_LEN).to(tl.int64)
    
    # 计算 q_adj 作为 stride_qh 乘以 (off_hz 对 H 求余) 加上 stride_qz 乘以 (off_hz 除以 H) 转换为 int64 类型
    q_adj = (stride_qh * (off_hz % H) + stride_qz * (off_hz // H)).to(tl.int64)
    
    # 计算 k_adj 作为 stride_kh 乘以 (off_hz 对 H 求余) 加上 stride_kz 乘以 (off_hz 除以 H) 转换为 int64 类型
    k_adj = (stride_kh * (off_hz % H) + stride_kz * (off_hz // H)).to(tl.int64)
    
    # 计算 v_adj 作为 stride_vh 乘以 (off_hz 对 H 求余) 加上 stride_vz 乘以 (off_hz 除以 H) 转换为 int64 类型
    v_adj = (stride_vh * (off_hz % H) + stride_vz * (off_hz // H)).to(tl.int64)

    # 对 Q 应用 q_adj 的偏移量
    Q += q_adj
    
    # 对 K 应用 k_adj 的偏移量
    K += k_adj
    
    # 对 V 应用 v_adj 的偏移量
    V += v_adj
    
    # 对 DO 应用 q_adj 的偏移量
    DO += q_adj
    
    # 对 DQ 应用 q_adj 的偏移量
    DQ += q_adj
    
    # 对 DV 应用 v_adj 的偏移量
    DV += v_adj
    
    # 对 LSE 应用 off_chz 的偏移量
    LSE += off_chz
    
    # 对 DELTA 应用 off_chz 的偏移量
    DELTA += off_chz

    # 生成范围在 [0, BLOCK_DMODEL) 内的整数序列
    offs_k = tl.arange(0, BLOCK_DMODEL)
# 注册一个降低操作，用于灵活注意力的反向传播
@register_lowering(
    torch.ops.higher_order.flex_attention_backward, type_promotion_kind=None
)
def flex_attention_backward(*args, **kwargs):
    # 解包参数元组
    (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        fw_graph,
        joint_graph,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
        SPARSE_KV_BLOCK_SIZE,
        SPARSE_Q_BLOCK_SIZE,
        *other_buffers,
    ) = args

    # 对于需要实现的缓冲区，进行实现操作
    for buf in [
        query,
        key,
        value,
        grad_out,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
    ]:
        buf.realize()

    # 获取查询张量的设备和数据类型
    device = query.get_device()
    dtype = query.get_dtype()

    # 创建前向子图所需的占位符输入
    fwd_placeholder_inps = [
        create_placeholder(name, dtype, device)
        for name, dtype in [
            ("score", dtype),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]

    # 构建前向子图缓冲区
    fw_subgraph_buffer = build_subgraph_buffer(
        args, fwd_placeholder_inps, fw_graph, graph_type=SubgraphType.JOINT_FWD
    )

    # 创建联合反向子图所需的占位符输入
    joint_placeholder_inps = fwd_placeholder_inps + [
        create_placeholder("grad_score_mod", dtype, device)
    ]

    # 构建联合反向子图缓冲区
    joint_subgraph_buffer = build_subgraph_buffer(
        args, joint_placeholder_inps, joint_graph, graph_type=SubgraphType.JOINT_BWD
    )

    # 使用固定的布局创建键的布局对象
    layout_k = FixedLayout(
        key.get_device(),
        key.get_dtype(),
        key.get_size(),
        key.get_stride(),
    )

    # 创建在反向传播核心中需要的 delta 值
    mul_delta = lowerings[aten.mul](out, grad_out)
    delta = lowerings[aten.sum](mul_delta, axis=-1)

    # 创建梯度查询和值的空张量，使用适当的尺寸和步幅
    grad_query = empty_strided(
        query.get_size(), query.get_stride(), dtype=dtype, device=device
    )
    grad_value = empty_strided(
        value.get_size(), value.get_stride(), dtype=dtype, device=device
    )

    # 初始化选择和配置列表
    choices: List[Any] = []
    configs: List[Tuple[int, int, int, int]] = []

    # 添加默认配置到配置列表
    configs.append(_get_default_config_bwd(query))

    # 如果存在最大自动调整配置，添加更多可能的配置选项
    if config.max_autotune:
        for BLOCK1 in [32, 64]:
            for BLOCK2 in [32, 64, 128]:
                if BLOCK2 % BLOCK1 != 0:
                    continue
                for w in [4, 8]:
                    for s in [1, 3, 4, 5]:
                        configs.append((BLOCK1, BLOCK2, w, s))
    # 对于每个配置元组中的BLOCK1, BLOCK2, num_warps, num_stages，依次进行迭代
    for BLOCK1, BLOCK2, num_warps, num_stages in configs:
        # 检查稀疏键值和查询块大小是否可以被BLOCK1整除，如果不能，跳过当前迭代
        if (
            SPARSE_KV_BLOCK_SIZE % BLOCK1 != 0
            or SPARSE_Q_BLOCK_SIZE % BLOCK1 != 0
            or SPARSE_KV_BLOCK_SIZE % BLOCK2 != 0
            or SPARSE_Q_BLOCK_SIZE % BLOCK2 != 0
        ):
            continue

        # 向后注意力模板中可能添加选项
        flex_attention_backward_template.maybe_append_choice(
            choices=choices,
            # 输入节点列表，包括查询、键、值、输出、logsumexp等
            input_nodes=[
                query,
                key,
                value,
                out,
                logsumexp,
                delta,
                grad_out,
                grad_query,
                grad_value,
                sparse_kv_num_blocks,
                sparse_kv_indices,
                sparse_q_num_blocks,
                sparse_q_indices,
            ],
            layout=layout_k,  # 我们仅用于grad_key时使用store_output
            subgraphs=[fw_subgraph_buffer, joint_subgraph_buffer],
            mutated_inputs=[grad_query, grad_value],
            call_sizes=query.get_size() + [key.get_size()[2]],  # 调用尺寸
            num_stages=num_stages,  # 阶段数目
            num_warps=num_warps,  # 线程束数目
            BLOCK_M1=BLOCK1,
            BLOCK_N1=BLOCK2,
            BLOCK_M2=BLOCK2,
            BLOCK_N2=BLOCK1,
            BLOCK_DMODEL=query.get_size()[-1],  # 模型维度
            # 目前，我们始终假设"sound"选项
            SCORE_MOD_IS_LINEAR=False,
            SPARSE_Q_BLOCK_SIZE=SPARSE_Q_BLOCK_SIZE,
            SPARSE_KV_BLOCK_SIZE=SPARSE_KV_BLOCK_SIZE,
        )

    # 自动调优的输入列表，包括查询、键、值、输出等及其他缓冲区
    inputs_for_autotuning = [
        query,
        key,
        value,
        out,
        logsumexp,
        delta,
        grad_out,
        grad_query,
        grad_value,
        sparse_kv_num_blocks,
        sparse_kv_indices,
        sparse_q_num_blocks,
        sparse_q_indices,
    ] + list(other_buffers)

    # 选择自动调优算法，用于"flex_attention_backward"
    grad_key = autotune_select_algorithm(
        "flex_attention_backward", choices, inputs_for_autotuning, layout_k
    )

    # 返回梯度查询、梯度键和梯度值
    return (
        grad_query,
        grad_key,
        grad_value,
    )
```