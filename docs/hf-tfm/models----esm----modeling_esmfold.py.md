# `.\models\esm\modeling_esmfold.py`

```
# 编码声明，指明代码文件采用的编码格式为utf-8
# 版权声明
import math  # 导入 math 模块
import sys  # 导入 sys 模块
from dataclasses import dataclass  # 从 dataclasses 模块中导入 dataclass 装饰器
from functools import partial  # 从 functools 模块中导入 partial 函数
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union  # 从 typing 模块中导入多种类型
import numpy as np  # 导入 numpy 模块并将其重命名为 np
import torch  # 导入 torch 模块
import torch.nn as nn  # 导入 torch.nn 模块并将其重命名为 nn
from torch.nn import LayerNorm  # 从 torch.nn 模块中导入 LayerNorm 类
# 导入相关模块和函数
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
    ContextManagers,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_scipy_available,
    logging,
    replace_return_docstrings,
)
# 从 .modeling_esm 模块中导入 EsmConfig, ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
    OFProtein,
    Rigid,
    Rotation,
    atom14_to_atom37,
    chunk_layer,
    compute_predicted_aligned_error,
    compute_tm,
    frames_and_literature_positions_to_atom14_pos,
    make_atom14_masks,
    residue_constants,
    to_pdb,
    torsion_angles_to_frames,
)

# 获取logger对象
logger = logging.get_logger(__name__)
# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "facebook/esmfold_v1"
# 用于文档的配置
_CONFIG_FOR_DOC = "EsmConfig"

@dataclass
class EsmForProteinFoldingOutput(ModelOutput):
    """
    Output type of [`EsmForProteinFoldingOutput`].
    # 定义函数的参数列表，说明每个参数的类型和含义
    Args:
        frames (`torch.FloatTensor`):
            输出帧。
        sidechain_frames (`torch.FloatTensor`):
            输出侧链帧。
        unnormalized_angles (`torch.FloatTensor`):
            预测的未归一化的主链和侧链扭转角度。
        angles (`torch.FloatTensor`):
            预测的主链和侧链扭转角度。
        positions (`torch.FloatTensor`):
            预测的主链和侧链原子的位置。
        states (`torch.FloatTensor`):
            蛋白质折叠主干中的隐藏状态。
        s_s (`torch.FloatTensor`):
            通过连接 ESM-2 LM stem 每层的隐藏状态导出的每个残基的嵌入。
        s_z (`torch.FloatTensor`):
            成对残基嵌入。
        distogram_logits (`torch.FloatTensor`):
            用于计算残基距离的直方图的输入 logits。
        lm_logits (`torch.FloatTensor`):
            ESM-2 蛋白质语言模型 stem 输出的 logits。
        aatype (`torch.FloatTensor`):
            输入的氨基酸 (AlphaFold2 索引)。
        atom14_atom_exists (`torch.FloatTensor`):
            每个原子在 atom14 表示中是否存在。
        residx_atom14_to_atom37 (`torch.FloatTensor`):
            在 atom14 和 atom37 表示之间的原子映射。
        residx_atom37_to_atom14 (`torch.FloatTensor`):
            在 atom37 和 atom14 表示之间的原子映射。
        atom37_atom_exists (`torch.FloatTensor`):
            每个原子在 atom37 表示中是否存在。
        residue_index (`torch.FloatTensor`):
            蛋白链中每个残基的索引。除非使用了内部填充标记，否则这将仅是从 0 到 `sequence_length` 的整数序列。
        lddt_head (`torch.FloatTensor`):
            用于计算 plddt 的 lddt head 的原始输出。
        plddt (`torch.FloatTensor`):
            每个残基的置信度分数。低置信度区域可能表示模型预测不确定的区域，或蛋白质结构失序的区域。
        ptm_logits (`torch.FloatTensor`):
            用于计算 ptm 的原始 logits。
        ptm (`torch.FloatTensor`):
            TM-score 输出，代表模型对整体结构的高级置信度。
        aligned_confidence_probs (`torch.FloatTensor`):
            对齐结构的每个残基置信度分数。
        predicted_aligned_error (`torch.FloatTensor`):
            模型预测值与真实值之间的预测误差。
        max_predicted_aligned_error (`torch.FloatTensor`):
            每个样本的最大预测误差。
    # 定义一个变量 positions，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    positions: torch.FloatTensor = None
    
    # 定义一个变量 states，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    states: torch.FloatTensor = None
    
    # 定义一个变量 s_s，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    s_s: torch.FloatTensor = None
    
    # 定义一个变量 s_z，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    s_z: torch.FloatTensor = None
    
    # 定义一个变量 distogram_logits，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    distogram_logits: torch.FloatTensor = None
    
    # 定义一个变量 lm_logits，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    lm_logits: torch.FloatTensor = None
    
    # 定义一个变量 aatype，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    aatype: torch.FloatTensor = None
    
    # 定义一个变量 atom14_atom_exists，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    atom14_atom_exists: torch.FloatTensor = None
    
    # 定义一个变量 residx_atom14_to_atom37，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    residx_atom14_to_atom37: torch.FloatTensor = None
    
    # 定义一个变量 residx_atom37_to_atom14，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    residx_atom37_to_atom14: torch.FloatTensor = None
    
    # 定义一个变量 atom37_atom_exists，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    atom37_atom_exists: torch.FloatTensor = None
    
    # 定义一个变量 residue_index，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    residue_index: torch.FloatTensor = None
    
    # 定义一个变量 lddt_head，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    lddt_head: torch.FloatTensor = None
    
    # 定义一个变量 plddt，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    plddt: torch.FloatTensor = None
    
    # 定义一个变量 ptm_logits，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    ptm_logits: torch.FloatTensor = None
    
    # 定义一个变量 ptm，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    ptm: torch.FloatTensor = None
    
    # 定义一个变量 aligned_confidence_probs，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    aligned_confidence_probs: torch.FloatTensor = None
    
    # 定义一个变量 predicted_aligned_error，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    predicted_aligned_error: torch.FloatTensor = None
    
    # 定义一个变量 max_predicted_aligned_error，数据类型为 torch 中的浮点型张量，并将其初始化为 None
    max_predicted_aligned_error: torch.FloatTensor = None
ESMFOLD_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        masking_pattern (`torch.LongTensor` of shape `({0})`, *optional*):
            Locations of tokens to mask during training as a form of regularization. Mask values selected in `[0, 1]`.
        num_recycles (`int`, *optional*, defaults to `None`):
            Number of times to recycle the input sequence. If `None`, defaults to `config.num_recycles`. "Recycling"
            consists of passing the output of the folding trunk back in as input to the trunk. During training, the
            number of recycles should vary with each batch, to ensure that the model learns to output valid predictions
            after each recycle. During inference, num_recycles should be set to the highest value that the model was
            trained with for maximum accuracy. Accordingly, when this value is set to `None`, config.max_recycles is
            used.
"""


def is_fp16_enabled():
    # 检查当前是否启用了FP16（半精度浮点数）计算
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    # 进一步检查自动混合精度计算是否已启用
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()

    return fp16_enabled


def is_deepspeed_initialized():
    # 检查是否导入了deepspeed模块
    if is_deepspeed_available():
        return False
    else:
        try:
            import deepspeed

            # 检查当前是否初始化了DeepSpeed
            return deepspeed.utils.is_initialized()
        except Exception:
            return False


def collate_dense_tensors(samples: List[torch.Tensor], pad_v: float = 0) -> torch.Tensor:
    """
    Takes a list of tensors with the following dimensions:
        [(d_11, ..., d_1K),
         (d_21, ..., d_2K), ..., (d_N1, ..., d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    # 检查样本列表是否为空
    if len(samples) == 0:
        return torch.Tensor()
    # 检查样本张量的维度是否一致
    if len({x.dim() for x in samples}) != 1:
        raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")
    # 从样本中获取设备信息，假设所有样本在同一设备上
    (device,) = tuple({x.device for x in samples})
    # 获取样本的最大形状
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    # 创建一个形状为最大形状的空张量，使用样本的设备，数据类型为第一个样本的数据类型
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    # 用填充值填充结果张量
    result.fill_(pad_v)
    # 遍历所有样本
    for i in range(len(samples)):
        # 获取结果张量中对应的元素
        result_i = result[i]
        # 获取当前样本
        t = samples[i]
        # 将当前样本的值赋给结果张量中对应的位置
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    # 返回结果张量
    return result
# 定义函数 flatten_final_dims，将张量 t 最后的 no_dims 维度展平
def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


# 定义函数 permute_final_dims，对张量 tensor 进行维度置换
def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    # 计算从最后向前数的维度索引
    zero_index = -1 * len(inds)
    # 构造维度索引列表，首先是前面的维度，然后是待置换的维度
    first_inds = list(range(len(tensor.shape[:zero_index])))
    # 执行置换操作
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# 定义函数 dict_multimap，对字典列表中的每个字典应用给定的函数
def dict_multimap(fn, dicts):
    # 获取第一个字典
    first = dicts[0]
    # 初始化新字典
    new_dict = {}
    # 遍历第一个字典的键值对
    for k, v in first.items():
        # 获取所有字典中相同键的值列表
        all_v = [d[k] for d in dicts]
        # 如果值是字典类型，则递归调用 dict_multimap
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            # 否则应用给定的函数到值列表上
            new_dict[k] = fn(all_v)
    # 返回新字典
    return new_dict


# 定义函数 trunc_normal_init_，使用截断正态分布初始化权重张量
def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    # 获取权重张量的形状
    shape = weights.shape
    # 计算标准差的缩放因子
    scale = scale / max(1, shape[1])

    # 如果没有安装 scipy 库，则使用警告
    if not is_scipy_available():
        logger.warning(
            "This init requires scipy, but scipy was not found, default to an approximation that might not be"
            " equivalent."
        )
        # 计算标准差并进行截断，然后填充权重张量
        std = math.sqrt(scale)
        torch.nn.init.normal_(weights, std=std).clamp(min=0.0, max=2.0 * std)

    else:
        from scipy.stats import truncnorm

        # 使用截断正态分布生成样本并填充权重张量
        std = math.sqrt(scale) / truncnorm.std(a=-2, b=2, loc=0, scale=1)
        samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=weights.numel())
        samples = np.reshape(samples, shape)
        weights.copy_(torch.tensor(samples, device=weights.device))


# 定义函数 ipa_point_weights_init_，用给定的常数填充权重张量
def ipa_point_weights_init_(weights):
    with torch.no_grad():
        # 使用 softplus 函数的反函数值填充权重张量
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


# 定义类 EsmFoldLinear，继承自 nn.Linear，具有内置的非标准初始化方法
class EsmFoldLinear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found in the code.
    """

    # 初始化方法
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                输入层的最终维度
            out_dim:
                层输出的最终维度
            bias:
                是否学习一个加性偏置。默认为True
            init:
                要使用的初始化器。可选项包括:

                "default": LeCun fan-in截断正态分布初始化 "relu": 使用截断正态分布的He初始化 "glorot": 均匀分布的均值初始化 "gating": 权重=0，偏置=1 "normal":
                标准差为1/sqrt(fan_in)的正态初始化 "final": 权重=0，偏置=0

                如果init_fn不是None，则会被init_fn覆盖。
            init_fn:
                一个自定义的初始化器，以权重和偏置作为输入。如果不是None，则覆盖init。
        """
        # 使用父类初始化方法初始化
        super().__init__(in_dim, out_dim, bias=bias)

        # 如果有偏置，则用0填充
        if bias:
            with torch.no_grad():
                self.bias.fill_(0)
        # 初始化属性值
        self.init = init
        self.init_fn = init_fn

        # 如果初始化不在指定的列表中，则抛出异常
        if init not in ["default", "relu", "glorot", "gating", "normal", "final"]:
            raise ValueError("Invalid init string.")
# 定义了一个名为 EsmFoldLayerNorm 的类，继承自 nn.Module 类
class EsmFoldLayerNorm(nn.Module):
    # 初始化方法，接受输入维度 c_in 和 eps（epsilon）
    def __init__(self, c_in, eps=1e-5):
        super().__init__()  # 调用父类的初始化方法

        # 将输入维度和 epsilon 赋值给对象的属性
        self.c_in = (c_in,)
        self.eps = eps

        # 初始化可学习参数 weight 和 bias，分别为尺度和偏置，均为可训练参数
        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 获取输入张量的数据类型
        d = x.dtype
        # 如果数据类型是 torch.bfloat16 并且没有初始化 deepspeed，则执行以下语句
        if d is torch.bfloat16 and not is_deepspeed_initialized():
            # 关闭自动混合精度，并执行 layer_norm 操作
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(x, self.c_in, self.weight.to(dtype=d), self.bias.to(dtype=d), self.eps)
        else:  # 否则执行以下语句
            # 执行 layer_norm 操作
            out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)

        # 返回处理后的张量
        return out

# 将 softmax 操作包装在一个忽略 Torch JIT 的函数中
@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of type bfloat16
    """
    # 获取输入张量的数据类型
    d = t.dtype
    # 如果数据类型是 torch.bfloat16 并且没有初始化 deepspeed，则执行以下语句
    if d is torch.bfloat16 and not is_deepspeed_initialized():
        # 关闭自动混合精度，并执行 softmax 操作
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:  # 否则执行以下语句
        # 执行 softmax 操作
        s = torch.nn.functional.softmax(t, dim=dim)

    # 返回处理后的张量
    return s

# 定义了一个名为 EsmFoldAttention 的类
class EsmFoldAttention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer initialization. Allows multiple bias vectors.
    """

    # 初始化方法，接受输入维度和参数设置
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super().__init__()  # 调用父类的初始化方法

        # 将输入维度和参数赋值给对象的属性
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # 初始化线性变换层，用于将输入维度变换到隐藏维度
        self.linear_q = EsmFoldLinear(self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_k = EsmFoldLinear(self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_v = EsmFoldLinear(self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot")
        self.linear_o = EsmFoldLinear(self.c_hidden * self.no_heads, self.c_q, init="final")

        # 如果需要 gating，则初始化 gating 层
        self.linear_g = None
        if self.gating:
            self.linear_g = EsmFoldLinear(self.c_q, self.c_hidden * self.no_heads, init="gating")

        # 初始化 sigmoid 函数
        self.sigmoid = nn.Sigmoid()
    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 使用全连接层对查询张量进行线性变换
        q = self.linear_q(q_x)
        # 使用全连接层对键值张量进行线性变换
        k = self.linear_k(kv_x)
        # 使用全连接层对值张量进行线性变换
        v = self.linear_v(kv_x)

        # 将查询张量重塑为多头注意力机制所需的形状
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        # 将键值张量重塑为多头注意力机制所需的形状
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        # 将值张量重塑为多头注意力机制所需的形状
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # 对查询张量的形状进行转置
        q = q.transpose(-2, -3)
        # 对键值张量的形状进行转置
        k = k.transpose(-2, -3)
        # 对值张量的形状进行转置
        v = v.transpose(-2, -3)

        # 对查询张量进行数值缩放
        q /= math.sqrt(self.c_hidden)

        # 返回查询、键值、值张量
        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        # 如果存在全连接层g，则对查询张量进行线性变换和激活函数处理
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # 将g张量重塑为多头注意力机制所需的形状
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            # 对输出张量进行加权处理
            o = o * g

        # 对输出张量进行形状变换
        o = flatten_final_dims(o, 2)

        # 对输出张量进行线性变换
        o = self.linear_o(o)

        # 返回处理后的输出张量
        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = 1024,
        lma_kv_chunk_size: int = 4096,
        use_flash: bool = False,
        flash_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] 查询数据（query data）
            kv_x:
                [*, K, C_k] 键值数据（key data）
            biases:
                广播到 [*, H, Q, K] 的偏置列表（List of biases that broadcast to [*, H, Q, K]）
            use_memory_efficient_kernel:
                是否使用自定义的内存高效注意力核。对大多数情况来说，这应该是默认选择。
                如果没有任何 "use_<...>" 标志为 True，则使用标准的 PyTorch 实现（Whether to use a custom memory-efficient attention kernel. This should be the default choice for most.
                If none of the "use_<...>" flags are True, a stock PyTorch implementation is used instead）
            use_lma:
                是否使用低内存注意力（Staats & Rabe 2021）。如果没有任何 "use_<...>" 标志为 True，则使用标准的 PyTorch 实现（Whether to use low-memory attention (Staats & Rabe 2021). If none of the "use_<...>" flags are True, a
                stock PyTorch implementation is used instead）
            lma_q_chunk_size:
                查询分块大小（用于 LMA）（Query chunk size (for LMA)）
            lma_kv_chunk_size:
                键/值分块大小（用于 LMA）（Key/Value chunk size (for LMA)）
        Returns
            [*, Q, C_q] 注意力更新（attention update）
        """
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError("如果指定了 use_lma，则必须提供 lma_q_chunk_size 和 lma_kv_chunk_size")

        if use_flash and biases is not None:
            raise ValueError("use_flash 与偏置选项不兼容。对于掩码，请改用 flash_mask")

        attn_options = [use_memory_efficient_kernel, use_lma, use_flash]
        if sum(attn_options) > 1:
            raise ValueError("最多选择一种替代注意力算法")

        if biases is None:
            biases = []

        # [*, H, Q/K, C_hidden]
        # 准备查询、键、值
        query, key, value = self._prep_qkv(q_x, kv_x)
        # 将键（key）的维度置换为 (1, 0)
        key = permute_final_dims(key, (1, 0))

        # [*, H, Q, K]
        # 执行矩阵乘法：query 和 key
        output = torch.matmul(query, key)
        # 将每个偏置 b 添加到输出中
        for b in biases:
            output += b
        # 在最后一个维度上进行 softmax 操作
        output = softmax_no_cast(output, -1)

        # [*, H, Q, C_hidden]
        # 执行矩阵乘法：output 和 value
        output = torch.matmul(output, value)
        # 将倒数第二维和倒数第三维交换
        output = output.transpose(-2, -3)
        # 完成后续处理
        output = self._wrap_up(output, q_x)

        return output
class EsmFoldTriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                输入通道维度
            c_hidden:
                总的隐藏通道维度（不是每个头的）
            no_heads:
                注意力头的数量
        """
        super().__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)  # 初始化LayerNorm层

        self.linear = EsmFoldLinear(c_in, self.no_heads, bias=False, init="normal")  # 初始化EsmFoldLinear层

        self.mha = EsmFoldAttention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)  # 初始化EsmFoldAttention层

    @torch.jit.ignore
    def _chunk(
        self,
        x: torch.Tensor,
        biases: List[torch.Tensor],
        chunk_size: int,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        # 传入多头注意力的输入参数
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        return chunk_layer(
            partial(self.mha, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        if mask is None:
            # 如果掩码为空，则创建一个全为1的掩码张量，形状与 x 的前 N-1 维相同
            # [*, I, J]
            mask = x.new_ones(
                x.shape[:-1],
            )

        if not self.starting:
            # 如果不是起始状态，则交换 x 的倒数第二和倒数第三个维度
            x = x.transpose(-2, -3)
            # 交换掩码的最后两个维度
            mask = mask.transpose(-1, -2)

        # 对 x 进行 Layer Normalization，最后一维度为 C_in
        # [*, I, J, C_in]
        x = self.layer_norm(x)

        # 根据掩码计算掩码偏置，将掩码转换为广播的形式
        # [*, I, 1, 1, J]
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # 将 x 经过一个线性层，并对最后一个维度进行置换，得到三角形偏置
        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # 在倒数第二个维度上增加一个维度，用于后续计算
        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        # 将偏置列表初始化为掩码偏置和三角形偏置
        biases = [mask_bias, triangle_bias]

        if chunk_size is not None:
            # 如果指定了 chunk_size，则对输入进行分块处理
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            # 否则直接调用多头注意力机制处理输入
            x = self.mha(
                q_x=x, kv_x=x, biases=biases, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma
            )

        if not self.starting:
            # 如果不是起始状态，则恢复 x 的原始形状
            x = x.transpose(-2, -3)

        return x
class EsmFoldTriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """

    def __init__(self, config, _outgoing=True):
        # 调用父类构造函数
        super().__init__()
        # 从配置中获取隐藏状态的维度
        c_hidden = config.pairwise_state_dim
        # 是否是传出连接
        self._outgoing = _outgoing

        # 创建线性层对象，用于处理 a 和 b
        self.linear_a_p = EsmFoldLinear(c_hidden, c_hidden)
        self.linear_a_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        self.linear_b_p = EsmFoldLinear(c_hidden, c_hidden)
        self.linear_b_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        # 创建用于处理门控信号的线性层对象
        self.linear_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        # 创建最终输出的线性层对象
        self.linear_z = EsmFoldLinear(c_hidden, c_hidden, init="final")

        # 创建输入层归一化对象和输出层归一化对象
        self.layer_norm_in = LayerNorm(c_hidden)
        self.layer_norm_out = LayerNorm(c_hidden)

        # 创建 Sigmoid 激活函数对象
        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self, a: torch.Tensor, b: torch.Tensor, _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # 如果是传出连接，则重新排列张量维度
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        # 如果指定了分块大小，则采用分块矩阵乘法
        if _inplace_chunk_size is not None:
            # 循环进行分块矩阵乘法
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                # 将结果存回 a 张量
                a[..., i : i + _inplace_chunk_size, :, :] = torch.matmul(
                    a_chunk,
                    b_chunk,
                )

            # 令 p 为 a
            p = a
        else:
            # 直接进行矩阵乘法
            p = torch.matmul(a, b)

        # 恢复张量维度
        return permute_final_dims(p, (1, 2, 0))

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        # 未实现的推断前向传播方法
        pass

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ):
        # 未实现的正向传播方法
        pass
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor，输入张量，形状为[*, N_res, N_res, C_z]
            mask:
                [*, N_res, N_res] input mask，输入的掩码张量，形状为[*, N_res, N_res]
        Returns:
            [*, N_res, N_res, C_z] output tensor，输出张量，形状为[*, N_res, N_res, C_z]
        """
        # 如果启用了原位操作的安全模式
        if inplace_safe:
            # 调用内部_forward方法进行推断，传入z、mask，以及其他参数
            x = self._inference_forward(
                z,
                mask,
                inplace_chunk_size=_inplace_chunk_size,  # 使用的原位操作块大小
                with_add=_add_with_inplace,  # 是否与原位操作结合
            )
            # 返回结果张量
            return x

        # 如果未提供掩码，则创建全1的掩码
        if mask is None:
            mask = z.new_ones(z.shape[:-1])

        # 将掩码张量增加一个维度
        mask = mask.unsqueeze(-1)

        # 对输入进行层归一化
        z = self.layer_norm_in(z)
        # 计算第一组注意力系数a
        a = mask
        a = a * self.sigmoid(self.linear_a_g(z))  # 计算gating值，将其应用到a上
        a = a * self.linear_a_p(z)  # 应用线性投影到a上
        # 计算第二组注意力系数b
        b = mask
        b = b * self.sigmoid(self.linear_b_g(z))  # 计算gating值，将其应用到b上
        b = b * self.linear_b_p(z)  # 应用线性投影到b上

        # 如果启用了混合精度训练
        if is_fp16_enabled():
            # 使用混合精度进行计算
            with torch.cuda.amp.autocast(enabled=False):
                # 将a和b组合为输出张量x
                x = self._combine_projections(a.float(), b.float())
        else:
            # 将a和b组合为输出张量x
            x = self._combine_projections(a, b)

        # 释放a和b的内存
        del a, b
        # 对输出进行层归一化
        x = self.layer_norm_out(x)
        # 应用线性投影到输出张量上
        x = self.linear_z(x)
        # 计算gating值
        g = self.sigmoid(self.linear_g(z))
        # 将输出张量与gating值相乘
        x = x * g

        # 返回输出张量
        return x
# 创建一个名为EsmFoldPreTrainedModel的类，继承自EsmPreTrainedModel，用于处理权重初始化和预训练模型的下载和加载
class EsmFoldPreTrainedModel(EsmPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 对特殊的初始化进行处理的`EsMPreTrainedModel`的子类
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, EsmFoldLinear):
            with torch.no_grad():
                if module.init_fn is not None:
                    module.init_fn(module.weight, module.bias)
                elif module.init == "default":
                    trunc_normal_init_(module.weight, scale=1.0)
                elif module.init == "relu":
                    trunc_normal_init_(module.weight, scale=2.0)
                elif module.init == "glorot":
                    nn.init.xavier_uniform_(module.weight, gain=1)
                elif module.init == "gating":
                    module.weight.fill_(0.0)
                    if module.bias:
                        module.bias.fill_(1.0)
                elif module.init == "normal":
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                elif module.init == "final":
                    module.weight.fill_(0.0)
        elif isinstance(module, EsmFoldInvariantPointAttention):
            ipa_point_weights_init_(module.head_weights)
        elif isinstance(module, EsmFoldTriangularSelfAttentionBlock):
            torch.nn.init.zeros_(module.tri_mul_in.linear_z.weight)
            torch.nn.init.zeros_(module.tri_mul_in.linear_z.bias)
            torch.nn.init.zeros_(module.tri_mul_out.linear_z.weight)
            torch.nn.init.zeros_(module.tri_mul_out.linear_z.bias)
            torch.nn.init.zeros_(module.tri_att_start.mha.linear_o.weight)
            torch.nn.init.zeros_(module.tri_att_start.mha.linear_o.bias)
            torch.nn.init.zeros_(module.tri_att_end.mha.linear_o.weight)
            torch.nn.init.zeros_(module.tri_att_end.mha.linear_o.bias)

            torch.nn.init.zeros_(module.sequence_to_pair.o_proj.weight)
            torch.nn.init.zeros_(module.sequence_to_pair.o_proj.bias)
            torch.nn.init.zeros_(module.pair_to_sequence.linear.weight)
            torch.nn.init.zeros_(module.seq_attention.o_proj.weight)
            torch.nn.init.zeros_(module.seq_attention.o_proj.bias)
            torch.nn.init.zeros_(module.mlp_seq.mlp[-2].weight)
            torch.nn.init.zeros_(module.mlp_seq.mlp[-2].bias)
            torch.nn.init.zeros_(module.mlp_pair.mlp[-2].weight)
            torch.nn.init.zeros_(module.mlp_pair.mlp[-2].bias)
        else:
            super()._init_weights(module)


class EsmFoldSelfAttention(nn.Module):
    # 初始化方法，创建自注意力层对象
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        super().__init__()
        assert embed_dim == num_heads * head_width

        # 初始化实例变量
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_width = head_width

        # 创建输入到查询、键、值的线性映射函数
        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # 创建输出投影层线性函数
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated
        # 如果启用门控机制，创建门控投影层线性函数
        if gated:
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            # 初始化门控投影层的权重和偏置
            torch.nn.init.zeros_(self.g_proj.weight)
            torch.nn.init.ones_(self.g_proj.bias)

        # 计算缩放因子
        self.rescale_factor = self.head_width**-0.5

        # 初始化输出投影层的偏置
        torch.nn.init.zeros_(self.o_proj.bias)

    # 前向传播方法，实现基本的自注意力机制，可选地包含掩码和外部注意力偏置
    def forward(self, x, mask=None, bias=None, indices=None):
        """
        Basic self attention with optional mask and external pairwise bias. To handle sequences of different lengths,
        use mask.

        Inputs:
            x: batch of input sequneces (.. x L x C) mask: batch of boolean masks where 1=valid, 0=padding position (..
            x L_k) bias: batch of scalar pairwise attention biases (.. x Lq x Lk x num_heads)

        Outputs:
          sequence projection (B x L x embed_dim), attention maps (B x L x L x num_heads)
        """

        # 将输入通过线性映射函数进行投影，并将结果重塑为 (B x L x num_heads x head_width*3) 的张量
        t = self.proj(x).view(*x.shape[:2], self.num_heads, -1)
        # 将张量维度重新排列为 (B x num_heads x L x head_width*3)
        t = t.permute(0, 2, 1, 3)
        # 拆分投影后的张量，得到查询、键、值
        q, k, v = t.chunk(3, dim=-1)

        # 缩放查询向量
        q = self.rescale_factor * q
        # 计算注意力分数
        a = torch.einsum("...qc,...kc->...qk", q, k)

        # 添加外部注意力偏置
        if bias is not None:
            a = a + bias.permute(0, 3, 1, 2)

        # 掩盖填充位置
        if mask is not None:
            mask = mask[:, None, None]
            a = a.masked_fill(mask == False, -np.inf)  # noqa: E712

        # 使用 softmax 函数计算注意力权重
        a = nn.functional.softmax(a, dim=-1)

        # 根据注意力权重计算加权后的值向量
        y = torch.einsum("...hqk,...hkc->...qhc", a, v)
        # 将结果张量重塑为 (B x L x embed_dim)
        y = y.reshape(*y.shape[:2], -1)

        # 如果启用门控机制，对结果应用门控
        if self.gated:
            y = self.g_proj(x).sigmoid() * y
        # 通过输出投影层得到最终输出
        y = self.o_proj(y)

        return y, a.permute(0, 3, 1, 2)
# 用于实现沿特定维度共享 dropout 掩模的 dropout
class EsmFoldDropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask along a particular dimension.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        super().__init__()
        
        # 初始化 dropout 概率
        self.r = r 
        # 如果 batch_dim 是整数，则转换为列表
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim
        # 创建一个 Dropout 层
        self.dropout = nn.Dropout(self.r)

    # 前向传播函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的形状
        shape = list(x.shape)
        # 如果 batch_dim 不为空，则将指定维度的大小设为1
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        # 返回带有 dropout 的输入张量
        return x * self.dropout(x.new_ones(shape))


# 将序列状态转换成成对状态的模块
class EsmFoldSequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super().__init__()

        # 初始化层归一化层和线性投影层
        self.layernorm = nn.LayerNorm(sequence_state_dim)
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)
        
        # 初始化偏置为0
        torch.nn.init.zeros_(self.proj.bias)
        torch.nn.init.zeros_(self.o_proj.bias)

    # 前向传播函数
    def forward(self, sequence_state):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim

        Output:
          pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """

        assert len(sequence_state.shape) == 3

        # 序列状态进行层归一化
        s = self.layernorm(sequence_state)
        # 线性投影
        s = self.proj(s)
        q, k = s.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]

        # 拼接张量
        x = torch.cat([prod, diff], dim=-1)
        x = self.o_proj(x)

        return x


# 将成对状态转换成序列状态的模块
class EsmFoldPairToSequence(nn.Module):
    def __init__(self, pairwise_state_dim, num_heads):
        super().__init__()

        # 初始化层归一化层和线性层
        self.layernorm = nn.LayerNorm(pairwise_state_dim)
        self.linear = nn.Linear(pairwise_state_dim, num_heads, bias=False)

    # 前向传播函数
    def forward(self, pairwise_state):
        """
        Inputs:
          pairwise_state: B x L x L x pairwise_state_dim

        Output:
          pairwise_bias: B x L x L x num_heads
        """
        assert len(pairwise_state.shape) == 4
        # 序列状态进行层归一化
        z = self.layernorm(pairwise_state)
        # 线性变换
        pairwise_bias = self.linear(z)
        return pairwise_bias


# 残差连接的多层感知机
class EsmFoldResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim, dropout=0):
        super().__init__()

        # 初始化多层感知机
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, embed_dim),
            nn.Dropout(dropout),
        )

    # 前向传播函数
    def forward(self, x):
        return x + self.mlp(x)


# 三角形自注意力模块
class EsmFoldTriangularSelfAttentionBlock(nn.Module):
    # 初始化函数，接受配置参数 config
    def __init__(self, config):
        # 调用父类构造函数初始化
        super().__init__()
        # 将配置参数保存在对象中
        self.config = config

        # 根据配置参数计算序列状态维度和成对状态维度
        sequence_state_dim = config.sequence_state_dim
        pairwise_state_dim = config.pairwise_state_dim
        sequence_num_heads = sequence_state_dim // config.sequence_head_width
        pairwise_num_heads = pairwise_state_dim // config.pairwise_head_width

        # 初始化 LayerNorm 层对象，并使用序列状态维度作为参数
        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        # 初始化序列到成对映射层对象，传入序列状态维度、成对状态维度的一半、成对状态维度作为参数
        self.sequence_to_pair = EsmFoldSequenceToPair(sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim)
        # 初始化成对到序列映射层对象，传入成对状态维度、序列头数作为参数
        self.pair_to_sequence = EsmFoldPairToSequence(pairwise_state_dim, sequence_num_heads)

        # 初始化序列自注意力层对象，传入序列状态维度、序列头数、序列头宽度、是否门控作为参数
        self.seq_attention = EsmFoldSelfAttention(
            sequence_state_dim, sequence_num_heads, config.sequence_head_width, gated=True
        )
        # 初始化三角形乘法更新层（出向），传入配置参数、是否出向更新作为参数
        self.tri_mul_out = EsmFoldTriangleMultiplicativeUpdate(config, _outgoing=True)
        # 初始化三角形乘法更新层（入向），传入配置参数、是否出向更新作为参数
        self.tri_mul_in = EsmFoldTriangleMultiplicativeUpdate(config, _outgoing=False)

        # 初始化三角形开始关注层，传入成对状态维度、成对头宽度、成对头数、无穷数值、是否开始作为参数
        self.tri_att_start = EsmFoldTriangleAttention(
            pairwise_state_dim, config.pairwise_head_width, pairwise_num_heads, inf=1e9, starting=True
        )
        # 初始化三角形结束关注层，传入成对状态维度、成对头宽度、成对头数、无穷数值、是否开始作为参数
        self.tri_att_end = EsmFoldTriangleAttention(
            pairwise_state_dim, config.pairwise_head_width, pairwise_num_heads, inf=1e9, starting=False
        )

        # 初始化基于残差连接的 MLP 层（序列），传入序列状态维度、4倍序列状态维度、dropout值作为参数
        self.mlp_seq = EsmFoldResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=config.dropout)
        # 初始化基于残差连接的 MLP 层（成对），传入成对状态维度、4倍成对状态维度、dropout值作为参数
        self.mlp_pair = EsmFoldResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=config.dropout)

        # 初始化 Dropout 层，传入dropout值作为参数
        self.drop = nn.Dropout(config.dropout)
        # 初始化 EsmFoldDropout 层（行），传入dropout值的两倍、2作为参数
        self.row_drop = EsmFoldDropout(config.dropout * 2, 2)
        # 初始化 EsmFoldDropout 层（列），传入dropout值的两倍、1作为参数
        self.col_drop = EsmFoldDropout(config.dropout * 2, 1)
class EsmCategoricalMixture:
    def __init__(self, param, bins=50, start=0, end=1):
        # 所有张量的形状都是..., bins.
        self.logits = param
        # 创建一个从 start 到 end 等间距取 bins + 1 个数的张量，device 和 dtype 与 self.logits 对应
        bins = torch.linspace(start, end, bins + 1, device=self.logits.device, dtype=self.logits.dtype)
        # 计算每一段的中点，存储在 self.v_bins 中
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # 形状分别是:
        #     self.probs: ... x bins
        #     true      : ...
        # 找到 true 在 self.v_bins 中最接近的索引
        true_index = (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        # 计算加权平均值，self.logits.softmax(-1) @ self.v_bins 的结果形状是... x 1
        return (self.logits.softmax(-1) @ self.v_bins.unsqueeze(1)).squeeze(-1)


def categorical_lddt(logits, bins=50):
    # logits 形状是..., 37, bins.
    return EsmCategoricalMixture(logits, bins=bins).mean()


def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used in row column attentions.

    Input:
      mask: B x L tensor of booleans

    Output:
      mask: B x L x L tensor of booleans
    """
    # 如果 mask 为空，则返回空值
    if mask is None:
        return None
    # 若 mask 的维度不是 2，则抛出异常
    if len(mask.shape) != 2:
        raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")
    batch_dim, seq_dim = mask.shape
    # 在第二个维度上扩展 mask，得到一个 B x L x L 的张量
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m


class EsmFoldRelativePosition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bins = config.position_bins

        # 注意，添加了额外的偏移，以便为屏蔽的配对保留第 0 位置。
        self.embedding = torch.nn.Embedding(2 * self.bins + 2, config.pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long) mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """
        # 如果 residue_index 的类型不是 torch.long，则抛出异常
        if residue_index.dtype != torch.long:
            raise ValueError(f"`residue_index` has dtype {residue_index.dtype}, it should be `torch.long`.")
        # 如果 mask 不为空且 residue_index 和 mask 的形状不一致，则抛出异常
        if mask is not None and residue_index.shape != mask.shape:
            raise ValueError(
                f"`residue_index` and `mask` have inconsistent shapes: {residue_index.shape} != {mask.shape}."
            )

        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # 添加 1 来调整填充索引。

        if mask is not None:
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0  # noqa: E712

        output = self.embedding(diff)
        return output


class EsmFoldAngleResnetBlock(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
    
        # 创建线性层对象，输入维度和输出维度都是配置中的 resnet_dim，初始化方式为“relu”
        self.linear_1 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="relu")
        # 创建线性层对象，输入维度和输出维度都是配置中的 resnet_dim，初始化方式为“final”
        self.linear_2 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="final")
    
        # 创建 ReLU 激活函数对象
        self.relu = nn.ReLU()
    
    # 前向传播方法，接受一个输入张量并返回一个张量
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # 保存输入张量的初始值
        s_initial = a
    
        # 对输入张量应用 ReLU 激活函数
        a = self.relu(a)
        # 使用线性层对象 1 处理输入张量
        a = self.linear_1(a)
        # 对处理后的张量再次应用 ReLU 激活函数
        a = self.relu(a)
        # 使用线性层对象 2 处理处理后的张量
        a = self.linear_2(a)
    
        # 返回处理后的张量与初始输入张量的和
        return a + s_initial
# 定义一个名为 EsmFoldAngleResnet 的类，继承自 nn.Module
# 实现 Algorithm 20 中 11-14 行的算法
class EsmFoldAngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用 nn.Module 的初始化方法
        super().__init__()
        # 保存传入的配置参数
        self.config = config

        # 初始化线性层，输入维度为 config.sequence_dim，输出维度为 config.resnet_dim
        self.linear_in = EsmFoldLinear(config.sequence_dim, config.resnet_dim)
        self.linear_initial = EsmFoldLinear(config.sequence_dim, config.resnet_dim)

        # 创建一个包含多个 EsmFoldAngleResnetBlock 实例的模块列表
        self.layers = nn.ModuleList()
        for _ in range(config.num_resnet_blocks):
            layer = EsmFoldAngleResnetBlock(config)
            self.layers.append(layer)

        # 初始化输出层，输入维度为 config.resnet_dim，输出维度为 config.num_angles * 2
        self.linear_out = EsmFoldLinear(config.resnet_dim, config.num_angles * 2)

        # 创建一个 ReLU 激活函数实例
        self.relu = nn.ReLU()

    # 前向传播方法，接受 s 和 s_initial 两个参数，并返回两个 torch.Tensor 类型的值
    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # 注释开始
        # 注意：输入的 ReLU 对于这里的输入在补充中没有提到，但在源代码中有。为了与预训练权重的最大兼容性，我将使用源代码中的方式。
        
        # 对 s_initial 应用 ReLU 激活函数
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        # 对 s 应用 ReLU 激活函数
        s = self.relu(s)
        s = self.linear_in(s)
        # 将 s 与 s_initial 相加
        s = s + s_initial

        # 对每个层进行前向传播
        for l in self.layers:
            s = l(s)

        # 再次对 s 应用 ReLU 激活函数
        s = self.relu(s)

        # 对 s 应用线性层得到预测的角度值
        s = self.linear_out(s)

        # 重新调整 s 的形状
        s = s.view(s.shape[:-1] + (-1, 2))

        # 计算 s 的归一化值
        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.config.epsilon,
            )
        )
        s = s / norm_denom

        # 返回未归一化的 s 和归一化后的 s
        return unnormalized_s, s


# 定义一个名为 EsmFoldInvariantPointAttention 的类，实现 Algorithm 22
class EsmFoldInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    # 初始化函数，用于初始化 IPA 模型的参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置信息保存到对象的属性中
        self.config = config
    
        # 获取配置中的维度信息
        c_s = config.sequence_dim
        c_z = config.pairwise_dim
        self.hidden_dim = config.ipa_dim
        self.num_heads = config.num_heads_ipa
        self.num_qk_points = config.num_qk_points
        self.num_v_points = config.num_v_points
    
        # 下面的线性层与补充材料中的规格不同。
        # 在补充材料中，它们没有偏置，并使用 Glorot 初始化。
        # 而在官方源代码中，它们有偏置并使用默认的 Lecun 初始化。
        # 这里与官方源代码保持一致，使用偏置和默认的 Lecun 初始化。
        hc = config.ipa_dim * config.num_heads_ipa
        # 创建线性层，用于计算查询向量
        self.linear_q = EsmFoldLinear(c_s, hc)
        # 创建线性层，用于计算键值向量
        self.linear_kv = EsmFoldLinear(c_s, 2 * hc)
    
        # 计算查询向量的位置编码
        hpq = config.num_heads_ipa * config.num_qk_points * 3
        self.linear_q_points = EsmFoldLinear(c_s, hpq)
    
        # 计算键值向量的位置编码
        hpkv = config.num_heads_ipa * (config.num_qk_points + config.num_v_points) * 3
        self.linear_kv_points = EsmFoldLinear(c_s, hpkv)
    
        # 创建线性层，用于计算 IPA 模型中的偏置项
        self.linear_b = EsmFoldLinear(c_z, config.num_heads_ipa)
    
        # 创建用于存储头部权重的可学习参数
        self.head_weights = nn.Parameter(torch.zeros((config.num_heads_ipa)))
    
        # 计算拼接后的输出维度
        concat_out_dim = config.num_heads_ipa * (c_z + config.ipa_dim + config.num_v_points * 4)
        # 创建线性层，用于最终的输出
        self.linear_out = EsmFoldLinear(concat_out_dim, c_s, init="final")
    
        # 创建 Softmax 函数，用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)
        # 创建 Softplus 函数，用于激活函数
        self.softplus = nn.Softplus()
    
    # 前向传播函数，用于计算 IPA 模型的前向传播过程
    def forward(
        self,
        s: torch.Tensor,
        z: Optional[torch.Tensor],
        r: Rigid,
        mask: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
class EsmFoldBackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, config):
        super().__init__()

        self.linear = EsmFoldLinear(config.sequence_dim, 6, init="final")
        # 初始化线性层，输入维度为config.sequence_dim，输出维度为6，使用"final"初始化法则

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)
        # 使用线性层处理输入数据s，得到update向量

        return update


class EsmFoldStructureModuleTransitionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear_1 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        # 初始化第一个线性层，输入输出维度都为config.sequence_dim，使用"relu"初始化法则
        self.linear_2 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        # 初始化第二个线性层，输入输出维度都为config.sequence_dim，使用"relu"初始化法则
        self.linear_3 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="final")
        # 初始化第三个线性层，输入输出维度都为config.sequence_dim，使用"final"初始化法则

        self.relu = nn.ReLU()
        # 初始化ReLU激活函数

    def forward(self, s):
        s_initial = s
        # 保存初始的输入数据s
        s = self.linear_1(s)
        s = self.relu(s)
        # 使用第一个线性层处理s，再使用ReLU激活函数激活
        s = self.linear_2(s)
        s = self.relu(s)
        # 使用第二个线性层处理s，再使用ReLU激活函数激活
        s = self.linear_3(s)
        # 使用第三个线性层处理s

        s = s + s_initial
        # 将处理后的s和初始的s相加

        return s


class EsmFoldStructureModuleTransition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList()
        # 初始化一个空的ModuleList

        for _ in range(config.num_transition_layers):
            l = EsmFoldStructureModuleTransitionLayer(config)
            # 根据config.num_transition_layers的数量，循环初始化EsmFoldStructureModuleTransitionLayer
            self.layers.append(l)
            # 将初始化的EsmFoldStructureModuleTransitionLayer添加到ModuleList中

        self.dropout = nn.Dropout(config.dropout_rate)
        # 初始化一个Dropout层，根据config中的dropout_rate
        self.layer_norm = LayerNorm(config.sequence_dim)
        # 根据config中的sequence_dim初始化LayerNorm

    def forward(self, s):
        for l in self.layers:
            s = l(s)
            # 对s进行多个EsmFoldStructureModuleTransitionLayer的处理，形成一个前向传播的过程

        s = self.dropout(s)
        # 对s进行dropout操作
        s = self.layer_norm(s)
        # 对s进行LayerNorm操作

        return s


class EsmFoldStructureModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Buffers to be lazily initialized later
        # self.default_frames
        # self.group_idx
        # self.atom_mask
        # self.lit_positions
        # 初始化一些缓冲区，稍后才能懒初始化

        self.layer_norm_s = LayerNorm(config.sequence_dim)
        # 根据config中的sequence_dim初始化LayerNorm
        self.layer_norm_z = LayerNorm(config.pairwise_dim)
        # 根据config中的pairwise_dim初始化LayerNorm

        self.linear_in = EsmFoldLinear(config.sequence_dim, config.sequence_dim)
        # 初始化线性层，输入输出维度都为config.sequence_dim

        self.ipa = EsmFoldInvariantPointAttention(config)
        # 初始化EsmFoldInvariantPointAttention

        self.ipa_dropout = nn.Dropout(config.dropout_rate)
        # 初始化一个Dropout层，根据config中的dropout_rate
        self.layer_norm_ipa = LayerNorm(config.sequence_dim)
        # 根据config中的sequence_dim初始化LayerNorm

        self.transition = EsmFoldStructureModuleTransition(config)
        # 初始化EsmFoldStructureModuleTransition

        self.bb_update = EsmFoldBackboneUpdate(config)
        # 初始化EsmFoldBackboneUpdate
        self.angle_resnet = EsmFoldAngleResnet(config)
        # 初始化EsmFoldAngleResnet

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        _offload_inference=False,
        # 前向传播函数，接受多个参数，包括evoformer_output_dict, aatype, mask和_offload_inference
    # 初始化残基常量，根据浮点数据类型和设备
    def _init_residue_constants(self, float_dtype, device):
        # 如果实例中没有默认帧属性，则注册默认帧属性
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    residue_constants.restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        # 如果实例中没有团索引属性，则注册团索引属性
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    residue_constants.restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        # 如果实例中没有原子掩码属性，则注册原子掩码属性
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    residue_constants.restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        # 如果实例中没有文献位置属性，则注册文献位置属性
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    residue_constants.restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
    
    # 将扭转角转换为帧
    def torsion_angles_to_frames(self, r, alpha, f):
        # 惰性地在正确的设备上初始化残基常量
        self._init_residue_constants(alpha.dtype, alpha.device)
        # 用于简化测试
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)
    
    # 将帧和文献位置转换为原子14位置
    def frames_and_literature_positions_to_atom14_pos(self, r, f):
        # 惰性地在正确的设备上初始化残基常量
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
class EsmFoldingTrunk(nn.Module):
    def __init__(self, config):
        # 继承父类构造函数
        super().__init__()
        # 保存配置参数
        self.config = config

        # 从配置中获取序列状态维度和成对状态维度
        c_s = config.sequence_state_dim
        c_z = config.pairwise_state_dim

        # 初始化成对位置嵌入层
        self.pairwise_positional_embedding = EsmFoldRelativePosition(config)

        # 创建多个 EsmFoldTriangularSelfAttentionBlock 组成的模块列表
        self.blocks = nn.ModuleList([EsmFoldTriangularSelfAttentionBlock(config) for _ in range(config.num_blocks)])

        # 初始化循环利用参数
        self.recycle_bins = 15
        # 序列状态的归一化层
        self.recycle_s_norm = nn.LayerNorm(c_s)
        # 成对状态的归一化层
        self.recycle_z_norm = nn.LayerNorm(c_z)
        # 循环利用距离的嵌入层
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        # 将循环利用距离的嵌入参数的第一个权重置零
        self.recycle_disto.weight[0].detach().zero_()

        # 初始化 EsmFoldStructureModule 结构模块
        self.structure_module = EsmFoldStructureModule(config.structure_module)
        # 序列状态映射到结构模块序列维度的线性层
        self.trunk2sm_s = nn.Linear(c_s, config.structure_module.sequence_dim)
        # 成对状态映射到结构模块成对维度的线性层
        self.trunk2sm_z = nn.Linear(c_z, config.structure_module.pairwise_dim)

        # 设置块大小参数
        self.chunk_size = config.chunk_size

    def set_chunk_size(self, chunk_size):
        # 设置块大小参数，影响分块计算的轴向注意力，可以降低内存使用
        self.chunk_size = chunk_size
    # 前向传播函数，接受序列特征、成对特征、真实氨基酸、残基索引、掩码和不透过的循环次数作为输入
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        """
        Inputs:
          seq_feats: B x L x C tensor of sequence features pair_feats: B x L x L x C tensor of pair features residx: B
          x L long tensor giving the position in the sequence mask: B x L boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """

        # 获取计算设备
        device = seq_feats.device
        # 初始序列特征和成对特征
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        # 如果没有给定不透过的循环次数，则设为最大循环次数
        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            # 如果循环次数为负数，引发异常
            if no_recycles < 0:
                raise ValueError("Number of recycles must not be negative.")
            # 增加循环次数1，第一个“循环”仅为模型的标准前向传递
            no_recycles += 1  

        # 定义块迭代函数
        def trunk_iter(s, z, residx, mask):
            # 在成对特征上加入位置编码
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            # 遍历所有块
            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        # 初始化变量
        s_s = s_s_0
        s_z = s_z_0
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        # 循环执行指定次数的循环
        for recycle_idx in range(no_recycles):
            # 使用ContextManagers包装，最后一个循环无需torch.no_grad()
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                # === 循环 ===
                recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
                recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
                recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                # === 结构模块 ===
                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa,
                    mask.float(),
                )

                recycle_s = s_s
                recycle_z = s_z
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    self.recycle_bins,
                )

        # 结构结果的保存和返回
        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure
    # 定义距离直方图函数，输入参数为坐标、最小区间、最大区间、区间数量
    def distogram(coords, min_bin, max_bin, num_bins):
        # Coords are [... L x 3 x 3], where it's [N, CA, C] x 3 coordinates.
        # 在给定最小值、最大值和区间数量的情况下，在给定设备上创建一组等间距的数值
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        # 对边界值进行平方
        boundaries = boundaries**2
        # 从坐标中分块提取 N、CA、C 坐标
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]
        # 推断 CB 坐标
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        # 通过线性组合计算 CB 坐标
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        # 计算 CB 之间的距离的平方
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)
        # 将距离平方与边界进行比较，以确定它们属于哪个区间
        bins = torch.sum(dists > boundaries, dim=-1)  # [..., L, L]
        # 返回距离直方图
        return bins
# TODO在文档字符串中添加关于任何转换为PDB格式或其他方式准备输出用于下游使用的方法的信息。
@add_start_docstrings(
    """
    ESMForProteinFolding是原始ESMFold模型的HuggingFace端口。它由一个ESM-2“干线”后面跟着一个蛋白质折叠“头部”组成，
    尽管与大多数其他输出头部不同，这个“头部”在大小和运行时与模型的其余部分相当类似！它输出一个包含有关输入蛋白质的预测结构信息的字典。
    """,
    ESM_START_DOCSTRING,
)
class EsmForProteinFolding(EsmPreTrainedModel):
    _no_split_modules = ["EsmFoldStructureModule", "EsmFoldTriangularSelfAttentionBlock"]
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 将配置参数存储到实例变量中
        self.config = config

        # 设定直方图的箱数
        self.distogram_bins = 64

        # 初始化 ESM 模型，不添加池化层
        self.esm = EsmModel(config, add_pooling_layer=False)

        # 设置 ESM 模型参数不可训练
        self.esm.requires_grad_(False)
        # 如果配置中指定使用 fp16 格式的 ESM 模型，则将模型转换为半精度
        if self.config.esmfold_config.fp16_esm:
            self.esm.half()

        # 设置 ESM 模型的特征维度
        self.esm_feats = self.config.hidden_size
        # 设置 ESM 模型的注意力头数
        self.esm_attns = self.config.num_hidden_layers * self.config.num_attention_heads
        # 设置 ESM 模型的层数
        self.esm_layers = self.config.num_hidden_layers

        # 注册缓冲区，存储从词汇列表到 ESM 输入的映射
        self.register_buffer("af2_to_esm", self._af2_to_esm_from_vocab_list(config.vocab_list))
        # 初始化参数，用于结合不同层的 ESM 输出
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm_layers + 1))

        # 设置 ESMFOLD 的 trunk 部分的配置
        trunk_config = self.config.esmfold_config.trunk
        # 获取序列状态维度和成对状态维度
        c_s = trunk_config.sequence_state_dim
        c_z = trunk_config.pairwise_state_dim

        # 构建用于处理 ESM 输出的 MLP 层
        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 计算嵌入的 token 数量，包括 padding、未知残基和 mask
        self.n_tokens_embed = residue_constants.restype_num + 3
        # 设置 padding 的索引为 0
        self.pad_idx = 0
        # 设置未知残基的索引
        self.unk_idx = self.n_tokens_embed - 2
        # 设置 mask 的索引
        self.mask_idx = self.n_tokens_embed - 1

        # 获取词汇列表中的特定标记的索引
        self.esm_dict_cls_idx = self.config.vocab_list.index("<cls>")
        self.esm_dict_mask_idx = self.config.vocab_list.index("<mask>")
        self.esm_dict_eos_idx = self.config.vocab_list.index("<eos>")
        self.esm_dict_padding_idx = self.config.vocab_list.index("<pad>")

        # 如果配置中指定了将氨基酸嵌入到模型中，则初始化嵌入层
        if self.config.esmfold_config.embed_aa:
            self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        # 初始化 ESMFOLD 的 trunk 部分
        self.trunk = EsmFoldingTrunk(trunk_config)

        # 创建直方图分布的输出层
        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        # 创建 PTM 预测的输出层
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        # 创建语言模型的输出层
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)
        
        # 设置 LDDT 预测的直方图分布的输出层
        self.lddt_bins = 50
        structure_module_config = trunk_config.structure_module
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(structure_module_config.sequence_dim),
            nn.Linear(structure_module_config.sequence_dim, self.config.esmfold_config.lddt_head_hid_dim),
            nn.Linear(self.config.esmfold_config.lddt_head_hid_dim, self.config.esmfold_config.lddt_head_hid_dim),
            nn.Linear(self.config.esmfold_config.lddt_head_hid_dim, 37 * self.lddt_bins),
        )

    @staticmethod
    # 根据词汇列表生成从 AF2 到 ESM 输入的映射
    def _af2_to_esm_from_vocab_list(vocab_list: List[str]) -> torch.Tensor:
        # 记住 t 是从 residue_constants 中偏移了 1（0 是 padding）。
        esm_reorder = [vocab_list.index("<pad>")] + [vocab_list.index(v) for v in residue_constants.restypes_with_x]
        return torch.tensor(esm_reorder)

    # 将输入文档字符串添加到模型前向方法的装饰器，用于提供输入格式的说明
    @add_start_docstrings_to_model_forward(ESMFOLD_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 将返回文档字符串的输出类型替换为 EsmForProteinFoldingOutput，配置类为 EsmConfig
    @replace_return_docstrings(output_type=EsmForProteinFoldingOutput, config_class=EsmConfig)
``` 
    # 定义了一个方法，用于前向传播模型
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        masking_pattern: Optional[torch.Tensor] = None,
        num_recycles: Optional[int] = None,
    # 将AF2序列索引映射到ESM序列索引的方法
    def af2_idx_to_esm_idx(self, aa, mask):
        # 避免在不同设备上进行索引
        if self.af2_to_esm.device != aa.device:
            self.af2_to_esm = self.af2_to_esm.to(aa.device)
        # 将aa加一并进行掩码填充
        aa = (aa + 1).masked_fill(mask != 1, 0)
        # 返回映射后的索引
        return self.af2_to_esm[aa]

    # 计算语言模型表示的方法
    def compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        B, L = esmaa.shape  # B = batch size, L = sequence length.

        # 如果配置为绕过语言模型，则返回全零张量
        if self.config.esmfold_config.bypass_lm:
            esm_s = torch.zeros(B, L, self.esm_s_combine.size[0], -1, self.esm_feats, device=device)
            return esm_s

        # 获取起始和结束标记的索引
        bosi, eosi = self.esm_dict_cls_idx, self.esm_dict_eos_idx
        bos = esmaa.new_full((B, 1), bosi)
        eos = esmaa.new_full((B, 1), self.esm_dict_padding_idx)
        # 在序列开头和结尾添加起始和结束标记
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # 在推断时使用第一个填充索引作为结束标记
        esmaa[range(B), (esmaa != 1).sum(1)] = eosi

        # 调用ESM模型获取隐藏状态
        esm_hidden_states = self.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)["hidden_states"]
        esm_s = torch.stack(esm_hidden_states, dim=2)

        # 去除起始和结束标记，保留中间部分的隐藏状态
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C

        return esm_s

    # 对输入进行BERT掩码处理的方法
    def bert_mask(self, aa, esmaa, mask, pattern):
        new_aa = aa.clone()
        target = aa.clone()
        new_esmaa = esmaa.clone()
        # 将模式中为1的位置替换为掩码索引
        new_aa[pattern == 1] = self.mask_idx
        # 将模式中不为1的位置替换为0
        target[pattern != 1] = 0
        # 将ESM序列中模式为1的位置替换为ESM掩码索引
        new_esmaa[pattern == 1] = self.esm_dict_mask_idx
        # 返回处理后的结果
        return new_aa, new_esmaa, target

    # 推断方法（不计算梯度）
    @torch.no_grad()
    def infer(
        self,
        seqs: Union[str, List[str]],
        position_ids=None,
    ): 
        # 如果输入的序列是字符串，则将其放入列表中
        if isinstance(seqs, str):
            lst = [seqs]
        else:
            lst = seqs
        # 给定一个输入序列，返回模型的原始输出
        device = next(self.parameters()).device
        # 将序列转换为 one-hot 编码的张量，并按维度取最大值
        aatype = collate_dense_tensors(
            [
                torch.from_numpy(
                    residue_constants.sequence_to_onehot(
                        sequence=seq,
                        mapping=residue_constants.restype_order_with_x,
                        map_unknown_to_x=True,
                    )
                )
                .to(device)
                .argmax(dim=1)
                for seq in lst
            ]
        )  # B=1 x L
        # 根据序列的长度生成掩码
        mask = collate_dense_tensors([aatype.new_ones(len(seq)) for seq in lst])
        # 生成位置 id
        position_ids = (
            torch.arange(aatype.shape[1], device=device).expand(len(lst), -1)
            if position_ids is None
            else position_ids.to(device)
        )
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        # 前向传播模型
        return self.forward(
            aatype,
            mask,
            position_ids=position_ids,
        )

    @staticmethod
    def output_to_pdb(output: Dict) -> List[str]:
        """从模型输出返回 PDB（文件）字符串。"""
        output = {k: v.to("cpu").numpy() for k, v in output.items()}
        pdbs = []
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        final_atom_mask = output["atom37_atom_exists"]
        for i in range(output["aatype"].shape[0]):
            aa = output["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = output["residue_index"][i] + 1
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=output["plddt"][i],
            )
            pdbs.append(to_pdb(pred))
        return pdbs

    def infer_pdb(self, seqs, *args, **kwargs) -> str:
        """给定输入序列，从模型返回 PDB（文件）字符串。"""
        assert isinstance(seqs, str)
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)[0]

    def infer_pdbs(self, seqs: List[str], *args, **kwargs) -> List[str]:
        """给定输入序列，从模型返回 PDB（文件）字符串列表。"""
        output = self.infer(seqs, *args, **kwargs)
        return self.output_to_pdb(output)
```