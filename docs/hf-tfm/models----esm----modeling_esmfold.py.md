# `.\models\esm\modeling_esmfold.py`

```py
# 设置编码格式为 UTF-8
# 版权声明和许可证信息，表明此代码的使用和分发需要遵循 Apache License, Version 2.0
# 导入必要的库和模块
import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
from torch.nn import LayerNorm  # 导入 PyTorch 的 LayerNorm 模块

# 导入相关的模块和函数，用于 DeepSpeed 集成、模型输出、文档字符串处理等
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
# 导入 ESM 模型的配置文件和模型定义
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
# 导入与蛋白质折叠相关的工具函数和类
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

# 获取日志记录器对象
logger = logging.get_logger(__name__)
# 文档中的模型检查点和配置信息
_CHECKPOINT_FOR_DOC = "facebook/esmfold_v1"
_CONFIG_FOR_DOC = "EsmConfig"

@dataclass
class EsmForProteinFoldingOutput(ModelOutput):
    """
    [`EsmForProteinFoldingOutput`] 的输出类型。
    """
    Args:
        frames (`torch.FloatTensor`):
            输出帧。
            模型预测的帧输出。
        sidechain_frames (`torch.FloatTensor`):
            侧链帧。
            模型预测的侧链帧输出。
        unnormalized_angles (`torch.FloatTensor`):
            预测的未归一化主链和侧链扭转角度。
            模型预测的未归一化主链和侧链扭转角度。
        angles (`torch.FloatTensor`):
            预测的主链和侧链扭转角度。
            模型预测的主链和侧链扭转角度。
        positions (`torch.FloatTensor`):
            预测的主链和侧链原子的位置。
            模型预测的主链和侧链原子位置。
        states (`torch.FloatTensor`):
            蛋白质折叠主干的隐藏状态。
            来自蛋白质折叠主干的隐藏状态。
        s_s (`torch.FloatTensor`):
            每个残基嵌入。
            通过连接ESM-2 LM stem每层的隐藏状态得到的每个残基嵌入。
        s_z (`torch.FloatTensor`):
            成对残基嵌入。
            成对残基嵌入。
        distogram_logits (`torch.FloatTensor`):
            距离直方图的输入对数。
            用于计算残基距离的输入对数。
        lm_logits (`torch.FloatTensor`):
            ESM-2蛋白质语言模型主干的输出对数。
            ESM-2蛋白质语言模型主干的输出对数。
        aatype (`torch.FloatTensor`):
            输入的氨基酸（AlphaFold2索引）。
            输入的氨基酸（AlphaFold2索引）。
        atom14_atom_exists (`torch.FloatTensor`):
            每个原子在atom14表示中是否存在。
            每个原子在atom14表示中是否存在。
        residx_atom14_to_atom37 (`torch.FloatTensor`):
            atom14到atom37表示之间的映射。
            atom14到atom37表示之间的映射。
        residx_atom37_to_atom14 (`torch.FloatTensor`):
            atom37到atom14表示之间的映射。
            atom37到atom14表示之间的映射。
        atom37_atom_exists (`torch.FloatTensor`):
            每个原子在atom37表示中是否存在。
            每个原子在atom37表示中是否存在。
        residue_index (`torch.FloatTensor`):
            蛋白链中每个残基的索引。
            蛋白链中每个残基的索引。
        lddt_head (`torch.FloatTensor`):
            lddt头部的原始输出。
            用于计算plddt的lddt头部的原始输出。
        plddt (`torch.FloatTensor`):
            每个残基的置信度分数。
            模型预测结构可能不确定或蛋白结构无序的区域可能表明低置信度的区域。
        ptm_logits (`torch.FloatTensor`):
            用于计算ptm的原始logits。
            用于计算ptm的原始logits。
        ptm (`torch.FloatTensor`):
            TM-score输出，代表模型对整体结构的高级置信度。
            TM-score输出，代表模型对整体结构的高级置信度。
        aligned_confidence_probs (`torch.FloatTensor`):
            对齐结构的每个残基置信度分数。
            对齐结构的每个残基置信度分数。
        predicted_aligned_error (`torch.FloatTensor`):
            模型预测与真实值之间的预测误差。
            模型预测与真实值之间的预测误差。
        max_predicted_aligned_error (`torch.FloatTensor`):
            每个样本的最大预测误差。
            每个样本的最大预测误差。
    """

    frames: torch.FloatTensor = None
    sidechain_frames: torch.FloatTensor = None
    unnormalized_angles: torch.FloatTensor = None
    angles: torch.FloatTensor = None
    # 定义一系列变量，每个变量的类型均为 torch.FloatTensor，初始赋值为 None
    positions: torch.FloatTensor = None  # 用于存储位置信息的张量
    states: torch.FloatTensor = None  # 用于存储状态信息的张量
    s_s: torch.FloatTensor = None  # 用于存储 s_s 信息的张量
    s_z: torch.FloatTensor = None  # 用于存储 s_z 信息的张量
    distogram_logits: torch.FloatTensor = None  # 用于存储距离直方图 logits 的张量
    lm_logits: torch.FloatTensor = None  # 用于存储语言模型 logits 的张量
    aatype: torch.FloatTensor = None  # 用于存储氨基酸类型的张量
    atom14_atom_exists: torch.FloatTensor = None  # 用于存储 atom14 是否存在的张量
    residx_atom14_to_atom37: torch.FloatTensor = None  # 用于存储 residue index 到 atom37 的映射的张量
    residx_atom37_to_atom14: torch.FloatTensor = None  # 用于存储 residue index 到 atom14 的映射的张量
    atom37_atom_exists: torch.FloatTensor = None  # 用于存储 atom37 是否存在的张量
    residue_index: torch.FloatTensor = None  # 用于存储残基索引的张量
    lddt_head: torch.FloatTensor = None  # 用于存储 lddt 头信息的张量
    plddt: torch.FloatTensor = None  # 用于存储 plddt 信息的张量
    ptm_logits: torch.FloatTensor = None  # 用于存储 ptm logits 的张量
    ptm: torch.FloatTensor = None  # 用于存储 ptm 信息的张量
    aligned_confidence_probs: torch.FloatTensor = None  # 用于存储对齐置信度概率的张量
    predicted_aligned_error: torch.FloatTensor = None  # 用于存储预测的对齐误差的张量
    max_predicted_aligned_error: torch.FloatTensor = None  # 用于存储最大预测对齐误差的张量
# 定义一个多行文档字符串，描述了函数 `ESMFOLD_INPUTS_DOCSTRING` 的参数及其含义
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
    # 检查当前是否启用了 FP16 自动转换
    fp16_enabled = torch.get_autocast_gpu_dtype() == torch.float16
    fp16_enabled = fp16_enabled and torch.is_autocast_enabled()
    return fp16_enabled


def is_deepspeed_initialized():
    # 检查是否初始化了 DeepSpeed，如果 DeepSpeed 可用但未初始化则返回 False
    if is_deepspeed_available():
        return False
    else:
        try:
            import deepspeed

            # 尝试调用 DeepSpeed 的初始化检查函数，部分版本可能不支持此功能
            return deepspeed.utils.is_initialized()
        except Exception:
            # 捕获所有异常，返回 False 表示未初始化
            return False


def collate_dense_tensors(samples: List[torch.Tensor], pad_v: float = 0) -> torch.Tensor:
    """
    将一个张量列表堆叠并填充成一个单一张量，所有张量的维度必须一致。
    参数：
        samples: 包含多个张量的列表，每个张量的维度必须相同。
        pad_v: 填充值，默认为 0。
    返回：
        堆叠并填充后的单一张量。
    异常：
        如果 samples 中张量的维度不一致，抛出 RuntimeError 异常。
    """
    if len(samples) == 0:
        return torch.Tensor()  # 如果 samples 列表为空，则返回空张量

    if len({x.dim() for x in samples}) != 1:
        # 检查 samples 中张量的维度是否一致，不一致则抛出异常
        raise RuntimeError(f"Samples has varying dimensions: {[x.dim() for x in samples]}")
    # 从 samples 中获取设备信息，假设所有样本都在同一设备上
    (device,) = tuple({x.device for x in samples})
    
    # 计算 samples 中每个样本的最大形状的每个维度的最大值
    max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
    
    # 使用 torch.empty 创建一个与最大形状匹配的张量 result，长度为 len(samples)，数据类型与 samples[0] 相同，设备与 samples 相同
    result = torch.empty(len(samples), *max_shape, dtype=samples[0].dtype, device=device)
    
    # 用 pad_v 填充 result 张量
    result.fill_(pad_v)
    
    # 遍历每个样本并将其复制到 result 张量的适当位置
    for i in range(len(samples)):
        result_i = result[i]  # 获取 result 中的第 i 个子张量
        t = samples[i]         # 获取第 i 个样本张量 t
        # 将样本张量 t 复制到 result_i 的正确位置
        result_i[tuple(slice(0, k) for k in t.shape)] = t
    
    # 返回填充后的 result 张量，其中包含了所有样本的数据
    return result
# 定义函数，用于将张量的最后几个维度展平成一个维度
def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


# 定义函数，用于对张量的最后几个维度进行置换
def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    # 计算最后几个维度的起始索引
    zero_index = -1 * len(inds)
    # 获取前面的维度索引列表
    first_inds = list(range(len(tensor.shape[:zero_index])))
    # 对张量进行置换操作
    return tensor.permute(first_inds + [zero_index + i for i in inds])


# 定义函数，对多个字典中相同键的值应用指定的函数
def dict_multimap(fn, dicts):
    # 获取第一个字典
    first = dicts[0]
    new_dict = {}
    # 遍历第一个字典的键值对
    for k, v in first.items():
        # 收集所有字典中相同键的值列表
        all_v = [d[k] for d in dicts]
        # 如果第一个字典中的值是字典类型，则递归调用dict_multimap函数
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            # 否则，对所有值应用给定的函数fn
            new_dict[k] = fn(all_v)
    # 返回应用函数后的新字典
    return new_dict


# 定义函数，使用截断正态分布初始化权重张量
def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    # 计算缩放系数
    scale = scale / max(1, shape[1])

    # 检查是否存在SciPy库，如果不存在，则给出警告
    if not is_scipy_available():
        logger.warning(
            "This init requires scipy, but scipy was not found, default to an approximation that might not be"
            " equivalent."
        )
        # 使用近似值初始化权重张量
        std = math.sqrt(scale)
        torch.nn.init.normal_(weights, std=std).clamp(min=0.0, max=2.0 * std)

    else:
        from scipy.stats import truncnorm

        # 使用SciPy的截断正态分布生成权重样本
        std = math.sqrt(scale) / truncnorm.std(a=-2, b=2, loc=0, scale=1)
        samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=weights.numel())
        samples = np.reshape(samples, shape)
        # 将生成的样本复制到权重张量中
        weights.copy_(torch.tensor(samples, device=weights.device))


# 定义函数，使用指定值初始化权重张量
def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        # 用给定值填充权重张量
        weights.fill_(softplus_inverse_1)


# 定义类，继承自torch.nn.Linear，实现了自定义的初始化方法
class EsmFoldLinear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
        # 继承父类构造方法，定义额外的初始化参数
        **kwargs
    ):
        super().__init__(in_dim, out_dim, bias=bias, **kwargs)
    ):
        """
        Args:
            in_dim:
                输入层的最终维度
            out_dim:
                层输出的最终维度
            bias:
                是否学习一个可加偏置，默认为True
            init:
                要使用的初始化器。可选项包括：

                "default": LeCun fan-in截断正态分布初始化
                "relu": 带截断正态分布的He初始化
                "glorot": Fan-average Glorot均匀分布初始化
                "gating": 权重=0，偏置=1
                "normal": 标准差为1/sqrt(fan_in)的正态分布初始化
                "final": 权重=0，偏置=0

                如果init_fn不为None，则被init_fn覆盖。
            init_fn:
                接受权重和偏置作为输入的自定义初始化器。如果不为None，则覆盖init。
        """
        # 调用父类构造函数，初始化输入维度、输出维度和是否有偏置
        super().__init__(in_dim, out_dim, bias=bias)

        # 如果有偏置，用0填充偏置项
        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        # 初始化器和自定义初始化器赋值
        self.init = init
        self.init_fn = init_fn

        # 检查init参数是否合法
        if init not in ["default", "relu", "glorot", "gating", "normal", "final"]:
            raise ValueError("Invalid init string.")
class EsmFoldLayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super().__init__()

        self.c_in = (c_in,)  # 输入通道数的元组，用于后续操作
        self.eps = eps  # Layer normalization 中的 epsilon 参数

        self.weight = nn.Parameter(torch.ones(c_in))  # 可学习的权重参数，默认为全1
        self.bias = nn.Parameter(torch.zeros(c_in))  # 可学习的偏置参数，默认为全0

    def forward(self, x):
        d = x.dtype  # 获取输入张量 x 的数据类型
        if d is torch.bfloat16 and not is_deepspeed_initialized():  # 如果输入是 bfloat16 并且没有启用深度速度优化
            with torch.cuda.amp.autocast(enabled=False):  # 禁用自动混合精度
                out = nn.functional.layer_norm(x, self.c_in, self.weight.to(dtype=d), self.bias.to(dtype=d), self.eps)  # 使用 layer normalization 进行归一化
        else:
            out = nn.functional.layer_norm(x, self.c_in, self.weight, self.bias, self.eps)  # 使用 layer normalization 进行归一化

        return out


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax, but without automatic casting to fp32 when the input is of type bfloat16
    """
    d = t.dtype  # 获取输入张量 t 的数据类型
    if d is torch.bfloat16 and not is_deepspeed_initialized():  # 如果输入是 bfloat16 并且没有启用深度速度优化
        with torch.cuda.amp.autocast(enabled=False):  # 禁用自动混合精度
            s = torch.nn.functional.softmax(t, dim=dim)  # 使用 softmax 计算张量 t 在指定维度上的概率分布
    else:
        s = torch.nn.functional.softmax(t, dim=dim)  # 使用 softmax 计算张量 t 在指定维度上的概率分布

    return s


class EsmFoldAttention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer initialization. Allows multiple bias vectors.
    """

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
        super().__init__()

        self.c_q = c_q  # 查询数据的输入维度
        self.c_k = c_k  # 键数据的输入维度
        self.c_v = c_v  # 值数据的输入维度
        self.c_hidden = c_hidden  # 每个注意力头的隐藏层维度
        self.no_heads = no_heads  # 注意力头的数量
        self.gating = gating  # 是否使用查询数据对输出进行门控

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = EsmFoldLinear(self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot")  # 查询线性变换层
        self.linear_k = EsmFoldLinear(self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot")  # 键线性变换层
        self.linear_v = EsmFoldLinear(self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot")  # 值线性变换层
        self.linear_o = EsmFoldLinear(self.c_hidden * self.no_heads, self.c_q, init="final")  # 输出线性变换层

        self.linear_g = None
        if self.gating:
            self.linear_g = EsmFoldLinear(self.c_q, self.c_hidden * self.no_heads, init="gating")  # 门控线性变换层

        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数的实例化
    # 准备 Q/K/V 查询、键、值的线性变换
    def _prep_qkv(self, q_x: torch.Tensor, kv_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 对查询向量 q_x 执行线性变换
        q = self.linear_q(q_x)
        # 对键向量 kv_x 执行线性变换
        k = self.linear_k(kv_x)
        # 对值向量 kv_x 执行线性变换
        v = self.linear_v(kv_x)

        # 重新塑形以适应多头注意力机制的输入格式
        # [*, Q/K/V, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # 将多头维度与注意力头部数交换位置，以便后续计算注意力权重
        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        # 缩放 Q 向量，以便在计算注意力权重时更稳定
        q /= math.sqrt(self.c_hidden)

        return q, k, v

    # 处理输出结果 o，并应用可选的全局门控线性变换
    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        if self.linear_g is not None:
            # 计算全局门控线性变换的输出，并应用 Sigmoid 激活函数
            g = self.sigmoid(self.linear_g(q_x))

            # 重新塑形以适应多头注意力机制的输入格式
            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # 将多头注意力机制的输出展平最后两个维度
        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # 对最终的输出应用线性变换，将其映射到输出空间
        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    # 实现模型的前向传播
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
                [*, Q, C_q] query data  # 输入的查询数据，形状为 [*, Q, C_q]
            kv_x:
                [*, K, C_k] key data  # 输入的键数据，形状为 [*, K, C_k]
            biases:
                List of biases that broadcast to [*, H, Q, K]  # 广播到 [*, H, Q, K] 的偏置列表
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel. This should be the default choice for most.
                If none of the "use_<...>" flags are True, a stock PyTorch implementation is used instead
                是否使用自定义的内存高效注意力核。对于大多数情况，这应该是默认选择。
                如果没有一个 "use_<...>" 标志为 True，则使用标准的 PyTorch 实现
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If none of the "use_<...>" flags are True, a
                stock PyTorch implementation is used instead
                是否使用低内存注意力 (Staats & Rabe 2021)。
                如果没有一个 "use_<...>" 标志为 True，则使用标准的 PyTorch 实现
            lma_q_chunk_size:
                Query chunk size (for LMA)  # 查询分块大小（用于低内存注意力）
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)  # 键/值分块大小（用于低内存注意力）
        Returns
            [*, Q, C_q] attention update  # 注意力更新后的输出，形状为 [*, Q, C_q]
        """
        if use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None):
            raise ValueError("If use_lma is specified, lma_q_chunk_size and lma_kv_chunk_size must be provided")
            # 如果使用低内存注意力，并且没有提供查询或键/值的分块大小，则抛出数值错误异常

        if use_flash and biases is not None:
            raise ValueError("use_flash is incompatible with the bias option. For masking, use flash_mask instead")
            # 如果同时使用闪存和偏置选项，则抛出数值错误异常。应使用 flash_mask 进行遮罩操作而非偏置。

        attn_options = [use_memory_efficient_kernel, use_lma, use_flash]
        if sum(attn_options) > 1:
            raise ValueError("Choose at most one alternative attention algorithm")
            # 如果选择了多个注意力算法选项，则抛出数值错误异常。只能选择最多一个备选注意力算法。

        if biases is None:
            biases = []

        # [*, H, Q/K, C_hidden]
        query, key, value = self._prep_qkv(q_x, kv_x)
        key = permute_final_dims(key, (1, 0))
        # 准备查询、键、值，形状为 [*, H, Q/K, C_hidden]，并将键的最后两个维度进行置换

        # [*, H, Q, K]
        output = torch.matmul(query, key)
        # 执行矩阵乘法得到注意力分数矩阵 [*, H, Q, K]
        for b in biases:
            output += b
        # 添加偏置到输出
        output = softmax_no_cast(output, -1)
        # 在最后一个维度上执行 softmax 操作，得到注意力权重

        # [*, H, Q, C_hidden]
        output = torch.matmul(output, value)
        # 使用注意力权重加权值，得到加权后的值矩阵，形状为 [*, H, Q, C_hidden]
        output = output.transpose(-2, -3)
        # 对输出进行维度转置，将倒数第二个和倒数第三个维度进行交换
        output = self._wrap_up(output, q_x)
        # 调用 _wrap_up 方法对输出进行包装处理，根据查询数据 q_x

        return output
class EsmFoldTriangleAttention(nn.Module):
    # 定义 EsmFoldTriangleAttention 类，继承自 nn.Module
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                输入通道维度
            c_hidden:
                总体隐藏通道维度（非每个注意力头）
            no_heads:
                注意力头的数量
        """
        super().__init__()
        
        # 初始化类的属性
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf
        
        # 初始化层归一化对象
        self.layer_norm = LayerNorm(self.c_in)
        
        # 初始化线性层对象
        self.linear = EsmFoldLinear(c_in, self.no_heads, bias=False, init="normal")
        
        # 初始化自定义的注意力对象
        self.mha = EsmFoldAttention(self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads)

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
        # 准备输入参数字典给多头注意力的 chunk_layer 方法
        mha_inputs = {
            "q_x": x,
            "kv_x": x,
            "biases": biases,
        }

        # 使用 chunk_layer 函数对注意力进行分块处理
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
        # 正向传播函数，接收输入张量 x 和可选的掩码 mask
        pass  # 实际实现在此处省略
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        # 如果没有提供掩码，则创建一个形状为 [*, I, J] 的新张量，所有元素为1
        if mask is None:
            mask = x.new_ones(
                x.shape[:-1],
            )

        # 如果不是起始状态，交换输入张量的倒数第二和倒数第三个维度
        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # 对输入张量进行 layer normalization，形状保持不变 [*, I, J, C_in]
        x = self.layer_norm(x)

        # 创建一个形状为 [*, I, 1, 1, J] 的张量，其中 mask_bias 的计算基于 mask 张量
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # 对线性层的输出进行维度变换，形状为 [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))

        # 在倒数第四个维度上扩展 triangle_bias，形状变为 [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)

        # 将 mask_bias 和 triangle_bias 放入列表中作为偏置项
        biases = [mask_bias, triangle_bias]

        # 如果指定了 chunk_size，则调用 _chunk 方法处理输入 x 和 biases
        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            # 否则调用 self.mha 进行多头注意力计算，使用给定的 biases
            x = self.mha(
                q_x=x, kv_x=x, biases=biases, use_memory_efficient_kernel=use_memory_efficient_kernel, use_lma=use_lma
            )

        # 如果不是起始状态，恢复 x 的倒数第二和倒数第三个维度的顺序
        if not self.starting:
            x = x.transpose(-2, -3)

        # 返回处理后的张量 x
        return x
    """
    Implements Algorithms 11 and 12.
    实现第 11 和第 12 算法。
    """

    def __init__(self, config, _outgoing=True):
        # 初始化函数，设置模型参数
        super().__init__()
        # 从配置中获取隐藏状态的维度
        c_hidden = config.pairwise_state_dim
        # 是否是外部输出
        self._outgoing = _outgoing

        # 定义线性层，用于算法中的计算
        self.linear_a_p = EsmFoldLinear(c_hidden, c_hidden)
        self.linear_a_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        self.linear_b_p = EsmFoldLinear(c_hidden, c_hidden)
        self.linear_b_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        self.linear_g = EsmFoldLinear(c_hidden, c_hidden, init="gating")
        self.linear_z = EsmFoldLinear(c_hidden, c_hidden, init="final")

        # 初始化输入和输出的 LayerNorm
        self.layer_norm_in = LayerNorm(c_hidden)
        self.layer_norm_out = LayerNorm(c_hidden)

        # 定义 Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def _combine_projections(
        self, a: torch.Tensor, b: torch.Tensor, _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        # 组合投影函数，根据 _outgoing 参数确定维度顺序
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))
            b = permute_final_dims(b, (2, 1, 0))
        else:
            a = permute_final_dims(a, (2, 1, 0))
            b = permute_final_dims(b, (2, 0, 1))

        # 如果指定了 _inplace_chunk_size，使用循环方式批量处理
        if _inplace_chunk_size is not None:
            # 待替换为 torch vmap 的部分
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.matmul(
                    a_chunk,
                    b_chunk,
                )

            p = a
        else:
            # 否则直接进行矩阵乘法运算
            p = torch.matmul(a, b)

        return permute_final_dims(p, (1, 2, 0))

    def _inference_forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_chunk_size: Optional[int] = None,
        with_add: bool = True,
    ):
        # 推断过程的前向传播函数，包括处理 mask、是否进行 in-place 操作和是否添加额外计算
        ...

    def forward(
        self,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: Optional[int] = 256,
    ):
        # 模型的前向传播函数，接受输入张量 z 和可选的 mask，执行模型计算
        ...
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_res, N_res, C_z] input tensor 输入张量，形状为 [*, N_res, N_res, C_z]
            mask:
                [*, N_res, N_res] input mask 输入的遮罩，形状为 [*, N_res, N_res]
        Returns:
            [*, N_res, N_res, C_z] output tensor 输出张量，形状为 [*, N_res, N_res, C_z]
        """
        if inplace_safe:
            x = self._inference_forward(
                z,
                mask,
                inplace_chunk_size=_inplace_chunk_size,  # 设置原地操作的块大小
                with_add=_add_with_inplace,  # 原地操作时是否进行加法
            )
            return x  # 返回处理后的张量

        if mask is None:
            mask = z.new_ones(z.shape[:-1])  # 使用输入 z 的形状创建全为 1 的遮罩

        mask = mask.unsqueeze(-1)  # 在最后一个维度上增加一个维度，形状变为 [*, N_res, N_res, 1]

        z = self.layer_norm_in(z)  # 输入 z 执行层归一化操作
        a = mask  # 将 mask 赋值给变量 a
        a = a * self.sigmoid(self.linear_a_g(z))  # a 乘以线性变换后经过 sigmoid 函数的结果
        a = a * self.linear_a_p(z)  # a 乘以另一个线性变换的结果
        b = mask  # 将 mask 赋值给变量 b
        b = b * self.sigmoid(self.linear_b_g(z))  # b 乘以线性变换后经过 sigmoid 函数的结果
        b = b * self.linear_b_p(z)  # b 乘以另一个线性变换的结果

        if is_fp16_enabled():  # 如果启用了 FP16 计算
            with torch.cuda.amp.autocast(enabled=False):  # 关闭自动混合精度计算
                x = self._combine_projections(a.float(), b.float())  # 使用浮点数进行投影组合
        else:
            x = self._combine_projections(a, b)  # 使用原始数据类型进行投影组合

        del a, b  # 删除变量 a 和 b
        x = self.layer_norm_out(x)  # 对输出 x 进行层归一化操作
        x = self.linear_z(x)  # 对归一化后的 x 进行线性变换
        g = self.sigmoid(self.linear_g(z))  # 对 z 执行线性变换后经过 sigmoid 函数的结果
        x = x * g  # 将 x 乘以 g

        return x  # 返回处理后的张量
class EsmFoldPreTrainedModel(EsmPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Subclass `EsmPreTrainedModel` to handle special initialization of weights
    def _init_weights(self, module):
        """Initialize the weights of the given module."""
        # Check if the module is an instance of `EsmFoldLinear`
        if isinstance(module, EsmFoldLinear):
            # Apply weight initialization based on module's initialization method
            with torch.no_grad():
                # Initialize using custom function if specified
                if module.init_fn is not None:
                    module.init_fn(module.weight, module.bias)
                # Initialize using truncated normal distribution with scale 1.0
                elif module.init == "default":
                    trunc_normal_init_(module.weight, scale=1.0)
                # Initialize using truncated normal distribution with scale 2.0
                elif module.init == "relu":
                    trunc_normal_init_(module.weight, scale=2.0)
                # Initialize using Xavier uniform initialization
                elif module.init == "glorot":
                    nn.init.xavier_uniform_(module.weight, gain=1)
                # Initialize weights to zero for "gating" type
                elif module.init == "gating":
                    module.weight.fill_(0.0)
                    # Initialize bias to 1.0 if bias exists
                    if module.bias:
                        module.bias.fill_(1.0)
                # Initialize using Kaiming normal distribution for "normal" type
                elif module.init == "normal":
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                # Initialize weights to zero for "final" type
                elif module.init == "final":
                    module.weight.fill_(0.0)
        # Initialize weights for `EsmFoldInvariantPointAttention` module
        elif isinstance(module, EsmFoldInvariantPointAttention):
            ipa_point_weights_init_(module.head_weights)
        # Initialize weights for `EsmFoldTriangularSelfAttentionBlock` module
        elif isinstance(module, EsmFoldTriangularSelfAttentionBlock):
            # Initialize various linear layers' weights and biases to zero
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
            # Call superclass method to initialize weights
            super()._init_weights(module)


class EsmFoldSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, head_width, gated=False):
        super().__init__()
        assert embed_dim == num_heads * head_width

        self.embed_dim = embed_dim  # 设置嵌入维度
        self.num_heads = num_heads  # 设置头的数量
        self.head_width = head_width  # 设置每个头的宽度

        # 定义投影层，将输入映射到更高维度的空间，不使用偏置项
        self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # 输出投影层，将多头注意力的结果映射回原始的嵌入维度，使用偏置项
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.gated = gated  # 是否启用门控机制
        if gated:
            # 门控投影层，用于门控机制的加权输出
            self.g_proj = nn.Linear(embed_dim, embed_dim)
            torch.nn.init.zeros_(self.g_proj.weight)  # 初始化权重为零
            torch.nn.init.ones_(self.g_proj.bias)  # 初始化偏置为一

        self.rescale_factor = self.head_width**-0.5  # 缩放因子

        torch.nn.init.zeros_(self.o_proj.bias)  # 输出投影层偏置初始化为零

    def forward(self, x, mask=None, bias=None, indices=None):
        """
        基础的自注意力机制，可选带掩码和外部的注意力偏置。用于处理不同长度的序列，使用掩码。

        Inputs:
            x: 输入序列的批量 (.. x L x C) mask: 批量的布尔掩码，其中 1=有效，0=填充位置 (.. x L_k) bias: 批量的标量注意力偏置 (.. x Lq x Lk x num_heads)

        Outputs:
            序列投影 (B x L x embed_dim), 注意力映射 (B x L x L x num_heads)
        """

        t = self.proj(x).view(*x.shape[:2], self.num_heads, -1)  # 投影并重塑张量形状
        t = t.permute(0, 2, 1, 3)  # 转置张量的维度顺序
        q, k, v = t.chunk(3, dim=-1)  # 拆分成查询、键、值

        q = self.rescale_factor * q  # 缩放查询向量
        a = torch.einsum("...qc,...kc->...qk", q, k)  # 执行注意力计算

        # 添加外部注意力偏置
        if bias is not None:
            a = a + bias.permute(0, 3, 1, 2)

        # 不参与填充令牌的注意力
        if mask is not None:
            mask = mask[:, None, None]
            a = a.masked_fill(mask == False, -np.inf)  # noqa: E712

        a = nn.functional.softmax(a, dim=-1)  # 执行 softmax 操作得到注意力权重

        y = torch.einsum("...hqk,...hkc->...qhc", a, v)  # 应用注意力权重到值上
        y = y.reshape(*y.shape[:2], -1)  # 重塑输出形状

        if self.gated:
            y = self.g_proj(x).sigmoid() * y  # 使用门控机制调节输出
        y = self.o_proj(y)  # 最终的输出投影

        return y, a.permute(0, 3, 1, 2)  # 返回结果及注意力权重的转置
class EsmFoldDropout(nn.Module):
    """
    Implementation of dropout with the ability to share the dropout mask along a particular dimension.
    """

    def __init__(self, r: float, batch_dim: Union[int, List[int]]):
        super().__init__()

        self.r = r  # 设定 dropout 的概率 r
        if isinstance(batch_dim, int):
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim  # 指定需要共享 dropout mask 的维度
        self.dropout = nn.Dropout(self.r)  # 初始化 Dropout 层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)  # 获取输入张量 x 的形状
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1  # 将指定维度的大小设为 1，用于共享 dropout mask
        return x * self.dropout(x.new_ones(shape))  # 对输入张量 x 应用 dropout 操作


class EsmFoldSequenceToPair(nn.Module):
    def __init__(self, sequence_state_dim, inner_dim, pairwise_state_dim):
        super().__init__()

        self.layernorm = nn.LayerNorm(sequence_state_dim)  # 序列归一化层
        self.proj = nn.Linear(sequence_state_dim, inner_dim * 2, bias=True)  # 线性投影层
        self.o_proj = nn.Linear(2 * inner_dim, pairwise_state_dim, bias=True)  # 输出线性投影层

        torch.nn.init.zeros_(self.proj.bias)  # 将投影层的偏置项初始化为零
        torch.nn.init.zeros_(self.o_proj.bias)  # 将输出投影层的偏置项初始化为零

    def forward(self, sequence_state):
        """
        Inputs:
          sequence_state: B x L x sequence_state_dim

        Output:
          pairwise_state: B x L x L x pairwise_state_dim

        Intermediate state:
          B x L x L x 2*inner_dim
        """

        assert len(sequence_state.shape) == 3  # 断言输入张量的形状为 B x L x sequence_state_dim

        s = self.layernorm(sequence_state)  # 序列归一化
        s = self.proj(s)  # 应用线性投影
        q, k = s.chunk(2, dim=-1)  # 将投影后的结果切分为两部分，q 和 k

        prod = q[:, None, :, :] * k[:, :, None, :]  # 计算乘积部分
        diff = q[:, None, :, :] - k[:, :, None, :]  # 计算差异部分

        x = torch.cat([prod, diff], dim=-1)  # 拼接乘积和差异部分
        x = self.o_proj(x)  # 应用输出投影层

        return x  # 返回输出张量


class EsmFoldPairToSequence(nn.Module):
    def __init__(self, pairwise_state_dim, num_heads):
        super().__init__()

        self.layernorm = nn.LayerNorm(pairwise_state_dim)  # 对成对状态维度进行归一化
        self.linear = nn.Linear(pairwise_state_dim, num_heads, bias=False)  # 线性层，用于生成成对偏置

    def forward(self, pairwise_state):
        """
        Inputs:
          pairwise_state: B x L x L x pairwise_state_dim

        Output:
          pairwise_bias: B x L x L x num_heads
        """
        assert len(pairwise_state.shape) == 4  # 断言输入张量的形状为 B x L x L x pairwise_state_dim
        z = self.layernorm(pairwise_state)  # 应用归一化层
        pairwise_bias = self.linear(z)  # 应用线性层生成成对偏置
        return pairwise_bias  # 返回成对偏置张量


class EsmFoldResidueMLP(nn.Module):
    def __init__(self, embed_dim, inner_dim, dropout=0):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),  # 对嵌入维度进行归一化
            nn.Linear(embed_dim, inner_dim),  # 第一个线性层
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(inner_dim, embed_dim),  # 第二个线性层
            nn.Dropout(dropout),  # Dropout 层
        )

    def forward(self, x):
        return x + self.mlp(x)  # 返回输入张量加上 MLP 处理后的结果


class EsmFoldTriangularSelfAttentionBlock(nn.Module):
    """
    Placeholder for a module implementing a triangular self-attention block.
    This class is not fully implemented in the provided code snippet.
    """
    # 初始化函数，用于创建对象实例时的初始化操作，接受一个配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数保存到实例的config属性中
        self.config = config

        # 从配置参数中获取序列状态维度和成对状态维度
        sequence_state_dim = config.sequence_state_dim
        pairwise_state_dim = config.pairwise_state_dim

        # 根据配置参数计算序列自注意力机制的头数
        sequence_num_heads = sequence_state_dim // config.sequence_head_width
        # 根据配置参数计算成对自注意力机制的头数
        pairwise_num_heads = pairwise_state_dim // config.pairwise_head_width

        # 创建一个序列层归一化模块，传入序列状态维度作为参数
        self.layernorm_1 = nn.LayerNorm(sequence_state_dim)

        # 创建一个将序列映射为成对表示的模块，传入序列状态维度和一半的成对状态维度作为参数
        self.sequence_to_pair = EsmFoldSequenceToPair(sequence_state_dim, pairwise_state_dim // 2, pairwise_state_dim)
        # 创建一个将成对表示映射回序列表示的模块，传入成对状态维度和序列自注意力机制头数作为参数
        self.pair_to_sequence = EsmFoldPairToSequence(pairwise_state_dim, sequence_num_heads)

        # 创建一个序列自注意力机制模块，传入序列状态维度、头数、头宽度和是否启用门控机制作为参数
        self.seq_attention = EsmFoldSelfAttention(
            sequence_state_dim, sequence_num_heads, config.sequence_head_width, gated=True
        )
        
        # 创建一个序列三角形形态更新模块（输出方向），传入配置参数和输出方向（True表示输出方向）
        self.tri_mul_out = EsmFoldTriangleMultiplicativeUpdate(config, _outgoing=True)
        # 创建一个序列三角形形态更新模块（输入方向），传入配置参数和输出方向（False表示输入方向）
        self.tri_mul_in = EsmFoldTriangleMultiplicativeUpdate(config, _outgoing=False)

        # 创建一个成对三角形注意力模块（起始方向），传入成对状态维度、头宽度、头数、无穷大值和是否起始方向为True作为参数
        self.tri_att_start = EsmFoldTriangleAttention(
            pairwise_state_dim, config.pairwise_head_width, pairwise_num_heads, inf=1e9, starting=True
        )
        # 创建一个成对三角形注意力模块（结束方向），传入成对状态维度、头宽度、头数、无穷大值和是否起始方向为False作为参数
        self.tri_att_end = EsmFoldTriangleAttention(
            pairwise_state_dim, config.pairwise_head_width, pairwise_num_heads, inf=1e9, starting=False
        )

        # 创建一个序列残差MLP模块，传入序列状态维度、4倍的序列状态维度和dropout概率作为参数
        self.mlp_seq = EsmFoldResidueMLP(sequence_state_dim, 4 * sequence_state_dim, dropout=config.dropout)
        # 创建一个成对残差MLP模块，传入成对状态维度、4倍的成对状态维度和dropout概率作为参数
        self.mlp_pair = EsmFoldResidueMLP(pairwise_state_dim, 4 * pairwise_state_dim, dropout=config.dropout)

        # 创建一个普通的dropout模块，传入dropout概率作为参数
        self.drop = nn.Dropout(config.dropout)
        # 创建一个行dropout模块，传入2倍的dropout概率和1作为参数
        self.row_drop = EsmFoldDropout(config.dropout * 2, 2)
        # 创建一个列dropout模块，传入2倍的dropout概率和1作为参数
        self.col_drop = EsmFoldDropout(config.dropout * 2, 1)
class EsmCategoricalMixture:
    # 定义一个混合分类分布的类
    def __init__(self, param, bins=50, start=0, end=1):
        # 初始化方法，接收参数和一些配置信息
        # 所有的张量都是形状为 ..., bins
        self.logits = param
        # 创建一个等间距的张量 bins，用于表示值的中心点
        bins = torch.linspace(start, end, bins + 1, device=self.logits.device, dtype=self.logits.dtype)
        # 计算每个 bin 的中心值
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # 计算给定值的对数概率
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ...
        # 找到最接近 true 的值在 v_bins 中的索引
        true_index = (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        # 计算 logits 的对数 softmax，并计算负对数似然
        nll = self.logits.log_softmax(-1)
        # 返回 true_index 处的对数概率
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        # 计算混合分布的均值
        return (self.logits.softmax(-1) @ self.v_bins.unsqueeze(1)).squeeze(-1)


def categorical_lddt(logits, bins=50):
    # 计算混合分类分布的均值
    # Logits are ..., 37, bins.
    return EsmCategoricalMixture(logits, bins=bins).mean()


def get_axial_mask(mask):
    """
    Helper to convert B x L mask of valid positions to axial mask used in row column attentions.

    Input:
      mask: B x L tensor of booleans

    Output:
      mask: B x L x L tensor of booleans
    """
    # 将 B x L 的有效位置掩码转换为用于行列注意力的轴向掩码的辅助函数

    if mask is None:
        return None

    if len(mask.shape) != 2:
        # 如果掩码的维度不是 2，则抛出异常
        raise ValueError(f"`mask` should be a 2d-tensor, got {len(mask.shape)} dims.")
    batch_dim, seq_dim = mask.shape
    # 在第二个维度上扩展掩码，以便生成 B x L x L 的掩码
    m = mask.unsqueeze(1).expand(batch_dim, seq_dim, seq_dim)
    m = m.reshape(batch_dim * seq_dim, seq_dim)
    return m


class EsmFoldRelativePosition(nn.Module):
    # 相对位置编码模块
    def __init__(self, config):
        super().__init__()
        self.bins = config.position_bins

        # Note an additional offset is used so that the 0th position
        # is reserved for masked pairs.
        # 使用额外的偏移量，确保第 0 位置留给掩码对

        self.embedding = torch.nn.Embedding(2 * self.bins + 2, config.pairwise_state_dim)

    def forward(self, residue_index, mask=None):
        """
        Input:
          residue_index: B x L tensor of indices (dytpe=torch.long) mask: B x L tensor of booleans

        Output:
          pairwise_state: B x L x L x pairwise_state_dim tensor of embeddings
        """
        # 前向传播函数，接收残基索引和掩码，返回残基的嵌入向量

        if residue_index.dtype != torch.long:
            # 如果残基索引的数据类型不是 torch.long，则抛出异常
            raise ValueError(f"`residue_index` has dtype {residue_index.dtype}, it should be `torch.long`.")
        if mask is not None and residue_index.shape != mask.shape:
            # 如果掩码不为空且形状与残基索引不一致，则抛出异常
            raise ValueError(
                f"`residue_index` and `mask` have inconsistent shapes: {residue_index.shape} != {mask.shape}."
            )

        # 计算残基索引之间的距离，并进行截断
        diff = residue_index[:, None, :] - residue_index[:, :, None]
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # Add 1 to adjust for padding index.

        if mask is not None:
            # 如果掩码不为空，则应用掩码
            mask = mask[:, None, :] * mask[:, :, None]
            diff[mask == False] = 0  # noqa: E712

        # 使用嵌入层将距离转换为嵌入向量
        output = self.embedding(diff)
        return output


class EsmFoldAngleResnetBlock(nn.Module):
    # ESM 折叠角度 ResNet 块，未完整提供代码，无需注释
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 创建第一个线性层，输入和输出维度都为 config.resnet_dim，使用 ReLU 激活函数初始化
        self.linear_1 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="relu")
        
        # 创建第二个线性层，输入和输出维度也为 config.resnet_dim，使用 "final" 方法进行初始化
        self.linear_2 = EsmFoldLinear(config.resnet_dim, config.resnet_dim, init="final")

        # 创建 ReLU 激活函数层
        self.relu = nn.ReLU()

    # 前向传播函数，接受一个 torch.Tensor 类型的输入 a，返回一个 torch.Tensor 类型的输出
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # 保存初始输入 a 到 s_initial 中
        s_initial = a

        # 对输入 a 应用 ReLU 激活函数
        a = self.relu(a)
        
        # 将经过 ReLU 激活函数后的输入 a 传入第一个线性层 self.linear_1
        a = self.linear_1(a)
        
        # 再次应用 ReLU 激活函数
        a = self.relu(a)
        
        # 将经过第一个线性层和 ReLU 后的输出 a 传入第二个线性层 self.linear_2
        a = self.linear_2(a)

        # 返回最终输出，它是第二个线性层的输出与初始输入的和
        return a + s_initial
class EsmFoldAngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 初始化输入线性层，将输入维度转换为ResNet维度
        self.linear_in = EsmFoldLinear(config.sequence_dim, config.resnet_dim)
        # 初始化初始线性层，将输入维度转换为ResNet维度
        self.linear_initial = EsmFoldLinear(config.sequence_dim, config.resnet_dim)

        # 初始化ResNet块的列表
        self.layers = nn.ModuleList()
        for _ in range(config.num_resnet_blocks):
            layer = EsmFoldAngleResnetBlock(config)
            self.layers.append(layer)

        # 初始化输出线性层，将ResNet维度转换为角度预测的维度（num_angles * 2）
        self.linear_out = EsmFoldLinear(config.resnet_dim, config.num_angles * 2)

        # 定义ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] 单个嵌入向量
            s_initial:
                [*, C_hidden] StructureModule 开始时的单个嵌入向量
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                [*, no_angles, 2] 预测的角度
        """
        # 注意：补充资料中未提及对输入应用ReLU，但在源代码中存在。
        # 为了最大兼容性，保留源代码中的实现方式。

        # 对 s_initial 应用ReLU激活函数
        s_initial = self.relu(s_initial)
        # 经过初始线性层处理
        s_initial = self.linear_initial(s_initial)
        # 对 s 应用ReLU激活函数
        s = self.relu(s)
        # 经过输入线性层处理
        s = self.linear_in(s)
        # 加上初始嵌入向量处理后的结果
        s = s + s_initial

        # 遍历所有的ResNet块
        for l in self.layers:
            s = l(s)

        # 对结果应用ReLU激活函数
        s = self.relu(s)

        # 经过输出线性层处理，得到未归一化的预测值
        s = self.linear_out(s)

        # 将输出形状变换为 [*, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        # 对 s 进行归一化处理
        unnormalized_s = s  # 保存未归一化的预测值
        norm_denom = torch.sqrt(
            torch.clamp(
                torch.sum(s**2, dim=-1, keepdim=True),
                min=self.config.epsilon,
            )
        )
        s = s / norm_denom  # 归一化处理

        return unnormalized_s, s


class EsmFoldInvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 将配置对象保存到实例属性中
        self.config = config

        # 从配置对象中获取各个维度的设定
        c_s = config.sequence_dim
        c_z = config.pairwise_dim
        self.hidden_dim = config.ipa_dim
        self.num_heads = config.num_heads_ipa
        self.num_qk_points = config.num_qk_points
        self.num_v_points = config.num_v_points

        # 下面的线性层与说明书中的规格不同。
        # 说明书中，它们没有偏置并使用Glorot初始化。
        # 在这里和官方源码中，它们带有偏置并使用默认的Lecun初始化。
        
        # 计算线性层q的输出维度
        hc = config.ipa_dim * config.num_heads_ipa
        # 创建线性层q，输入维度为c_s，输出维度为hc
        self.linear_q = EsmFoldLinear(c_s, hc)
        
        # 计算线性层kv的输出维度
        self.linear_kv = EsmFoldLinear(c_s, 2 * hc)

        # 计算线性层q_points的输出维度
        hpq = config.num_heads_ipa * config.num_qk_points * 3
        self.linear_q_points = EsmFoldLinear(c_s, hpq)

        # 计算线性层kv_points的输出维度
        hpkv = config.num_heads_ipa * (config.num_qk_points + config.num_v_points) * 3
        self.linear_kv_points = EsmFoldLinear(c_s, hpkv)

        # 创建线性层b，输入维度为c_z，输出维度为config.num_heads_ipa
        self.linear_b = EsmFoldLinear(c_z, config.num_heads_ipa)

        # 创建可学习的参数，用于存储头部权重
        self.head_weights = nn.Parameter(torch.zeros((config.num_heads_ipa)))

        # 计算拼接后的输出维度
        concat_out_dim = config.num_heads_ipa * (c_z + config.ipa_dim + config.num_v_points * 4)
        # 创建线性层out，输入维度为concat_out_dim，输出维度为c_s，使用"final"初始化方式
        self.linear_out = EsmFoldLinear(concat_out_dim, c_s, init="final")

        # 创建softmax激活函数，沿着最后一个维度进行softmax操作
        self.softmax = nn.Softmax(dim=-1)
        # 创建softplus激活函数
        self.softplus = nn.Softplus()

    # 前向传播函数定义，接受多个输入参数并返回一个输出
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

        # Initialize a linear layer for updating the backbone with 6 output features
        self.linear = EsmFoldLinear(config.sequence_dim, 6, init="final")

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # Compute the update vector using the linear layer
        update = self.linear(s)

        return update


class EsmFoldStructureModuleTransitionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Initialize three linear layers for transformation, using ReLU activation for the first two
        self.linear_1 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        self.linear_2 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="relu")
        self.linear_3 = EsmFoldLinear(config.sequence_dim, config.sequence_dim, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        # Save the initial input for later residual connection
        s_initial = s

        # Pass through the three linear layers with ReLU activations in between
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        # Add the initial input to the transformed output (residual connection)
        s = s + s_initial

        return s


class EsmFoldStructureModuleTransition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize a series of transition layers based on the specified number in config
        self.layers = nn.ModuleList()
        for _ in range(config.num_transition_layers):
            l = EsmFoldStructureModuleTransitionLayer(config)
            self.layers.append(l)

        # Apply dropout and layer normalization
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = LayerNorm(config.sequence_dim)

    def forward(self, s):
        # Forward pass through each transition layer
        for l in self.layers:
            s = l(s)

        # Apply dropout and layer normalization to the final output
        s = self.dropout(s)
        s = self.layer_norm(s)

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

        # Initialize layer normalization for sequence and pairwise dimensions
        self.layer_norm_s = LayerNorm(config.sequence_dim)
        self.layer_norm_z = LayerNorm(config.pairwise_dim)

        # Linear layer for initial transformation of input sequence
        self.linear_in = EsmFoldLinear(config.sequence_dim, config.sequence_dim)

        # Initialize Invariant Point Attention and its associated dropout and layer normalization
        self.ipa = EsmFoldInvariantPointAttention(config)
        self.ipa_dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm_ipa = LayerNorm(config.sequence_dim)

        # Initialize transition module, backbone update, and angle resnet modules
        self.transition = EsmFoldStructureModuleTransition(config)
        self.bb_update = EsmFoldBackboneUpdate(config)
        self.angle_resnet = EsmFoldAngleResnet(config)

    def forward(
        self,
        evoformer_output_dict,
        aatype,
        mask=None,
        _offload_inference=False,
    ):
        # Implementation of forward pass for the entire structure module is not provided here
        pass
    # 初始化残基常量，如果不存在默认帧，则注册为缓冲区张量
    def _init_residue_constants(self, float_dtype, device):
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
        # 如果不存在组索引，则注册为缓冲区张量
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
        # 如果不存在原子掩码，则注册为缓冲区张量
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
        # 如果不存在文献位置，则注册为缓冲区张量
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
        # 懒惰地在正确的设备上初始化残基常量
        self._init_residue_constants(alpha.dtype, alpha.device)
        # 将扭转角转换为帧，使用默认帧作为参数之一
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    # 将帧和文献位置转换为原子14位置
    def frames_and_literature_positions_to_atom14_pos(self, r, f):  # [*, N, 8]  # [*, N]
        # 懒惰地在正确的设备上初始化残基常量
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        # 使用帧、组索引、原子掩码和文献位置将帧和文献位置转换为原子14位置
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
# 定义一个名为 EsmFoldingTrunk 的神经网络模块类，继承自 nn.Module
class EsmFoldingTrunk(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 将传入的 config 参数保存在实例变量 self.config 中
        self.config = config

        # 从 config 中获取序列状态维度和成对状态维度，并保存到本地变量 c_s 和 c_z 中
        c_s = config.sequence_state_dim
        c_z = config.pairwise_state_dim

        # 创建一个 EsmFoldRelativePosition 实例，用于生成成对位置嵌入
        self.pairwise_positional_embedding = EsmFoldRelativePosition(config)

        # 创建一个由多个 EsmFoldTriangularSelfAttentionBlock 实例组成的模块列表，
        # 列表的长度由 config.num_blocks 决定
        self.blocks = nn.ModuleList([EsmFoldTriangularSelfAttentionBlock(config) for _ in range(config.num_blocks)])

        # 设置循环使用的桶数为 15
        self.recycle_bins = 15
        # 创建一个用于序列状态归一化的 LayerNorm 实例，参数为 c_s
        self.recycle_s_norm = nn.LayerNorm(c_s)
        # 创建一个用于成对状态归一化的 LayerNorm 实例，参数为 c_z
        self.recycle_z_norm = nn.LayerNorm(c_z)
        # 创建一个嵌入层，用于存储循环分布信息，有 recycle_bins 个桶，每个桶长度为 c_z
        self.recycle_disto = nn.Embedding(self.recycle_bins, c_z)
        # 将嵌入层的第一个权重向量初始化为零
        self.recycle_disto.weight[0].detach().zero_()

        # 创建一个 EsmFoldStructureModule 实例，用于处理结构模块相关任务
        self.structure_module = EsmFoldStructureModule(config.structure_module)
        # 创建一个线性层，将序列状态映射到结构模块的序列维度大小
        self.trunk2sm_s = nn.Linear(c_s, config.structure_module.sequence_dim)
        # 创建一个线性层，将成对状态映射到结构模块的成对维度大小
        self.trunk2sm_z = nn.Linear(c_z, config.structure_module.pairwise_dim)

        # 初始化块的默认大小，用于分块处理注意力机制的输入
        self.chunk_size = config.chunk_size

    # 设置块的大小，用于分块处理注意力机制的输入
    def set_chunk_size(self, chunk_size):
        # 参数 chunk_size 指示将使用分块方式计算轴向注意力机制。
        # 这可以使得内存使用大致为 O(L) 而不是 O(L^2)。
        # 相当于在我们迭代的维度的块上运行一个 for 循环，
        # 其中 chunk_size 是块的大小，比如如果设置为 128，则意味着解析长度为 128 的块。
        self.chunk_size = chunk_size
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles):
        """
        Inputs:
          seq_feats: B x L x C tensor of sequence features pair_feats: B x L x L x C tensor of pair features residx: B
          x L long tensor giving the position in the sequence mask: B x L boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """

        # 获取输入张量 seq_feats 的设备信息
        device = seq_feats.device
        # 初始化原始的序列特征和对特征
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        # 如果未提供 no_recycles 参数，则使用配置中的最大循环次数
        if no_recycles is None:
            no_recycles = self.config.max_recycles
        else:
            # 如果提供了 no_recycles 参数，确保其不为负数
            if no_recycles < 0:
                raise ValueError("Number of recycles must not be negative.")
            # 将 no_recycles 值增加 1，因为第一个 'recycle' 是通过模型的标准前向传播
            no_recycles += 1

        def trunk_iter(s, z, residx, mask):
            # 为 z 添加位置编码嵌入
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            # 遍历所有的块（blocks），每个块执行一次
            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=self.chunk_size)
            return s, z

        # 将初始的序列特征和对特征赋值给 s_s 和 s_z
        s_s = s_s_0
        s_z = s_z_0
        # 初始化用于循环的张量
        recycle_s = torch.zeros_like(s_s)
        recycle_z = torch.zeros_like(s_z)
        recycle_bins = torch.zeros(*s_z.shape[:-1], device=device, dtype=torch.int64)

        # 执行循环指定的次数（no_recycles）
        for recycle_idx in range(no_recycles):
            with ContextManagers([] if recycle_idx == no_recycles - 1 else [torch.no_grad()]):
                # === Recycling ===
                # 对 recycle_s 和 recycle_z 进行归一化处理，并转移到指定设备上
                recycle_s = self.recycle_s_norm(recycle_s.detach()).to(device)
                recycle_z = self.recycle_z_norm(recycle_z.detach()).to(device)
                # 添加距离约束到 recycle_z
                recycle_z += self.recycle_disto(recycle_bins.detach()).to(device)

                # 执行 trunk_iter 函数，更新 s_s 和 s_z
                s_s, s_z = trunk_iter(s_s_0 + recycle_s, s_z_0 + recycle_z, residx, mask)

                # === Structure module ===
                # 使用结构模块生成结构预测，传入单体和对体的转换结果，真实的氨基酸序列和掩码
                structure = self.structure_module(
                    {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
                    true_aa,
                    mask.float(),
                )

                # 更新 recycle_s 和 recycle_z 为当前的 s_s 和 s_z
                recycle_s = s_s
                recycle_z = s_z
                # 计算距离直方图所需的 bins，调用 distogram 方法
                recycle_bins = EsmFoldingTrunk.distogram(
                    structure["positions"][-1][:, :, :3],
                    3.375,
                    21.375,
                    self.recycle_bins,
                )

        # 将最终的 s_s 和 s_z 存储在结构对象中，并返回结构对象
        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure

    @staticmethod
    def distogram(coords, min_bin, max_bin, num_bins):
        # 计算距离直方图，输入参数分别为坐标数组，最小bin值，最大bin值，bin的数量

        # 使用 torch.linspace 在设备上生成一组均匀间隔的边界值
        boundaries = torch.linspace(
            min_bin,
            max_bin,
            num_bins - 1,
            device=coords.device,
        )
        boundaries = boundaries**2  # 将边界值平方

        # 将输入的坐标数组按照特定维度切分成 N, CA, C 坐标数组
        N, CA, C = [x.squeeze(-2) for x in coords.chunk(3, dim=-2)]

        # 推断出 CB 坐标
        b = CA - N
        c = C - CA
        a = b.cross(c, dim=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        # 计算 CB 坐标之间的距离的平方和，得到距离矩阵
        dists = (CB[..., None, :, :] - CB[..., :, None, :]).pow(2).sum(dim=-1, keepdims=True)

        # 计算每对 CB 坐标之间的距离所属的 bin 编号
        bins = torch.sum(dists > boundaries, dim=-1)  # 得到距离直方图的矩阵

        return bins
# 导入函数用于添加文档字符串（docstring）信息到类
@add_start_docstrings(
    """
    ESMForProteinFolding is the HuggingFace port of the original ESMFold model. It consists of an ESM-2 "stem" followed
    by a protein folding "head", although unlike most other output heads, this "head" is similar in size and runtime to
    the rest of the model combined! It outputs a dictionary containing predicted structural information about the input
    protein(s).
    """,
    ESM_START_DOCSTRING,
)
# 定义 EsmForProteinFolding 类，继承自 EsmPreTrainedModel 类
class EsmForProteinFolding(EsmPreTrainedModel):
    # 不需要拆分的模块列表，用于模型训练和推理阶段的处理
    _no_split_modules = ["EsmFoldStructureModule", "EsmFoldTriangularSelfAttentionBlock"]
    # 初始化函数，接受一个配置参数，并调用父类的初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)

        # 将配置参数保存到当前对象的属性中
        self.config = config

        # 定义直方图分箱的数量为64
        self.distogram_bins = 64

        # 创建一个 EsmModel 对象，禁用添加池化层的选项
        self.esm = EsmModel(config, add_pooling_layer=False)

        # 将 EsmModel 的参数设置为不需要梯度
        self.esm.requires_grad_(False)
        
        # 如果配置中指定使用 fp16 模式，则将 EsmModel 切换为半精度
        if self.config.esmfold_config.fp16_esm:
            self.esm.half()

        # 设置 ESM 特征的维度为配置中指定的隐藏层大小
        self.esm_feats = self.config.hidden_size

        # 计算 ESM 注意力头的数量
        self.esm_attns = self.config.num_hidden_layers * self.config.num_attention_heads

        # 设置 ESM 层数为配置中指定的隐藏层数
        self.esm_layers = self.config.num_hidden_layers

        # 使用从词汇表中得到的映射创建一个缓冲区，用于将序列特征映射到 ESM 的表示
        self.register_buffer("af2_to_esm", self._af2_to_esm_from_vocab_list(config.vocab_list))

        # 创建一个可学习的参数，用于结合不同层的 ESM 输出
        self.esm_s_combine = nn.Parameter(torch.zeros(self.esm_layers + 1))

        # 从配置中获取 ESMFold 的 trunk 配置
        trunk_config = self.config.esmfold_config.trunk

        # 定义序列状态维度和配对状态维度
        c_s = trunk_config.sequence_state_dim
        c_z = trunk_config.pairwise_state_dim

        # 定义一个序列，包含一系列的层次归一化和线性变换，用于将 ESM 特征映射到序列状态维度
        self.esm_s_mlp = nn.Sequential(
            LayerNorm(self.esm_feats),
            nn.Linear(self.esm_feats, c_s),
            nn.ReLU(),
            nn.Linear(c_s, c_s),
        )

        # 定义序列的嵌入标记数量，包括填充标记、未知残基标记和掩码标记
        self.n_tokens_embed = residue_constants.restype_num + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1

        # 获取词汇表中特定标记的索引，如 "<cls>", "<mask>", "<eos>", "<pad>"
        self.esm_dict_cls_idx = self.config.vocab_list.index("<cls>")
        self.esm_dict_mask_idx = self.config.vocab_list.index("<mask>")
        self.esm_dict_eos_idx = self.config.vocab_list.index("<eos>")
        self.esm_dict_padding_idx = self.config.vocab_list.index("<pad>")

        # 如果配置指定要嵌入氨基酸标记，则创建一个嵌入层
        if self.config.esmfold_config.embed_aa:
            self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        # 创建 ESMFold 的 trunk 部分
        self.trunk = EsmFoldingTrunk(trunk_config)

        # 定义直方图头部的线性层和蛋白质结构的头部线性层
        self.distogram_head = nn.Linear(c_z, self.distogram_bins)
        self.ptm_head = nn.Linear(c_z, self.distogram_bins)
        self.lm_head = nn.Linear(c_s, self.n_tokens_embed)

        # 定义 LDDT 预测的分箱数量
        self.lddt_bins = 50

        # 获取 trunk 配置中结构模块的配置信息
        structure_module_config = trunk_config.structure_module

        # 定义 LDDT 预测头部的线性层序列
        self.lddt_head = nn.Sequential(
            nn.LayerNorm(structure_module_config.sequence_dim),
            nn.Linear(structure_module_config.sequence_dim, self.config.esmfold_config.lddt_head_hid_dim),
            nn.Linear(self.config.esmfold_config.lddt_head_hid_dim, self.config.esmfold_config.lddt_head_hid_dim),
            nn.Linear(self.config.esmfold_config.lddt_head_hid_dim, 37 * self.lddt_bins),
        )
    # 定义模型的前向传播方法，接受多个输入张量作为参数
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        masking_pattern: Optional[torch.Tensor] = None,
        num_recycles: Optional[int] = None,
    ):
        # 省略的是前向传播的具体实现，根据输入参数计算模型输出
        pass

    # 将从AF2空间到ESM空间的索引映射转换为与输入设备相同的设备，以避免设备上的索引错误
    def af2_idx_to_esm_idx(self, aa, mask):
        if self.af2_to_esm.device != aa.device:
            self.af2_to_esm = self.af2_to_esm.to(aa.device)
        # 将aa中的每个元素加一，并将非1的位置用0填充
        aa = (aa + 1).masked_fill(mask != 1, 0)
        # 使用af2_to_esm映射aa中的每个元素，返回对应的索引
        return self.af2_to_esm[aa]

    # 计算语言模型的表示，接受ESM的张量作为输入，并返回处理后的张量
    def compute_language_model_representations(self, esmaa: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        B, L = esmaa.shape  # B为批次大小，L为序列长度

        # 如果配置要求绕过语言模型，则返回全零的张量作为输出
        if self.config.esmfold_config.bypass_lm:
            esm_s = torch.zeros(B, L, self.esm_s_combine.size[0], -1, self.esm_feats, device=device)
            return esm_s

        # 获取开始和结束的特殊标记索引
        bosi, eosi = self.esm_dict_cls_idx, self.esm_dict_eos_idx
        # 在序列的开头和结尾添加特殊标记索引
        bos = esmaa.new_full((B, 1), bosi)
        eos = esmaa.new_full((B, 1), self.esm_dict_padding_idx)
        esmaa = torch.cat([bos, esmaa, eos], dim=1)
        # 在推断过程中，使用第一个填充索引作为结束标记
        esmaa[range(B), (esmaa != 1).sum(1)] = eosi

        # 计算ESM模型的隐藏状态，返回多层隐藏状态的张量
        esm_hidden_states = self.esm(esmaa, attention_mask=esmaa != 1, output_hidden_states=True)["hidden_states"]
        esm_s = torch.stack(esm_hidden_states, dim=2)

        # 移除序列开头和结尾的特殊标记
        esm_s = esm_s[:, 1:-1]  # B, L, nLayers, C

        return esm_s

    # 对输入的aa和esmaa张量进行BERT掩码操作，并返回处理后的新张量
    def bert_mask(self, aa, esmaa, mask, pattern):
        new_aa = aa.clone()
        target = aa.clone()
        new_esmaa = esmaa.clone()
        # 将pattern为1的位置在new_aa中替换为mask_idx
        new_aa[pattern == 1] = self.mask_idx
        # 将pattern不为1的位置在target中替换为0
        target[pattern != 1] = 0
        # 将pattern为1的位置在new_esmaa中替换为esm_dict_mask_idx
        new_esmaa[pattern == 1] = self.esm_dict_mask_idx
        return new_aa, new_esmaa, target

    # 声明推断方法，接受序列文本或列表作为输入，不进行梯度计算
    @torch.no_grad()
    def infer(
        self,
        seqs: Union[str, List[str]],
        position_ids=None,
    ):
        if isinstance(seqs, str):
            # 如果输入的序列是字符串，则转换为单元素列表
            lst = [seqs]
        else:
            # 否则，直接使用输入的序列列表
            lst = seqs
        # 获取模型参数的设备信息
        device = next(self.parameters()).device
        # 使用自定义函数将输入序列转换为 one-hot 编码的张量
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
        # 为每个序列生成掩码张量
        mask = collate_dense_tensors([aatype.new_ones(len(seq)) for seq in lst])
        # 生成位置 ID 张量，如果未提供则创建一个新的
        position_ids = (
            torch.arange(aatype.shape[1], device=device).expand(len(lst), -1)
            if position_ids is None
            else position_ids.to(device)
        )
        # 如果位置 ID 张量的维度为 1，则扩展为二维张量
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        # 调用模型的 forward 方法进行推断
        return self.forward(
            aatype,
            mask,
            position_ids=position_ids,
        )

    @staticmethod
    def output_to_pdb(output: Dict) -> List[str]:
        """Returns the pdb (file) string from the model given the model output."""
        # 将模型输出中的张量转移到 CPU 上，并转换为 numpy 数组
        output = {k: v.to("cpu").numpy() for k, v in output.items()}
        pdbs = []
        # 获取最终的原子位置和掩码信息
        final_atom_positions = atom14_to_atom37(output["positions"][-1], output)
        final_atom_mask = output["atom37_atom_exists"]
        # 遍历每个样本的预测结果，并生成相应的 PDB 对象
        for i in range(output["aatype"].shape[0]):
            aa = output["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = output["residue_index"][i] + 1
            # 使用预测的信息创建 OFProtein 对象
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=output["plddt"][i],
            )
            # 将生成的 PDB 对象转换为 PDB 文件格式字符串并添加到列表中
            pdbs.append(to_pdb(pred))
        return pdbs

    def infer_pdb(self, seqs, *args, **kwargs) -> str:
        """Returns the pdb (file) string from the model given an input sequence."""
        # 确保输入序列为字符串
        assert isinstance(seqs, str)
        # 调用 infer 方法进行推断
        output = self.infer(seqs, *args, **kwargs)
        # 将推断结果转换为 PDB 文件格式字符串并返回第一个结果
        return self.output_to_pdb(output)[0]

    def infer_pdbs(self, seqs: List[str], *args, **kwargs) -> List[str]:
        """Returns the pdb (file) string from the model given an input sequence."""
        # 调用 infer 方法进行推断
        output = self.infer(seqs, *args, **kwargs)
        # 将推断结果转换为 PDB 文件格式字符串列表并返回
        return self.output_to_pdb(output)
```