# `.\diffusers\models\attention_processor.py`

```
# 版权声明，标明该文件的版权归 HuggingFace 团队所有
# 该文件根据 Apache 2.0 许可证进行许可
# 在遵守许可证的情况下，您可以使用该文件
# 许可证的副本可以在以下网址获取
# http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则软件按 "现状" 提供，不附带任何明示或暗示的担保
# 请参阅许可证以了解有关权限和限制的具体信息

import inspect  # 导入 inspect 模块，用于获取对象的信息
import math  # 导入 math 模块，提供数学函数
from typing import Callable, List, Optional, Tuple, Union  # 导入类型提示相关的类型

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的神经网络功能模块，并重命名为 F
from torch import nn  # 从 PyTorch 导入 nn 模块，提供神经网络的构建块

from ..image_processor import IPAdapterMaskProcessor  # 从上层模块导入 IPAdapterMaskProcessor
from ..utils import deprecate, logging  # 从上层模块导入弃用和日志记录功能
from ..utils.import_utils import is_torch_npu_available, is_xformers_available  # 导入检查 PyTorch NPU 和 xformers 可用性的工具
from ..utils.torch_utils import is_torch_version, maybe_allow_in_graph  # 导入与 PyTorch 版本和图形相关的工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例，便于记录日志信息

if is_torch_npu_available():  # 检查是否可以使用 PyTorch NPU
    import torch_npu  # 如果可用，则导入 torch_npu 模块

if is_xformers_available():  # 检查是否可以使用 xformers 库
    import xformers  # 如果可用，导入 xformers 模块
    import xformers.ops  # 导入 xformers 中的操作模块
else:  # 如果 xformers 不可用
    xformers = None  # 将 xformers 设为 None

@maybe_allow_in_graph  # 装饰器，可能允许在图中使用该类
class Attention(nn.Module):  # 定义 Attention 类，继承自 nn.Module
    r"""  # 文档字符串，描述该类是一个交叉注意力层
    A cross attention layer.
    """

    def __init__(  # 初始化方法，定义构造函数
        self,
        query_dim: int,  # 查询维度，类型为整数
        cross_attention_dim: Optional[int] = None,  # 可选的交叉注意力维度，默认为 None
        heads: int = 8,  # 注意力头的数量，默认为 8
        kv_heads: Optional[int] = None,  # 可选的键值头数量，默认为 None
        dim_head: int = 64,  # 每个头的维度，默认为 64
        dropout: float = 0.0,  # dropout 概率，默认为 0.0
        bias: bool = False,  # 是否使用偏置，默认为 False
        upcast_attention: bool = False,  # 是否上升注意力精度，默认为 False
        upcast_softmax: bool = False,  # 是否上升 softmax 精度，默认为 False
        cross_attention_norm: Optional[str] = None,  # 可选的交叉注意力归一化方式，默认为 None
        cross_attention_norm_num_groups: int = 32,  # 交叉注意力归一化的组数量，默认为 32
        qk_norm: Optional[str] = None,  # 可选的查询键归一化方式，默认为 None
        added_kv_proj_dim: Optional[int] = None,  # 可选的添加键值投影维度，默认为 None
        added_proj_bias: Optional[bool] = True,  # 是否为添加的投影使用偏置，默认为 True
        norm_num_groups: Optional[int] = None,  # 可选的归一化组数量，默认为 None
        spatial_norm_dim: Optional[int] = None,  # 可选的空间归一化维度，默认为 None
        out_bias: bool = True,  # 是否使用输出偏置，默认为 True
        scale_qk: bool = True,  # 是否缩放查询和键，默认为 True
        only_cross_attention: bool = False,  # 是否仅使用交叉注意力，默认为 False
        eps: float = 1e-5,  # 为数值稳定性引入的微小常数，默认为 1e-5
        rescale_output_factor: float = 1.0,  # 输出重标定因子，默认为 1.0
        residual_connection: bool = False,  # 是否使用残差连接，默认为 False
        _from_deprecated_attn_block: bool = False,  # 可选参数，指示是否来自弃用的注意力块，默认为 False
        processor: Optional["AttnProcessor"] = None,  # 可选的处理器，默认为 None
        out_dim: int = None,  # 输出维度，默认为 None
        context_pre_only=None,  # 上下文前处理，默认为 None
        pre_only=False,  # 是否仅进行前处理，默认为 False
    # 设置是否使用来自 `torch_npu` 的 npu flash attention
    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        r"""
        设置是否使用来自 `torch_npu` 的 npu flash attention。
        """
        # 如果选择使用 npu flash attention
        if use_npu_flash_attention:
            # 创建 NPU 注意力处理器实例
            processor = AttnProcessorNPU()
        else:
            # 设置注意力处理器
            # 默认情况下使用 AttnProcessor2_0，当使用 torch 2.x 时，
            # 它利用 torch.nn.functional.scaled_dot_product_attention 进行本地 Flash/内存高效注意力
            # 仅在其具有默认 `scale` 参数时适用。TODO: 在迁移到 torch 2.1 时移除 scale_qk 检查
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        # 设置当前的处理器
        self.set_processor(processor)
    
    # 设置是否使用内存高效的 xformers 注意力
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        pass  # 此处可能缺少实现
    
    # 设置注意力计算的切片大小
    def set_attention_slice(self, slice_size: int) -> None:
        r"""
        设置注意力计算的切片大小。
    
        参数：
            slice_size (`int`):
                用于注意力计算的切片大小。
        """
        # 如果切片大小不为 None 且大于可切片头维度
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            # 抛出值错误，切片大小必须小于或等于可切片头维度
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")
    
        # 如果切片大小不为 None 且添加的 kv 投影维度不为 None
        if slice_size is not None and self.added_kv_proj_dim is not None:
            # 创建带切片大小的 KV 处理器实例
            processor = SlicedAttnAddedKVProcessor(slice_size)
        # 如果切片大小不为 None
        elif slice_size is not None:
            # 创建带切片大小的注意力处理器实例
            processor = SlicedAttnProcessor(slice_size)
        # 如果添加的 kv 投影维度不为 None
        elif self.added_kv_proj_dim is not None:
            # 创建 KV 注意力处理器实例
            processor = AttnAddedKVProcessor()
        else:
            # 设置注意力处理器
            # 默认情况下使用 AttnProcessor2_0，当使用 torch 2.x 时，
            # 它利用 torch.nn.functional.scaled_dot_product_attention 进行本地 Flash/内存高效注意力
            # 仅在其具有默认 `scale` 参数时适用。TODO: 在迁移到 torch 2.1 时移除 scale_qk 检查
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
    
        # 设置当前的处理器
        self.set_processor(processor)
    # 设置要使用的注意力处理器
    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        设置要使用的注意力处理器。
    
        参数：
            processor (`AttnProcessor`):
                要使用的注意力处理器。
        """
        # 如果当前处理器在 `self._modules` 中，且传入的 `processor` 不在其中，则需要从 `self._modules` 中移除当前处理器
        if (
            hasattr(self, "processor")  # 检查当前对象是否有处理器属性
            and isinstance(self.processor, torch.nn.Module)  # 确保当前处理器是一个 PyTorch 模块
            and not isinstance(processor, torch.nn.Module)  # 检查传入的处理器不是 PyTorch 模块
        ):
            # 记录日志，指出将移除已训练权重的处理器
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            # 从模块中移除当前处理器
            self._modules.pop("processor")
    
        # 设置当前对象的处理器为传入的处理器
        self.processor = processor
    
    # 获取正在使用的注意力处理器
    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        获取正在使用的注意力处理器。
    
        参数：
            return_deprecated_lora (`bool`, *可选*, 默认为 `False`):
                设置为 `True` 以返回过时的 LoRA 注意力处理器。
    
        返回：
            "AttentionProcessor": 正在使用的注意力处理器。
        """
        # 如果不需要返回过时的 LoRA 处理器，则返回当前处理器
        if not return_deprecated_lora:
            return self.processor
    
    # 前向传播方法，处理输入的隐藏状态
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量
        **cross_attention_kwargs,  # 可变参数，用于交叉注意力
    ) -> torch.Tensor:
        r"""  # 文档字符串，描述此方法的功能和参数
        The forward method of the `Attention` class.

        Args:  # 参数说明
            hidden_states (`torch.Tensor`):  # 查询的隐藏状态，类型为张量
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):  # 编码器的隐藏状态，可选参数
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):  # 注意力掩码，可选参数
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:  # 额外的关键字参数，传递给交叉注意力
                Additional keyword arguments to pass along to the cross attention.

        Returns:  # 返回值说明
            `torch.Tensor`: The output of the attention layer.  # 返回注意力层的输出
        """
        # `Attention` 类可以调用不同的注意力处理器/函数
        # 这里我们简单地将所有张量传递给所选的处理器类
        # 对于此处定义的标准处理器，`**cross_attention_kwargs` 是空的

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())  # 获取处理器调用方法的参数名集合
        quiet_attn_parameters = {"ip_adapter_masks"}  # 定义不需要警告的参数集合
        unused_kwargs = [  # 筛选出未被使用的关键字参数
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:  # 如果存在未使用的关键字参数
            logger.warning(  # 记录警告日志
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}  # 过滤出有效的关键字参数

        return self.processor(  # 调用处理器并返回结果
            self,
            hidden_states,  # 传递隐藏状态
            encoder_hidden_states=encoder_hidden_states,  # 传递编码器的隐藏状态
            attention_mask=attention_mask,  # 传递注意力掩码
            **cross_attention_kwargs,  # 解包有效的额外关键字参数
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:  # 定义方法，输入张量并返回处理后的张量
        r"""  # 文档字符串，描述此方法的功能和参数
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`  # 将张量从 `[batch_size, seq_len, dim]` 重新形状为 `[batch_size // heads, seq_len, dim * heads]`，`heads` 为初始化时的头数量
        is the number of heads initialized while constructing the `Attention` class.

        Args:  # 参数说明
            tensor (`torch.Tensor`): The tensor to reshape.  # 要重新形状的张量

        Returns:  # 返回值说明
            `torch.Tensor`: The reshaped tensor.  # 返回重新形状后的张量
        """
        head_size = self.heads  # 获取头的数量
        batch_size, seq_len, dim = tensor.shape  # 解包输入张量的形状
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)  # 重新调整张量的形状
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)  # 调整维度顺序并重新形状
        return tensor  # 返回处理后的张量
    # 将输入张量从形状 `[batch_size, seq_len, dim]` 转换为 `[batch_size, seq_len, heads, dim // heads]`
    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
            r"""
            将张量从 `[batch_size, seq_len, dim]` 重塑为 `[batch_size, seq_len, heads, dim // heads]`，其中 `heads` 是
            在构造 `Attention` 类时初始化的头数。
    
            参数：
                tensor (`torch.Tensor`): 要重塑的张量。
                out_dim (`int`, *可选*, 默认值为 `3`): 张量的输出维度。如果为 `3`，则张量被
                    重塑为 `[batch_size * heads, seq_len, dim // heads]`。
    
            返回：
                `torch.Tensor`: 重塑后的张量。
            """
            # 获取头的数量
            head_size = self.heads
            # 检查输入张量的维度，如果是三维则提取形状信息
            if tensor.ndim == 3:
                batch_size, seq_len, dim = tensor.shape
                extra_dim = 1
            else:
                # 如果不是三维，提取四维形状信息
                batch_size, extra_dim, seq_len, dim = tensor.shape
            # 重塑张量为 `[batch_size, seq_len * extra_dim, head_size, dim // head_size]`
            tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
            # 调整张量维度顺序为 `[batch_size, heads, seq_len * extra_dim, dim // heads]`
            tensor = tensor.permute(0, 2, 1, 3)
    
            # 如果输出维度为 3，进一步重塑张量为 `[batch_size * heads, seq_len * extra_dim, dim // heads]`
            if out_dim == 3:
                tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)
    
            # 返回重塑后的张量
            return tensor
    
    # 计算注意力得分的函数
    def get_attention_scores(
            self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            r"""
            计算注意力得分。
    
            参数：
                query (`torch.Tensor`): 查询张量。
                key (`torch.Tensor`): 键张量。
                attention_mask (`torch.Tensor`, *可选*): 使用的注意力掩码。如果为 `None`，则不应用掩码。
    
            返回：
                `torch.Tensor`: 注意力概率/得分。
            """
            # 获取查询张量的数据类型
            dtype = query.dtype
            # 如果需要上升类型，将查询和键张量转换为浮点型
            if self.upcast_attention:
                query = query.float()
                key = key.float()
    
            # 如果没有提供注意力掩码，创建空的输入张量
            if attention_mask is None:
                baddbmm_input = torch.empty(
                    query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
                )
                # 设置 beta 为 0
                beta = 0
            else:
                # 如果有注意力掩码，将其用作输入
                baddbmm_input = attention_mask
                # 设置 beta 为 1
                beta = 1
    
            # 计算注意力得分
            attention_scores = torch.baddbmm(
                baddbmm_input,
                query,
                key.transpose(-1, -2),
                beta=beta,
                alpha=self.scale,
            )
            # 删除临时的输入张量
            del baddbmm_input
    
            # 如果需要上升类型，将注意力得分转换为浮点型
            if self.upcast_softmax:
                attention_scores = attention_scores.float()
    
            # 计算注意力概率
            attention_probs = attention_scores.softmax(dim=-1)
            # 删除注意力得分张量
            del attention_scores
    
            # 将注意力概率转换回原始数据类型
            attention_probs = attention_probs.to(dtype)
    
            # 返回注意力概率
            return attention_probs
    
    # 准备注意力掩码的函数
    def prepare_attention_mask(
            self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:  # 定义一个函数的返回类型为 torch.Tensor
        r"""  # 开始文档字符串，描述函数的作用和参数
        Prepare the attention mask for the attention computation.  # 准备注意力计算的注意力掩码
        Args:  # 参数说明
            attention_mask (`torch.Tensor`):  # 输入参数，注意力掩码，类型为 torch.Tensor
                The attention mask to prepare.  # 待准备的注意力掩码
            target_length (`int`):  # 输入参数，目标长度，类型为 int
                The target length of the attention mask. This is the length of the attention mask after padding.  # 注意力掩码的目标长度，经过填充后的长度
            batch_size (`int`):  # 输入参数，批处理大小，类型为 int
                The batch size, which is used to repeat the attention mask.  # 批处理大小，用于重复注意力掩码
            out_dim (`int`, *optional*, defaults to `3`):  # 可选参数，输出维度，类型为 int，默认为 3
                The output dimension of the attention mask. Can be either `3` or `4`.  # 注意力掩码的输出维度，可以是 3 或 4
        Returns:  # 返回说明
            `torch.Tensor`: The prepared attention mask.  # 返回准备好的注意力掩码，类型为 torch.Tensor
        """  # 结束文档字符串
        head_size = self.heads  # 获取头部大小，来自类的属性 heads
        if attention_mask is None:  # 检查注意力掩码是否为 None
            return attention_mask  # 如果是 None，直接返回

        current_length: int = attention_mask.shape[-1]  # 获取当前注意力掩码的长度
        if current_length != target_length:  # 检查当前长度是否与目标长度不匹配
            if attention_mask.device.type == "mps":  # 如果设备类型是 "mps"
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.  # HACK: MPS 不支持填充超过输入张量的维度
                # Instead, we can manually construct the padding tensor.  # 所以我们手动构建填充张量
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)  # 定义填充张量的形状
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)  # 创建全零填充张量
                attention_mask = torch.cat([attention_mask, padding], dim=2)  # 在最后一个维度上拼接填充张量
            else:  # 如果不是 "mps" 设备
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:  # TODO: 对于如 stable-diffusion 的管道，填充交叉注意力掩码
                #       we want to instead pad by (0, remaining_length), where remaining_length is:  # 我们希望用 (0, remaining_length) 填充，其中 remaining_length 是
                #       remaining_length: int = target_length - current_length  # remaining_length 的计算
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding  # TODO: 重新启用相关测试
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)  # 用零填充注意力掩码到目标长度

        if out_dim == 3:  # 如果输出维度是 3
            if attention_mask.shape[0] < batch_size * head_size:  # 检查注意力掩码的第一维是否小于批处理大小乘以头部大小
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)  # 在第一维上重复注意力掩码
        elif out_dim == 4:  # 如果输出维度是 4
            attention_mask = attention_mask.unsqueeze(1)  # 在第一维增加一个维度
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)  # 在第二维上重复注意力掩码

        return attention_mask  # 返回准备好的注意力掩码
    # 定义一个函数用于规范化编码器的隐藏状态，接受一个张量作为输入并返回一个张量
    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        规范化编码器隐藏状态。构造 `Attention` 类时需要指定 `self.norm_cross`。

        参数:
            encoder_hidden_states (`torch.Tensor`): 编码器的隐藏状态。

        返回:
            `torch.Tensor`: 规范化后的编码器隐藏状态。
        """
        # 确保在调用此方法之前已定义 `self.norm_cross`
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        # 检查 `self.norm_cross` 是否为 LayerNorm 类型
        if isinstance(self.norm_cross, nn.LayerNorm):
            # 对编码器隐藏状态进行层归一化
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        # 检查 `self.norm_cross` 是否为 GroupNorm 类型
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # GroupNorm 沿通道维度进行归一化，并期望输入形状为 (N, C, *)。
            # 此时我们希望沿隐藏维度进行归一化，因此需要调整形状
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)  # 转置张量以调整维度顺序
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)  # 对转置后的张量进行归一化
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)  # 再次转置回原始顺序
        else:
            # 如果 `self.norm_cross` 既不是 LayerNorm 也不是 GroupNorm，则触发断言失败
            assert False

        # 返回规范化后的编码器隐藏状态
        return encoder_hidden_states

    # 该装饰器在计算图中禁止梯度计算，以节省内存和加快推理速度
    @torch.no_grad()
    # 定义一个融合投影的方法，默认参数 fuse 为 True
    def fuse_projections(self, fuse=True):
        # 获取 to_q 权重的设备信息
        device = self.to_q.weight.data.device
        # 获取 to_q 权重的数据类型
        dtype = self.to_q.weight.data.dtype

        # 如果不是交叉注意力
        if not self.is_cross_attention:
            # 获取权重矩阵的拼接
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            # 输入特征数为拼接后权重的列数
            in_features = concatenated_weights.shape[1]
            # 输出特征数为拼接后权重的行数
            out_features = concatenated_weights.shape[0]

            # 创建一个新的线性投影层并复制权重
            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            # 复制拼接后的权重到新的层
            self.to_qkv.weight.copy_(concatenated_weights)
            # 如果使用偏置
            if self.use_bias:
                # 拼接 q、k、v 的偏置
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                # 复制拼接后的偏置到新的层
                self.to_qkv.bias.copy_(concatenated_bias)

        # 如果是交叉注意力
        else:
            # 获取 k 和 v 权重的拼接
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            # 输入特征数为拼接后权重的列数
            in_features = concatenated_weights.shape[1]
            # 输出特征数为拼接后权重的行数
            out_features = concatenated_weights.shape[0]

            # 创建一个新的线性投影层并复制权重
            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            # 复制拼接后的权重到新的层
            self.to_kv.weight.copy_(concatenated_weights)
            # 如果使用偏置
            if self.use_bias:
                # 拼接 k 和 v 的偏置
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                # 复制拼接后的偏置到新的层
                self.to_kv.bias.copy_(concatenated_bias)

        # 处理 SD3 和其他添加的投影
        if hasattr(self, "add_q_proj") and hasattr(self, "add_k_proj") and hasattr(self, "add_v_proj"):
            # 获取额外投影的权重拼接
            concatenated_weights = torch.cat(
                [self.add_q_proj.weight.data, self.add_k_proj.weight.data, self.add_v_proj.weight.data]
            )
            # 输入特征数为拼接后权重的列数
            in_features = concatenated_weights.shape[1]
            # 输出特征数为拼接后权重的行数
            out_features = concatenated_weights.shape[0]

            # 创建一个新的线性投影层并复制权重
            self.to_added_qkv = nn.Linear(
                in_features, out_features, bias=self.added_proj_bias, device=device, dtype=dtype
            )
            # 复制拼接后的权重到新的层
            self.to_added_qkv.weight.copy_(concatenated_weights)
            # 如果使用偏置
            if self.added_proj_bias:
                # 拼接额外投影的偏置
                concatenated_bias = torch.cat(
                    [self.add_q_proj.bias.data, self.add_k_proj.bias.data, self.add_v_proj.bias.data]
                )
                # 复制拼接后的偏置到新的层
                self.to_added_qkv.bias.copy_(concatenated_bias)

        # 将融合状态存储到属性中
        self.fused_projections = fuse
# 定义一个处理器类，用于执行与注意力相关的计算
class AttnProcessor:
    r"""
    默认处理器，用于执行与注意力相关的计算。
    """

    # 实现可调用方法，处理注意力计算
    def __call__(
        self,
        attn: Attention,  # 注意力对象
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态（可选）
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码（可选）
        temb: Optional[torch.Tensor] = None,  # 额外的时间嵌入（可选）
        *args,  # 额外的位置参数
        **kwargs,  # 额外的关键字参数
    ) -> torch.Tensor:  # 返回处理后的张量
        # 检查是否有额外参数或已弃用的 scale 参数
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 构建弃用警告消息
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用弃用处理函数
            deprecate("scale", "1.0.0", deprecation_message)

        # 初始化残差为隐藏状态
        residual = hidden_states

        # 如果空间归一化存在，则应用于隐藏状态
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # 获取输入张量的维度
        input_ndim = hidden_states.ndim

        # 如果输入是四维的，则调整形状
        if input_ndim == 4:
            # 解包隐藏状态的形状
            batch_size, channel, height, width = hidden_states.shape
            # 重新调整形状为(batch_size, channel, height*width)并转置
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 根据编码器隐藏状态的存在与否，获取批次大小和序列长度
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 如果组归一化存在，则应用于隐藏状态
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)

        # 如果没有编码器隐藏状态，使用隐藏状态作为编码器隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要规范化编码器隐藏状态，则应用规范化
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 从编码器隐藏状态中获取键和值
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # 将查询、键和值转换为批次维度
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 通过注意力分数加权求值
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态转换回头维度
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 如果输入是四维的，调整回原始形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 如果存在残差连接，则将残差加回隐藏状态
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # 将隐藏状态归一化到输出因子
        hidden_states = hidden_states / attn.rescale_output_factor

        # 返回最终的隐藏状态
        return hidden_states


# 定义一个处理器类，用于实现自定义扩散方法的注意力
class CustomDiffusionAttnProcessor(nn.Module):
    r"""
    实现自定义扩散方法的注意力处理器。
    # 定义参数说明
    Args:
        train_kv (`bool`, defaults to `True`):  # 是否重新训练对应于文本特征的键值矩阵
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):  # 是否重新训练对应于潜在图像特征的查询矩阵
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):  # 注意力层的隐藏大小
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):  # 编码器隐藏状态中的通道数量
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):  # 是否在 `train_q_out` 中包含偏置参数
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):  # 使用的 dropout 概率
            The dropout probability to use.
    """

    # 初始化方法
    def __init__(
        self,  # 初始化方法的第一个参数，表示对象本身
        train_kv: bool = True,  # 设置键值矩阵训练的默认值为 True
        train_q_out: bool = True,  # 设置查询矩阵训练的默认值为 True
        hidden_size: Optional[int] = None,  # 隐藏层大小，默认为 None
        cross_attention_dim: Optional[int] = None,  # 跨注意力维度，默认为 None
        out_bias: bool = True,  # 输出偏置参数的默认值为 True
        dropout: float = 0.0,  # 默认的 dropout 概率为 0.0
    ):
        super().__init__()  # 调用父类的初始化方法
        self.train_kv = train_kv  # 保存键值训练标志
        self.train_q_out = train_q_out  # 保存查询输出训练标志

        self.hidden_size = hidden_size  # 保存隐藏层大小
        self.cross_attention_dim = cross_attention_dim  # 保存跨注意力维度

        # `_custom_diffusion` id 方便序列化和加载
        if self.train_kv:  # 如果需要训练键值
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)  # 创建键的线性层
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)  # 创建值的线性层
        if self.train_q_out:  # 如果需要训练查询输出
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)  # 创建查询的线性层
            self.to_out_custom_diffusion = nn.ModuleList([])  # 初始化输出层的模块列表
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))  # 添加线性输出层
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))  # 添加 dropout 层

    # 可调用方法
    def __call__(  # 定义对象被调用时的行为
        self,  # 第一个参数，表示对象本身
        attn: Attention,  # 注意力对象
        hidden_states: torch.Tensor,  # 隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
    # 返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 获取隐藏状态的批量大小和序列长度
        batch_size, sequence_length, _ = hidden_states.shape
        # 准备注意力掩码以适应当前批量和序列长度
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        # 如果需要训练查询输出，则使用自定义扩散进行转换
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            # 否则使用标准的查询转换
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))
    
        # 检查编码器隐藏状态是否为 None
        if encoder_hidden_states is None:
            # 如果是，则不进行交叉注意力
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            # 否则，启用交叉注意力
            crossattn = True
            # 如果需要归一化编码器隐藏状态，则进行归一化
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
        # 如果需要训练键值对
        if self.train_kv:
            # 使用自定义扩散获取键和值
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
            # 将键和值转换为查询的权重数据类型
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            # 否则使用标准的键和值转换
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
    
        # 如果进行交叉注意力
        if crossattn:
            # 创建与键相同形状的张量以进行detach操作
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            # 应用detach逻辑以阻止梯度流动
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()
    
        # 将查询、键和值转换为批次维度
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
    
        # 计算注意力分数
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 使用注意力分数和值进行批量矩阵乘法
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态转换回头维度
        hidden_states = attn.batch_to_head_dim(hidden_states)
    
        # 如果需要训练查询输出
        if self.train_q_out:
            # 线性投影
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # 应用dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # 否则使用标准的线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # 应用dropout
            hidden_states = attn.to_out[1](hidden_states)
    
        # 返回最终的隐藏状态
        return hidden_states
# 定义一个带有额外可学习的键和值矩阵的注意力处理器类
class AttnAddedKVProcessor:
    r"""
    处理器，用于执行与文本编码器相关的注意力计算
    """

    # 定义调用方法，以实现注意力计算
    def __call__(
        self,
        attn: Attention,  # 注意力对象
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态（可选）
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码（可选）
        *args,  # 其他位置参数
        **kwargs,  # 其他关键字参数
    ) -> torch.Tensor:  # 返回类型为张量
        # 检查是否传递了多余的参数或已弃用的 scale 参数
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 发出弃用警告
            deprecate("scale", "1.0.0", deprecation_message)

        # 将隐藏状态赋值给残差
        residual = hidden_states

        # 重塑隐藏状态的形状，并转置维度
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        # 获取批大小和序列长度
        batch_size, sequence_length, _ = hidden_states.shape

        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 如果没有编码器隐藏状态，则使用输入的隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要进行归一化处理
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 对隐藏状态进行分组归一化处理
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 将隐藏状态转换为查询
        query = attn.to_q(hidden_states)
        # 将查询从头维度转换为批维度
        query = attn.head_to_batch_dim(query)

        # 将编码器隐藏状态投影为键和值
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        # 将投影结果转换为批维度
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        # 如果不是仅进行交叉注意力
        if not attn.only_cross_attention:
            # 将隐藏状态转换为键和值
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            # 转换为批维度
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            # 将编码器键和值与当前键和值拼接
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            # 仅使用编码器的键和值
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        # 获取注意力概率
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # 计算隐藏状态的新值
        hidden_states = torch.bmm(attention_probs, value)
        # 将隐藏状态转换回头维度
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 重塑隐藏状态，并将残差加回
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        # 返回处理后的隐藏状态
        return hidden_states


# 定义另一个注意力处理器类
class AttnAddedKVProcessor2_0:
    r"""
    # 处理缩放点积注意力的处理器（如果使用 PyTorch 2.0，默认启用），
    # 其中为文本编码器添加了额外的可学习的键和值矩阵。
        """
    
        # 初始化方法
        def __init__(self):
            # 检查 F 中是否有 "scaled_dot_product_attention" 属性
            if not hasattr(F, "scaled_dot_product_attention"):
                # 如果没有，抛出 ImportError，提示用户需要升级到 PyTorch 2.0
                raise ImportError(
                    "AttnAddedKVProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
                )
    
        # 定义调用方法
        def __call__(
            self,
            attn: Attention,  # 输入的注意力机制对象
            hidden_states: torch.Tensor,  # 隐藏状态张量
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态张量
            attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量
            *args,  # 额外的位置参数
            **kwargs,  # 额外的关键字参数
    ) -> torch.Tensor:  # 指定函数返回类型为 torch.Tensor
        # 检查参数是否存在或 scale 参数是否被提供
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            # 设置弃用消息，告知 scale 参数将被忽略
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # 调用 deprecate 函数发出弃用警告
            deprecate("scale", "1.0.0", deprecation_message)

        # 将输入的 hidden_states 赋值给 residual
        residual = hidden_states

        # 调整 hidden_states 的形状并进行转置
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        # 获取 batch_size 和 sequence_length
        batch_size, sequence_length, _ = hidden_states.shape

        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)

        # 如果没有提供 encoder_hidden_states，则使用 hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要归一化交叉隐藏状态
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 对 hidden_states 进行分组归一化
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 计算查询向量
        query = attn.to_q(hidden_states)
        # 将查询向量转换为批次维度
        query = attn.head_to_batch_dim(query, out_dim=4)

        # 生成 encoder_hidden_states 的键和值的投影
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        # 将键和值转换为批次维度
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)

        # 如果不是只进行交叉注意力
        if not attn.only_cross_attention:
            # 计算当前 hidden_states 的键和值
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            # 转换为批次维度
            key = attn.head_to_batch_dim(key, out_dim=4)
            value = attn.head_to_batch_dim(value, out_dim=4)
            # 将键和值与 encoder 的键和值连接
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            # 如果只进行交叉注意力，使用 encoder 的键和值
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        # 计算缩放点积注意力的输出，形状为 (batch, num_heads, seq_len, head_dim)
        # TODO: 在迁移到 Torch 2.1 时添加对 attn.scale 的支持
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        # 转置并重塑 hidden_states
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, residual.shape[1])

        # 进行线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # 进行 dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 转置并重塑回 residual 的形状
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        # 将 residual 加到 hidden_states 上
        hidden_states = hidden_states + residual

        # 返回最终的 hidden_states
        return hidden_states
# 定义一个名为 JointAttnProcessor2_0 的类，用于处理自注意力投影
class JointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    # 初始化方法
    def __init__(self):
        # 检查 F 是否有 scaled_dot_product_attention 属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，抛出导入错误，提示需要升级 PyTorch 到 2.0
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 定义调用方法，接受多个参数
    def __call__(
        self,
        attn: Attention,  # 自注意力对象
        hidden_states: torch.FloatTensor,  # 当前隐藏状态的张量
        encoder_hidden_states: torch.FloatTensor = None,  # 编码器的隐藏状态，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码，默认为 None
        *args,  # 额外的位置参数
        **kwargs,  # 额外的关键字参数
    # 返回一个浮点张量
    ) -> torch.FloatTensor:
        # 保存输入的隐藏状态，以便后续使用
        residual = hidden_states
    
        # 获取隐藏状态的维度
        input_ndim = hidden_states.ndim
        # 如果隐藏状态是四维的
        if input_ndim == 4:
            # 解包隐藏状态的形状为批大小、通道、高度和宽度
            batch_size, channel, height, width = hidden_states.shape
            # 将隐藏状态重塑为三维，并进行转置
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        # 获取编码器隐藏状态的维度
        context_input_ndim = encoder_hidden_states.ndim
        # 如果编码器隐藏状态是四维的
        if context_input_ndim == 4:
            # 解包编码器隐藏状态的形状为批大小、通道、高度和宽度
            batch_size, channel, height, width = encoder_hidden_states.shape
            # 将编码器隐藏状态重塑为三维，并进行转置
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    
        # 获取编码器隐藏状态的批大小
        batch_size = encoder_hidden_states.shape[0]
    
        # 计算 `sample` 投影
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
    
        # 计算 `context` 投影
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
    
        # 合并注意力查询、键和值
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
    
        # 获取键的最后一维大小
        inner_dim = key.shape[-1]
        # 计算每个头的维度
        head_dim = inner_dim // attn.heads
        # 重塑查询、键和值以适应多个头
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    
        # 计算缩放点积注意力
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # 转置并重塑隐藏状态
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # 转换为查询的类型
        hidden_states = hidden_states.to(query.dtype)
    
        # 拆分注意力输出
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],  # 获取原隐藏状态的部分
            hidden_states[:, residual.shape[1] :],  # 获取编码器隐藏状态的部分
        )
    
        # 进行线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # 进行 dropout
        hidden_states = attn.to_out[1](hidden_states)
        # 如果上下文不是仅限于编码器
        if not attn.context_pre_only:
            # 对编码器隐藏状态进行额外处理
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
    
        # 如果输入是四维的，进行转置和重塑
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        # 如果上下文输入是四维的，进行转置和重塑
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    
        # 返回处理后的隐藏状态和编码器隐藏状态
        return hidden_states, encoder_hidden_states
# 定义一个类，PAGJointAttnProcessor2_0，用于处理自注意力投影
class PAGJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    # 初始化方法
    def __init__(self):
        # 检查是否存在名为"scaled_dot_product_attention"的属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果不存在，则抛出导入错误，提示需要升级PyTorch到2.0
            raise ImportError(
                "PAGJointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    # 可调用方法，接受注意力对象和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        # 其他可选参数
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
# 定义另一个类，PAGCFGJointAttnProcessor2_0，类似于PAGJointAttnProcessor2_0
class PAGCFGJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    # 初始化方法
    def __init__(self):
        # 检查是否存在名为"scaled_dot_product_attention"的属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果不存在，则抛出导入错误，提示需要升级PyTorch到2.0
            raise ImportError(
                "PAGCFGJointAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    # 可调用方法，接受注意力对象和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        # 其他可选参数
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
# 定义第三个类，FusedJointAttnProcessor2_0，处理自注意力投影
class FusedJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    # 初始化方法
    def __init__(self):
        # 检查是否存在名为"scaled_dot_product_attention"的属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果不存在，则抛出导入错误，提示需要升级PyTorch到2.0
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 可调用方法，接受注意力对象和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        # 其他可选参数
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # 将隐藏状态赋值给残差变量
        residual = hidden_states

        # 获取隐藏状态的维度
        input_ndim = hidden_states.ndim
        # 如果隐藏状态是四维的，进行维度变换
        if input_ndim == 4:
            # 解包隐藏状态的形状
            batch_size, channel, height, width = hidden_states.shape
            # 将隐藏状态变形为(batch_size, channel, height * width)并转置
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        # 获取编码器隐藏状态的维度
        context_input_ndim = encoder_hidden_states.ndim
        # 如果编码器隐藏状态是四维的，进行维度变换
        if context_input_ndim == 4:
            # 解包编码器隐藏状态的形状
            batch_size, channel, height, width = encoder_hidden_states.shape
            # 将编码器隐藏状态变形为(batch_size, channel, height * width)并转置
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 获取编码器隐藏状态的批量大小
        batch_size = encoder_hidden_states.shape[0]

        # `sample` 进行投影
        qkv = attn.to_qkv(hidden_states)
        # 计算每个分量的大小
        split_size = qkv.shape[-1] // 3
        # 将qkv拆分为query、key和value
        query, key, value = torch.split(qkv, split_size, dim=-1)

        # `context` 进行投影
        encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
        # 计算编码器qkv的分量大小
        split_size = encoder_qkv.shape[-1] // 3
        # 将编码器qkv拆分为查询、键和值的投影
        (
            encoder_hidden_states_query_proj,
            encoder_hidden_states_key_proj,
            encoder_hidden_states_value_proj,
        ) = torch.split(encoder_qkv, split_size, dim=-1)

        # 进行注意力计算
        # 将query、key、value进行连接
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        # 获取key的最后一维大小
        inner_dim = key.shape[-1]
        # 计算每个头的维度
        head_dim = inner_dim // attn.heads
        # 调整query的形状以适应多头注意力
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # 调整key的形状以适应多头注意力
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # 调整value的形状以适应多头注意力
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 进行缩放点积注意力计算
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        # 调整hidden_states的形状
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # 将hidden_states转换为与query相同的数据类型
        hidden_states = hidden_states.to(query.dtype)

        # 拆分注意力输出
        hidden_states, encoder_hidden_states = (
            # 保留残差形状的部分
            hidden_states[:, : residual.shape[1]],
            # 剩余的部分
            hidden_states[:, residual.shape[1] :],
        )

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # 进行dropout
        hidden_states = attn.to_out[1](hidden_states)
        # 如果不是只使用上下文，进行编码器输出的投影
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # 如果输入是四维的，调整hidden_states的形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        # 如果上下文输入是四维的，调整encoder_hidden_states的形状
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 返回hidden_states和encoder_hidden_states
        return hidden_states, encoder_hidden_states
# 定义一个用于处理 Aura Flow 的注意力处理器类
class AuraFlowAttnProcessor2_0:
    """Attention processor used typically in processing Aura Flow."""

    # 初始化方法
    def __init__(self):
        # 检查 F 是否具有 scaled_dot_product_attention 属性，并确保 PyTorch 版本符合要求
        if not hasattr(F, "scaled_dot_product_attention") and is_torch_version("<", "2.1"):
            # 如果不满足条件，抛出导入错误，提示用户升级 PyTorch
            raise ImportError(
                "AuraFlowAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to at least 2.1 or above as we use `scale` in `F.scaled_dot_product_attention()`. "
            )

    # 可调用方法，用于处理输入的注意力和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        *args,
        **kwargs,
# 定义一个用于处理 Aura Flow 的融合投影注意力处理器类
class FusedAuraFlowAttnProcessor2_0:
    """Attention processor used typically in processing Aura Flow with fused projections."""

    # 初始化方法
    def __init__(self):
        # 检查 F 是否具有 scaled_dot_product_attention 属性，并确保 PyTorch 版本符合要求
        if not hasattr(F, "scaled_dot_product_attention") and is_torch_version("<", "2.1"):
            # 如果不满足条件，抛出导入错误，提示用户升级 PyTorch
            raise ImportError(
                "FusedAuraFlowAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to at least 2.1 or above as we use `scale` in `F.scaled_dot_product_attention()`. "
            )

    # 可调用方法，用于处理输入的注意力和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        *args,
        **kwargs,
# YiYi 待办事项：重构与 rope 相关的函数/类
def apply_rope(xq, xk, freqs_cis):
    # 将 xq 转换为浮点型，并重新调整形状以便处理
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    # 将 xk 转换为浮点型，并重新调整形状以便处理
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    # 计算 xq 的输出，结合频率复数
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    # 计算 xk 的输出，结合频率复数
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    # 返回调整形状后的 xq_out 和 xk_out，并确保与原始类型匹配
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# 定义一个实现缩放点积注意力的处理器类
class FluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 是否具有 scaled_dot_product_attention 属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果不满足条件，抛出导入错误，提示用户升级 PyTorch
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 可调用方法，用于处理输入的注意力和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    # 定义函数的返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 获取 hidden_states 的维度数量
        input_ndim = hidden_states.ndim
    
        # 如果输入的维度为 4
        if input_ndim == 4:
            # 解包 hidden_states 的形状为 batch_size, channel, height, width
            batch_size, channel, height, width = hidden_states.shape
            # 将 hidden_states 视图调整为 (batch_size, channel, height * width) 并转置
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    
        # 如果 encoder_hidden_states 为 None，则获取 hidden_states 的形状
        # 否则获取 encoder_hidden_states 的形状
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    
        # 将 hidden_states 转换为查询向量
        query = attn.to_q(hidden_states)
        # 如果 encoder_hidden_states 为 None，将其设置为 hidden_states
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
    
        # 将 encoder_hidden_states 转换为键向量
        key = attn.to_k(encoder_hidden_states)
        # 将 encoder_hidden_states 转换为值向量
        value = attn.to_v(encoder_hidden_states)
    
        # 获取键的最后一个维度的大小
        inner_dim = key.shape[-1]
        # 计算每个头的维度
        head_dim = inner_dim // attn.heads
    
        # 将查询向量调整视图为 (batch_size, -1, attn.heads, head_dim) 并转置
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    
        # 将键向量调整视图为 (batch_size, -1, attn.heads, head_dim) 并转置
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # 将值向量调整视图为 (batch_size, -1, attn.heads, head_dim) 并转置
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    
        # 如果存在规范化查询的层，则对查询进行规范化
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        # 如果存在规范化键的层，则对键进行规范化
        if attn.norm_k is not None:
            key = attn.norm_k(key)
    
        # 如果需要应用 RoPE
        if image_rotary_emb is not None:
            # 应用旋转嵌入到查询和键上
            query, key = apply_rope(query, key, image_rotary_emb)
    
        # 计算缩放点积注意力，输出形状为 (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    
        # 转置并调整 hidden_states 的形状为 (batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # 将 hidden_states 转换为与查询相同的数据类型
        hidden_states = hidden_states.to(query.dtype)
    
        # 如果输入维度为 4，将 hidden_states 转置并调整形状回原始维度
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    
        # 返回处理后的 hidden_states
        return hidden_states
# 定义一个名为 FluxAttnProcessor2_0 的类，通常用于处理 SD3 类自注意力投影
class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    # 初始化方法
    def __init__(self):
        # 检查 F 是否有 scaled_dot_product_attention 属性，如果没有则抛出 ImportError
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 定义调用方法，使类实例可被调用
    def __call__(
        self,
        attn: Attention,  # 接收 Attention 对象
        hidden_states: torch.FloatTensor,  # 接收隐藏状态张量
        encoder_hidden_states: torch.FloatTensor = None,  # 可选的编码器隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码张量
        image_rotary_emb: Optional[torch.Tensor] = None,  # 可选的图像旋转嵌入张量
    ):
        # 此处将实现自注意力的具体处理逻辑

# 定义一个名为 CogVideoXAttnProcessor2_0 的类，专用于 CogVideoX 模型的缩放点积注意力处理
class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 是否有 scaled_dot_product_attention 属性，如果没有则抛出 ImportError
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 定义调用方法，使类实例可被调用
    def __call__(
        self,
        attn: Attention,  # 接收 Attention 对象
        hidden_states: torch.Tensor,  # 接收隐藏状态张量
        encoder_hidden_states: torch.Tensor,  # 接收编码器隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量
        image_rotary_emb: Optional[torch.Tensor] = None,  # 可选的图像旋转嵌入张量
    ):
        # 此处将实现自注意力的具体处理逻辑
    ) -> torch.Tensor:  # 函数返回一个张量，表示隐藏状态
        text_seq_length = encoder_hidden_states.size(1)  # 获取编码器隐藏状态的序列长度

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # 在维度1上连接编码器隐藏状态和当前隐藏状态

        batch_size, sequence_length, _ = (  # 解包 batch_size 和 sequence_length
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape  # 根据编码器隐藏状态的存在性决定形状
        )

        if attention_mask is not None:  # 如果存在注意力掩码
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)  # 准备注意力掩码
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])  # 调整注意力掩码的形状以适应头数

        query = attn.to_q(hidden_states)  # 将隐藏状态转换为查询向量
        key = attn.to_k(hidden_states)  # 将隐藏状态转换为键向量
        value = attn.to_v(hidden_states)  # 将隐藏状态转换为值向量

        inner_dim = key.shape[-1]  # 获取键向量的最后一个维度大小
        head_dim = inner_dim // attn.heads  # 计算每个头的维度

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # 调整查询向量形状并转置以适应多头注意力
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # 调整键向量形状并转置以适应多头注意力
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # 调整值向量形状并转置以适应多头注意力

        if attn.norm_q is not None:  # 如果查询归一化层存在
            query = attn.norm_q(query)  # 对查询向量进行归一化
        if attn.norm_k is not None:  # 如果键归一化层存在
            key = attn.norm_k(key)  # 对键向量进行归一化

        # Apply RoPE if needed  # 如果需要应用旋转位置编码
        if image_rotary_emb is not None:  # 如果图像旋转嵌入存在
            from .embeddings import apply_rotary_emb  # 导入应用旋转嵌入的函数

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)  # 应用旋转嵌入到查询向量的后半部分
            if not attn.is_cross_attention:  # 如果不是交叉注意力
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)  # 应用旋转嵌入到键向量的后半部分

        hidden_states = F.scaled_dot_product_attention(  # 计算缩放点积注意力
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False  # 输入查询、键和值，以及注意力掩码
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)  # 转置和重塑隐藏状态以合并头维度

        # linear proj  # 线性投影
        hidden_states = attn.to_out[0](hidden_states)  # 对隐藏状态应用输出线性变换
        # dropout  # 进行dropout操作
        hidden_states = attn.to_out[1](hidden_states)  # 对隐藏状态应用dropout

        encoder_hidden_states, hidden_states = hidden_states.split(  # 将隐藏状态分割为编码器和当前隐藏状态
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1  # 根据文本序列长度和剩余部分进行分割
        )
        return hidden_states, encoder_hidden_states  # 返回当前隐藏状态和编码器隐藏状态
# 定义一个用于实现 CogVideoX 模型的缩放点积注意力的处理器类
class FusedCogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 是否具有 scaled_dot_product_attention 属性，如果没有则抛出导入错误
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 定义可调用方法，处理注意力计算
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 获取编码器隐藏状态的序列长度
        text_seq_length = encoder_hidden_states.size(1)

        # 将编码器和当前隐藏状态按维度 1 连接
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 获取批次大小和序列长度，依据编码器隐藏状态是否为 None
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # 如果提供了注意力掩码，则准备掩码
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # 将掩码调整为适当的形状
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # 将隐藏状态转换为查询、键、值
        qkv = attn.to_qkv(hidden_states)
        # 计算每个部分的大小
        split_size = qkv.shape[-1] // 3
        # 分割成查询、键和值
        query, key, value = torch.split(qkv, split_size, dim=-1)

        # 获取键的内部维度
        inner_dim = key.shape[-1]
        # 计算每个头的维度
        head_dim = inner_dim // attn.heads

        # 调整查询、键和值的形状以适应多头注意力
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # 如果存在查询的归一化，则应用归一化
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        # 如果存在键的归一化，则应用归一化
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 如果需要应用 RoPE
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            # 对查询的特定部分应用旋转嵌入
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            # 如果不是交叉注意力，则对键的特定部分应用旋转嵌入
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # 计算缩放点积注意力
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # 调整隐藏状态的形状以便输出
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # 线性投影
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 将隐藏状态拆分为编码器隐藏状态和当前隐藏状态
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        # 返回当前隐藏状态和编码器隐藏状态
        return hidden_states, encoder_hidden_states


# 定义用于实现内存高效注意力的处理器类
class XFormersAttnAddedKVProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.
    # 文档字符串，说明可选参数 attention_op 的作用
        Args:
            attention_op (`Callable`, *optional*, defaults to `None`):
                使用的基本注意力操作符，推荐设置为 `None` 让 xFormers 选择最佳操作符
        """
    
        # 构造函数，初始化注意力操作符
        def __init__(self, attention_op: Optional[Callable] = None):
            # 将传入的注意力操作符赋值给实例变量
            self.attention_op = attention_op
    
        # 可调用方法，用于执行注意力计算
        def __call__(
            self,
            attn: Attention,  # 注意力对象
            hidden_states: torch.Tensor,  # 隐藏状态张量
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，默认为 None
            attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
        ) -> torch.Tensor:
            # 将当前隐藏状态保存为残差以便后续使用
            residual = hidden_states
            # 调整隐藏状态的形状并转置
            hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
            # 获取批次大小和序列长度
            batch_size, sequence_length, _ = hidden_states.shape
    
            # 准备注意力掩码
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    
            # 如果没有编码器隐藏状态，则将其设置为当前的隐藏状态
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            # 如果需要，则对编码器隐藏状态进行归一化处理
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
            # 对隐藏状态进行分组归一化处理
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    
            # 生成查询向量
            query = attn.to_q(hidden_states)
            # 将查询向量从头部维度转换为批次维度
            query = attn.head_to_batch_dim(query)
    
            # 对编码器隐藏状态进行键和值的投影
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            # 将编码器隐藏状态的键和值转换为批次维度
            encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)
    
            # 如果不是仅使用交叉注意力
            if not attn.only_cross_attention:
                # 生成当前隐藏状态的键和值
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
                # 转换键和值到批次维度
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)
                # 将编码器的键和值与当前的键和值连接起来
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                # 如果仅使用交叉注意力，则直接使用编码器的键和值
                key = encoder_hidden_states_key_proj
                value = encoder_hidden_states_value_proj
    
            # 计算高效的注意力
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            # 将结果转换为查询的 dtype
            hidden_states = hidden_states.to(query.dtype)
            # 将隐藏状态从批次维度转换回头部维度
            hidden_states = attn.batch_to_head_dim(hidden_states)
    
            # 线性变换
            hidden_states = attn.to_out[0](hidden_states)
            # 应用 dropout
            hidden_states = attn.to_out[1](hidden_states)
    
            # 调整隐藏状态的形状以匹配残差
            hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
            # 将当前隐藏状态与残差相加
            hidden_states = hidden_states + residual
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个用于实现基于 xFormers 的内存高效注意力的处理器类
class XFormersAttnProcessor:
    r"""
    处理器，用于实现基于 xFormers 的内存高效注意力。

    参数：
        attention_op (`Callable`, *可选*, 默认为 `None`):
            基础
            [操作符](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase)，
            用作注意力操作符。建议将其设置为 `None`，并让 xFormers 选择最佳操作符。
    """

    # 初始化方法，接受一个可选的注意力操作符
    def __init__(self, attention_op: Optional[Callable] = None):
        # 将传入的注意力操作符赋值给实例变量
        self.attention_op = attention_op

    # 定义可调用方法，用于执行注意力计算
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
# 定义一个用于实现 flash attention 的处理器类，使用 torch_npu
class AttnProcessorNPU:
    r"""
    处理器，用于使用 torch_npu 实现 flash attention。torch_npu 仅支持 fp16 和 bf16 数据类型。如果
    使用 fp32，将使用 F.scaled_dot_product_attention 进行计算，但在 NPU 上加速效果不明显。

    """

    # 初始化方法
    def __init__(self):
        # 检查是否可用 torch_npu，如果不可用则抛出异常
        if not is_torch_npu_available():
            raise ImportError("AttnProcessorNPU requires torch_npu extensions and is supported only on npu devices.")

    # 定义可调用方法，用于执行注意力计算
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
# 定义一个用于实现 scaled dot-product attention 的处理器类，默认在 PyTorch 2.0 中启用
class AttnProcessor2_0:
    r"""
    处理器，用于实现 scaled dot-product attention（如果您使用的是 PyTorch 2.0，默认启用）。
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 中是否有 scaled_dot_product_attention 属性，如果没有则抛出异常
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 定义可调用方法，用于执行注意力计算
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
# 定义一个用于实现 scaled dot-product attention 的处理器类，适用于稳定音频模型
class StableAudioAttnProcessor2_0:
    r"""
    处理器，用于实现 scaled dot-product attention（如果您使用的是 PyTorch 2.0，默认启用）。此处理器用于
    稳定音频模型。它在查询和键向量上应用旋转嵌入，并允许 MHA、GQA 或 MQA。
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 中是否有 scaled_dot_product_attention 属性，如果没有则抛出异常
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    # 定义方法，用于应用部分旋转嵌入
    def apply_partial_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor],
    # 定义返回类型为 torch.Tensor 的函数
    ) -> torch.Tensor:
        # 从当前模块导入 apply_rotary_emb 函数
        from .embeddings import apply_rotary_emb
    
        # 获取频率余弦的最后一个维度大小，用于旋转
        rot_dim = freqs_cis[0].shape[-1]
        # 将输入张量 x 划分为需要旋转和不需要旋转的部分
        x_to_rotate, x_unrotated = x[..., :rot_dim], x[..., rot_dim:]
    
        # 应用旋转嵌入到需要旋转的部分
        x_rotated = apply_rotary_emb(x_to_rotate, freqs_cis, use_real=True, use_real_unbind_dim=-2)
    
        # 将旋转后的部分与未旋转的部分在最后一个维度上连接
        out = torch.cat((x_rotated, x_unrotated), dim=-1)
        # 返回连接后的输出张量
        return out
    
    # 定义可调用方法，接收注意力和隐藏状态
    def __call__(
        self,
        # 输入的注意力对象
        attn: Attention,
        # 隐藏状态的张量
        hidden_states: torch.Tensor,
        # 可选的编码器隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 可选的注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的旋转嵌入张量
        rotary_emb: Optional[torch.Tensor] = None,
# 定义 HunyuanAttnProcessor2_0 类，处理缩放的点积注意力
class HunyuanAttnProcessor2_0:
    r"""
    处理器用于实现缩放的点积注意力（如果使用 PyTorch 2.0，默认启用）。这是
    HunyuanDiT 模型中使用的。它在查询和键向量上应用归一化层和旋转嵌入。
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 中是否有 scaled_dot_product_attention 属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，则抛出导入错误，提示需要升级 PyTorch 到 2.0
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 定义调用方法
    def __call__(
        self,
        attn: Attention,  # 注意力机制实例
        hidden_states: torch.Tensor,  # 当前隐藏状态的张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态的可选张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码的可选张量
        temb: Optional[torch.Tensor] = None,  # 时间嵌入的可选张量
        image_rotary_emb: Optional[torch.Tensor] = None,  # 图像旋转嵌入的可选张量
class FusedHunyuanAttnProcessor2_0:
    r"""
    处理器用于实现缩放的点积注意力（如果使用 PyTorch 2.0，默认启用），带有融合的
    投影层。这是 HunyuanDiT 模型中使用的。它在查询和键向量上应用归一化层和旋转嵌入。
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 中是否有 scaled_dot_product_attention 属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，则抛出导入错误，提示需要升级 PyTorch 到 2.0
            raise ImportError(
                "FusedHunyuanAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    # 定义调用方法
    def __call__(
        self,
        attn: Attention,  # 注意力机制实例
        hidden_states: torch.Tensor,  # 当前隐藏状态的张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态的可选张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码的可选张量
        temb: Optional[torch.Tensor] = None,  # 时间嵌入的可选张量
        image_rotary_emb: Optional[torch.Tensor] = None,  # 图像旋转嵌入的可选张量
class PAGHunyuanAttnProcessor2_0:
    r"""
    处理器用于实现缩放的点积注意力（如果使用 PyTorch 2.0，默认启用）。这是
    HunyuanDiT 模型中使用的。它在查询和键向量上应用归一化层和旋转嵌入。该处理器
    变体采用了 [Pertubed Attention Guidance](https://arxiv.org/abs/2403.17377)。
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 中是否有 scaled_dot_product_attention 属性
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，则抛出导入错误，提示需要升级 PyTorch 到 2.0
            raise ImportError(
                "PAGHunyuanAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    # 定义调用方法
    def __call__(
        self,
        attn: Attention,  # 注意力机制实例
        hidden_states: torch.Tensor,  # 当前隐藏状态的张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态的可选张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码的可选张量
        temb: Optional[torch.Tensor] = None,  # 时间嵌入的可选张量
        image_rotary_emb: Optional[torch.Tensor] = None,  # 图像旋转嵌入的可选张量
class PAGCFGHunyuanAttnProcessor2_0:
    r"""
    处理器用于实现缩放的点积注意力（如果使用 PyTorch 2.0，默认启用）。这是
    HunyuanDiT 模型中使用的。它在查询和键向量上应用归一化层和旋转嵌入。该处理器
    变体采用了 [Pertubed Attention Guidance](https://arxiv.org/abs/2403.17377)。
    """
    # 初始化方法，用于创建类的实例
        def __init__(self):
            # 检查模块 F 是否具有属性 "scaled_dot_product_attention"
            if not hasattr(F, "scaled_dot_product_attention"):
                # 如果没有该属性，则抛出 ImportError，提示用户升级 PyTorch
                raise ImportError(
                    "PAGCFGHunyuanAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
                )
    
    # 可调用方法，允许类的实例像函数一样被调用
        def __call__(
            self,
            attn: Attention,  # 注意力机制对象
            hidden_states: torch.Tensor,  # 当前隐藏状态的张量
            encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，可选参数
            attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选参数
            temb: Optional[torch.Tensor] = None,  # 时间嵌入，可选参数
            image_rotary_emb: Optional[torch.Tensor] = None,  # 图像旋转嵌入，可选参数
# 定义一个用于实现缩放点积注意力的处理器类
class LuminaAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LuminaNextDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    # 初始化方法
    def __init__(self):
        # 检查 PyTorch 是否具有缩放点积注意力功能
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，抛出导入错误，提示用户升级 PyTorch 到 2.0
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    # 定义调用方法，使类实例可调用
    def __call__(
        self,
        # 接收注意力对象
        attn: Attention,
        # 接收隐藏状态张量
        hidden_states: torch.Tensor,
        # 接收编码器隐藏状态张量
        encoder_hidden_states: torch.Tensor,
        # 可选的注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的查询旋转嵌入张量
        query_rotary_emb: Optional[torch.Tensor] = None,
        # 可选的键旋转嵌入张量
        key_rotary_emb: Optional[torch.Tensor] = None,
        # 可选的基本序列长度
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:  # 函数返回一个张量，表示处理后的隐藏状态
        from .embeddings import apply_rotary_emb  # 从当前包导入应用旋转嵌入的函数

        input_ndim = hidden_states.ndim  # 获取隐藏状态的维度数

        if input_ndim == 4:  # 如果隐藏状态是四维张量
            batch_size, channel, height, width = hidden_states.shape  # 解包出批次大小、通道、高度和宽度
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)  # 重塑并转置隐藏状态

        batch_size, sequence_length, _ = hidden_states.shape  # 解包出批次大小和序列长度

        # Get Query-Key-Value Pair  # 获取查询、键、值对
        query = attn.to_q(hidden_states)  # 将隐藏状态转换为查询张量
        key = attn.to_k(encoder_hidden_states)  # 将编码器的隐藏状态转换为键张量
        value = attn.to_v(encoder_hidden_states)  # 将编码器的隐藏状态转换为值张量

        query_dim = query.shape[-1]  # 获取查询的最后一个维度（特征维度）
        inner_dim = key.shape[-1]  # 获取键的最后一个维度
        head_dim = query_dim // attn.heads  # 计算每个头的维度
        dtype = query.dtype  # 获取查询张量的数据类型

        # Get key-value heads  # 获取键值头的数量
        kv_heads = inner_dim // head_dim  # 计算每个头的键值数量

        # Apply Query-Key Norm if needed  # 如果需要，应用查询-键归一化
        if attn.norm_q is not None:  # 如果定义了查询的归一化
            query = attn.norm_q(query)  # 对查询进行归一化
        if attn.norm_k is not None:  # 如果定义了键的归一化
            key = attn.norm_k(key)  # 对键进行归一化

        query = query.view(batch_size, -1, attn.heads, head_dim)  # 重塑查询张量以适应头的维度

        key = key.view(batch_size, -1, kv_heads, head_dim)  # 重塑键张量以适应头的维度
        value = value.view(batch_size, -1, kv_heads, head_dim)  # 重塑值张量以适应头的维度

        # Apply RoPE if needed  # 如果需要，应用旋转位置嵌入
        if query_rotary_emb is not None:  # 如果定义了查询的旋转嵌入
            query = apply_rotary_emb(query, query_rotary_emb, use_real=False)  # 应用旋转嵌入到查询
        if key_rotary_emb is not None:  # 如果定义了键的旋转嵌入
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)  # 应用旋转嵌入到键

        query, key = query.to(dtype), key.to(dtype)  # 将查询和键转换为相同的数据类型

        # Apply proportional attention if true  # 如果为真，应用比例注意力
        if key_rotary_emb is None:  # 如果没有键的旋转嵌入
            softmax_scale = None  # 设置缩放因子为 None
        else:  # 如果有键的旋转嵌入
            if base_sequence_length is not None:  # 如果定义了基础序列长度
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale  # 计算缩放因子
            else:  # 如果没有定义基础序列长度
                softmax_scale = attn.scale  # 使用注意力的缩放因子

        # perform Grouped-query Attention (GQA)  # 执行分组查询注意力
        n_rep = attn.heads // kv_heads  # 计算每个键值头的重复数量
        if n_rep >= 1:  # 如果重复数量大于等于 1
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)  # 扩展并重复键
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)  # 扩展并重复值

        # scaled_dot_product_attention expects attention_mask shape to be  # 缩放点积注意力期望的注意力掩码形状
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)  # 将注意力掩码转换为布尔值并调整形状
        attention_mask = attention_mask.expand(-1, attn.heads, sequence_length, -1)  # 扩展注意力掩码以匹配头的数量

        query = query.transpose(1, 2)  # 转置查询张量
        key = key.transpose(1, 2)  # 转置键张量
        value = value.transpose(1, 2)  # 转置值张量

        # the output of sdp = (batch, num_heads, seq_len, head_dim)  # 缩放点积注意力的输出形状
        # TODO: add support for attn.scale when we move to Torch 2.1  # TODO: 在迁移到 Torch 2.1 时支持 attn.scale
        hidden_states = F.scaled_dot_product_attention(  # 计算缩放点积注意力
            query, key, value, attn_mask=attention_mask, scale=softmax_scale  # 输入查询、键、值及注意力掩码和缩放因子
        )
        hidden_states = hidden_states.transpose(1, 2).to(dtype)  # 转置输出并转换为相应的数据类型

        return hidden_states  # 返回处理后的隐藏状态
# 定义一个用于实现缩放点积注意力的处理器类，默认启用（如果使用 PyTorch 2.0）
class FusedAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). It uses
    fused projection layers. For self-attention modules, all projection matrices (i.e., query, key, value) are fused.
    For cross-attention modules, key and value projection matrices are fused.

    <Tip warning={true}>

    This API is currently 🧪 experimental in nature and can change in future.

    </Tip>
    """

    # 初始化方法
    def __init__(self):
        # 检查 F 库是否具有缩放点积注意力功能
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，抛出导入错误，提示用户升级 PyTorch 版本
            raise ImportError(
                "FusedAttnProcessor2_0 requires at least PyTorch 2.0, to use it. Please upgrade PyTorch to > 2.0."
            )

    # 调用方法，处理注意力计算
    def __call__(
        self,
        attn: Attention,  # 注意力模块
        hidden_states: torch.Tensor,  # 隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选
        temb: Optional[torch.Tensor] = None,  # 时间嵌入，可选
        *args,  # 可变位置参数
        **kwargs,  # 可变关键字参数
    ):
        pass  # 此处省略具体实现

# 定义一个用于实现内存高效注意力的处理器类，使用 xFormers 方法
class CustomDiffusionXFormersAttnProcessor(nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers for the Custom Diffusion method.

    Args:
    train_kv (`bool`, defaults to `True`):
        Whether to newly train the key and value matrices corresponding to the text features.
    train_q_out (`bool`, defaults to `True`):
        Whether to newly train query matrices corresponding to the latent image features.
    hidden_size (`int`, *optional*, defaults to `None`):
        The hidden size of the attention layer.
    cross_attention_dim (`int`, *optional*, defaults to `None`):
        The number of channels in the `encoder_hidden_states`.
    out_bias (`bool`, defaults to `True`):
        Whether to include the bias parameter in `train_q_out`.
    dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability to use.
    attention_op (`Callable`, *optional*, defaults to `None`):
        The base
        [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to use
        as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best operator.
    """

    # 初始化方法，设置各种参数
    def __init__(
        self,
        train_kv: bool = True,  # 是否训练与文本特征对应的键值矩阵
        train_q_out: bool = False,  # 是否训练与潜在图像特征对应的查询矩阵
        hidden_size: Optional[int] = None,  # 注意力层的隐藏大小
        cross_attention_dim: Optional[int] = None,  # 编码器隐藏状态的通道数
        out_bias: bool = True,  # 是否在 train_q_out 中包含偏置参数
        dropout: float = 0.0,  # 使用的丢弃概率
        attention_op: Optional[Callable] = None,  # 要使用的基础注意力操作
    ):
        pass  # 此处省略具体实现
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 存储训练键值对的标志
        self.train_kv = train_kv
        # 存储训练查询输出的标志
        self.train_q_out = train_q_out

        # 存储隐藏层大小
        self.hidden_size = hidden_size
        # 存储交叉注意力维度
        self.cross_attention_dim = cross_attention_dim
        # 存储注意力操作类型
        self.attention_op = attention_op

        # `_custom_diffusion` id 用于简化序列化和加载
        if self.train_kv:
            # 创建线性层，将交叉注意力维度或隐藏层大小映射到隐藏层大小，且不使用偏置
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            # 创建线性层，将交叉注意力维度或隐藏层大小映射到隐藏层大小，且不使用偏置
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            # 创建线性层，将隐藏层大小映射到隐藏层大小，且不使用偏置
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            # 创建一个空的模块列表以存储输出相关的层
            self.to_out_custom_diffusion = nn.ModuleList([])
            # 将线性层添加到模块列表中，用于输出映射
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            # 将 Dropout 层添加到模块列表中，用于正则化
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(
        # 定义调用方法，接收注意力对象和隐藏状态张量
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        # 可选参数：编码器的隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 可选参数：注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
    # 定义函数的返回类型为 torch.Tensor
        ) -> torch.Tensor:
            # 获取批量大小和序列长度，根据 encoder_hidden_states 是否为 None 决定来源
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
    
            # 准备注意力掩码
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    
            # 判断是否在训练阶段并应用不同的查询生成方式
            if self.train_q_out:
                query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
            else:
                query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))
    
            # 判断是否存在编码器隐藏状态，并设置 crossattn 标志
            if encoder_hidden_states is None:
                crossattn = False
                encoder_hidden_states = hidden_states
            else:
                crossattn = True
                # 如果需要对编码器隐藏状态进行归一化处理
                if attn.norm_cross:
                    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
            # 判断是否在训练阶段并应用不同的键值生成方式
            if self.train_kv:
                key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
                value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
                key = key.to(attn.to_q.weight.dtype)
                value = value.to(attn.to_q.weight.dtype)
            else:
                key = attn.to_k(encoder_hidden_states)
                value = attn.to_v(encoder_hidden_states)
    
            # 如果使用交叉注意力，进行键值的分离和处理
            if crossattn:
                detach = torch.ones_like(key)
                detach[:, :1, :] = detach[:, :1, :] * 0.0
                key = detach * key + (1 - detach) * key.detach()
                value = detach * value + (1 - detach) * value.detach()
    
            # 将查询、键、值转换为批处理维度并保持连续性
            query = attn.head_to_batch_dim(query).contiguous()
            key = attn.head_to_batch_dim(key).contiguous()
            value = attn.head_to_batch_dim(value).contiguous()
    
            # 使用内存高效的注意力计算隐藏状态
            hidden_states = xformers.ops.memory_efficient_attention(
                query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            # 将隐藏状态转换为查询的类型
            hidden_states = hidden_states.to(query.dtype)
            # 将隐藏状态转换回头部维度
            hidden_states = attn.batch_to_head_dim(hidden_states)
    
            # 根据训练标志决定输出的处理方式
            if self.train_q_out:
                # 线性变换
                hidden_states = self.to_out_custom_diffusion[0](hidden_states)
                # 进行 dropout 操作
                hidden_states = self.to_out_custom_diffusion[1](hidden_states)
            else:
                # 线性变换
                hidden_states = attn.to_out[0](hidden_states)
                # 进行 dropout 操作
                hidden_states = attn.to_out[1](hidden_states)
    
            # 返回处理后的隐藏状态
            return hidden_states
# 自定义扩散注意力处理器类，继承自 PyTorch 的 nn.Module
class CustomDiffusionAttnProcessor2_0(nn.Module):
    r"""
    用于实现自定义扩散方法的注意力处理器，使用 PyTorch 2.0 的内存高效缩放
    点积注意力。

    参数:
        train_kv (`bool`, 默认值为 `True`):
            是否新训练与文本特征对应的键和值矩阵。
        train_q_out (`bool`, 默认值为 `True`):
            是否新训练与潜在图像特征对应的查询矩阵。
        hidden_size (`int`, *可选*, 默认值为 `None`):
            注意力层的隐藏大小。
        cross_attention_dim (`int`, *可选*, 默认值为 `None`):
            `encoder_hidden_states` 中的通道数。
        out_bias (`bool`, 默认值为 `True`):
            是否在 `train_q_out` 中包含偏置参数。
        dropout (`float`, *可选*, 默认值为 0.0):
            使用的 dropout 概率。
    """

    # 初始化方法，设置类的属性
    def __init__(
        self,
        train_kv: bool = True,
        train_q_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置是否训练键值矩阵的标志
        self.train_kv = train_kv
        # 设置是否训练查询输出矩阵的标志
        self.train_q_out = train_q_out

        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置交叉注意力的维度
        self.cross_attention_dim = cross_attention_dim

        # 如果需要训练键值矩阵，则创建对应的线性层
        if self.train_kv:
            # 创建从交叉注意力维度到隐藏层的线性变换，且不使用偏置
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            # 创建从交叉注意力维度到隐藏层的线性变换，且不使用偏置
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        # 如果需要训练查询输出，则创建对应的线性层
        if self.train_q_out:
            # 创建从隐藏层到隐藏层的线性变换，且不使用偏置
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            # 创建一个空的模块列表，用于存储输出层
            self.to_out_custom_diffusion = nn.ModuleList([])
            # 将线性层添加到输出模块列表中
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            # 添加 dropout 层到输出模块列表中
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    # 定义类的调用方法，处理输入的注意力和隐藏状态
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # 定义返回类型为 torch.Tensor
        batch_size, sequence_length, _ = hidden_states.shape  # 解包 hidden_states 的形状，获取批大小和序列长度
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)  # 准备注意力掩码
        if self.train_q_out:  # 检查是否在训练查询输出
            query = self.to_q_custom_diffusion(hidden_states)  # 使用自定义扩散方法生成查询向量
        else:  # 否则
            query = attn.to_q(hidden_states)  # 使用标准方法生成查询向量

        if encoder_hidden_states is None:  # 检查编码器隐藏状态是否为空
            crossattn = False  # 设置交叉注意力标志为假
            encoder_hidden_states = hidden_states  # 将编码器隐藏状态设置为隐藏状态
        else:  # 如果编码器隐藏状态不为空
            crossattn = True  # 设置交叉注意力标志为真
            if attn.norm_cross:  # 如果需要归一化
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)  # 归一化编码器隐藏状态

        if self.train_kv:  # 检查是否在训练键值对
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))  # 生成键向量
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))  # 生成值向量
            key = key.to(attn.to_q.weight.dtype)  # 将键向量转换为查询权重的数据类型
            value = value.to(attn.to_q.weight.dtype)  # 将值向量转换为查询权重的数据类型
        else:  # 否则
            key = attn.to_k(encoder_hidden_states)  # 使用标准方法生成键向量
            value = attn.to_v(encoder_hidden_states)  # 使用标准方法生成值向量

        if crossattn:  # 如果进行交叉注意力
            detach = torch.ones_like(key)  # 创建与键相同形状的全1张量
            detach[:, :1, :] = detach[:, :1, :] * 0.0  # 将第一时间步的值设置为0
            key = detach * key + (1 - detach) * key.detach()  # 根据 detach 张量计算键的最终值
            value = detach * value + (1 - detach) * value.detach()  # 根据 detach 张量计算值的最终值

        inner_dim = hidden_states.shape[-1]  # 获取隐藏状态的最后一维大小

        head_dim = inner_dim // attn.heads  # 计算每个头的维度
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # 重新调整查询的形状并转置
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # 重新调整键的形状并转置
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # 重新调整值的形状并转置

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(  # 计算缩放点积注意力
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False  # 输入查询、键和值以及注意力掩码
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)  # 转置并重塑隐藏状态
        hidden_states = hidden_states.to(query.dtype)  # 将隐藏状态转换为查询的类型

        if self.train_q_out:  # 如果在训练查询输出
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)  # 线性变换
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)  # 应用 dropout
        else:  # 否则
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)  # 线性变换
            # dropout
            hidden_states = attn.to_out[1](hidden_states)  # 应用 dropout

        return hidden_states  # 返回最终的隐藏状态
# 定义一个用于实现切片注意力的处理器类
class SlicedAttnProcessor:
    r"""
    处理器用于实现切片注意力。

    参数:
        slice_size (`int`, *可选*):
            计算注意力的步骤数量。使用的切片数量为 `attention_head_dim // slice_size`，并且
            `attention_head_dim` 必须是 `slice_size` 的整数倍。
    """

    # 初始化方法，接受切片大小作为参数
    def __init__(self, slice_size: int):
        # 将传入的切片大小保存为实例变量
        self.slice_size = slice_size

    # 定义可调用方法，以便实例可以像函数一样被调用
    def __call__(
        self,
        attn: Attention,  # 输入的注意力对象
        hidden_states: torch.Tensor,  # 当前隐藏状态的张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可选参数
    ) -> torch.Tensor:
        # 保存输入的隐藏状态，用于残差连接
        residual = hidden_states

        # 获取隐藏状态的维度数量
        input_ndim = hidden_states.ndim

        # 如果输入维度是4，调整隐藏状态的形状
        if input_ndim == 4:
            # 解包隐藏状态的形状为批量大小、通道、高度和宽度
            batch_size, channel, height, width = hidden_states.shape
            # 将隐藏状态展平为(batch_size, channel, height * width)并转置
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # 确定序列长度和批量大小，根据是否有编码器隐藏状态决定
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # 准备注意力掩码
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # 如果有分组归一化，应用于隐藏状态
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # 将隐藏状态转换为查询向量
        query = attn.to_q(hidden_states)
        # 获取查询向量的最后一维大小
        dim = query.shape[-1]
        # 将查询向量转换为批量维度格式
        query = attn.head_to_batch_dim(query)

        # 如果没有编码器隐藏状态，使用当前隐藏状态
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # 如果需要，归一化编码器隐藏状态
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 将编码器隐藏状态转换为键和值
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # 将键和值转换为批量维度格式
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # 获取查询向量的批量大小和令牌数量
        batch_size_attention, query_tokens, _ = query.shape
        # 初始化隐藏状态张量为零
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        # 按切片处理查询、键和值
        for i in range((batch_size_attention - 1) // self.slice_size + 1):
            # 计算当前切片的起始和结束索引
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            # 获取当前切片的查询、键和注意力掩码
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            # 计算当前切片的注意力分数
            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            # 将注意力分数与值相乘，获取注意力结果
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            # 将注意力结果存储到隐藏状态中
            hidden_states[start_idx:end_idx] = attn_slice

        # 将隐藏状态转换回头维度格式
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # 对隐藏状态进行线性变换
        hidden_states = attn.to_out[0](hidden_states)
        # 应用 dropout
        hidden_states = attn.to_out[1](hidden_states)

        # 如果输入维度是4，调整隐藏状态的形状
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 如果需要残差连接，将残差加到隐藏状态中
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        # 根据缩放因子调整输出
        hidden_states = hidden_states / attn.rescale_output_factor

        # 返回最终的隐藏状态
        return hidden_states
# 定义一个处理器类，用于实现切片注意力，并额外学习键和值矩阵
class SlicedAttnAddedKVProcessor:
    r"""
    处理器，用于实现带有额外可学习的键和值矩阵的切片注意力，用于文本编码器。

    参数：
        slice_size (`int`, *可选*):
            计算注意力的步数。使用 `attention_head_dim // slice_size` 的切片数量，
            并且 `attention_head_dim` 必须是 `slice_size` 的倍数。
    """

    # 初始化方法，接收切片大小作为参数
    def __init__(self, slice_size):
        # 将传入的切片大小赋值给实例变量
        self.slice_size = slice_size

    # 定义调用方法，使类的实例可以像函数一样被调用
    def __call__(
        self,
        attn: "Attention",  # 接收一个注意力对象
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量
        temb: Optional[torch.Tensor] = None,  # 可选的时间嵌入张量
    # 返回类型为 torch.Tensor
        ) -> torch.Tensor:
            # 保存输入的隐藏状态作为残差
            residual = hidden_states
    
            # 如果空间归一化存在，则应用于隐藏状态和时间嵌入
            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)
    
            # 将隐藏状态重塑为三维张量并转置维度
            hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
    
            # 获取批量大小和序列长度
            batch_size, sequence_length, _ = hidden_states.shape
    
            # 准备注意力掩码
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    
            # 如果没有编码器隐藏状态，则将其设置为当前隐藏状态
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            # 如果需要归一化编码器隐藏状态
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    
            # 对隐藏状态应用组归一化并转置维度
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    
            # 生成查询向量
            query = attn.to_q(hidden_states)
            # 获取查询向量的最后一维大小
            dim = query.shape[-1]
            # 将查询向量的维度转换为批次维度
            query = attn.head_to_batch_dim(query)
    
            # 生成编码器隐藏状态的键和值的投影
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
    
            # 将编码器隐藏状态的键和值转换为批次维度
            encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
            encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)
    
            # 如果不只使用交叉注意力
            if not attn.only_cross_attention:
                # 生成当前隐藏状态的键和值
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
                # 将键和值转换为批次维度
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)
                # 将编码器键与当前键拼接
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
                # 将编码器值与当前值拼接
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                # 直接使用编码器的键和值
                key = encoder_hidden_states_key_proj
                value = encoder_hidden_states_value_proj
    
            # 获取批量大小、查询令牌数量和最后一维大小
            batch_size_attention, query_tokens, _ = query.shape
            # 初始化隐藏状态为零张量
            hidden_states = torch.zeros(
                (batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype
            )
    
            # 按切片大小进行迭代处理
            for i in range((batch_size_attention - 1) // self.slice_size + 1):
                start_idx = i * self.slice_size  # 切片起始索引
                end_idx = (i + 1) * self.slice_size  # 切片结束索引
    
                # 获取当前查询切片、键切片和注意力掩码切片
                query_slice = query[start_idx:end_idx]
                key_slice = key[start_idx:end_idx]
                attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None
    
                # 获取当前切片的注意力分数
                attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
    
                # 将注意力分数与当前值进行批量矩阵乘法
                attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
    
                # 将结果存储到隐藏状态
                hidden_states[start_idx:end_idx] = attn_slice
    
            # 将隐藏状态的维度转换回头部维度
            hidden_states = attn.batch_to_head_dim(hidden_states)
    
            # 线性投影
            hidden_states = attn.to_out[0](hidden_states)
            # 应用丢弃层
            hidden_states = attn.to_out[1](hidden_states)
    
            # 转置最后两维并重塑为残差形状
            hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
            # 将残差添加到当前隐藏状态
            hidden_states = hidden_states + residual
    
            # 返回最终的隐藏状态
            return hidden_states
# 定义一个空间归一化类，继承自 nn.Module
class SpatialNorm(nn.Module):
    """
    空间条件归一化，定义在 https://arxiv.org/abs/2209.09002 中。

    参数:
        f_channels (`int`):
            输入到组归一化层的通道数，以及空间归一化层的输出通道数。
        zq_channels (`int`):
            量化向量的通道数，如论文中所述。
    """

    # 初始化方法，接收通道数作为参数
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 创建组归一化层，指定通道数、组数和其他参数
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        # 创建卷积层，输入通道为 zq_channels，输出通道为 f_channels
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        # 创建另一个卷积层，功能相同但用于偏置项
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    # 前向传播方法，定义输入和输出
    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        # 获取输入张量 f 的空间尺寸
        f_size = f.shape[-2:]
        # 对 zq 进行上采样，使其尺寸与 f 相同
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        # 对输入 f 应用归一化层
        norm_f = self.norm_layer(f)
        # 计算新的输出张量，通过归一化后的 f 和卷积结果结合
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        # 返回新的张量
        return new_f


# 定义一个 IPAdapter 注意力处理器类，继承自 nn.Module
class IPAdapterAttnProcessor(nn.Module):
    r"""
    多个 IP-Adapter 的注意力处理器。

    参数:
        hidden_size (`int`):
            注意力层的隐藏尺寸。
        cross_attention_dim (`int`):
            `encoder_hidden_states` 中的通道数。
        num_tokens (`int`, `Tuple[int]` 或 `List[int]`, 默认为 `(4,)`):
            图像特征的上下文长度。
        scale (`float` 或 List[`float`], 默认为 1.0):
            图像提示的权重缩放。
    """

    # 初始化方法，接收多个参数
    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        # 调用父类的初始化方法
        super().__init__()

        # 保存隐藏尺寸
        self.hidden_size = hidden_size
        # 保存交叉注意力维度
        self.cross_attention_dim = cross_attention_dim

        # 确保 num_tokens 为元组或列表
        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        # 保存 num_tokens
        self.num_tokens = num_tokens

        # 确保 scale 为列表
        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        # 验证 scale 和 num_tokens 长度相同
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        # 保存缩放因子
        self.scale = scale

        # 创建用于键的线性变换列表
        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        # 创建用于值的线性变换列表
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    # 定义调用方法，处理输入的注意力信息
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
class IPAdapterAttnProcessor2_0(torch.nn.Module):
    r"""
    PyTorch 2.0 的 IP-Adapter 注意力处理器。
    # 定义参数说明文档，列出类构造函数的参数及其类型和默认值
        Args:
            hidden_size (`int`):
                注意力层的隐藏层大小
            cross_attention_dim (`int`):
                编码器隐藏状态的通道数
            num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
                图像特征的上下文长度
            scale (`float` or `List[float]`, defaults to 1.0):
                图像提示的权重比例
        """
    
        # 初始化类的构造函数，设置类属性
        def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
            # 调用父类的构造函数
            super().__init__()
    
            # 检查 PyTorch 是否支持缩放点积注意力
            if not hasattr(F, "scaled_dot_product_attention"):
                # 如果不支持，抛出导入错误
                raise ImportError(
                    f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
                )
    
            # 设置隐藏层大小属性
            self.hidden_size = hidden_size
            # 设置交叉注意力维度属性
            self.cross_attention_dim = cross_attention_dim
    
            # 如果 num_tokens 不是元组或列表，则将其转换为列表
            if not isinstance(num_tokens, (tuple, list)):
                num_tokens = [num_tokens]
            # 设置 num_tokens 属性
            self.num_tokens = num_tokens
    
            # 如果 scale 不是列表，则创建与 num_tokens 长度相同的列表
            if not isinstance(scale, list):
                scale = [scale] * len(num_tokens)
            # 检查 scale 的长度是否与 num_tokens 相同
            if len(scale) != len(num_tokens):
                # 如果不同，抛出值错误
                raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
            # 设置 scale 属性
            self.scale = scale
    
            # 创建一个包含多个线性层的模块列表，用于输入到 K 的映射
            self.to_k_ip = nn.ModuleList(
                [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
            )
            # 创建一个包含多个线性层的模块列表，用于输入到 V 的映射
            self.to_v_ip = nn.ModuleList(
                [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
            )
    
        # 定义类的调用方法，用于执行注意力计算
        def __call__(
            self,
            attn: Attention,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            temb: Optional[torch.Tensor] = None,
            scale: float = 1.0,
            ip_adapter_masks: Optional[torch.Tensor] = None,
# 定义用于实现 PAG 的处理器类，使用缩放点积注意力（默认启用 PyTorch 2.0）
class PAGIdentitySelfAttnProcessor2_0:
    r"""
    处理器用于实现 PAG，使用缩放点积注意力（默认在 PyTorch 2.0 中启用）。
    PAG 参考: https://arxiv.org/abs/2403.17377
    """

    # 初始化函数
    def __init__(self):
        # 检查 F 中是否有缩放点积注意力功能
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，则抛出导入错误，提示需要升级 PyTorch
            raise ImportError(
                "PAGIdentitySelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    # 可调用方法，定义了注意力处理的输入参数
    def __call__(
        self,
        attn: Attention,  # 输入的注意力对象
        hidden_states: torch.FloatTensor,  # 当前的隐藏状态
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器的隐藏状态（可选）
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码（可选）
        temb: Optional[torch.FloatTensor] = None,  # 额外的时间嵌入（可选）
class PAGCFGIdentitySelfAttnProcessor2_0:
    r"""
    处理器用于实现 PAG，使用缩放点积注意力（默认启用 PyTorch 2.0）。
    PAG 参考: https://arxiv.org/abs/2403.17377
    """

    # 初始化函数
    def __init__(self):
        # 检查 F 中是否有缩放点积注意力功能
        if not hasattr(F, "scaled_dot_product_attention"):
            # 如果没有，则抛出导入错误，提示需要升级 PyTorch
            raise ImportError(
                "PAGCFGIdentitySelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    # 可调用方法，定义了注意力处理的输入参数
    def __call__(
        self,
        attn: Attention,  # 输入的注意力对象
        hidden_states: torch.FloatTensor,  # 当前的隐藏状态
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器的隐藏状态（可选）
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码（可选）
        temb: Optional[torch.FloatTensor] = None,  # 额外的时间嵌入（可选）
class LoRAAttnProcessor:
    # 初始化函数
    def __init__(self):
        # 该类的构造函数，目前没有初始化操作
        pass


class LoRAAttnProcessor2_0:
    # 初始化函数
    def __init__(self):
        # 该类的构造函数，目前没有初始化操作
        pass


class LoRAXFormersAttnProcessor:
    # 初始化函数
    def __init__(self):
        # 该类的构造函数，目前没有初始化操作
        pass


class LoRAAttnAddedKVProcessor:
    # 初始化函数
    def __init__(self):
        # 该类的构造函数，目前没有初始化操作
        pass


# 定义一个包含添加键值注意力处理器的元组
ADDED_KV_ATTENTION_PROCESSORS = (
    AttnAddedKVProcessor,  # 添加键值注意力处理器
    SlicedAttnAddedKVProcessor,  # 切片添加键值注意力处理器
    AttnAddedKVProcessor2_0,  # 添加键值注意力处理器版本2.0
    XFormersAttnAddedKVProcessor,  # XFormers 添加键值注意力处理器
)

# 定义一个包含交叉注意力处理器的元组
CROSS_ATTENTION_PROCESSORS = (
    AttnProcessor,  # 注意力处理器
    AttnProcessor2_0,  # 注意力处理器版本2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
    SlicedAttnProcessor,  # 切片注意力处理器
    IPAdapterAttnProcessor,  # IPAdapter 注意力处理器
    IPAdapterAttnProcessor2_0,  # IPAdapter 注意力处理器版本2.0
)

# 定义一个包含所有注意力处理器的联合类型
AttentionProcessor = Union[
    AttnProcessor,  # 注意力处理器
    AttnProcessor2_0,  # 注意力处理器版本2.0
    FusedAttnProcessor2_0,  # 融合注意力处理器版本2.0
    XFormersAttnProcessor,  # XFormers 注意力处理器
    SlicedAttnProcessor,  # 切片注意力处理器
    AttnAddedKVProcessor,  # 添加键值注意力处理器
    SlicedAttnAddedKVProcessor,  # 切片添加键值注意力处理器
    AttnAddedKVProcessor2_0,  # 添加键值注意力处理器版本2.0
    XFormersAttnAddedKVProcessor,  # XFormers 添加键值注意力处理器
    CustomDiffusionAttnProcessor,  # 自定义扩散注意力处理器
    CustomDiffusionXFormersAttnProcessor,  # 自定义扩散 XFormers 注意力处理器
    CustomDiffusionAttnProcessor2_0,  # 自定义扩散注意力处理器版本2.0
    PAGCFGIdentitySelfAttnProcessor2_0,  # PAGCFG 身份自注意力处理器版本2.0
    PAGIdentitySelfAttnProcessor2_0,  # PAG 身份自注意力处理器版本2.0
    PAGCFGHunyuanAttnProcessor2_0,  # PAGCGHunyuan 注意力处理器版本2.0
    PAGHunyuanAttnProcessor2_0,  # PAG Hunyuan 注意力处理器版本2.0
]
```