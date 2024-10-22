# `.\diffusers\models\transformers\dual_transformer_2d.py`

```py
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件按“原样”分发，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 有关许可的特定权限和限制，请参见许可证。
from typing import Optional  # 从 typing 模块导入 Optional 类型，用于指示可选参数类型

from torch import nn  # 从 torch 模块导入 nn 子模块，提供神经网络的构建块

from ..modeling_outputs import Transformer2DModelOutput  # 从上级模块导入 Transformer2DModelOutput，用于模型输出格式
from .transformer_2d import Transformer2DModel  # 从当前模块导入 Transformer2DModel，用于构建 Transformer 模型


class DualTransformer2DModel(nn.Module):  # 定义 DualTransformer2DModel 类，继承自 nn.Module
    """
    Dual transformer wrapper that combines two `Transformer2DModel`s for mixed inference.
    
    这个类是一个双重变换器的封装器，结合了两个 `Transformer2DModel` 用于混合推理。

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """
    # 初始化方法，用于设置模型参数
        def __init__(
            # 注意力头的数量，默认值为16
            num_attention_heads: int = 16,
            # 每个注意力头的维度，默认值为88
            attention_head_dim: int = 88,
            # 输入通道数，可选参数
            in_channels: Optional[int] = None,
            # 模型层数，默认值为1
            num_layers: int = 1,
            # dropout比率，默认值为0.0
            dropout: float = 0.0,
            # 归一化的组数，默认值为32
            norm_num_groups: int = 32,
            # 交叉注意力维度，可选参数
            cross_attention_dim: Optional[int] = None,
            # 是否使用注意力偏差，默认值为False
            attention_bias: bool = False,
            # 样本大小，可选参数
            sample_size: Optional[int] = None,
            # 向量嵌入的数量，可选参数
            num_vector_embeds: Optional[int] = None,
            # 激活函数，默认值为"geglu"
            activation_fn: str = "geglu",
            # 自适应归一化的嵌入数量，可选参数
            num_embeds_ada_norm: Optional[int] = None,
        ):
            # 调用父类初始化方法
            super().__init__()
            # 创建一个包含两个Transformer2DModel的模块列表
            self.transformers = nn.ModuleList(
                [
                    # 实例化Transformer2DModel，使用传入的参数
                    Transformer2DModel(
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        in_channels=in_channels,
                        num_layers=num_layers,
                        dropout=dropout,
                        norm_num_groups=norm_num_groups,
                        cross_attention_dim=cross_attention_dim,
                        attention_bias=attention_bias,
                        sample_size=sample_size,
                        num_vector_embeds=num_vector_embeds,
                        activation_fn=activation_fn,
                        num_embeds_ada_norm=num_embeds_ada_norm,
                    )
                    # 创建两个Transformer实例
                    for _ in range(2)
                ]
            )
    
            # 可通过管道设置的变量：
    
            # 推理时组合transformer1和transformer2输出状态的比率
            self.mix_ratio = 0.5
    
            # `encoder_hidden_states`的形状预期为
            # `(batch_size, condition_lengths[0]+condition_lengths[1], num_features)`
            self.condition_lengths = [77, 257]
    
            # 指定编码条件时使用哪个transformer。
            # 例如，`(1, 0)`表示使用`transformers[1](conditions[0])`和`transformers[0](conditions[1])`
            self.transformer_index_for_condition = [1, 0]
    
        # 前向传播方法，用于模型的推理过程
        def forward(
            # 隐藏状态输入
            hidden_states,
            # 编码器的隐藏状态
            encoder_hidden_states,
            # 时间步，可选参数
            timestep=None,
            # 注意力掩码，可选参数
            attention_mask=None,
            # 交叉注意力的额外参数，可选参数
            cross_attention_kwargs=None,
            # 是否返回字典格式的输出，默认值为True
            return_dict: bool = True,
    ):
        """
        参数:
            hidden_states ( 当为离散时，`torch.LongTensor` 形状为 `(batch size, num latent pixels)`。
                当为连续时，`torch.Tensor` 形状为 `(batch size, channel, height, width)`): 输入的 hidden_states。
            encoder_hidden_states ( `torch.LongTensor` 形状为 `(batch size, encoder_hidden_states dim)`，*可选*):
                用于交叉注意力层的条件嵌入。如果未提供，交叉注意力将默认为
                自注意力。
            timestep ( `torch.long`，*可选*):
                可选的时间步长，将作为 AdaLayerNorm 中的嵌入使用。用于指示去噪步骤。
            attention_mask (`torch.Tensor`，*可选*):
                可选的注意力掩码，应用于注意力。
            cross_attention_kwargs (`dict`，*可选*):
                如果指定，将传递给 `AttentionProcessor` 的关键字参数字典，如
                在 `self.processor` 中定义的
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)。
            return_dict (`bool`，*可选*，默认为 `True`):
                是否返回 [`models.unets.unet_2d_condition.UNet2DConditionOutput`] 而不是简单的
                元组。

        返回:
            [`~models.transformers.transformer_2d.Transformer2DModelOutput`] 或 `tuple`:
            如果 `return_dict` 为 True，则返回 [`~models.transformers.transformer_2d.Transformer2DModelOutput`]，否则返回
            `tuple`。当返回元组时，第一个元素是样本张量。
        """
        # 将输入的 hidden_states 赋值给 input_states
        input_states = hidden_states

        # 初始化一个空列表用于存储编码后的状态
        encoded_states = []
        # 初始化 token 的起始位置为 0
        tokens_start = 0
        # attention_mask 目前尚未使用
        for i in range(2):
            # 对于两个变换器中的每一个，传递相应的条件标记
            condition_state = encoder_hidden_states[:, tokens_start : tokens_start + self.condition_lengths[i]]
            # 根据条件标记的索引获取对应的变换器
            transformer_index = self.transformer_index_for_condition[i]
            # 调用变换器处理输入状态和条件状态，并获取输出
            encoded_state = self.transformers[transformer_index](
                input_states,
                encoder_hidden_states=condition_state,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]  # 只获取输出的第一个元素
            # 将编码后的状态与输入状态相减，存入列表
            encoded_states.append(encoded_state - input_states)
            # 更新 token 的起始位置
            tokens_start += self.condition_lengths[i]

        # 结合两个编码后的状态，计算输出状态
        output_states = encoded_states[0] * self.mix_ratio + encoded_states[1] * (1 - self.mix_ratio)
        # 将计算后的输出状态与输入状态相加
        output_states = output_states + input_states

        # 如果不返回字典格式
        if not return_dict:
            # 返回输出状态的元组
            return (output_states,)

        # 返回包含样本输出的 Transformer2DModelOutput 对象
        return Transformer2DModelOutput(sample=output_states)
```