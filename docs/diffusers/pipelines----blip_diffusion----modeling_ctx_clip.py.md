# `.\diffusers\pipelines\blip_diffusion\modeling_ctx_clip.py`

```py
# 版权所有 2024 Salesforce.com, inc.
# 版权所有 2024 The HuggingFace Team. 保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（"许可证"）许可；
# 除非遵循许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 是按“原样”基础提供的，不提供任何形式的保证或条件，
# 明示或暗示。有关许可证的特定语言的权限和限制，请参见
# 许可证。
from typing import Optional, Tuple, Union  # 从 typing 模块导入 Optional、Tuple 和 Union 类型提示

import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 导入神经网络模块
from transformers import CLIPPreTrainedModel  # 从 transformers 导入 CLIP 预训练模型基类
from transformers.modeling_outputs import BaseModelOutputWithPooling  # 导入带池化的基础模型输出
from transformers.models.clip.configuration_clip import CLIPTextConfig  # 导入 CLIP 文本配置
from transformers.models.clip.modeling_clip import CLIPEncoder  # 导入 CLIP 编码器


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    扩展 attention_mask 从 `[bsz, seq_len]` 到 `[bsz, 1, tgt_seq_len, src_seq_len]`。
    """
    bsz, src_len = mask.size()  # 获取输入掩码的批量大小和源序列长度
    tgt_len = tgt_len if tgt_len is not None else src_len  # 如果目标长度为 None，则使用源长度

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)  # 扩展掩码到目标维度并转换类型

    inverted_mask = 1.0 - expanded_mask  # 反转掩码，将 1 变为 0，0 变为 1

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)  # 用最小浮点数填充反转掩码中为 True 的位置


# 这是 transformers.models.clip.modeling_clip 中的 CLIPTextModel 的修改版本
# 允许额外输入“上下文嵌入”，即 Qformer 中使用的查询嵌入
# 它们与文本嵌入一起通过 clip 模型，并使用自注意力与之交互
class ContextCLIPTextModel(CLIPPreTrainedModel):  # 定义上下文 CLIP 文本模型类，继承自 CLIP 预训练模型
    config_class = CLIPTextConfig  # 指定配置类为 CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]  # 定义不应被拆分的模块列表

    def __init__(self, config: CLIPTextConfig):  # 初始化方法，接受 CLIPTextConfig 配置
        super().__init__(config)  # 调用父类的初始化方法
        self.text_model = ContextCLIPTextTransformer(config)  # 创建上下文 CLIP 文本转换器模型
        # 初始化权重并应用最终处理
        self.post_init()  # 调用后处理方法

    def forward(  # 定义前向传播方法
        self,
        ctx_embeddings: torch.Tensor = None,  # 上下文嵌入，默认为 None
        ctx_begin_pos: list = None,  # 上下文开始位置列表，默认为 None
        input_ids: Optional[torch.Tensor] = None,  # 输入 ID，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，默认为 None
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，默认为 None
    ) -> Union[Tuple, BaseModelOutputWithPooling]:  # 定义返回类型为元组或带池化的基础模型输出
        return self.text_model(  # 调用文本模型的前向传播方法
            ctx_embeddings=ctx_embeddings,  # 传递上下文嵌入
            ctx_begin_pos=ctx_begin_pos,  # 传递上下文开始位置
            input_ids=input_ids,  # 传递输入 ID
            attention_mask=attention_mask,  # 传递注意力掩码
            position_ids=position_ids,  # 传递位置 ID
            output_attentions=output_attentions,  # 传递输出注意力参数
            output_hidden_states=output_hidden_states,  # 传递输出隐藏状态参数
            return_dict=return_dict,  # 传递返回字典参数
        )  # 返回文本模型的前向传播结果
# 定义一个名为 ContextCLIPTextTransformer 的类，继承自 nn.Module
class ContextCLIPTextTransformer(nn.Module):
    # 初始化方法，接受一个配置对象 config，类型为 CLIPTextConfig
    def __init__(self, config: CLIPTextConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置对象存储为实例属性
        self.config = config
        # 获取隐藏层的维度
        embed_dim = config.hidden_size
        # 创建上下文 CLIP 文本嵌入对象
        self.embeddings = ContextCLIPTextEmbeddings(config)
        # 创建 CLIP 编码器对象
        self.encoder = CLIPEncoder(config)
        # 创建最终层的归一化层
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    # 定义前向传播方法，处理输入的张量和其他参数
    def forward(
        self,
        # 上下文嵌入的张量
        ctx_embeddings: torch.Tensor,
        # 上下文开始位置的列表
        ctx_begin_pos: list,
        # 可选的输入 ID 张量
        input_ids: Optional[torch.Tensor] = None,
        # 可选的注意力掩码张量
        attention_mask: Optional[torch.Tensor] = None,
        # 可选的位置 ID 张量
        position_ids: Optional[torch.Tensor] = None,
        # 可选的输出注意力标志
        output_attentions: Optional[bool] = None,
        # 可选的输出隐藏状态标志
        output_hidden_states: Optional[bool] = None,
        # 可选的返回字典标志
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r""" 
        # 文档字符串，说明返回值类型
        Returns:

        """
        # 如果 output_attentions 为 None，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 为 None，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为 None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 input_ids 为 None，抛出错误
        if input_ids is None:
            raise ValueError("You have to specify either input_ids")

        # 获取输入张量的形状
        input_shape = input_ids.size()
        # 将 input_ids 调整为二维张量，第二维为输入的最后一维
        input_ids = input_ids.view(-1, input_shape[-1])

        # 使用嵌入层处理输入 ids 以获取隐藏状态
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            ctx_embeddings=ctx_embeddings,
            ctx_begin_pos=ctx_begin_pos,
        )

        # 获取批次大小和序列长度
        bsz, seq_len = input_shape
        # 如果存在上下文嵌入，更新序列长度
        if ctx_embeddings is not None:
            seq_len += ctx_embeddings.size(1)
        # 准备因果注意力掩码
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # 如果提供了注意力掩码，则扩展它
        if attention_mask is not None:
            # 将 [bsz, seq_len] 扩展为 [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        # 将嵌入的隐藏状态传入编码器，并获取输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后的隐藏状态进行层归一化处理
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # 从 eot 嵌入中获取特征（eot_token 是每个序列中的最大值）
        # 为了与 onnx 兼容，转换为 torch.int：argmax 不支持 opset 14 的 int64 输入
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=input_ids.device),
            input_ids.to(torch.int).argmax(dim=-1),
        ]

        # 如果不需要返回字典格式，则返回最后隐藏状态、池化输出和编码器其他输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回带有池化输出的模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 定义构建因果注意力掩码的方法，参数包括批次大小、序列长度和数据类型
    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # 延迟创建因果注意力掩码，确保视觉标记之间有完全注意力
        # pytorch 使用加法注意力掩码；填充 -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)  # 创建一个空的张量作为掩码
        mask.fill_(torch.tensor(torch.finfo(dtype).min))  # 用数据类型的最小值填充掩码
        mask.triu_(1)  # 将下三角部分置零，保留上三角部分
        mask = mask.unsqueeze(1)  # 扩展掩码的维度，以便与其他张量兼容
        return mask  # 返回生成的因果注意力掩码
# 定义一个名为 ContextCLIPTextEmbeddings 的类，继承自 nn.Module
class ContextCLIPTextEmbeddings(nn.Module):
    # 初始化方法，接受一个配置对象 config
    def __init__(self, config: CLIPTextConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 获取嵌入维度，来自配置对象的 hidden_size 属性
        embed_dim = config.hidden_size

        # 创建一个词嵌入层，输入词汇大小和嵌入维度
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 创建一个位置嵌入层，输入最大位置嵌入数和嵌入维度
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 创建一个名为 position_ids 的缓冲区，表示位置 ID 的张量
        # 位置 ID 为 (1, len position emb)，在内存中是连续的，并在序列化时导出
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    # 前向传播方法，接受上下文嵌入和其他可选参数
    def forward(
        self,
        ctx_embeddings: torch.Tensor,
        ctx_begin_pos: list,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 如果 ctx_embeddings 为空，设置上下文长度为 0
        if ctx_embeddings is None:
            ctx_len = 0
        else:
            # 获取上下文嵌入的长度
            ctx_len = ctx_embeddings.shape[1]

        # 计算序列长度，如果 input_ids 为空，则使用 inputs_embeds 的倒数第二维度
        seq_length = (input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]) + ctx_len

        # 如果 position_ids 为空，从位置缓冲区获取相应的 position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 inputs_embeds 为空，从 token_embedding 获取嵌入
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

            # 为每个输入嵌入，在正确位置添加上下文嵌入
            input_embeds_ctx = []
            # 获取批次大小
            bsz = inputs_embeds.shape[0]

            # 如果 ctx_embeddings 不为空，进行上下文嵌入的拼接
            if ctx_embeddings is not None:
                # 遍历每个样本
                for i in range(bsz):
                    # 获取当前样本的上下文开始位置
                    cbp = ctx_begin_pos[i]

                    # 获取输入嵌入的前缀部分
                    prefix = inputs_embeds[i, :cbp]
                    # 获取输入嵌入的后缀部分，移除特殊标记的嵌入
                    suffix = inputs_embeds[i, cbp:]

                    # 将前缀、上下文嵌入和后缀拼接起来
                    input_embeds_ctx.append(torch.cat([prefix, ctx_embeddings[i], suffix], dim=0))

                # 将所有样本的输入嵌入堆叠成一个张量
                inputs_embeds = torch.stack(input_embeds_ctx, dim=0)

        # 获取位置嵌入
        position_embeddings = self.position_embedding(position_ids)
        # 计算最终的嵌入，将输入嵌入与位置嵌入相加
        embeddings = inputs_embeds + position_embeddings

        # 返回计算得到的嵌入
        return embeddings
```