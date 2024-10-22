# `.\diffusers\pipelines\blip_diffusion\modeling_blip2.py`

```py
# 版权信息，表示该代码归 HuggingFace 团队所有
# 
# 根据 Apache 许可证 2.0 版授权；
# 除非遵守许可证，否则不得使用此文件。
# 可以在以下地址获取许可证：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则根据许可证分发的软件是“按现状”提供的，
# 不提供任何明示或暗示的担保或条件。
# 请参见许可证，以了解管理权限和
# 限制的具体条款。
from typing import Optional, Tuple, Union  # 导入类型提示模块，包含可选、元组和联合类型

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 检查点工具
from torch import nn  # 从 PyTorch 导入神经网络模块
from transformers import BertTokenizer  # 从 transformers 导入 BERT 分词器
from transformers.activations import QuickGELUActivation as QuickGELU  # 导入快速 GELU 激活函数并重命名
from transformers.modeling_outputs import (  # 从 transformers 导入多种模型输出格式
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.blip_2.configuration_blip_2 import Blip2Config, Blip2VisionConfig  # 导入 BLIP-2 配置类
from transformers.models.blip_2.modeling_blip_2 import (  # 从 BLIP-2 导入模型类
    Blip2Encoder,
    Blip2PreTrainedModel,
    Blip2QFormerAttention,
    Blip2QFormerIntermediate,
    Blip2QFormerOutput,
)
from transformers.pytorch_utils import apply_chunking_to_forward  # 导入应用前向分块的工具
from transformers.utils import (  # 从 transformers 导入工具函数
    logging,
    replace_return_docstrings,
)

logger = logging.get_logger(__name__)  # 创建一个日志记录器，以当前模块名为标识

# 在 `transformers` 中有 BLIP2 的实现：https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py。
# 但它不支持获取多模态嵌入。因此，可以用将来支持此功能的 `transformers` 版本替换此模块。
class Blip2TextEmbeddings(nn.Module):  # 定义 Blip2 文本嵌入类，继承自 nn.Module
    """从词和位置嵌入构建嵌入。"""

    def __init__(self, config):  # 初始化方法，接受配置参数
        super().__init__()  # 调用父类构造函数
        # 创建词嵌入层，使用词汇大小和隐藏层大小
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，使用最大位置嵌入和隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 将 LayerNorm 命名为非蛇形格式，以便与 TensorFlow 模型变量名一致，从而能够加载
        # 任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建 Dropout 层

        # 创建位置 ID 缓冲区，表示连续内存中的位置嵌入
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # 获取位置嵌入类型，默认为绝对位置
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.config = config  # 保存配置

    def forward(  # 定义前向传播方法
        self,
        input_ids=None,  # 输入的 ID
        position_ids=None,  # 输入的位置 ID
        query_embeds=None,  # 查询嵌入
        past_key_values_length=0,  # 过去的键值长度
    # 方法体开始，接受参数
        ):
            # 如果输入ID不为None
            if input_ids is not None:
                # 获取输入序列的长度
                seq_length = input_ids.size()[1]
            else:
                # 如果输入ID为None，序列长度设为0
                seq_length = 0
    
            # 如果位置ID为None
            if position_ids is None:
                # 从位置ID矩阵中提取所需的部分，并克隆
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length].clone()
    
            # 如果输入ID不为None
            if input_ids is not None:
                # 根据输入ID获取词嵌入
                embeddings = self.word_embeddings(input_ids)
                # 如果位置嵌入类型为绝对位置
                if self.position_embedding_type == "absolute":
                    # 获取位置嵌入
                    position_embeddings = self.position_embeddings(position_ids)
                    # 将词嵌入与位置嵌入相加
                    embeddings = embeddings + position_embeddings
    
                # 如果查询嵌入不为None
                if query_embeds is not None:
                    # 获取批次大小
                    batch_size = embeddings.shape[0]
                    # 重复查询嵌入以匹配批次大小
                    query_embeds = query_embeds.repeat(batch_size, 1, 1)
                    # 将查询嵌入和词嵌入在维度1上拼接
                    embeddings = torch.cat((query_embeds, embeddings), dim=1)
            else:
                # 如果输入ID为None，使用查询嵌入
                embeddings = query_embeds
            # 将嵌入转换为查询嵌入的数据类型
            embeddings = embeddings.to(query_embeds.dtype)
            # 对嵌入进行层归一化
            embeddings = self.LayerNorm(embeddings)
            # 对嵌入应用dropout
            embeddings = self.dropout(embeddings)
            # 返回最终的嵌入
            return embeddings
# 从 transformers.models.blip.modeling_blip.BlipVisionEmbeddings 复制而来，进行了 Blip 到 Blip2 的修改
class Blip2VisionEmbeddings(nn.Module):
    # 初始化 Blip2VisionEmbeddings 类，接收配置参数
    def __init__(self, config: Blip2VisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置参数
        self.config = config
        # 获取隐藏层大小作为嵌入维度
        self.embed_dim = config.hidden_size
        # 获取图像大小
        self.image_size = config.image_size
        # 获取补丁大小
        self.patch_size = config.patch_size

        # 初始化类嵌入参数，随机生成
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # 创建补丁嵌入卷积层，输入通道为3，输出通道为嵌入维度
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        # 计算总补丁数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # 总位置数量比补丁数量多一个（类嵌入）
        self.num_positions = self.num_patches + 1

        # 初始化位置嵌入参数，随机生成
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    # 前向传播方法，接收像素值输入
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取批大小
        batch_size = pixel_values.shape[0]
        # 获取补丁嵌入的权重数据类型
        target_dtype = self.patch_embedding.weight.dtype
        # 通过补丁嵌入层处理像素值，得到补丁嵌入
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        # 将补丁嵌入展平并转置
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 扩展类嵌入以匹配批大小，并转换为目标数据类型
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # 将类嵌入和补丁嵌入进行拼接
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 加上位置嵌入
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        # 返回最终的嵌入
        return embeddings


# Qformer 编码器，接收视觉嵌入和文本输入，以获取多模态嵌入
class Blip2QFormerEncoder(nn.Module):
    # 初始化 Blip2QFormerEncoder 类，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置参数
        self.config = config
        # 创建一个包含多个 Blip2QFormerLayer 的模块列表
        self.layer = nn.ModuleList(
            [Blip2QFormerLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收隐藏状态和可选参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,
# 构成 Qformer 编码器的各层
class Blip2QFormerLayer(nn.Module):
    # 初始化方法，接收配置和层索引作为参数
    def __init__(self, config, layer_idx):
        # 调用父类的初始化方法
        super().__init__()
        # 设置前馈网络的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 初始化注意力机制
        self.attention = Blip2QFormerAttention(config)

        # 存储当前层的索引
        self.layer_idx = layer_idx

        # 判断当前层是否需要交叉注意力
        if layer_idx % config.cross_attention_frequency == 0:
            # 初始化交叉注意力机制
            self.crossattention = Blip2QFormerAttention(config, is_cross_attention=True)
            # 标记当前层有交叉注意力
            self.has_cross_attention = True
        else:
            # 标记当前层没有交叉注意力
            self.has_cross_attention = False

        # 初始化中间层的前馈网络
        self.intermediate = Blip2QFormerIntermediate(config)
        # 初始化中间查询层的前馈网络
        self.intermediate_query = Blip2QFormerIntermediate(config)
        # 初始化输出查询层的前馈网络
        self.output_query = Blip2QFormerOutput(config)
        # 初始化输出层的前馈网络
        self.output = Blip2QFormerOutput(config)

    # 前向传播方法，定义网络的输入和输出
    def forward(
        self,
        hidden_states,  # 隐藏状态输入
        attention_mask=None,  # 注意力掩码（可选）
        head_mask=None,  # 注意力头掩码（可选）
        encoder_hidden_states=None,  # 编码器的隐藏状态（可选）
        encoder_attention_mask=None,  # 编码器的注意力掩码（可选）
        past_key_value=None,  # 过去的键值（可选）
        output_attentions=False,  # 是否输出注意力权重
        query_length=0,  # 查询的长度
    ):
        # 解码器单向自注意力的缓存键/值元组位于位置 1 和 2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用自注意力机制，传入隐藏状态及相关参数
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出中的主要注意力结果
        attention_output = self_attention_outputs[0]
        # 获取自注意力输出中的其他结果，排除首尾
        outputs = self_attention_outputs[1:-1]

        # 获取当前自注意力的键/值元组
        present_key_value = self_attention_outputs[-1]

        # 如果查询长度大于 0
        if query_length > 0:
            # 获取查询的注意力输出
            query_attention_output = attention_output[:, :query_length, :]

            # 如果有交叉注意力
            if self.has_cross_attention:
                # 检查编码器隐藏状态是否提供
                if encoder_hidden_states is None:
                    raise ValueError("encoder_hidden_states must be given for cross-attention layers")
                # 调用交叉注意力机制
                cross_attention_outputs = self.crossattention(
                    query_attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                # 获取交叉注意力的输出
                query_attention_output = cross_attention_outputs[0]
                # 如果输出注意力权重，添加交叉注意力的输出
                outputs = outputs + cross_attention_outputs[1:-1]

            # 应用前馈网络到查询注意力输出
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk_query,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                query_attention_output,
            )

            # 如果注意力输出的序列长度大于查询长度
            if attention_output.shape[1] > query_length:
                # 应用前馈网络到注意力输出的后半部分
                layer_output_text = apply_chunking_to_forward(
                    self.feed_forward_chunk,
                    self.chunk_size_feed_forward,
                    self.seq_len_dim,
                    attention_output[:, query_length:, :],
                )
                # 合并层输出
                layer_output = torch.cat([layer_output, layer_output_text], dim=1)
        else:
            # 如果查询长度为 0，直接应用前馈网络
            layer_output = apply_chunking_to_forward(
                self.feed_forward_chunk,
                self.chunk_size_feed_forward,
                self.seq_len_dim,
                attention_output,
            )
        # 将层输出与其他输出组合
        outputs = (layer_output,) + outputs

        # 将当前键/值元组添加到输出中
        outputs = outputs + (present_key_value,)

        # 返回所有输出
        return outputs

    # 前馈网络的块函数
    def feed_forward_chunk(self, attention_output):
        # 计算中间输出
        intermediate_output = self.intermediate(attention_output)
        # 计算最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    # 查询的前馈网络块函数
    def feed_forward_chunk_query(self, attention_output):
        # 计算查询的中间输出
        intermediate_output = self.intermediate_query(attention_output)
        # 计算最终查询层输出
        layer_output = self.output_query(intermediate_output, attention_output)
        return layer_output
# ProjLayer 用于将多模态 Blip2 嵌入投影到文本编码器中
class ProjLayer(nn.Module):
    # 初始化方法，定义输入维度、输出维度、隐藏层维度、丢弃率和 epsilon
    def __init__(self, in_dim, out_dim, hidden_dim, drop_p=0.1, eps=1e-12):
        # 调用父类构造函数
        super().__init__()

        # 定义全连接层1 -> 激活函数 -> 全连接层2 -> 丢弃层 -> 残差连接 -> 归一化层
        self.dense1 = nn.Linear(in_dim, hidden_dim)  # 第一层全连接
        self.act_fn = QuickGELU()  # 激活函数使用 QuickGELU
        self.dense2 = nn.Linear(hidden_dim, out_dim)  # 第二层全连接
        self.dropout = nn.Dropout(drop_p)  # 定义丢弃层，减少过拟合

        self.LayerNorm = nn.LayerNorm(out_dim, eps=eps)  # 归一化层

    # 前向传播方法
    def forward(self, x):
        x_in = x  # 保存输入以用于残差连接

        x = self.LayerNorm(x)  # 对输入进行层归一化
        # 通过全连接层1 -> 激活函数 -> 全连接层2 -> 丢弃层，进行处理并加上输入（残差连接）
        x = self.dropout(self.dense2(self.act_fn(self.dense1(x)))) + x_in

        return x  # 返回处理后的输出


# 从 transformers.models.blip.modeling_blip.BlipVisionModel 复制并修改 Blip->Blip2, BLIP->BLIP_2
class Blip2VisionModel(Blip2PreTrainedModel):
    main_input_name = "pixel_values"  # 主要输入的名称
    config_class = Blip2VisionConfig  # 配置类

    # 初始化方法，传入配置对象
    def __init__(self, config: Blip2VisionConfig):
        # 调用父类构造函数
        super().__init__(config)
        self.config = config  # 保存配置
        embed_dim = config.hidden_size  # 嵌入维度
        self.embeddings = Blip2VisionEmbeddings(config)  # 初始化嵌入层
        self.pre_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 前置层归一化
        self.encoder = Blip2Encoder(config)  # 初始化编码器
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)  # 后置层归一化

        self.post_init()  # 后初始化处理

    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=Blip2VisionConfig)
    # 前向传播方法
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 输入的像素值
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回
    ) -> Union[Tuple, BaseModelOutputWithPooling]:  # 指定函数返回类型为元组或带有池化输出的基础模型输出
        r"""  # 文档字符串的开始，通常用于描述函数的用途
        Returns:  # 返回部分的说明
        """  # 文档字符串的结束
        # 判断是否需要输出注意力权重，如果未指定则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 判断是否需要输出隐藏状态，如果未指定则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 判断是否返回字典形式的输出，如果未指定则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果没有提供像素值，则抛出错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 通过嵌入层将像素值转换为隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 进行层归一化处理
        hidden_states = self.pre_layernorm(hidden_states)
        # 将隐藏状态输入编码器，获取编码器输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,  # 输入的嵌入
            output_attentions=output_attentions,  # 是否输出注意力
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
        )
        # 获取编码器输出中的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0]
        # 对最后的隐藏状态进行后续层归一化
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 从最后的隐藏状态中提取池化输出，通常是[CLS]标记的表示
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后续层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典，则返回最后的隐藏状态、池化输出和其他编码器输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 返回包含最后隐藏状态、池化输出、隐藏状态和注意力的基础模型输出
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,  # 最后的隐藏状态
            pooler_output=pooled_output,  # 池化输出
            hidden_states=encoder_outputs.hidden_states,  # 编码器的隐藏状态
            attentions=encoder_outputs.attentions,  # 编码器的注意力
        )

    def get_input_embeddings(self):  # 定义获取输入嵌入的方法
        return self.embeddings  # 返回嵌入层
# Qformer model, used to get multimodal embeddings from the text and image inputs
class Blip2QFormerModel(Blip2PreTrainedModel):
    """ 
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self, config: Blip2Config):
        # 初始化父类，传入配置
        super().__init__(config)
        # 保存配置对象
        self.config = config
        # 创建文本嵌入层，使用 Q-Former 的配置
        self.embeddings = Blip2TextEmbeddings(config.qformer_config)
        # 创建视觉编码器，使用视觉模型的配置
        self.visual_encoder = Blip2VisionModel(config.vision_config)
        # 初始化查询 token 的参数，形状为 (1, num_query_tokens, hidden_size)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        # 检查配置是否包含 tokenizer，如果没有则使用默认的 BERT tokenizer
        if not hasattr(config, "tokenizer") or config.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        else:
            # 使用配置中的 tokenizer
            self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer, truncation_side="right")
        # 添加特殊的开始 token
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # 创建投影层，设置输入、输出维度和隐藏层维度
        self.proj_layer = ProjLayer(
            in_dim=config.qformer_config.hidden_size,
            out_dim=config.qformer_config.hidden_size,
            hidden_dim=config.qformer_config.hidden_size * 4,
            drop_p=0.1,
            eps=1e-12,
        )
        # 创建 Q-Former 编码器，使用配置
        self.encoder = Blip2QFormerEncoder(config.qformer_config)
        # 调用后初始化方法
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层的词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层的词嵌入
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历每一层和需要剪枝的头
        for layer, heads in heads_to_prune.items():
            # 对指定层的注意力头进行剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    def get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int],
        device: torch.device,
        has_query: bool = False,
        # ...
    ) -> torch.Tensor:  # 指定该函数返回一个 torch.Tensor 类型的值
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.  # 准备可广播的注意力和因果掩码，以忽略未来和被掩盖的标记。

        Arguments:  # 参数说明
            attention_mask (`torch.Tensor`):  # 注意力掩码，类型为 torch.Tensor
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.  # 掩码中，1表示要关注的标记，0表示要忽略的标记。
            input_shape (`Tuple[int]`):  # 输入的形状，类型为整数元组
                The shape of the input to the model.  # 模型输入的形状。
            device (`torch.device`):  # 输入的设备类型
                The device of the input to the model.  # 模型输入的设备。
        
        Returns:  # 返回值说明
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.  # 返回扩展的注意力掩码，其数据类型与 attention_mask 的数据类型相同。
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]  # 可以提供自注意力掩码，维度为 [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.  # 在这种情况下，只需将其设置为可广播到所有头部。
        if attention_mask.dim() == 3:  # 如果注意力掩码是 3 维
            extended_attention_mask = attention_mask[:, None, :, :]  # 扩展掩码以增加一个维度，使其可以广播到所有头
        elif attention_mask.dim() == 2:  # 如果注意力掩码是 2 维
            # Provided a padding mask of dimensions [batch_size, seq_length]  # 提供的填充掩码维度为 [batch_size, seq_length]
            # - the model is an encoder, so make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]  # - 模型是编码器，因此将掩码扩展为可广播到 [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]  # 扩展掩码以增加两个维度
        else:  # 如果不是以上情况
            raise ValueError(  # 抛出值错误
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(  # 错误信息，说明 input_ids 或 attention_mask 的形状不正确
                    input_shape, attention_mask.shape  # 显示输入形状和注意力掩码形状
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for  # 由于 attention_mask 在要关注的位置为 1.0，在掩盖的位置为 0.0
        # masked positions, this operation will create a tensor which is 0.0 for  # 这个操作将创建一个张量，在要关注的位置为 0.0，在被掩盖的位置为 -10000.0
        # positions we want to attend and -10000.0 for masked positions.  # 由于我们在 softmax 之前将其添加到原始分数，这实际上与完全删除这些位置相同。
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility  # 将扩展的注意力掩码转换为与模型数据类型兼容的格式（例如 fp16）
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 将掩码转换为注意力得分，关注位置为 0.0，被掩盖位置为 -10000.0
        return extended_attention_mask  # 返回扩展的注意力掩码

    def forward(  # 定义前向传播函数
        self,  # self 参数，引用当前实例
        text_input=None,  # 文本输入，默认为 None
        image_input=None,  # 图像输入，默认为 None
        head_mask=None,  # 头部掩码，默认为 None
        encoder_hidden_states=None,  # 编码器隐藏状态，默认为 None
        encoder_attention_mask=None,  # 编码器注意力掩码，默认为 None
        past_key_values=None,  # 过去的键值，默认为 None
        use_cache=None,  # 是否使用缓存，默认为 None
        output_attentions=None,  # 是否输出注意力，默认为 None
        output_hidden_states=None,  # 是否输出隐藏状态，默认为 None
        return_dict=None,  # 是否返回字典，默认为 None
```