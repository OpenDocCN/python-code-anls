# `.\transformers\models\blip\modeling_blip.py`

```py
# 定义一个数据类，用于存储 BLIP 有条件生成模型的输出
@dataclass
class BlipForConditionalGenerationModelOutput(ModelOutput):
    """
    从包含最后隐藏状态的视觉模型输出的基类进行了调整，还包含了图像嵌入的池化。该类还添加了文本解码器的损失项。
    """
    Args:
        loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Languge modeling loss from the text decoder.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`, *optional`):
            Prediction scores of the language modeling head of the text decoder model.
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional`):
            The image embeddings obtained after applying the Vision Transformer model to the input image.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 定义可选的属性，用于存储不同类型的数据
    loss: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # 定义属性的装饰器，用于获取`decoder_logits`属性，同时发出警告
    @property
    def decoder_logits(self):
        warnings.warn(
            "`decoder_logits` attribute is deprecated and will be removed in version 5 of Transformers."
            " Please use the `logits` attribute to retrieve the final output instead.",
            FutureWarning,
        )
        return self.logits
# 定义一个数据类，用于存储 BlipTextVisionModel 的输出结果
@dataclass
class BlipTextVisionModelOutput(ModelOutput):
    """
    从包含图像嵌入的最后隐藏状态的视觉模型输出基类进行调整。该类还添加了文本解码器的损失项。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            文本解码器的语言建模损失。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            通过将投影层应用于池化器输出获得的图像嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor` 元组，
            包含模型在每一层输出的隐藏状态（如果模型有嵌入层，则包含嵌入层的输出）。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `torch.FloatTensor` 元组，
            包含注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，用于存储 BlipImageTextMatchingModel 的输出结果
@dataclass
class BlipImageTextMatchingModelOutput(ModelOutput):
    """
    从包含图像嵌入的最后隐藏状态的视觉模型输出基类进行调整。该类还添加了文本解码器的损失项以及图像文本相似度分数。
    """
    Args:
        itm_score (`torch.FloatTensor`):
            图像文本相似度得分。
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            文本解码器产生的语言建模损失。
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            通过将池化输出应用于投影层而获得的图像嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（如果模型有嵌入层，则一个用于嵌入的输出，每个层的一个用于输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            模型在每一层的隐藏状态加上可选的初始嵌入输出。
        vision_pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            模型的仅视觉分支的视觉最后一层隐藏状态。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（每层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
        question_embeds (`torch.FloatTensor`):
            通过文本投影层获得的问题嵌入。
    """

    itm_score: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_pooler_output: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    question_embeds: Optional[Tuple[torch.FloatTensor]] = None
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling
import torch
import torch.nn as nn

# 定义一个名为BlipOutput的数据类，用于存储BLIP模型的输出结果
@dataclass
class BlipOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for image-text similarity.
        logits_per_image:(`torch.FloatTensor` of shape `(image_batch_size, text_batch_size)`):
            The scaled dot product scores between `image_embeds` and `text_embeds`. This represents the image-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, image_batch_size)`):
            The scaled dot product scores between `text_embeds` and `image_embeds`. This represents the text-image
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`BlipTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`BlipVisionModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipTextModel`].
        vision_model_output(`BaseModelOutputWithPooling`):
            The output of the [`BlipVisionModel`].
    """

    # 用于存储对比损失的张量
    loss: Optional[torch.FloatTensor] = None
    # 存储图像与文本之间的相似度得分
    logits_per_image: torch.FloatTensor = None
    # 存储文本与图像之间的相似度得分
    logits_per_text: torch.FloatTensor = None
    # 存储文本嵌入
    text_embeds: torch.FloatTensor = None
    # 存储图像嵌入
    image_embeds: torch.FloatTensor = None
    # 存储文本模型的输出
    text_model_output: BaseModelOutputWithPooling = None
    # 存储图像模型的输出
    vision_model_output: BaseModelOutputWithPooling = None

    # 将BlipOutput对象转换为元组形式
    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


# 定义BlipVisionEmbeddings类，用于处理BLIP模型的图像嵌入
class BlipVisionEmbeddings(nn.Module):
    def __init__(self, config: BlipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 类别嵌入，用于表示图像的类别信息
        self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # 图像分块嵌入，用于提取图像的局部特征
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        # 计算图像分块的数量和位置嵌入的数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # 位置嵌入，用于表示图像中不同位置的信息
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))


```  
    # 前向传播函数，接受像素值张量作为输入，并返回张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取输入张量的批量大小
        batch_size = pixel_values.shape[0]
        # 获取目标数据类型，与嵌入权重的数据类型相同
        target_dtype = self.patch_embedding.weight.dtype
        # 使用嵌入层将像素值转换为嵌入表示，shape = [*, width, grid, grid]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        # 将嵌入表示扁平化并转置，以便与类别嵌入进行拼接
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 复制类别嵌入以匹配批量大小，并转换为目标数据类型
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        # 将类别嵌入与图像嵌入拼接在一起
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 添加位置嵌入到拼接后的嵌入中
        embeddings = embeddings + self.position_embedding[:, : embeddings.size(1), :].to(target_dtype)
        # 返回嵌入表示
        return embeddings
# 从transformers.models.clip.modeling_clip.CLIPTextEmbeddings复制代码，并将CLIP->Blip
class BlipTextEmbeddings(nn.Module):
    def __init__(self, config: BlipTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        # 创建一个词嵌入层，将词索引映射为嵌入向量
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        # 创建一个位置嵌入层，将位置索引映射为嵌入向量
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # 创建一个持久化的position_ids缓冲区，用于存储位置索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # 获取输入序列的长度
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        # 如果未提供位置索引，则使用预先定义的position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供嵌入向量，则使用token_embedding层将输入序列转换为嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        # 计算位置嵌入向量
        position_embeddings = self.position_embedding(position_ids)
        # 将词嵌入向量和位置嵌入向量相加得到最终的嵌入向量
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class BlipAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除（得到`embed_dim`：{self.embed_dim}和`num_heads`：{self.num_heads}）."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)

        # 创建一个线性层，用于计算查询、键、值的线性变换
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim)

        # 创建一个线性层，用于将多头注意力的输出映射回原始维度
        self.projection = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """定义了一个方法，用于计算自注意力机制的输出。
        
        输入形状: Batch x Time x Channel
        返回一个元组，包含输出张量、可选的注意力张量以及可选的注意力头张量元组。
        """

        # 获取隐藏状态张量的批量大小、目标长度和嵌入维度
        bsz, tgt_len, embed_dim = hidden_states.size()

        # 使用 QKV 线性层对隐藏状态进行线性变换，然后重塑成多头格式，并进行维度置换
        mixed_qkv = (
            self.qkv(hidden_states)
            .reshape(bsz, tgt_len, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query_states, key_states, value_states = mixed_qkv[0], mixed_qkv[1], mixed_qkv[2]

        # 计算原始注意力分数，即“查询”与“键”的点积
        attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        # 缩放注意力分数
        attention_scores = attention_scores * self.scale

        # 将注意力分数标准化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 进行 dropout 操作，随机丢弃部分 token 的注意力概率
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对注意力概率进行头部遮罩
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量，通过注意力概率与值张量的乘积并进行维度置换
        context_layer = torch.matmul(attention_probs, value_states).permute(0, 2, 1, 3)

        # 重塑上下文层的形状以匹配原始形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.embed_dim,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        # 通过投影层将上下文层映射到输出空间
        output = self.projection(context_layer)

        # 根据是否需要输出注意力权重，构建返回的输出元组
        outputs = (output, attention_probs) if output_attentions else (output, None)

        return outputs
# 从 transformers.models.clip.modeling_clip.CLIPMLP 复制并修改为 BlipMLP
class BlipMLP(nn.Module):
    def __init__(self, config):
        # 初始化 BlipMLP 类
        super().__init__()
        # 保存配置信息
        self.config = config
        # 获取激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 定义全连接层1，输入维度为隐藏尺寸，输出维度为中间尺寸
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # 定义全连接层2，输入维度为中间尺寸，输出维度为隐藏尺寸
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播
        # 使用全连接层1进行线性变换
        hidden_states = self.fc1(hidden_states)
        # 使用激活函数进行非线性变换
        hidden_states = self.activation_fn(hidden_states)
        # 使用全连接层2进行线性变换
        hidden_states = self.fc2(hidden_states)
        # 返回隐藏状态
        return hidden_states


class BlipEncoderLayer(nn.Module):
    def __init__(self, config: BlipConfig):
        # 初始化 BlipEncoderLayer 类
        super().__init__()
        # 获取嵌入维度
        self.embed_dim = config.hidden_size
        # 定义 BlipAttention 层
        self.self_attn = BlipAttention(config)
        # LayerNorm1 层，用于归一化
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 定义 BlipMLP 层
        self.mlp = BlipMLP(config)
        # LayerNorm2 层，用于归一化
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保留残差连接
        residual = hidden_states

        # 使用 LayerNorm1 进行归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 进行自注意力机制计算
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            head_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 添加残差连接
        hidden_states = hidden_states + residual
        # 更新残差
        residual = hidden_states
        # 使用 LayerNorm2 进行归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 使用 BlipMLP 进行前向传播
        hidden_states = self.mlp(hidden_states)

        # 添加残差连接
        hidden_states = hidden_states + residual

        outputs = (hidden_states,)

        if output_attentions:
            # 如果需要输出注意力权重，则加入输出中
            outputs += (attn_weights,)

        return outputs


class BlipPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # BlipPreTrainedModel 的配置类
    config_class = BlipConfig
    # base_model_prefix
    base_model_prefix = "blip"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化模型的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_range
        # 如果模块是卷积层、嵌入层或全连接层
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
            # 对权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=factor)
            # 如果模块有偏置项且不为None，则将偏置项初始化为0
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()

        # 如果模块是BlipVisionEmbeddings类型
        if isinstance(module, BlipVisionEmbeddings):
            # 如果配置中有视觉配置，则使用视觉配置的初始化因子
            if hasattr(self.config, "vision_config"):
                factor = self.config.vision_config.initializer_range
            # 对位置嵌入进行截断正态分布初始化
            nn.init.trunc_normal_(
                module.position_embedding,
                mean=0.0,
                std=factor,
            )

            # 对类别嵌入进行截断正态分布初始化
            nn.init.trunc_normal_(
                module.class_embedding,
                mean=0.0,
                std=factor,
            )

        # 如果模块是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
        # 如果模块是全连接层且有偏置项，则将偏置项初始化为0
        elif isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
# BLIP_START_DOCSTRING 是一个包含模型说明文档的字符串，描述了该模型的继承关系和参数信息
BLIP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BlipConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# BLIP_TEXT_INPUTS_DOCSTRING 是一个包含文本输入参数说明的字符串，描述了输入参数的类型和作用
BLIP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoProcessor`]. See [`BlipProcessor.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# BLIP_VISION_INPUTS_DOCSTRING 是一个空字符串，可能用于描述视觉输入参数的说明
BLIP_VISION_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。默认情况下，如果您提供了填充(padding)，将会忽略它。可以使用[`BlipImageProcessor`]获取像素值。有关详细信息，请参阅[`BlipImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的`attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量下的`hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通的元组。
"""
定义了 BLIP_INPUTS_DOCSTRING，包含了对输入参数的详细说明
"""

class BlipEncoder(nn.Module):
    """
    定义了 BlipEncoder 类，是由 config.num_hidden_layers 个自注意力层组成的 Transformer 编码器

    Args:
        config (`BlipConfig`):
            BlipEncoder 对应的视觉配置
    """

    def __init__(self, config: BlipConfig):
        # 初始化函数，接受 BlipConfig 类型的 config 参数
        super().__init__()
        # 调用父类的初始化函数
        self.config = config
        # 将传入的 config 参数保存到实例变量中
        self.layers = nn.ModuleList([BlipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建包含 config.num_hidden_layers 个 BlipEncoderLayer 实例的 ModuleList
        self.gradient_checkpointing = False
        # 初始化梯度检查点为 False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # 设置输出的注意力张量是否包含所有注意力层的信息
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出的隐藏状态是否包含所有层的信息
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回一个 ModelOutput 对象而不是一个普通元组
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果不输出隐藏状态，则初始化为空
        encoder_states = () if output_hidden_states else None
        # 如果不输出注意力，则初始化为空
        all_attentions = () if output_attentions else None

        # 将输入的嵌入表示设置为隐藏状态
        hidden_states = inputs_embeds
        # 遍历每个编码层
        for idx, encoder_layer in enumerate(self.layers):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 encoder_states 中
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # 如果启用渐变检查点且处于训练模式，则使用渐变检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用编码层
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=output_attentions,
                )

            # 更新隐藏状态为编码层的输出
            hidden_states = layer_outputs[0]

            # 如果输出注意力，则将当前层的注意力添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到 encoder_states 中
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # 如果不返回字典，则返回包含隐藏状态、编码状态和注意力的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        # 否则返回一个 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
class BlipVisionModel(BlipPreTrainedModel):
    # 主要输入名称为像素值
    main_input_name = "pixel_values"
    # 配置类为BlipVisionConfig
    config_class = BlipVisionConfig

    # 初始化函数，接受BlipVisionConfig类型的参数config
    def __init__(self, config: BlipVisionConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将参数config保存到实例变量中
        self.config = config
        # 嵌入维度为配置中的隐藏大小
        embed_dim = config.hidden_size

        # 初始化嵌入层对象
        self.embeddings = BlipVisionEmbeddings(config)
        # 初始化编码器对象
        self.encoder = BlipEncoder(config)
        # 初始化后层归一化对象，输入维度为embed_dim，eps为配置中的layer_norm_eps
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        # 调用后初始化函数
        self.post_init()

    # 前向传播函数，接受一系列输入参数，返回一个包含隐藏状态和池化输出的元组或BaseModelOutputWithPooling对象
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=BlipVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        # 如果output_attentions为None，则使用配置中的output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states为None，则使用配置中的output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict为None，则使用配置中的use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为None，则引发值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传递给嵌入层，并获取隐藏状态
        hidden_states = self.embeddings(pixel_values)

        # 将隐藏状态传递给编码器，并获取编码器的输出
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的最后隐藏状态，并对其进行后层归一化
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # 从最后隐藏状态中提取池化输出
        pooled_output = last_hidden_state[:, 0, :]
        # 对池化输出进行后层归一化
        pooled_output = self.post_layernorm(pooled_output)

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则返回BaseModelOutputWithPooling对象
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.embeddings


# 添加起始文档字符串的BlipModel类
@add_start_docstrings(BLIP_START_DOCSTRING)
class BlipModel(BlipPreTrainedModel):
    # 配置类为BlipConfig
    config_class = BlipConfig
    # 初始化方法，接受一个 BlipConfig 对象作为参数
    def __init__(self, config: BlipConfig):
        # 调用父类的初始化方法，传入 BlipConfig 对象
        super().__init__(config)

        # 检查 config.text_config 是否为 BlipTextConfig 类型，如果不是则抛出 ValueError 异常
        if not isinstance(config.text_config, BlipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type BlipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.vision_config 是否为 BlipVisionConfig 类型，如果不是则抛出 ValueError 异常
        if not isinstance(config.vision_config, BlipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type BlipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        # 将 config.text_config 和 config.vision_config 分别赋值给对应的变量
        text_config = config.text_config
        vision_config = config.vision_config

        # 将 projection_dim、text_embed_dim 和 vision_embed_dim 分别赋值给对应的属性
        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        # 创建 BlipTextModel 和 BlipVisionModel 对象，并赋值给对应的属性
        self.text_model = BlipTextModel(text_config)
        self.vision_model = BlipVisionModel(vision_config)

        # 创建线性层，用于将视觉嵌入维度映射到 projection_dim，不使用偏置项
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        # 创建线性层，用于将文本嵌入维度映射到 projection_dim，不使用偏置项
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        # 创建可学习参数 logit_scale，其初始值为 config.logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    # 对模型的前向传播方法添加文档字符串，该方法用于获取文本特征
    @add_start_docstrings_to_model_forward(BLIP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`BlipTextModel`].

        Examples:

        ```py
        >>> from transformers import AutoProcessor, BlipModel

        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 BlipTextModel 的前向传播方法，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=return_dict,
        )

        # 获取文本池化后的输出
        pooled_output = text_outputs[1]
        # 将文本池化后的输出映射到 projection_dim 维度
        text_features = self.text_projection(pooled_output)

        # 返回文本特征
        return text_features

    # 对模型的前向传播方法添加文档字符串，用于获取视觉特征
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,  # 定义函数，获取图片特征
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
    ) -> torch.FloatTensor:  # 返回类型为torch.FloatTensor的张量
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`BlipVisionModel`].

        Examples:

        ```py
        >>> from PIL import Image  # 导入Image类
        >>> import requests  # 导入requests模块
        >>> from transformers import AutoProcessor, BlipModel  # 导入AutoProcessor和BlipModel类

        >>> model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")  # 从预训练模型创建BlipModel对象
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # 从预训练模型创建AutoProcessor对象

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 图片URL
        >>> image = Image.open(requests.get(url, stream=True).raw)  # 从URL打开图像，并转换为PIL.Image对象

        >>> inputs = processor(images=image, return_tensors="pt")  # 处理图像并返回PyTorch张量

        >>> image_features = model.get_image_features(**inputs)  # 获得图像特征
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict不为None，则使用return_dict；否则使用self.config.use_return_dict

        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=return_dict)  # 使用视觉模型获得输出

        pooled_output = vision_outputs[1]  # pooled_output，池化输出
        image_features = self.visual_projection(pooled_output)  # 使用可视化投影层对池化输出进行投影

        return image_features  # 返回图像特征

    @add_start_docstrings_to_model_forward(BLIP_INPUTS_DOCSTRING)  # 添加模型前向传播的文档字符串
    @replace_return_docstrings(output_type=BlipOutput, config_class=BlipConfig)  # 替换返回的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID序列
        pixel_values: Optional[torch.FloatTensor] = None,  # 图像像素值
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID
        return_loss: Optional[bool] = None,  # 是否返回损失
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
```py  
# 添加模型文档字符串，描述 BLIP 模型用于图像字幕生成，包含视觉编码器和文本解码器，可以传入 input_ids 作为文本提示，让文本解码器继续提示，否则从 [BOS] 标记开始生成文本
@add_start_docstrings(
    """
    BLIP Model for image captioning. The model consists of a vision encoder and a text decoder. One can optionally pass
    `input_ids` to the model, which serve as a text prompt, to make the text decoder continue the prompt. Otherwise,
    the decoder starts generating text from the [BOS] (beginning-of-sequence) token. will start generating the caption
    from the text input. If no text input is provided, the decoder will start with the [BOS] token only.
    """,
    BLIP_START_DOCSTRING,
)
# 定义 BlipForConditionalGeneration 类，继承自 BlipPreTrainedModel
class BlipForConditionalGeneration(BlipPreTrainedModel):
    # 指定配置类为 BlipConfig
    config_class = BlipConfig
    # 定义权重绑定的键
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化方法，接受 BlipConfig 类型的参数 config
    def __init__(self, config: BlipConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建视觉模型对象
        self.vision_model = BlipVisionModel(config.vision_config)

        # 创建文本解码器对象
        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        # 设置解码器的输入标记为 bos_token_id
        self.decoder_input_ids = config.text_config.bos_token_id
        # 设置解码器的填充标记为 pad_token_id
        self.decoder_pad_token_id = config.text_config.pad_token_id

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipForConditionalGenerationModelOutput, config_class=BlipVisionConfig)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BlipForConditionalGenerationModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A picture of"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> outputs = model(**inputs)
        ```py"""

        # 设置返回字典，如果未指定则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置是否输出注意力权重，如果未指定则使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态，如果未指定则使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 使用视觉模型处理像素值，获取图像嵌入
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]

        # 使用文本解码器生成输出
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        # 如果不返回字典，则返回指定的输出
        if not return_dict:
            outputs = (outputs[0], outputs[1], image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 返回生成的模型输出
        return BlipForConditionalGenerationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            pixel_values (*torch.FloatTensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForConditionalGeneration

        >>> model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        two cats sleeping on a couch
        ```py
        """

        # 获取输入图片的批量大小
        batch_size = pixel_values.shape[0]
        # 使用视觉模型处理输入图片
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        # 从视觉输出中提取图像嵌入
        image_embeds = vision_outputs[0]

        # 创建图像注意力遮罩，全为1，形状与图像嵌入维度一致
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # 将输入的序列转换为LongTensor类型，若为None，则构造起始标记序列
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        elif input_ids is None:
            input_ids = (
                torch.LongTensor([[self.decoder_input_ids, self.config.text_config.eos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )

        # 在输入序列的开头添加起始标记
        input_ids[:, 0] = self.config.text_config.bos_token_id
        # 若存在attention_mask，则移除最后一个标记的attention_mask
        attention_mask = attention_mask[:, :-1] if attention_mask is not None else None

        # 使用文本解码器生成序列
        outputs = self.text_decoder.generate(
            input_ids=input_ids[:, :-1],
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            **generate_kwargs,
        )

        # 返回生成的序列
        return outputs
# BLIP 模型用于视觉问答。该模型由一个视觉编码器、一个文本编码器以及一个文本解码器组成。
# 视觉编码器将编码输入图像，文本编码器将编码输入问题以及图像的编码，文本解码器将输出问题的答案。
@add_start_docstrings(
    """
    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text
    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together
    with the encoding of the image, and the text decoder will output the answer to the question.
    """,
    BLIP_START_DOCSTRING,
)
class BlipForQuestionAnswering(BlipPreTrainedModel):
    # BLIP 模型的配置类
    config_class = BlipConfig
    # 要绑定权重的键列表
    _tied_weights_keys = ["text_decoder.cls.predictions.decoder.bias"]

    def __init__(self, config: BlipConfig):
        # 调用父类初始化方法
        super().__init__(config)

        # 初始化视觉模型
        self.vision_model = BlipVisionModel(config.vision_config)

        # 初始化文本编码器
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # 初始化文本解码器
        self.text_decoder = BlipTextLMHeadModel(config.text_config)

        # 解码器的填充标记 ID
        self.decoder_pad_token_id = config.text_config.pad_token_id
        # 解码器的起始标记 ID
        self.decoder_start_token_id = config.text_config.bos_token_id

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):
    # 生成方法，不计算梯度
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        r"""
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*):
                The sequence used as a prompt for the generation.
            pixel_values (*torch.FloatTensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            attention_mask (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            **generate_kwargs:
                Additional arguments passed to the *generate* function of the decoder


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForQuestionAnswering

        >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```py
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)

        question_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )

        question_embeds = question_outputs[0]

        question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(question_embeds.device)

        bos_ids = torch.full(
            (question_embeds.size(0), 1), fill_value=self.decoder_start_token_id, device=question_embeds.device
        )

        outputs = self.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.config.text_config.sep_token_id,
            pad_token_id=self.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask,
            **generate_kwargs,
        )

        return outputs
# 使用 add_start_docstrings 装饰器为模型添加文档字符串，描述了 BLIP 模型在图像-文本检索任务中的作用和输入输出
@add_start_docstrings(
    """
    BLIP 模型具有视觉和文本投影器以及顶部的分类头。该模型用于图像-文本检索任务。
    给定一张图像和一段文本，模型返回文本与图像相关的概率。
    """,
    BLIP_START_DOCSTRING,  # 导入的 BLIP_START_DOCSTRING
)
# 定义 BlipForImageTextRetrieval 类，继承自 BlipPreTrainedModel
class BlipForImageTextRetrieval(BlipPreTrainedModel):
    # 配置类为 BlipConfig
    config_class = BlipConfig

    # 初始化方法
    def __init__(self, config: BlipConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化视觉模型
        self.vision_model = BlipVisionModel(config.vision_config)

        # 初始化文本编码器
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)

        # 视觉投影层
        self.vision_proj = nn.Linear(config.vision_config.hidden_size, config.image_text_hidden_size)

        # 文本投影层
        self.text_proj = nn.Linear(config.text_config.hidden_size, config.image_text_hidden_size)

        # 图像文本匹配头
        self.itm_head = nn.Linear(config.text_config.hidden_size, 2)

        # 解码器填充标记 ID
        self.decoder_pad_token_id = (
            config.text_config.pad_token_id
            if not hasattr(config, "decoder_pad_token_id")
            else config.decoder_pad_token_id
        )

        # 解码器起始标记 ID
        self.decoder_start_token_id = (
            config.text_config.bos_token_id
            if not hasattr(config, "decoder_start_token_id")
            else config.decoder_start_token_id
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    # 前向传播方法
    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)  # 添加输入文档字符串
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)  # 替换返回文档字符串
    def forward(
        self,
        input_ids: torch.LongTensor,  # 输入文本的 token IDs
        pixel_values: torch.FloatTensor,  # 输入图像的像素值
        use_itm_head: Optional[bool] = True,  # 是否使用图像-文本匹配头，默认为 True
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式
```  
        ) -> Union[Tuple, BlipTextVisionModelOutput]:
        r"""
        Returns:

        Examples:

        ```py
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForImageTextRetrieval

        >>> model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "an image of a cat"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```
        """
        # 设置返回值为 return_dict 或者 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置输出注意力值为 output_attentions 或者 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态为 output_hidden_states 或者 self.config.output_hidden_states

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取图像嵌入
        image_embeds = vision_outputs[0]
        # 创建图像注意力张量
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        if use_itm_head:
            # 如果使用 itm_head，则计算问题嵌入
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            # 使用 itm_head 计算输出
            output = self.itm_head(question_embeds[:, 0, :])
        else:
            # 如果不使用 itm_head，则计算问题嵌入
            question_embeds = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
            )
            question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state

            # 标准化图像特征和文本特征，计算输出
            image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)

            output = image_feat @ text_feat.t()

        if not return_dict:
            # 如果不返回字典，则返回输出和其他相关值
            outputs = (output, vision_outputs[0]) + vision_outputs[2:] + (question_embeds,)
            return tuple(output for output in outputs if output is not None)

        # 返回 BlipImageTextMatchingModelOutput 对象
        return BlipImageTextMatchingModelOutput(
            itm_score=output,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
            question_embeds=question_embeds,
        )
```