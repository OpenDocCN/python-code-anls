# `.\models\vivit\modeling_vivit.py`

```py
# 定义 VivitTubeletEmbeddings 类，用于构建 Vivit 模型的 Tubelet embeddings
class VivitTubeletEmbeddings(nn.Module):
    """
    Construct Vivit Tubelet embeddings.

    This module turns a batch of videos of shape (batch_size, num_frames, num_channels, height, width) into a tensor of
    shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size[0]) * (height // tubelet_size[1]) *
    (width // tubelet_size[2]).
    """

    def __init__(self, config):
        super().__init__()
        # 初始化 Tubelet embeddings 相关参数
        self.num_frames = config.num_frames  # 视频帧数
        self.image_size = config.image_size  # 视频帧的尺寸
        self.patch_size = config.tubelet_size  # Tubelet 的尺寸
        # 计算 patches 的数量，用于 Transformer 编码器的输入长度
        self.num_patches = (
            (self.image_size // self.patch_size[2])
            * (self.image_size // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )
        self.embed_dim = config.hidden_size  # 嵌入向量的维度

        # 使用 3D 卷积层将视频帧转换为嵌入向量
        self.projection = nn.Conv3d(
            config.num_channels,  # 输入视频的通道数
            config.hidden_size,   # 输出嵌入向量的维度
            kernel_size=config.tubelet_size,  # 卷积核大小，即 Tubelet 的尺寸
            stride=config.tubelet_size  # 卷积的步长，与 Tubelet 尺寸相同
        )
    # 定义前向传播方法，接受像素值作为输入
    def forward(self, pixel_values):
        # 获取输入张量的批大小、帧数、通道数、高度和宽度
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 检查输入图像的高度和宽度是否与模型要求的大小相匹配
        if height != self.image_size or width != self.image_size:
            # 如果不匹配，抛出数值错误异常
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )

        # 将输入像素值重新排列为 (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        # 使用投影层处理重新排列后的像素值
        x = self.projection(pixel_values)
        # 对处理后的输出进行扁平化，并转置维度，以便得到期望的形状
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回处理后的张量作为前向传播的输出
        return x
# 定义 VivitEmbeddings 类，继承自 nn.Module，用于视频数据的嵌入处理
class VivitEmbeddings(nn.Module):
    """
    Vivit Embeddings.

    Creates embeddings from a video using VivitTubeletEmbeddings, adds CLS token and positional embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # 初始化一个可学习的 CLS token 参数，形状为 [1, 1, hidden_size]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # 使用 VivitTubeletEmbeddings 类生成视频帧的嵌入表示
        self.patch_embeddings = VivitTubeletEmbeddings(config)

        # 初始化位置嵌入，形状为 [1, num_patches + 1, hidden_size]
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )
        
        # 定义一个 dropout 层，用于随机置零输入张量的部分元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 存储配置信息
        self.config = config

    def forward(self, pixel_values):
        # 获取输入张量的 batch size
        batch_size = pixel_values.shape[0]
        
        # 生成视频帧的嵌入表示
        embeddings = self.patch_embeddings(pixel_values)

        # 复制并添加 CLS token 到嵌入表示中，维度变为 [batch_size, num_patches + 1, hidden_size]
        cls_tokens = self.cls_token.repeat([batch_size, 1, 1])
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 添加位置编码到每个 token 上
        embeddings = embeddings + self.position_embeddings

        # 对嵌入表示进行 dropout 处理
        embeddings = self.dropout(embeddings)

        return embeddings


# 从 transformers.models.vit.modeling_vit.ViTSelfAttention 复制并修改为 VivitSelfAttention
class VivitSelfAttention(nn.Module):
    def __init__(self, config: VivitConfig) -> None:
        super().__init__()
        
        # 检查 hidden_size 是否能被 num_attention_heads 整除，否则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头的数量和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义 dropout 层，用于注意力概率的随机置零
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量重塑为注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 定义自注意力层的前向传播函数
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ):
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用 self.query 对隐藏状态进行查询，得到混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 对隐藏状态进行键的转换，并调整维度以备点积计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        
        # 使用 self.value 对隐藏状态进行值的转换，并调整维度以备点积计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 使用混合查询层和键层的转置进行点积操作，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 根据注意力头的大小对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行归一化得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 对注意力概率进行随机丢弃处理
        attention_probs = self.dropout(attention_probs)

        # 如果有头部遮罩，则将注意力概率与头部遮罩相乘
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 将注意力概率与值层进行加权求和，得到上下文层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文层的维度顺序，并确保连续的存储
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 将调整后的上下文层重塑为指定的形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 返回上下文层和可能的注意力概率，如果需要输出注意力信息
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从 transformers.models.vit.modeling_vit.ViTSelfOutput 复制而来，进行了 ViT -> Vivit 的改名
class VivitSelfOutput(nn.Module):
    """
    The residual connection is defined in VivitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VivitConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出的维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，以 config.hidden_dropout_prob 的概率随机置零输入张量的元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用全连接层 self.dense
        hidden_states = self.dense(hidden_states)
        # 对应用全连接层后的 hidden_states 应用 dropout 层
        hidden_states = self.dropout(hidden_states)

        # 返回处理后的 hidden_states
        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTAttention 复制而来，进行了 ViT -> Vivit 的改名
class VivitAttention(nn.Module):
    def __init__(self, config: VivitConfig) -> None:
        super().__init__()
        # 实例化 VivitSelfAttention 类，传入配置 config，并赋值给 self.attention
        self.attention = VivitSelfAttention(config)
        # 实例化 VivitSelfOutput 类，传入配置 config，并赋值给 self.output
        self.output = VivitSelfOutput(config)
        # 初始化一个空集合，用于存储要剪枝的注意力头部
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到要剪枝的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用 self.attention 的 forward 方法，传入 hidden_states, head_mask 和 output_attentions
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将 self_outputs[0] 和 hidden_states 作为输入，调用 self.output 的 forward 方法
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力，则将 attentions 添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class VivitIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 定义一个 dropout 层，以 config.hidden_dropout_prob 的概率随机置零输入张量的元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果 config.hidden_act 是字符串类型，则选择相应的激活函数；否则直接使用 config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数对线性变换后的结果进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对非线性变换后的结果应用dropout操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)

        # 返回处理后的隐藏状态作为输出
        return hidden_states
class VivitOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入尺寸为config中的中间层大小，输出尺寸为config中的隐藏层大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个dropout层，使用config中的隐藏层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 对输入的hidden_states进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的hidden_states进行dropout
        hidden_states = self.dropout(hidden_states)
        # 将dropout后的hidden_states与输入的input_tensor相加作为最终输出
        hidden_states = hidden_states + input_tensor
        return hidden_states


class VivitLayer(nn.Module):
    """This corresponds to the EncoderBlock class in the scenic/vivit implementation."""

    def __init__(self, config):
        super().__init__()
        # 定义VivitLayer的属性，用于分块的前馈传播chunk_size_feed_forward，序列长度维度seq_len_dim为1
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 定义VivitLayer中的注意力层、中间层、输出层，分别使用给定的config初始化
        self.attention = VivitAttention(config)
        self.intermediate = VivitIntermediate(config)
        self.output = VivitOutput(config)
        # 在self-attention之前应用层归一化
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 在self-attention之后应用层归一化
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        # 将输入的hidden_states和head_mask传递给self.attention，获取self-attention的输出
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在self-attention之前应用层归一化
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        # 如果需要输出注意力权重，则将其添加到outputs中
        outputs = self_attention_outputs[1:]

        # 第一个残差连接，将self-attention的输出加到原始的hidden_states上
        hidden_states = attention_output + hidden_states

        # 在self-attention之后应用层归一化
        layer_output = self.layernorm_after(hidden_states)
        # 将归一化后的输出传递给中间层进行处理
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接，在输出层中应用处理后的layer_output，并与原始的hidden_states相加
        layer_output = self.output(layer_output, hidden_states)

        # 将最终的输出组装成outputs
        outputs = (layer_output,) + outputs

        return outputs


class VivitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个由VivitLayer实例组成的列表，长度为config中指定的隐藏层数量num_hidden_layers
        self.layer = nn.ModuleList([VivitLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点标志设置为False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则初始化一个空元组，否则置为None
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 根据头部掩码是否存在，决定是否应用头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点且处于训练模式，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用层模块进行前向传播
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则添加当前层输出的注意力权重到元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不使用返回字典格式，则返回一个元组，过滤掉None值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 使用BaseModelOutput格式返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        """
        # VivitPooler 类，用于汇集模型隐藏状态的第一个令牌的隐藏状态
        An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
        models.
        """

    config_class = VivitConfig
    base_model_prefix = "vivit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """
        # 初始化模型权重
        Initialize the weights
        """
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # 略有不同于 TF 版本，使用正态分布初始化权重
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`VivitImageProcessor`]. See
            [`VivitImageProcessor.preprocess`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare ViViT Transformer model outputting raw hidden-states without any specific head on top.",
    VIVIT_START_DOCSTRING,
)
"""
class VivitModel(VivitPreTrainedModel):
    """
    ViViT Transformer model for raw hidden-states.

    Args:
        config: ViViT model configuration instance.
        add_pooling_layer: Whether to add a pooling layer on top of the encoder.

    Attributes:
        embeddings: ViViT embeddings module.
        encoder: ViViT encoder module.
        layernorm: Layer normalization module.
        pooler: Optional pooling layer for final representation.

    Methods:
        get_input_embeddings(): Retrieve the patch embeddings from the model.
        _prune_heads(heads_to_prune): Prune attention heads in the model.
        forward(...): Forward pass of the ViViT model.

    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # Initialize ViViT components
        self.embeddings = VivitEmbeddings(config)
        self.encoder = VivitEncoder(config)

        # Layer normalization and optional pooling layer
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = VivitPooler(config) if add_pooling_layer else None

        # Initialize weights and final processing steps
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieve the patch embeddings from the ViViT model.

        Returns:
            embeddings: Patch embeddings used by the model.
        """
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes specific attention heads in the ViViT model.

        Args:
            heads_to_prune (dict): Dictionary mapping layer numbers to lists of heads to prune.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the ViViT model.

        Args:
            pixel_values: Pixel values of the input video frames.
            head_mask: Mask to exclude certain attention heads.
            output_attentions: Whether to output attentions weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.

        Returns:
            BaseModelOutputWithPooling or torch.Tensor: Model outputs.
        """
        # Forward pass logic goes here
        pass

"""
@add_start_docstrings(
    """ViViT Transformer model with a video classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for Kinetics-400.""",
    VIVIT_START_DOCSTRING,
)
"""
class VivitForVideoClassification(VivitPreTrainedModel):
    """
    ViViT Transformer model with a video classification head.

    Args:
        config: ViViT model configuration instance.

    Attributes:
        num_labels: Number of classification labels.
        vivit: ViViT base model.
        classifier: Linear classification layer.

    Methods:
        forward(...): Forward pass of the model for video classification.

    """
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vivit = VivitModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and final processing steps
        self.post_init()

    @add_start_docstrings_to_model_forward(VIVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass of the ViViT model for video classification.

        Args:
            pixel_values: Pixel values of the input video frames.
            head_mask: Mask to exclude certain attention heads.
            labels: Labels for classification.
            output_attentions: Whether to output attentions weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dictionary.

        Returns:
            ImageClassifierOutput or torch.Tensor: Model outputs.
        """
        # Forward pass logic goes here
        pass
```