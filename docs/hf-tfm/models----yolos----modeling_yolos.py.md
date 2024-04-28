# `.\transformers\models\yolos\modeling_yolos.py`

```
# 定义 YolosObjectDetectionOutput 类，继承自 ModelOutput，用于 YOLOS 检测模型的输出类型
@dataclass
class YolosObjectDetectionOutput(ModelOutput):
    """
    Output type of [`YolosForObjectDetection`].
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~YolosImageProcessor.post_process`] to retrieve the unnormalized bounding
            boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """
    
    # 定义函数的参数类型和可能的返回值类型
    loss: Optional[torch.FloatTensor] = None  # 总损失，如果提供了标签，则返回
    loss_dict: Optional[Dict] = None  # 包含个体损失的字典，用于记录
    logits: torch.FloatTensor = None  # 所有查询的分类对数（包括无对象）
    pred_boxes: torch.FloatTensor = None  # 所有查询的标准化框坐标，表示为（中心_x，中心_y，宽度，高度）
    auxiliary_outputs: Optional[List[Dict]] = None  # 辅助输出，仅当使用辅助损失并提供标签时才返回
    last_hidden_state: Optional[torch.FloatTensor] = None  # 模型解码器最后一层的隐藏状态序列
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型在每层输出的隐藏状态和可选的初始嵌入输出的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 在注意力 softmax 之后的注意力权重，用于计算自注意头中的加权平均值
class YolosEmbeddings(nn.Module):
    """
    构建 CLS token、检测 tokens、位置和路径嵌入。

    """

    def __init__(self, config: YolosConfig) -> None:
        super().__init__()

        # 初始化 CLS token 参数为全零的张量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 初始化检测 tokens 参数为全零的张量
        self.detection_tokens = nn.Parameter(torch.zeros(1, config.num_detection_tokens, config.hidden_size))
        # 初始化路径嵌入
        self.patch_embeddings = YolosPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        # 初始化位置嵌入参数为全零的张量
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + config.num_detection_tokens + 1, config.hidden_size)
        )
        
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化插值层
        self.interpolation = InterpolateInitialPositionEmbeddings(config)
        # 保存配置
        self.config = config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        # 获取图片中的嵌入
        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.size()

        # 将 [CLS] 和检测 tokens 添加到嵌入的路径 tokens 中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        detection_tokens = self.detection_tokens.expand(batch_size, -1, -1)
        # 拼接嵌入的路径 tokens
        embeddings = torch.cat((cls_tokens, embeddings, detection_tokens), dim=1)

        # 为每个 tokens 添加位置编码
        # 这可能需要对现有位置嵌入进行插值
        position_embeddings = self.interpolation(self.position_embeddings, (height, width))

        # 将位置编码添加到嵌入中
        embeddings = embeddings + position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class InterpolateInitialPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
    # 前向传播函数，接受位置编码和图像大小作为输入，并返回一个张量
    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        # 提取类别位置编码，保留批次维度
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        # 提取检测位置编码，保留批次维度
        det_pos_embed = pos_embed[:, -self.config.num_detection_tokens :, :]
        # 提取补丁位置编码，去除类别和检测位置编码，然后转置维度
        patch_pos_embed = pos_embed[:, 1 : -self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        # 获取批次大小、隐藏大小和序列长度
        batch_size, hidden_size, seq_len = patch_pos_embed.shape

        # 计算补丁高度和宽度
        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        # 重新形状补丁位置编码
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, patch_height, patch_width)

        # 获取图像高度和宽度
        height, width = img_size
        # 计算新的补丁高度和宽度
        new_patch_heigth, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        # 插值操作，将补丁位置编码调整为新的尺寸
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_heigth, new_patch_width), mode="bicubic", align_corners=False
        )
        # 展平补丁位置编码，并转置维度
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        # 合并所有位置编码
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        # 返回合并后的位置编码
        return scale_pos_embed
# YolosPatchEmbeddings 类将形状为 (batch_size, num_channels, height, width) 的像素值转换为形状为
# (batch_size, seq_length, hidden_size) 的初始隐藏状态（分块嵌入），以供 Transformer 使用。

class YolosPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        # 初始化函数，接收配置参数
        super().__init__()
        # 从配置参数中获取图像大小、分块大小、通道数和隐藏层大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小和分块大小不是可迭代对象，则将它们转换为元组形式
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的分块数
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 通过卷积操作将输入的像素值转换为隐藏状态的尺寸
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    # 前向传播函数，接受像素数值作为输入，返回嵌入向量
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查通道数是否与模型配置中设置的通道数匹配
        if num_channels != self.num_channels:
            # 若通道数不匹配，则引发 ValueError 异常
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 将输入像素值投影到嵌入空间，并展平成二维张量，然后进行维度转置
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回嵌入向量
        return embeddings
# 从transformers.models.vit.modeling_vit.ViTSelfAttention复制代码，将ViT改成Yolos
class YolosSelfAttention(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        如果隐藏层大小不能被注意力头的数量整除，且配置中没有embedding_size属性，则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        初始化注意力头的数量和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    将输入张量转换为注意力分数张量的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    前向传播函数
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        混合查询层
        mixed_query_layer = self.query(hidden_states)

        转置键层和值层的形状以便计算注意力分数
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算"查询"和"键"之间的点积，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        将注意力分数除以数学上的注意力头大小的平方根
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        实际上这里是对整个token进行dropout，看起来有点不寻常，但是源自原始Transformer论文
        attention_probs = self.dropout(attention_probs)

        如果需要，对注意力概率进行头部遮罩
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        计算上下文张量
        context_layer = torch.matmul(attention_probs, value_layer)

        调整上下文张量的形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        如果需要输出注意力信息，则返回上下文张量和注意力概率，否则只返回上下文张量
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        返回结果
        return outputs
# 从 transformers.models.vit.modeling_vit.ViTSelfOutput 复制并将 ViT 改为 Yolos
class YolosSelfOutput(nn.Module):
    """
    The residual connection is defined in YolosLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    # YolosSelfOutput 类的初始化函数
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个以 config.hidden_dropout_prob 为参数的 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # YolosSelfOutput 类的前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 处理 hidden_states
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTAttention 复制并将 ViT 改为 Yolos
class YolosAttention(nn.Module):
    # YolosAttention 类的初始化函数
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 创建一个 YolosSelfAttention 实例
        self.attention = YolosSelfAttention(config)
        # 创建一个 YolosSelfOutput 实例
        self.output = YolosSelfOutput(config)
        # 创建一个空集合 pruned_heads
        self.pruned_heads = set()

    # YolosAttention 类的 prune_heads 方法
    def prune_heads(self, heads: Set[int]) -> None:
        # 如果输入头部集合为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可剪枝的头部索引
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

    # YolosAttention 类的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 对 hidden_states 进行自注意力机制处理
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 使用 YolosSelfOutput 处理自注意力输出和原始 hidden_states
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出 注意力权重，则添加到 outputs 中
        return outputs


# 从 transformers.models.vit.modeling_vit.ViTIntermediate 复制并将 ViT 改为 Yolos
class YolosIntermediate(nn.Module):
    # YolosIntermediate 类的初始化函数
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入维度是 config.hidden_size，输出维度是 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串类型，则使用 ACT2FN 中的函数，否则使用 config.hidden_act 作为激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，接收隐藏状态张量并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理线性变换后的隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态张量
        return hidden_states
# 定义一个 PyTorch 模块 YolosOutput，继承自 nn.Module
# 该模块负责处理 Yolos 模型输出的特征
class YolosOutput(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 创建一个线性层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 dropout 层，dropout 概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 通过线性层和 dropout 层
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # 将处理后的 hidden_states 与原始的 input_tensor 相加，得到最终的输出
        hidden_states = hidden_states + input_tensor
        return hidden_states


# 定义一个 PyTorch 模块 YolosLayer，继承自 nn.Module
# 该模块对应 timm 实现中的 Block 类
class YolosLayer(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 设置 chunk_size_feed_forward 和 seq_len_dim
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建 YolosAttention、YolosIntermediate 和 YolosOutput 模块
        self.attention = YolosAttention(config)
        self.intermediate = YolosIntermediate(config)
        self.output = YolosOutput(config)
        # 创建两个 LayerNorm 层，分别应用于注意力计算前后
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 先对 hidden_states 进行 LayerNorm
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则将其添加到输出中

        # 进行第一个残差连接
        hidden_states = attention_output + hidden_states

        # 对 hidden_states 进行第二次 LayerNorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 进行第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# 定义 PyTorch 模块 YolosEncoder
class YolosEncoder(nn.Module):
    # 该模块实现 Yolos 编码器的功能
    pass
    # 初始化函数，接受一个 YolosConfig 对象作为参数
    def __init__(self, config: YolosConfig) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置对象
        self.config = config
        # 创建包含多个 YolosLayer 对象的 ModuleList
        self.layer = nn.ModuleList([YolosLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

        # 根据配置计算序列长度
        seq_length = (
            1 + (config.image_size[0] * config.image_size[1] // config.patch_size**2) + config.num_detection_tokens
        )
        # 如果配置中使用中间位置嵌入，则创建中间位置嵌入参数
        self.mid_position_embeddings = (
            nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers - 1,
                    1,
                    seq_length,
                    config.hidden_size,
                )
            )
            if config.use_mid_position_embeddings
            else None
        )

        # 如果配置中使用中间位置嵌入，则创建 InterpolateMidPositionEmbeddings 对象
        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        height,
        width,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 初始化隐藏状态和注意力矩阵的输出
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 如果配置中使用中间位置嵌入，则进行插值计算
        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))

        # 遍历每个 YolosLayer，并进行前向传播计算
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则保存当前的隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果梯度检查点为开启且处于训练模式，则使用梯度检查点函数进行前向传播计算
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态
            hidden_states = layer_outputs[0]

            # 如果配置中使用中间位置嵌入且不是最后一层，则加上中间位置嵌入
            if self.config.use_mid_position_embeddings:
                if i < (self.config.num_hidden_layers - 1):
                    hidden_states = hidden_states + interpolated_mid_position_embeddings[i]

            # 如果输出注意力矩阵，则保存当前的注意力矩阵
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则保存当前的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出，则将当前的结果以元组的形式返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回字典形式的结果
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义一个 YolosPreTrainedModel 类，继承自 PreTrainedModel
class YolosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 YolosConfig 类为配置类
    config_class = YolosConfig
    # 设置基础模型前缀为 "vit"
    base_model_prefix = "vit"
    # 设置主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重函数
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果模块是 Linear 或 Conv2d 类型
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 初始化权重为正态分布，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为 0
            module.bias.data.zero_()
            # 初始化权重为 1
            module.weight.data.fill_(1.0)


# YolosModel 类的文档字符串
YOLOS_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`YolosConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# YolosModel 类输入的文档字符串
YOLOS_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`YolosImageProcessor.__call__`] for details.

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

# 添加 YOLOS Model 的文档字符串
@add_start_docstrings(
    "The bare YOLOS Model transformer outputting raw hidden-states without any specific head on top.",
    YOLOS_START_DOCSTRING,
)
# YolosModel 类继承自 YolosPreTrainedModel
class YolosModel(YolosPreTrainedModel):
    # 初始化方法，接受config对象和一个布尔值参数add_pooling_layer，默认为True
    def __init__(self, config: YolosConfig, add_pooling_layer: bool = True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置实例的config属性为传入的config对象
        self.config = config
    
        # 创建YolosEmbeddings对象并赋值给实例的embeddings属性
        self.embeddings = YolosEmbeddings(config)
        # 创建YolosEncoder对象并赋值给实例的encoder属性
        self.encoder = YolosEncoder(config)
    
        # 使用config中的hidden_size和layer_norm_eps参数创建nn.LayerNorm对象，并赋值给实例的layernorm属性
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果add_pooling_layer为True，则创建YolosPooler对象并赋值给实例的pooler属性，否则设置为None
        self.pooler = YolosPooler(config) if add_pooling_layer else None
    
        # 调用post_init方法进行权重初始化和最终处理
        self.post_init()
    
    # 获取输入嵌入对象的方法，返回YolosPatchEmbeddings对象
    def get_input_embeddings(self) -> YolosPatchEmbeddings:
        return self.embeddings.patch_embeddings
    
    # 对模型的头部进行修剪的方法
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model.
    
        Args:
            heads_to_prune (`dict` of {layer_num: list of heads to prune in this layer}):
                See base class `PreTrainedModel`.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    # 前向传播方法，接受多个可选的torch.Tensor类型参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义模型的前向传播函数，接收输入的像素值和一些控制参数，返回模型输出
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 如果未提供输出注意力权重，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供返回字典参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为空，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # 头部掩码中的 1.0 表示保留该头部
        # 注意力权重的形状为 bsz x n_heads x N x N
        # 输入的头部掩码形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 将头部掩码转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将像素值传递给嵌入层进行处理
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传递给编码器进行处理
        encoder_outputs = self.encoder(
            embedding_output,
            height=pixel_values.shape[-2],
            width=pixel_values.shape[-1],
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行 LayerNormalization
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化层，则对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不要求返回字典，则将输出格式化为元组
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            # 返回编码器的输出
            return head_outputs + encoder_outputs[1:]

        # 如果需要返回字典格式的输出，则创建 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class YolosPooler(nn.Module):
    def __init__(self, config: YolosConfig):
        super().__init__()
        # 初始化线性层，用于池化模型，将隐藏状态映射到相同大小的空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化激活函数，用于激活线性层的输出
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 我们通过简单地取与第一个令牌对应的隐藏状态来“池化”模型。
        # 提取第一个令牌的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 通过线性层处理第一个令牌的隐藏状态，得到池化输出
        pooled_output = self.dense(first_token_tensor)
        # 对池化输出进行激活
        pooled_output = self.activation(pooled_output)
        return pooled_output


@add_start_docstrings(
    """
    YOLOS Model（包含一个 ViT 编码器）的目标检测头部，用于诸如 COCO 检测等任务。
    """,
    YOLOS_START_DOCSTRING,
)
class YolosForObjectDetection(YolosPreTrainedModel):
    def __init__(self, config: YolosConfig):
        super().__init__(config)

        # YOLOS（ViT）编码器模型
        self.vit = YolosModel(config, add_pooling_layer=False)

        # 目标检测头部
        # 为“无目标”类别添加一个头部
        self.class_labels_classifier = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=config.num_labels + 1, num_layers=3
        )
        self.bbox_predictor = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 参考 https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个解决方案，使 torchscript 快乐，因为 torchscript
        # 不支持具有非同构值的字典，例如
        # 同时具有张量和列表的字典。
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=YolosObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[List[Dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 从 transformers.models.detr.modeling_detr.dice_loss 复制
def dice_loss(inputs, targets, num_boxes):
    """
    计算 DICE 损失，类似于掩码的广义 IOU

    Args:
        inputs: 任意形状的浮点张量。
                每个示例的预测。
        targets: 与输入具有相同形状的浮点张量。存储了每个输入元素的二进制分类标签（负类为 0，正类为 1）。
    """
    # 对输入进行 sigmoid 操作
    inputs = inputs.sigmoid()
    # 将输入展平
    inputs = inputs.flatten(1)
```  
    # 计算输入和目标的乘积，并将其按行求和，然后乘以2
    numerator = 2 * (inputs * targets).sum(1)
    # 分别对输入和目标在最后一个维度上求和
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 根据公式计算损失值
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 对损失值按行求和后，再求平均值
    return loss.sum() / num_boxes
# 从transformers.models.detr.modeling_detr.sigmoid_focal_loss中复制而来的函数
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    在RetinaNet中使用的损失函数：https://arxiv.org/abs/1708.02002。

    参数:
        inputs (`torch.FloatTensor` of arbitrary shape):
            每个例子的预测结果。
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            一个存储了每个`inputs`中元素的二元分类标签的张量（0代表负类，1代表正类）。
        alpha (`float`, *optional*, defaults to `0.25`):
            用于平衡正负样本的可选权重因子，在范围(0,1)内。
        gamma (`int`, *optional*, defaults to `2`):
            用于平衡易与难样本的调节因子（1 - p_t）的指数。

    返回:
        损失张量
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 添加调节因子
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


# 从transformers.models.detr.modeling_detr.DetrLoss中复制而来的类，将Detr->Yolos
class YolosLoss(nn.Module):
    """
    此类用于计算YolosForObjectDetection/YolosForSegmentation的损失。运行过程分两步:
    1) 我们计算真实边界框和模型输出之间的匈牙利分配 2) 监督每对匹配的真实边界框/预测（监督类别和边界框）。

    关于`num_classes`参数的注意事项 (从原始detr.py中复制): “损失函数中`num_classes`参数的命名有点误导。实际上它对应于`max_obj_id` + 1，其中`max_obj_id`是数据集中类别的最大id。
    例如，COCO数据集的`max_obj_id`为90，所以我们传入`num_classes`为91。再举一个例子，对于只有一个类别id为1的数据集，应该传入`num_classes`为2（`max_obj_id` + 1）。更多详情请参考以下讨论 https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"

    参数:
        matcher (`YolosHungarianMatcher`):
            能够计算目标和提议之间匹配的模块。
        num_classes (`int`):
            对象类别的数量，不包括特殊的非对象类别。
        eos_coef (`float`):
            应用于非对象类别的相对分类权重。
        losses (`List[str]`):
            应用的所有损失的列表。见`get_loss`获取所有可用损失列表。
    """
    # 初始化方法，初始化匹配器、类别数量、eos系数和损失函数
    def __init__(self, matcher, num_classes, eos_coef, losses):
        # 调用父类构造函数
        super().__init__()
        # 设置匹配器
        self.matcher = matcher
        # 设置类别数量
        self.num_classes = num_classes
        # 设置eos系数
        self.eos_coef = eos_coef
        # 设置损失函数
        self.losses = losses
        # 创建全为1的权重向量
        empty_weight = torch.ones(self.num_classes + 1)
        # 将最后一个元素设置为eos系数
        empty_weight[-1] = self.eos_coef
        # 将权重向量作为缓冲区注册进来
        self.register_buffer("empty_weight", empty_weight)

    # 移除logging参数，这是原始实现的一部分
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        分类损失（NLL）目标字典必须包含键"类别标签"，其中包含维度为[nb_target_boxes]的张量
        """
        # 检查输出中是否有"logits"
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        # 获取源logits
        source_logits = outputs["logits"]

        # 获取排序后的索引
        idx = self._get_source_permutation_idx(indices)
        # 获取目标类别
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        # 创建与源logits形状相同的目标类别张量
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        # 计算交叉熵损失
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        计算基数错误，即预测的非空框数量的绝对误差。

        这实际上不是损失，仅用于记录目的。不会传播梯度。
        """
        # 获取logits
        logits = outputs["logits"]
        device = logits.device
        # 获取目标长度
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测非"no-object"（最后一个类别）的数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 计算绝对误差
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        计算与边界框相关的损失，包括 L1 回归损失和 GIoU 损失。

        目标字典必须包含键“boxes”，其中包含维度为[nb_target_boxes，4]的张量。目标框
        应以（center_x，center_y，w，h）格式提供，按图像大小标准化。
        """
        # 检查输出中是否有“pred_boxes”
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 获取源排列索引
        idx = self._get_source_permutation_idx(indices)
        # 获取源边界框和目标边界框
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算 L1 损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        # 计算并添加边界框损失
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        计算与蒙版相关的损失：焦点损失和 Dice 损失。

        目标字典必须包含键“masks”，其中包含维度为[nb_target_boxes，h，w]的张量。
        """
        # 检查输出中是否有“pred_masks”
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        # 获取源排列索引和目标排列索引
        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        # TODO 使用有效性标记来屏蔽由于填充而得到的无效区域
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # 将源掩码插值到目标大小
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        # 计算并返回损失
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # 重排预测值以符合索引
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx
    # 获取目标重新排列的索引
    def _get_target_permutation_idx(self, indices):
        # 按照给定的索引对目标进行排列
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    # 计算损失函数
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 损失函数映射表
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    # 前向传播
    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        # 排除辅助输出的情况
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # 获取输出的最后一层和目标之间的匹配索引
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点平均目标框的数量，用于归一化
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # 注释掉下面的函数，分布式训练将被添加
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # 在原始实现中，num_boxes 会被 get_world_size() 分割
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果存在辅助损失，我们将针对每个中间层的输出重复这个过程
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # 中间层的 mask 损失计算成本过高，我们忽略它
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 返回损失字典
        return losses
# 从transformers.models.detr.modeling_detr.DetrMLPPredictionHead复制而来，将Detr->Yolos
class YolosMLPPredictionHead(nn.Module):
    """
    非常简单的多层感知机（MLP，也称为FFN），用于预测边界框相对于图像的标准化中心坐标、高度和宽度。

    从https://github.com/facebookresearch/detr/blob/master/models/detr.py复制而来
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从transformers.models.detr.modeling_detr.DetrHungarianMatcher复制而来，将Detr->Yolos
class YolosHungarianMatcher(nn.Module):
    """
    此类计算网络的目标和预测之间的分配。

    为了效率原因，目标不包括no_object。因此，一般情况下，预测数量比目标数量多。
    在这种情况下，我们对最佳预测进行一对一匹配，而其他预测则未匹配（因此被视为非对象）。

    参数：
        class_cost:
            匹配成本中分类错误的相对权重。
        bbox_cost:
            边界框坐标的L1误差在匹配成本中的相对权重。
        giou_cost:
            边界框的giou损失在匹配成本中的相对权重。
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("匹配器的所有成本不能为0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # Softmax over the classification logits
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        # Flatten the predicted box coordinates
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # Concatenate the class labels of target boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        # Concatenate the target box coordinates
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        # Calculate the classification cost as -probability of target labels
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        # Calculate L1 distance cost between the predicted and target boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        # Calculate GIoU cost between predicted and target boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        # Combine costs for classification, L1 distance, and GIoU into a final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        # Reshape and move to CPU for further computation
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # Split the cost matrix into pieces based on target box sizes and do linear sum assignment
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        # Convert the indices to torch tensors for the indices of selected predictions and targets
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# 从 transformers.models.detr.modeling_detr._upcast 复制而来
def _upcast(t: Tensor) -> Tensor:
    # 通过将类型提升到等效的更高类型，防止乘法产生的数值溢出
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# 从 transformers.models.detr.modeling_detr.box_area 复制而来
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由它们的 (x1, y1, x2, y2) 坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            要计算面积的边界框。它们应该以 (x1, y1, x2, y2) 格式给出，要求 `0 <= x1 < x2` 和 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框的面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# 从 transformers.models.detr.modeling_detr.box_iou 复制而来
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# 从 transformers.models.detr.modeling_detr.generalized_box_iou 复制而来
def generalized_box_iou(boxes1, boxes2):
    """
    从 https://giou.stanford.edu/ 计算广义 IoU。边界框应该是以 [x0, y0, x1, y1] (角) 格式给出。

    Returns:
        `torch.FloatTensor`: 一个 [N, M] 的成对矩阵，其中 N = len(boxes1) and M = len(boxes2)
    """
    # 退化的边界框会产生 inf / nan 结果，因此进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# 从 transformers.models.detr.modeling_detr._max_by_axis 复制而来
def _max_by_axis(the_list):
    # 类型: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


# 从 transformers.models.detr.modeling_detr.NestedTensor 复制而来
class NestedTensor(object):
    # 初始化函数，接受一组张量和可选的遮罩，并保存它们
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    # 将对象的张量和遮罩移动到指定设备上
    def to(self, device):
        # 将张量移动到指定设备上
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    # 返回对象的张量和遮罩
    def decompose(self):
        return self.tensors, self.mask

    # 返回对象的字符串表示形式，即张量的内容
    def __repr__(self):
        return str(self.tensors)


# 从给定的张量列表创建一个NestedTensor对象
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:  # 检查张量维度是否为3
        # 计算最大的尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # 创建批次形状
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 创建全零张量和全一遮罩
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 将每个张量复制到全零张量对应的位置，更新遮罩
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果张量不是3维的，抛出异常
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
```