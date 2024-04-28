# `.\transformers\models\nat\modeling_nat.py`

```
# 设置编码器输出的数据结构，包括可能的隐藏状态和注意力信息
@dataclass
class NatEncoderOutput(ModelOutput):
    """
    Nat 编码器的输出，包括可能的隐藏状态和注意力信息。
    """
    # 定义函数参数说明
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每层输出的隐藏状态序列组成的元组。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            用于计算自注意力头中加权平均值的注意力权重。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含空间维度的每层输出隐藏状态序列组成的元组。
    
        # 初始化函数参数变量
        last_hidden_state: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
        reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，表示 NAT 模型的输出，同时包含最后隐藏状态的汇聚。
@dataclass
class NatModelOutput(ModelOutput):
    """
    Nat model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态的序列。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            最后一层隐藏状态的平均池化。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor`（一个用于嵌入输出 + 一个用于每个阶段输出）的形状为 `(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态加上初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含 `torch.FloatTensor`（每个阶段一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor`（一个用于嵌入输出 + 一个用于每个阶段输出）的形状为 `(batch_size, hidden_size, height, width)`。

            每层模型的隐藏状态加上初始嵌入输出，重塑以包括空间维度。
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个数据类，表示图像分类的 NAT 输出。
@dataclass
class NatImageClassifierOutput(ModelOutput):
    """
    Nat outputs for image classification.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（如果config.num_labels==1，则为回归）损失。
            （如果提供了`labels`，则返回）。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（如果config.num_labels==1，则为回归）得分（SoftMax之前）。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor`元组（包含嵌入输出和每个阶段的输出），
            形状为`(batch_size, sequence_length, hidden_size)`。

            每个层的模型隐藏状态以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor`元组（每个阶段一个），
            形状为`(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor`元组（包含嵌入输出和每个阶段的输出），
            形状为`(batch_size, hidden_size, height, width)`。

            每个层的模型隐藏状态以及初始嵌入输出，重塑以包含空间维度。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
class NatEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings.
    """

    def __init__(self, config):
        # 初始化 NatEmbeddings 类
        super().__init__()

        # 创建 NatPatchEmbeddings 对象
        self.patch_embeddings = NatPatchEmbeddings(config)

        # 初始化 LayerNorm 和 Dropout 层
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        # 对输入的像素值进行嵌入处理
        embeddings = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)

        # 对嵌入结果进行 Dropout 处理
        embeddings = self.dropout(embeddings)

        return embeddings


class NatPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, height, width, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        # 初始化 NatPatchEmbeddings 类
        super().__init__()
        patch_size = config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        self.num_channels = num_channels

        if patch_size == 4:
            pass
        else:
            # 如果补丁大小不为4，引发异常
            raise ValueError("Dinat only supports patch size of 4 at the moment.")

        # 创建卷积层进行特征映射
        self.projection = nn.Sequential(
            nn.Conv2d(self.num_channels, hidden_size // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        # 检查像素值的通道数与配置是否匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对像素值进行特征映射
        embeddings = self.projection(pixel_values)
        embeddings = embeddings.permute(0, 2, 3, 1)

        return embeddings


class NatDownsampler(nn.Module):
    """
    Convolutional Downsampling Layer.

    Args:
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        # 初始化 NatDownsampler 类
        super().__init__()
        self.dim = dim
        # 创建卷积层进行降采样
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        # 对输入特征进行卷积降采样
        input_feature = self.reduction(input_feature.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        input_feature = self.norm(input_feature)
        return input_feature


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    在每个样本上应用路径丢弃（随机深度），当应用于残差块的主路径时。

    Ross Wightman 的注释：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，但原始名称是具有误导性的，
    因为 'Drop Connect' 是另一篇论文中的一种不同形式的丢弃...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择更改层和参数名称为
    'drop path' 而不是混用 DropConnect 作为层名称，并使用 'survival rate' 作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output


# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制而来，将 Beit->Nat
class NatDropPath(nn.Module):
    """每个样本上应用路径丢弃（随机深度），当应用于残差块的主路径时。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class NeighborhoodAttention(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kernel_size = kernel_size

        # rpb is learnable relative positional biases; same concept is used Swin.
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)))

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 3, 1, 2, 4)
 def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 将输入的 hidden_states 通过 self.query、self.key 和 self.value 线性变换得到查询、键和值
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 在计算注意力权重之前，对查询进行缩放，将 query_layer 除以 sqrt(attention_head_size)
        # 这样做通常更高效，因为注意力权重通常是一个比查询更大的张量
        # 这样做不会改变结果，因为标量可以在矩阵乘法中交换
        query_layer = query_layer / math.sqrt(self.attention_head_size)

        # 计算“查询”和“键”之间的注意力得分，并添加相对位置偏差
        attention_scores = natten2dqkrpb(query_layer, key_layer, self.rpb, self.kernel_size, 1)

        # 将注意力得分归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 这实际上是通过丢弃整个 token 来进行自我注意，这可能看起来有点不寻常，但其来自于原始 Transformer 论文
        # 这里的 self.dropout 是一个 Dropout 层，用于对 attention_probs 进行随机丢弃
        attention_probs = self.dropout(attention_probs)

        # 通过将注意力概率权重与值进行加权求和，得到注意力上下文向量
        context_layer = natten2dav(attention_probs, value_layer, self.kernel_size, 1)
        # 将上下文向量的维度进行调整和置换
        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否需要输出 attention_probs，来构建输出元组
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 邻域注意力输出模块
class NeighborhoodAttentionOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层将输入转换为输出维度
        self.dense = nn.Linear(dim, dim)
        # 定义一个dropout层以防止过拟合
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态传入线性层并进行激活
        hidden_states = self.dense(hidden_states)
        # 将激活后的结果传入dropout层
        hidden_states = self.dropout(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states


# 邻域注意力模块
class NeighborhoodAttentionModule(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size):
        super().__init__()
        # 定义自注意力层
        self.self = NeighborhoodAttention(config, dim, num_heads, kernel_size)
        # 定义注意力输出层
        self.output = NeighborhoodAttentionOutput(config, dim)
        # 定义一个存储已剪枝头的集合
        self.pruned_heads = set()

    # 剪枝注意力头的函数
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可以剪枝的头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取自注意力的输出
        self_outputs = self.self(hidden_states, output_attentions)
        # 将自注意力输出传入注意力输出层
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将注意力输出与其他输出组成元组返回
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# NAT中间层模块
class NatIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层用于扩展特征维度
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 定义一个激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态传入线性层并进行激活
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# NAT输出模块
class NatOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层用于缩减特征维度
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 定义一个dropout层以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义一个前向传播方法，接受隐藏状态作为输入，并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行dropout操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为NatLayer的类，继承自nn.Module
class NatLayer(nn.Module):
    # 初始化方法
    def __init__(self, config, dim, num_heads, drop_path_rate=0.0):
        super().__init__()
        # 设置前馈的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置卷积核大小
        self.kernel_size = config.kernel_size
        # 使用LayerNorm对输入进行归一化处理
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建邻域注意力模块
        self.attention = NeighborhoodAttentionModule(config, dim, num_heads, kernel_size=self.kernel_size)
        # 设置删除路径的比率
        self.drop_path = NatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 对输出进行归一化处理
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建自然语言中间层
        self.intermediate = NatIntermediate(config, dim)
        # 创建自然语言输出
        self.output = NatOutput(config, dim)
        # 初始化层比例参数
        self.layer_scale_parameters = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((2, dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )

    # 判断是否需要对隐藏状态进行填充
    def maybe_pad(self, hidden_states, height, width):
        # 设置卷积窗口大小
        window_size = self.kernel_size
        pad_values = (0, 0, 0, 0, 0, 0)
        if height < window_size or width < window_size:
            # 计算需要填充的值
            pad_l = pad_t = 0
            pad_r = max(0, window_size - width)
            pad_b = max(0, window_size - height)
            pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
            # 对隐藏状态进行填充
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取隐藏状态的形状信息
        batch_size, height, width, channels = hidden_states.size()
        shortcut = hidden_states

        # 对隐藏状态进行归一化处理
        hidden_states = self.layernorm_before(hidden_states)
        # 如果隐藏状态小于卷积核大小，进行填充
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape

        # 获取注意力输出
        attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)

        attention_output = attention_outputs[0]

        # 判断是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_output = attention_output[:, :height, :width, :].contiguous()

        # 如果存在层比例参数，则进行比例调整
        if self.layer_scale_parameters is not None:
            attention_output = self.layer_scale_parameters[0] * attention_output

        # 对隐藏状态添加删除路径后得到新的隐藏状态
        hidden_states = shortcut + self.drop_path(attention_output)

        # 对层输出进行归一化处理
        layer_output = self.layernorm_after(hidden_states)
        # 输出层结果
        layer_output = self.output(self.intermediate(layer_output))

        # 如果存在层比例参数，则进行比例调整
        if self.layer_scale_parameters is not None:
            layer_output = self.layer_scale_parameters[1] * layer_output

        # 对层输出添加删除路径后得到新的层输出
        layer_output = hidden_states + self.drop_path(layer_output)

        # 如果输出注意力结果，则返回包含注意力结果的元组，否则只返回层输出结果
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
    # 初始化函数，设置模型参数
    def __init__(self, config, dim, depth, num_heads, drop_path_rate, downsample):
        # 调用父类初始化函数
        super().__init__()
        # 初始化模型的配置、维度等参数
        self.config = config
        self.dim = dim
        # 创建模型层列表
        self.layers = nn.ModuleList(
            [
                NatLayer(
                    config=config,
                    dim=dim,
                    num_heads=num_heads,
                    drop_path_rate=drop_path_rate[i],
                )
                for i in range(depth)
            ]
        )

        # 池化层
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 初始化指针
        self.pointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取隐藏状态的大小
        _, height, width, _ = hidden_states.size()
        for i, layer_module in enumerate(self.layers):
            # 对每个模型层进行前向传播
            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        # 如果存在池化层，则应用池化
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states_before_downsampling)

        # 组装输出
        stage_outputs = (hidden_states, hidden_states_before_downsampling)

        # 如果需要输出注意力矩阵，则补充到输出中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs
# 定义一个自然语言编码器的神经网络模型
class NatEncoder(nn.Module):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 确定自然语言编码器的层级数量
        self.num_levels = len(config.depths)
        # 保存配置信息
        self.config = config
        # 计算每一层的丢弃路径比例
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建每一层的自然语言编码器阶段
        self.levels = nn.ModuleList(
            [
                NatStage(
                    # 配置信息
                    config=config,
                    # 计算每一层的维度
                    dim=int(config.embed_dim * 2**i_layer),
                    # 每一层的深度
                    depth=config.depths[i_layer],
                    # 每一层的头数
                    num_heads=config.num_heads[i_layer],
                    # 每一层的丢弃路径比例
                    drop_path_rate=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    # 如果不是最后一层，则使用自然语言下采样器，否则为None
                    downsample=NatDownsampler if (i_layer < self.num_levels - 1) else None,
                )
                # 遍历每一层
                for i_layer in range(self.num_levels)
            ]
        )

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 定义函数，输入为隐藏状态和是否输出隐藏状态，输出为元组或 NatEncoderOutput 结果
    ) -> Union[Tuple, NatEncoderOutput]:
        # 如果需要输出隐藏状态，则初始化 all_hidden_states 为空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化 all_reshaped_hidden_states 为空元组，否则为 None
        all_reshaped_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力矩阵，则初始化 all_self_attentions 为空元组，否则为 None
        all_self_attentions = () if output_attentions else None

        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 重排隐藏状态张量的维度，从 b h w c 重排为 b c h w
            reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
            # 将当前隐藏状态添加到 all_hidden_states 中
            all_hidden_states += (hidden_states,)
            # 将重排后的隐藏状态添加到 all_reshaped_hidden_states 中
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        # 对每个层进行迭代
        for i, layer_module in enumerate(self.levels):
            # 调用当前层的前向传播方法
            layer_outputs = layer_module(hidden_states, output_attentions)

            # 更新隐藏状态为当前层输出的隐藏状态
            hidden_states = layer_outputs[0]
            # 记录当前层经过下采样之前的隐藏状态
            hidden_states_before_downsampling = layer_outputs[1]

            # 如果需要输出隐藏状态以及隐藏状态下采样之前的隐藏状态
            if output_hidden_states and output_hidden_states_before_downsampling:
                # 重排下采样之前的隐藏状态张量的维度，从 b h w c 重排为 b c h w
                reshaped_hidden_state = hidden_states_before_downsampling.permute(0, 3, 1, 2)
                # 将下采样之前的隐藏状态添加到 all_hidden_states 中
                all_hidden_states += (hidden_states_before_downsampling,)
                # 将重排后的下采样之前的隐藏状态添加到 all_reshaped_hidden_states 中
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            # 如果需要输出隐藏状态但不需要隐藏状态下采样之前的隐藏状态
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                # 重排隐藏状态张量的维度，从 b h w c 重排为 b c h w
                reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
                # 将当前隐藏状态添加到 all_hidden_states 中
                all_hidden_states += (hidden_states,)
                # 将重排后��隐藏状态添加到 all_reshaped_hidden_states 中
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        # 如果不需要返回字典形式的结果
        if not return_dict:
            # 返回隐藏状态、所有隐藏状态和所有注意力矩阵的元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # 返回 NatEncoderOutput 类的实例，包括最后的隐藏状态、所有隐藏状态、所有注意力矩阵和重排后的所有隐藏状态
        return NatEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )
# 抽象基类，用于处理权重初始化和加载预训练模型的简单接口
class NatPreTrainedModel(PreTrainedModel):
    # 配置类
    config_class = NatConfig
    # 基础模型前缀
    base_model_prefix = "nat"
    # 主要输入名称
    main_input_name = "pixel_values"

    # 初始化权重
    def _init_weights(self, module):
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 用正态分布初始化权重，标准差为配置的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 开始文档字符串
NAT_START_DOCSTRING = r"""
    这是一个PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)的子类。可以作为常规的PyTorch Module使用，并参考PyTorch文档了解更多关于一般用法和行为的信息。

    参数:
        config ([`NatConfig`]): 包含模型所有参数的配置类。
            使用配置文件初始化不会加载与模型关联的权重，只会加载配置。
            查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
"""


# 输入文档字符串
NAT_INPUTS_DOCSTRING = r"""
    参数:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。可以使用 [`AutoImageProcessor`] 获得像素值。详见 [`ViTImageProcessor.__call__`]。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。更多详情见返回张量部分的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。更多详情见返回张量部分的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
"""


# 添加开始文档字符串
@add_start_docstrings(
    "The bare Nat Model transformer outputting raw hidden-states without any specific head on top.",
    NAT_START_DOCSTRING,
)
# NatModel 类
class NatModel(NatPreTrainedModel):
    pass
    # 定义 NatModel 类，继承自 nn.Module
        def __init__(self, config, add_pooling_layer=True):
            # 调用父类 nn.Module 的构造函数
            super().__init__(config)
            # 检查所需的后端是否可用
            requires_backends(self, ["natten"])
            # 保存配置参数
            self.config = config
            # 获取金字塔层数
            self.num_levels = len(config.depths)
            # 计算最终特征维度
            self.num_features = int(config.embed_dim * 2 ** (self.num_levels - 1))
            # 创建嵌入层
            self.embeddings = NatEmbeddings(config)
            # 创建编码器
            self.encoder = NatEncoder(config)
            # 创建层归一化层
            self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
            # 创建自适应平均池化层
            self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None
            # 进行权重初始化
            self.post_init()
    
        # 获取输入嵌入层
        def get_input_embeddings(self):
            return self.embeddings.patch_embeddings
    
        # 剪枝注意力头
        def _prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.encoder.layer[layer].attention.prune_heads(heads)
    
        # 前向传播
        @add_start_docstrings_to_model_forward(NAT_INPUTS_DOCSTRING)
        @add_code_sample_docstrings(
            checkpoint=_CHECKPOINT_FOR_DOC,
            output_type=NatModelOutput,
            config_class=_CONFIG_FOR_DOC,
            modality="vision",
            expected_output=_EXPECTED_OUTPUT_SHAPE,
        )
        def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
    # 根据函数的参数设置是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    # 根据函数的参数设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
    # 根据函数的参数设置是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 如果未传入像素值，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

    # 将像素值传入嵌入层得到嵌入输出
        embedding_output = self.embeddings(pixel_values)

    # 将嵌入输出传入编码器，根据参数输出注意力权重和隐藏状态
        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # 获取编码器输出中的序列输出并进行 LayerNorm 处理
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

    # 初始化池化输出为 None，并且如果存在池化层，则进行池化操作
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.flatten(1, 2).transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

    # 如果不返回字典形式的输出，则将需要返回的内容组成元组输出
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

    # 如果返回字典形式的输出，则返回经过格式处理的输出结构体 NatModelOutput
        return NatModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
    # 使用 add_start_docstrings 装饰器添加模型类的文档字符串
    """
    Nat Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """
    # 继承自 NatPreTrainedModel 的 NatForImageClassification 类
    # 通过该装饰器添加文档字符串
    @add_start_docstrings(
    NAT_START_DOCSTRING,
    )
class NatForImageClassification(NatPreTrainedModel):
    # 初始化函数，接受参数 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 需要 natten 后端支持
        requires_backends(self, ["natten"])

        # 记录标签数量
        self.num_labels = config.num_labels
        # 创建 NatModel 对象
        self.nat = NatModel(config)

        # 分类器头部
        self.classifier = (
            # 如果标签数量大于 0，则创建线性层；否则创建一个恒等函数
            nn.Linear(self.nat.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 和 add_code_sample_docstrings 装饰器添加前向传播函数的文档字符串
    @add_start_docstrings_to_model_forward(NAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=NatImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 前向传播函数，接受多个输入参数
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 计算图像分类或回归的损失函数
    def forward(
        self,
        pixel_values,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, NatImageClassifierOutput]:
        # 判断是否使用return_dict，如果未指定则使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 调用self.nat函数获取输出
        outputs = self.nat(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 从输出中获取池化后的输出
        pooled_output = outputs[1]
        
        # 将池化后的输出传入分类器得到logits
        logits = self.classifier(pooled_output)
        
        # 如果提供了labels，则计算损失函数
        loss = None
        if labels is not None:
            # 自动判断问题类型(回归或分类)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            # 根据不同问题类型计算损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 根据return_dict决定返回形式
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        return NatImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
# 用于添加初始文档字符串，描述 NAT 骨干网络的用途，可与 DETR 和 MaskFormer 等框架一起使用
@add_start_docstrings(
    "NAT backbone, to be used with frameworks like DETR and MaskFormer.",
    NAT_START_DOCSTRING,
)
# 定义 NatBackbone 类，继承自 NatPreTrainedModel 和 BackboneMixin
class NatBackbone(NatPreTrainedModel, BackboneMixin):
    # 构造函数，初始化类实例
    def __init__(self, config):
        # 调用父类 NatPreTrainedModel 的构造函数
        super().__init__(config)
        # 调用 BackboneMixin 的 _init_backbone 方法
        super()._init_backbone(config)

        # 确保依赖的后端库已加载
        requires_backends(self, ["natten"])

        # 初始化嵌入层
        self.embeddings = NatEmbeddings(config)
        # 初始化编码器
        self.encoder = NatEncoder(config)

        # 计算特征数，用于定义层次结构
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]

        # 为输出特征的隐藏状态添加层归一化
        hidden_states_norms = {}
        for stage, num_channels in zip(self.out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 前向传播方法，定义了模型的计算流程
    @add_start_docstrings_to_model_forward(NAT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,  # 输入像素值张量
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
        ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "shi-labs/nat-mini-in1k-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 512, 7, 7]
        ```"""
        # 如果 return_dict 参数为 None，则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 output_hidden_states 参数为 None，则使用模型配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_attentions 参数为 None，则使用模型配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 使用输入像素值计算嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 对嵌入输出进行编码
        outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=True,
        )

        # 获取编码后的隐藏状态
        hidden_states = outputs.reshaped_hidden_states

        # 初始化特征图为空元组
        feature_maps = ()
        # 遍历每个阶段的名称和隐藏状态
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            # 如果当前阶段在所需输出特征中
            if stage in self.out_features:
                # 转置隐藏状态以匹配预期形状
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                # 重新形状隐藏状态以应用标准化
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                # 添加处理后的隐藏状态到特征图中
                feature_maps += (hidden_state,)

        # 如果不返回字典，则返回特征图和可能的隐藏状态
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回带有特征图、隐藏状态和注意力的字典
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```