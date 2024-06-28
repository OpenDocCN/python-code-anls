# `.\models\beit\modeling_flax_beit.py`

```py
# BEIT_START_DOCSTRING 是一个原始文档字符串的标记，用于后续的文档字符串生成
BEIT_START_DOCSTRING = r"""
    # 这个模型继承自 `FlaxPreTrainedModel`。查看超类的文档，了解库为所有模型实现的通用方法（如下载、保存和从PyTorch模型转换权重）。
    
    # 这个模型还是一个 `flax.linen.Module` 的子类。可以将其作为常规的 Flax linen 模块使用，并参考 Flax 文档了解与一般使用和行为相关的所有内容。
    
    # 最后，这个模型支持 JAX 的一些内置特性，如：
    # - Just-In-Time (JIT) 编译：https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit
    # - 自动微分：https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation
    # - 向量化：https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap
    # - 并行化：https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap
    
    # 参数：
    # config (`BeitConfig`): 包含模型所有参数的配置类。
    # 初始化时使用配置文件不会加载模型的权重，只加载配置。查看 `~FlaxPreTrainedModel.from_pretrained` 方法以加载模型权重。
    
    # dtype (`jax.numpy.dtype`, *optional*, 默认为 `jax.numpy.float32`):
    # 计算时的数据类型。可以是 `jax.numpy.float32`、`jax.numpy.float16`（在GPU上）和 `jax.numpy.bfloat16`（在TPU上）之一。
    # 可用于在GPU或TPU上启用混合精度训练或半精度推理。如果指定了 dtype，所有计算都将使用给定的 `dtype` 进行。
    
    # **注意，这只指定了计算时的数据类型，不影响模型参数的数据类型。**
    # 如果希望更改模型参数的数据类型，请参阅 `~FlaxPreTrainedModel.to_fp16` 和 `~FlaxPreTrainedModel.to_bf16`。
"""

BEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`numpy.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

def relative_position_index_init(window_size: Tuple[int, int]) -> jnp.ndarray:
    """
    Initialize a matrix of relative position indices for tokens inside a window.

    Args:
        window_size: Tuple specifying the height and width of the window.

    Returns:
        jnp.ndarray: Matrix of relative position indices.

    This function computes the relative positions between tokens in a window based on the specified window size.
    """

    num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3

    coords_h = np.arange(window_size[0])
    coords_w = np.arange(window_size[1])
    coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
    coords_flatten = np.reshape(coords, (2, -1))
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
    relative_coords[:, :, 1] += window_size[1] - 1
    relative_coords[:, :, 0] *= 2 * window_size[1] - 1

    relative_position_index = np.zeros(shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
    relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
    relative_position_index[0, 0:] = num_relative_distance - 3
    relative_position_index[0:, 0] = num_relative_distance - 2
    relative_position_index[0, 0] = num_relative_distance - 1
    return jnp.array(relative_position_index)


def ones_with_scale(key, shape, scale, dtype=jnp.float32):
    """
    Create a tensor filled with ones scaled by a specified factor.

    Args:
        key: Random key for JAX randomness.
        shape: Shape of the tensor.
        scale: Scaling factor for the ones tensor.
        dtype: Data type of the tensor.

    Returns:
        jnp.ndarray: Tensor filled with ones scaled by `scale`.
    """
    return jnp.ones(shape, dtype) * scale


class FlaxBeitDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    rate: float

    @nn.module.compact
    def __call__(self, inputs, deterministic: Optional[bool] = True):
        """
        Apply drop path regularization to inputs.

        Args:
            inputs: Input tensor to which drop path is applied.
            deterministic: Whether to apply deterministic or stochastic drop path.

        Returns:
            jnp.ndarray: Output tensor after applying drop path regularization.
        """
        if self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        if deterministic:
            return inputs
        else:
            shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
            rng = self.make_rng("droppath")
            random_tensor = keep_prob + jax.random.uniform(rng, shape=shape, dtype=inputs.dtype)
            binary_tensor = jnp.floor(random_tensor)
            output = inputs / keep_prob * binary_tensor
            return output
    # 定义一个名为 FlaxBeitPatchEmbeddings 的新模块，继承自 nn.Module
    class FlaxBeitPatchEmbeddings(nn.Module):
        # 引入配置类 BeitConfig
        config: BeitConfig
        # 定义计算时使用的数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

        # 模块的设置方法
        def setup(self):
            # 从配置中获取通道数和图像大小
            self.num_channels = self.config.num_channels
            image_size = self.config.image_size
            patch_size = self.config.patch_size
            # 计算图像被分成的块数和每个块的形状
            num_patches = (image_size // patch_size) * (image_size // patch_size)
            patch_shape = (image_size // patch_size, image_size // patch_size)
            # 设置模块的属性
            self.num_patches = num_patches
            self.patch_shape = patch_shape
            # 创建一个卷积层投影，用于将输入投影到隐藏尺寸空间
            self.projection = nn.Conv(
                self.config.hidden_size,
                kernel_size=(patch_size, patch_size),
                strides=(patch_size, patch_size),
                padding="VALID",
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )

        # 模块的调用方法，处理输入像素值
        def __call__(self, pixel_values):
            # 检查输入像素值的通道数是否与配置中设置的通道数匹配
            num_channels = pixel_values.shape[-1]
            if num_channels != self.num_channels:
                raise ValueError(
                    "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                )
            # 使用投影层处理像素值，得到嵌入表示
            embeddings = self.projection(pixel_values)
            batch_size, _, _, channels = embeddings.shape
            # 将嵌入表示重塑为适当的形状，以便后续处理
            return jnp.reshape(embeddings, (batch_size, -1, channels))


    # 定义一个名为 FlaxBeitEmbeddings 的新模块，继承自 nn.Module
    class FlaxBeitEmbeddings(nn.Module):
        """构建CLS令牌、位置和补丁嵌入。"""

        # 引入配置类 BeitConfig
        config: BeitConfig
        # 定义计算时使用的数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32  # 计算时使用的数据类型

        # 模块的设置方法
        def setup(self):
            # 定义一个CLS令牌，初始化为全零，形状为 (1, 1, hidden_size)
            self.cls_token = self.param("cls_token", nn.initializers.zeros, (1, 1, self.config.hidden_size))
            # 如果配置要求使用掩码令牌，则定义一个掩码令牌，初始化为全零，形状也为 (1, 1, hidden_size)
            if self.config.use_mask_token:
                self.mask_token = self.param("mask_token", nn.initializers.zeros, (1, 1, self.config.hidden_size))
            # 创建补丁嵌入模块实例，使用给定的配置和数据类型
            self.patch_embeddings = FlaxBeitPatchEmbeddings(self.config, dtype=self.dtype)
            num_patches = self.patch_embeddings.num_patches
            # 如果配置要求使用绝对位置嵌入，则定义一个绝对位置嵌入参数，初始化为全零，形状为 (1, num_patches + 1, hidden_size)
            if self.config.use_absolute_position_embeddings:
                self.position_embeddings = self.param(
                    "position_embeddings", nn.initializers.zeros, (1, num_patches + 1, self.config.hidden_size)
                )
            # 定义一个Dropout层，用于随机断开输入单元，以防止过拟合
            self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
    # 定义一个类的调用方法，接受像素值和可选的布尔掩码作为输入参数，并返回嵌入表示
    def __call__(self, pixel_values, bool_masked_pos=None, deterministic=True):
        # 使用patch_embeddings方法将像素值转换为嵌入表示
        embeddings = self.patch_embeddings(pixel_values)
        # 获取嵌入表示的维度信息：批量大小、序列长度、嵌入维度
        batch_size, seq_len, _ = embeddings.shape

        # 创建一个形状与嵌入表示相同的CLS token，并将其数据类型转换为embeddings的数据类型
        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        cls_tokens = cls_tokens.astype(embeddings.dtype)

        # 如果给定了布尔掩码，替换被掩码的视觉令牌为mask_tokens
        if bool_masked_pos is not None:
            # 创建一个形状与嵌入表示相同的mask token，并将其数据类型转换为embeddings的数据类型
            mask_tokens = jnp.broadcast_to(self.mask_token, (batch_size, seq_len, self.config.hidden_size))
            mask_tokens = mask_tokens.astype(embeddings.dtype)
            # 使用布尔掩码来选择性地应用mask_tokens替换embeddings中的视觉令牌
            w = jnp.expand_dims(bool_masked_pos, axis=-1)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        # 将CLS token与嵌入表示连接起来，形成完整的嵌入表示序列
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

        # 如果配置中使用了绝对位置嵌入，将位置嵌入加到嵌入表示上
        if self.config.use_absolute_position_embeddings:
            embeddings = embeddings + self.position_embeddings.astype(embeddings.dtype)

        # 使用dropout方法对嵌入表示进行随机失活，根据deterministic参数确定是否确定性地进行操作
        embeddings = self.dropout(embeddings, deterministic=deterministic)
        # 返回最终的嵌入表示
        return embeddings
    # FlaxBeitRelativePositionBias 类，用于计算相对位置偏置
    class FlaxBeitRelativePositionBias(nn.Module):
        # BeitConfig 类型的配置信息
        config: BeitConfig
        # 窗口大小的元组，表示注意力窗口的尺寸
        window_size: Tuple[int, int]
        # 计算中使用的数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32  # the dtype of the computation

        # 模块初始化方法
        def setup(self):
            # 计算相对距离的数量，形状为 (2*Wh-1)*(2*Ww-1) + 3
            num_relative_distance = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) + 3
            # 创建参数，相对位置偏置表，形状为 (num_relative_distance, num_attention_heads)
            self.relative_position_bias_table = self.param(
                "relative_position_bias_table",
                nn.initializers.zeros,
                (num_relative_distance, self.config.num_attention_heads),
            )  # 2*Wh-1 * 2*Ww-1, nH
            # 类别到标记 & 标记到类别 & 类别到类别

            # 初始化相对位置索引
            self.relative_position_index = relative_position_index_init(self.window_size)

        # 对象调用方法
        def __call__(self):
            # 将相对位置索引重塑为一维数组
            index = self.relative_position_index.reshape(-1)
            # 定义相对位置偏置的形状
            shape = (self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
            # 根据索引从相对位置偏置表中获取相对位置偏置，并重塑为指定形状
            relative_position_bias = self.relative_position_bias_table[index].reshape(shape)  # Wh*Ww,Wh*Ww,nH
            # 返回相对位置偏置，并进行维度转置
            return jnp.transpose(relative_position_bias, (2, 0, 1))


    # FlaxBeitSelfAttention 类，实现自注意力机制
    class FlaxBeitSelfAttention(nn.Module):
        # BeitConfig 类型的配置信息
        config: BeitConfig
        # 窗口大小的元组，表示注意力窗口的尺寸
        window_size: Tuple[int, int]
        # 计算中使用的数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32  # the dtype of the computation

        # 模块初始化方法
        def setup(self):
            # 检查隐藏层大小是否是注意力头数的倍数，且不是嵌入大小的属性
            if self.config.hidden_size % self.config.num_attention_heads != 0 and not hasattr(
                self.config, "embedding_size"
            ):
                # 抛出数值错误，提示隐藏大小不是注意力头数的倍数
                raise ValueError(
                    f"The hidden size {self.config.hidden_size,} is not a multiple of the number of attention "
                    f"heads {self.config.num_attention_heads}."
                )

            # 初始化查询、键、值的线性变换层
            self.query = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )
            self.key = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
                use_bias=False,
            )
            self.value = nn.Dense(
                self.config.hidden_size,
                dtype=self.dtype,
                kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            )

            # 如果定义了窗口大小，创建相对位置偏置对象
            self.relative_position_bias = (
                FlaxBeitRelativePositionBias(self.config, window_size=self.window_size, dtype=self.dtype)
                if self.window_size
                else None
            )

        # 对象调用方法，实现自注意力计算
        def __call__(
            self, hidden_states, relative_position_bias=None, deterministic: bool = True, output_attentions: bool = False
    ):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        # 将查询向量转换成多头格式：(batch_size, seq_length, num_attention_heads, head_dim)
        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        
        # 将数值向量转换成多头格式：(batch_size, seq_length, num_attention_heads, head_dim)
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        
        # 将键向量转换成多头格式：(batch_size, seq_length, num_attention_heads, head_dim)
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        # 如果非确定性计算且设置了注意力概率的丢弃率，则创建一个用于丢弃的随机数生成器
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attention_bias = jnp.array(0.0, dtype=self.dtype)
        # 如果存在相对位置偏置，则添加到注意力偏置中
        if self.relative_position_bias is not None:
            attention_bias = jnp.expand_dims(self.relative_position_bias(), 0)
            attention_bias = attention_bias.astype(query_states.dtype)

        # 如果提供了共享的相对位置偏置，则将其加到注意力偏置中
        if relative_position_bias is not None:
            attention_bias = attention_bias + relative_position_bias.astype(attention_bias.dtype)

        # 计算点积注意力的权重
        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        # 使用注意力权重计算注意力输出
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        # 如果需要输出注意力权重，则返回注意力输出和注意力权重；否则只返回注意力输出
        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs
# 定义一个 FlaxBeitSelfOutput 类，继承自 nn.Module
class FlaxBeitSelfOutput(nn.Module):
    # 配置项，使用 BeitConfig 类型
    config: BeitConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 定义一个全连接层，输出大小为 self.config.hidden_size
        # 初始化权重使用正态分布，标准差为 self.config.initializer_range
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 定义一个 Dropout 层，丢弃率为 self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 调用方法，接收 hidden_states 和 deterministic 参数
    def __call__(self, hidden_states, deterministic: bool = True):
        # 将 hidden_states 输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 处理 hidden_states
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个 FlaxBeitAttention 类，继承自 nn.Module
class FlaxBeitAttention(nn.Module):
    # 配置项，使用 BeitConfig 类型
    config: BeitConfig
    # 窗口大小，为元组类型，存储两个整数值
    window_size: Tuple[int, int]
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 初始化方法
    def setup(self):
        # 定义一个自注意力层，使用 FlaxBeitSelfAttention 类
        self.attention = FlaxBeitSelfAttention(self.config, self.window_size, dtype=self.dtype)
        # 定义一个输出层，使用 FlaxBeitSelfOutput 类
        self.output = FlaxBeitSelfOutput(self.config, dtype=self.dtype)

    # 调用方法，接收 hidden_states、relative_position_bias、deterministic 和 output_attentions 参数
    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic=True, output_attentions: bool = False
    ):
        # 执行自注意力层的调用方法，传入相关参数
        attn_outputs = self.attention(
            hidden_states, relative_position_bias, deterministic=deterministic, output_attentions=output_attentions
        )
        # 获取注意力输出的第一个元素
        attn_output = attn_outputs[0]
        # 将注意力输出传入输出层进行处理
        attn_output = self.output(attn_output, deterministic=deterministic)

        # 初始化 outputs 为包含 attn_output 的元组
        outputs = (attn_output,)

        # 如果 output_attentions 为 True，则将注意力输出的第二个元素加入 outputs 中
        if output_attentions:
            outputs += (attn_outputs[1],)

        # 返回 outputs
        return outputs


# 定义一个 FlaxBeitIntermediate 类，继承自 nn.Module
class FlaxBeitIntermediate(nn.Module):
    # 配置项，使用 BeitConfig 类型
    config: BeitConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 定义一个全连接层，输出大小为 self.config.intermediate_size
        # 初始化权重使用正态分布，标准差为 self.config.initializer_range
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 激活函数为配置中指定的隐藏层激活函数
        self.activation = ACT2FN[self.config.hidden_act]

    # 调用方法，接收 hidden_states 参数
    def __call__(self, hidden_states):
        # 将 hidden_states 输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理 hidden_states
        hidden_states = self.activation(hidden_states)

        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个 FlaxBeitOutput 类，继承自 nn.Module
class FlaxBeitOutput(nn.Module):
    # 配置项，使用 BeitConfig 类型
    config: BeitConfig
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    # 初始化方法
    def setup(self):
        # 定义一个全连接层，输出大小为 self.config.hidden_size
        # 初始化权重使用正态分布，标准差为 self.config.initializer_range
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        # 定义一个 Dropout 层，丢弃率为 self.config.hidden_dropout_prob
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    # 调用方法，接收 hidden_states 和 deterministic 参数
    def __call__(self, hidden_states, deterministic: bool = True):
        # 将 hidden_states 输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 处理 hidden_states
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)

        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个 FlaxBeitLayer 类，继承自 nn.Module
class FlaxBeitLayer(nn.Module):
    # 配置项，使用 BeitConfig 类型
    config: BeitConfig
    # 窗口大小，为元组类型，存储两个整数值
    window_size: Tuple[int, int]
    # DropPath 的概率
    drop_path_rate: float
    # 计算时的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    # 在初始化方法中设置模型的各个组件
    def setup(self):
        # 初始化注意力机制组件
        self.attention = FlaxBeitAttention(self.config, self.window_size, dtype=self.dtype)
        # 初始化中间层组件
        self.intermediate = FlaxBeitIntermediate(self.config, dtype=self.dtype)
        # 初始化输出层组件
        self.output = FlaxBeitOutput(self.config, dtype=self.dtype)
        # 初始化前层归一化组件，使用给定的 epsilon 参数
        self.layernorm_before = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 初始化 DropPath 组件，使用给定的丢弃率
        self.drop_path = FlaxBeitDropPath(rate=self.drop_path_rate)
        # 初始化后层归一化组件，使用给定的 epsilon 参数
        self.layernorm_after = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        # 初始化 lambda_1 和 lambda_2 参数，如果初始化值大于 0，则创建参数；否则设为 None
        self.init_values = self.config.layer_scale_init_value
        if self.init_values > 0:
            self.lambda_1 = self.param("lambda_1", ones_with_scale, (self.config.hidden_size), self.init_values)
            self.lambda_2 = self.param("lambda_2", ones_with_scale, (self.config.hidden_size), self.init_values)
        else:
            self.lambda_1 = None
            self.lambda_2 = None

    # 实现调用方法，处理输入的隐藏状态，执行模型的前向传播
    def __call__(
        self, hidden_states, relative_position_bias=None, deterministic: bool = True, output_attentions: bool = False
    ):
        # 执行自注意力机制，包括前层归一化
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 BEiT 中，自注意力前先进行归一化
            relative_position_bias,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        # 获取自注意力的输出
        attention_output = self_attention_outputs[0]

        # 如果 lambda_1 参数存在，则应用于注意力输出
        if self.lambda_1 is not None:
            attention_output = self.lambda_1.astype(attention_output.dtype) * attention_output

        # 第一次残差连接
        hidden_states = self.drop_path(attention_output, deterministic=deterministic) + hidden_states

        # 在 BEiT 中，层归一化也应用于自注意力后
        layer_output = self.layernorm_after(hidden_states)

        # 执行中间层操作
        layer_output = self.intermediate(layer_output)
        # 执行输出层操作，包括确定性与否
        layer_output = self.output(layer_output, deterministic=deterministic)

        # 如果 lambda_2 参数存在，则应用于中间层输出
        if self.lambda_2 is not None:
            layer_output = self.lambda_2.astype(layer_output.dtype) * layer_output

        # 第二次残差连接，将中间层输出与原始隐藏状态相加
        layer_output = self.drop_path(layer_output, deterministic=deterministic) + hidden_states

        # 返回最终输出，包括中间层输出或者中间层输出及注意力权重（根据需求）
        outputs = (layer_output,)

        if output_attentions:
            outputs += (self_attention_outputs[1],)

        return outputs
class FlaxBeitLayerCollection(nn.Module):
    config: BeitConfig  # 类型注解，指定 config 属性的类型为 BeitConfig
    window_size: Tuple[int, int]  # 类型注解，指定 window_size 属性的类型为元组，包含两个整数
    drop_path_rates: List[float]  # 类型注解，指定 drop_path_rates 属性的类型为列表，包含浮点数
    relative_position_bias: Callable[[], jnp.ndarray]  # 类型注解，指定 relative_position_bias 属性的类型为可调用对象，返回 jnp.ndarray
    dtype: jnp.dtype = jnp.float32  # 类型注解，默认值为 jnp.float32，指定 dtype 属性的类型为 jnp.dtype，表示计算时的数据类型

    def setup(self):
        # 初始化 layers 属性为一个列表，每个元素是一个 FlaxBeitLayer 实例
        self.layers = [
            FlaxBeitLayer(
                self.config,
                window_size=self.window_size if self.config.use_relative_position_bias else None,
                drop_path_rate=self.drop_path_rates[i],
                name=str(i),
                dtype=self.dtype,
            )
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果 output_attentions 为 True，则初始化空元组 all_attentions，否则设为 None
        all_attentions = () if output_attentions else None
        # 如果 output_hidden_states 为 True，则初始化空元组 all_hidden_states，否则设为 None
        all_hidden_states = () if output_hidden_states else None

        # 遍历 layers 列表中的每个层，并处理 hidden_states
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)  # 将当前 hidden_states 添加到 all_hidden_states 元组中
            # 根据 self.relative_position_bias 的值初始化 relative_position_bias
            relative_position_bias = self.relative_position_bias() if self.relative_position_bias is not None else None
            # 调用当前层 layer 的处理方法，更新 hidden_states
            layer_outputs = layer(
                hidden_states, relative_position_bias, deterministic=deterministic, output_attentions=output_attentions
            )
            # 更新 hidden_states 为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果 output_attentions 为 True，则将当前层的注意力加入 all_attentions 元组
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 如果 output_hidden_states 为 True，则将最终的 hidden_states 加入 all_hidden_states 元组
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 初始化输出为包含最终 hidden_states 的元组 outputs
        outputs = (hidden_states,)
        # 如果 return_dict 为 False，则返回 outputs 中不为 None 的元素组成的元组
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回 FlaxBaseModelOutput 对象，包含最终的 hidden_states、所有 hidden_states 和所有 attentions
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class FlaxBeitEncoder(nn.Module):
    config: BeitConfig  # 类型注解，指定 config 属性的类型为 BeitConfig
    window_size: Tuple[int, int]  # 类型注解，指定 window_size 属性的类型为元组，包含两个整数
    dtype: jnp.dtype = jnp.float32  # 类型注解，默认值为 jnp.float32，指定 dtype 属性的类型为 jnp.dtype，表示计算时的数据类型

    def setup(self):
        # 如果 self.config.use_shared_relative_position_bias 为 True，则初始化 relative_position_bias
        if self.config.use_shared_relative_position_bias:
            self.relative_position_bias = FlaxBeitRelativePositionBias(
                config=self.config, window_size=self.window_size, dtype=self.dtype
            )

        # 根据 stochastic depth decay rule 初始化 drop_path_rates 列表
        drop_path_rates = list(np.linspace(0, self.config.drop_path_rate, self.config.num_hidden_layers))
        # 初始化 layer 属性为 FlaxBeitLayerCollection 实例
        self.layer = FlaxBeitLayerCollection(
            self.config,
            window_size=self.window_size,
            drop_path_rates=drop_path_rates,
            relative_position_bias=self.relative_position_bias
            if self.config.use_shared_relative_position_bias
            else None,
            dtype=self.dtype,
        )
    # 定义一个特殊方法 __call__，使对象可以像函数一样被调用
    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用对象的 layer 方法，传入参数和关键字参数，并返回结果
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class FlaxBeitPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用BeitConfig作为配置类
    config_class = BeitConfig
    # base_model_prefix指定基础模型的前缀为"beit"
    base_model_prefix = "beit"
    # main_input_name指定主要输入名称为"pixel_values"
    main_input_name = "pixel_values"
    # module_class用于存储模块类，初始时未指定
    module_class: nn.Module = None

    def __init__(
        self,
        config: BeitConfig,
        input_shape=None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 根据配置类和其他参数初始化模块
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 如果未提供输入形状，则使用默认形状
        if input_shape is None:
            input_shape = (1, config.image_size, config.image_size, config.num_channels)
        # 调用父类的初始化方法，传递配置、模块、输入形状、种子、数据类型等参数
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化像素值张量
        pixel_values = jnp.zeros(input_shape, dtype=self.dtype)

        # 分割随机数生成器，用于不同的参数初始化
        params_rng, dropout_rng = jax.random.split(rng)
        dropout_rng, droppath_rng = jax.random.split(dropout_rng)
        rngs = {"params": params_rng, "dropout": dropout_rng, "droppath": droppath_rng}

        # 使用模块的初始化方法初始化随机参数
        random_params = self.module.init(rngs, pixel_values, return_dict=False)["params"]

        if params is not None:
            # 如果提供了参数，则将随机初始化的参数与提供的参数进行合并
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            # 如果未提供参数，则直接返回随机初始化的参数
            return random_params

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def __call__(
        self,
        pixel_values,
        bool_masked_pos=None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
            # 如果 output_attentions 参数为 None，则使用 self.config.output_attentions 的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果 output_hidden_states 参数为 None，则使用 self.config.output_hidden_states 的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果 return_dict 参数为 None，则使用 self.config.return_dict 的默认值
            return_dict = return_dict if return_dict is not None else self.config.return_dict

            # 将像素值张量进行转置，调整通道顺序为 (batch_size, height, width, channels)
            pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))
            # 如果需要处理任何 PRNG（伪随机数发生器），则初始化一个空字典用于存储不同的 PRNG
            rngs = {}
            if dropout_rng is not None:
                # 如果 dropout_rng 不为 None，则使用 JAX 提供的随机数分割函数拆分 PRNG
                dropout_rng, droppath_rng = jax.random.split(dropout_rng)
                rngs["dropout"] = dropout_rng
                rngs["droppath"] = droppath_rng

            # 调用模块的 apply 方法，传递参数、像素值张量、布尔掩码、训练标志、输出注意力、隐藏状态、返回字典和 PRNGs
            return self.module.apply(
                {"params": params or self.params},  # 模型参数
                jnp.array(pixel_values, dtype=jnp.float32),  # 像素值张量，转换为 JAX 的 float32 类型数组
                bool_masked_pos,  # 布尔掩码，指示哪些位置需要屏蔽
                not train,  # 是否为推断模式（非训练模式）
                output_attentions,  # 是否输出注意力权重
                output_hidden_states,  # 是否输出隐藏状态
                return_dict,  # 是否以字典形式返回输出
                rngs=rngs,  # PRNGs 字典，用于模型中的随机数生成
            )
# 定义了一个用于池化操作的 FlaxBeitPooler 类，继承自 nn.Module
class FlaxBeitPooler(nn.Module):
    # 用于配置的 BeitConfig 对象
    config: BeitConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型

    # 初始化方法
    def setup(self):
        # 如果配置中使用了均值池化
        if self.config.use_mean_pooling:
            # 初始化一个 LayerNorm 层，用于对池化后的输出进行归一化，设定 epsilon 值为配置中的 layer_norm_eps
            self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    # 调用方法，实现池化操作
    def __call__(self, hidden_states):
        # 如果配置中使用了均值池化
        if self.config.use_mean_pooling:
            # 提取除第一个 token 以外的所有 token 的隐藏状态，进行均值池化操作
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(jnp.mean(patch_tokens, axis=1))  # 对所有 token 的均值进行 LayerNorm 归一化
        else:
            # 否则，直接使用第一个 token 的隐藏状态作为池化输出
            pooled_output = hidden_states[:, 0]

        return pooled_output


# 定义了一个 FlaxBeitModule 类，继承自 nn.Module
class FlaxBeitModule(nn.Module):
    # 用于配置的 BeitConfig 对象
    config: BeitConfig
    # 计算中使用的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32  # 计算的数据类型
    # 是否添加池化层，默认为 True
    add_pooling_layer: bool = True

    # 初始化方法
    def setup(self):
        # 初始化嵌入层对象，使用配置和数据类型作为参数
        self.embeddings = FlaxBeitEmbeddings(self.config, dtype=self.dtype)
        # 初始化编码器对象，使用配置、窗口大小和数据类型作为参数
        self.encoder = FlaxBeitEncoder(
            self.config, window_size=self.embeddings.patch_embeddings.patch_shape, dtype=self.dtype
        )
        # 如果不使用均值池化，初始化一个 LayerNorm 层，用于对编码器输出进行归一化，设定 epsilon 值为配置中的 layer_norm_eps
        if not self.config.use_mean_pooling:
            self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        # 如果需要添加池化层，初始化一个 FlaxBeitPooler 对象
        self.pooler = FlaxBeitPooler(self.config, dtype=self.dtype) if self.add_pooling_layer else None

    # 调用方法，实现模型的前向传播
    def __call__(
        self,
        pixel_values,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 使用嵌入层对象，根据输入像素值和掩码位置，获取隐藏状态
        hidden_states = self.embeddings(pixel_values, bool_masked_pos, deterministic=deterministic)

        # 使用编码器对象，对隐藏状态进行编码处理，返回输出对象
        outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的最终隐藏状态作为处理后的隐藏状态
        hidden_states = outputs[0]

        # 如果不使用均值池化，对隐藏状态进行 LayerNorm 归一化处理
        if not self.config.use_mean_pooling:
            hidden_states = self.layernorm(hidden_states)

        # 如果需要添加池化层，对处理后的隐藏状态进行池化操作
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        # 如果不返回字典形式的输出
        if not return_dict:
            # 如果池化结果为空，则不返回池化结果
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            # 否则返回隐藏状态和池化结果以及其他输出
            return (hidden_states, pooled) + outputs[1:]

        # 返回带有池化输出的 FlaxBeitModelOutputWithPooling 对象
        return FlaxBeitModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加文档字符串注释，描述了 FlaxBeitModel 类的基本信息和用法
@add_start_docstrings(
    "The bare Beit Model transformer outputting raw hidden-states without any specific head on top.",
    BEIT_START_DOCSTRING,
)
# 定义了 FlaxBeitModel 类，继承自 FlaxBeitPreTrainedModel
class FlaxBeitModel(FlaxBeitPreTrainedModel):
    # 指定模块的类为 FlaxBeitModule
    module_class = FlaxBeitModule


# 定义了 FLAX_BEIT_MODEL_DOCSTRING，包含返回值和示例信息
FLAX_BEIT_MODEL_DOCSTRING = """
    Returns:

    Examples:

    ```
    # 导入所需的库和模块
    >>> from transformers import AutoImageProcessor, FlaxBeitModel
    >>> from PIL import Image
    >>> import requests
    
    # 定义要处理的图像的 URL
    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 模块获取图像的原始字节流，并使用 PIL 打开图像
    >>> image = Image.open(requests.get(url, stream=True).raw)
    
    # 从预训练模型加载图像处理器（AutoImageProcessor）
    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    # 从预训练模型加载 BEiT 模型（FlaxBeitModel）
    >>> model = FlaxBeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
    
    # 使用图像处理器处理图像，返回 NumPy 张量作为输入
    >>> inputs = image_processor(images=image, return_tensors="np")
    # 使用 BEiT 模型进行推理，输入处理后的图像数据
    >>> outputs = model(**inputs)
    # 获取模型输出中的最后一个隐藏状态
    >>> last_hidden_states = outputs.last_hidden_state
"""

# 调用函数覆盖文档字符串，将 FlaxBeitModel 的文档字符串替换为 FLAX_BEIT_MODEL_DOCSTRING 的内容
overwrite_call_docstring(FlaxBeitModel, FLAX_BEIT_MODEL_DOCSTRING)

# 附加和替换函数返回值的文档字符串，指定输出类型为 FlaxBeitModelOutputWithPooling，配置类为 BeitConfig
append_replace_return_docstrings(FlaxBeitModel, output_type=FlaxBeitModelOutputWithPooling, config_class=BeitConfig)

# FlaxBeitForMaskedImageModelingModule 类定义
class FlaxBeitForMaskedImageModelingModule(nn.Module):
    # 类的配置参数为 BeitConfig 类型
    config: BeitConfig
    # 计算过程中的数据类型，默认为 jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 模块初始化方法
    def setup(self):
        # 创建 FlaxBeitModule 实例，不添加池化层，并使用指定的数据类型
        self.beit = FlaxBeitModule(self.config, add_pooling_layer=False, dtype=self.dtype)

        # 分类器头部初始化
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)  # LayerNorm 初始化
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),  # 使用正态分布初始化权重
            dtype=self.dtype,
        )

    # 对象调用方法
    def __call__(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 如果 return_dict 为 None，则使用配置中指定的 return_dict 参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.beit 对象进行前向计算
        outputs = self.beit(
            pixel_values,
            bool_masked_pos,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的序列输出
        sequence_output = outputs[0]
        # 应用 Layernorm
        sequence_output = self.layernorm(sequence_output)
        # 对序列输出进行预测得分计算，去除第一个位置的特殊标记
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        # 如果不使用 return_dict，则返回预测得分和额外的输出状态
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return output

        # 使用 FlaxMaskedLMOutput 类封装返回结果，包括预测得分、隐藏状态和注意力权重
        return FlaxMaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加文档字符串说明到 FlaxBeitForMaskedImageModeling 类
@add_start_docstrings(
    "Beit Model transformer with a 'language' modeling head on top (to predict visual tokens).",
    BEIT_START_DOCSTRING,
)
class FlaxBeitForMaskedImageModeling(FlaxBeitPreTrainedModel):
    module_class = FlaxBeitForMaskedImageModelingModule


# 定义 FLAX_BEIT_MLM_DOCSTRING，提供 Beit 模型的文档字符串信息
FLAX_BEIT_MLM_DOCSTRING = """
    bool_masked_pos (`numpy.ndarray` of shape `(batch_size, num_patches)`):
        Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

    Returns:
        Beit 模型的输出结果，包含 logits、hidden_states 和 attentions。

    Examples:

    ```
    >>> from transformers import AutoImageProcessor, BeitForMaskedImageModeling
    >>> from PIL import Image
    >>> import requests

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> image_processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
    >>> model = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

    >>> inputs = image_processor(images=image, return_tensors="np")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""
# 使用 overwrite_call_docstring 函数，将 FlaxBeitForMaskedImageModeling 类的文档字符串替换为 FLAX_BEIT_MLM_DOCSTRING 中定义的文档字符串
overwrite_call_docstring(FlaxBeitForMaskedImageModeling, FLAX_BEIT_MLM_DOCSTRING)

# 使用 append_replace_return_docstrings 函数，为 FlaxBeitForMaskedImageModeling 类附加或替换输出类型为 FlaxMaskedLMOutput 和配置类为 BeitConfig 的文档字符串
append_replace_return_docstrings(
    FlaxBeitForMaskedImageModeling, output_type=FlaxMaskedLMOutput, config_class=BeitConfig
)


class FlaxBeitForImageClassificationModule(nn.Module):
    config: BeitConfig
    dtype: jnp.dtype = jnp.float32

    # 设置方法，初始化模块中的 Beit 和分类器组件
    def setup(self):
        self.beit = FlaxBeitModule(config=self.config, dtype=self.dtype, add_pooling_layer=True)
        self.classifier = nn.Dense(
            self.config.num_labels,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )

    # 调用方法，接受多个输入参数，并根据配置返回相应的输出
    def __call__(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 根据配置或者默认设置 return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 self.beit，传递参数并接收输出
        outputs = self.beit(
            pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化后的输出
        pooled_output = outputs[1]
        # 使用分类器计算 logits
        logits = self.classifier(pooled_output)

        # 如果 return_dict 为 False，则返回 logits 和其他输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        # 如果 return_dict 为 True，则返回 FlaxSequenceClassifierOutput 类的实例，包含 logits、hidden_states 和 attentions
        return FlaxSequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 附加或替换 FlaxBeitForImageClassification 类的文档字符串，包括 BEIT_START_DOCSTRING 和从 FLAX_BEIT_CLASSIF_DOCSTRING 中定义的描述
@add_start_docstrings(
    """
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """,
    BEIT_START_DOCSTRING,
)
class FlaxBeitForImageClassification(FlaxBeitPreTrainedModel):
    module_class = FlaxBeitForImageClassificationModule


# 将 FLAX_BEIT_CLASSIF_DOCSTRING 中定义的文档字符串替换为 FlaxBeitForImageClassification 类的文档字符串
overwrite_call_docstring(FlaxBeitForImageClassification, FLAX_BEIT_CLASSIF_DOCSTRING)

# 使用 append_replace_return_docstrings 函数，为 FlaxBeitForImageClassification 类附加或替换输出类型为 FlaxSequenceClassifierOutput 的文档字符串
append_replace_return_docstrings(
    # 导入FlaxBeitForImageClassification类，指定输出类型为FlaxSequenceClassifierOutput，使用BeitConfig配置类
    FlaxBeitForImageClassification, output_type=FlaxSequenceClassifierOutput, config_class=BeitConfig
# 创建一个名为文件的空列表
files = []
# 遍历整数 i 从 0 到 9（不包括 10）
for i in range(10):
    # 向文件列表添加字符串形式的 i
    files.append(f"file_{i}.txt")
```