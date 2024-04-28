# `.\models\gptj\modeling_flax_gptj.py`

```
# 设置文件编码格式为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 官网获取许可证的副本
# 如果未有适用法律或书面同意的要求，不得使用本文件
# 在"AS IS"基础上分发，不附带任何明示或暗示的担保或条件
# 查看特定语言的许可证授权权限和限制
# 导入所需的库和模块
# 从 functools 模块导入 partial 函数
# 从 typing 模块导入 Optional 和 Tuple 类型
# 导入第三方库 flax.linen
# 导入 JAX 库
# 导入 JAX 的 numpy 模块并命名为 jnp
# 导入 numpy 库并命名为 np
# 导入 flax.core.frozen_dict 模块中的 FrozenDict, freeze, unfreeze 函数
# 导入 flax.linen 模块
# 导入 flax.linen.attention 模块中的 dot_product_attention_weights 函数
# 导入 flax.traverse_util 模块中的 flatten_dict, unflatten_dict 函数
# 导入 jax 库中的 lax 模块
# 导入 modeling_flax_outputs 模块中的 FlaxBaseModelOutput, FlaxCausalLMOutput 类
# 导入 modeling_flax_utils 模块中的 ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring 函数
# 导入工具包中的 add_start_docstrings, add_start_docstrings_to_model_forward, logging 函数
# 从配置文件中导入 GPTJConfig 类
# 导入 logging 模块中的 logger 对象

# 用于文档的示例检查点
_CHECKPOINT_FOR_DOC = "gptj"
# 用于文档的配置示例
_CONFIG_FOR_DOC = "GPTJConfig"
# GPTJ 模型的文档起始字符串
GPTJ_START_DOCSTRING = r"""

    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.

    Finally, this model supports inherent JAX features such as:

    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
"""
    def __init__(self, config: Union[PretrainedConfig, Dict[str, Any]], dtype: jnp.dtype = jnp.float32):
        """
        初始化函数，接受一个模型配置类及所有模型参数。
        初始化一个配置文件不会加载模型关联的权重，只会加载配置信息。
        查看 `~FlaxPreTrainedModel.from_pretrained` 方法以加载模型权重。
    
        Parameters:
            config ([`GPTJConfig`]): 包含模型所有参数的模型配置类。
                使用配置文件初始化不会加载模型关联的权重，只会加载配置信息。
                查看 `~FlaxPreTrainedModel.from_pretrained` 方法以加载模型权重。
            dtype (`jax.numpy.dtype`, *可选*, 默认为 `jax.numpy.float32`):
                计算的数据类型。可以是 `jax.numpy.float32`, `jax.numpy.float16`（在GPU上）和 `jax.numpy.bfloat16`（在TPU上）中的一种。
    
                这可以用于在GPU或TPU上启用混合精度训练或半精度推断。如果指定了dtype，则所有计算将使用给定的 `dtype` 进行。
    
                **请注意，这仅指定了计算的dtype，不影响模型参数的dtype。**
    
                如果您希望更改模型参数的dtype，请参阅 [`~FlaxPreTrainedModel.to_fp16`] 和 [`~FlaxPreTrainedModel.to_bf16`]。
        """
"""
# GPTJ_INPUTS_DOCSTRING是一个包含参数说明的文档字符串

# 创建一个函数，用于生成正弦位置编码
def create_sinusoidal_positions(num_pos, dim):
    # 计算正弦位置编码的频率
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    # 生成正弦位置编码
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    # 设置一个标志，用于确定sin和cos的维度
    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    # 将sin和cos分配给输出张量
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out)

# 创建一个函数，用于旋转张量的每两个元素
def rotate_every_two(tensor):
    rotate_half_tensor = jnp.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1)
    rotate_half_tensor = rotate_half_tensor.reshape(rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor

# 创建一个函数，用于应用旋转位置编码
def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    # 计算旋转后的位置编码
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)

# 创建一个名为FlaxGPTJAttention的类，继承自nn.Module
class FlaxGPTJAttention(nn.Module):
    # GPTJConfig配置属性
    config: GPTJConfig
    # 数据类型
    dtype: jnp.dtype = jnp.float32
    # 定义 causal 变量，类型为布尔值，初始值为 True
    causal: bool = True
    # 定义 is_cross_attention 变量，类型为布尔值，初始值为 False
    is_cross_attention: bool = False

    # 定义 setup 方法
    def setup(self):
        # 获取配置信息
        config = self.config
        # 初始化 embed_dim 变量为配置中的 hidden_size
        self.embed_dim = config.hidden_size
        # 初始化 num_heads 变量为配置中的 num_attention_heads
        self.num_heads = config.num_attention_heads
        # 初始化 head_dim 变量为 embed_dim 除以 num_heads
        self.head_dim = self.embed_dim // self.num_heads

        # 初始化 rotary_dim 变量为配置中的 rotary_dim
        self.rotary_dim = config.rotary_dim

        # 创建部分函数 dense，参数为 nn.Dense
        dense = partial(
            nn.Dense,
            self.embed_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        # 初始化 q_proj、k_proj、v_proj 为 dense 函数的返回值
        self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
        # 初始化 out_proj 为 dense 函数的返回值
        self.out_proj = dense()

        # 初始化 resid_dropout 为 nn.Dropout，丢弃率为配置中的 resid_pdrop
        self.resid_dropout = nn.Dropout(rate=config.resid_pdrop)

        # 创建一个因果关系遮罩，大小为 (1, max_position_embeddings)，数据类型为布尔值
        self.causal_mask = make_causal_mask(jnp.ones((1, config.max_position_embeddings), dtype="bool"), dtype="bool")

        # 初始化 pos_embd_dim 为 rotary_dim 或 embed_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        # 初始化 embed_positions 为 create_sinusoidal_positions 函数的返回值
        self.embed_positions = create_sinusoidal_positions(config.max_position_embeddings, pos_embd_dim)

    # 定义 _split_heads 方法，参数为 hidden_states
    def _split_heads(self, hidden_states):
        # 将 hidden_states 重新调整为原来维度的形状加上头数和头维度
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_heads, self.head_dim))

    # 定义 _merge_heads 方法，参数为 hidden_states
    def _merge_heads(self, hidden_states):
        # 将 hidden_states 重新调整为原来维度的形状加上 embed_dim
        return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

    # 定义一个装饰器为 nn.compact
    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        """
        This function takes projected key, value states from a single input token and concatenates the states to cached
        states from previous steps. This function is slightly adapted from the official Flax repository:
        https://github.com/google/flax/blob/491ce18759622506588784b4fca0e4bf05f8c8cd/flax/linen/attention.py#L252
        """
        # 检测是否通过缺少现有缓存数据来进行初始化。
        is_initialized = self.has_variable("cache", "cached_key")
        # 获取缓存的键状态，并初始化为零矩阵，如果未初始化的话。
        cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
        # 获取缓存的值状态，并初始化为零矩阵，如果未初始化的话。
        cached_value = self.variable("cache", "cached_value", jnp.zeros, value.shape, value.dtype)
        # 获取缓存索引，并初始化为0。
        cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            # 提取批次维度、最大长度、注意力头数和每个头的深度
            *batch_dims, max_length, num_heads, depth_per_head = cached_key.value.shape
            # 更新键、值缓存，使用新的一维空间切片
            cur_index = cache_index.value
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cached_key.value, key, indices)
            value = lax.dynamic_update_slice(cached_value.value, value, indices)
            cached_key.value = key
            cached_value.value = value
            # 更新缓存向量的数量
            num_updated_cache_vectors = query.shape[1]
            cache_index.value = cache_index.value + num_updated_cache_vectors
            # 用于缓存的因果掩码：我们的单个查询位置应该仅关注已经生成和缓存的键位置，而不是剩余的零元素。
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            # 合并填充掩码和输入的注意力掩码
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
class FlaxGPTJMLP(nn.Module):
    # 定义了一个 FlaxGPTJMLP 类，继承自 nn.Module
    config: GPTJConfig
    # 设置类属性 config 为 GPTJConfig 类型
    intermediate_size: int
    # 设置类属性 intermediate_size 为整数类型
    dtype: jnp.dtype = jnp.float32
    # 设置类属性 dtype，默认为 jnp.float32

    def setup(self):
        # 定义类方法 setup
        embed_dim = self.config.hidden_size
        # 从 config 中获取 hidden_size 并赋值给 embed_dim
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        # 使用 jax.nn.initializers.normal 函数初始化 kernel

        self.fc_in = nn.Dense(self.intermediate_size, dtype=self.dtype, kernel_init=kernel_init)
        # 创建一个全连接层对象，并赋值给实例属性 fc_in
        self.fc_out = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init)
        # 创建一个全连接层对象，并赋值给实例属性 fc_out

        self.act = ACT2FN[self.config.activation_function]
        # 从 ACT2FN 字典中获取激活函数，并赋值给实例属性 act
        self.dropout = nn.Dropout(rate=self.config.resid_pdrop)
        # 创建一个 Dropout 层对象，并赋值给实例属性 dropout

    def __call__(self, hidden_states, deterministic: bool = True):
        # 定义魔法方法 __call__
        hidden_states = self.fc_in(hidden_states)
        # 调用全连接层对象 fc_in 处理输入 hidden_states
        hidden_states = self.act(hidden_states)
        # 对结果应用激活函数
        hidden_states = self.fc_out(hidden_states)
        # 调用全连接层对象 fc_out 处理结果
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # 对结果应用 Dropout 层
        return hidden_states
        # 返回处理后的 hidden_states


class FlaxGPTJBlock(nn.Module):
    # 定义了一个 FlaxGPTJBlock 类，继承自 nn.Module
    config: GPTJConfig
    # 设置类属性 config 为 GPTJConfig 类型
    dtype: jnp.dtype = jnp.float32
    # 设置类属性 dtype，默认为 jnp.float32

    def setup(self):
        # 定义类方法 setup
        hidden_size = self.config.hidden_size
        # 从 config 里获取 hidden_size 并赋值给 hidden_size
        inner_dim = self.config.n_inner if self.config.n_inner is not None else 4 * hidden_size
        # 判断 config 中是否有 n_inner，没有则设为 4 倍 hidden_size

        self.ln_1 = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)
        # 创建一个 LayerNorm 层对象，并赋值给实例属性 ln_1
        self.attn = FlaxGPTJAttention(self.config, dtype=self.dtype)
        # 创建一个 FlaxGPTJAttention 层对象，并赋值给实例属性 attn

        self.mlp = FlaxGPTJMLP(self.config, inner_dim, dtype=self.dtype)
        # 创建一个 FlaxGPTJMLP 实例，并赋值给实例属性 mlp

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        # 定义魔法方法 __call__
        residual = hidden_states
        # 将输入赋值给 residual，用于残差连接
        hidden_states = self.ln_1(hidden_states)
        # 调用 LayerNorm 层处理 hidden_states
        attn_outputs = self.attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
        )
        # 调用注意力层对象处理 hidden_states
        attn_output = attn_outputs[0]
        # 获取注意力层输出结果

        feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        # 调用 MLP 层对象处理 hidden_states
        hidden_states = attn_output + feed_forward_hidden_states + residual
        # 残差连接

        return (hidden_states,) + attn_outputs[1:]
        # 返回处理后的 hidden_states 和额外输出信息


class FlaxGPTJPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTJConfig
    # 设置类属性 config_class 为 GPTJConfig 类型
    base_model_prefix = "transformer"
    # 设置类属性 base_model_prefix 为 "transformer"
    module_class: nn.Module = None
    # 设置类属性 module_class 为 nn.Module 类型，默认为 None

    def __init__(
        self,
        config: GPTJConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        # 初始化方法
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        # 使用 module_class 创建一个模型实例
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)
        # 调用父类的初始化方法
    # 初始化模型的参数权重
    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # 初始化输入张量
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_shape)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        # 如果模型需要跨注意力头，则初始化编码器隐藏状态和注意力掩码
        if self.config.add_cross_attention:
            encoder_hidden_states = jnp.zeros(input_shape + (self.config.n_embd,))
            encoder_attention_mask = attention_mask
            module_init_outputs = self.module.init(
                rngs,
                input_ids,
                attention_mask,
                position_ids,
                encoder_hidden_states,
                encoder_attention_mask,
                return_dict=False,
            )
        else:
            module_init_outputs = self.module.init(rngs, input_ids, attention_mask, position_ids, return_dict=False)

        random_params = module_init_outputs["params"]

        # 如果提供了预训练参数，则使用预训练参数，否则使用随机初始化的参数
        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    # 初始化缓存以用于快速自回归解码
    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                用于快速自回归解码的批大小。定义了初始化缓存时的批大小。
            max_length (`int`):
                自回归解码的最大可能长度。定义了初始化缓存时的序列长度。
        """
        # 初始化用于检索缓存的输入变量
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)
        position_ids = jnp.broadcast_to(jnp.arange(jnp.atleast_2d(input_ids).shape[-1]), input_ids.shape)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, position_ids, return_dict=False, init_cache=True
        )
        return init_variables["cache"]

    # 模型的调用函数，对输入进行前向计算
    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        params: dict = None,
        past_key_values: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 检查是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏层状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否需要返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # 获取输入的 batch_size 和 sequence_length
        batch_size, sequence_length = input_ids.shape

        # 如果未提供位置信息且存在过去的 key-value，则抛出数值错误
        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("Make sure to provide `position_ids` when passing `past_key_values`.")

            # 根据 sequence_length 广播生成位置信息
            position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        # 如果未提供注意力掩码，则生成全 1 注意力掩码
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # 如果需要处理任何 PRNG
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        # 将参数添加到输入中
        inputs = {"params": params or self.params}

        # 如果存在过去的 key-values，则将其添加到输入中，并标记为可变
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        # 应用模块，生成输出
        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            jnp.array(position_ids, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # 将更新后的 cache 添加到模型输出
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        # 返回输出
        return outputs
# 定义一个名为FlaxGPTJBlockCollection的类，继承自nn.Module类
class FlaxGPTJBlockCollection(nn.Module):
    # 定义类属性config，类型为GPTJConfig
    config: GPTJConfig
    # 定义类属性dtype，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义初始化方法setup，用于设置模型
    def setup(self):
        # 创建self.blocks列表，其中包含self.config.num_hidden_layers个FlaxGPTJBlock对象
        self.blocks = [
            FlaxGPTJBlock(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
        ]

    # 定义__call__方法，用于模型的调用
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 如果需要输出attentions，则初始化一个空tuple变量all_attentions
        all_attentions = () if output_attentions else None
        # 如果需要输出hidden_states，则初始化一个空tuple变量all_hidden_states
        all_hidden_states = () if output_hidden_states else None

        # 遍历self.blocks列表
        for block in self.blocks:
            # 如果需要输出hidden_states，则将当前hidden_states加入all_hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 对当前block进行调用，获取层级输出
            layer_outputs = block(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
            )
            # 更新hidden_states为当前block的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出attentions，则将当前层的attentions加入all_attentions
            if output_attentions:
                all_attentions += (layer_outputs[1],)

        # 构建输出结果，包括hidden_states、all_hidden_states和all_attentions
        # 这里可能包含了None值，会在FlaxGPTJModule中进行过滤处理
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


# 定义一个名为FlaxGPTJModule的类，继承自nn.Module类
class FlaxGPTJModule(nn.Module):
    # 定义类属性config，类型为GPTJConfig
    config: GPTJConfig
    # 定义类属性dtype，默认为jnp.float32
    dtype: jnp.dtype = jnp.float32

    # 定义初始化方法setup，用于设置模型
    def setup(self):
        # 设置self.embed_dim为self.config.hidden_size
        self.embed_dim = self.config.hidden_size
        # 初始化词嵌入self.wte，将词汇量、隐藏层大小等作为参数传入
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )
        # 初始化dropout层self.dropout，将config.embd_pdrop作为参数传入
        self.dropout = nn.Dropout(rate=self.config.embd_pdrop)
        # 初始化GPTJBlockCollection层self.h，将self.config和dtype作为参数传入
        self.h = FlaxGPTJBlockCollection(self.config, dtype=self.dtype)
        # 初始化LayerNorm层self.ln_f，将config.layer_norm_epsilon和dtype作为参数传入
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    # 定义__call__方法，用于模型的调用
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 将输入的ids转换为i4类型并嵌入
        input_embeds = self.wte(input_ids.astype("i4"))

        # 对嵌入的输入进行dropout处理
        hidden_states = self.dropout(input_embeds, deterministic=deterministic)

        # 使用Transformer模型处理隐藏状态（hidden_states），包括传入注意力掩码、位置ids等参数
        outputs = self.h(
            hidden_states,
            attention_mask,
            position_ids=position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取处理后的隐藏状态
        hidden_states = outputs[0]
        # 对最终的隐藏状态进行Layer Normalization操作
        hidden_states = self.ln_f(hidden_states)

        # 如果需要输出所有隐藏状态，则将处理后的隐藏状态与之前的隐藏状态存入all_hidden_states
        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 如果不需要返回dict，则返回outputs中非None的部分
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        # 返回FlaxBaseModelOutput对象，包括最终的隐藏状态、所有隐藏状态和注意力的结果
        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )
# 添加起始文档字符串到 FlaxGPTJModel 类
@add_start_docstrings(
    "The bare GPTJ Model transformer outputting raw hidden-states without any specific head on top.", # 添加起始文档字符串描述模型输出原始隐藏状态，没有特定的头部
    GPTJ_START_DOCSTRING,
)
class FlaxGPTJModel(FlaxGPTJPreTrainedModel):
    module_class = FlaxGPTJModule # 设置模型类为 FlaxGPTJModule


# 添加调用示例文档字符串到 FlaxGPTJModel 类
append_call_sample_docstring(
    FlaxGPTJModel,
    _CHECKPOINT_FOR_DOC,
    FlaxCausalLMOutput,
    _CONFIG_FOR_DOC,
)


# 定义用于 Causal Language Modeling 的 FlaxGPTJForCausalLMModule 类
class FlaxGPTJForCausalLMModule(nn.Module):
    config: GPTJConfig # 配置参数为 GPTJConfig 类型
    dtype: jnp.dtype = jnp.float32 # 数据类型默认为 jnp.float32

    def setup(self):
        # 设置模型的 Transformer 和语言模型头部
        self.transformer = FlaxGPTJModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size, # 输出维度为词汇表大小
            dtype=self.dtype, # 使用指定数据类型
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range), # 使用正态分布初始化权重
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 调用 Transformer 模型
        outputs = self.transformer(
            input_ids,
            attention_mask,
            position_ids,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] # 获取隐藏状态

        if self.config.tie_word_embeddings:
            # 如果词嵌入被绑定
            shared_kernel = self.transformer.variables["params"]["wte"]["embedding"].T # 获取共享的词嵌入权重矩阵
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states) # 应用共享的词嵌入权重到 LM 头部
        else:
            lm_logits = self.lm_head(hidden_states) # 应用普通的 LM 头部

        if not return_dict:
            return (lm_logits,) + outputs[1:] # 返回 logits 和可能的额外输出

        # 返回 CausalLMOutput 对象
        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 添加起始文档字符串到 FlaxGPTJForCausalLM 类
@add_start_docstrings(
    """
    The GPTJ Model transformer with a language modeling head on top.
    """,
    GPTJ_START_DOCSTRING,
)
class FlaxGPTJForCausalLM(FlaxGPTJPreTrainedModel):
    module_class = FlaxGPTJForCausalLMModule # 设置模型类为 FlaxGPTJForCausalLMModule
    # 为生成准备输入数据，初始化缓存
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jax.Array] = None):
        # 获取输入数据的批量大小和序列长度
        batch_size, seq_length = input_ids.shape
        # 初始化缓存
        past_key_values = self.init_cache(batch_size, max_length)
        
        # 注意：通常情况下，需要为 attention_mask 中 x > input_ids.shape[-1] 和 x < cache_length 的位置放入 0
        # 但由于 GPTJ 使用的是因果关系掩码，这些位置已经被掩盖了
        # 因此我们可以在这里创建一个静态的 attention_mask，这样更有效率
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            # 根据 attention_mask 计算位置
            position_ids = attention_mask.cumsum(axis=-1) - 1
            # 更新 extended_attention_mask 的部分值为 attention_mask
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
        else:
            # 如果没有传入 attention_mask，则使用默认的位置信息
            position_ids = jnp.broadcast_to(jnp.arange(seq_length, dtype="i4")[None, :], (batch_size, seq_length))

        # 返回输入数据字典
        return {
            "past_key_values": past_key_values,  # 过去的键值对
            "attention_mask": extended_attention_mask,  # 注意力掩码
            "position_ids": position_ids,  # 位置信息
        }

    # 更新生成时的输入数据
    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        # 更新键值对和位置信息
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        model_kwargs["position_ids"] = model_kwargs["position_ids"][:, -1:] + 1
        return model_kwargs  # 返回更新后的输入数据
# 调用函数append_call_sample_docstring，传入参数(FlaxGPTJForCausalLM, _CHECKPOINT_FOR_DOC, FlaxCausalLMOutput, _CONFIG_FOR_DOC)
# 参数1: FlaxGPTJForCausalLM，表示要添加文档字符串的函数或类
# 参数2: _CHECKPOINT_FOR_DOC，表示用于示例的检查点名称
# 参数3: FlaxCausalLMOutput，表示函数或类的输出类型
# 参数4: _CONFIG_FOR_DOC，表示用于示例的配置名称
```