# `.\transformers\models\t5\modeling_tf_t5.py`

```
# 设置编码
# 版权声明
# 获取许可证
# 引入类型注解
# 引入警告模块
# 引入 numpy 模块
# 引入 tensorflow 模块
# 引入 dynamic_slice 方法
# 引入 TFBaseModelOutput, TFBaseModelOutputWithPastAndCrossAttentions, TFSeq2SeqLMOutput, TFSeq2SeqModelOutput 类型
# 引入模型相关计算的辅助方法
# 引入 T5Config 类
# 获取日志处理器
# T5 预训练模型的存档列表
# 定义 T5LayerNorm 类
    # 初始化方法
        # 构建一个 T5 风格的 layernorm 模块，不计算偏差和平均值的差值
    # 构建方法
        # 构建共享的词嵌入层
    # 调用方法
        # 计算隐藏状态的方差
        # 根据方差调整隐藏状态
class TFT5DenseActDense(tf.keras.layers.Layer):
    # 定义类TFT5DenseActDense，继承自tf.keras.layers.Layer
    def __init__(self, config, **kwargs):
        # 初始化函数，接受config作为参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        wi_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model**-0.5)
        )
        # 创建一个RandomNormal类型的初始化器wi_initializer
        wo_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff**-0.5)
        )
        # 创建一个RandomNormal类型的初始化器wo_initializer
        self.wi = tf.keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        # 创建一个全连接层，不使用偏置，名为wi，使用wi_initializer初始化
        self.wo = tf.keras.layers.Dense(
            config.d_model, use_bias=False, name="wo", kernel_initializer=wo_initializer
        )  # Update init weights as in flax
        # 创建一个全连接层，不使用偏置，名为wo，使用wo_initializer初始化
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        # 创建一个Dropout层，使用给定的dropout率
        self.act = get_tf_activation(config.dense_act_fn)
        # 调用get_tf_activation函数获取激活函数配置
        self.config = config
        # 将config参数保存到类属性中

    def call(self, hidden_states, training=False):
        # 定义类的调用函数，传入隐藏状态hidden_states和训练标志training
        hidden_states = self.wi(hidden_states)
        # 对隐藏状态进行wi的全连接操作
        hidden_states = self.act(hidden_states)
        # 使用激活函数处理隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)
        # 对处理后的隐藏状态进行Dropout操作
        hidden_states = self.wo(hidden_states)
        # 对Dropout后的隐藏状态进行wo的全连接操作
        return hidden_states
        # 返回处理后的隐藏状态

    def build(self, input_shape=None):
        # 定义模型的构建函数，接受输入形状input_shape作为参数，默认为None
        if self.built:
            return
        # 如果模型已经构建好，则直接返回
        self.built = True
        # 将模型的built属性设置为True
        if getattr(self, "wi", None) is not None:
            # 如果wi存在
            with tf.name_scope(self.wi.name):
                # 在TensorFlow的命名域中使用wi的名称
                self.wi.build([None, None, self.config.d_model])
                # 构建wi模型，���定输入形状
        if getattr(self, "wo", None) is not None:
            # 如果wo存在
            with tf.name_scope(self.wo.name):
                # 在TensorFlow的命名域中使用wo的名称
                self.wo.build([None, None, self.config.d_ff])
                # 构建wo模型，指定输入形状

class TFT5DenseGatedActDense(tf.keras.layers.Layer):
    # 定义类TFT5DenseGatedActDense，继承自tf.keras.layers.Layer
    def __init__(self, config, **kwargs):
        # 初始化函数，接受config作为参数
        super().__init__(**kwargs)
        # 调用父类的初始化函数
        wi_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_model**-0.5)
        )
        # 创建一个RandomNormal类型的初始化器wi_initializer
        wo_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (config.d_ff**-0.5)
        )
        # 创建一个RandomNormal类型的初始化器wo_initializer
        self.wi_0 = tf.keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi_0", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        # 创建一个全连接层，不使用偏置，名为wi_0，使用wi_initializer初始化
        self.wi_1 = tf.keras.layers.Dense(
            config.d_ff, use_bias=False, name="wi_1", kernel_initializer=wi_initializer
        )  # Update init weights as in flax
        # 创建一个全连接层，不使用偏置，名为wi_1，使用wi_initializer初始化
        self.wo = tf.keras.layers.Dense(
            config.d_model, use_bias=False, name="wo", kernel_initializer=wo_initializer
        )  # Update init weights as in flax
        # 创建一个全连接层，不使用偏置，名为wo，使用wo_initializer初始化
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
        # 创建一个Dropout层，使用给定的dropout率
        self.act = get_tf_activation(config.dense_act_fn)
        # 调用get_tf_activation函数获取激活函数配置
        self.config = config
        # 将config参数保存到类属性中
    # 定义神经网络模型中的调用方法，用于前向传播
    def call(self, hidden_states, training=False):
        # 对隐藏状态应用激活函数，并将结果赋值给hidden_gelu
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 对隐藏状态进行线性变换，并将结果赋值给hidden_linear
        hidden_linear = self.wi_1(hidden_states)
        # 将激活函数和线性变换的结果相乘，得到新的隐藏状态
        hidden_states = hidden_gelu * hidden_linear
        # 根据训练状态进行dropout操作，防止过拟合
        hidden_states = self.dropout(hidden_states, training=training)
        # 对隐藏状态进行另一个线性变换
        hidden_states = self.wo(hidden_states)
        # 返回最终的隐藏状态
        return hidden_states

    # 构建神经网络模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果wi_0存在，则构建wi_0
        if getattr(self, "wi_0", None) is not None:
            with tf.name_scope(self.wi_0.name):
                # 构建wi_0并指定输入形状
                self.wi_0.build([None, None, self.config.d_model])
        # 如果wi_1存在，则构建wi_1
        if getattr(self, "wi_1", None) is not None:
            with tf.name_scope(self.wi_1.name):
                # 构建wi_1并指定输入形状
                self.wi_1.build([None, None, self.config.d_model])
        # 如果wo存在，则构建wo
        if getattr(self, "wo", None) is not None:
            with tf.name_scope(self.wo.name):
                # 构建wo并指定输入形状
                self.wo.build([None, None, self.config.d_ff])
# 定义一个名为TFT5LayerFF的自定义层，继承于tf.keras.layers.Layer类
class TFT5LayerFF(tf.keras.layers.Layer):
    # 初始化方法
    def __init__(self, config, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 如果config中包含门控激活函数的标志，则创建TFT5DenseGatedActDense对象，否则创建TFT5DenseActDense对象，并赋值给DenseReluDense属性
        if config.is_gated_act:
            self.DenseReluDense = TFT5DenseGatedActDense(config, name="DenseReluDense")
        else:
            self.DenseReluDense = TFT5DenseActDense(config, name="DenseReluDense")
        # 创建TFT5LayerNorm对象，并赋值给layer_norm属性
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name="layer_norm")
        # 创建tf.keras.layers.Dropout对象，并赋值给dropout属性
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    # 调用方法
    def call(self, hidden_states, training=False):
        # 对输入的hidden_states进行layer normalization，得到normed_hidden_states
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将normed_hidden_states作为输入，调用DenseReluDense对象的call方法，得到dense_output
        dense_output = self.DenseReluDense(normed_hidden_states, training=training)
        # 计算新的hidden_states，并执行dropout操作
        hidden_states = hidden_states + self.dropout(dense_output, training=training)
        # 返回计算得到的新的hidden_states
        return hidden_states

    # 构建方法
    def build(self, input_shape=None):
        # 如果已经构建过了，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        # 如果layer_norm存在，则构建layer_norm
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)
        # 如果DenseReluDense存在，则构建DenseReluDense
        if getattr(self, "DenseReluDense", None) is not None:
            with tf.name_scope(self.DenseReluDense.name):
                self.DenseReluDense.build(None)

# 定义一个名为TFT5Attention的自定义层，继承于tf.keras.layers.Layer类
class TFT5Attention(tf.keras.layers.Layer):
    # 创建一个新的计数器对象
    NEW_ID = itertools.count()
    # 这是 TFT5Attention 类的构造函数
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 获取一个新的层 ID
        self.layer_id = next(TFT5Attention.NEW_ID)
        # 设置是否为解码器层
        self.is_decoder = config.is_decoder
        # 设置是否使用缓存
        self.use_cache = config.use_cache
        # 设置是否有相对注意力偏置
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置是否输出注意力权重
        self.output_attentions = config.output_attentions
    
        # 设置相对注意力偏置的超参数
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 设置模型的维度大小
        self.d_model = config.d_model
        # 设置 key 和 value 的投影维度
        self.key_value_proj_dim = config.d_kv
        # 设置注意力头的数量
        self.n_heads = config.num_heads
        # 计算每个注意力头的内部维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim
    
        # 使用 Mesh TensorFlow 的权重初始化方式
        # 初始化 query 层的权重
        q_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * ((self.inner_dim * self.key_value_proj_dim) ** -0.5)
        )
        # 初始化 key 层的权重
        k_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        # 初始化 value 层的权重
        v_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        # 初始化输出层的权重
        o_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
        # 初始化相对注意力偏置的权重
        self.relative_attention_bias_initializer = tf.keras.initializers.RandomNormal(
            mean=0, stddev=config.initializer_factor * (self.inner_dim**-0.5)
        )
    
        # 创建 query 层
        self.q = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="q", kernel_initializer=q_initializer
        )
        # 创建 key 层
        self.k = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="k", kernel_initializer=k_initializer
        )
        # 创建 value 层
        self.v = tf.keras.layers.Dense(
            self.inner_dim, use_bias=False, name="v", kernel_initializer=v_initializer
        )
        # 创建输出层
        self.o = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="o", kernel_initializer=o_initializer
        )
        # 创建dropout层
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)
    
        # 初始化一个空的剪枝头集合
        self.pruned_heads = set()
    # 构建自注意力层模型，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果存在相对注意力偏置
        if self.has_relative_attention_bias:
            # 使用命名空间创建相对注意力偏置
            with tf.name_scope("relative_attention_bias"):
                self.relative_attention_bias = self.add_weight(
                    name="embeddings",
                    shape=[self.relative_attention_num_buckets, self.n_heads],
                    initializer=self.relative_attention_bias_initializer,  # 添加初始化器
                )
        # 如果存在查询向量q
        if getattr(self, "q", None) is not None:
            with tf.name_scope(self.q.name):
                self.q.build([None, None, self.d_model])
        # 如果存在键向量k
        if getattr(self, "k", None) is not None:
            with tf.name_scope(self.k.name):
                self.k.build([None, None, self.d_model])
        # 如果存在数值向量v
        if getattr(self, "v", None) is not None:
            with tf.name_scope(self.v.name):
                self.v.build([None, None, self.d_model])
        # 如果存在输出向量o
        if getattr(self, "o", None) is not None:
            with tf.name_scope(self.o.name):
                self.o.build([None, None, self.inner_dim])

    # 剪枝头部
    def prune_heads(self, heads):
        raise NotImplementedError

    # 静态方法
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        # Calculate the relative buckets for relative attention
        # Adjust number of buckets if bidirectional
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (
                tf.cast(tf.math.greater(relative_position, 0), dtype=relative_position.dtype) * num_buckets
            )
            relative_position = tf.math.abs(relative_position)
        else:
            relative_position = -tf.math.minimum(relative_position, 0)
        # Determine position range and adjust for large relative positions
        max_exact = num_buckets // 2
        is_small = tf.math.less(relative_position, max_exact)
        relative_position_if_large = max_exact + tf.cast(
            tf.math.log(tf.cast(relative_position, tf.float32) / tf.cast(max_exact, tf.float32))
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        relative_position_if_large = tf.math.minimum(relative_position_if_large, num_buckets - 1)
        relative_buckets += tf.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
        def compute_bias(self, query_length, key_length):
            """Compute binned relative position bias"""
            # 创建一个表示查询长度的一维张量
            context_position = tf.range(query_length)[:, None]
            # 创建一个表示关键长度的一维张量
            memory_position = tf.range(key_length)[None, :]
            # 计算相对位置，得到一个二维张量，其形状为(query_length, key_length)
            relative_position = memory_position - context_position
            # 将相对位置进行分桶处理，并根据参数配置，得到一个表示相对位置分桶的张量
            relative_position_bucket = self._relative_position_bucket(
                relative_position,
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            # 从相对位置分桶张量中获取对应位置的相对注意力偏置值，形状为(query_length, key_length, num_heads)
            values = tf.gather(
                self.relative_attention_bias, relative_position_bucket
            )
            # 将获取的相对注意力偏置值进行形状变换，变换为(1, num_heads, query_length, key_length)
            values = tf.expand_dims(
                tf.transpose(values, [2, 0, 1]), axis=0
            )
            # 返回计算得到的相对位置偏置值
            return values

        def call(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            training=False,
            output_attentions=False,
class TFT5LayerSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数初始化对象
        # 创建自注意力层对象，使用给定的配置和是否具有相对注意偏置参数
        self.SelfAttention = TFT5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            name="SelfAttention",
        )
        # 创建层归一化对象，使用给定的隐藏层维度和层归一化参数
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name="layer_norm")
        # 创建丢弃层对象，使用给定的丢弃率参数
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        # 对隐藏状态进行归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用自注意力层对象，传入归一化后的隐藏状态和其他参数
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 更新隐藏状态，加上自注意力输出经过丢弃层的结果
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        # 构建输出元组，包含更新后的隐藏状态和注意力输出（如果有的话）
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果已经构建了自注意力层对象，则构建它
        if getattr(self, "SelfAttention", None) is not None:
            with tf.name_scope(self.SelfAttention.name):
                self.SelfAttention.build(None)
        # 如果已经构建了层归一化对象，则构建它
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)


class TFT5LayerCrossAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)  # 调用父类构造函数初始化对象
        # 创建编码-解码注意力层对象，使用给定的配置
        self.EncDecAttention = TFT5Attention(
            config,
            has_relative_attention_bias=False,
            name="EncDecAttention",
        )
        # 创建层归一化对象，使用给定的隐藏层维度和层归一化参数
        self.layer_norm = TFT5LayerNorm(config.d_model, epsilon=config.layer_norm_epsilon, name="layer_norm")
        # 创建丢弃层对象，使用给定的丢弃率参数
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def call(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        training=False,
    ):
        # 调用编码-解码注意力层对象，传入隐藏状态和键值状态以及其他参数
        attention_output = self.EncDecAttention(
            hidden_states,
            key_value_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 对编码-解码注意力输出经过丢弃层并更新隐藏状态
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        # 构建输出元组，包含更新后的隐藏状态和注意力输出（如果有的话）
        outputs = (hidden_states,) + attention_output[1:]
        return outputs
    ):
        # 将输入的隐藏状态进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 使用编码-解码注意力机制
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            query_length=query_length,
            use_cache=use_cache,
            output_attentions=output_attentions,
            training=training,
        )
        # 将原始的隐藏状态与注意力机制的输出相加，并进行dropout
        hidden_states = hidden_states + self.dropout(attention_output[0], training=training)
        # 如果输出注意力机制，则在输出中添加注意力机制
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在编码-解码注意力机制，则构建它
        if getattr(self, "EncDecAttention", None) is not None:
            with tf.name_scope(self.EncDecAttention.name):
                self.EncDecAttention.build(None)
        # 如果存在层归一化，则构建它
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build(None)
class TFT5Block(tf.keras.layers.Layer):
    # 定义TFT5Block类，继承自tf.keras.layers.Layer
    def __init__(self, config, has_relative_attention_bias=False, **kwargs):
        super().__init__(**kwargs)
        # 初始化函数，设置TFT5Block对象的属性
        self.is_decoder = config.is_decoder
        self.layer = []
        self.layer.append(
            # 向layer列表中添加TFT5LayerSelfAttention层
            TFT5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                name="layer_._0",
            )
        )
        if self.is_decoder:
            self.layer.append(
                # 向layer列表中添加TFT5LayerCrossAttention层（若是decoder）
                TFT5LayerCrossAttention(
                    config,
                    name="layer_._1",
                )
            )

        # 向layer列表中添加TFT5LayerFF层
        self.layer.append(TFT5LayerFF(config, name=f"layer_._{len(self.layer)}"))

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        encoder_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        training=False,
        # 定义TFT5Block类的call方法，其中包含多个参数

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        for layer_module in self.layer:
            if hasattr(layer_module, "name"):
                with tf.name_scope(layer_module.name):
                    layer_module.build(None)
        # 构建TFT5Block类

####################################################
# The full model without a specific pretrained or finetuning head is
# provided as a tf.keras.layers.Layer usually called "TFT5MainLayer"
####################################################
@keras_serializable
class TFT5MainLayer(tf.keras.layers.Layer):
    # 定义TFT5MainLayer类，继承自tf.keras.layers.Layer
    config_class = T5Config

    def __init__(self, config, embed_tokens=None, **kwargs):
        super().__init__(**kwargs)

        # 初始化函数，设置TFT5MainLayer对象的属性
        self.config = config
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.use_cache = config.use_cache

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.config = config
        self.num_hidden_layers = config.num_layers

        # 创建TFT5Block块列表
        self.block = [
            TFT5Block(config, has_relative_attention_bias=bool(i == 0), name=f"block_._{i}")
            for i in range(config.num_layers)
        ]
        self.final_layer_norm = TFT5LayerNorm(
            config.d_model, epsilon=config.layer_norm_epsilon, name="final_layer_norm"
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout_rate)

    def _prune_heads(self, heads_to_prune):
        raise NotImplementedError  # Not implemented yet in the library fr TF 2.0 models
        # 定义_prune_heads方法，抛出NotImplementedError异常

    @unpack_inputs
    # 装饰器
    # 定义一个 call 方法，接受多个输入参数
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    ):
        # 这里是方法的实现部分，具体做什么操作没有给出
    
    # 定义一个 build 方法
    def build(self, input_shape=None):
        # 如果已经构建过了，直接返回
        if self.built:
            return
        # 设置 built 标志为 True，表示已构建
        self.built = True
        
        # 如果存在 final_layer_norm，在 tf.name_scope 下构建它
        if getattr(self, "final_layer_norm", None) is not None:
            with tf.name_scope(self.final_layer_norm.name):
                self.final_layer_norm.build(None)
        
        # 如果存在 block，遍历每个 layer，在 tf.name_scope 下构建它们
        if getattr(self, "block", None) is not None:
            for layer in self.block:
                with tf.name_scope(layer.name):
                    layer.build(None)
####################################################
# TFT5PreTrainedModel是tf.keras.Model的子类，负责加载和保存预训练权重以及各种常见的实用工具。
# 在这里，您只需要为您的模型指定一些（不言而喻的）指针。
####################################################
class TFT5PreTrainedModel(TFPreTrainedModel):
    """
    处理权重初始化和简单接口以下载和加载预训练模型的抽象类。
    """

    config_class = T5Config
    base_model_prefix = "transformer"
    # 当从PT模型加载TF模型时，具有'.'的名称表示授权的意外/丢失层
    _keys_to_ignore_on_load_unexpected = [r"decoder\Wblock[\W_0]+layer[\W_1]+EncDecAttention\Wrelative_attention_bias"]

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        if hasattr(self, "decoder"):
            self.decoder.embed_tokens = self.shared

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id必须定义。 在TF T5中，它通常设置为pad_token_id。 有关更多信息，请参阅T5文档"
        )

        start_tokens = tf.fill((shape_list(input_ids)[0], 1), decoder_start_token_id)
        start_tokens = tf.cast(start_tokens, input_ids.dtype)  # 确保连接时的兼容dtype
        shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], -1)

        assert pad_token_id is not None, "self.model.config.pad_token_id必须定义。"
        # 将标签中可能的-100值替换为`pad_token_id`
        shifted_input_ids = tf.where(
            shifted_input_ids == -100,
            tf.cast(tf.fill(shape_list(shifted_input_ids), pad_token_id), shifted_input_ids.dtype),
            shifted_input_ids,
        )

        # "验证`标签`仅具有正值和-100"
        assert_gte0 = tf.debugging.assert_greater_equal(
            shifted_input_ids, tf.constant(0, dtype=shifted_input_ids.dtype)
        )

        # 确保通过将结果包装在身份no-op中来调用断言操作
        with tf.control_dependencies([assert_gte0]):
            shifted_input_ids = tf.identity(shifted_input_ids)

        return shifted_input_ids


T5_START_DOCSTRING = r"""

    T5模型在[Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) 中由Colin Raffel、Noam Shazeer、Adam Roberts、Katherine Lee、Sharan
    Narang、Michael Matena、Yanqi Zhou、Wei Li、Peter J. Liu提出。 它是一个在文本到文本转换中进行了预训练的编码器解码器变换器。

```    
    # text-to-text去噪生成模型
    
        # 该模型继承自`TFPreTrainedModel`。请查看超类的文档以获取库为其所有模型实现的通用方法（如下载或保存，调整输入嵌入，裁剪头等）。
        
        # 该模型还是[tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)的子类。可以将其视为常规的TF 2.0 Keras模型，并参考TF 2.0文档以了解与一般用法和行为相关的所有内容。
        
        <提示>
        
        `transformers`中的TensorFlow模型和层可接受两种格式的输入：
        
        - 将所有输入作为关键字参数（类似于PyTorch模型），或
        - 将所有输入作为列表、元组或字典传递给第一个位置参数。
        
        支持第二种格式的原因是，当将输入传递给模型和层时，Keras方法更喜欢此格式。由于支持此格式，当使用`model.fit()`等方法时，您只需以`model.fit()`支持的任何格式传递输入和标签即可！但是，如果您要在Keras方法之外使用第二种格式，例如在使用Keras `Functional` API创建自己的层或模型时，有三种可能性可以使用第一个位置参数中的所有输入张量进行汇总：
        
        - 仅具有`input_ids`（即无其他参数）的单个张量：`model(input_ids)`
        - 长度可变的列表，其中包含一个或多个输入张量，并按照说明文档中给出的顺序：`model([input_ids, attention_mask])`或`model([input_ids, attention_mask, token_type_ids])`
        - 具有与说明文档中给出的输入名称相关联的一个或多个输入张量的字典：`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`
        
        请注意，当使用
        [子类化](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)创建模型和层时，不需要担心这些问题，因为您可以像对任何其他Python函数一样传递输入！
        
        </提示>
        
        参数：
            config ([`T5Config`]）：模型配置类，包含模型的所有参数。使用配置文件初始化不会加载与模型相关的权重，只会加载配置。请查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""
# T5 Model Transformer
实现了不带特定头部的T5模型转换器，输出原始的隐藏状态。

## T5_INPUTS_DOCSTRING
`inputs`参数文档字符串，用于描述模型输入。

- `inputs`(`tf.Tensor`，形状为`(batch_size, sequence_length)`):
  输入序列标记在词汇表中的索引。T5是一个具有相对位置嵌入的模型，因此您可以在右侧或左侧填充输入。

  您可以使用[`AutoTokenizer`]获取索引。有关详细信息，请参阅[`PreTrainedTokenizer.__call__`]和[`PreTrainedTokenizer.encode`]。

- `attention_mask`(`tf.Tensor`，形状为`(batch_size, sequence_length)`，*可选*):
  避免在填充标记索引上执行注意力的掩码。掩码值选择在`[0, 1]`范围内:

  - 1表示**不掩码**的标记，
  - 0表示**掩码**的标记。

- `inputs_embeds`(`tf.Tensor`，形状为`(batch_size, sequence_length, hidden_size)`，*可选*):
  如果选择直接传递嵌入表示而不是`input_ids`，则可以直接传递。这对于希望对`input_ids`索引如何转换为相关矢量具有更多控制权的情况很有用。

- `head_mask`(`tf.Tensor`，形状为`(num_heads,)`或`(num_layers, num_heads)`，*可选*):
  头部掩码，用于使自注意力模块选择性地失效。掩码值选择在`[0, 1]`范围内:

  - 1表示该头部**不被掩码**，
  - 0表示该头部**被掩码**。

- `output_attentions`(`bool`，*可选*):
  是否返回所有注意力层的注意力张量。有关详细信息，请参阅返回张量中的`attentions`。

- `output_hidden_states`(`bool`，*可选*):
  是否返回所有层的隐藏状态。有关详细信息，请参阅返回张量中的`hidden_states`。

- `return_dict`(`bool`，*可选*):
  是否返回[`~utils.ModelOutput`]而不是普通元组。

- `training`(`bool`，*可选*，默认为`False`)：
  是否在训练模式下使用模型（某些模块如投放模块在训练和评估之间有不同的行为）。

## _HEAD_MASK_WARNING_MSG
头部掩码警告消息。

当前，输入参数`head_mask`已分为两个参数`head_mask`和`decoder_head_mask`。
目前`decoder_head_mask`设置为复制`head_mask`，但此功能在将来的版本中将被弃用。

如果您现在不想使用任何`decoder_head_mask`，请将`decoder_head_mask = tf.ones((num_layers, num_heads))`。

## bare T5 Model转换器输出原始隐藏状态
给出了T5模型转换器的简要文档字符串和T5的起始文档字符串。
"""
class TFT5Model(TFT5PreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化共享层，使用配置中的词汇大小和模型维度
        self.shared = tf.keras.layers.Embedding(
            input_dim=config.vocab_size,
            output_dim=config.d_model,
            embeddings_initializer=tf.keras.initializers.TruncatedNormal(self.config.initializer_factor),
            name="shared",
        )
        # 附加属性，用于指定该层的名称范围（用于加载/存储权重）
        self.shared.load_weight_prefix = "shared"

        # 复制编码器配置，设置使用缓存为False，构建编码器层
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name="encoder")

        # 复制解码器配置，设置为解码器，设置层数为配置中的解码器层数，构建解码器层
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = TFT5MainLayer(decoder_config, self.shared, name="decoder")

    # 返回编码器
    def get_encoder(self):
        return self.encoder

    # 返回解码器
    def get_decoder(self):
        return self.decoder

    # 调用方法，传入各种输入参数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,
        encoder_outputs: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
        
    # 构建模型
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 共享/绑定权重希望位于模型基本命名空间中
        # 在tf.name_scope末尾添加“/”（而不是开头！）将其放入根命名空间，而不是当前命名空间。
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            self.shared.build(None)
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
# 添加自动生成的文档字符串，T5 模型在其基础上加上一个 `language modeling` 头部
@add_start_docstrings("""T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING)
class TFT5ForConditionalGeneration(TFT5PreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法，传入配置和额外参数
        super().__init__(config, *inputs, **kwargs)
        # 设置模型维度为配置文件中的 d_model 值
        self.model_dim = config.d_model
        # 创建一个共享的嵌入层，用于分享编码器和解码器中的嵌入层参数
        self.shared = tf.keras.layers.Embedding(
            config.vocab_size,
            config.d_model,
            name="shared",
            embeddings_initializer=get_initializer(self.config.initializer_factor),
        )
        # 附加属性，用于指定层的预期名称空间（用于加载/存储权重）
        self.shared.load_weight_prefix = "shared"

        # 复制配置用于编码器
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        # 创建编码器，传入配置和共享的嵌入层
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name="encoder")

        # 复制配置用于解码器
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器，传入配置和共享的嵌入层
        self.decoder = TFT5MainLayer(decoder_config, self.shared, name="decoder")

        # 如果不共享词嵌入，则创建一个新的 LM 头部
        if not config.tie_word_embeddings:
            lm_head_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=config.initializer_factor)
            self.lm_head = tf.keras.layers.Dense(
                config.vocab_size, use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # 更新初始权重与 flax 中一致
        # 保存配置
        self.config = config

    # 获取输出嵌入层
    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return self.get_input_embeddings()
        else:
            # 在密集层中，核的形状为 (last_dim, units)，对于我们来说是 (dim, num_tokens)
            # value 的形状为 (num_tokens, dim)，然后需要转置
            return tf.transpose(self.lm_head.kernel)

    # 设置输出嵌入层
    def set_output_embeddings(self, value):
        if self.config.tie_word_embeddings:
            self.set_input_embeddings(value)
        else:
            lm_head_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=self.config.initializer_factor)
            self.lm_head = tf.keras.layers.Dense(
                shape_list(value)[0], use_bias=False, name="lm_head", kernel_initializer=lm_head_initializer
            )  # 更新初始权重与 flax 中一致
            # 在密集层中，核的形状为 (last_dim, units)，对于我们来说是 (dim, num_tokens)
            # value 的形状为 (num_tokens, dim)，然后需要转置
            transposed_value = tf.transpose(value)
            self.lm_head.kernel = transposed_value

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 解包输入，添加模型前向方法的文档字符串
    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法，用于调用模型
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入文本的标识符
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 输入文本的注意力掩码
        decoder_input_ids: np.ndarray | tf.Tensor | None = None,  # 解码器的输入标识符
        decoder_attention_mask: np.ndarray | tf.Tensor | None = None,  # 解码器的注意力掩码
        head_mask: np.ndarray | tf.Tensor | None = None,  # 头部掩码
        decoder_head_mask: np.ndarray | tf.Tensor | None = None,  # 解码器的头部掩码
        encoder_outputs: np.ndarray | tf.Tensor | None = None,  # 编码器输出
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,  # 过去的关键值
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 输入的嵌入向量
        decoder_inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 解码器的输入嵌入向量
        labels: np.ndarray | tf.Tensor | None = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
        training: Optional[bool] = False,  # 是否在训练中
    def serving_output(self, output):
        # 根据配置判断是否使用缓存并转换成张量
        pkv = tf.convert_to_tensor(output.past_key_values[1:]) if self.config.use_cache else None
        dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
        dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
        cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
        enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
        enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None

        # 返回序列到序列模型的输出
        return TFSeq2SeqLMOutput(
            logits=output.logits,  # 预测的对数概率值
            past_key_values=pkv,  # 过去的关键值
            decoder_hidden_states=dec_hs,  # 解码器的隐藏状态
            decoder_attentions=dec_attns,  # 解码器的注意力
            cross_attentions=cross_attns,  # 交叉注意力
            encoder_last_hidden_state=output.encoder_last_hidden_state,  # 编码器的最后隐藏状态
            encoder_hidden_states=enc_hs,  # 编码器的隐藏状态
            encoder_attentions=enc_attns,  # 编码器的注意力
        )

    # 为生成准备输入
    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入标识符
        past_key_values=None,  # 过去的关键值
        attention_mask=None,  # 注意力掩码
        decoder_attention_mask=None,  # 解码器注意力掩码
        head_mask=None,  # 头部掩码
        decoder_head_mask=None,  # 解码器头部掩码
        use_cache=None,  # 是否使用缓存
        encoder_outputs=None,  # 编码器输出
        **kwargs,  # 其他参数
    ):
        # 如果使用过去的关键值，截取解码器的输入标识符
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": None,  # 传递给Keras.layer.__call__的参数
            "decoder_input_ids": input_ids,  # 解码器的输入标识符
            "past_key_values": past_key_values,  # 过去的关键值
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "attention_mask": attention_mask,  # 注意力掩码
            "decoder_attention_mask": decoder_attention_mask,  # 解码器的注意力掩码
            "head_mask": head_mask,  # 头部掩码
            "decoder_head_mask": decoder_head_mask,  # 解码器的头部掩码
            "use_cache": use_cache,  # 是否使用缓存
        }
    # 生成一个用于解码的输入序列，将标签数据向右移动一位
    def prepare_decoder_input_ids_from_labels(self, labels: tf.Tensor):
        return self._shift_right(labels)

    # 构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完毕，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 共享/绑定的权重需要在模型基本命名空间中
        # 将 tf.name_scope 结尾添加 "/" (不是开头!) 可将其放入根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享的权重
            self.shared.build(None)
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在解码器，则构建解码器
        if getattr(self, "decoder", None) is not None:
            with tf.name_scope(self.decoder.name):
                self.decoder.build(None)
        # 如果存在语言模型头，构建语言模型头
        if getattr(self, "lm_head", None) is not None:
            with tf.name_scope(self.lm_head.name):
                # 构建语言模型头，输入形状为 [None, None, self.config.d_model]
                self.lm_head.build([None, None, self.config.d_model])
# 这是一个 TFT5EncoderModel 类，继承自 TFT5PreTrainedModel 类
@add_start_docstrings(
    "The bare T5 Model transformer outputting encoder's raw hidden-stateswithout any specific head on top.",
    T5_START_DOCSTRING,
)
class TFT5EncoderModel(TFT5PreTrainedModel):
    # 初始化函数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 创建一个共享的 Embedding 层
        self.shared = tf.keras.layers.Embedding(
            config.vocab_size,
            config.d_model,
            name="shared",
            embeddings_initializer=get_initializer(self.config.initializer_factor),
        )
        # 设置共享层的加载/保存权重的前缀
        self.shared.load_weight_prefix = "shared"

        # 复制配置并设置 use_cache 为 False
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        # 创建 TFT5MainLayer 作为编码器
        self.encoder = TFT5MainLayer(encoder_config, self.shared, name="encoder")

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 前向传播函数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[Tuple, TFBaseModelOutput]:
        # 使用编码器进行前向传播
        encoder_outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 根据 return_dict 决定返回值的形式
        if not return_dict:
            return encoder_outputs

        return TFBaseModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    # 构建方法用于构建模型
    def build(self, input_shape=None):
        # 如果模型已经构建完成，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 共享/绑定的权重应该存在于模型的基本命名空间中
        # 在当前命名空间添加"/"到结尾，将其放在根命名空间而不是当前命名空间
        with tf.name_scope(self.shared.load_weight_prefix + "/" + self.shared.name + "/"):
            # 构建共享组件
            self.shared.build(None)
        # 如果存在编码器
        if getattr(self, "encoder", None) is not None:
            # 在编码器的命名空间内构建
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
```