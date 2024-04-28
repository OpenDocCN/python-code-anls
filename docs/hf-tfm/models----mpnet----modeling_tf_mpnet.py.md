# `.\transformers\models\mpnet\modeling_tf_mpnet.py`

```py
# 设置文件的编码格式为utf-8
# 版权声明
# 引入模块
# 引入必要的库
# 设置函数返回值的注解类型
# 导入模型和模块的输出
# 导入模型工具
# 导入必要的库和依赖项
# 设置模型的损失函数
# 设置模型输入类型
# 引入一些注释
# 导入依赖项
# 定义模型的输入类型
# 设置模型预训练的路径列表
# 拿到 logger 模块
# 搜索检查点路径
# 搜索配置信息路径
# 设置预训练模型的路径列表

# 定义 TFMPNetPreTrainedModel 类，用于初始化权重和处理预训练模型的下载和加载
class TFMPNetPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 初始化配置类
    config_class = MPNetConfig
    # 模型前缀
    base_model_prefix = "mpnet"

# 定义 TFMPNetEmbeddings 类，用于从词嵌入和位置嵌入构造嵌入
class TFMPNetEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position embeddings."""
    # 初始化函数
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 设定 padding 索引为1
        self.padding_idx = 1
        # 获取配置信息
        self.config = config
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 获取最大位置嵌入数
        self.max_position_embeddings = config.max_position_embeddings
        # 获取初始化范围
        self.initializer_range = config.initializer_range
        # 设置 Layer Norm 层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 设置 dropout 层
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    # 在构建模型时创建词嵌入层
    def build(self, input_shape=None):
        # 使用 TensorFlow 的命名作用域，方便在 TensorBoard 中组织可视化结果
        with tf.name_scope("word_embeddings"):
            # 添加词嵌入层的权重参数，形状为 [词汇表大小, 隐藏层大小]
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                # 初始化权重参数的方法
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 使用 TensorFlow 的命名作用域，方便在 TensorBoard 中组织可视化结果
        with tf.name_scope("position_embeddings"):
            # 添加位置嵌入层的权重参数，形状为 [最大位置编码数, 隐藏层大小]
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                # 初始化权重参数的方法
                initializer=get_initializer(initializer_range=self.initializer_range),
            )

        # 如果模型已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果模型包含 LayerNorm 层，则构建 LayerNorm 层
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建 LayerNorm 层，输入形状为 [None, None, 隐藏层大小]
                self.LayerNorm.build([None, None, self.config.hidden_size])

    # 根据输入的 token ids 创建位置编码
    def create_position_ids_from_input_ids(self, input_ids):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            input_ids: tf.Tensor
        Returns: tf.Tensor
        """
        # 创建非填充符号的位置编码。位置编号从 padding_idx+1 开始。填充符号被忽略。
        mask = tf.cast(tf.math.not_equal(input_ids, self.padding_idx), dtype=input_ids.dtype)
        incremental_indices = tf.math.cumsum(mask, axis=1) * mask

        return incremental_indices + self.padding_idx

    # 根据输入生成嵌入向量
    def call(self, input_ids=None, position_ids=None, inputs_embeds=None, training=False):
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 断言输入不为空
        assert not (input_ids is None and inputs_embeds is None)

        # 如果输入为 token ids，则根据 token ids 检查嵌入是否超出边界，并从权重参数中获取对应的嵌入向量
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果位置编码为空，则根据输入 token ids 创建位置编码
        if position_ids is None:
            if input_ids is not None:
                # 从输入的 token ids 创建位置编码，保留任何填充的 token 的填充位置
                position_ids = self.create_position_ids_from_input_ids(input_ids=input_ids)
            else:
                # 如果没有输入的 token ids，则创建默认的位置编码
                position_ids = tf.expand_dims(
                    tf.range(start=self.padding_idx + 1, limit=input_shape[-1] + self.padding_idx + 1), axis=0
                )

        # 根据位置编码获取位置嵌入
        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        # 最终的嵌入向量为输入嵌入向量和位置嵌入向量的和
        final_embeddings = inputs_embeds + position_embeds
        # 对最终嵌入向量进行 LayerNorm 归一化处理
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        # 使用 dropout 进行正则化处理
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings
# 从transformers.models.bert.modeling_tf_bert.TFBertPooler复制到MPNet，定义MPNetPooler类
class TFMPNetPooler(tf.keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs):
        super().__init__(**kwargs)

        # 创建一个全连接层，单元数为config.hidden_size，激活函数为tanh
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        self.config = config

    # 定义call方法，用于对输入的hidden_states进行处理
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 通过取第一个标记对模型进行“汇总”
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)
        return pooled_output

    # 构建方法，用于构建层数是否已经构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 定义TFMPNetSelfAttention类，继承自Layer
class TFMPNetSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 如果config.hidden_size不能整除config.num_attention_heads，则抛出ValueError异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads}"
            )

        # 初始化注意力头数和大小
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化Q、K、V、O层，按照config配置进行初始化
        self.q = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="q"
        )
        self.k = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="k"
        )
        self.v = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="v"
        )
        self.o = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="o"
        )
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
        self.config = config

    # 将x转置用于计算分数
    def transpose_for_scores(self, x, batch_size):
        # 从[batch_size, seq_length, all_head_size]转换为[batch_size, seq_length, num_attention_heads, attention_head_size]
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))

        return tf.transpose(x, perm=[0, 2, 1, 3])
    # 定义一个调用方法，接收隐藏状态、注意力掩码、头部掩码、输出注意力、位置偏差和训练标志作为参数
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        # 获取隐藏状态的批量大小
        batch_size = shape_list(hidden_states)[0]

        # 通过q、k、v网络分别对隐藏状态进行处理得到查询、键、值
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        # 为了进行注意力计算，对查询、键、值进行维度变换
        q = self.transpose_for_scores(q, batch_size)
        k = self.transpose_for_scores(k, batch_size)
        v = self.transpose_for_scores(v, batch_size)

        # 计算注意力分数
        attention_scores = tf.matmul(q, k, transpose_b=True)
        # 获取密钥的维度并转换为注意力分数的数据类型
        dk = tf.cast(shape_list(k)[-1], attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)

        # 如果提供了位置偏差，则应用相对位置嵌入
        if position_bias is not None:
            attention_scores += position_bias

        # 如果存在注意力掩码，则应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对注意力分数进行稳定的softmax操作
        attention_probs = stable_softmax(attention_scores, axis=-1)

        # 对注意力概率进行dropout操作
        attention_probs = self.dropout(attention_probs, training=training)

        # 如果存在头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 通过注意力概率对值进行加权求和操作
        c = tf.matmul(attention_probs, v)
        c = tf.transpose(c, perm=[0, 2, 1, 3])
        c = tf.reshape(c, (batch_size, -1, self.all_head_size))
        
        # 通过输出网络对c进行处理得到最终输出o
        o = self.o(c)

        # 如果需要输出注意力，则返回输出o和注意力概率；否则只返回输出o
        outputs = (o, attention_probs) if output_attentions else (o,)
        return outputs

    # 构建方法，用于构建注意力层网络
    def build(self, input_shape=None):
        # 如果已经构建过网络，则直接返回
        if self.built:
            return
        self.built = True
        # 如果网络中存在查询、键、值、输出网络，则分别构建它们
        if getattr(self, "q", None) is not None:
            with tf.name_scope(self.q.name):
                self.q.build([None, None, self.config.hidden_size])
        if getattr(self, "k", None) is not None:
            with tf.name_scope(self.k.name):
                self.k.build([None, None, self.config.hidden_size])
        if getattr(self, "v", None) is not None:
            with tf.name_scope(self.v.name):
                self.v.build([None, None, self.config.hidden_size])
        if getattr(self, "o", None) is not None:
            with tf.name_scope(self.o.name):
                self.o.build([None, None, self.config.hidden_size])
# 定义了一个 TFMPNetAttention 类，继承自 tf.keras.layers.Layer
class TFMPNetAttention(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)
        
        # 创建 TFMPNetSelfAttention 对象
        self.attn = TFMPNetSelfAttention(config, name="attn")
        # 创建 LayerNorm 层
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建 Dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 保存配置参数
        self.config = config

    # 剪枝 head
    def prune_heads(self, heads):
        # 抛出 NotImplementedError 异常
        raise NotImplementedError

    # 前向传播函数
    def call(self, input_tensor, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        # 调用 self.attn 的前向传播函数
        self_outputs = self.attn(
            input_tensor, attention_mask, head_mask, output_attentions, position_bias=position_bias, training=training
        )
        # 注意力输出和输入张量相加并经过 LayerNorm、dropout 处理
        attention_output = self.LayerNorm(self.dropout(self_outputs[0]) + input_tensor)
        # 构造输出元组
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    # 构建函数
    def build(self, input_shape=None):
        # 若已构建则返回
        if self.built:
            return
        self.built = True
        # 构建 self.attn
        if getattr(self, "attn", None) is not None:
            with tf.name_scope(self.attn.name):
                self.attn.build(None)
        # 构建 self.LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertIntermediate 中复制过来，将 Bert->MPNet
# 定义了一个 TFMPNetIntermediate 类，继承自 tf.keras.layers.Layer
class TFMPNetIntermediate(tf.keras.layers.Layer):
    # 初始化函数
    def __init__(self, config: MPNetConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建 Dense 层
        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 根据配置参数创建 intermediate_act_fn 函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        # 保存配置参数
        self.config = config

    # 前向传播函数
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 经过 dense 层和 intermediate_act_fn 函数的处理
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 构建函数
    def build(self, input_shape=None):
        # 若已构建则返回
        if self.built:
            return
        self.built = True
        # 构建 self.dense
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


# 从 transformers.models.bert.modeling_tf_bert.TFBertOutput 中复制过来，将 Bert->MPNet
# 定义了一个 TFMPNetOutput 类，继承自 tf.keras.layers.Layer
class TFMPNetOutput(tf.keras.layers.Layer):
    def __init__(self, config: MPNetConfig, **kwargs):
        # 调用父类构造函数初始化对象
        super().__init__(**kwargs)

        # 初始化全连接层，设置单元数、初始化器和名称
        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 初始化 LayerNormalization 层，设置 epsilon 和名称
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 初始化 Dropout 层，设置丢弃率
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 将隐藏状态输入全连接层
        hidden_states = self.dense(inputs=hidden_states)
        # 对输出进行 Dropout 处理
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        # 将输出与输入张量相加后进行 LayerNormalization
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

    def build(self, input_shape=None):
        # 如果已构建过网络结构，则直接返回
        if self.built:
            return
        # 设置网络已构建标志
        self.built = True
        # 如果存在全连接层 dense
        if getattr(self, "dense", None) is not None:
            # 在全连接层 dense 上下文中设置名称作用域，构建全连接层
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        # 如果存在 LayerNormalization 层 LayerNorm
        if getattr(self, "LayerNorm", None) is not None:
            # 在 LayerNormalization 层 LayerNorm 上下文中设置名称作用域，构建 LayerNormalization 层
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
# TFMPNetLayer 是一个 Keras 层,包含注意力机制、中间层和输出层
class TFMPNetLayer(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 初始化注意力机制层、中间层和输出层
        self.attention = TFMPNetAttention(config, name="attention")
        self.intermediate = TFMPNetIntermediate(config, name="intermediate")
        self.out = TFMPNetOutput(config, name="output")

    # 定义前向传播过程
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, position_bias=None, training=False):
        # 计算注意力输出
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions, position_bias=position_bias, training=training
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重,则添加到输出中

        # 计算中间层输出
        intermediate_output = self.intermediate(attention_output)
        # 计算输出层输出
        layer_output = self.out(intermediate_output, attention_output, training=training)
        outputs = (layer_output,) + outputs  # 将输出层输出添加到输出中

        return outputs

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 构建注意力机制层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        # 构建中间层
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        # 构建输出层
        if getattr(self, "out", None) is not None:
            with tf.name_scope(self.out.name):
                self.out.build(None)


# TFMPNetEncoder 是一个 Keras 层,包含多个 TFMPNetLayer 层
class TFMPNetEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 保存配置信息
        self.config = config
        self.n_heads = config.num_attention_heads
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.initializer_range = config.initializer_range

        # 创建多个 TFMPNetLayer 层
        self.layer = [TFMPNetLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

    # 构建层
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 创建相对注意力偏置权重
        with tf.name_scope("relative_attention_bias"):
            self.relative_attention_bias = self.add_weight(
                name="embeddings",
                shape=[self.relative_attention_num_buckets, self.n_heads],
                initializer=get_initializer(self.initializer_range),
            )
        # 构建每个 TFMPNetLayer 层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)
    # 定义了一个方法用于模型的调用，接受多个参数，包括隐藏状态、注意力掩码、头部掩码、是否输出注意力权重、是否输出隐藏状态、是否以字典形式返回结果以及是否处于训练状态
    def call(
        self,
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions,
        output_hidden_states,
        return_dict,
        training=False,
    ):
        # 计算位置偏置
        position_bias = self.compute_position_bias(hidden_states)
        # 如果设置了输出隐藏状态，初始化存储所有隐藏状态的元组
        all_hidden_states = () if output_hidden_states else None
        # 如果设置了输出注意力权重，初始化存储所有注意力权重的元组
        all_attentions = () if output_attentions else None

        # 遍历每个层次的模块
        for i, layer_module in enumerate(self.layer):
            # 如果设置了输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层次的模块，传入隐藏状态、注意力掩码、头部掩码、是否输出注意力权重、位置偏置以及是否处于训练状态
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions,
                position_bias=position_bias,
                training=training,
            )
            # 更新隐藏状态为当前层次的模块输出的隐藏状态
            hidden_states = layer_outputs[0]

            # 如果设置了输出注意力权重，则将当前层次模块输出的注意力权重加入到所有注意力权重的元组中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不要求以字典形式返回结果，则返回一个元组，其中包含隐藏状态、所有隐藏状态和所有注意力权重，如果某个元素为None则不加入到元组中
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 如果要求以字典形式返回结果，则返回一个TFBaseModelOutput对象，包含最后的隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

    # 静态方法，用于计算相对位置的哈希桶
    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        # 初始化相对位置哈希桶的值
        ret = 0
        # 计算相对位置的负值
        n = -relative_position

        # 哈希桶数量除以2
        num_buckets //= 2
        # 如果相对位置为负，则将哈希桶数量加上一半
        ret += tf.cast(tf.math.less(n, 0), dtype=relative_position.dtype) * num_buckets
        # 取相对位置的绝对值
        n = tf.math.abs(n)

        # 现在n的范围在[0, inf)
        max_exact = num_buckets // 2
        # 判断n是否小于哈希桶数量的一半
        is_small = tf.math.less(n, max_exact)

        # 如果n较小，直接使用n作为哈希值
        val_if_large = max_exact + tf.cast(
            tf.math.log(n / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact),
            dtype=relative_position.dtype,
        )
        # 限制哈希值的范围在[0, num_buckets - 1]
        val_if_large = tf.math.minimum(val_if_large, num_buckets - 1)
        # 根据n的大小选择合适的哈希值
        ret += tf.where(is_small, n, val_if_large)
        # 返回最终的哈希桶值
        return ret
    # 计算分桶后的相对位置偏置
    def compute_position_bias(self, x, position_ids=None):
        """Compute binned relative position bias"""
        # 获取输入张量的形状
        input_shape = shape_list(x)
        qlen, klen = input_shape[1], input_shape[1]

        # 如果给定了位置ID，使用位置ID创建上下文位置和记忆位置张量
        if position_ids is not None:
            context_position = position_ids[:, :, None]
            memory_position = position_ids[:, None, :]
        else:
            # 否则使用默认位置ID范围
            context_position = tf.range(qlen)[:, None]
            memory_position = tf.range(klen)[None, :]

        # 计算相对位置，shape为(qlen, klen)
        relative_position = memory_position - context_position  

        # 将相对位置分桶，使用相对注意力的桶数
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.relative_attention_num_buckets,
        )
        # 从相对注意力偏置中检索值，shape为(qlen, klen, num_heads)
        values = tf.gather(self.relative_attention_bias, rp_bucket)  
        # 对值进行转置并添加维度，shape为(1, num_heads, qlen, klen)
        values = tf.expand_dims(tf.transpose(values, [2, 0, 1]), axis=0)  
        # 返回计算结果
        return values
# 声明一个自定义的 TFMPNetMainLayer 类，并使其可序列化
@keras_serializable
class TFMPNetMainLayer(tf.keras.layers.Layer):
    # 指定配置类
    config_class = MPNetConfig

    # 初始化方法
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        # 初始化配置属性
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        # 创建编码器、池化器和嵌入层对象
        self.encoder = TFMPNetEncoder(config, name="encoder")
        self.pooler = TFMPNetPooler(config, name="pooler")
        # 嵌入层必须在最后声明，以遵循权重顺序
        self.embeddings = TFMPNetEmbeddings(config, name="embeddings")

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.get_input_embeddings 复制的方法
    def get_input_embeddings(self) -> tf.keras.layers.Layer:
        return self.embeddings

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer.set_input_embeddings 复制的方法
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # 从 transformers.models.bert.modeling_tf_bert.TFBertMainLayer._prune_heads 复制的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 对 call 方法进行输入解包
    @unpack_inputs
    def call(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        training=False,
    # 构建方法
    def build(self, input_shape=None):
        # 如果已构建过，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在编码器，则构建编码器
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
        # 如果存在池化器，则构建池化器
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        # 如果存在嵌入层，则构建嵌入层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)


MPNET_START_DOCSTRING = r"""

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
    as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage and
    behavior.

    <Tip>
    TensorFlow models and layers in `transformers` accept two formats as input:

    - having all inputs as keyword arguments (like PyTorch models), or
    - having all inputs as a list, tuple or dict in the first positional argument.

    The reason the second format is supported is that Keras methods prefer this format when passing inputs to models
    and layers. Because of this support, when using methods like `model.fit()` things should "just work" for you - just
    pass your inputs and labels in any format that `model.fit()` supports! If, however, you want to use the second
    format outside of Keras methods like `fit()` and `predict()`, such as when creating your own layers or models with
    the Keras `Functional` API, there are three possibilities you can use to gather all the input Tensors in the first
    positional argument:

    - a single Tensor with `input_ids` only and nothing else: `model(input_ids)`
    - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
    `model([input_ids, attention_mask])` or `model([input_ids, attention_mask, token_type_ids])`
    - a dictionary with one or several input Tensors associated to the input names given in the docstring:
    `model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Note that when creating models and layers with
    [subclassing](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) then you don't need to worry
    about any of this, as you can just pass inputs like you would to any other Python function!

    </Tip>

    Args:
        config ([`MPNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# MPNet 模型的输入文档字符串，用于说明每个输入参数的作用

@add_start_docstrings(
    "The bare MPNet Model transformer outputting raw hidden-states without any specific head on top.",
    设置 MPNet 模型的基本转换器输出，输出原始的隐藏状态，没有任何特定的头部层
    # 定义一个变量，可能是用于开始文档字符串的标记
    MPNET_START_DOCSTRING,
# 定义 TFMPNetModel 类，继承自 TFMPNetPreTrainedModel 类
class TFMPNetModel(TFMPNetPreTrainedModel):
    # 构造函数，接受 config 参数和输入，并将其传递给父类构造函数
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        # 创建 TFMPNetMainLayer 实例
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")

    # 使用装饰器指定 call 方法的调用方式，添加文档字符串和示例代码
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 call 方法，接受多个输入参数并调用 self.mpnet 方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        outputs = self.mpnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        return outputs

    # 构建方法，根据输入形状构建模型
    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 设置已构建标志为 True
        self.built = True
        if getattr(self, "mpnet", None) is not None:
            # 在指定名称空间下构建 mpnet
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)


# 定义 TFMPNetLMHead 类，继承自 tf.keras.layers.Layer 类
class TFMPNetLMHead(tf.keras.layers.Layer):
    """MPNet head for masked and permuted language modeling"""

    # 构造函数，接受 config，input_embeddings 和其他参数
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        # 创建一个全连接层，设置权重初始化器为 config 中定义的初始化范围
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 创建一个层归一化层
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="layer_norm")
        # 获取 GELU 激活函数
        self.act = get_tf_activation("gelu")

        # 输出权重和输入嵌入一样，但每个标记有一个输出偏差
        self.decoder = input_embeddings
    # 构建模型的方法，输入形状为可选参数，默认为 None
    def build(self, input_shape=None):
        # 添加偏置项，并初始化为零向量
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
    
        # 如果模型已经构建好，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 dense 层，则构建它
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 layer_norm 层，则构建它
        if getattr(self, "layer_norm", None) is not None:
            with tf.name_scope(self.layer_norm.name):
                self.layer_norm.build([None, None, self.config.hidden_size])
    
    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.decoder
    
    # 设置输出嵌入
    def set_output_embeddings(self, value):
        self.decoder.weight = value
        self.decoder.vocab_size = shape_list(value)[0]
    
    # 获取偏置项
    def get_bias(self):
        return {"bias": self.bias}
    
    # 设置偏置项
    def set_bias(self, value):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]
    
    # 调用方法
    def call(self, hidden_states):
        # 使用 dense 层进行变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数进行激活
        hidden_states = self.act(hidden_states)
        # 使用 layer_norm 层进行归一化
    
        hidden_states = self.layer_norm(hidden_states)
    
        # 通过偏置项将隐藏状态投影回词汇表大小
        seq_length = shape_list(tensor=hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.decoder.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)
    
        return hidden_states
# 用语言模型头部在 MPNet 模型上创建一个 `语言建模` 的模型
@add_start_docstrings("""MPNet Model with a `language modeling` head on top.""", MPNET_START_DOCSTRING)
class TFMPNetForMaskedLM(TFMPNetPreTrainedModel, TFMaskedLanguageModelingLoss):
    # 在加载时忽略的键值
    _keys_to_ignore_on_load_missing = [r"pooler"]

    # 初始化模型
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 创建 MPNet 主层
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 创建 MPNet 语言模型头部
        self.lm_head = TFMPNetLMHead(config, self.mpnet.embeddings, name="lm_head")

    # 获取语言模型头部
    def get_lm_head(self):
        return self.lm_head

    # 获取前缀偏置名称
    def get_prefix_bias_name(self):
        warnings.warn("The method get_prefix_bias_name is deprecated. Please use `get_bias` instead.", FutureWarning)
        return self.name + "/" + self.lm_head.name

    # 调用模型前的特殊处理
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型前向传播函数
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 调用 MPNet 模型传播
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 使用 lm_head 对序列输出进行预测
        prediction_scores = self.lm_head(sequence_output)

        # 如果没有标签，则损失为 None；否则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels, prediction_scores)

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 MaskedLMOutput 对象
        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 构建神经网络模型
        def build(self, input_shape=None):
            # 如果模型已经构建过，直接返回
            if self.built:
                return
            # 标记模型已经构建
            self.built = True
            # 如果存在 mpnet 属性，使用 tf.name_scope 创建命名作用域
            if getattr(self, "mpnet", None) is not None:
                with tf.name_scope(self.mpnet.name):
                    # 构建 mpnet
                    self.mpnet.build(None)
            # 如果存在 lm_head 属性，使用 tf.name_scope 创建命名作用域
            if getattr(self, "lm_head", None) is not None:
                with tf.name_scope(self.lm_head.name):
                    # 构建 lm_head
                    self.lm_head.build(None)
class TFMPNetClassificationHead(tf.keras.layers.Layer):
    """Head for sentence-level classification tasks."""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，输出维度为config.hidden_size，激活函数为tanh
        self.dense = tf.keras.layers.Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )
        # 添加一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 创建一个全连接层，输出维度为config.num_labels
        self.out_proj = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="out_proj"
        )
        self.config = config

    def call(self, features, training=False):
        # 取features的第一个token作为输入
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        x = self.out_proj(x)
        return x

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果dense层存在，就构建Dense层
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果out_proj层存在，就构建Dense层
        if getattr(self, "out_proj", None) is not None:
            with tf.name_scope(self.out_proj.name):
                self.out_proj.build([None, None, self.config.hidden_size])


@add_start_docstrings(
    """
    MPNet Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForSequenceClassification(TFMPNetPreTrainedModel, TFSequenceClassificationLoss):
    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels

        # 创建MPNet主体层
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 创建分类任务的头部
        self.classifier = TFMPNetClassificationHead(config, name="classifier")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels用于计算序列分类/回归损失。索引应在`[0, ..., config.num_labels - 1]`范围内。如果`config.num_labels == 1`，则计算回归损失（均方损失）, 如果`config.num_labels > 1`则计算分类损失（交叉熵）。
        """
        # 调用mpnet模型进行计算
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取序列输出
        sequence_output = outputs[0]
        # 根据序列输出计算logits
        logits = self.classifier(sequence_output, training=training)

        # 如果没有labels，则loss为None，否则调用hf_compute_loss函数计算loss
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果return_dict为False，则生成output结果
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回TFSequenceClassifierOutput对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已被构建，则直接返回
        if self.built:
            return
        self.built = True
        # 如果mpnet模型存在，则构建mpnet
        if getattr(self, "mpnet", None) is not None:
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)
        # 如果classifier模型存在，则构建classifier
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build(None)
# 定义一个 MPNet 模型，该模型在顶部包含一个用于多项选择分类的线性层（在汇集输出之上）和一个 softmax 层，例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    MPNet Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    MPNET_START_DOCSTRING,
)
class TFMPNetForMultipleChoice(TFMPNetPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 初始化 MPNet 主层
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 初始化 dropout 层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 初始化分类器，输出维度为 1
        self.classifier = tf.keras.layers.Dense(
            1, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存模型配置
        self.config = config

    # 定义模型前向传播方法
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    # 定义一个函数，用于处理输入并返回模型输出或损失值
    ) -> Union[TFMultipleChoiceModelOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ..., num_choices]`
            where `num_choices` is the size of the second dimension of the input tensors. (See `input_ids` above)
        """
        # 如果存在输入标识符，则获取选择数量和序列长度
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]

        # 将输入标识符压平，方便后续处理
        flat_input_ids = tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None
        flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None
        flat_position_ids = tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None
        flat_inputs_embeds = (
            tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3]))
            if inputs_embeds is not None
            else None
        )
        # 调用 mpnet 模型进行处理，得到输出
        outputs = self.mpnet(
            flat_input_ids,
            flat_attention_mask,
            flat_position_ids,
            head_mask,
            flat_inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        pooled_output = outputs[1]
        # 对输出应用 dropout 处理
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        # 如果存在标签，则计算损失值
        loss = None if labels is None else self.hf_compute_loss(labels, reshaped_logits)

        # 如果不需要返回字典，则返回损失值和输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回损失值、输出日志、隐藏状态和注意力
        return TFMultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 定义一个方法，用于建立模型
    def build(self, input_shape=None):
        # 如果已经建立过模型，则直接返回
        if self.built:
            return
        self.built = True
        # 如果存在 mpnet 模型，则构建 mpnet 模型
        if getattr(self, "mpnet", None) is not None:
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)
        # 如果存在分类器模型，则构建分类器模型
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
# 添加关于 MPNet 模型用于token分类任务的文档字符串
@add_start_docstrings(
    """
       MPNet Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
       Named-Entity-Recognition (NER) tasks.
       """,
    MPNET_START_DOCSTRING,
)
# 定义 TFMPNetForTokenClassification 类，继承自 TFMPNetPreTrainedModel 和 TFTokenClassificationLoss
class TFMPNetForTokenClassification(TFMPNetPreTrainedModel, TFTokenClassificationLoss):
    # 忽略某些在加载模型时丢失的键
    _keys_to_ignore_on_load_missing = [r"pooler"]

    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置标签数量
        self.num_labels = config.num_labels
        # 实例化 MPNet 主体层
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 实例化dropout层
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        # 实例化分类器Dense层
        self.classifier = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )
        # 保存配置
        self.config = config

    # 解包输入的装饰器
    @unpack_inputs
    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: tf.Tensor | None = None,
        training: bool = False,
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        # 文档字符串中的 labels 参数说明
        r"""
        labels (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 MPNet 主体层的前向传播
        outputs = self.mpnet(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出
        sequence_output = outputs[0]

        # 应用dropout
        sequence_output = self.dropout(sequence_output, training=training)
        # 应用分类器
        logits = self.classifier(sequence_output)

        # 计算loss
        loss = None if labels is None else self.hf_compute_loss(labels, logits)

        # 如果不需要返回字典，则返回一个tuple
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 否则返回 TFTokenClassifierOutput 对象
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    # 构建模型
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 将构建标志置为 True
        self.built = True
        # 如果存在 mpnet 属性
        if getattr(self, "mpnet", None) is not None:
            # 在 TensorFlow 中创建一个命名作用域，命名为 mpnet.name
            with tf.name_scope(self.mpnet.name):
                # 使用 mpnet 的 build 方法构建模型，输入形状为 None，表示未知
                self.mpnet.build(None)
        # 如果存在 classifier 属性
        if getattr(self, "classifier", None) is not None:
            # 在 TensorFlow 中创建一个命名作用域，命名为 classifier.name
            with tf.name_scope(self.classifier.name):
                # 使用 classifier 的 build 方法构建模型，输入形状为 [None, None, self.config.hidden_size]
                self.classifier.build([None, None, self.config.hidden_size])
# 使用装饰器为模型添加文档字符串，描述其用途和功能
@add_start_docstrings(
    """
    MPNet Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MPNET_START_DOCSTRING,
)
# 定义 TFMPNetForQuestionAnswering 类，继承自 TFMPNetPreTrainedModel 和 TFQuestionAnsweringLoss
class TFMPNetForQuestionAnswering(TFMPNetPreTrainedModel, TFQuestionAnsweringLoss):
    # 定义在加载模型时忽略的键
    _keys_to_ignore_on_load_missing = [r"pooler"]

    # 初始化函数，接受配置和其他参数
    def __init__(self, config, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(config, *inputs, **kwargs)
        # 设置模型输出类别数量
        self.num_labels = config.num_labels

        # 创建 MPNet 主层
        self.mpnet = TFMPNetMainLayer(config, name="mpnet")
        # 创建 QA 输出层，包括线性层和初始化方法
        self.qa_outputs = tf.keras.layers.Dense(
            config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 保存配置信息
        self.config = config

    # 使用装饰器为 call 方法添加文档字符串，描述输入输出参数
    @unpack_inputs
    @add_start_docstrings_to_model_forward(MPNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: Optional[Union[np.array, tf.Tensor]] = None,
        position_ids: Optional[Union[np.array, tf.Tensor]] = None,
        head_mask: Optional[Union[np.array, tf.Tensor]] = None,
        inputs_embeds: tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: tf.Tensor | None = None,
        end_positions: tf.Tensor | None = None,
        training: bool = False,
        **kwargs,
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        对于给定的输入张量，执行模型的前向传播并返回输出。
        start_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            起始位置的标签（索引），用于计算标记分类损失的起始位置。位置被限制在序列的长度 (`sequence_length`) 内。
            位于序列之外的位置不会用于计算损失。
        end_positions (`tf.Tensor` of shape `(batch_size,)`, *optional*):
            结束位置的标签（索引），用于计算标记分类损失的结束位置。位置被限制在序列的长度 (`sequence_length`) 内。
            位于序列之外的位置不会用于计算损失。
        """
        # 调用 MPNet 模型执行前向传播
        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 获取序列输出
        sequence_output = outputs[0]

        # 通过 QA 输出层获得预测的起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None

        # 如果提供了起始和结束位置标签，则计算损失
        if start_positions is not None and end_positions is not None:
            labels = {"start_position": start_positions, "end_position": end_positions}
            loss = self.hf_compute_loss(labels, (start_logits, end_logits))

        # 如果不返回字典格式的输出，则按照元组格式返回输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFQuestionAnsweringModelOutput 对象，其中包含损失、起始位置 logits、结束位置 logits、隐藏状态和注意力权重
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，则直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果存在 MPNet 模型，则构建 MPNet 层
        if getattr(self, "mpnet", None) is not None:
            with tf.name_scope(self.mpnet.name):
                self.mpnet.build(None)
        # 如果存在 QA 输出层，则构建 QA 输出层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```