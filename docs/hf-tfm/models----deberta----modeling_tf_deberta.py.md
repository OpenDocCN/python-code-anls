# `.\models\deberta\modeling_tf_deberta.py`

```py
# 定义 TFDebertaContextPooler 类，用于处理 DeBERTa 模型的上下文池化操作
class TFDebertaContextPooler(keras.layers.Layer):
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建一个全连接层，用于池化上下文表示
        self.dense = keras.layers.Dense(config.pooler_hidden_size, name="dense")
        # 初始化一个稳定的 Dropout 层，用于在训练过程中进行正则化
        self.dropout = TFDebertaStableDropout(config.pooler_dropout, name="dropout")
        # 存储配置信息
        self.config = config

    def call(self, hidden_states, training: bool = False):
        # 通过仅使用第一个 token 对应的隐藏状态来进行模型的“池化”
        context_token = hidden_states[:, 0]
        # 在训练过程中，应用 Dropout 正则化到 context_token
        context_token = self.dropout(context_token, training=training)
        # 将经过 Dropout 后的 context_token 输入全连接层
        pooled_output = self.dense(context_token)
        # 应用激活函数到池化后的输出
        pooled_output = get_tf_activation(self.config.pooler_hidden_act)(pooled_output)
        # 返回池化后的输出表示
        return pooled_output

    @property
    def output_dim(self) -> int:
        # 返回输出的维度，即隐藏大小
        return self.config.hidden_size
    # 定义神经网络模型的 build 方法，用于构建模型的结构
    def build(self, input_shape=None):
        # 如果模型已经构建完成，直接返回，避免重复构建
        if self.built:
            return
        # 将模型标记为已构建状态
        self.built = True
        # 如果存在 dense 属性（密集连接层），则构建该层
        if getattr(self, "dense", None) is not None:
            # 使用 tf.name_scope 为 dense 层创建命名空间，命名空间名称为 dense.name
            with tf.name_scope(self.dense.name):
                # 调用 dense 层的 build 方法，指定输入的形状为 [None, None, self.config.pooler_hidden_size]
                self.dense.build([None, None, self.config.pooler_hidden_size])
        # 如果存在 dropout 属性，则构建 dropout 层
        if getattr(self, "dropout", None) is not None:
            # 使用 tf.name_scope 为 dropout 层创建命名空间，命名空间名称为 dropout.name
            with tf.name_scope(self.dropout.name):
                # 调用 dropout 层的 build 方法，输入形状为 None（表示任意形状）
                self.dropout.build(None)
class TFDebertaXSoftmax(keras.layers.Layer):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`tf.Tensor`): The input tensor that will apply softmax.
        mask (`tf.Tensor`): The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs: tf.Tensor, mask: tf.Tensor):
        # 创建反向的掩码张量，将 mask 张量转换成布尔类型取反
        rmask = tf.logical_not(tf.cast(mask, tf.bool))
        # 将输入张量中掩码为 True 的位置置为负无穷，保证 softmax 计算时被忽略
        output = tf.where(rmask, float("-inf"), inputs)
        # 对处理后的张量应用稳定的 softmax 函数
        output = stable_softmax(output, self.axis)
        # 将之前处理的掩码位置重新置为 0.0，保证输出符合预期
        output = tf.where(rmask, 0.0, output)
        return output


class TFDebertaStableDropout(keras.layers.Layer):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    @tf.custom_gradient
    def xdropout(self, inputs):
        """
        Applies dropout to the inputs, as vanilla dropout, but also scales the remaining elements up by 1/drop_prob.
        """
        # 使用 Bernoulli 分布生成 dropout 掩码
        mask = tf.cast(
            1
            - tf.compat.v1.distributions.Bernoulli(probs=1.0 - self.drop_prob).sample(sample_shape=shape_list(inputs)),
            tf.bool,
        )
        # 计算缩放因子
        scale = tf.convert_to_tensor(1.0 / (1 - self.drop_prob), dtype=tf.float32)
        if self.drop_prob > 0:
            # 如果 dropout 概率大于 0，则对输入张量应用 dropout 并乘以缩放因子
            inputs = tf.where(mask, 0.0, inputs) * scale

        def grad(upstream):
            if self.drop_prob > 0:
                # 计算 dropout 操作的反向传播梯度
                return tf.where(mask, 0.0, upstream) * scale
            else:
                return upstream

        return inputs, grad

    def call(self, inputs: tf.Tensor, training: tf.Tensor = False):
        if training:
            # 在训练模式下应用自定义的 dropout 操作
            return self.xdropout(inputs)
        # 在推断模式下直接返回输入张量
        return inputs


class TFDebertaLayerNorm(keras.layers.Layer):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.eps = eps

    def build(self, input_shape):
        # 添加权重参数 gamma 和 beta
        self.gamma = self.add_weight(shape=[self.size], initializer=tf.ones_initializer(), name="weight")
        self.beta = self.add_weight(shape=[self.size], initializer=tf.zeros_initializer(), name="bias")
        return super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # 计算输入张量的均值、方差和标准差
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        std = tf.math.sqrt(variance + self.eps)
        # 应用 LayerNorm 公式，输出归一化后的张量
        return self.gamma * (x - mean) / std + self.beta


class TFDebertaSelfOutput(keras.layers.Layer):
    # 这部分代码还未完整给出，故不做注释
    pass
    # 初始化函数，用于创建一个新的实例
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 创建一个全连接层，用于处理隐藏状态，输出维度为config.hidden_size
        self.dense = keras.layers.Dense(config.hidden_size, name="dense")
        # 创建一个 LayerNormalization 层，设置 epsilon 为 config.layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # 创建一个 dropout 层，使用 TFDebertaStableDropout 类，dropout 概率为 config.hidden_dropout_prob
        self.dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="dropout")
        # 将 config 对象存储在实例中，供后续调用使用
        self.config = config

    # 前向传播函数，接收隐藏状态和输入张量，根据训练标志进行处理
    def call(self, hidden_states, input_tensor, training: bool = False):
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用 dropout 处理后的隐藏状态
        hidden_states = self.dropout(hidden_states, training=training)
        # 使用 LayerNormalization 层处理 dropout 后的隐藏状态和输入张量的和
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states

    # 构建函数，用于构建层的结构
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 设置标志为已构建
        self.built = True
        # 如果存在 dense 层，则构建 dense 层，并指定输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])
        # 如果存在 LayerNorm 层，则构建 LayerNorm 层，并指定输入形状为 [None, None, self.config.hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        # 如果存在 dropout 层，则构建 dropout 层，输入形状为 None
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
# 定义 TFDebertaAttention 类，继承自 keras 的 Layer 类，实现自定义的注意力层
class TFDebertaAttention(keras.layers.Layer):
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建 TFDebertaDisentangledSelfAttention 实例，用于自注意力机制
        self.self = TFDebertaDisentangledSelfAttention(config, name="self")
        # 创建 TFDebertaSelfOutput 实例，用于处理自注意力输出
        self.dense_output = TFDebertaSelfOutput(config, name="output")
        self.config = config

    # 定义 call 方法，实现层的前向传播逻辑
    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 使用 self.self 实例进行自注意力计算
        self_outputs = self.self(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        # 如果 query_states 为 None，则将其设置为输入张量 input_tensor
        if query_states is None:
            query_states = input_tensor
        # 使用 self.dense_output 实例处理自注意力输出
        attention_output = self.dense_output(
            hidden_states=self_outputs[0], input_tensor=query_states, training=training
        )
        # 将处理后的输出和 self_outputs 的其余部分组成元组返回
        output = (attention_output,) + self_outputs[1:]

        return output

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 self.self 实例存在，则在 tf 的命名空间下构建其参数
        if getattr(self, "self", None) is not None:
            with tf.name_scope(self.self.name):
                self.self.build(None)
        # 如果 self.dense_output 实例存在，则在 tf 的命名空间下构建其参数
        if getattr(self, "dense_output", None) is not None:
            with tf.name_scope(self.dense_output.name):
                self.dense_output.build(None)


# 定义 TFDebertaIntermediate 类，继承自 keras 的 Layer 类，实现中间层
class TFDebertaIntermediate(keras.layers.Layer):
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        # 创建 Dense 层实例，用于中间层的线性变换
        self.dense = keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        # 根据配置获取激活函数，用于中间层的非线性变换
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
        self.config = config

    # 定义 call 方法，实现层的前向传播逻辑
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 使用 Dense 层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 使用配置中的激活函数进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

    # 定义 build 方法，用于构建层的参数
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果 self.dense 实例存在，则在 tf 的命名空间下构建其参数
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.hidden_size])


class TFDebertaOutput(keras.layers.Layer):
    # 此处为 TFDebertaOutput 类的定义，未提供具体实现和方法，不需要添加额外注释
    # 初始化函数，接受一个DebertaConfig对象和其他关键字参数
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        # 创建一个全连接层，输出单元数为config中指定的隐藏层大小，
        # 初始化方式使用config中指定的initializer_range
        self.dense = keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )

        # 创建LayerNormalization层，epsilon值为config中指定的layer_norm_eps
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

        # 创建TFDebertaStableDropout层，dropout率为config中指定的hidden_dropout_prob
        self.dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="dropout")

        # 将传入的config对象保存在self.config中
        self.config = config

    # 调用函数，接收隐藏状态(hidden_states)、输入张量(input_tensor)和训练标志(training)，
    # 返回经过全连接层、dropout和LayerNormalization处理后的隐藏状态
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        # 使用self.dense对hidden_states进行全连接操作
        hidden_states = self.dense(inputs=hidden_states)

        # 使用self.dropout对全连接结果进行dropout操作，根据training参数决定是否使用训练模式
        hidden_states = self.dropout(hidden_states, training=training)

        # 将dropout后的结果与输入张量input_tensor相加，再使用self.LayerNorm进行LayerNormalization处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        # 返回处理后的隐藏状态
        return hidden_states

    # 构建函数，用于构建模型层次结构
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        
        # 标记为已经构建
        self.built = True
        
        # 如果存在self.dense属性，则使用tf.name_scope为dense层命名空间，
        # 并调用self.dense的build方法，传入输入形状[None, None, self.config.intermediate_size]
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                self.dense.build([None, None, self.config.intermediate_size])
        
        # 如果存在self.LayerNorm属性，则使用tf.name_scope为LayerNorm层命名空间，
        # 并调用self.LayerNorm的build方法，传入输入形状[None, None, self.config.hidden_size]
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                self.LayerNorm.build([None, None, self.config.hidden_size])
        
        # 如果存在self.dropout属性，则使用tf.name_scope为dropout层命名空间，
        # 并调用self.dropout的build方法，传入None作为输入形状
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
class TFDebertaLayer(keras.layers.Layer):
    # TFDebertaLayer 类定义，继承自 keras 的 Layer 类
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 初始化注意力、中间层和输出层组件
        self.attention = TFDebertaAttention(config, name="attention")
        self.intermediate = TFDebertaIntermediate(config, name="intermediate")
        self.bert_output = TFDebertaOutput(config, name="output")

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # 调用注意力机制模块，返回注意力输出
        attention_outputs = self.attention(
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = attention_outputs[0]
        
        # 调用中间层，传入注意力输出，得到中间层输出
        intermediate_output = self.intermediate(hidden_states=attention_output)
        
        # 调用输出层，传入中间层输出和注意力输出，得到最终层输出
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        
        # 构建返回的输出元组，包括最终层输出和可能的注意力信息
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        
        return outputs

    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        self.built = True
        
        # 构建注意力、中间层和输出层
        if getattr(self, "attention", None) is not None:
            with tf.name_scope(self.attention.name):
                self.attention.build(None)
        if getattr(self, "intermediate", None) is not None:
            with tf.name_scope(self.intermediate.name):
                self.intermediate.build(None)
        if getattr(self, "bert_output", None) is not None:
            with tf.name_scope(self.bert_output.name):
                self.bert_output.build(None)


class TFDebertaEncoder(keras.layers.Layer):
    # TFDebertaEncoder 类定义，继承自 keras 的 Layer 类
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 根据配置参数构建多层 TFDebertaLayer 组成的列表
        self.layer = [TFDebertaLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]
        
        # 检查是否使用相对注意力机制
        self.relative_attention = getattr(config, "relative_attention", False)
        self.config = config
        
        # 如果使用相对注意力，设置最大相对位置
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
    # 构建模型层，如果已经构建过则直接返回
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        # 如果使用相对注意力机制，添加相对位置嵌入权重
        if self.relative_attention:
            self.rel_embeddings = self.add_weight(
                name="rel_embeddings.weight",
                shape=[self.max_relative_positions * 2, self.config.hidden_size],
                initializer=get_initializer(self.config.initializer_range),
            )
        # 如果存在子层，则逐个构建这些子层
        if getattr(self, "layer", None) is not None:
            for layer in self.layer:
                with tf.name_scope(layer.name):
                    layer.build(None)

    # 获取相对位置嵌入权重
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings if self.relative_attention else None
        return rel_embeddings

    # 根据输入的注意力掩码，生成扩展后的注意力掩码
    def get_attention_mask(self, attention_mask):
        if len(shape_list(attention_mask)) <= 2:
            extended_attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 2)
            attention_mask = extended_attention_mask * tf.expand_dims(tf.squeeze(extended_attention_mask, -2), -1)
            attention_mask = tf.cast(attention_mask, tf.uint8)
        elif len(shape_list(attention_mask)) == 3:
            attention_mask = tf.expand_dims(attention_mask, 1)

        return attention_mask

    # 获取相对位置编码
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        # 如果使用相对注意力且没有提供相对位置编码，则根据隐藏状态的形状生成
        if self.relative_attention and relative_pos is None:
            q = shape_list(query_states)[-2] if query_states is not None else shape_list(hidden_states)[-2]
            relative_pos = build_relative_position(q, shape_list(hidden_states)[-2])
        return relative_pos

    # 模型的调用函数，处理输入的隐藏状态和注意力掩码
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        training: bool = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果输出隐藏状态为True，则初始化一个空元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重为True，则初始化一个空元组，否则为None
        all_attentions = () if output_attentions else None

        # 调用self对象的get_attention_mask方法，生成注意力掩码
        attention_mask = self.get_attention_mask(attention_mask)
        # 调用self对象的get_rel_pos方法，生成相对位置编码
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        # 如果hidden_states是一个序列对象，则将第一个元素作为next_kv，否则直接使用hidden_states
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states

        # 调用self对象的get_rel_embedding方法，生成相对位置嵌入
        rel_embeddings = self.get_rel_embedding()

        # 遍历self.layer中的每个层模块
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为True，则记录当前隐藏状态到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层模块的__call__方法，计算当前层的输出
            layer_outputs = layer_module(
                hidden_states=next_kv,
                attention_mask=attention_mask,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
                training=training,
            )
            # 更新hidden_states为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果query_states不为None，则更新query_states为当前隐藏状态
            if query_states is not None:
                query_states = hidden_states
                # 如果hidden_states是一个序列对象，则更新next_kv为下一个层的隐藏状态
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                # 否则直接更新next_kv为当前隐藏状态
                next_kv = hidden_states

            # 如果输出注意力权重为True，则记录当前层的注意力权重到all_attentions中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果return_dict为False，则返回非None的结果组成的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        # 如果return_dict为True，则返回TFBaseModelOutput对象，包含最后的隐藏状态、所有隐藏状态和所有注意力权重
        return TFBaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
def build_relative_position(query_size, key_size):
    """
    根据查询和键构建相对位置关系

    假设查询的绝对位置 \\(P_q\\) 范围在 (0, query_size)，键的绝对位置 \\(P_k\\) 范围在 (0, key_size)，
    查询到键的相对位置为 \\(R_{q \\rightarrow k} = P_q - P_k\\)

    Args:
        query_size (int): 查询的长度
        key_size (int): 键的长度

    Return:
        `tf.Tensor`: 形状为 [1, query_size, key_size] 的张量，表示相对位置索引

    """
    q_ids = tf.range(query_size, dtype=tf.int32)  # 生成查询位置的索引
    k_ids = tf.range(key_size, dtype=tf.int32)    # 生成键位置的索引
    rel_pos_ids = q_ids[:, None] - tf.tile(tf.reshape(k_ids, [1, -1]), [query_size, 1])  # 计算相对位置
    rel_pos_ids = rel_pos_ids[:query_size, :]     # 裁剪得到查询长度范围内的相对位置
    rel_pos_ids = tf.expand_dims(rel_pos_ids, axis=0)  # 扩展维度，形成 [1, query_size, key_size]
    return tf.cast(rel_pos_ids, tf.int64)          # 转换为 int64 类型的张量返回


def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    shapes = [
        shape_list(query_layer)[0],        # 查询层的批量大小
        shape_list(query_layer)[1],        # 查询层的序列长度
        shape_list(query_layer)[2],        # 查询层的隐藏单元数
        shape_list(relative_pos)[-1],      # 相对位置张量的最后一个维度大小
    ]
    return tf.broadcast_to(c2p_pos, shapes)  # 将 c2p_pos 广播扩展到指定形状的张量


def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    shapes = [
        shape_list(query_layer)[0],        # 查询层的批量大小
        shape_list(query_layer)[1],        # 查询层的序列长度
        shape_list(key_layer)[-2],         # 键层的序列长度
        shape_list(key_layer)[-2],         # 键层的序列长度
    ]
    return tf.broadcast_to(c2p_pos, shapes)  # 将 c2p_pos 广播扩展到指定形状的张量


def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    shapes = shape_list(p2c_att)[:2] + [shape_list(pos_index)[-2], shape_list(key_layer)[-2]]
    return tf.broadcast_to(pos_index, shapes)  # 将 pos_index 广播扩展到指定形状的张量


def torch_gather(x, indices, gather_axis):
    if gather_axis < 0:
        gather_axis = tf.rank(x) + gather_axis  # 将负数索引转换为正数索引

    if gather_axis != tf.rank(x) - 1:
        pre_roll = tf.rank(x) - 1 - gather_axis
        permutation = tf.roll(tf.range(tf.rank(x)), pre_roll, axis=0)  # 创建索引重排的置换
        x = tf.transpose(x, perm=permutation)   # 根据置换重新排列张量 x
        indices = tf.transpose(indices, perm=permutation)  # 根据置换重新排列索引张量 indices
    else:
        pre_roll = 0

    flat_x = tf.reshape(x, (-1, tf.shape(x)[-1]))    # 将张量 x 展平
    flat_indices = tf.reshape(indices, (-1, tf.shape(indices)[-1]))  # 将索引张量 indices 展平
    gathered = tf.gather(flat_x, flat_indices, batch_dims=1)  # 根据展平后的索引从 flat_x 中收集数据
    gathered = tf.reshape(gathered, tf.shape(indices))  # 将收集的数据重新 reshape 成原始索引张量的形状

    if pre_roll != 0:
        permutation = tf.roll(tf.range(tf.rank(x)), -pre_roll, axis=0)  # 创建索引重排的逆置换
        gathered = tf.transpose(gathered, perm=permutation)  # 根据逆置换重新排列 gathered 张量

    return gathered


class TFDebertaDisentangledSelfAttention(keras.layers.Layer):
    """
    Disentangled self-attention module

    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]

    """
    # 初始化函数，接受一个DebertaConfig对象和其他关键字参数
    def __init__(self, config: DebertaConfig, **kwargs):
        # 调用父类（AssumeRoleModel）的初始化方法
        super().__init__(**kwargs)
        # 检查隐藏大小是否是注意力头数的倍数，如果不是则引发异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算总的头大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 创建输入投影层，用于将输入转换为模型可用的格式
        self.in_proj = keras.layers.Dense(
            self.all_head_size * 3,
            kernel_initializer=get_initializer(config.initializer_range),
            name="in_proj",
            use_bias=False,
        )
        # 设置位置注意力类型，如果未指定，则为空列表
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        # 是否使用相对注意力机制和对话头模式的标志
        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)

        # 如果启用对话头模式，创建头权重和头选择的投影层
        if self.talking_head:
            self.head_logits_proj = keras.layers.Dense(
                self.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                name="head_logits_proj",
                use_bias=False,
            )
            self.head_weights_proj = keras.layers.Dense(
                self.num_attention_heads,
                kernel_initializer=get_initializer(config.initializer_range),
                name="head_weights_proj",
                use_bias=False,
            )

        # 使用自定义的softmax层（TFDebertaXSoftmax），在最后一个轴上进行softmax操作
        self.softmax = TFDebertaXSoftmax(axis=-1)

        # 如果启用相对注意力机制，配置最大相对位置，设置位置丢弃层，并根据pos_att_type设置位置投影层
        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="pos_dropout")
            if "c2p" in self.pos_att_type:
                self.pos_proj = keras.layers.Dense(
                    self.all_head_size,
                    kernel_initializer=get_initializer(config.initializer_range),
                    name="pos_proj",
                    use_bias=False,
                )
            if "p2c" in self.pos_att_type:
                self.pos_q_proj = keras.layers.Dense(
                    self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="pos_q_proj"
                )

        # 设置注意力概率丢弃层
        self.dropout = TFDebertaStableDropout(config.attention_probs_dropout_prob, name="dropout")
        # 保存配置信息
        self.config = config
    # 定义神经网络层的构建方法，初始化权重等操作
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True

        # 添加查询偏置权重
        self.q_bias = self.add_weight(
            name="q_bias", shape=(self.all_head_size), initializer=keras.initializers.Zeros()
        )
        # 添加数值偏置权重
        self.v_bias = self.add_weight(
            name="v_bias", shape=(self.all_head_size), initializer=keras.initializers.Zeros()
        )

        # 如果存在输入投影层，则构建该层
        if getattr(self, "in_proj", None) is not None:
            with tf.name_scope(self.in_proj.name):
                self.in_proj.build([None, None, self.config.hidden_size])

        # 如果存在 dropout 层，则构建该层
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)

        # 如果存在头部 logits 投影层，则构建该层
        if getattr(self, "head_logits_proj", None) is not None:
            with tf.name_scope(self.head_logits_proj.name):
                self.head_logits_proj.build(None)

        # 如果存在头部权重投影层，则构建该层
        if getattr(self, "head_weights_proj", None) is not None:
            with tf.name_scope(self.head_weights_proj.name):
                self.head_weights_proj.build(None)

        # 如果存在位置 dropout 层，则构建该层
        if getattr(self, "pos_dropout", None) is not None:
            with tf.name_scope(self.pos_dropout.name):
                self.pos_dropout.build(None)

        # 如果存在位置投影层，则构建该层
        if getattr(self, "pos_proj", None) is not None:
            with tf.name_scope(self.pos_proj.name):
                self.pos_proj.build([self.config.hidden_size])

        # 如果存在位置查询投影层，则构建该层
        if getattr(self, "pos_q_proj", None) is not None:
            with tf.name_scope(self.pos_q_proj.name):
                self.pos_q_proj.build([self.config.hidden_size])

    # 将输入张量重塑为注意力得分所需的形状
    def transpose_for_scores(self, tensor: tf.Tensor) -> tf.Tensor:
        # 获取张量的形状列表，去除最后一个维度，并将最后两个维度合并
        shape = shape_list(tensor)[:-1] + [self.num_attention_heads, -1]
        tensor = tf.reshape(tensor=tensor, shape=shape)

        # 将张量从 [batch_size, seq_length, all_head_size] 转置为 [batch_size, seq_length, num_attention_heads, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # 神经网络层的调用方法，执行注意力计算等操作
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        query_states: tf.Tensor = None,
        relative_pos: tf.Tensor = None,
        rel_embeddings: tf.Tensor = None,
        output_attentions: bool = False,
        training: bool = False,
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        # 如果未提供相对位置信息，根据查询层的形状获取相对位置
        if relative_pos is None:
            q = shape_list(query_layer)[-2]
            relative_pos = build_relative_position(q, shape_list(key_layer)[-2])
        
        shape_list_pos = shape_list(relative_pos)
        
        # 如果相对位置的形状是二维，则扩展维度使其成为四维
        if len(shape_list_pos) == 2:
            relative_pos = tf.expand_dims(tf.expand_dims(relative_pos, 0), 0)
        # 如果相对位置的形状是三维，则在第二个维度上扩展维度
        elif len(shape_list_pos) == 3:
            relative_pos = tf.expand_dims(relative_pos, 1)
        # 如果相对位置的形状不是二维或三维，则抛出异常
        elif len(shape_list_pos) != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {len(shape_list_pos)}")

        # 计算注意力跨度，确保不超过最大相对位置数，并转换为整型
        att_span = tf.cast(
            tf.minimum(
                tf.maximum(shape_list(query_layer)[-2], shape_list(key_layer)[-2]), self.max_relative_positions
            ),
            tf.int64,
        )
        
        # 根据注意力跨度选择相对位置嵌入，并扩展维度以匹配张量形状
        rel_embeddings = tf.expand_dims(
            rel_embeddings[self.max_relative_positions - att_span : self.max_relative_positions + att_span, :], 0
        )

        score = 0

        # 若位置注意力类型包含 "c2p"，执行内容到位置的注意力计算
        if "c2p" in self.pos_att_type:
            # 使用位置投影层对相对位置嵌入进行处理，并转置以便进行注意力计算
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            # 计算内容到位置的注意力分数
            c2p_att = tf.matmul(query_layer, tf.transpose(pos_key_layer, [0, 1, 3, 2]))
            # 对相对位置进行调整，并利用调整后的位置索引收集注意力分数
            c2p_pos = tf.clip_by_value(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_att = torch_gather(c2p_att, c2p_dynamic_expand(c2p_pos, query_layer, relative_pos), -1)
            score += c2p_att

        # 若位置注意力类型包含 "p2c"，执行位置到内容的注意力计算
        if "p2c" in self.pos_att_type:
            # 使用位置投影层对相对位置嵌入进行处理，并转置以便进行注意力计算
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            # 根据缩放因子对位置查询层进行归一化处理
            pos_query_layer /= tf.math.sqrt(tf.cast(shape_list(pos_query_layer)[-1] * scale_factor, dtype=tf.float32))
            # 如果查询层和键层的长度不同，重新构建相对位置
            if shape_list(query_layer)[-2] != shape_list(key_layer)[-2]:
                r_pos = build_relative_position(shape_list(key_layer)[-2], shape_list(key_layer)[-2])
            else:
                r_pos = relative_pos
            # 对位置到内容的相对位置进行调整，并利用调整后的位置索引收集注意力分数
            p2c_pos = tf.clip_by_value(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_att = tf.matmul(key_layer, tf.transpose(pos_query_layer, [0, 1, 3, 2]))
            p2c_att = tf.transpose(
                torch_gather(p2c_att, p2c_dynamic_expand(p2c_pos, query_layer, key_layer), -1), [0, 1, 3, 2]
            )
            # 如果查询层和键层的长度不同，利用位置索引对注意力分数进行再次调整
            if shape_list(query_layer)[-2] != shape_list(key_layer)[-2]:
                pos_index = tf.expand_dims(relative_pos[:, :, :, 0], -1)
                p2c_att = torch_gather(p2c_att, pos_dynamic_expand(pos_index, p2c_att, key_layer), -2)
            score += p2c_att

        # 返回最终的注意力分数
        return score
class TFDebertaEmbeddings(keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.position_biased_input = getattr(config, "position_biased_input", True)
        self.initializer_range = config.initializer_range
        if self.embedding_size != config.hidden_size:
            # 如果embedding_size不等于hidden_size，则使用全连接层进行投影
            self.embed_proj = keras.layers.Dense(
                config.hidden_size,
                kernel_initializer=get_initializer(config.initializer_range),
                name="embed_proj",
                use_bias=False,
            )
        # LayerNormalization层，用于标准化层的输出
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        # dropout层，用于随机丢弃一部分神经元，防止过拟合
        self.dropout = TFDebertaStableDropout(config.hidden_dropout_prob, name="dropout")

    def build(self, input_shape=None):
        with tf.name_scope("word_embeddings"):
            # 创建词嵌入权重矩阵，形状为[vocab_size, embedding_size]
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.embedding_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            if self.config.type_vocab_size > 0:
                # 如果有token_type信息，则创建token_type嵌入矩阵，形状为[type_vocab_size, embedding_size]
                self.token_type_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.config.type_vocab_size, self.embedding_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.token_type_embeddings = None

        with tf.name_scope("position_embeddings"):
            if self.position_biased_input:
                # 如果需要使用位置信息偏置，则创建位置嵌入矩阵，形状为[max_position_embeddings, hidden_size]
                self.position_embeddings = self.add_weight(
                    name="embeddings",
                    shape=[self.max_position_embeddings, self.hidden_size],
                    initializer=get_initializer(self.initializer_range),
                )
            else:
                self.position_embeddings = None

        if self.built:
            return
        self.built = True
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建LayerNormalization层
                self.LayerNorm.build([None, None, self.config.hidden_size])
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                # 构建dropout层
                self.dropout.build(None)
        if getattr(self, "embed_proj", None) is not None:
            with tf.name_scope(self.embed_proj.name):
                # 构建全连接投影层
                self.embed_proj.build([None, None, self.embedding_size])
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        mask: tf.Tensor = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        # 检查是否提供了有效的输入数据，至少需要提供 `input_ids` 或 `input_embeds`
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        # 如果提供了 `input_ids`，则使用权重张量 `self.weight` 来获取嵌入向量
        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        # 获取输入嵌入张量的形状
        input_shape = shape_list(inputs_embeds)[:-1]

        # 如果未提供 `token_type_ids`，则将其初始化为全零张量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 如果未提供 `position_ids`，则根据输入张量的最后一个维度创建位置张量
        if position_ids is None:
            position_ids = tf.expand_dims(tf.range(start=0, limit=input_shape[-1]), axis=0)

        # 初始的最终嵌入张量即为输入嵌入张量
        final_embeddings = inputs_embeds

        # 如果模型配置要求在输入中加入位置偏置
        if self.position_biased_input:
            position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
            final_embeddings += position_embeds

        # 如果模型配置要求加入类型标记嵌入
        if self.config.type_vocab_size > 0:
            token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
            final_embeddings += token_type_embeds

        # 如果嵌入大小与隐藏层大小不一致，则使用 `self.embed_proj` 进行投影
        if self.embedding_size != self.hidden_size:
            final_embeddings = self.embed_proj(final_embeddings)

        # 对最终嵌入张量进行 Layer Normalization 处理
        final_embeddings = self.LayerNorm(final_embeddings)

        # 如果提供了掩码张量 `mask`
        if mask is not None:
            # 如果掩码张量的维度与最终嵌入张量的维度不同，进行维度调整
            if len(shape_list(mask)) != len(shape_list(final_embeddings)):
                if len(shape_list(mask)) == 4:
                    mask = tf.squeeze(tf.squeeze(mask, axis=1), axis=1)
                mask = tf.cast(tf.expand_dims(mask, axis=2), tf.float32)

            # 应用掩码到最终嵌入张量上
            final_embeddings = final_embeddings * mask

        # 对最终嵌入张量应用 dropout，如果处于训练模式则启用
        final_embeddings = self.dropout(final_embeddings, training=training)

        # 返回处理后的最终嵌入张量
        return final_embeddings
# 定义 TFDebertaPredictionHeadTransform 类，作为 Keras 层
class TFDebertaPredictionHeadTransform(keras.layers.Layer):
    # 初始化方法，接受 DebertaConfig 对象和额外的关键字参数
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        
        # 根据配置获取嵌入大小，如果未指定则使用 hidden_size
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        
        # 创建一个全连接层，输出单元数为嵌入大小，初始化器为配置中的初始化范围
        self.dense = keras.layers.Dense(
            units=self.embedding_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )
        
        # 如果 hidden_act 是字符串，则根据字符串获取激活函数；否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        
        # 创建 LayerNormalization 层，使用配置中的 epsilon 值，命名为 LayerNorm
        self.LayerNorm = keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        
        # 保存配置对象
        self.config = config

    # 定义调用方法，接受隐藏状态张量并返回转换后的张量
    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # 先通过全连接层进行线性变换
        hidden_states = self.dense(inputs=hidden_states)
        # 然后应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 最后对结果应用 LayerNormalization
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states

    # 构建方法，用于构建层的权重
    def build(self, input_shape=None):
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果存在 dense 层，则构建其权重
        if getattr(self, "dense", None) is not None:
            with tf.name_scope(self.dense.name):
                # 构建时指定输入形状为 [None, None, hidden_size]
                self.dense.build([None, None, self.config.hidden_size])
        
        # 如果存在 LayerNorm 层，则构建其权重
        if getattr(self, "LayerNorm", None) is not None:
            with tf.name_scope(self.LayerNorm.name):
                # 构建时指定输入形状为 [None, None, embedding_size]
                self.LayerNorm.build([None, None, self.embedding_size])


# 定义 TFDebertaLMPredictionHead 类，作为 Keras 层
class TFDebertaLMPredictionHead(keras.layers.Layer):
    # 初始化方法，接受 DebertaConfig 对象、输入嵌入层和额外的关键字参数
    def __init__(self, config: DebertaConfig, input_embeddings: keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)
        
        # 保存配置对象和嵌入大小（默认为 hidden_size）
        self.config = config
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        
        # 创建 TFDebertaPredictionHeadTransform 实例，命名为 transform
        self.transform = TFDebertaPredictionHeadTransform(config, name="transform")
        
        # 输入的嵌入层
        self.input_embeddings = input_embeddings

    # 构建方法，用于构建层的权重
    def build(self, input_shape=None):
        # 添加一个全零的偏置，形状为 (vocab_size,)
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")
        
        # 如果已经构建过，直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果存在 transform 层，则构建其权重
        if getattr(self, "transform", None) is not None:
            with tf.name_scope(self.transform.name):
                self.transform.build(None)

    # 获取输出嵌入层
    def get_output_embeddings(self) -> keras.layers.Layer:
        return self.input_embeddings

    # 设置输出嵌入层
    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        # 更新嵌入层的词汇大小
        self.input_embeddings.vocab_size = shape_list(value)[0]

    # 获取偏置
    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    # 设置偏置
    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        # 更新配置中的词汇大小
        self.config.vocab_size = shape_list(value["bias"])[0]
    # 调用方法，输入隐藏状态张量，并通过 self.transform 方法进行转换
    hidden_states = self.transform(hidden_states=hidden_states)

    # 获取隐藏状态张量的序列长度
    seq_length = shape_list(hidden_states)[1]

    # 将隐藏状态张量重塑为二维张量，形状为 [-1, self.embedding_size]
    hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.embedding_size])

    # 执行矩阵乘法，计算隐藏状态张量与 self.input_embeddings.weight 的乘积，转置 self.input_embeddings.weight
    hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)

    # 将结果重新塑造为三维张量，形状为 [-1, seq_length, self.config.vocab_size]
    hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])

    # 使用偏置 self.bias 添加到隐藏状态张量上
    hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

    # 返回处理后的隐藏状态张量
    return hidden_states
class TFDebertaOnlyMLMHead(keras.layers.Layer):
    # 定义 TFDebertaOnlyMLMHead 类，继承自 keras 的 Layer 类
    def __init__(self, config: DebertaConfig, input_embeddings: keras.layers.Layer, **kwargs):
        # 初始化方法，接受 DebertaConfig 类型的 config 和 keras.Layer 类型的 input_embeddings 参数
        super().__init__(**kwargs)
        # 调用父类的初始化方法

        # 创建 TFDebertaLMPredictionHead 实例，并命名为 predictions
        self.predictions = TFDebertaLMPredictionHead(config, input_embeddings, name="predictions")

    # 定义 call 方法，接受 tf.Tensor 类型的 sequence_output 参数，返回 tf.Tensor 类型的 prediction_scores
    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        # 调用 self.predictions 的 __call__ 方法，传入 hidden_states=sequence_output 参数
        prediction_scores = self.predictions(hidden_states=sequence_output)

        # 返回 prediction_scores
        return prediction_scores

    # 定义 build 方法，接受 input_shape 参数，默认为 None
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 将 built 属性设置为 True，表示已经构建
        self.built = True
        
        # 如果 self.predictions 存在
        if getattr(self, "predictions", None) is not None:
            # 使用 tf.name_scope 来限定作用域为 self.predictions.name
            with tf.name_scope(self.predictions.name):
                # 调用 self.predictions 的 build 方法，传入 None 参数
                self.predictions.build(None)


# @keras_serializable
class TFDebertaMainLayer(keras.layers.Layer):
    # 类变量 config_class，指定为 DebertaConfig 类
    config_class = DebertaConfig

    # 初始化方法，接受 DebertaConfig 类型的 config 参数和其他关键字参数
    def __init__(self, config: DebertaConfig, **kwargs):
        super().__init__(**kwargs)
        # 调用父类的初始化方法

        # 将 config 参数赋值给 self.config
        self.config = config

        # 创建 TFDebertaEmbeddings 实例，并命名为 embeddings
        self.embeddings = TFDebertaEmbeddings(config, name="embeddings")
        
        # 创建 TFDebertaEncoder 实例，并命名为 encoder
        self.encoder = TFDebertaEncoder(config, name="encoder")

    # 返回 embeddings 属性
    def get_input_embeddings(self) -> keras.layers.Layer:
        return self.embeddings

    # 设置 embeddings 属性的权重和词汇表大小
    def set_input_embeddings(self, value: tf.Variable):
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    # _prune_heads 方法，用于剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    # 使用 unpack_inputs 装饰器，接受多个输入参数，并按需解包
    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
        # 省略部分参数...
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 如果同时指定了 input_ids 和 inputs_embeds，抛出数值错误异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids，则获取其形状信息
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        # 如果指定了 inputs_embeds，则获取其形状信息去掉最后一维
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则抛出数值错误异常
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 如果 attention_mask 为 None，则使用输入形状创建全为1的张量
        if attention_mask is None:
            attention_mask = tf.fill(dims=input_shape, value=1)

        # 如果 token_type_ids 为 None，则使用输入形状创建全为0的张量
        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        # 使用 embeddings 层处理输入，获取嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            mask=attention_mask,
            training=training,
        )

        # 使用 encoder 层处理嵌入输出，获取编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        # 获取编码器输出中的序列输出（通常是最后一层的隐藏状态）
        sequence_output = encoder_outputs[0]

        # 如果不要求返回字典形式的输出，则返回编码器的输出
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 如果要求返回字典形式的输出，则构造 TFBaseModelOutput 对象并返回
        return TFBaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过模型，直接返回
        if self.built:
            return
        # 标记模型已构建
        self.built = True
        # 如果模型具有 embeddings 属性，则构建 embeddings 层
        if getattr(self, "embeddings", None) is not None:
            with tf.name_scope(self.embeddings.name):
                self.embeddings.build(None)
        # 如果模型具有 encoder 属性，则构建 encoder 层
        if getattr(self, "encoder", None) is not None:
            with tf.name_scope(self.encoder.name):
                self.encoder.build(None)
"""
    The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build
    on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a [keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it
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

    Parameters:
        config ([`DebertaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
    Args:
        input_ids (`np.ndarray`, `tf.Tensor`, `List[tf.Tensor]` or `Dict[str, tf.Tensor]` or `Dict[str, np.ndarray]` and each example must have the shape `({0})`):
            # 输入序列的 token 索引，在词汇表中的索引表示。
            # 可以使用 [`AutoTokenizer`] 获取。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            # 避免对填充的 token 索引进行注意力计算的掩码。掩码值为 `[0, 1]`：

            - 1 表示 **未被掩码的** token，
            - 0 表示 **被掩码的** token。

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            # 段 token 索引，用于指示输入的第一部分和第二部分。索引选取在 `[0, 1]`：

            - 0 对应 *句子 A* 的 token，
            - 1 对应 *句子 B* 的 token。

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`np.ndarray` or `tf.Tensor` of shape `({0})`, *optional*):
            # 每个输入序列 token 在位置嵌入中的位置索引。选取范围在 `[0, config.max_position_embeddings - 1]`。

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`np.ndarray` or `tf.Tensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示而不是 `input_ids`。如果您希望更多地控制如何将 `input_ids` 索引转换为关联向量，
            # 则此选项非常有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
"""
@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
"""
# 定义 TFDebertaModel 类，继承自 TFDebertaPreTrainedModel 类
class TFDebertaModel(TFDebertaPreTrainedModel):
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 初始化 DeBERTa 主层，使用给定的配置
        self.deberta = TFDebertaMainLayer(config, name="deberta")

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 call 方法，接收多种输入并调用 DeBERTa 主层
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: Optional[bool] = False,
    ) -> Union[TFBaseModelOutput, Tuple[tf.Tensor]]:
        # 调用 DeBERTa 主层进行前向传播，返回输出
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        return outputs

    # 构建模型，确保 DeBERTa 主层已经被构建
    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)


"""
@add_start_docstrings("DeBERTa Model with a `language modeling` head on top.", DEBERTA_START_DOCSTRING)
"""
# 定义 TFDebertaForMaskedLM 类，继承自 TFDebertaPreTrainedModel 和 TFMaskedLanguageModelingLoss 类
class TFDebertaForMaskedLM(TFDebertaPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        # 如果配置中设定为解码器，发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `TFDebertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 DeBERTa 主层和 MLM 头部
        self.deberta = TFDebertaMainLayer(config, name="deberta")
        self.mlm = TFDebertaOnlyMLMHead(config, input_embeddings=self.deberta.embeddings, name="cls")

    # 返回 MLM 头部
    def get_lm_head(self) -> keras.layers.Layer:
        return self.mlm.predictions

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFMaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,  # 输入模型的输入序列的 ID，可以为 None
        attention_mask: np.ndarray | tf.Tensor | None = None,  # 可选的注意力掩码，用于指示哪些位置需要注意力
        token_type_ids: np.ndarray | tf.Tensor | None = None,  # 可选的标记类型 ID，用于区分不同类型的输入
        position_ids: np.ndarray | tf.Tensor | None = None,  # 可选的位置 ID，用于指示输入中的位置
        inputs_embeds: np.ndarray | tf.Tensor | None = None,  # 可选的嵌入输入，用于直接传递嵌入向量
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态
        return_dict: Optional[bool] = None,  # 是否以字典形式返回输出
        labels: np.ndarray | tf.Tensor | None = None,  # 可选的标签，用于计算掩码语言建模损失
        training: Optional[bool] = False,  # 是否处于训练模式
    ) -> Union[TFMaskedLMOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            计算掩码语言建模损失的标签。索引应在 `[-100, 0, ..., config.vocab_size]` 范围内。索引为 `-100` 的标记被忽略（掩码），
            损失仅计算标签在 `[0, ..., config.vocab_size]` 范围内的标记。
        """
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        sequence_output = outputs[0]  # 提取模型输出的序列输出
        prediction_scores = self.mlm(sequence_output=sequence_output, training=training)  # 使用序列输出预测得分
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=prediction_scores)  # 计算损失，如果没有标签则为 None

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]  # 如果不以字典形式返回，构建输出元组
            return ((loss,) + output) if loss is not None else output  # 返回损失和输出元组或者仅输出元组

        return TFMaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        if self.built:
            return
        self.built = True  # 标记模型已构建
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)  # 构建模型部件
        if getattr(self, "mlm", None) is not None:
            with tf.name_scope(self.mlm.name):
                self.mlm.build(None)  # 构建掩码语言建模部件
# 使用装饰器添加模型的文档字符串，说明这是一个在DeBERTa模型基础上增加了序列分类/回归头的Transformer模型
@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaForSequenceClassification(TFDebertaPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(config, *inputs, **kwargs)

        # 设置分类的标签数目
        self.num_labels = config.num_labels

        # 初始化DeBERTa主层和池化层
        self.deberta = TFDebertaMainLayer(config, name="deberta")
        self.pooler = TFDebertaContextPooler(config, name="pooler")

        # 从配置中获取分类器的dropout值或者使用默认的隐藏层dropout概率
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # 初始化稳定的Dropout层用于分类器
        self.dropout = TFDebertaStableDropout(drop_out, name="cls_dropout")
        # 初始化分类器的全连接层
        self.classifier = keras.layers.Dense(
            units=config.num_labels,
            kernel_initializer=get_initializer(config.initializer_range),
            name="classifier",
        )
        # 设置输出维度为池化层的输出维度
        self.output_dim = self.pooler.output_dim

    # 使用装饰器添加模型前向传播方法的文档字符串，描述输入参数和输出类型
    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFSequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ) -> Union[TFSequenceClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 使用 `->` 符号指定函数返回类型为 TFSequenceClassifierOutput 或者包含 tf.Tensor 的元组
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从模型输出中取出序列输出（第一个元素）
        sequence_output = outputs[0]
        # 将序列输出通过池化层得到汇总输出
        pooled_output = self.pooler(sequence_output, training=training)
        # 对汇总输出应用 dropout 操作
        pooled_output = self.dropout(pooled_output, training=training)
        # 将经过 dropout 处理后的汇总输出输入分类器，得到 logits
        logits = self.classifier(pooled_output)
        # 如果提供了 labels，则计算损失；否则将损失设为 None
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果 return_dict 为 False，则返回一个包含 logits 和额外输出的元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # 如果 return_dict 为 True，则返回一个 TFSequenceClassifierOutput 对象
        return TFSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回
        if self.built:
            return
        # 标记为已构建
        self.built = True
        
        # 如果存在 self.deberta，则在 TensorFlow 的命名空间下构建 self.deberta
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        
        # 如果存在 self.pooler，则在 TensorFlow 的命名空间下构建 self.pooler
        if getattr(self, "pooler", None) is not None:
            with tf.name_scope(self.pooler.name):
                self.pooler.build(None)
        
        # 如果存在 self.dropout，则在 TensorFlow 的命名空间下构建 self.dropout
        if getattr(self, "dropout", None) is not None:
            with tf.name_scope(self.dropout.name):
                self.dropout.build(None)
        
        # 如果存在 self.classifier，则在 TensorFlow 的命名空间下构建 self.classifier
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.output_dim])
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class TFDebertaForTokenClassification(TFDebertaPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.num_labels = config.num_labels  # 初始化模型的标签数量

        self.deberta = TFDebertaMainLayer(config, name="deberta")  # 使用配置初始化 DeBERTa 主层
        self.dropout = keras.layers.Dropout(rate=config.hidden_dropout_prob)  # 根据配置添加 dropout 层
        self.classifier = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="classifier"
        )  # 添加分类器层，输出维度为标签数量，使用配置的初始化器范围初始化
        self.config = config  # 保存配置对象

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFTokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
    ):
        """
        DeBERTa 模型的前向传播方法，处理输入并返回输出结果。

        Args:
            input_ids (TFModelInputType | None): 输入的 token IDs
            attention_mask (np.ndarray | tf.Tensor | None): 注意力遮罩
            token_type_ids (np.ndarray | tf.Tensor | None): token 类型 IDs
            position_ids (np.ndarray | tf.Tensor | None): 位置 IDs
            inputs_embeds (np.ndarray | tf.Tensor | None): 嵌入式输入
            output_attentions (Optional[bool]): 是否输出注意力权重
            output_hidden_states (Optional[bool]): 是否输出隐藏状态
            return_dict (Optional[bool]): 是否以字典形式返回结果
            labels (np.ndarray | tf.Tensor | None): 标签数据
            training (Optional[bool]): 是否在训练模式下

        Returns:
            TFTokenClassifierOutput: DeBERTa 模型的输出结果对象
        """
        # 调用 TFDebertaForTokenClassification 模型的前向传播
        # 详细参数和用法示例请参考文档和代码样例
        return super().call(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
            training=training,
        )
    ) -> Union[TFTokenClassifierOutput, Tuple[tf.Tensor]]:
        r"""
        labels (`tf.Tensor` or `np.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 调用 DeBERTa 模型进行前向传播，获取输出结果
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从 DeBERTa 模型的输出中获取序列输出（通常是隐藏状态）
        sequence_output = outputs[0]
        # 根据训练状态进行 dropout 操作，以防止过拟合
        sequence_output = self.dropout(sequence_output, training=training)
        # 将序列输出传递给分类器，生成分类预测 logits
        logits = self.classifier(inputs=sequence_output)
        # 如果有标签，则计算损失
        loss = None if labels is None else self.hf_compute_loss(labels=labels, logits=logits)

        # 如果不要求返回字典格式的输出，则按照元组形式返回结果
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TFTokenClassifierOutput 格式的输出，包括损失、logits、隐藏状态和注意力权重
        return TFTokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果模型已经构建，则直接返回
        if self.built:
            return
        # 设置模型为已构建状态
        self.built = True
        # 如果存在 DeBERTa 模型，则构建其参数
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果存在分类器模型，则构建其参数
        if getattr(self, "classifier", None) is not None:
            with tf.name_scope(self.classifier.name):
                self.classifier.build([None, None, self.config.hidden_size])
"""
DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
"""
# 基于DeBERTa模型，在其隐藏状态输出顶部添加一个用于提取性问答任务（如SQuAD）的跨度分类头部（通过线性层计算`span start logits`和`span end logits`）。

@add_start_docstrings(
    """
    添加文档字符串注释到模型的前向传播函数，描述输入的详细信息。
    """,
    DEBERTA_START_DOCSTRING,
)
# 装饰器：添加起始文档字符串到模型的前向传播函数，使用了预定义的DeBERTa起始文档字符串格式。

class TFDebertaForQuestionAnswering(TFDebertaPreTrainedModel, TFQuestionAnsweringLoss):
    # TFDebertaForQuestionAnswering类，继承自TFDebertaPreTrainedModel和TFQuestionAnsweringLoss。

    def __init__(self, config: DebertaConfig, *inputs, **kwargs):
        # 初始化函数，接收DebertaConfig类型的配置参数config和其他输入。

        super().__init__(config, *inputs, **kwargs)
        # 调用父类（TFDebertaPreTrainedModel和TFQuestionAnsweringLoss）的初始化函数。

        self.num_labels = config.num_labels
        # 设置类属性num_labels为配置参数config中的标签数目。

        self.deberta = TFDebertaMainLayer(config, name="deberta")
        # 创建TFDebertaMainLayer实例self.deberta，使用配置参数config并命名为"deberta"。

        self.qa_outputs = keras.layers.Dense(
            units=config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name="qa_outputs"
        )
        # 创建Dense层self.qa_outputs，用于输出QA任务结果，设置单元数为config.num_labels，初始化器使用config中的范围初始化器。

        self.config = config
        # 设置类属性config为配置参数config。

    @unpack_inputs
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TFQuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 装饰器：添加起始文档字符串到模型的前向传播函数，描述输入的详细信息，并添加代码示例文档字符串。

    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_positions: np.ndarray | tf.Tensor | None = None,
        end_positions: np.ndarray | tf.Tensor | None = None,
        training: Optional[bool] = False,
        # 定义模型的前向传播函数call，接收多个输入参数和可选的控制参数。
    ) -> Union[TFQuestionAnsweringModelOutput, Tuple[tf.Tensor]]:
        r"""
        start_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`tf.Tensor` or `np.ndarray` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 调用模型的前向传播函数，并获取输出
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )
        # 从模型输出中提取序列输出
        sequence_output = outputs[0]
        # 使用序列输出计算问题回答的 logits
        logits = self.qa_outputs(inputs=sequence_output)
        # 将 logits 分割为起始位置和结束位置的预测
        start_logits, end_logits = tf.split(value=logits, num_or_size_splits=2, axis=-1)
        # 移除起始位置和结束位置 logits 的最后一个维度，使得维度降为 (batch_size,)
        start_logits = tf.squeeze(input=start_logits, axis=-1)
        end_logits = tf.squeeze(input=end_logits, axis=-1)
        # 初始化损失为 None
        loss = None

        # 如果提供了起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            # 构建标签字典
            labels = {"start_position": start_positions}
            labels["end_position"] = end_positions
            # 使用损失计算函数计算损失
            loss = self.hf_compute_loss(labels=labels, logits=(start_logits, end_logits))

        # 如果不需要返回字典形式的输出，则按照元组形式返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则创建 TFQuestionAnsweringModelOutput 对象并返回
        return TFQuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def build(self, input_shape=None):
        # 如果已经构建过网络结构，则直接返回
        if self.built:
            return
        # 标记模型已经构建
        self.built = True
        # 如果存在 self.deberta 属性，则构建 self.deberta 模型
        if getattr(self, "deberta", None) is not None:
            with tf.name_scope(self.deberta.name):
                self.deberta.build(None)
        # 如果存在 self.qa_outputs 属性，则构建 self.qa_outputs 层
        if getattr(self, "qa_outputs", None) is not None:
            with tf.name_scope(self.qa_outputs.name):
                self.qa_outputs.build([None, None, self.config.hidden_size])
```