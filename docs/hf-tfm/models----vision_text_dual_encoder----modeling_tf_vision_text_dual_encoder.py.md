# `.\models\vision_text_dual_encoder\modeling_tf_vision_text_dual_encoder.py`

```
# 定义一个文本与视觉双编码模型的文档字符串，用于说明如何初始化并使用预训练的视觉和文本编码器。
VISION_TEXT_DUAL_ENCODER_START_DOCSTRING = r"""
    This class can be used to initialize a vision-text dual encoder model with any pretrained vision autoencoding model
    as the vision encoder and any pretrained text model as the text encoder. The vision and text encoders are loaded
    via the [`~TFAutoModel.from_pretrained`] method. The projection layers are automatically added to the model and
    should be fine-tuned on a downstream task, like contrastive image-text modeling.

    In [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991) it is shown how
    leveraging pre-trained (locked/frozen) image and text model for contrastive learning yields significant improvement
    on new zero-shot vision tasks such as image classification or retrieval.

    After such a Vision-Text-Dual-Encoder model has been trained/fine-tuned, it can be saved/loaded just like any other
    models (see the examples for more information).

    This model inherits from [`TFPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a Keras [Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model) subclass. Use it as a
    regular Keras Model and refer to the TF documentation for all matter related to general usage and behavior.
"""
    Parameters:
        config ([`VisionEncoderDecoderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~TFPreTrainedModel.from_pretrained`] method to load the model weights.
"""
Args:
    input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
        输入序列 token 的索引，位于词汇表中。默认情况下会忽略填充部分。
        可以使用 `PreTrainedTokenizer` 获取索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

        [什么是输入 ID？](../glossary#input-ids)
    attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        避免在填充 token 索引上执行注意力的掩码。掩码值在 `[0, 1]` 范围内：

        - 对于 **未被掩码** 的 token，为 1，
        - 对于 **被掩码** 的 token，为 0。

        [什么是注意力掩码？](../glossary#attention-mask)
    position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
        每个输入序列 token 在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]` 内。

        [什么是位置 ID？](../glossary#position-ids)
    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。查看返回张量下的 `attentions` 获取更多细节。
    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。查看返回张量下的 `hidden_states` 获取更多细节。
    return_dict (`bool`, *optional*):
        是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
    # 输入参数为输入序列的标记索引，形状为(batch_size, sequence_length)
    # 使用AutoTokenizer获取输入索引，详见PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__
    input_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`):
    
    # 可选参数，注意力掩码，形状为(batch_size, sequence_length)
    # 用于避免在填充标记索引上执行注意力操作，值为0和1：
    # - 1表示**未遮蔽**的标记
    # - 0表示**遮蔽**的标记
    attention_mask (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
    
    # 可选参数，位置索引，形状为(batch_size, sequence_length)
    # 每个输入序列标记在位置嵌入中的位置索引，取值范围为[0, config.max_position_embeddings - 1]
    position_ids (`tf.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
    
    # 输入参数为像素值，形状为(batch_size, num_channels, height, width)
    # 像素值。如果提供填充，则默认忽略。可以使用图像处理器获取像素值（例如，使用ViT作为编码器时应使用AutoImageProcessor）。
    # 详见ViTImageProcessor.__call__
    pixel_values (`tf.Tensor` of shape `(batch_size, num_channels, height, width)`):
    
    # 可选参数，是否返回对比损失
    return_loss (`bool`, *optional*):
    
    # 可选参数，是否返回所有注意力层的注意力张量
    # 返回值中包含更多详细信息，详见返回的tensors下的attentions
    output_attentions (`bool`, *optional*):
    
    # 可选参数，是否返回所有层的隐藏状态
    # 返回值中包含更多详细信息，详见返回的tensors下的hidden_states
    output_hidden_states (`bool`, *optional*):
    
    # 可选参数，是否返回utils.ModelOutput而不是普通元组
    return_dict (`bool`, *optional*):
"""
# 从 transformers.models.clip.modeling_tf_clip.contrastive_loss 复制的对比损失函数定义
def contrastive_loss(logits: tf.Tensor) -> tf.Tensor:
    # 计算稀疏分类交叉熵损失的均值，用于对比损失的计算
    return tf.math.reduce_mean(
        keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(shape_list(logits)[0]), y_pred=logits, from_logits=True
        )
    )


# 从 transformers.models.clip.modeling_tf_clip.clip_loss 复制的 CLIP 损失函数定义
def clip_loss(similarity: tf.Tensor) -> tf.Tensor:
    # 计算标题和图像的对比损失，由对比损失函数 contrastive_loss 计算
    caption_loss = contrastive_loss(similarity)
    # 转置相似度矩阵并计算图像和标题的对比损失，同样使用 contrastive_loss 函数
    image_loss = contrastive_loss(tf.transpose(similarity))
    # 返回标题损失和图像损失的平均值作为 CLIP 损失
    return (caption_loss + image_loss) / 2.0


# 使用 @add_start_docstrings(VISION_TEXT_DUAL_ENCODER_START_DOCSTRING) 装饰器注释的双编码器模型类
class TFVisionTextDualEncoderModel(TFPreTrainedModel):
    # 指定模型配置类
    config_class = VisionTextDualEncoderConfig
    # 指定基础模型前缀
    base_model_prefix = "vision_text_dual_encoder"
    # 指定加载权重前缀
    load_weight_prefix = "tf_vision_text_dual_encoder_model"

    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[TFPreTrainedModel] = None,
        text_model: Optional[TFPreTrainedModel] = None,
    ):
        # 如果未提供配置且视觉模型或文本模型任一未提供，则引发 ValueError
        if config is None and (vision_model is None or text_model is None):
            raise ValueError("Either a configuration or an vision and a text model has to be provided")

        # 如果未提供配置，则从视觉和文本模型的配置中创建 VisionTextDualEncoderConfig
        if config is None:
            config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_model.config, text_model.config)
        else:
            # 如果提供的配置不是 VisionTextDualEncoderConfig 类型，则引发 ValueError
            if not isinstance(config, self.config_class):
                raise ValueError(f"config: {config} has to be of type {self.config_class}")

        # 使用配置初始化父类
        super().__init__(config)

        # 如果未提供视觉模型，则根据配置创建适当的视觉模型
        if vision_model is None:
            if isinstance(config.vision_config, CLIPVisionConfig):
                vision_model = TFCLIPVisionModel.from_config(config.vision_config, name="vision_model")
            else:
                vision_model = TFAutoModel.from_config(config.vision_config, name="vision_model")

        # 如果未提供文本模型，则根据配置创建适当的文本模型
        if text_model is None:
            text_model = TFAutoModel.from_config(config.text_config, name="text_model")

        # 分别设置视觉模型和文本模型
        self.vision_model = vision_model
        self.text_model = text_model

        # 确保各模型的配置引用共享配置，以保持配置更新同步
        self.vision_model.config = self.config.vision_config
        self.text_model.config = self.config.text_config

        # 设置视觉嵌入维度、文本嵌入维度和投影维度
        self.vision_embed_dim = config.vision_config.hidden_size
        self.text_embed_dim = config.text_config.hidden_size
        self.projection_dim = config.projection_dim

        # 定义视觉和文本的投影层，不使用偏置项
        self.visual_projection = keras.layers.Dense(self.projection_dim, use_bias=False, name="visual_projection")
        self.text_projection = keras.layers.Dense(self.projection_dim, use_bias=False, name="text_projection")

        # 初始化日志尺度为 None
        self.logit_scale = None

        # 设置模型配置
        self.config = config
    # 在构建方法中构建模型，确保命名正确
    def build(self, input_shape=None):
        # 如果已经构建过，则直接返回，避免重复构建
        if self.built:
            return
        # 设置标志表示模型已经构建
        self.built = True
        # 使用常量初始化器设置logit_scale权重，shape为(1,)
        initializer = keras.initializers.Constant(self.config.logit_scale_init_value)
        self.logit_scale = self.add_weight(shape=(1,), initializer=initializer, name="logit_scale")

        # 如果存在visual_projection属性，则构建它并设置命名空间
        if getattr(self, "visual_projection", None) is not None:
            with tf.name_scope(self.visual_projection.name):
                self.visual_projection.build([None, None, self.vision_embed_dim])
        
        # 如果存在text_projection属性，则构建它并设置命名空间
        if getattr(self, "text_projection", None) is not None:
            with tf.name_scope(self.text_projection.name):
                self.text_projection.build([None, None, self.text_embed_dim])
        
        # 设置vision_model的命名空间并构建其模型
        with tf.name_scope(self.vision_model.name):
            self.vision_model.build(None)
        
        # 设置text_model的命名空间并构建其模型
        with tf.name_scope(self.text_model.name):
            self.text_model.build(None)

    # 将TensorFlow的权重名称转换为PyTorch风格的权重名称
    def tf_to_pt_weight_rename(self, tf_weight):
        # 如果权重名称中包含"vision_model"，则根据不同情况进行重命名处理
        if "vision_model" in tf_weight:
            if tf_weight.count("vision_model") == 1:
                return (re.sub(r"vision_model\..*?\.", "vision_model.", tf_weight),)
            elif tf_weight.count("vision_model") == 2:
                return (re.sub(r"vision_model\..*?\.vision_model", "vision_model.vision_model", tf_weight),)
            else:
                raise ValueError(
                    f"Unexpected weight name {tf_weight}. Please file an issue on the"
                    " Transformers repo to let us know about this error!"
                )
        # 如果权重名称中包含"text_model"，则进行相应的重命名处理
        elif "text_model" in tf_weight:
            return (re.sub(r"text_model\..*?\.", "text_model.", tf_weight),)
        # 如果以上条件都不符合，则返回原始的权重名称
        else:
            return (tf_weight,)

    # 添加模型前向传播的文档字符串，并用VISION_TEXT_DUAL_ENCODER_TEXT_INPUTS_DOCSTRING进行注释
    def get_text_features(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        token_type_ids=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            text_features (`tf.Tensor` of shape `(batch_size, output_dim`): The text embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPTextModel`].

        Examples:

        ```python
        >>> from transformers import TFVisionTextDualEncoderModel, AutoTokenizer

        >>> model = TFVisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian", from_pt=True)
        >>> tokenizer = AutoTokenizer.from_pretrained("clip-italian/clip-italian")

        >>> inputs = tokenizer(["una foto di un gatto", "una foto di un cane"], padding=True, return_tensors="np")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # 使用 self.text_model 处理输入，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中获取池化后的输出
        pooled_output = text_outputs[1]
        # 使用 self.text_projection 对池化输出进行投影，得到文本特征
        text_features = self.text_projection(pooled_output)

        return text_features

    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_VISION_INPUTS_DOCSTRING)
    def get_image_features(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Returns:
            image_features (`tf.Tensor` of shape `(batch_size, output_dim`): The image embeddings obtained by applying
            the projection layer to the pooled output of [`TFCLIPVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import TFVisionTextDualEncoderModel, AutoImageProcessor

        >>> model = TFVisionTextDualEncoderModel.from_pretrained("clip-italian/clip-italian", from_pt=True)
        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="np")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        # 使用 self.vision_model 处理输入，获取视觉输出
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从视觉输出中获取池化后的输出
        pooled_output = vision_outputs[1]  # pooled_output
        # 使用 self.visual_projection 对池化输出进行投影，得到图像特征
        image_features = self.visual_projection(pooled_output)

        return image_features

    @unpack_inputs
    @add_start_docstrings_to_model_forward(VISION_TEXT_DUAL_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFCLIPOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `call`，用于执行模型推理或训练过程的输入处理和参数设置
    def call(
        self,
        input_ids: tf.Tensor | None = None,  # 输入文本的token IDs张量，默认为None
        pixel_values: tf.Tensor | None = None,  # 输入图像的像素值张量，默认为None
        attention_mask: tf.Tensor | None = None,  # 注意力掩码张量，默认为None
        position_ids: tf.Tensor | None = None,  # 位置编码张量，默认为None
        return_loss: Optional[bool] = None,  # 是否返回损失张量的布尔值，可选，默认为None
        token_type_ids: tf.Tensor | None = None,  # token类型 IDs 张量，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量的布尔值，可选，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态张量的布尔值，可选，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出结果的布尔值，可选，默认为None
        training: bool = False,  # 是否为训练模式的布尔值，默认为False
    ):
        
    # 类方法，用于从预训练的视觉-文本模型加载模型
    @classmethod
    def from_vision_text_pretrained(
        cls,
        vision_model_name_or_path: str = None,  # 视觉模型名称或路径的字符串，默认为None
        text_model_name_or_path: str = None,  # 文本模型名称或路径的字符串，默认为None
        *model_args,  # 模型参数的位置参数
        **kwargs,  # 模型参数的关键字参数
    ):
        
    # 属性方法，返回构建网络所需的虚拟输入数据字典
    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            `Dict[str, tf.Tensor]`: The dummy inputs.
        """
        # 使用预定义的虚拟输入数据构建输入文本的token IDs张量
        input_ids = tf.constant(DUMMY_INPUTS, dtype=tf.int32)
        batch_size, seq_len = input_ids.shape

        # 使用随机生成的虚拟输入数据构建输入图像的像素值张量
        VISION_DUMMY_INPUTS = tf.random.uniform(
            shape=(
                batch_size,
                self.config.vision_config.num_channels,
                self.config.vision_config.image_size,
                self.config.vision_config.image_size,
            ),
            dtype=tf.float32,
        )
        pixel_values = tf.constant(VISION_DUMMY_INPUTS)
        # 构建并返回包含虚拟输入数据的字典
        dummy = {"pixel_values": pixel_values, "input_ids": input_ids}
        return dummy
```