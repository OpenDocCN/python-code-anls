# `.\models\fuyu\modeling_fuyu.py`

```
    """
    The bare Fuyu Model outputting raw hidden-states without any specific head on top.

    This model inherits from `PreTrainedModel`. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch `torch.nn.Module` subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    """
    # 定义一个字符串，描述 Fuyu 模型及其用途，包括语言建模头部，用于基于图像补丁和文本进行因果语言建模。
    "Fuyu Model with a language modeling head on top for causal language model conditioned on image patches and text.",
    # 引用 FUYU_START_DOCSTRING，可能是一个预定义的文档字符串或标记。
    FUYU_START_DOCSTRING,
)
class FuyuForCausalLM(FuyuPreTrainedModel):
    # FuyuForCausalLM 类，继承自 FuyuPreTrainedModel 类
    def __init__(self, config: FuyuConfig):
        # 初始化方法，接受一个 FuyuConfig 类型的参数 config
        super().__init__(config)
        # 调用父类的初始化方法
        self.padding_idx = config.pad_token_id
        # 设置 padding_idx 属性为配置中的 pad_token_id
        self.vocab_size = config.vocab_size
        # 设置 vocab_size 属性为配置中的 vocab_size
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)
        # 创建一个 AutoModelForCausalLM 实例作为语言模型，使用给定的 text_config 配置

        self.vision_embed_tokens = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels, config.hidden_size
        )
        # 创建一个线性层，用于将图像嵌入的输入映射到隐藏大小的维度

        self.gradient_checkpointing = False
        # 初始化梯度检查点为 False，不使用梯度检查点优化

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取语言模型的输入嵌入
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置语言模型的输入嵌入
        self.language_model.set_input_embeddings(value)

    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: List[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """This function places the continuous_embeddings into the word_embeddings at the locations
        indicated by image_patch_input_indices. Different batch elements can have different numbers of continuous
        embeddings.

        Args:
            word_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Tensor of word embeddings.
            continuous_embeddings (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
                Tensor of continuous embeddings. The length of the list is the batch size. Each entry is shape
                [num_image_embeddings, hidden], and num_image_embeddings needs to match the number of non-negative
                indices in image_patch_input_indices for that batch element.
            image_patch_input_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Tensor of indices of the image patches in the input_ids tensor.
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"Batch sizes must match! Got {len(continuous_embeddings)=} and {word_embeddings.shape[0]=}"
            )

        # Clone the word_embeddings tensor to preserve the original and store modified embeddings
        output_embeddings = word_embeddings.clone()
        
        # Iterate through each batch element
        for batch_idx in range(word_embeddings.shape[0]):
            # Find indices in word_embeddings where non-negative values exist in image_patch_input_indices
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            
            # Retrieve corresponding indices from image_patch_input_indices to locate continuous_embeddings
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            
            # Check if the number of continuous embeddings matches the number of indices
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"Number of continuous embeddings {continuous_embeddings[batch_idx].shape=} does not match "
                    f"number of continuous token ids {src_indices.shape=} in batch element {batch_idx}."
                )
            
            # Replace selected word embeddings with corresponding continuous embeddings
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        
        # Return the modified output_embeddings tensor
        return output_embeddings
    # 定义一个方法用于模型的前向传播
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs
        image_patches: torch.Tensor = None,  # 图像片段张量，形状为 [batch_size, num_total_patches, patch_size x patch_size x num_channels]
        image_patches_indices: torch.Tensor = None,  # 图像片段的索引张量
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs 张量
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 用于存储过去的键值对的列表
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入张量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        labels: Optional[torch.Tensor] = None,  # 标签张量
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出
    ):
        # 如果传入了 `past_key_values`，则仅使用最后一个 token 的输入
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 获取额外的关键字参数中的 `position_ids`
        position_ids = kwargs.get("position_ids", None)
        # 如果存在 `attention_mask` 且不存在 `position_ids`
        if attention_mask is not None and position_ids is None:
            # 动态生成位置 IDs 用于批量生成
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在 `past_key_values`，则仅使用最后一个位置 ID
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # 如果传入了 `inputs_embeds`，则仅在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 如果存在 `image_patches_indices`，将其添加到模型输入中
        if image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices

        # 更新模型输入字典
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # 如果存在 `past_key_values`，则将以下两个键置为 None
                "image_patches_indices": image_patches_indices if past_key_values is None else None,
                "image_patches": image_patches if past_key_values is None else None,
            }
        )
        # 返回模型输入字典
        return model_inputs
```