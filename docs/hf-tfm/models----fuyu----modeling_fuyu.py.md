# `.\models\fuyu\modeling_fuyu.py`

```py
# coding=utf-8
# 以 UTF-8 编码的文件

# 版权信息
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
#
# 根据 Apache License, Version 2.0 授权
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依此许可证分发的软件以 "原样" 分发，
# 不附带任何明示或暗示的担保或条件。
# 有关特定语言的许可证，查看特定语言的许可证
""" PyTorch Fuyu model."""
# 导入所需的模块和包
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
# 导入自定义的模块和包
from ...modeling_outputs import CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...models.auto.modeling_auto import AutoModelForCausalLM
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_fuyu import FuyuConfig
# 获取日志记录器
logger = logging.get_logger(__name__)
# 用于文档化的配置对象
_CONFIG_FOR_DOC = "FuyuConfig"
# Fuyu 模型的文档字符串
FUYU_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FuyuConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 添加文档字符串
@add_start_docstrings(
    "The bare Fuyu Model outputting raw hidden-states without any specific head on top.",
    FUYU_START_DOCSTRING,
)
# Fuyu 预训练模型类
class FuyuPreTrainedModel(PreTrainedModel):
    config_class = FuyuConfig
    base_model_prefix = "fuyu"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    # 初始化权重方法
    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# Fuyu 模型输入文档字符串
FUYU_INPUTS_DOCSTRING = r"""
"""

# 添加文档字符串
@add_start_docstrings(
    # 定义了一个描述 Fuyu 模型的字符串，包含了对该模型的语言建模头，以及对图像补丁和文本进行条件化的因果语言模型
    "Fuyu Model with a language modeling head on top for causal language model conditioned on image patches and text.",
    # 调用 FUYU_START_DOCSTRING，可能用于生成模型文档
    FUYU_START_DOCSTRING,
# 定义一个名为 FuyuForCausalLM 的类，继承自 FuyuPreTrainedModel
class FuyuForCausalLM(FuyuPreTrainedModel):
    # 初始化方法，接受一个 FuyuConfig 类型的参数
    def __init__(self, config: FuyuConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置 padding 索引为 config 中的 pad_token_id
        self.padding_idx = config.pad_token_id
        # 设置词汇表大小为 config 中的 vocab_size
        self.vocab_size = config.vocab_size
        # 使用 AutoModelForCausalLM 的配置创建一个语言模型
        self.language_model = AutoModelForCausalLM.from_config(config.text_config)

        # 将图像的 token 嵌入转换为隐藏大小的线性层
        self.vision_embed_tokens = nn.Linear(
            config.patch_size * config.patch_size * config.num_channels, config.hidden_size
        )

        # 是否使用梯度检查点优化，默认为 False
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 收集连续嵌入
    def gather_continuous_embeddings(
        self,
        word_embeddings: torch.Tensor,
        continuous_embeddings: List[torch.Tensor],
        image_patch_input_indices: torch.Tensor,
    ) -> torch.Tensor:
        """将连续的嵌入放入单词嵌入中，位置由image_patch_input_indices指定。不同的批次元素可以有不同数量的连续嵌入。

        Args:
            word_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                词嵌入张量。
            continuous_embeddings (`torch.FloatTensor` of shape `(batch_size, num_patches, hidden_size)`):
                连续嵌入的张量。列表的长度为批次大小。每个条目的形状为 [num_image_embeddings, hidden]，num_image_embeddings 需要匹配该批次元素中 image_patch_input_indices 中非负索引的数量。
            image_patch_input_indices (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                在 input_ids 张量中图像补丁的索引张量。
        """
        if not (word_embeddings.shape[0] == len(continuous_embeddings)):
            raise ValueError(
                f"批次大小必须匹配！得到的 {len(continuous_embeddings)=} 和 {word_embeddings.shape[0]=}"
            )

        output_embeddings = word_embeddings.clone()
        for batch_idx in range(word_embeddings.shape[0]):
            # 首先找到 image_patch_input_indices 中所有非负值的位置，这些位置是我们要用 continuous_embeddings 内容替换的 word_embeddings 中的位置。
            dst_indices = torch.nonzero(image_patch_input_indices[batch_idx] >= 0, as_tuple=True)[0]
            # 接下来查找这些索引在 image_patch_input_indices 中，以查找在 continuous_embeddings 中要用来替换 word_embeddings 中值的索引。
            src_indices = image_patch_input_indices[batch_idx][dst_indices]
            # 检查是否有更多的索引比嵌入。请注意，如果图像被截断，可能会有较少的索引。
            if src_indices.shape[0] > continuous_embeddings[batch_idx].shape[0]:
                raise ValueError(
                    f"连续嵌入的数量 {continuous_embeddings[batch_idx].shape=} 与批次元素 {batch_idx} 中的连续标记 id 的数量 {src_indices.shape=} 不匹配。"
                )
            output_embeddings[batch_idx, dst_indices] = continuous_embeddings[batch_idx][src_indices]
        return output_embeddings

    @add_start_docstrings_to_model_forward(FUYU_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的标记 ID 张量，默认为空
        image_patches: torch.Tensor = None,  # 输入的图像分块张量，[batch_size, num_total_patches, patch_size_ x patch_size x num_channels ]
        image_patches_indices: torch.Tensor = None,  # 图像分块的索引张量，默认为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，默认为空
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID 张量，默认为空
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值列表，默认为空
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入张量，默认为空
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为空
        labels: Optional[torch.Tensor] = None,  # 标签张量，默认为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为空
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为空
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入的标记 ID 张量
        past_key_values=None,  # 过去的键值，默认为空
        attention_mask=None,  # 注意力掩码，默认为空
        inputs_embeds=None,  # 嵌入输入，默认为空
        image_patches=None,  # 图像分块，默认为空
        image_patches_indices=None,  # 图像分块索引，默认为空
        **kwargs,  # 其他关键字参数
    ):
        # 如果存在过去的键值，则仅使用最后一个输入标记
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # 获取位置 ID
        position_ids = kwargs.get("position_ids", None)
        # 如果存在注意力掩码但不存在位置 ID，则在批次生成时动态创建位置 ID
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在过去的键值，则仅使用最后一个位置 ID
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # 如果传入了嵌入输入，且不存在过去的键值，则只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 如果存在图像分块索引，则将其添加到模型输入中
        if image_patches_indices is not None:
            model_inputs["image_patches_indices"] = image_patches_indices

        # 更新模型输入字典
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # 如果不存在过去的键值，则添加图像分块索引和图像分块
                "image_patches_indices": image_patches_indices if past_key_values is None else None,
                "image_patches": image_patches if past_key_values is None else None,
            }
        )
        return model_inputs
```