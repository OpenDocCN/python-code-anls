# `.\models\lxmert\configuration_lxmert.py`

```
# coding=utf-8
# Copyright 2018, Hao Tan, Mohit Bansal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" LXMERT model configuration"""


from ...configuration_utils import PretrainedConfig  # 导入预训练配置的类
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/config.json",
}


class LxmertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LxmertModel`] or a [`TFLxmertModel`]. It is used
    to instantiate a LXMERT model according to the specified arguments, defining the model architecture. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the Lxmert
    [unc-nlp/lxmert-base-uncased](https://huggingface.co/unc-nlp/lxmert-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    """

    model_type = "lxmert"  # 设置模型类型为 "lxmert"
    attribute_map = {}  # 定义一个空的属性映射字典

    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小默认为 30522
        hidden_size=768,  # 隐藏层大小默认为 768
        num_attention_heads=12,  # 注意力头数目默认为 12
        num_qa_labels=9500,  # QA 标签数目默认为 9500
        num_object_labels=1600,  # 对象标签数目默认为 1600
        num_attr_labels=400,  # 属性标签数目默认为 400
        intermediate_size=3072,  # 中间层大小默认为 3072
        hidden_act="gelu",  # 隐藏层激活函数默认为 gelu
        hidden_dropout_prob=0.1,  # 隐藏层 dropout 概率默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率 dropout 概率默认为 0.1
        max_position_embeddings=512,  # 最大位置嵌入数默认为 512
        type_vocab_size=2,  # 类型词汇表大小默认为 2
        initializer_range=0.02,  # 初始化范围默认为 0.02
        layer_norm_eps=1e-12,  # 层归一化的 epsilon 默认为 1e-12
        l_layers=9,  # L 层默认为 9
        x_layers=5,  # X 层默认为 5
        r_layers=5,  # R 层默认为 5
        visual_feat_dim=2048,  # 视觉特征维度默认为 2048
        visual_pos_dim=4,  # 视觉位置维度默认为 4
        visual_loss_normalizer=6.67,  # 视觉损失正则化默认为 6.67
        task_matched=True,  # 匹配任务默认启用
        task_mask_lm=True,  # Masked LM 任务默认启用
        task_obj_predict=True,  # 对象预测任务默认启用
        task_qa=True,  # QA 任务默认启用
        visual_obj_loss=True,  # 视觉对象损失默认启用
        visual_attr_loss=True,  # 视觉属性损失默认启用
        visual_feat_loss=True,  # 视觉特征损失默认启用
        **kwargs,
        ):
        # 初始化 BERT 模型的参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_attention_heads = num_attention_heads  # 注意力头的数量
        self.hidden_act = hidden_act  # 隐藏层激活函数
        self.intermediate_size = intermediate_size  # 中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob  # 隐藏层dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 注意力dropout概率
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.type_vocab_size = type_vocab_size  # 类型词汇表大小
        self.initializer_range = initializer_range  # 初始化范围
        self.layer_norm_eps = layer_norm_eps  # 层归一化的 epsilon
        self.num_qa_labels = num_qa_labels  # QA 标签数量
        self.num_object_labels = num_object_labels  # 对象标签数量
        self.num_attr_labels = num_attr_labels  # 属性标签数量
        self.l_layers = l_layers  # 语言层的数量
        self.x_layers = x_layers  # 交叉编码器层的数量
        self.r_layers = r_layers  # 视觉层的数量
        self.visual_feat_dim = visual_feat_dim  # 视觉特征维度
        self.visual_pos_dim = visual_pos_dim  # 视觉位置维度
        self.visual_loss_normalizer = visual_loss_normalizer  # 视觉损失归一化器
        self.task_matched = task_matched  # 匹配任务
        self.task_mask_lm = task_mask_lm  # Masked LM 任务
        self.task_obj_predict = task_obj_predict  # 对象预测任务
        self.task_qa = task_qa  # QA 任务
        self.visual_obj_loss = visual_obj_loss  # 视觉对象损失
        self.visual_attr_loss = visual_attr_loss  # 视觉属性损失
        self.visual_feat_loss = visual_feat_loss  # 视觉特征损失
        self.num_hidden_layers = {"vision": r_layers, "cross_encoder": x_layers, "language": l_layers}  # 隐藏层的数量字典
        super().__init__(**kwargs)  # 调用父类的初始化方法，并传入额外参数
```