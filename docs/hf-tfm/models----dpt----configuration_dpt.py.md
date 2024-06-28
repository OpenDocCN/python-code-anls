# `.\models\dpt\configuration_dpt.py`

```
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" DPT model configuration"""

# 引入必要的模块和类
import copy

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志模块
from ...utils import logging
# 从自动配置中导入配置映射
from ..auto.configuration_auto import CONFIG_MAPPING
# 从bit模块导入BitConfig类
from ..bit import BitConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型与配置文件的映射关系
DPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Intel/dpt-large": "https://huggingface.co/Intel/dpt-large/resolve/main/config.json",
    # See all DPT models at https://huggingface.co/models?filter=dpt
}

# 定义DPTConfig类，继承自PretrainedConfig类
class DPTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DPTModel`]. It is used to instantiate an DPT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DPT
    [Intel/dpt-large](https://huggingface.co/Intel/dpt-large) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import DPTModel, DPTConfig

    >>> # Initializing a DPT dpt-large style configuration
    >>> configuration = DPTConfig()

    >>> # Initializing a model from the dpt-large style configuration
    >>> model = DPTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 指定模型类型为"dpt"
    model_type = "dpt"
    def __init__(
        self,
        hidden`
    def __init__(
        self,
        hidden_size=768,  # 设置隐藏层大小，默认为768
        num_hidden_layers=12,  # 设置隐藏层数量，默认为12
        num_attention_heads=12,  # 设置注意力头的数量，默认为12
        intermediate_size=3072,  # 设置中间层的大小，默认为3072
        hidden_act="gelu",  # 设置隐藏层激活函数，默认为'gelu'
        hidden_dropout_prob=0.0,  # 设置隐藏层的丢弃概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 设置注意力概率的丢弃概率，默认为0.0
        initializer_range=0.02,  # 设置初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 设置层标准化的 epsilon 值，默认为1e-12
        image_size=384,  # 设置输入图像的尺寸，默认为384
        patch_size=16,  # 设置图像切片的大小，默认为16
        num_channels=3,  # 设置输入图像的通道数，默认为3
        is_hybrid=False,  # 设置是否为混合模型，默认为False
        qkv_bias=True,  # 设置 QKV 是否使用偏置，默认为True
        backbone_out_indices=[2, 5, 8, 11],  # 设置骨干网络输出的层索引，默认为[2, 5, 8, 11]
        readout_type="project",  # 设置读取类型，默认为'project'
        reassemble_factors=[4, 2, 1, 0.5],  # 设置重组因子，默认为[4, 2, 1, 0.5]
        neck_hidden_sizes=[96, 192, 384, 768],  # 设置脖子层隐藏层的大小，默认为[96, 192, 384, 768]
        fusion_hidden_size=256,  # 设置融合层隐藏层的大小，默认为256
        head_in_index=-1,  # 设置头部输入的索引，默认为-1
        use_batch_norm_in_fusion_residual=False,  # 设置在融合残差中是否使用批量归一化，默认为False
        use_bias_in_fusion_residual=None,  # 设置融合残差中是否使用偏置，默认为None
        add_projection=False,  # 设置是否添加投影层，默认为False
        use_auxiliary_head=True,  # 设置是否使用辅助头，默认为True
        auxiliary_loss_weight=0.4,  # 设置辅助损失权重，默认为0.4
        semantic_loss_ignore_index=255,  # 设置语义损失忽略索引，默认为255
        semantic_classifier_dropout=0.1,  # 设置语义分类器的丢弃概率，默认为0.1
        backbone_featmap_shape=[1, 1024, 24, 24],  # 设置骨干特征图的形状，默认为[1, 1024, 24, 24]
        neck_ignore_stages=[0, 1],  # 设置忽略的颈部阶段，默认为[0, 1]
        backbone_config=None,  # 设置骨干配置，默认为None
        backbone=None,  # 设置骨干网络，默认为None
        use_pretrained_backbone=False,  # 设置是否使用预训练骨干，默认为False
        use_timm_backbone=False,  # 设置是否使用 timm 骨干，默认为False
        backbone_kwargs=None,  # 设置骨干网络的关键字参数，默认为None
        **kwargs,  # 允许额外的关键字参数
    ):
        """
        初始化方法，设置模型的各种参数。
        """
        super().__init__(**kwargs)  # 调用父类的初始化方法

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)  # 深拷贝当前实例的字典属性

        if output["backbone_config"] is not None:  # 如果骨干配置不为空
            output["backbone_config"] = self.backbone_config.to_dict()  # 将骨干配置转换为字典

        output["model_type"] = self.__class__.model_type  # 设置模型类型为当前类的模型类型
        return output  # 返回字典表示
```