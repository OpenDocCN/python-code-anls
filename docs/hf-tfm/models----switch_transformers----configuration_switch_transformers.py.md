# `.\models\switch_transformers\configuration_switch_transformers.py`

```
# coding=utf-8
# Copyright 2022, Google and HuggingFace Inc.
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
""" Switch Transformers model configuration"""
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志模块


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/switch-base-8": "https://huggingface.co/google/switch-base-8/blob/main/config.json",
}

class SwitchTransformersConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`SwitchTransformersModel`]. It is used to
    instantiate a SwitchTransformers model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    SwitchTransformers [google/switch-base-8](https://huggingface.co/google/switch-base
        ):
        # 初始化 Transformer 参数
        self.vocab_size = vocab_size  # 设置词汇表大小
        self.d_model = d_model  # 设置 Transformer 模型的维度大小
        self.d_kv = d_kv  # 设置键和值的维度大小
        self.d_ff = d_ff  # 设置前馈网络的隐藏层大小

        self.num_sparse_encoder_layers = num_sparse_encoder_layers  # 编码器中稀疏层的数量
        self.num_layers = num_layers  # 总层数
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 解码器层数，默认与总层数相同
        self.num_sparse_decoder_layers = num_sparse_decoder_layers  # 解码器中稀疏层的数量

        # 每隔多少层设置一个稀疏层，用于编码器
        if self.num_sparse_encoder_layers > 0:
            self.encoder_sparse_step = self.num_layers // self.num_sparse_encoder_layers
        else:
            self.encoder_sparse_step = self.num_layers  # 如果没有稀疏层，则步长为总层数，这会创建0个稀疏层

        # 每隔多少层设置一个稀疏层，用于解码器
        if self.num_sparse_decoder_layers > 0:
            self.decoder_sparse_step = self.num_decoder_layers // self.num_sparse_decoder_layers
        else:
            self.decoder_sparse_step = self.num_decoder_layers  # 如果没有稀疏层，则步长为总层数，这会创建0个稀疏层

        self.num_heads = num_heads  # 设置注意力头的数量
        self.num_experts = num_experts  # 设置专家的数量
        self.expert_capacity = expert_capacity  # 设置每个专家的容量
        self.router_bias = router_bias  # 设置路由器偏置
        self.router_jitter_noise = router_jitter_noise  # 设置路由器抖动噪声
        if router_dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"`router_dtype` must be one of 'float32', 'float16' or 'bfloat16', got {router_dtype}")
        self.router_dtype = router_dtype  # 设置路由器数据类型

        self.router_ignore_padding_tokens = router_ignore_padding_tokens  # 是否忽略填充标记的路由
        self.relative_attention_num_buckets = relative_attention_num_buckets  # 相对注意力的桶数量
        self.relative_attention_max_distance = relative_attention_max_distance  # 相对注意力的最大距离

        self.dropout_rate = dropout_rate  # 设置丢弃率
        self.layer_norm_epsilon = layer_norm_epsilon  # 层归一化的 epsilon 值
        self.initializer_factor = initializer_factor  # 初始化因子
        self.use_cache = use_cache  # 是否使用缓存
        self.add_router_probs = add_router_probs  # 是否添加路由概率

        self.router_z_loss_coef = router_z_loss_coef  # 路由 Z 损失系数
        self.router_aux_loss_coef = router_aux_loss_coef  # 路由辅助损失系数
        self.dense_act_fn = dense_act_fn  # 密集层的激活函数

        super().__init__(
            pad_token_id=pad_token_id,  # 填充标记 ID
            eos_token_id=eos_token_id,  # 终止标记 ID
            is_encoder_decoder=is_encoder_decoder,  # 是否是编码解码器
            **kwargs,  # 其它关键字参数
        )
```