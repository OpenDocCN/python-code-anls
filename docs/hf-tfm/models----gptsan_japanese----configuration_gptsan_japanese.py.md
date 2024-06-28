# `.\models\gptsan_japanese\configuration_gptsan_japanese.py`

```
# coding=utf-8
# 指定代码文件的编码格式为UTF-8

# Copyright 2023, HuggingFace Inc.
# 版权声明，版权归HuggingFace Inc.所有，日期为2023年

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证授权使用本文件

# you may not use this file except in compliance with the License.
# 除非遵守许可证的规定，否则不得使用本文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则按"原样"分发软件，无论是明示还是隐含的，不提供任何担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证了解特定语言的授权内容及限制

"""  GPTSAN-japanese model configuration"""
# 模型配置的文档字符串说明，这是GPTSAN-japanese模型的配置

from ...configuration_utils import PretrainedConfig
# 导入PretrainedConfig类，用于存储预训练配置信息

from ...utils import logging
# 导入logging工具类，用于记录日志

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "tanreinama/GPTSAN-2.8B-spout_is_uniform": (
        "https://huggingface.co/tanreinama/GPTSAN-2.8B-spout_is_uniform/resolve/main/config.json"
    ),
}
# 预训练配置映射表，将模型名称映射到其配置文件的URL

class GPTSanJapaneseConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTSanJapaneseModel`]. It is used to instantiate
    a GPTSANJapanese model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPTSANJapanese
    [Tanrei/GPTSAN-japanese](https://huggingface.co/Tanrei/GPTSAN-japanese) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    """
    # GPTSanJapaneseConfig类的文档字符串，用于存储GPTSanJapaneseModel的配置信息

    model_type = "gptsan-japanese"
    # 模型类型定义为"gptsan-japanese"

    keys_to_ignore_at_inference = [
        "past_key_values",
    ]
    # 推理过程中忽略的键列表，在推理时不使用"past_key_values"

    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
    # 属性映射字典，将配置中的部分属性名映射为其他名称，如"hidden_size"映射为"d_model"

    def __init__(
        self,
        vocab_size=36000,
        max_position_embeddings=1280,
        d_model=1024,
        d_ff=8192,
        d_ext=4096,
        d_spout=128,
        num_switch_layers=10,
        num_ext_layers=0,
        num_heads=16,
        num_experts=16,
        expert_capacity=128,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-5,
        router_bias=False,
        router_jitter_noise=0.0,
        router_dtype="float32",
        router_ignore_padding_tokens=False,
        output_hidden_states=False,
        output_attentions=False,
        initializer_factor=0.002,
        output_router_logits=False,
        use_cache=True,
        separator_token_id=35998,
        pad_token_id=35995,
        eos_token_id=35999,
        **kwargs,
    ):
        # 初始化方法，用于创建一个新的GPTSanJapaneseConfig对象，设置模型的各种配置参数及其默认值
        ):
        # 初始化 TransformerXLConfig 类的实例，设定模型的各种超参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置嵌入数
        self.d_model = d_model  # 模型的隐藏层大小
        self.d_ff = d_ff  # 前向传播神经网络中间层的大小
        self.d_ext = d_ext  # 扩展层的大小
        self.d_spout = d_spout  # 接口层的大小
        self.num_switch_layers = num_switch_layers  # 切换层的数量
        self.num_ext_layers = num_ext_layers  # 扩展层的数量
        self.num_layers = num_switch_layers + num_ext_layers  # 总层数
        self.num_heads = num_heads  # 注意力头的数量
        self.num_experts = num_experts  # 专家的数量
        self.expert_capacity = expert_capacity  # 专家的容量
        self.dropout_rate = dropout_rate  # 丢弃率
        self.layer_norm_epsilon = layer_norm_epsilon  # 层归一化的 epsilon 参数
        self.router_bias = router_bias  # 路由器的偏置
        self.router_jitter_noise = router_jitter_noise  # 路由器的抖动噪声
        self.router_dtype = router_dtype  # 路由器的数据类型
        self.router_ignore_padding_tokens = router_ignore_padding_tokens  # 是否忽略填充标记的路由
        self.output_hidden_states = output_hidden_states  # 是否输出隐藏状态
        self.output_attentions = output_attentions  # 是否输出注意力权重
        self.initializer_factor = initializer_factor  # 初始化因子
        self.output_router_logits = output_router_logits  # 是否输出路由器的对数
        self.use_cache = use_cache  # 是否使用缓存

        # 调用父类 TransformerXLConfig 的初始化方法，设置分隔符、填充符、终止符等参数
        super().__init__(
            separator_token_id=separator_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
```