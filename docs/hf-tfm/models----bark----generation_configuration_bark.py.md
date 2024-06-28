# `.\models\bark\generation_configuration_bark.py`

```
# coding=utf-8
# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" BARK model generation configuration"""

import copy
from typing import Dict

from ...generation.configuration_utils import GenerationConfig
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 BarkSemanticGenerationConfig 类，继承自 GenerationConfig
class BarkSemanticGenerationConfig(GenerationConfig):
    # 模型类型为语义生成
    model_type = "semantic"

    def __init__(
        self,
        eos_token_id=10_000,
        renormalize_logits=True,
        max_new_tokens=768,
        output_scores=False,
        return_dict_in_generate=False,
        output_hidden_states=False,
        output_attentions=False,
        temperature=1.0,
        do_sample=False,
        text_encoding_offset=10_048,
        text_pad_token=129_595,
        semantic_infer_token=129_599,
        semantic_vocab_size=10_000,
        max_input_semantic_length=256,
        semantic_rate_hz=49.9,
        min_eos_p=None,
        **kwargs,
    ):
        # 调用父类 GenerationConfig 的构造函数，初始化配置参数
        super().__init__(
            eos_token_id=eos_token_id,
            renormalize_logits=renormalize_logits,
            max_new_tokens=max_new_tokens,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs,
        )
        # 设置额外的语义生成配置参数
        self.text_encoding_offset = text_encoding_offset
        self.text_pad_token = text_pad_token
        self.semantic_infer_token = semantic_infer_token
        self.semantic_vocab_size = semantic_vocab_size
        self.max_input_semantic_length = max_input_semantic_length
        self.semantic_rate_hz = semantic_rate_hz
        self.min_eos_p = min_eos_p

# 定义 BarkCoarseGenerationConfig 类，继承自 GenerationConfig
class BarkCoarseGenerationConfig(GenerationConfig):
    # 模型类型为粗粒度声学生成
    model_type = "coarse_acoustics"

    def __init__(
        self,
        renormalize_logits=True,
        output_scores=False,
        return_dict_in_generate=False,
        output_hidden_states=False,
        output_attentions=False,
        temperature=1.0,
        do_sample=False,
        coarse_semantic_pad_token=12_048,
        coarse_rate_hz=75,
        n_coarse_codebooks=2,
        coarse_infer_token=12_050,
        max_coarse_input_length=256,
        max_coarse_history: int = 630,
        sliding_window_len: int = 60,
        **kwargs,
    ):
        # 调用父类 GenerationConfig 的构造函数，初始化配置参数
        super().__init__(
            renormalize_logits=renormalize_logits,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            temperature=temperature,
            do_sample=do_sample,
            **kwargs,
        )
        # 设置额外的粗粒度声学生成配置参数
        self.coarse_semantic_pad_token = coarse_semantic_pad_token
        self.coarse_rate_hz = coarse_rate_hz
        self.n_coarse_codebooks = n_coarse_codebooks
        self.coarse_infer_token = coarse_infer_token
        self.max_coarse_input_length = max_coarse_input_length
        self.max_coarse_history = max_coarse_history
        self.sliding_window_len = sliding_window_len

# 定义 BarkFineGenerationConfig 类，继承自 GenerationConfig
class BarkFineGenerationConfig(GenerationConfig):
    # 模型类型为细粒度声学生成
    model_type = "fine_acoustics"

    def __init__(
        self,
        temperature=1.0,
        max_fine_history_length=512,
        max_fine_input_length=1024,
        n_fine_codebooks=8,
        **kwargs,
    ):
        # 调用父类 GenerationConfig 的构造函数，初始化配置参数
        super().__init__(
            temperature=temperature,
            **kwargs,
        )
        # 设置额外的细粒度声学生成配置参数
        self.max_fine_history_length = max_fine_history_length
        self.max_fine_input_length = max_fine_input_length
        self.n_fine_codebooks = n_fine_codebooks
    ):
        """
        Class that holds a generation configuration for `BarkFineModel`.

        `BarkFineModel` is an autoencoder model, so should not usually be used for generation. However, under the
        hood, it uses `temperature` when used by `BarkModel`.

        This configuration inherits from `GenerationConfig` and can be used to control the model generation. Read the
        documentation from `GenerationConfig` for more information.

        Args:
            temperature (`float`, *optional*):
                The value used to modulate the next token probabilities.
            max_fine_history_length (`int`, *optional*, defaults to 512):
                Max length of the fine history vector.
            max_fine_input_length (`int`, *optional*, defaults to 1024):
                Max length of fine input vector.
            n_fine_codebooks (`int`, *optional*, defaults to 8):
                Number of codebooks used.
        """
        # 调用父类构造函数，初始化基类的 temperature 参数
        super().__init__(temperature=temperature)

        # 设置当前类的属性值，用于控制生成配置
        self.max_fine_history_length = max_fine_history_length
        self.max_fine_input_length = max_fine_input_length
        self.n_fine_codebooks = n_fine_codebooks

    def validate(self, **kwargs):
        """
        Overrides GenerationConfig.validate because BarkFineGenerationConfig don't use any parameters outside
        temperature.
        """
        # 由于 BarkFineGenerationConfig 不使用除 temperature 外的任何参数，因此重写了 GenerationConfig.validate 方法。
        pass
class BarkGenerationConfig(GenerationConfig):
    model_type = "bark"
    is_composition = True

    # TODO (joao): nested from_dict
    # 定义一个待办事项，表示需要从字典中嵌套生成配置对象

    def __init__(
        self,
        semantic_config: Dict = None,
        coarse_acoustics_config: Dict = None,
        fine_acoustics_config: Dict = None,
    ):
        r"""
        Instantiate a [`BarkGenerationConfig`] (or a derived class) from bark sub-models generation configuration.

        Returns:
            [`BarkGenerationConfig`]: An instance of a configuration object
        """
        return cls(
            semantic_config=semantic_config.to_dict(),
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),
            fine_acoustics_config=fine_acoustics_config.to_dict(),
            **kwargs,
        )


    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # Deep copy the internal dictionary representation of the object
        output = copy.deepcopy(self.__dict__)

        # Convert nested `semantic_config`, `coarse_acoustics_config`, and `fine_acoustics_config` objects to dictionaries
        output["semantic_config"] = self.semantic_config.to_dict()
        output["coarse_acoustics_config"] = self.coarse_acoustics_config.to_dict()
        output["fine_acoustics_config"] = self.fine_acoustics_config.to_dict()

        # Add the class-level attribute `model_type` to the output dictionary
        output["model_type"] = self.__class__.model_type
        return output
```