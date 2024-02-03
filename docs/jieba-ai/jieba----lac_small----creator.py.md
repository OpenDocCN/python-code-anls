# `jieba\jieba\lac_small\creator.py`

```py
# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""
Define the function to create lexical analysis model and model's data reader
"""
import sys
import os
import math

import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer
import jieba.lac_small.nets as nets

# 定义创建词法分析模型和数据读取器的函数

def create_model(vocab_size, num_labels, mode='train'):
    """create lac model"""

    # model's input data
    # 定义模型的输入数据
    words = fluid.data(name='words', shape=[-1, 1], dtype='int64', lod_level=1)
    targets = fluid.data(
        name='targets', shape=[-1, 1], dtype='int64', lod_level=1)

    # for inference process
    # 用于推理过程
    if mode == 'infer':
        # 调用词法分析网络，设置为推理模式
        crf_decode = nets.lex_net(
            words, vocab_size, num_labels, for_infer=True, target=None)
        return {
            "feed_list": [words],
            "words": words,
            "crf_decode": crf_decode,
        }
    return ret
```