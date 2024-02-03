# `jieba\jieba\lac_small\nets.py`

```
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
The function lex_net(args) define the lexical analysis network structure
"""
import sys
import os
import math

import paddle.fluid as fluid
from paddle.fluid.initializer import NormalInitializer

# 定义词法分析网络结构的函数
def lex_net(word, vocab_size, num_labels, for_infer=True, target=None):
    """
    define the lexical analysis network structure
    word: stores the input of the model
    for_infer: a boolean value, indicating if the model to be created is for training or predicting.

    return:
        for infer: return the prediction
        otherwise: return the prediction
    """
   
    word_emb_dim=128
    grnn_hidden_dim=128
    bigru_num=2
    emb_lr = 1.0
    crf_lr = 1.0
    init_bound = 0.1
    IS_SPARSE = True
    # 定义双向 GRU 层
    def _bigru_layer(input_feature):
        """
        define the bidirectional gru layer
        """
        # 前向 GRU，全连接层
        pre_gru = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        # 前向 GRU，动态展开
        gru = fluid.layers.dynamic_gru(
            input=pre_gru,
            size=grnn_hidden_dim,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        # 反向 GRU，全连接层
        pre_gru_r = fluid.layers.fc(
            input=input_feature,
            size=grnn_hidden_dim * 3,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))
        # 反向 GRU，动态展开
        gru_r = fluid.layers.dynamic_gru(
            input=pre_gru_r,
            size=grnn_hidden_dim,
            is_reverse=True,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        # 合并前向和反向 GRU 的输出
        bi_merge = fluid.layers.concat(input=[gru, gru_r], axis=1)
        return bi_merge
    # 定义一个函数用于配置网络
    def _net_conf(word, target=None):
        """
        Configure the network
        """
        # 创建词嵌入层，用于将输入的词转换为词向量
        word_embedding = fluid.embedding(
            input=word,
            size=[vocab_size, word_emb_dim],
            dtype='float32',
            is_sparse=IS_SPARSE,
            param_attr=fluid.ParamAttr(
                learning_rate=emb_lr,
                name="word_emb",
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound)))

        # 将词嵌入层的输出作为输入特征
        input_feature = word_embedding
        # 使用循环构建多层双向 GRU 网络
        for i in range(bigru_num):
            bigru_output = _bigru_layer(input_feature)
            input_feature = bigru_output

        # 创建一个全连接层，用于输出预测结果
        emission = fluid.layers.fc(
            size=num_labels,
            input=bigru_output,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=-init_bound, high=init_bound),
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=1e-4)))

        # 获取输出的维度大小
        size = emission.shape[1]
        # 创建一个参数用于 CRF 解码
        fluid.layers.create_parameter(
            shape=[size + 2, size], dtype=emission.dtype, name='crfw')
        # 使用 CRF 解码层进行解码
        crf_decode = fluid.layers.crf_decoding(
            input=emission, param_attr=fluid.ParamAttr(name='crfw'))

        # 返回 CRF 解码结果
        return crf_decode
    # 调用网络配置函数并返回结果
    return _net_conf(word)
```