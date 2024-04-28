# `.\models\deprecated\transfo_xl\modeling_tf_transfo_xl_utilities.py`

```
# 导入必要的 Python 模块
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# 导入 Apache License, Version 2.0
# 用于许可该软件的使用
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
 A TF 2.0 Adaptive Softmax for Transformer XL model.
"""


import tensorflow as tf

from ....tf_utils import shape_list


class TFAdaptiveSoftmaxMask(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_embed, d_proj, cutoffs, div_val=1, keep_order=False, **kwargs):
        # 初始化父类 tf.keras.layers.Layer
        super().__init__(**kwargs)

        # 存储模型参数
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_proj = d_proj

        # 计算分类阈值
        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        # 计算模型结构相关参数
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        self.keep_order = keep_order

        # 初始化输出层和投影层
        self.out_layers = []
        self.out_projs = []
    # 在模型构建时根据输入形状进行相关初始化操作
    def build(self, input_shape):
        # 如果聚类数大于0，添加聚类权重和偏置
        if self.n_clusters > 0:
            self.cluster_weight = self.add_weight(
                shape=(self.n_clusters, self.d_embed), initializer="zeros", trainable=True, name="cluster_weight"
            )
            self.cluster_bias = self.add_weight(
                shape=(self.n_clusters,), initializer="zeros", trainable=True, name="cluster_bias"
            )

        # 如果div_val等于1，在循环截断列表长度范围内进行相关初始化操作
        if self.div_val == 1:
            for i in range(len(self.cutoffs)):
                # 如果投影维度不等于嵌入维度，添加权重
                if self.d_proj != self.d_embed:
                    weight = self.add_weight(
                        shape=(self.d_embed, self.d_proj),
                        initializer="zeros",
                        trainable=True,
                        name=f"out_projs_._{i}",
                    )
                    self.out_projs.append(weight)
                else:
                    self.out_projs.append(None)
                # 添加权重和偏置
                weight = self.add_weight(
                    shape=(self.vocab_size, self.d_embed),
                    initializer="zeros",
                    trainable=True,
                    name=f"out_layers_._{i}_._weight",
                )
                bias = self.add_weight(
                    shape=(self.vocab_size,),
                    initializer="zeros",
                    trainable=True,
                    name=f"out_layers_._{i}_._bias",
                )
                self.out_layers.append((weight, bias))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = self.d_embed // (self.div_val**i)
                # 添加权重
                weight = self.add_weight(
                    shape=(d_emb_i, self.d_proj), initializer="zeros", trainable=True, name=f"out_projs_._{i}"
                )
                self.out_projs.append(weight)
                # 添加权重和偏置
                weight = self.add_weight(
                    shape=(r_idx - l_idx, d_emb_i),
                    initializer="zeros",
                    trainable=True,
                    name=f"out_layers_._{i}_._weight",
                )
                bias = self.add_weight(
                    shape=(r_idx - l_idx,),
                    initializer="zeros",
                    trainable=True,
                    name=f"out_layers_._{i}_._bias",
                )
                self.out_layers.append((weight, bias))
        # 调用父类的构建方法
        super().build(input_shape)

    # 线性操作
    @staticmethod
    def _logit(x, W, b, proj=None):
        y = x
        # 如果存在投影，进行投影操作
        if proj is not None:
            y = tf.einsum("ibd,ed->ibe", y, proj)
        return tf.einsum("ibd,nd->ibn", y, W) + b

    # 获取对数概率
    @staticmethod
    def _gather_logprob(logprob, target):
        lp_size = shape_list(logprob)
        r = tf.range(lp_size[0], dtype=target.dtype)
        idx = tf.stack([r, target], 1)
        return tf.gather_nd(logprob, idx)
```