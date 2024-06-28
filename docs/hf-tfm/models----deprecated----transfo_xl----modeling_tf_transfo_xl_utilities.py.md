# `.\models\deprecated\transfo_xl\modeling_tf_transfo_xl_utilities.py`

```py
# coding=utf-8
# 文件编码声明，指定使用UTF-8编码
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# 版权声明，版权归属于Google AI、Google Brain、Carnegie Mellon University以及HuggingFace Inc.团队
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，版权归属于NVIDIA CORPORATION，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 授权许可声明，采用Apache License, Version 2.0，详细内容可访问指定URL获取
# you may not use this file except in compliance with the License.
# 除非符合许可证规定，否则不得使用此文件
# You may obtain a copy of the License at
# 可在指定URL获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 根据适用法律或书面同意，软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证分发在"原样"基础上
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何形式的担保或条件，无论是明示的还是隐含的
# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以了解特定语言的权限和限制
"""
 A TF 2.0 Adaptive Softmax for Transformer XL model.
"""
# 模块说明文档字符串，描述了此文件实现了基于Transformer XL的TF 2.0自适应Softmax模型


import tensorflow as tf

from ....modeling_tf_utils import keras
from ....tf_utils import shape_list


class TFAdaptiveSoftmaxMask(keras.layers.Layer):
    def __init__(self, vocab_size, d_embed, d_proj, cutoffs, div_val=1, keep_order=False, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = cutoffs + [vocab_size]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters
        self.keep_order = keep_order

        self.out_layers = []
        self.out_projs = []
    # 定义神经网络层的构建方法，根据输入形状 input_shape 动态构建网络层
    def build(self, input_shape):
        # 如果聚类数大于0，则添加聚类权重和偏置
        if self.n_clusters > 0:
            self.cluster_weight = self.add_weight(
                shape=(self.n_clusters, self.d_embed), initializer="zeros", trainable=True, name="cluster_weight"
            )
            self.cluster_bias = self.add_weight(
                shape=(self.n_clusters,), initializer="zeros", trainable=True, name="cluster_bias"
            )

        # 根据 div_val 的值分支处理
        if self.div_val == 1:
            # 遍历 self.cutoffs 列表的长度，依次处理每个 cutoff
            for i in range(len(self.cutoffs)):
                # 如果投影维度 d_proj 不等于嵌入维度 d_embed，则添加输出投影权重
                if self.d_proj != self.d_embed:
                    weight = self.add_weight(
                        shape=(self.d_embed, self.d_proj),
                        initializer="zeros",
                        trainable=True,
                        name=f"out_projs_._{i}",
                    )
                    self.out_projs.append(weight)
                else:
                    # 否则添加 None，表示无需额外投影
                    self.out_projs.append(None)
                # 添加输出层权重和偏置
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
            # 处理 div_val 不为1的情况
            for i in range(len(self.cutoffs)):
                # 获取当前 cutoff 的左右索引
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                # 计算当前层的嵌入维度 d_emb_i
                d_emb_i = self.d_embed // (self.div_val**i)

                # 添加输出投影权重
                weight = self.add_weight(
                    shape=(d_emb_i, self.d_proj), initializer="zeros", trainable=True, name=f"out_projs_._{i}"
                )
                self.out_projs.append(weight)
                # 添加输出层权重和偏置
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
        
        # 调用父类的 build 方法，完成神经网络层的构建
        super().build(input_shape)

    @staticmethod
    # 静态方法：计算对数概率的对数几率
    def _logit(x, W, b, proj=None):
        # 将输入 x 赋值给 y
        y = x
        # 如果提供了投影矩阵 proj，则进行投影操作
        if proj is not None:
            y = tf.einsum("ibd,ed->ibe", y, proj)
        # 计算最终的对数几率，使用 tf.einsum 实现张量乘法和加法
        return tf.einsum("ibd,nd->ibn", y, W) + b

    @staticmethod
    # 静态方法：根据目标索引从对数概率中收集对应的对数概率值
    def _gather_logprob(logprob, target):
        # 获取对数概率 logprob 的形状信息
        lp_size = shape_list(logprob)
        # 生成一个范围张量 r，其长度为 lp_size[0]，数据类型与目标张量一致
        r = tf.range(lp_size[0], dtype=target.dtype)
        # 构造索引张量 idx，形状为 [lp_size[0], 2]，每行包含一个范围值和对应的目标索引
        idx = tf.stack([r, target], 1)
        # 使用 tf.gather_nd 根据索引 idx 从 logprob 中收集对应的对数概率值
        return tf.gather_nd(logprob, idx)
```