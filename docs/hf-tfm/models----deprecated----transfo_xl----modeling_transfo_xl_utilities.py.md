# `.\models\deprecated\transfo_xl\modeling_transfo_xl_utilities.py`

```
# coding=utf-8
# 定义编码方式为 UTF-8

# Utilities for PyTorch Transformer XL model. Directly adapted from https://github.com/kimiyoung/transformer-xl.
# PyTorch Transformer XL 模型的实用工具，直接从 https://github.com/kimiyoung/transformer-xl 进行了适应。

import torch
# 导入 PyTorch 库
from torch import nn
# 从 PyTorch 库中导入 nn 模块

# CUDA_MAJOR = int(torch.version.cuda.split('.')[0])
# CUDA_MINOR = int(torch.version.cuda.split('.')[1])

class ProjectedAdaptiveLogSoftmax(nn.Module):
    # 定义 ProjectedAdaptiveLogSoftmax 类，继承自 nn.Module
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super().__init__()
        # 调用父类的初始化方法

        self.n_token = n_token
        # 设置词汇表大小
        self.d_embed = d_embed
        # 设置嵌入维度大小
        self.d_proj = d_proj
        # 设置投影维度大小

        self.cutoffs = cutoffs + [n_token]
        # 设置分隔阈值列表，并加入词汇表大小作为最后一个阈值
        self.cutoff_ends = [0] + self.cutoffs
        # 设置分隔阈值结束位置列表，包含从0开始到每个阈值结束的位置
        self.div_val = div_val
        # 设置划分值

        self.shortlist_size = self.cutoffs[0]
        # 设置短列表大小为第一个阈值
        self.n_clusters = len(self.cutoffs) - 1
        # 设置集群数量为阈值数量减一
        self.head_size = self.shortlist_size + self.n_clusters
        # 设置头部大小为短列表大小加上集群数量

        if self.n_clusters > 0:
            # 如果集群数量大于零
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            # 设置集群权重为 nn.Parameter，大小为集群数量乘以嵌入维度
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))
            # 设置集群偏置为 nn.Parameter，大小为集群数量

        self.out_layers = nn.ModuleList()
        # 设置输出层为 nn.ModuleList
        self.out_projs = nn.ParameterList()
        # 设置输出投影为 nn.ParameterList

        if div_val == 1:
            # 如果划分值为1
            for i in range(len(self.cutoffs)):
                # 对于阈值列表的每个元素
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                    # 如果投影维度不等于嵌入维度，添加投影参数
                else:
                    self.out_projs.append(None)
                    # 否则添加空值

            self.out_layers.append(nn.Linear(d_embed, n_token))
            # 添加线性层，输入维度为嵌入维度，输出维度为词汇表大小
        else:
            for i in range(len(self.cutoffs)):
                # 对于阈值列表的每个元素
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                # 获取左右索引值

                d_emb_i = d_embed // (div_val**i)
                # 计算当前嵌入维度

                self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))
                # 添加投影参数

                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))
                # 添加线性层，输入维度为当前嵌入维度，输出维度为右索引减去左索引

        self.keep_order = keep_order
        # 设置保持顺序标志位
    # 定义一个方法，用于计算逻辑输出
    def _compute_logit(self, hidden, weight, bias, proj):
        # 如果没有投影矩阵，则直接使用线性变换计算逻辑输出
        if proj is None:
            logit = nn.functional.linear(hidden, weight, bias=bias)
        else:
            # 如果有投影矩阵，则先对隐藏层进行投影操作
            proj_hid = nn.functional.linear(hidden, proj.t().contiguous())
            # 然后使用投影后的结果与权重进行线性变换，计算逻辑输出
            logit = nn.functional.linear(proj_hid, weight, bias=bias)
            # CUDA_MAJOR 和 CUDA_MINOR 小于等于 9.1 时使用下面的方法计算逻辑输出
            # logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            # 如果有偏置，则加上偏置值
            # if bias is not None:
            #     logit = logit + bias

        # 返回计算得到的逻辑输出
        return logit
    def log_prob(self, hidden):
        r"""
        Computes log probabilities for all \\(n\_classes\\) From:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.p

        Args:
            hidden (Tensor): a minibatch of example

        Returns:
            log-probabilities of for each class \\(c\\) in range \\(0 <= c <= n\_classes\\), where \\(n\_classes\\) is
            a parameter passed to `AdaptiveLogSoftmaxWithLoss` constructor. Shape:

            - Input: \\((N, in\_features)\\)
            - Output: \\((N, n\_classes)\\)
        """
        # Check if the number of clusters is zero
        if self.n_clusters == 0:
            # Compute logit using the first output layer's weights, biases, and projection
            logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
            # Apply log_softmax to the computed logit along the last dimension
            return nn.functional.log_softmax(logit, dim=-1)
        else:
            # Initialize empty lists for weights and biases
            weights, biases = [], []
            # Iterate over cutoffs to construct weights and biases
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    # Calculate left and right indices based on cutoff ends
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    # Extract weight and bias from the first output layer within the specified range
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:
                    # Use weights and biases directly from respective output layers
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                # Concatenate cluster weights and biases if it's the first iteration
                if i == 0:
                    weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                # Append constructed weight and bias to lists
                weights.append(weight_i)
                biases.append(bias_i)

            # Extract head weight, bias, and projection
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
            # Compute logit using the extracted parameters
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)

            # Initialize an empty tensor 'out' with the same device as 'hidden'
            out = hidden.new_empty((head_logit.size(0), self.n_token))
            # Compute log softmax along the second dimension (classes) of head logit
            head_logprob = nn.functional.log_softmax(head_logit, dim=1)

            # Define cutoff values with 0 as the initial value
            cutoff_values = [0] + self.cutoffs
            # Iterate over cutoff values to compute log probabilities
            for i in range(len(cutoff_values) - 1):
                start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]

                if i == 0:
                    # Assign head log probabilities to the initial segment of 'out'
                    out[:, : self.cutoffs[0]] = head_logprob[:, : self.cutoffs[0]]
                else:
                    # Extract weight, bias, and projection for the current segment
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]
                    # Compute logit for the current segment
                    tail_logit_i = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                    # Compute log softmax for the computed logit
                    tail_logprob_i = nn.functional.log_softmax(tail_logit_i, dim=1)

                    # Combine head log probabilities and tail log probabilities
                    logprob_i = head_logprob[:, -i] + tail_logprob_i
                    # Assign computed log probabilities to the respective segment of 'out'
                    out[:, start_idx: stop_idx] = logprob_i

            # Return the computed log probabilities 'out'
            return out
```