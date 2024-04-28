# `.\models\deprecated\transfo_xl\modeling_transfo_xl_utilities.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，来源于 Google AI、Google Brain、Carnegie Mellon University 作者以及 HuggingFace Inc. 团队和 NVIDIA CORPORATION
# 根据 Apache 许可证版本 2.0 许可使用此文件
# 只有在符合许可证的情况下才能使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件根据“原样”基础分发
# 无论是明示的还是暗示的，都没有任何担保或条件
# 请参阅许可证获取特定语言的权限和限制
"""
用于 PyTorch Transformer XL 模型的实用工具。直接改编自 https://github.com/kimiyoung/transformer-xl。
"""

import torch
from torch import nn

# 定义一个类，用于实现投影自适应对数softmax操作
class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super().__init__()

        self.n_token = n_token  # 词汇表大小
        self.d_embed = d_embed  # 嵌入维度
        self.d_proj = d_proj  # 投影维度

        # 分段线性函数的断点列表，用于将词汇表划分为不同的子集
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val  # 用于确定不同子集的维度缩放比例的值

        self.shortlist_size = self.cutoffs[0]  # 第一个子集的大小
        self.n_clusters = len(self.cutoffs) - 1  # 子集的数量
        self.head_size = self.shortlist_size + self.n_clusters  # 投影后的向量大小

        # 如果有子集，则初始化子集的投影权重和偏置
        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        # 输出层和投影参数的初始化
        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        # 根据不同的维度缩放比例和子集断点，初始化输出层和投影参数
        if div_val == 1:
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:
                    self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                else:
                    self.out_projs.append(None)

            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)

                self.out_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))

                self.out_layers.append(nn.Linear(d_emb_i, r_idx - l_idx))

        self.keep_order = keep_order  # 是否保持原始顺序
    # 计算 logit 值，根据隐藏层、权重、偏置和投影矩阵
    def _compute_logit(self, hidden, weight, bias, proj):
        # 如果没有投影矩阵，直接使用线性函数计算logit值
        if proj is None:
            logit = nn.functional.linear(hidden, weight, bias=bias)
        else:
            # 如果有投影矩阵，则先将隐藏层与投影矩阵进行线性运算
            proj_hid = nn.functional.linear(hidden, proj.t().contiguous())
            # 再将得到的结果与权重进行线性运算得到最终的logit值
            logit = nn.functional.linear(proj_hid, weight, bias=bias)
            # 如果CUDA版本较低，使用另一种计算方式（注释部分代码不执行）
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit
        def log_prob(self, hidden):
            # 计算给定隐藏状态的所有类别的对数概率
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

            if self.n_clusters == 0:
                # 计算logit
                logit = self._compute_logit(hidden, self.out_layers[0].weight, self.out_layers[0].bias, self.out_projs[0])
                # 对logit进行log_softmax处理
                return nn.functional.log_softmax(logit, dim=-1)
            else:
                # 构建权重和偏置
                weights, biases = [], []
                for i in range(len(self.cutoffs)):
                    if self.div_val == 1:
                        l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                        weight_i = self.out_layers[0].weight[l_idx:r_idx]
                        bias_i = self.out_layers[0].bias[l_idx:r_idx]
                    else:
                        weight_i = self.out_layers[i].weight
                        bias_i = self.out_layers[i].bias

                    if i == 0:
                        weight_i = torch.cat([weight_i, self.cluster_weight], dim=0)
                        bias_i = torch.cat([bias_i, self.cluster_bias], dim=0)

                    weights.append(weight_i)
                    biases.append(bias_i)

                head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]
                head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)

                out = hidden.new_empty((head_logit.size(0), self.n_token))
                head_logprob = nn.functional.log_softmax(head_logit, dim=1)

                cutoff_values = [0] + self.cutoffs
                for i in range(len(cutoff_values) - 1):
                    start_idx, stop_idx = cutoff_values[i], cutoff_values[i + 1]

                    if i == 0:
                        out[:, : self.cutoffs[0]] = head_logprob[:, : self.cutoffs[0]]
                    else:
                        weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                        tail_logit_i = self._compute_logit(hidden, weight_i, bias_i, proj_i)
                        tail_logprob_i = nn.functional.log_softmax(tail_logit_i, dim=1)

                        logprob_i = head_logprob[:, -i] + tail_logprob_i
                        out[:, start_idx, stop_idx] = logprob_i

                return out
```