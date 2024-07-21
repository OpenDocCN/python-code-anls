# `.\pytorch\benchmarks\tensorexpr\attention.py`

```
# 这是从MLPerf复制的rnn_attention代码，一些常见大小被硬编码用于基准测试，并且去除了一些控制流。
# https://github.com/mlperf/training/blob/master/rnn_translator/pytorch/seq2seq/models/attention.py

import torch

from . import benchmark  # 导入benchmark模块


class BahdanauAttention(benchmark.Benchmark):
    def __init__(self, mode, device, dtype, b, t_q, t_k, n):
        super().__init__(mode, device, dtype)
        self.b = b  # 批量大小
        self.t_q = t_q  # 查询序列长度
        self.t_k = t_k  # 键序列长度
        self.n = n  # 特征维度大小
        # 初始化注意力查询向量，形状为 [b, t_q, n]
        self.att_query = self.rand(
            [b, t_q, n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 初始化注意力键向量，形状为 [b, t_k, n]
        self.att_keys = self.rand(
            [b, t_k, n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 初始化用于归一化的偏置向量，形状为 [n]
        self.normalize_bias = self.rand(
            [n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 初始化用于线性变换的权重向量，形状为 [n]
        self.linear_att = self.rand(
            [n], device=device, dtype=dtype, requires_grad=self.requires_grad
        )
        # 输入列表，包含初始化的各个张量
        self.inputs = [
            self.att_query,
            self.att_keys,
            self.normalize_bias,
            self.linear_att,
        ]

    def forward(self, att_query, att_keys, normalize_bias, linear_att):
        """
        计算Bahdanau注意力分数

        :param att_query: b x t_q x n，注意力查询向量
        :param att_keys: b x t_k x n，注意力键向量

        return b x t_q x t_k 的注意力分数
        """

        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)

        # 扩展张量以便进行注意力计算
        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys + normalize_bias  # 计算注意力分数的总和
        out = torch.tanh(sum_qk).matmul(linear_att)  # 应用tanh并进行线性变换
        return out

    def reference(self):
        return self.numpy(self.forward(*self.inputs))  # 返回前向传播的numpy数组结果

    def config(self):
        return [self.b, self.t_q, self.t_k, self.n]  # 返回模型配置参数列表

    @staticmethod
    def module():
        return "attention"  # 返回模块类型为"attention"

    def memory_workload(self):
        def memsize(t):
            return t.numel() * t.element_size()

        # 计算输入和输出的内存工作量
        input_size = (
            memsize(self.att_query)
            + memsize(self.att_keys)
            + memsize(self.normalize_bias)
            + memsize(self.linear_att)
        )
        output_size = 4 * torch.Size([self.b, self.t_q, self.t_k]).numel()
        io_size = input_size + output_size

        # 如果矩阵乘法未融合，必须先写入然后读取'sum_qk'。
        intermediate_size = (
            2 * 4 * torch.Size([self.b, self.t_q, self.t_k, self.n]).numel()
        )
        return {"sol": io_size, "algorithmic": io_size + intermediate_size}

    @staticmethod
    def default_configs():
        mlperf_inference = [1280, 1, 66, 1024]  # MLPerf推理默认配置
        nvidia = [128, 10, 128, 1024]  # Nvidia默认配置
        return [mlperf_inference, nvidia]  # 返回默认配置列表


# 注册BahdanauAttention类到benchmark模块
benchmark.register_benchmark_class(BahdanauAttention)
```