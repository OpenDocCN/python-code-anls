# `.\pytorch\benchmarks\operator_benchmark\pt\qatembedding_ops_test.py`

```py
# 导入所需的模块numpy和configs
import numpy
from pt import configs

# 导入操作基准库op_bench，以及torch和torch.quantization模块中的特定内容
import operator_benchmark as op_bench
import torch
import torch.ao.nn.qat as nnqat
from torch.ao.quantization import default_embedding_qat_qconfig

"""
QAT Embedding + EmbeddingBag 运算的微基准。
"""

# 定义基准类QATEmbeddingBagBenchmark，继承自op_bench.TorchBenchmarkBase
class QATEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化函数，设置各类参数
    def init(
        self,
        embeddingbags,
        dim,
        mode,
        input_size,
        offset,
        sparse,
        include_last_offset,
        device,
    ):
        # 设置量化配置qconfig为默认的嵌入QAT量化配置
        qconfig = default_embedding_qat_qconfig
        # 创建nnqat.EmbeddingBag实例，用于量化感知训练
        self.embedding = nnqat.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
            device=device,
            qconfig=qconfig,
        )
        # 设置随机数种子为(1 << 32) - 1，确保随机性
        numpy.random.seed((1 << 32) - 1)
        # 创建长整型Tensor offsets，包含偏移量offset
        offsets = torch.LongTensor([offset], device=device)
        # 创建输入Tensor input，包含随机整数，用于嵌入索引
        input = torch.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        # 设置inputs字典，包含input和offset
        self.inputs = {
            "input": input,
            "offset": torch.cat(
                (offsets, torch.tensor([input.size(0)], dtype=torch.long)), 0
            ),
        }
        # 设置模块名为"qatEmbeddingBag"
        self.set_module_name("qatEmbeddingBag")

    # 前向传播函数，执行嵌入操作
    def forward(self, input, offset):
        return self.embedding(input, offset)


# 当前版本的EmbeddingBag QAT不支持稀疏嵌入。
# 过滤出embeddingbag_short_configs中不包含稀疏配置的config
embeddingbag_short_dense_configs = [
    config
    for config in configs.embeddingbag_short_configs
    if {"sparse": True} not in config
]

# 生成QATEmbeddingBagBenchmark的基准测试
op_bench.generate_pt_test(embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark)
# 生成QATEmbeddingBagBenchmark的梯度测试
op_bench.generate_pt_gradient_test(
    embeddingbag_short_dense_configs, QATEmbeddingBagBenchmark
)

# 定义基准类QATEmbeddingBenchmark，继承自op_bench.TorchBenchmarkBase
class QATEmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化函数，设置嵌入数量、嵌入维度、输入大小和设备
    def init(self, num_embeddings, embedding_dim, input_size, device):
        # 设置量化配置qconfig为默认的嵌入QAT量化配置
        qconfig = default_embedding_qat_qconfig
        # 创建nnqat.Embedding实例，用于量化感知训练
        self.embedding = nnqat.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            qconfig=qconfig,
            device=device,
        )
        # 设置嵌入的量化配置为默认的嵌入QAT量化配置
        self.embedding.qconfig = default_embedding_qat_qconfig
        # 设置随机数种子为(1 << 32) - 1，确保随机性
        numpy.random.seed((1 << 32) - 1)
        # 创建输入Tensor input，包含随机整数，用于嵌入索引
        self.input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        # 设置inputs字典，包含input
        self.inputs = {"input": self.input}
        # 设置模块名为"qatEmbedding"
        self.set_module_name("qatEmbedding")

    # 前向传播函数，执行嵌入操作
    def forward(self, input):
        return self.embedding(input)


# 生成QATEmbeddingBenchmark的基准测试
op_bench.generate_pt_test(configs.embedding_short_configs, QATEmbeddingBenchmark)
# 生成QATEmbeddingBenchmark的梯度测试
op_bench.generate_pt_gradient_test(
    configs.embedding_short_configs, QATEmbeddingBenchmark
)

# 如果当前脚本作为主程序运行，则执行基准运行器的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```