# `.\pytorch\benchmarks\operator_benchmark\pt\embeddingbag_test.py`

```
# 导入NumPy库
import numpy
# 导入pt模块中的configs模块
from pt import configs

# 导入operator_benchmark模块，并将其重命名为op_bench
import operator_benchmark as op_bench
# 导入PyTorch库
import torch

"""Embedding and EmbeddingBag Operator Benchmark"""

# 定义EmbeddingBagBenchmark类，继承自op_bench.TorchBenchmarkBase类
class EmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置实例变量
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
        # 创建一个EmbeddingBag实例，并移动到指定的设备
        self.embedding = torch.nn.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
            sparse=sparse,
        ).to(device=device)
        # 设置随机种子为固定值
        numpy.random.seed((1 << 32) - 1)
        # 创建偏移量Tensor，移动到指定的设备
        offsets = torch.LongTensor([offset], device=device)
        # 创建输入Tensor，包含随机整数，移动到指定的设备
        input = torch.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        # 设置实例的inputs字典，包含输入和偏移量Tensor
        self.inputs = {
            "input": input,
            "offset": torch.cat(
                (offsets, torch.tensor([input.size(0)], dtype=torch.long)), 0
            ),
        }
        # 设置模块名称为"embeddingbag"
        self.set_module_name("embeddingbag")

    # 前向传播方法，调用EmbeddingBag实例的forward方法
    def forward(self, input, offset):
        return self.embedding(input, offset)


# 使用configs.embeddingbag_short_configs生成EmbeddingBagBenchmark的性能测试
op_bench.generate_pt_test(configs.embeddingbag_short_configs, EmbeddingBagBenchmark)
# 使用configs.embeddingbag_short_configs生成EmbeddingBagBenchmark的梯度测试
op_bench.generate_pt_gradient_test(
    configs.embeddingbag_short_configs, EmbeddingBagBenchmark
)


# 定义EmbeddingBenchmark类，继承自op_bench.TorchBenchmarkBase类
class EmbeddingBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置实例变量
    def init(self, num_embeddings, embedding_dim, input_size, device):
        # 创建一个Embedding实例，并移动到指定的设备
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        ).to(device=device)
        # 设置随机种子为固定值
        numpy.random.seed((1 << 32) - 1)
        # 创建输入Tensor，包含随机整数，移动到指定的设备
        input = torch.tensor(
            numpy.random.randint(0, num_embeddings, input_size), device=device
        ).long()
        # 设置实例的inputs字典，包含输入Tensor
        self.inputs = {"input": input}
        # 设置模块名称为"embedding"
        self.set_module_name("embedding")

    # 前向传播方法，调用Embedding实例的forward方法
    def forward(self, input):
        return self.embedding(input)


# 使用configs.embedding_short_configs生成EmbeddingBenchmark的性能测试
op_bench.generate_pt_test(configs.embedding_short_configs, EmbeddingBenchmark)
# 使用configs.embedding_short_configs生成EmbeddingBenchmark的梯度测试
op_bench.generate_pt_gradient_test(configs.embedding_short_configs, EmbeddingBenchmark)

# 如果当前文件被直接运行
if __name__ == "__main__":
    # 运行性能基准测试的主函数
    op_bench.benchmark_runner.main()
```