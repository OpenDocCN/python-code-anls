# `.\pytorch\benchmarks\operator_benchmark\pt\qembeddingbag_test.py`

```
# 导入必要的库 numpy
import numpy
# 从 pt 模块中导入 configs
from pt import configs

# 导入 operator_benchmark 库并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 torch 库
import torch
# 导入 torch 中的量化神经网络模块 nnq
import torch.ao.nn.quantized as nnq

"""
qEmbeddingBag 操作的微基准测试。
"""

# 定义 QEmbeddingBagBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class QEmbeddingBagBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，设置基准测试所需的参数和状态
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
        # 创建 nnq.EmbeddingBag 对象并移动到指定设备上
        self.embedding = nnq.EmbeddingBag(
            num_embeddings=embeddingbags,
            embedding_dim=dim,
            mode=mode,
            include_last_offset=include_last_offset,
        ).to(device=device)
        # 设置随机数种子
        numpy.random.seed((1 << 32) - 1)
        # 创建 tensor 输入数据，使用 numpy 随机整数填充
        self.input = torch.tensor(
            numpy.random.randint(0, embeddingbags, input_size), device=device
        ).long()
        # 创建 tensor 偏移量数据，使用给定的 offset 参数
        offset = torch.LongTensor([offset], device=device)
        # 拼接偏移量和输入数据大小，构成完整的偏移量 tensor
        self.offset = torch.cat(
            (offset, torch.tensor([self.input.size(0)], dtype=torch.long)), 0
        )
        # 构建输入字典，包含输入数据和偏移量数据
        self.inputs = {"input": self.input, "offset": self.offset}
        # 设置模块名称
        self.set_module_name("qEmbeddingBag")

    # 前向方法，定义基准测试的执行逻辑
    def forward(self, input, offset):
        # 调用 embedding 对象进行前向计算，返回结果
        return self.embedding(input, offset)

# 生成基于 configs.embeddingbag_short_configs 的 PyTorch 测试用例，使用 QEmbeddingBagBenchmark 类
op_bench.generate_pt_test(configs.embeddingbag_short_configs, QEmbeddingBagBenchmark)

# 当文件作为脚本直接执行时，运行基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```