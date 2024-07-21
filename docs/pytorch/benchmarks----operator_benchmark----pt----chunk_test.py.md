# `.\pytorch\benchmarks\operator_benchmark\pt\chunk_test.py`

```
# 导入自定义的操作基准测试模块
import operator_benchmark as op_bench
# 导入PyTorch库
import torch

"""Microbenchmarks for Chunk operator"""

# 配置短测试用例的PT Chunk操作
chunk_short_configs = op_bench.config_list(
    attr_names=["M", "N", "chunks"],  # 定义属性名称列表
    attrs=[
        [8, 8, 2],     # 第一个配置：M=8, N=8, chunks=2
        [256, 512, 2], # 第二个配置：M=256, N=512, chunks=2
        [512, 512, 2], # 第三个配置：M=512, N=512, chunks=2
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 在不同设备上进行交叉测试：CPU和CUDA
    },
    tags=["short"],  # 标记为短测试用例
)

# 长测试用例的配置：跨多个参数进行交叉测试
chunks_long_configs = op_bench.cross_product_configs(
    M=[128, 1024],   # M的取值：128和1024
    N=[128, 1024],   # N的取值：128和1024
    chunks=[2, 4],   # chunks的取值：2和4
    device=["cpu", "cuda"],  # 在CPU和CUDA上进行测试
    tags=["long"]    # 标记为长测试用例
)

# 定义ChunkBenchmark类，继承自op_bench.TorchBenchmarkBase
class ChunkBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, chunks, device):
        # 初始化输入字典，包含输入张量和chunk数目
        self.inputs = {"input_one": torch.rand(M, N, device=device), "chunks": chunks}
        # 设置模块名称为"chunk"
        self.set_module_name("chunk")

    def forward(self, input_one, chunks: int):
        # 调用torch.chunk进行张量分块操作
        return torch.chunk(input_one, chunks)

# 生成PyTorch的性能测试用例，包括短测试用例和长测试用例
op_bench.generate_pt_test(chunk_short_configs + chunks_long_configs, ChunkBenchmark)

# 如果当前脚本作为主程序运行，则执行操作基准测试的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```