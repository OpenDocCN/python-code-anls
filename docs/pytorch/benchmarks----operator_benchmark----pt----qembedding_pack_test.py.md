# `.\pytorch\benchmarks\operator_benchmark\pt\qembedding_pack_test.py`

```py
# 导入 operator_benchmark 库和 torch 库
import operator_benchmark as op_bench
import torch

# 定义短配置的嵌入袋转换基准测试参数组合
embeddingbag_conversion_short_configs = op_bench.cross_product_configs(
    num_embeddings=(80,), embedding_dim=(128, 256, 512), tags=("short",)
)

# 定义长配置的嵌入袋转换基准测试参数组合
embeddingbag_conversion_long_configs = op_bench.cross_product_configs(
    num_embeddings=(100, 120, 1000),
    embedding_dim=(16, 64, 128, 256, 512, 1024, 2048),
    tags=("long",),
)

# 定义三维配置的嵌入袋转换基准测试参数组合
embeddingbag_conversion_three_dim_configs = op_bench.cross_product_configs(
    num_embeddings=(80,),
    embedding_dim=(128, 256, 512),
    batch_size=(10,),
    tags=("short",),
)

# 定义转换操作列表
conversion_ops = op_bench.op_list(
    attrs=(
        ("qembeddingbag_byte_prepack", torch.ops.quantized.embedding_bag_byte_prepack),
        ("qembeddingbag_4bit_prepack", torch.ops.quantized.embedding_bag_4bit_prepack),
        ("qembeddingbag_2bit_prepack", torch.ops.quantized.embedding_bag_2bit_prepack),
    ),
    attr_names=("op_name", "op_func"),
)

# 定义解包操作列表
unpack_ops = op_bench.op_list(
    attrs=(
        ("qembeddingbag_byte_unpack", torch.ops.quantized.embedding_bag_byte_unpack),
        ("qembeddingbag_4bit_unpack", torch.ops.quantized.embedding_bag_4bit_unpack),
        ("qembeddingbag_2bit_unpack", torch.ops.quantized.embedding_bag_2bit_unpack),
    ),
    attr_names=("op_name", "op_func"),
)

# 定义嵌入袋浮点到融合格式基准测试类
class EmbeddingBagFloatToFusedBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, op_func):
        self.inputs = {
            "weight": torch.rand(num_embeddings, embedding_dim, dtype=torch.float) + 1
        }
        self.op_func = op_func

    def forward(self, weight):
        return self.op_func(weight)

# 定义三维嵌入袋浮点到融合格式基准测试类
class EmbeddingBagThreeDimFloatToFusedBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, batch_size, op_func):
        self.inputs = {
            "weight": torch.rand(
                batch_size, num_embeddings, embedding_dim, dtype=torch.float
            )
            + 1
        }
        self.op_func = op_func

    def forward(self, weight):
        return self.op_func(weight)

# 定义融合格式到浮点嵌入袋基准测试类
class EmbeddingBagFusedToFloatBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, op_func):
        weight = torch.randn(num_embeddings, embedding_dim + 8, dtype=torch.float)
        self.inputs = {"packed_weight": weight.to(torch.uint8)}
        self.op_func = op_func

    def forward(self, packed_weight):
        return self.op_func(packed_weight)

# 定义三维融合格式到浮点嵌入袋基准测试类
class EmbeddingBagThreeDimFusedToFloatBase(op_bench.TorchBenchmarkBase):
    def init(self, num_embeddings, embedding_dim, batch_size, op_func):
        weight = torch.randn(
            batch_size, num_embeddings, embedding_dim + 8, dtype=torch.float
        )
        self.inputs = {"packed_weight": weight.to(torch.uint8)}
        self.op_func = op_func

    def forward(self, packed_weight):
        return self.op_func(packed_weight)

# 从转换操作列表生成 PyTorch 测试
op_bench.generate_pt_tests_from_op_list(
    conversion_ops,
    # 将 embeddingbag_conversion_short_configs 和 embeddingbag_conversion_long_configs 的内容合并为一个列表
    embeddingbag_conversion_short_configs + embeddingbag_conversion_long_configs,
    # 使用 EmbeddingBagFloatToFusedBase 类进行转换
    EmbeddingBagFloatToFusedBase,
# 调用op_bench.generate_pt_tests_from_op_list函数，生成性能测试的操作列表，并生成测试用例
op_bench.generate_pt_tests_from_op_list(
    # 使用unpack_ops作为操作列表
    unpack_ops,
    # 将embeddingbag_conversion_short_configs和embeddingbag_conversion_long_configs合并作为配置参数传递
    embeddingbag_conversion_short_configs + embeddingbag_conversion_long_configs,
    # 使用EmbeddingBagFusedToFloatBase作为基础类进行测试
    EmbeddingBagFusedToFloatBase,
)

# 再次调用op_bench.generate_pt_tests_from_op_list函数，生成性能测试的操作列表，并生成测试用例
op_bench.generate_pt_tests_from_op_list(
    # 使用conversion_ops作为操作列表
    conversion_ops,
    # 将embeddingbag_conversion_three_dim_configs作为配置参数传递
    embeddingbag_conversion_three_dim_configs,
    # 使用EmbeddingBagThreeDimFloatToFusedBase作为基础类进行测试
    EmbeddingBagThreeDimFloatToFusedBase,
)

# 再次调用op_bench.generate_pt_tests_from_op_list函数，生成性能测试的操作列表，并生成测试用例
op_bench.generate_pt_tests_from_op_list(
    # 使用unpack_ops作为操作列表
    unpack_ops,
    # 将embeddingbag_conversion_three_dim_configs作为配置参数传递
    embeddingbag_conversion_three_dim_configs,
    # 使用EmbeddingBagThreeDimFusedToFloatBase作为基础类进行测试
    EmbeddingBagThreeDimFusedToFloatBase,
)

# 如果当前脚本被直接执行（而非被导入到其他脚本中）
if __name__ == "__main__":
    # 运行op_bench.benchmark_runner模块的主函数，启动性能测试
    op_bench.benchmark_runner.main()
```