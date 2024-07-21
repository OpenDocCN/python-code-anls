# `.\pytorch\benchmarks\operator_benchmark\pt\qembedding_bag_lookups_test.py`

```
# 导入必要的模块和类型声明
from typing import Optional

# 导入常用的科学计算库 numpy
import numpy as np

# 导入性能基准测试相关的模块和函数
import operator_benchmark as op_bench

# 导入 PyTorch 深度学习框架
import torch

# 导入内部用于量化的常见函数
from torch.testing._internal.common_quantization import lengths_to_offsets

# 加载自定义的 C++ 扩展库，包含稀疏神经网络相关的运算符
torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")

# 定义短配置参数，用于 embedding_bag_rowwise_offsets 测试
embedding_bag_rowwise_offsets_short_configs = op_bench.cross_product_configs(
    num_embeddings=(80,),  # 嵌入的数量为 80
    embedding_dim=(128, 256),  # 嵌入的维度为 128 或 256
    num_offsets=range(2, 10),  # 偏移量的数量在 2 到 9 之间
    enable_per_sample_weights=(True, False),  # 是否启用每个样本的权重
    include_last_offset=(True, False),  # 是否包含最后一个偏移量
    is_pruned_weights=(True, False),  # 是否为剪枝权重
    use_32bit_indices=(True, False),  # 是否使用 32 位索引
    use_32bit_offsets=(True, False),  # 是否使用 32 位偏移量
    tags=["short"],  # 配置标签为 "short"
)

# 定义长配置参数，用于 embedding_bag_rowwise_offsets 测试
embedding_bag_rowwise_offsets_long_configs = op_bench.cross_product_configs(
    num_embeddings=(100, 120, 1000, 10_000, 20_000),  # 多个不同的嵌入数量
    embedding_dim=(16, 64, 128, 256),  # 多个不同的嵌入维度
    num_offsets=range(10, 20),  # 偏移量的数量在 10 到 19 之间
    enable_per_sample_weights=(True, False),  # 是否启用每个样本的权重
    include_last_offset=(True, False),  # 是否包含最后一个偏移量
    is_pruned_weights=(True, False),  # 是否为剪枝权重
    use_32bit_indices=(True, False),  # 是否使用 32 位索引
    use_32bit_offsets=(True, False),  # 是否使用 32 位偏移量
    tags=["long"],  # 配置标签为 "long"
)

# 将短配置和长配置合并成完整的配置列表
full_configs = (
    embedding_bag_rowwise_offsets_short_configs
    + embedding_bag_rowwise_offsets_long_configs
)

# 定义支持 4 位量化的行逐偏移操作列表
four_bit_rowwise_ops = op_bench.op_list(
    attrs=(
        (
            "qembeddingbag_4bit_rowwise_offsets",  # 操作的名称
            torch.ops.quantized.embedding_bag_4bit_rowwise_offsets,  # 操作的函数
        ),
    ),
    attr_names=("op_name", "op_func"),  # 属性名称列表
)

# 定义支持字节量化的行逐偏移操作列表
byte_rowwise_ops = op_bench.op_list(
    attrs=(
        (
            "qembeddingbag_byte_rowwise_offsets",  # 操作的名称
            torch.ops.quantized.embedding_bag_byte_rowwise_offsets,  # 操作的函数
        ),
    ),
    attr_names=("op_name", "op_func"),  # 属性名称列表
)

# 定义函数用于获取剪枝权重和索引映射
def get_pruned_weights_and_mapping(q_weights):
    # 创建一个指示器张量，其值在 -1.0 到 1.0 之间的均匀分布
    indicator = torch.from_numpy(
        np.random.uniform(low=-1.0, high=1.0, size=[q_weights.shape[0]]).astype(
            np.float32
        )
    )

    # 调用 C++ 扩展的函数进行行逐偏移剪枝
    q_pruned_weights, compressed_indices_mapping = torch.ops.fb.embedding_bag_rowwise_prune(
        q_weights, indicator, 0.01, torch.int32
    )

    return q_pruned_weights, compressed_indices_mapping

# 定义一个基于 op_bench.TorchBenchmarkBase 的测试类
class EmbedddingBag4BitRowwiseOffsetsTest(op_bench.TorchBenchmarkBase):
    # 初始化方法，设定测试参数和操作函数
    def init(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_offsets: int,
        enable_per_sample_weights: bool,
        include_last_offset: bool,
        is_pruned_weights: bool,
        use_32bit_indices: bool,
        use_32bit_offsets: bool,
        op_func,
        # 初始化类的成员变量，指定嵌入向量的数量
        ):
        # 设置类的成员变量：嵌入向量的数量
        self.num_embeddings = num_embeddings
        # 设置类的成员变量：嵌入向量的维度
        self.embedding_dim = embedding_dim
        # 设置类的成员变量：偏移量的数量
        self.num_offsets = num_offsets
        # 设置类的成员变量：是否启用每个样本的权重
        self.enable_per_sample_weights = enable_per_sample_weights
        # 设置类的成员变量：是否包括最后一个偏移量
        self.include_last_offset = include_last_offset
        # 设置类的成员变量：最大段长度为20
        self.max_segment_length = 20
        # 根据随机生成的段数，初始化类的成员变量：长度列表
        self.num_lengths = np.random.randint(1, num_offsets + 1)
        self.lengths = np.random.randint(
            0, self.max_segment_length + 1, size=self.num_lengths
        ).astype(np.int32)
        # 计算总索引数量
        self.num_indices = np.sum(self.lengths)
        # 设置类的成员变量：是否为裁剪过的权重
        self.is_pruned_weights = is_pruned_weights
        # 设置类的成员变量：是否使用32位整数索引
        self.use_32bit_indices = use_32bit_indices
        # 设置类的成员变量：是否使用32位整数偏移量
        self.use_32bit_offsets = use_32bit_offsets

        # 根据长度列表计算偏移量列表
        self.offsets = lengths_to_offsets(self.lengths)
        # 随机生成索引数组，并转换为PyTorch张量
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )

        # 如果使用32位整数索引，则将索引张量转换为32位整数类型
        self.indices = self.indices.int() if self.use_32bit_indices else self.indices
        # 如果使用32位整数偏移量，则将偏移量张量转换为32位整数类型
        self.offsets = self.offsets.int() if self.use_32bit_offsets else self.offsets

        # 如果包括最后一个偏移量，则将其追加到偏移量张量末尾
        if self.include_last_offset:
            self.offsets = torch.cat(
                (self.offsets, torch.tensor([self.indices.size(0)], dtype=torch.long)),
                0,
            )

        # 随机初始化权重矩阵，并转换为PyTorch张量
        self.weights = torch.from_numpy(
            (
                np.random.random_sample((self.num_embeddings, self.embedding_dim)) + 1
            ).astype(np.float32)
        )
        # 重新生成索引数组，并转换为PyTorch张量
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=self.num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )
        # 设置预打包函数为指定的量化嵌入包装函数
        self.prepack_func = torch.ops.quantized.embedding_bag_4bit_prepack

        # 使用预打包函数对权重进行预打包
        self.prepacked_weights = self.prepack_func(self.weights)
        
        # 如果启用每个样本的权重，则随机生成权重数组，并转换为PyTorch张量
        self.per_sample_weights = (
            torch.from_numpy(
                np.random.uniform(low=0.01, high=0.5, size=[len(self.indices)]).astype(
                    np.float32
                )
            )
            if self.enable_per_sample_weights
            else None
        )

        # 初始化压缩后的索引为空
        self.compressed_indices = None

        # 如果为裁剪过的权重，则获取裁剪后的权重和映射
        if self.is_pruned_weights:
            (
                self.prepacked_weights,
                self.compressed_indices,
            ) = get_pruned_weights_and_mapping(self.prepacked_weights)

        # 构建输入字典，包含所有必要的张量和标志
        self.inputs = {
            "prepacked_weights": self.prepacked_weights,
            "indices": self.indices,
            "offsets": self.offsets,
            "mode": 0,
            "per_sample_weights": self.per_sample_weights,
            "include_last_offset": self.include_last_offset,
            "is_pruned_weights": self.is_pruned_weights,
            "compressed_indices": self.compressed_indices,
        }

        # 设置操作函数为指定的操作函数
        self.op_func = op_func
    # 定义一个方法 `forward`，用于执行前向传播操作
    def forward(
        self,
        prepacked_weights,                      # 输入参数：预打包的权重数据
        indices,                                # 输入参数：索引数据
        offsets,                                # 输入参数：偏移量数据
        mode: int,                              # 输入参数：模式，整数类型
        per_sample_weights: Optional[torch.Tensor],  # 输入参数：每个样本的权重，可选的张量类型
        include_last_offset: bool,              # 输入参数：是否包含最后一个偏移量，布尔类型
        is_pruned_weights: bool,                # 输入参数：是否为裁剪后的权重，布尔类型
        compressed_indices: Optional[torch.Tensor],  # 输入参数：压缩后的索引映射，可选的张量类型
    ):
        # 调用实例对象的 `op_func` 方法来执行操作
        return self.op_func(
            prepacked_weights,                  # 将预打包的权重传递给操作函数
            indices,                            # 将索引数据传递给操作函数
            offsets,                            # 将偏移量数据传递给操作函数
            mode=mode,                          # 将模式传递给操作函数
            per_sample_weights=per_sample_weights,  # 将每个样本的权重传递给操作函数
            include_last_offset=include_last_offset,  # 将是否包含最后一个偏移量传递给操作函数
            pruned_weights=is_pruned_weights,   # 将是否裁剪后的权重传递给操作函数
            compressed_indices_mapping=compressed_indices,  # 将压缩后的索引映射传递给操作函数
        )
# 定义一个名为 EmbedddingBagByteRowwiseOffsetsTest 的类，继承自 op_bench.TorchBenchmarkBase
class EmbedddingBagByteRowwiseOffsetsTest(op_bench.TorchBenchmarkBase):

    # 初始化方法，设置类的初始属性和参数
    def init(
        self,
        num_embeddings: int,             # 参数：嵌入数量（整数）
        embedding_dim: int,              # 参数：嵌入维度（整数）
        num_offsets: int,                # 参数：偏移量数量（整数）
        enable_per_sample_weights: bool, # 参数：是否启用每个样本的权重（布尔值）
        include_last_offset: bool,       # 参数：是否包括最后一个偏移量（布尔值）
        is_pruned_weights: bool,         # 参数：是否是修剪过的权重（布尔值）
        use_32bit_indices: bool,         # 参数：是否使用32位索引（布尔值）
        use_32bit_offsets: bool,         # 参数：是否使用32位偏移量（布尔值）
        op_func,                         # 参数：操作函数（未指定类型）
        ):
        # 初始化 EmbeddingBag 参数
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_offsets = num_offsets
        self.enable_per_sample_weights = enable_per_sample_weights
        self.include_last_offset = include_last_offset
        self.max_segment_length = 20
        # 随机生成一个小于等于 num_offsets 的整数，表示长度数组的长度
        self.num_lengths = np.random.randint(1, num_offsets + 1)
        # 随机生成长度数组，每个元素的取值范围是 [0, max_segment_length]
        self.lengths = np.random.randint(
            0, self.max_segment_length + 1, size=self.num_lengths
        ).astype(np.int32)
        self.is_pruned_weights = is_pruned_weights
        self.use_32bit_indices = use_32bit_indices
        self.use_32bit_offsets = use_32bit_offsets

        # 计算总索引数
        self.num_indices = np.sum(self.lengths)
        # 将长度数组转换为偏移数组
        self.offsets = lengths_to_offsets(self.lengths)
        # 生成随机索引数组，元素取值范围是 [0, num_embeddings)
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )

        # 根据 use_32bit_indices 决定是否转换索引数组的数据类型为 int
        self.indices = self.indices.int() if self.use_32bit_indices else self.indices
        # 根据 use_32bit_offsets 决定是否转换偏移数组的数据类型为 int
        self.offsets = self.offsets.int() if self.use_32bit_offsets else self.offsets

        # 如果 include_last_offset 为真，则将最后一个索引的偏移添加到偏移数组末尾
        if include_last_offset:
            self.offsets = torch.cat(
                (self.offsets, torch.tensor([self.indices.size(0)], dtype=torch.long)),
                0,
            )

        # 随机生成权重矩阵，元素取值范围是 (1, 2)
        self.weights = torch.from_numpy(
            (
                np.random.random_sample((self.num_embeddings, self.embedding_dim)) + 1
            ).astype(np.float32)
        )
        # 重新生成索引数组，元素取值范围是 [0, num_embeddings)
        self.indices = torch.from_numpy(
            np.random.randint(
                low=0, high=self.num_embeddings, size=self.num_indices, dtype=np.int64
            )
        )

        # 设置预打包函数为量化嵌入包预打包函数
        self.prepack_func = torch.ops.quantized.embedding_bag_byte_prepack

        # 使用预打包函数对权重矩阵进行预打包
        self.prepacked_weights = self.prepack_func(self.weights)
        
        # 如果启用每样本权重，则生成每样本权重数组，元素取值范围是 [0.01, 0.5)
        self.per_sample_weights = (
            torch.from_numpy(
                np.random.uniform(low=0.01, high=0.5, size=[len(self.indices)]).astype(
                    np.float32
                )
            )
            if self.enable_per_sample_weights
            else None
        )

        # 初始化压缩索引为 None
        self.compressed_indices = None

        # 如果使用剪枝权重，则获取剪枝后的权重和映射索引
        if self.is_pruned_weights:
            (
                self.prepacked_weights,
                self.compressed_indices,
            ) = get_pruned_weights_and_mapping(self.prepacked_weights)

        # 构建输入字典，包含所需的所有输入参数
        self.inputs = {
            "prepacked_weights": self.prepacked_weights,
            "indices": self.indices,
            "offsets": self.offsets,
            "mode": 0,
            "per_sample_weights": self.per_sample_weights,
            "include_last_offset": self.include_last_offset,
            "is_pruned_weights": self.is_pruned_weights,
            "compressed_indices": self.compressed_indices,
        }

        # 设置操作函数
        self.op_func = op_func
    # 定义一个方法 forward，接受多个参数：
    # - prepacked_weights: 预打包的权重数据
    # - indices: 索引数据
    # - offsets: 偏移量数据
    # - mode: 模式，应为整数
    # - per_sample_weights: 可选的每个样本权重张量
    # - include_last_offset: 布尔值，指示是否包含最后一个偏移量
    # - is_pruned_weights: 布尔值，指示是否是裁剪过的权重
    # - compressed_indices: 可选的压缩索引张量
    def forward(
        self,
        prepacked_weights,
        indices,
        offsets,
        mode: int,
        per_sample_weights: Optional[torch.Tensor],
        include_last_offset: bool,
        is_pruned_weights: bool,
        compressed_indices: Optional[torch.Tensor],
    ):
        # 调用 self.op_func 方法，传递参数进行前向计算
        return self.op_func(
            prepacked_weights,
            indices,
            offsets,
            mode=0,  # 将 mode 参数设为固定值 0
            per_sample_weights=per_sample_weights,
            include_last_offset=self.include_last_offset,  # 使用对象自身的 include_last_offset 属性
            pruned_weights=self.is_pruned_weights,  # 使用对象自身的 is_pruned_weights 属性
            compressed_indices_mapping=self.compressed_indices,  # 使用对象自身的 compressed_indices 属性
        )
# 从给定的操作列表和配置生成 PyTorch 测试，用于四位行逐行偏移操作
op_bench.generate_pt_tests_from_op_list(
    four_bit_rowwise_ops, full_configs, EmbedddingBag4BitRowwiseOffsetsTest
)

# 从给定的操作列表和配置生成 PyTorch 测试，用于字节行逐行偏移操作
op_bench.generate_pt_tests_from_op_list(
    byte_rowwise_ops, full_configs, EmbedddingBagByteRowwiseOffsetsTest
)

# 如果当前脚本作为主程序运行，执行基准测试运行器的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```