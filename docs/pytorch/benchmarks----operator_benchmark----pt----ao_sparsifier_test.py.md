# `.\pytorch\benchmarks\operator_benchmark\pt\ao_sparsifier_test.py`

```py
# 导入 operator_benchmark 模块作为 op_bench 别名，用于性能基准测试
import operator_benchmark as op_bench
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 torch.ao 模块中导入 pruning 模块
from torch.ao import pruning

# 定义短稀疏配置列表 sparse_configs_short
sparse_configs_short = op_bench.config_list(
    # 设置属性名称列表 ["M", "SL", "SBS", "ZPB"]
    attr_names=["M", "SL", "SBS", "ZPB"],
    # 设置属性值列表
    attrs=[
        [(32, 16), 0.3, (4, 1), 2],
        [(32, 16), 0.6, (1, 4), 4],
        [(17, 23), 0.9, (1, 1), 1],
    ],
    # 设置标签为 ("short",)
    tags=("short",),
)

# 定义长稀疏配置列表 sparse_configs_long
sparse_configs_long = op_bench.cross_product_configs(
    # 设置 M 参数的取值范围
    M=((128, 128), (255, 324)),  # Mask shape
    # 设置 SL 参数的取值范围
    SL=(0.0, 1.0, 0.3, 0.6, 0.9, 0.99),  # Sparsity level
    # 设置 SBS 参数的取值范围
    SBS=((1, 4), (1, 8), (4, 1), (8, 1)),  # Sparse block shape
    # 设置 ZPB 参数的取值范围
    ZPB=(0, 1, 2, 3, 4, None),  # Zeros per block
    # 设置标签为 ("long",)
    tags=("long",),
)

# 定义 WeightNormSparsifierBenchmark 类，继承自 op_bench.TorchBenchmarkBase 类
class WeightNormSparsifierBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法，接受 M, SL, SBS, ZPB 四个参数
    def init(self, M, SL, SBS, ZPB):
        # 创建尺寸为 M 的全 1 张量 weight
        weight = torch.ones(M)
        # 创建一个空的 nn.Module 实例 model
        model = nn.Module()
        # 将 weight 注册为 model 的缓冲区（buffer）
        model.register_buffer("weight", weight)

        # 定义稀疏配置 sparse_config，指定稀疏化的目标张量为 "weight"
        sparse_config = [{"tensor_fqn": "weight"}]
        # 创建 WeightNormSparsifier 实例 self.sparsifier
        self.sparsifier = pruning.WeightNormSparsifier(
            sparsity_level=SL,  # 稀疏水平参数
            sparse_block_shape=SBS,  # 稀疏块形状参数
            zeros_per_block=ZPB,  # 每个块的零数参数
        )
        # 在模型 model 上准备稀疏化，应用 sparse_config 的配置
        self.sparsifier.prepare(model, config=sparse_config)
        # 初始化输入 inputs，这是基准测试所需的输入
        self.inputs = {}  # All benchmarks need inputs :)
        # 设置模块名称为 "weight_norm_sparsifier_step"
        self.set_module_name("weight_norm_sparsifier_step")

    # 前向方法，执行稀疏化的一步操作
    def forward(self):
        self.sparsifier.step()

# 将短稀疏配置列表和长稀疏配置列表合并为一个列表 all_tests
all_tests = sparse_configs_short + sparse_configs_long
# 生成基于 PyTorch 的性能测试，测试类为 WeightNormSparsifierBenchmark
op_bench.generate_pt_test(all_tests, WeightNormSparsifierBenchmark)

# 如果当前脚本作为主程序运行，则执行 operator_benchmark 的主函数
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```