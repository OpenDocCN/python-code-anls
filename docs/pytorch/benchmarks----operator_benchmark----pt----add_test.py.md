# `.\pytorch\benchmarks\operator_benchmark\pt\add_test.py`

```py
# 导入operator_benchmark和torch库
import operator_benchmark as op_bench
import torch

"""Microbenchmarks for add_ operator. Supports both Caffe2/PyTorch."""

# 针对PT的add操作符的配置
add_long_configs = op_bench.cross_product_configs(
    M=[8, 128], N=[32, 64], K=[256, 512], device=["cpu", "cuda"], tags=["long"]
)

# 针对add操作的短配置
add_short_configs = op_bench.config_list(
    attr_names=["M", "N", "K"],
    attrs=[
        [1, 1, 1],
        [64, 64, 64],
        [64, 64, 128],
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],
    },
    tags=["short"],
)

# 定义一个用于测试add操作的类
class AddBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
            "input_two": torch.rand(
                M, N, K, device=device, requires_grad=self.auto_set()
            ),
        }
        self.set_module_name("add")

    def forward(self, input_one, input_two):
        return torch.add(input_one, input_two)

# 生成基于add_short_configs的测试
# 生成的测试名称遵循以下模式：
# add_M8_N16_K32_devicecpu
# add_M8_N16_K32_devicecpu_bwdall
# add_M8_N16_K32_devicecpu_bwd1
# add_M8_N16_K32_devicecpu_bwd2
# ...
# 可以用这些名称来筛选测试用例。
op_bench.generate_pt_test(add_long_configs + add_short_configs, AddBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddBenchmark)

"""Mircobenchmark for addmm operator."""

# 定义一个用于测试addmm操作的类
class AddmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(M, K, device=device, requires_grad=self.auto_set()),
            "mat1": torch.rand(M, N, device=device, requires_grad=self.auto_set()),
            "mat2": torch.rand(N, K, device=device, requires_grad=self.auto_set()),
        }
        self.set_module_name("addmm")

    def forward(self, input_one, mat1, mat2):
        return torch.addmm(input_one, mat1, mat2)

# 生成基于add_long_configs和add_short_configs的测试
op_bench.generate_pt_test(add_long_configs + add_short_configs, AddmmBenchmark)
op_bench.generate_pt_gradient_test(add_long_configs + add_short_configs, AddmmBenchmark)

"""Mircobenchmark for addr operator."""

# 定义一个用于测试addr操作的类
class AddrBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, M, N, device, dtype):
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec1": torch.rand(
                (M,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
            "vec2": torch.rand(
                (N,), device=device, requires_grad=self.auto_set(), dtype=dtype
            ),
        }
        self.set_module_name("addr")

    def forward(self, input_one, vec1, vec2):
        return torch.addr(input_one, vec1, vec2)

# addr操作的配置
addr_configs = op_bench.cross_product_configs(
    M=[8, 256],
    N=[256, 16],
    device=["cpu", "cuda"],
    dtype=[torch.float32, torch.float64]  # dtype被增加到了配置中
)
    # 定义数据类型为双精度和半精度
    dtype=[torch.double, torch.half],
    # 定义标签为"addr"
    tags=["addr"],
)

op_bench.generate_pt_test(addr_configs, AddrBenchmark)
op_bench.generate_pt_gradient_test(addr_configs, AddrBenchmark)


"""Mircobenchmark for addbmm operator."""


class AddbmmBenchmark(op_bench.TorchBenchmarkBase):
    def init(self, B, M, N, K, device):
        self.inputs = {
            "input_one": torch.rand(
                (M, N), device=device, requires_grad=self.auto_set()
            ),
            "batch1": torch.rand(
                (B, M, K), device=device, requires_grad=self.auto_set()
            ),
            "batch2": torch.rand(
                (
                    B,
                    K,
                    N,
                ),
                device=device,
                requires_grad=self.auto_set(),
            ),
        }
        self.set_module_name("addbmm")

    def forward(self, input_one, batch1, batch2):
        return torch.addbmm(input_one, batch1, batch2)


# 设置 addbmm 运算的基准测试配置
addbmm_configs = op_bench.cross_product_configs(
    B=[2, 100],  # 批次大小 B 的取值范围
    M=[8, 256],   # 矩阵维度 M 的取值范围
    N=[256, 16],  # 矩阵维度 N 的取值范围
    K=[15, 16],   # 矩阵维度 K 的取值范围
    device=["cpu", "cuda"],  # 设备选择，支持 CPU 和 CUDA
    tags=["addbmm"],  # 标记为 addbmm 操作
)

# 生成 addbmm 操作的性能测试
op_bench.generate_pt_test(addbmm_configs, AddbmmBenchmark)
# 生成 addbmm 操作的梯度性能测试
op_bench.generate_pt_gradient_test(addbmm_configs, AddbmmBenchmark)

if __name__ == "__main__":
    # 运行基准测试的主函数
    op_bench.benchmark_runner.main()
```