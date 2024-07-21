# `.\pytorch\benchmarks\operator_benchmark\pt\diag_test.py`

```
# 导入 operator_benchmark 库并重命名为 op_bench
import operator_benchmark as op_bench
# 导入 PyTorch 库
import torch

"""Microbenchmarks for diag operator"""

# 配置 PT diag 操作符的参数组合列表
diag_configs_short = op_bench.config_list(
    attr_names=["dim", "M", "N", "diagonal", "out"],  # 定义参数名列表
    attrs=[
        [1, 64, 64, 0, True],      # 第一组参数值
        [2, 128, 128, -10, False], # 第二组参数值
        [1, 256, 256, 20, True],   # 第三组参数值
    ],
    cross_product_configs={
        "device": ["cpu", "cuda"],  # 针对 device 参数进行交叉组合
    },
    tags=["short"],  # 标记为 "short"
)

# 定义 DiagBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class DiagBenchmark(op_bench.TorchBenchmarkBase):
    # 初始化方法
    def init(self, dim, M, N, diagonal, out, device):
        self.inputs = {
            "input": torch.rand(M, N, device=device) if dim == 2 else torch.rand(M, device=device),  # 根据 dim 创建随机张量
            "diagonal": diagonal,  # 设置 diagonal 参数
            "out": out,  # 设置 out 参数
            "out_tensor": torch.tensor(()),  # 初始化一个空张量 out_tensor
        }
        self.set_module_name("diag")  # 设置模块名为 "diag"

    # 前向方法
    def forward(self, input, diagonal: int, out: bool, out_tensor):
        if out:
            return torch.diag(input, diagonal=diagonal, out=out_tensor)  # 如果 out 为 True，则使用预先提供的 out_tensor 计算对角线
        else:
            return torch.diag(input, diagonal=diagonal)  # 否则，仅计算对角线

# 生成基于 diag_configs_short 的 PyTorch 测试
op_bench.generate_pt_test(diag_configs_short, DiagBenchmark)

# 如果脚本被直接执行
if __name__ == "__main__":
    op_bench.benchmark_runner.main()  # 运行 benchmark_runner 的主函数
```