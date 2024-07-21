# `.\pytorch\benchmarks\operator_benchmark\common\tests\jit_forward_test.py`

```py
# 导入 operator_benchmark 模块并重命名为 op_bench
import operator_benchmark as op_bench

# 导入 torch 模块
import torch

# 定义了一个包含单个配置的属性列表，表示两个元素 [8, 16]
# 这些属性的名称分别是 "M" 和 "N"，并且带有 "short" 标签
intraop_bench_configs = op_bench.config_list(
    attrs=[
        [8, 16],
    ],
    attr_names=["M", "N"],
    tags=["short"],
)

# 使用 torch.jit.script 装饰器声明了一个脚本化函数 torch_sumall
@torch.jit.script
def torch_sumall(a, iterations):
    # type: (Tensor, int)
    # 初始化结果为 0.0
    result = 0.0
    # 进行 iterations 次迭代
    for _ in range(iterations):
        # 将 Tensor a 的所有元素求和，并将结果转换为 float 类型后累加到 result 中
        result += float(torch.sum(a))
        # 修改 Tensor a 的第一个元素的第一个元素的值增加 0.01
        a[0][0] += 0.01
    # 返回累加结果
    return result

# 定义了 TorchSumBenchmark 类，继承自 op_bench.TorchBenchmarkBase
class TorchSumBenchmark(op_bench.TorchBenchmarkBase):

    # 初始化方法，接受参数 M 和 N
    def init(self, M, N):
        # 生成一个大小为 MxN 的随机 Tensor，作为类的属性 input_one
        self.input_one = torch.rand(M, N)
        # 设置模块名称为 "sum"
        self.set_module_name("sum")

    # 这是一个非常临时的方法，将会很快移除，因此不要在您的基准测试中使用此方法
    # TODO(mingzhe): 使用一个前向方法来兼容 JIT 和 Eager
    # jit_forward 方法接受参数 iters，并调用 torch_sumall 函数
    def jit_forward(self, iters):
        return torch_sumall(self.input_one, iters)

# 使用 op_bench 模块的 generate_pt_test 函数生成基于 intraop_bench_configs 配置的基准测试
op_bench.generate_pt_test(intraop_bench_configs, TorchSumBenchmark)

# 如果该脚本作为主程序运行，则调用 op_bench 模块的 benchmark_runner.main() 函数来运行基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```