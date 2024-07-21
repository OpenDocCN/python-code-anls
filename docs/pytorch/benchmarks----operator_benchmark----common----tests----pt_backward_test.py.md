# `.\pytorch\benchmarks\operator_benchmark\common\tests\pt_backward_test.py`

```
# 导入名为operator_benchmark的模块，并重命名为op_bench
import operator_benchmark as op_bench

# 导入torch模块，用于进行张量操作
import torch

# 定义add_configs，通过op_bench.cross_product_configs生成不同参数组合的配置
add_configs = op_bench.cross_product_configs(
    M=[8, 1], N=[8, 2], K=[8, 4], tags=["short"]
)

# 定义AddBenchmark类，继承自op_bench.TorchBenchmarkBase类
# 用于执行加法操作的性能测试
class AddBenchmark(op_bench.TorchBenchmarkBase):
    
    # 初始化方法，接受参数M, N, K，并初始化输入张量input_one和input_two
    def init(self, M, N, K):
        # 使用torch.rand生成随机张量input_one，形状为(M, N, K)，设置requires_grad参数为self.auto_set()
        self.input_one = torch.rand(M, N, K, requires_grad=self.auto_set())
        # 使用torch.rand生成随机张量input_two，形状为(M, N, K)，设置requires_grad参数为self.auto_set()
        self.input_two = torch.rand(M, N, K, requires_grad=self.auto_set())
        # 设置模块名称为"add"
        self.set_module_name("add")
    
    # 前向传播方法，执行torch.add操作，返回input_one和input_two的和
    def forward(self):
        return torch.add(self.input_one, self.input_two)

# 使用add_configs生成基准测试的测试用例
op_bench.generate_pt_test(add_configs, AddBenchmark)

# 使用add_configs生成基准测试的梯度测试用例
op_bench.generate_pt_gradient_test(add_configs, AddBenchmark)

# 如果当前脚本作为主程序运行，则执行性能基准测试
if __name__ == "__main__":
    op_bench.benchmark_runner.main()
```