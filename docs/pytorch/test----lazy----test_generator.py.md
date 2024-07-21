# `.\pytorch\test\lazy\test_generator.py`

```
# Owner(s): ["oncall: jit"]

# 导入PyTorch库
import torch
# 导入PyTorch的度量模块
import torch._lazy.metrics as metrics
# 导入PyTorch的时间序列后端模块
import torch._lazy.ts_backend
# 导入测试工具函数和类
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase

# 初始化时间序列后端
torch._lazy.ts_backend.init()

# 测试用例类，继承自TestCase
class LazyGeneratorTest(TestCase):
    
    # 测试生成器函数
    def test_generator(self):
        """
        Test that generators are being inserted into the TorchScript
        graph by setting different seeds before each call to
        generate_tensor but the resulting tensor is the same
        """
        
        # 定义生成张量的函数
        def generate_tensor():
            # 创建第一个生成器并设置种子为2023
            g1 = torch.Generator()
            g1.manual_seed(2023)
            # 创建张量t1，并使用g1生成器生成均匀分布的随机数
            t1 = torch.tensor(1.0)
            t1.uniform_(generator=g1)
            
            # 创建第二个生成器并设置种子为2024
            g2 = torch.Generator()
            g2.manual_seed(2024)
            # 创建张量t2，并使用g2生成器生成正态分布的随机数
            t2 = torch.tensor(1.0)
            t2.normal_(generator=g2)
            
            return t1, t2
        
        # 设置全局种子为1
        torch.manual_seed(1)
        
        # 在CPU设备上生成张量cpu_t1和cpu_t2
        with torch.device("cpu"):
            cpu_t1, cpu_t2 = generate_tensor()
        
        # 设置全局种子为2
        torch.manual_seed(2)
        
        # 在"lazy"设备上生成张量lazy_t1和lazy_t2
        with torch.device("lazy"):
            lazy_t1, lazy_t2 = generate_tensor()
        
        # 在Lazy模式下标记步骤
        torch._lazy.mark_step()
        
        # 断言cpu_t1与lazy_t1.to("cpu")在数值上的接近性
        assert torch.allclose(
            cpu_t1, lazy_t1.to("cpu")
        ), f"Expected {cpu_t1}, got {lazy_t1.to('cpu')}"
        
        # 断言cpu_t2与lazy_t2.to("cpu")在数值上的接近性
        assert torch.allclose(
            cpu_t2, lazy_t2.to("cpu")
        ), f"Expected {cpu_t2}, got {lazy_t2.to('cpu')}"

    # 如果是Torch Dynamo环境，则跳过该测试
    @skipIfTorchDynamo("Torch Dynamo does not support torch.Generator type")
    # 定义一个测试函数，用于验证生成器使用不同种子会导致重新编译
    def test_generator_causes_multiple_compiles(self):
        """
        Test that inserting generators with different seed caused recompile
        """

        # 定义一个生成张量的函数，接受一个种子作为参数
        def generate_tensor(seed):
            # 创建一个值为 1.0 的张量
            t = torch.tensor(1.0)
            # 创建一个随机数生成器对象
            g = torch.Generator()
            # 设置随机数生成器的种子为给定的种子
            g.manual_seed(seed)
            # 使用指定的生成器生成均匀分布在[-1, 1]之间的随机数填充张量t
            t.uniform_(-1, 1, generator=g)
            return t

        # 重置度量指标
        metrics.reset()

        # 使用lazy设备上下文管理器
        with torch.device("lazy"):
            # 生成种子为1的张量
            t = generate_tensor(1)
            # 标记一个步骤
            torch._lazy.mark_step()

            # 检查未缓存编译的计数器值
            uncached_compile = metrics.counter_value("UncachedCompile")
            assert (
                uncached_compile == 1
            ), f"Expected 1 uncached compiles, got {uncached_compile}"

            # 生成种子为2的张量
            t = generate_tensor(2)
            # 标记一个步骤
            torch._lazy.mark_step()

            # 再次检查未缓存编译的计数器值
            uncached_compile = metrics.counter_value("UncachedCompile")
            assert (
                uncached_compile == 2
            ), f"Expected 2 uncached compiles, got {uncached_compile}"

            # 再次生成种子为1的张量
            t = generate_tensor(1)
            # 标记一个步骤
            torch._lazy.mark_step()

            # 再次检查未缓存编译的计数器值
            uncached_compile = metrics.counter_value("UncachedCompile")
            assert (
                uncached_compile == 2
            ), f"Expected 2 uncached compiles, got {uncached_compile}"
            
            # 检查缓存编译的计数器值
            cached_compile = metrics.counter_value("CachedCompile")
            assert (
                cached_compile == 1
            ), f"Expected 1 cached compile, got {cached_compile}"

        # 重置度量指标
        metrics.reset()

        # 获取最新的计算图
        latest_graph = torch._C._lazy_ts_backend._get_latest_computation_graph()
        # 断言生成的计算图中包含特定信息
        assert 'torch.Generator(device="cpu", seed=1)' in latest_graph
        assert "aten::uniform" in latest_graph
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```