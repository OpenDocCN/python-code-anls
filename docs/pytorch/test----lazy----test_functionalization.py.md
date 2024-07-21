# `.\pytorch\test\lazy\test_functionalization.py`

```
# Owner(s): ["oncall: jit"]

# 导入正则表达式模块
import re

# 导入PyTorch相关模块
import torch
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend
from torch.testing._internal.common_utils import run_tests, TestCase

# 初始化懒加载的时间序列后端
torch._lazy.ts_backend.init()

# 定义一个正则表达式模式，用于匹配节点类型信息
NODE_TYPE_PATTERN = re.compile(r", NodeType=[^\n]+")


# 测试用例类，继承自TestCase
class LazyFuncionalizationTest(TestCase):

    # 测试懒加载初始化和视图功能
    def test_lazy_init_with_view(self):
        
        # 定义内部函数f，接受设备类型和是否重置存储参数
        def f(device, reset_storage=False):
            # 设定随机种子
            torch.manual_seed(2023)

            # 如果设备为"lazy"，则重置度量指标
            if device == "lazy":
                metrics.reset()

            # 定义一个简单的神经网络模型
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = torch.nn.Linear(4, 2, bias=False)

                def forward(self, x):
                    return x @ self.fc1.weight.transpose(0, 1)

            # 使用特定设备上下文创建模型
            with torch.device(device):
                model = Model()

                # 如果设备为"lazy"，进行一系列操作
                if device == "lazy":
                    # 如果需要重置存储，调用不安全的存储重置操作
                    if reset_storage:
                        torch._C._unsafe_reset_storage(model.fc1.weight)

                    # 标记步骤的懒加载操作
                    torch._lazy.mark_step()

                    # 获取与IR同步的张量计数
                    sync_tensors = metrics.counter_value("SyncedTensorsWithIR")
                    if reset_storage:
                        assert sync_tensors == 1
                    else:
                        # 如果功能存储没有重置，则会有一个额外的张量被不必要地同步
                        assert sync_tensors == 2

                # 创建输入张量
                x = torch.ones(4)
                # 将输入传递给模型
                out = model(x)

                # 如果设备为"lazy"，再次标记步骤的懒加载操作
                if device == "lazy":
                    torch._lazy.mark_step()

                return out

        # 调用f函数，使用cpu设备进行计算
        cpu_out = f("cpu")
        # 使用"lazy"设备进行计算，不重置存储
        lazy_out_1 = f("lazy", reset_storage=False)
        # 使用"lazy"设备进行计算，重置存储
        lazy_out_2 = f("lazy", reset_storage=True)

        # 断言cpu输出与lazy输出1相等（转换为cpu设备）
        self.assertEqual(cpu_out, lazy_out_1.to("cpu"))
        # 断言cpu输出与lazy输出2相等（转换为cpu设备）
        self.assertEqual(cpu_out, lazy_out_2.to("cpu"))

    # 测试数据分配功能
    def test_data_assign(self):
        
        # 定义内部函数text，接受懒加载张量作为输入，返回其文本表示
        def text(lazyt):
            raw = torch._C._lazy._get_tensors_text([lazyt])
            return NODE_TYPE_PATTERN.sub("", raw)

        # 创建一个原始张量，形状为(3,)，数据类型为float32
        origin = torch.rand(3, dtype=torch.float32)
        # 将原始张量转换为"lazy"设备上的张量
        tensor = origin.to("lazy")

        # 断言获取的文本表示与预期的IR结构相匹配
        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0, ROOT=0
}
""",
        )

        # 修改张量的数据类型，并赋值给'data'
        # 这将更新FunctionalTensorWrapper内部张量，改变对应的IR节点
        modified_tensor = tensor.to(torch.bfloat16)
        tensor.data = modified_tensor

        # 断言获取的文本表示与更新后的预期IR结构相匹配
        self.assertExpectedInline(
            text(tensor),
            """\
IR {
  %0 = [Float[3]] lazy_tensors::device_data(), device=CPU0
  %1 = [BFloat16[3]] aten::_to_copy(%0), dtype=BFloat16, layout=null, device=null, pin_memory=null, non_blocking=0, memory_format=null, ROOT=0
}
""",  # noqa: B950
        )


# 如果运行在主程序中，则执行测试
if __name__ == "__main__":
    run_tests()
```