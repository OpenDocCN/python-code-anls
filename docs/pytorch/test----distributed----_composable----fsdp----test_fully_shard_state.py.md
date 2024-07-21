# `.\pytorch\test\distributed\_composable\fsdp\test_fully_shard_state.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块
import copy  # 导入复制模块
import unittest  # 导入单元测试模块

import torch.nn as nn  # 导入PyTorch神经网络模块
from torch.distributed._composable.fsdp import FSDPModule, fully_shard  # 导入FSDP相关模块
from torch.testing._internal.common_cuda import TEST_CUDA  # 导入CUDA测试相关模块
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, MLP  # 导入FSDP测试相关模块和MLP模型
from torch.testing._internal.common_utils import run_tests  # 导入运行测试的辅助函数

# 测试类，继承自FSDPTestMultiThread
class TestFullyShardState(FSDPTestMultiThread):
    
    @property
    def world_size(self) -> int:
        return 1  # 返回虚拟的世界大小为1
    
    @unittest.skipIf(not TEST_CUDA, "no cuda")  # 如果没有CUDA支持，则跳过测试
    def test_fully_shard_state(self):
        """
        Tests the ability to get the state object from a fully sharded module.
        测试从完全分片模块获取状态对象的能力。
        """
        num_mlps = 3  # 定义MLP模型的数量
        model = nn.Sequential(*[MLP(8) for _ in range(num_mlps)])  # 创建包含多个MLP模块的序列模型
        for mlp in model:
            fully_shard(mlp)  # 对每个MLP模块进行完全分片
        fully_shard(model)  # 对整个模型进行完全分片
        root_state = fully_shard.state(model)  # 获取根模型的状态对象
        self.assertTrue(root_state is not None)  # 断言根状态对象不为空
        all_states = [root_state] + [fully_shard.state(mlp) for mlp in model]  # 收集所有模型及其子模块的状态对象
        # 检查每次`fully_shard`调用是否构造了不同的状态对象
        self.assertEqual(len(set(all_states)), num_mlps + 1)

    @unittest.skipIf(not TEST_CUDA, "no cuda")  # 如果没有CUDA支持，则跳过测试
    def test_fully_shard_reapply(self):
        model = MLP(8)  # 创建一个MLP模型
        fully_shard(model)  # 对模型进行完全分片
        with self.assertRaisesRegex(
            AssertionError,
            "Each distinct composable distributed API can only be applied to a module once.",
        ):
            fully_shard(model)  # 再次对同一模型应用`fully_shard`，预期抛出断言错误

    @unittest.skipIf(not TEST_CUDA, "no cuda")  # 如果没有CUDA支持，则跳过测试
    def test_fully_shard_cls(self):
        # 检查仅交换传递给`fully_shard`的模块的类
        model = MLP(8)  # 创建一个MLP模型
        fully_shard(model)  # 对模型进行完全分片
        self.assertTrue(isinstance(model, MLP))  # 断言模型仍然是MLP类的实例
        self.assertTrue(isinstance(model, FSDPModule))  # 断言模型现在是FSDPModule类的实例
        self.assertEqual(model.__class__.__name__, "FSDPMLP")  # 断言模型类名为"FSDPMLP"
        for module in model.modules():
            if module is model:
                continue
            self.assertFalse(isinstance(module, FSDPModule))  # 断言模型中除了根模块外的其他模块不是FSDPModule的实例

        # 检查切片操作不会保留FSDP特性
        model = nn.Sequential(*[MLP(8) for _ in range(3)])  # 创建包含多个MLP模块的序列模型
        fully_shard(model)  # 对序列模型进行完全分片
        self.assertTrue(isinstance(model, nn.Sequential))  # 断言模型仍然是nn.Sequential类的实例
        self.assertTrue(isinstance(model, FSDPModule))  # 断言模型现在是FSDPModule类的实例
        self.assertEqual(model.__class__.__name__, "FSDPSequential")  # 断言模型类名为"FSDPSequential"
        sliced_model = model[:2]  # 对模型进行切片操作
        self.assertTrue(isinstance(sliced_model, nn.Sequential))  # 断言切片后的模型仍然是nn.Sequential类的实例
        self.assertFalse(isinstance(sliced_model, FSDPModule))  # 断言切片后的模型不是FSDPModule的实例

    @unittest.skipIf(not TEST_CUDA, "no cuda")  # 如果没有CUDA支持，则跳过测试
    # 定义测试方法，用于测试不支持完全分片的模块类
    def test_fully_shard_unsupported_module_cls(self):
        # 定义正则表达式，用于匹配错误信息
        regex = (
            r"fully\_shard does not support containers that do not implement forward"
        )
        # 创建包含三个MLP(8)模块的模块列表
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        # 断言抛出值错误，并匹配指定的正则表达式
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)
        # 创建包含键为"1"和"2"的MLP(8)模块的模块字典
        model = nn.ModuleDict({"1": MLP(8), "2": MLP(8)})
        # 断言抛出值错误，并匹配指定的正则表达式
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)

    # 根据CUDA测试条件，如果没有CUDA，则跳过该测试
    @unittest.skipIf(not TEST_CUDA, "no cuda")
    # 定义测试方法，用于测试完全分片后的深度复制
    def test_fully_shard_deepcopy(self):
        # 创建一个MLP(8)模型
        model = MLP(8)
        # 对模型应用完全分片
        fully_shard(model)
        # 断言抛出断言错误，并匹配指定的错误信息
        with self.assertRaisesRegex(AssertionError, "FSDP does not support deepcopy"):
            copy.deepcopy(model)
# 如果当前模块被直接执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 调用名为 run_tests 的函数，用于执行测试代码
    run_tests()
```