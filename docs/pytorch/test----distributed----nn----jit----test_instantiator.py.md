# `.\pytorch\test\distributed\nn\jit\test_instantiator.py`

```
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# 导入标准库模块
import sys
# 导入处理路径的模块
from pathlib import Path
# 导入类型提示模块
from typing import Tuple

# 导入PyTorch相关模块
import torch
import torch.distributed as dist
# 导入PyTorch的神经网络模块和张量模块
from torch import nn, Tensor

# 如果分布式功能不可用，输出消息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入PyTorch的分布式神经网络JIT实例化器
from torch.distributed.nn.jit import instantiator
# 导入PyTorch的测试工具类和测试用例类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义一个接口，继承自torch.jit.interface，声明了一个forward方法
@torch.jit.interface
class MyModuleInterface:
    def forward(
        self, tensor: Tensor, number: int, word: str = "default"
    ) -> Tuple[Tensor, int, str]:
        pass

# 定义一个空的MyModule类，继承自nn.Module
class MyModule(nn.Module):
    pass

# 返回一个空的MyModule对象
def create_module():
    return MyModule()

# 定义一个测试类TestInstantiator，继承自TestCase
class TestInstantiator(TestCase):
    # 测试从接口中获取参数和返回类型的方法
    def test_get_arg_return_types_from_interface(self):
        (
            args_str,
            arg_types_str,
            return_type_str,
        ) = instantiator.get_arg_return_types_from_interface(MyModuleInterface)
        self.assertEqual(args_str, "tensor, number, word")
        self.assertEqual(arg_types_str, "tensor: Tensor, number: int, word: str")
        self.assertEqual(return_type_str, "Tuple[Tensor, int, str]")

    # 测试实例化脚本化远程模块模板的方法
    def test_instantiate_scripted_remote_module_template(self):
        dir_path = Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)

        # 清理操作
        file_paths = dir_path.glob(f"{instantiator._FILE_PREFIX}*.py")
        for file_path in file_paths:
            file_path.unlink()

        # 检查运行前的状态
        file_paths = dir_path.glob(f"{instantiator._FILE_PREFIX}*.py")
        num_files_before = len(list(file_paths))
        self.assertEqual(num_files_before, 0)

        # 实例化一个脚本化的远程模块模板
        generated_module = instantiator.instantiate_scriptable_remote_module_template(
            MyModuleInterface
        )
        # 断言生成的模块对象具有"_remote_forward"和"_generated_methods"属性
        self.assertTrue(hasattr(generated_module, "_remote_forward"))
        self.assertTrue(hasattr(generated_module, "_generated_methods"))

        # 检查运行后的状态
        file_paths = dir_path.glob(f"{instantiator._FILE_PREFIX}*.py")
        num_files_after = len(list(file_paths))
        self.assertEqual(num_files_after, 1)
    def test_instantiate_non_scripted_remote_module_template(self):
        dir_path = Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)

        # Cleanup.
        # 获取所有以指定前缀开头的文件路径并删除
        file_paths = dir_path.glob(f"{instantiator._FILE_PREFIX}*.py")
        for file_path in file_paths:
            file_path.unlink()

        # Check before run.
        # 再次获取符合条件的文件路径，并计算数量
        file_paths = dir_path.glob(f"{instantiator._FILE_PREFIX}*.py")
        num_files_before = len(list(file_paths))
        self.assertEqual(num_files_before, 0)

        # 生成非脚本化远程模块模板
        generated_module = (
            instantiator.instantiate_non_scriptable_remote_module_template()
        )
        # 断言生成的模块具有 "_remote_forward" 属性
        self.assertTrue(hasattr(generated_module, "_remote_forward"))
        # 断言生成的模块具有 "_generated_methods" 属性
        self.assertTrue(hasattr(generated_module, "_generated_methods"))

        # Check after run.
        # 再次获取符合条件的文件路径，并计算数量
        file_paths = dir_path.glob(f"{instantiator._FILE_PREFIX}*.py")
        num_files_after = len(list(file_paths))
        self.assertEqual(num_files_after, 1)
# 如果当前脚本作为主程序运行，则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```