# `.\pytorch\test\jit\test_module_apis.py`

```
# Owner(s): ["oncall: jit"]

# 导入标准库和第三方库
import os
import sys
from typing import Any, Dict, List

# 导入 PyTorch 库及测试用的 JIT 相关工具
import torch
from torch.testing._internal.jit_utils import JitTestCase

# 将测试文件夹路径添加到系统路径，使得其中的文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 如果当前脚本被直接执行，则抛出运行时错误，建议通过指定的方式运行
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类，继承自 JitTestCase，用于测试模块的 API
class TestModuleAPIs(JitTestCase):
    
    # 测试默认状态字典方法是否自动可用
    def test_default_state_dict_methods(self):
        """Tests that default state dict methods are automatically available"""

        # 定义一个继承自 torch.nn.Module 的子类，用于测试
        class DefaultStateDictModule(torch.nn.Module):
            
            # 初始化函数，创建网络的结构
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)

            # 前向传播函数，定义网络的计算流程
            def forward(self, x):
                x = self.conv(x)
                x = self.fc(x)
                return x

        # 使用 JIT 编译器对 DefaultStateDictModule 进行脚本化编译
        m1 = torch.jit.script(DefaultStateDictModule())
        m2 = torch.jit.script(DefaultStateDictModule())
        
        # 获取模型 m1 的状态字典
        state_dict = m1.state_dict()
        
        # 将状态字典加载到模型 m2 中，测试状态字典加载功能
        m2.load_state_dict(state_dict)
    # 定义一个测试函数，用于测试自定义状态字典方法是否有效
    def test_customized_state_dict_methods(self):
        """Tests that customized state dict methods are in effect"""

        # 定义一个自定义的 PyTorch 模块 CustomStateDictModule
        class CustomStateDictModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 初始化模块包含的神经网络层
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)
                # 添加两个布尔类型的标志位，用于记录自定义状态字典方法是否被调用
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                # 定义前向传播函数，其中包括卷积和全连接层
                x = self.conv(x)
                x = self.fc(x)
                return x

            @torch.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool
            ):
                # 自定义保存状态字典方法，设置保存标志位为 True
                self.customized_save_state_dict_called = True
                # 返回一个包含虚拟键值对的字典，用于测试目的
                return {"dummy": torch.ones(1)}

            @torch.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, torch.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                # 自定义加载状态字典方法，设置加载标志位为 True
                self.customized_load_state_dict_called = True
                # 不执行任何操作，仅用于测试

        # 使用 torch.jit.script 方法将 CustomStateDictModule 脚本化
        m1 = torch.jit.script(CustomStateDictModule())
        # 断言初始时自定义保存状态字典方法未被调用
        self.assertFalse(m1.customized_save_state_dict_called)
        # 获取模块的状态字典
        state_dict = m1.state_dict()
        # 断言自定义保存状态字典方法已被调用
        self.assertTrue(m1.customized_save_state_dict_called)

        # 创建另一个脚本化的 CustomStateDictModule 实例 m2
        m2 = torch.jit.script(CustomStateDictModule())
        # 断言初始时自定义加载状态字典方法未被调用
        self.assertFalse(m2.customized_load_state_dict_called)
        # 加载之前保存的状态字典到 m2 中
        m2.load_state_dict(state_dict)
        # 断言自定义加载状态字典方法已被调用
        self.assertTrue(m2.customized_load_state_dict_called)
    def test_submodule_customized_state_dict_methods(self):
        """Tests that customized state dict methods on submodules are in effect"""
        
        class CustomStateDictModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义一个卷积层和一个全连接层作为模块的成员变量
                self.conv = torch.nn.Conv2d(6, 16, 5)
                self.fc = torch.nn.Linear(16 * 5 * 5, 120)
                # 初始化自定义的状态字典方法的调用状态为False
                self.customized_save_state_dict_called: bool = False
                self.customized_load_state_dict_called: bool = False

            def forward(self, x):
                # 模块的前向传播方法，包含卷积层和全连接层的操作
                x = self.conv(x)
                x = self.fc(x)
                return x

            @torch.jit.export
            def _save_to_state_dict(
                self, destination: Dict[str, torch.Tensor], prefix: str, keep_vars: bool
            ):
                # 自定义的保存状态字典方法，在保存时将状态设置为已调用
                self.customized_save_state_dict_called = True
                return {"dummy": torch.ones(1)}

            @torch.jit.export
            def _load_from_state_dict(
                self,
                state_dict: Dict[str, torch.Tensor],
                prefix: str,
                local_metadata: Any,
                strict: bool,
                missing_keys: List[str],
                unexpected_keys: List[str],
                error_msgs: List[str],
            ):
                # 自定义的加载状态字典方法，在加载时将状态设置为已调用
                self.customized_load_state_dict_called = True
                return

        class ParentModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 创建一个自定义状态字典模块的实例作为其子模块
                self.sub = CustomStateDictModule()

            def forward(self, x):
                # 前向传播方法，调用子模块的前向传播方法
                return self.sub(x)

        # 使用 torch.jit.script 方法将父模块进行脚本化
        m1 = torch.jit.script(ParentModule())
        # 断言子模块的保存状态字典方法未被调用
        self.assertFalse(m1.sub.customized_save_state_dict_called)
        # 获取模块的状态字典
        state_dict = m1.state_dict()
        # 断言子模块的保存状态字典方法已被调用
        self.assertTrue(m1.sub.customized_save_state_dict_called)

        # 使用 torch.jit.script 方法将另一个父模块进行脚本化
        m2 = torch.jit.script(ParentModule())
        # 断言子模块的加载状态字典方法未被调用
        self.assertFalse(m2.sub.customized_load_state_dict_called)
        # 加载之前保存的状态字典到模型 m2
        m2.load_state_dict(state_dict)
        # 断言子模块的加载状态字典方法已被调用
        self.assertTrue(m2.sub.customized_load_state_dict_called)
```