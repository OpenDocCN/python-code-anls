# `.\pytorch\test\distributed\checkpoint\test_nested_dict.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入需要的模块和函数
import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义测试类 TestFlattening，继承自 TestCase 类
class TestFlattening(TestCase):
    
    # 定义测试函数 test_flattening_round_trip，无返回值
    def test_flattening_round_trip(self) -> None:
        # 定义测试用的状态字典 state_dict
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {"1": 2, "2": 3},
            "key3": torch.tensor([1]),
            "key4": [[torch.tensor(2), "x"], [1, 2, 3], {"key6": [44]}],
        }

        # 调用 flatten_state_dict 函数，获取扁平化后的字典和映射关系
        flatten_dict, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
            {
                'key0': 1,
                'key1': [1, 2],
                'key2': {'1': 2, '2': 3},
                'key3': tensor([1]),
                'key4.0.0': tensor(2),
                'key4.0.1': 'x',
                'key4.1': [1, 2, 3],
                'key4.2': {'key6': [44]}
            }
        """
        
        # 调用 unflatten_state_dict 函数，使用扁平化字典和映射关系恢复状态字典
        restored = unflatten_state_dict(flatten_dict, mapping)

        # 使用 self.assertEqual 进行断言，验证状态字典恢复后与原始状态字典是否相等
        self.assertEqual(state_dict, restored)

    # 定义测试函数 test_mapping，无返回值
    def test_mapping(self) -> None:
        # 定义测试用的状态字典 state_dict
        state_dict = {
            "k0": [1],
            "k2": [torch.tensor([1]), 99, [{"k3": torch.tensor(1)}]],
            "k3": ["x", 99, [{"k3": "y"}]],
        }

        # 调用 flatten_state_dict 函数，获取扁平化后的字典和映射关系
        flatten_dict, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
        {'k0': [1], 'k2.0': tensor([1]), 'k2.1': 99, 'k2.2.0.k3': tensor(1), 'k3': ['x', 99, [{'k3': 'y'}]]}
        mapping:
        {'k0': ('k0',), 'k2.0': ('k2', 0), 'k2.1': ('k2', 1), 'k2.2.0.k3': ('k2', 2, 0, 'k3'), 'k3': ('k3',)}
        """

        # 使用 self.assertEqual 进行断言，验证映射关系的正确性
        self.assertEqual(("k0",), mapping["k0"])
        self.assertEqual(("k2", 0), mapping["k2.0"])
        self.assertEqual(("k2", 1), mapping["k2.1"])
        self.assertEqual(("k2", 2, 0, "k3"), mapping["k2.2.0.k3"])
        self.assertEqual(("k3", 0), mapping["k3.0"])
        self.assertEqual(("k3", 1), mapping["k3.1"])
        self.assertEqual(("k3", 2, 0, "k3"), mapping["k3.2.0.k3"])

# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```