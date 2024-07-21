# `.\pytorch\test\distributed\checkpoint\test_traverse.py`

```
# Owner(s): ["oncall: distributed"]

# 引入有序字典模块
from collections import OrderedDict
# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入 PyTorch 库
import torch

# 引入 Torch 分布式检查点模块的遍历函数
import torch.distributed.checkpoint._traverse as _traverse
# 引入测试工具函数和测试用例基类
from torch.testing._internal.common_utils import run_tests, TestCase

# 如果正在进行类型检查，则引入状态字典类型
if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


# TODO: add comments for TestTraverse
# 定义 TestTraverse 类，继承自 TestCase
class TestTraverse(TestCase):
    
    # 定义 test_traverse_shallow 方法，测试浅层结构的状态字典遍历
    def test_traverse_shallow(self) -> None:
        # 定义状态字典，包含不同类型的数据结构
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {1: 2, 2: 3},
            "key3": torch.tensor([1]),
        }

        # 初始化数据字典
        data = {}

        # 定义收集数据的回调函数
        def collect_data(path, value):
            nonlocal data
            data[path] = value

        # 调用 _traverse.traverse_state_dict 函数，遍历状态字典并收集数据
        _traverse.traverse_state_dict(state_dict, collect_data)

        # 断言收集的数据
        self.assertIn(("key0",), data)
        self.assertEqual(data[("key0",)], 1)

        self.assertIn(("key1",), data)
        self.assertEqual(data[("key1",)], [1, 2])

        self.assertIn(("key2", "1"), data)
        self.assertEqual(data[("key2", "1")], 2)
        self.assertIn(("key2", "2"), data)
        self.assertEqual(data[("key2", "2")], 3)

        self.assertIn(("key3",), data)
        self.assertEqual(data[("key3",)], torch.tensor([1]))

    # 定义 test_traverse_nested_list 方法，测试嵌套列表的状态字典遍历
    def test_traverse_nested_list(self) -> None:
        # 定义状态字典，包含嵌套列表结构
        state_dict = {
            "key1": [
                torch.tensor([1]),
                [33, torch.tensor([2]), [44, 55]],
                [66, 77],
            ],
        }

        # 初始化数据字典
        data = {}

        # 定义收集数据的回调函数
        def collect_data(path, value):
            nonlocal data
            data[path] = value

        # 调用 _traverse.traverse_state_dict 函数，遍历状态字典并收集数据
        _traverse.traverse_state_dict(state_dict, collect_data)

        # 断言收集的数据
        self.assertNotIn(("key1"), data)

        self.assertIn(("key1", 0), data)
        self.assertEqual(data[("key1", 0)], torch.tensor([1]))

        self.assertIn(("key1", 1, 0), data)
        self.assertEqual(data[("key1", 1, 0)], 33)

        self.assertIn(("key1", 1, 1), data)
        self.assertEqual(data[("key1", 1, 1)], torch.tensor([2]))

        self.assertIn(("key1", 1, 2), data)
        self.assertEqual(data[("key1", 1, 2)], [44, 55])
        self.assertNotIn(("key1", 1, 2, 0), data)

        self.assertIn(("key1", 2), data)
        self.assertEqual(data[("key1", 2)], [66, 77])

    # 定义 test_traverse_nested_dict 方法，测试嵌套字典的状态字典遍历
    def test_traverse_nested_dict(self) -> None:
        # 定义状态字典，包含嵌套字典结构
        state_dict = {
            "key0": {"key1": 99, "key2": torch.tensor([1])},
        }

        # 初始化数据字典
        data = {}

        # 定义收集数据的回调函数
        def collect_data(path, value):
            nonlocal data
            data[path] = value

        # 调用 _traverse.traverse_state_dict 函数，遍历状态字典并收集数据
        _traverse.traverse_state_dict(state_dict, collect_data)

        # 断言收集的数据
        self.assertNotIn(("key0",), data)

        self.assertIn(("key0", "key1"), data)
        self.assertEqual(data[("key0", "key1")], 99)

        self.assertIn(("key0", "key2"), data)
        self.assertEqual(data[("key0", "key2")], torch.tensor([1]))
    def test_traverse_doesnt_ignore_intermediate_collections(self) -> None:
        # 初始化状态字典，包含嵌套结构
        state_dict: STATE_DICT_TYPE = {"key0": [{"key1": {"key2": torch.tensor([1])}}]}
    
        # 初始化空数据字典
        data = {}
    
        # 定义收集数据的函数
        def collect_data(path, value):
            nonlocal data
            # 将路径和对应数值存入数据字典
            data[path] = value
    
        # 使用_traverse模块的函数遍历状态字典并收集数据
        _traverse.traverse_state_dict(state_dict, collect_data)
    
        # 断言特定路径数据已经收集
        self.assertIn(("key0", 0, "key1", "key2"), data)
        self.assertEqual(
            data[("key0", 0, "key1", "key2")],
            torch.tensor([1]),
        )
    
    def test_traverse_with_ordered_dict(self) -> None:
        # 使用OrderedDict初始化状态字典
        state_dict = OrderedDict(
            {
                "key0": [
                    99,
                    torch.tensor([3]),
                ]
            }
        )
    
        # 初始化空数据字典
        data = {}
    
        # 定义收集数据的函数
        def collect_data(path, value):
            nonlocal data
            # 将路径和对应数值存入数据字典
            data[path] = value
    
        # 使用_traverse模块的函数遍历状态字典并收集数据
        _traverse.traverse_state_dict(state_dict, collect_data)
    
        # 断言特定路径数据已经收集
        self.assertIn(("key0", 0), data)
        self.assertEqual(data[("key0", 0)], 99)
    
        self.assertIn(("key0", 1), data)
        self.assertEqual(data[("key0", 1)], torch.tensor([3]))
    
    def test_set_element(self) -> None:
        # 初始化空状态字典
        state_dict: STATE_DICT_TYPE = {}
    
        # 使用_traverse模块的函数设置字典中元素
        _traverse.set_element(state_dict, ("k",), 10)
        self.assertEqual(state_dict["k"], 10)
    
        _traverse.set_element(state_dict, ("k1", 2), 1)
        self.assertEqual(state_dict["k1"], [None, None, 1])
    
        _traverse.set_element(state_dict, ("k1", 1), 99)
        self.assertEqual(state_dict["k1"], [None, 99, 1])
    
        _traverse.set_element(state_dict, ("k1", 3), 88)
        self.assertEqual(state_dict["k1"], [None, 99, 1, 88])
    
        _traverse.set_element(state_dict, ("k2", "k3"), 3)
        self.assertEqual(state_dict["k2"], {"k3": 3})
    
        _traverse.set_element(state_dict, ("k2", "k4", 0, 0), 99)
        self.assertEqual(state_dict["k2"]["k4"][0], [99])
    
    def test_get_element(self) -> None:
        # 初始化包含多种数据类型的状态字典
        state_dict = {"a": [0, 1], "b": [2, {"c": "d"}]}
        # 断言获取指定路径的元素值
        self.assertEqual(_traverse.get_element(state_dict, ("a",)), [0, 1])
        self.assertEqual(_traverse.get_element(state_dict, ("b", 0)), 2)
        self.assertEqual(_traverse.get_element(state_dict, ("b", 1, "c")), "d")
    
        # 断言访问不存在路径时返回None
        self.assertIsNone(_traverse.get_element(state_dict, ("c",)))
        self.assertIsNone(_traverse.get_element(state_dict, ("a", 33)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 88)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 0, 2)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 1, 2)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 1, "d")))
# 如果当前脚本作为主程序运行，则调用 run_tests() 函数执行测试
if __name__ == "__main__":
    run_tests()
```