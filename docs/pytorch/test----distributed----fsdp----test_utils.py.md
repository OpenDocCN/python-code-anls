# `.\pytorch\test\distributed\fsdp\test_utils.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的模块
import random
import sys
import unittest
from collections import OrderedDict
from dataclasses import dataclass
from typing import List

# 导入 PyTorch 相关模块
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.utils import _apply_to_tensors, _replace_by_prefix
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
)

# 如果分布式不可用，跳过测试
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果启用了测试开发调试模式，跳过相关测试
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 定义测试类 TestUtils，继承自 unittest 的 TestCase
class TestUtils(TestCase):
    
    # 参数化测试，测试 apply_to_tensors 函数
    @parametrize(
        "devices", [["cpu"], ["cuda"], subtest(["cpu", "cuda"], name="cpu_cuda")]
    )
    def test_apply_to_tensors(self, devices):
        # 如果设备中包含 cuda 且 CUDA 不可用或者没有 GPU，则跳过测试
        if "cuda" in devices and (
            not torch.cuda.is_available() or torch.cuda.device_count() < 1
        ):
            raise unittest.SkipTest("Skipped due to lack of GPU")

        expected = 0

        # 定义函数 get_a_tensor，返回一个随机设备上的随机张量
        def get_a_tensor():
            """Return a random tensor on random device."""
            dev = random.choice(devices)
            shape = random.choice(((1), (2, 3), (4, 5, 6), (7, 8, 9, 10)))
            t = torch.rand(shape).to(dev)
            nonlocal expected
            expected += t.numel()
            return t

        # 数据类 SomeDataClass，包含字符串、浮点数和张量列表
        @dataclass
        class SomeDataClass:
            some_key: str
            some_float: float
            some_tensor: List[torch.Tensor]

        # 创建混合数据
        data = [1, "str"]
        data.append({"key1": get_a_tensor(), "key2": {1: get_a_tensor()}, "key3": 3})
        data.insert(0, {"x", get_a_tensor(), get_a_tensor()})
        data.append(([1], get_a_tensor(), (1), [get_a_tensor()], {1, 2}))
        data.append({"abc": SomeDataClass("some_key", 1.0, [get_a_tensor()])})
        od = OrderedDict()
        od["k"] = "value"
        data.append(od)

        total = 0

        # 定义处理函数 fn，计算总张量元素数，并返回张量本身
        def fn(t):
            nonlocal total
            total += t.numel()
            return t

        # 对数据应用处理函数 fn
        new_data = _apply_to_tensors(fn, data)
        
        # 断言处理后的总张量元素数等于预期
        self.assertEqual(total, expected)
        
        # 检查新数据类型与原始数据类型相匹配
        for i, v in enumerate(data):
            self.assertEqual(type(new_data[i]), type(v))

    # 测试替换函数 replace_by_prefix
    def test_replace_by_prefix(self):
        # 初始状态字典
        state_dict = {
            "layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "layer.b": torch.tensor(3),
        }
        original_state_dict = state_dict.copy()
        
        # 使用 replace_by_prefix 替换指定前缀
        _replace_by_prefix(state_dict, "layer.", "module.layer.")
        
        # 断言替换后的状态字典符合预期
        assert state_dict == {
            "module.layer.a": torch.tensor(1),
            "abc.layer.def": torch.tensor(2),
            "module.layer.b": torch.tensor(3),
        }
        
        # 再次使用 replace_by_prefix 恢复原始状态字典
        _replace_by_prefix(state_dict, "module.layer.", "layer.")
        
        # 断言恢复后的状态字典与初始状态一致
        assert state_dict == original_state_dict
    # 定义一个测试方法，用于验证 RNN 的压缩序列是否正确修改
    def test_packed_sequence(self):
        """Test to ensure RNN packed sequences are modified correctly."""
        # 创建一个 RNN 模型，输入维度为 5，输出维度为 5
        rnn = nn.RNN(5, 5)

        # 生成一个大小为 (5, 1, 5) 的随机张量 x，数据类型为 float
        x = torch.rand((5, 1, 5), dtype=torch.float)
        # 创建一个包含序列长度的张量 seq_length，其值为 [4]，数据类型为 int
        seq_length = torch.tensor([4], dtype=torch.int)

        # 定义一个填充函数 fill_fn，用于填充输入张量 x 的数据
        def fill_fn(x):
            x.fill_(0)

        # 对输入张量 x 进行压缩序列处理
        x = nn.utils.rnn.pack_padded_sequence(x, seq_length)
        # 将压缩序列 x 输入到 RNN 模型中，获取输出 x 和隐藏状态 h
        x, h = rnn(x)
        # 对输出 x 中的张量应用填充函数 fill_fn
        x = _apply_to_tensors(fill_fn, x)
        # 对填充后的压缩序列 x 进行解压缩处理
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        # 断言解压缩后的张量 x 的所有元素之和应为 0
        self.assertEqual(torch.sum(x), 0)
# 调用一个函数来实例化带有参数的测试用例，参数为 TestUtils
instantiate_parametrized_tests(TestUtils)

# 检查当前脚本是否作为主程序执行
if __name__ == "__main__":
    # 运行测试函数
    run_tests()
```