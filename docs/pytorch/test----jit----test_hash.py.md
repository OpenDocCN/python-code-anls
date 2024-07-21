# `.\pytorch\test\jit\test_hash.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块和类型
import os
import sys

from typing import List, Tuple

import torch

# 将 test/ 目录下的辅助文件设为可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

# 如果此脚本被直接运行，则引发运行时错误，建议使用指定方式运行
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类 TestHash，继承自 JitTestCase
class TestHash(JitTestCase):

    # 测试元组的哈希值比较
    def test_hash_tuple(self):
        def fn(t1: Tuple[int, int], t2: Tuple[int, int]) -> bool:
            return hash(t1) == hash(t2)

        # 使用 JitTestCase 提供的 checkScript 方法检查脚本化函数的行为
        self.checkScript(fn, ((1, 2), (1, 2)))
        self.checkScript(fn, ((1, 2), (3, 4)))
        self.checkScript(fn, ((1, 2), (2, 1)))

    # 测试包含不可哈希类型的元组，预期引发运行时错误
    def test_hash_tuple_nested_unhashable_type(self):
        @torch.jit.script
        def fn_unhashable(t1: Tuple[int, List[int]]):
            return hash(t1)

        # 使用 self.assertRaisesRegexWithHighlight 检查是否引发预期的异常
        with self.assertRaisesRegexWithHighlight(RuntimeError, "unhashable", "hash"):
            fn_unhashable((1, [1]))

    # 测试张量的哈希值比较，预期由对象标识决定哈希值
    def test_hash_tensor(self):
        """Tensors should hash by identity"""

        def fn(t1, t2):
            return hash(t1) == hash(t2)

        # 创建张量实例
        tensor1 = torch.tensor(1)
        tensor1_clone = torch.tensor(1)
        tensor2 = torch.tensor(2)

        # 使用 checkScript 方法检查脚本化函数的行为
        self.checkScript(fn, (tensor1, tensor1))
        self.checkScript(fn, (tensor1, tensor1_clone))
        self.checkScript(fn, (tensor1, tensor2))

    # 测试 None 的哈希值比较
    def test_hash_none(self):
        def fn():
            n1 = None
            n2 = None
            return hash(n1) == hash(n2)

        # 使用 checkScript 方法检查脚本化函数的行为
        self.checkScript(fn, ())

    # 测试布尔值的哈希值比较
    def test_hash_bool(self):
        def fn(b1: bool, b2: bool):
            return hash(b1) == hash(b2)

        # 使用 checkScript 方法检查脚本化函数的行为
        self.checkScript(fn, (True, False))
        self.checkScript(fn, (True, True))
        self.checkScript(fn, (False, True))
        self.checkScript(fn, (False, False))

    # 测试浮点数的哈希值比较
    def test_hash_float(self):
        def fn(f1: float, f2: float):
            return hash(f1) == hash(f2)

        # 使用 checkScript 方法检查脚本化函数的行为
        self.checkScript(fn, (1.2345, 1.2345))
        self.checkScript(fn, (1.2345, 6.789))
        self.checkScript(fn, (1.2345, float("inf")))
        self.checkScript(fn, (float("inf"), float("inf")))
        self.checkScript(fn, (1.2345, float("nan")))
        
        # 在 Python 版本低于 3.10 时，两个 NaN 的哈希值不保证相等
        if sys.version_info < (3, 10):
            self.checkScript(fn, (float("nan"), float("nan")))
        
        self.checkScript(fn, (float("nan"), float("inf")))
    # 定义一个测试函数，测试两个整数的哈希值是否相同
    def test_hash_int(self):
        # 定义内部函数fn，接受两个整数参数，比较它们的哈希值是否相等
        def fn(i1: int, i2: int):
            return hash(i1) == hash(i2)

        # 使用测试框架中的checkScript方法测试fn函数的结果
        self.checkScript(fn, (123, 456))
        self.checkScript(fn, (123, 123))
        self.checkScript(fn, (123, -123))
        self.checkScript(fn, (-123, -123))
        self.checkScript(fn, (123, 0))

    # 定义一个测试函数，测试两个字符串的哈希值是否相同
    def test_hash_string(self):
        # 定义内部函数fn，接受两个字符串参数，比较它们的哈希值是否相等
        def fn(s1: str, s2: str):
            return hash(s1) == hash(s2)

        # 使用测试框架中的checkScript方法测试fn函数的结果
        self.checkScript(fn, ("foo", "foo"))
        self.checkScript(fn, ("foo", "bar"))
        self.checkScript(fn, ("foo", ""))

    # 定义一个测试函数，测试两个torch设备对象的哈希值是否相同
    def test_hash_device(self):
        # 定义内部函数fn，接受两个torch设备对象参数，比较它们的哈希值是否相等
        def fn(d1: torch.device, d2: torch.device):
            return hash(d1) == hash(d2)

        # 创建几个torch设备对象
        gpu0 = torch.device("cuda:0")
        gpu1 = torch.device("cuda:1")
        cpu = torch.device("cpu")

        # 使用测试框架中的checkScript方法测试fn函数的结果
        self.checkScript(fn, (gpu0, gpu0))
        self.checkScript(fn, (gpu0, gpu1))
        self.checkScript(fn, (gpu0, cpu))
        self.checkScript(fn, (cpu, cpu))
```