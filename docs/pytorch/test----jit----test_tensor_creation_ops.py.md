# `.\pytorch\test\jit\test_tensor_creation_ops.py`

```py
# Owner(s): ["oncall: jit"]

# 引入必要的模块和库
import os
import sys

import torch  # 导入 PyTorch 库

# 将 test/ 目录下的 helper 文件添加到 Python 搜索路径中，使其可导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase  # 导入测试相关的 JitTestCase 类

# 如果当前文件被直接运行，抛出运行时错误，提示正确的运行方式
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类 TestTensorCreationOps，继承自 JitTestCase 类
class TestTensorCreationOps(JitTestCase):
    """
    A suite of tests for ops that create tensors.
    一组测试操作创建张量的测试套件。
    """

    # 定义测试函数 test_randperm_default_dtype
    def test_randperm_default_dtype(self):
        # 定义内部函数 randperm，接受一个整数参数 x
        def randperm(x: int):
            # 调用 torch.randperm 函数生成一个大小为 x 的随机排列张量 perm
            perm = torch.randperm(x)
            # 进行断言，检查 perm 的数据类型是否为 torch.int64
            # 因为 TorchScript 返回的数据类型是整数，而不是即时模式下的 torch.dtype，所以需要进行比较
            assert perm.dtype == torch.int64

        # 调用 JitTestCase 类的方法，验证 randperm 函数在 TorchScript 中的行为
        self.checkScript(randperm, (3,))

    # 定义测试函数 test_randperm_specifed_dtype
    def test_randperm_specifed_dtype(self):
        # 定义内部函数 randperm，接受一个整数参数 x
        def randperm(x: int):
            # 调用 torch.randperm 函数生成一个大小为 x 的随机排列张量 perm，
            # 并指定数据类型为 torch.float
            perm = torch.randperm(x, dtype=torch.float)
            # 进行断言，检查 perm 的数据类型是否为 torch.float
            # 因为 TorchScript 返回的数据类型是整数，而不是即时模式下的 torch.dtype，所以需要进行比较
            assert perm.dtype == torch.float

        # 调用 JitTestCase 类的方法，验证 randperm 函数在 TorchScript 中的行为
        self.checkScript(randperm, (3,))

    # 定义测试函数 test_triu_indices_default_dtype
    def test_triu_indices_default_dtype(self):
        # 定义内部函数 triu_indices，接受两个整数参数 rows 和 cols
        def triu_indices(rows: int, cols: int):
            # 调用 torch.triu_indices 函数生成一个大小为 (rows, cols) 的上三角索引张量 indices
            indices = torch.triu_indices(rows, cols)
            # 进行断言，检查 indices 的数据类型是否为 torch.int64
            # 因为 TorchScript 返回的数据类型是整数，而不是即时模式下的 torch.dtype，所以需要进行比较
            assert indices.dtype == torch.int64

        # 调用 JitTestCase 类的方法，验证 triu_indices 函数在 TorchScript 中的行为
        self.checkScript(triu_indices, (3, 3))

    # 定义测试函数 test_triu_indices_specified_dtype
    def test_triu_indices_specified_dtype(self):
        # 定义内部函数 triu_indices，接受两个整数参数 rows 和 cols
        def triu_indices(rows: int, cols: int):
            # 调用 torch.triu_indices 函数生成一个大小为 (rows, cols) 的上三角索引张量 indices，
            # 并指定数据类型为 torch.int32
            indices = torch.triu_indices(rows, cols, dtype=torch.int32)
            # 进行断言，检查 indices 的数据类型是否为 torch.int32
            # 因为 TorchScript 返回的数据类型是整数，而不是即时模式下的 torch.dtype，所以需要进行比较
            assert indices.dtype == torch.int32

        # 调用 JitTestCase 类的方法，验证 triu_indices 函数在 TorchScript 中的行为
        self.checkScript(triu_indices, (3, 3))

    # 定义测试函数 test_tril_indices_default_dtype
    def test_tril_indices_default_dtype(self):
        # 定义内部函数 tril_indices，接受两个整数参数 rows 和 cols
        def tril_indices(rows: int, cols: int):
            # 调用 torch.tril_indices 函数生成一个大小为 (rows, cols) 的下三角索引张量 indices
            indices = torch.tril_indices(rows, cols)
            # 进行断言，检查 indices 的数据类型是否为 torch.int64
            # 因为 TorchScript 返回的数据类型是整数，而不是即时模式下的 torch.dtype，所以需要进行比较
            assert indices.dtype == torch.int64

        # 调用 JitTestCase 类的方法，验证 tril_indices 函数在 TorchScript 中的行为
        self.checkScript(tril_indices, (3, 3))

    # 定义测试函数 test_tril_indices_specified_dtype
    def test_tril_indices_specified_dtype(self):
        # 定义内部函数 tril_indices，接受两个整数参数 rows 和 cols
        def tril_indices(rows: int, cols: int):
            # 调用 torch.tril_indices 函数生成一个大小为 (rows, cols) 的下三角索引张量 indices，
            # 并指定数据类型为 torch.int32
            indices = torch.tril_indices(rows, cols, dtype=torch.int32)
            # 进行断言，检查 indices 的数据类型是否为 torch.int32
            # 因为 TorchScript 返回的数据类型是整数，而不是即时模式下的 torch.dtype，所以需要进行比较
            assert indices.dtype == torch.int32

        # 调用 JitTestCase 类的方法，验证 tril_indices 函数在 TorchScript 中的行为
        self.checkScript(tril_indices, (3, 3))
```