# `.\numpy\numpy\f2py\tests\test_common.py`

```py
# 导入 pytest 模块，用于测试
import pytest
# 导入 numpy 模块并重命名为 np，用于数值计算和数组操作
import numpy as np
# 从当前包中导入 util 模块
from . import util

# 用 pytest 的标记将该类标记为慢速测试
@pytest.mark.slow
# 测试类 TestCommonBlock，继承自 util.F2PyTest 类
class TestCommonBlock(util.F2PyTest):
    # 指定源文件列表
    sources = [util.getpath("tests", "src", "common", "block.f")]

    # 定义测试方法 test_common_block
    def test_common_block(self):
        # 调用 self.module 的 initcb 方法进行初始化
        self.module.initcb()
        # 断言 self.module.block.long_bn 等于一个浮点数数组，数值为 1.0，数据类型为 np.float64
        assert self.module.block.long_bn == np.array(1.0, dtype=np.float64)
        # 断言 self.module.block.string_bn 等于一个字符串数组，内容为 "2"，数据类型为 '|S1'
        assert self.module.block.string_bn == np.array("2", dtype="|S1")
        # 断言 self.module.block.ok 等于一个整数数组，数值为 3，数据类型为 np.int32
        assert self.module.block.ok == np.array(3, dtype=np.int32)


# 测试类 TestCommonWithUse，继承自 util.F2PyTest 类
class TestCommonWithUse(util.F2PyTest):
    # 指定源文件列表
    sources = [util.getpath("tests", "src", "common", "gh19161.f90")]

    # 定义测试方法 test_common_gh19161
    def test_common_gh19161(self):
        # 断言 self.module.data.x 等于 0
        assert self.module.data.x == 0
```