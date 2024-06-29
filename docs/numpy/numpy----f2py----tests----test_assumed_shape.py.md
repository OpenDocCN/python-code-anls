# `.\numpy\numpy\f2py\tests\test_assumed_shape.py`

```py
# 导入必要的库和模块
import os  # 导入操作系统相关的功能
import pytest  # 导入 pytest 测试框架
import tempfile  # 导入临时文件相关的功能

# 从当前包中导入 util 模块
from . import util


# 定义一个测试类 TestAssumedShapeSumExample，继承自 util.F2PyTest
class TestAssumedShapeSumExample(util.F2PyTest):
    # 定义一个列表，包含多个源文件的路径
    sources = [
        util.getpath("tests", "src", "assumed_shape", "foo_free.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_use.f90"),
        util.getpath("tests", "src", "assumed_shape", "precision.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_mod.f90"),
        util.getpath("tests", "src", "assumed_shape", ".f2py_f2cmap"),
    ]

    # 标记此测试方法为慢速测试
    @pytest.mark.slow
    def test_all(self):
        # 调用 self.module 对象的 fsum 方法，传入参数 [1, 2]
        r = self.module.fsum([1, 2])
        # 断言返回结果 r 等于 3
        assert r == 3
        # 调用 self.module 对象的 sum 方法，传入参数 [1, 2]
        r = self.module.sum([1, 2])
        # 断言返回结果 r 等于 3
        assert r == 3
        # 调用 self.module 对象的 sum_with_use 方法，传入参数 [1, 2]
        r = self.module.sum_with_use([1, 2])
        # 断言返回结果 r 等于 3
        assert r == 3

        # 调用 self.module.mod 对象的 sum 方法，传入参数 [1, 2]
        r = self.module.mod.sum([1, 2])
        # 断言返回结果 r 等于 3
        assert r == 3
        # 调用 self.module.mod 对象的 fsum 方法，传入参数 [1, 2]
        r = self.module.mod.fsum([1, 2])
        # 断言返回结果 r 等于 3
        assert r == 3


# 定义一个测试类 TestF2cmapOption，继承自 TestAssumedShapeSumExample
class TestF2cmapOption(TestAssumedShapeSumExample):
    # 设置每个测试方法的初始化方法
    def setup_method(self):
        # 创建 self.sources 列表的副本，并移除最后一个元素，保存到 f2cmap_src 变量中
        self.sources = list(self.sources)
        f2cmap_src = self.sources.pop(-1)

        # 创建一个临时命名文件，不会自动删除
        self.f2cmap_file = tempfile.NamedTemporaryFile(delete=False)
        # 以二进制读取 f2cmap_src 文件内容，并写入临时文件中
        with open(f2cmap_src, "rb") as f:
            self.f2cmap_file.write(f.read())
        # 关闭临时文件
        self.f2cmap_file.close()

        # 将临时文件名添加到 self.sources 列表末尾
        self.sources.append(self.f2cmap_file.name)
        # 设置选项参数列表，包含 "--f2cmap" 和临时文件名
        self.options = ["--f2cmap", self.f2cmap_file.name]

        # 调用父类的初始化方法
        super().setup_method()

    # 设置每个测试方法的清理方法
    def teardown_method(self):
        # 删除临时文件
        os.unlink(self.f2cmap_file.name)
```