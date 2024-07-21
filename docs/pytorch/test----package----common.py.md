# `.\pytorch\test\package\common.py`

```py
# 导入必要的模块
import os
import sys
from tempfile import NamedTemporaryFile

# 导入 PyTorch 相关的模块
import torch.package.package_exporter
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase

# 创建一个测试用例类，继承自 TestCase 类
class PackageTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化一个临时文件列表
        self._temporary_files = []

    # 创建临时文件方法
    def temp(self):
        # 使用 NamedTemporaryFile 创建临时文件对象 t
        t = NamedTemporaryFile()
        # 获取临时文件名
        name = t.name
        # 如果运行环境是 Windows
        if IS_WINDOWS:
            # 关闭临时文件对象（在 Windows 中无法在文件打开状态下读取文件）
            t.close()
        else:
            # 将临时文件对象添加到临时文件列表中
            self._temporary_files.append(t)
        # 返回临时文件名
        return name

    # 设置测试环境方法
    def setUp(self):
        # 调用父类的 setUp 方法
        super().setUp()
        # 获取当前测试文件所在目录
        self.package_test_dir = os.path.dirname(os.path.realpath(__file__))
        # 备份当前的 sys.path
        self.orig_sys_path = sys.path.copy()
        # 将当前测试目录添加到 sys.path 中，以便可以导入测试中的伪造包
        sys.path.append(self.package_test_dir)
        # 禁用 TorchScript 序列化的门限，便于测试
        torch.package.package_exporter._gate_torchscript_serialization = False

    # 清理测试环境方法
    def tearDown(self):
        # 调用父类的 tearDown 方法
        super().tearDown()
        # 恢复原始的 sys.path
        sys.path = self.orig_sys_path

        # 关闭所有临时文件对象
        for t in self._temporary_files:
            t.close()
        # 清空临时文件列表
        self._temporary_files = []
```