# `.\pytorch\test\test_license.py`

```py
# Owner(s): ["module: unknown"]

# 导入所需的模块和库
import glob  # 用于文件路径的模式匹配
import io  # 提供了对字节流的支持
import os  # 提供了与操作系统相关的功能
import unittest  # 提供了单元测试框架

import torch  # PyTorch深度学习库
from torch.testing._internal.common_utils import run_tests, TestCase  # 引入测试相关的工具类和函数

# 尝试导入第三方模块，用于创建打包文件
try:
    from third_party.build_bundled import create_bundled
except ImportError:
    create_bundled = None

# 指定许可证文件和起始文本
license_file = "third_party/LICENSES_BUNDLED.txt"
starting_txt = "The PyTorch repository and source distributions bundle"
# 获取安装位置的site-packages路径
site_packages = os.path.dirname(os.path.dirname(torch.__file__))
# 查找所有以torch-*dist-info结尾的目录
distinfo = glob.glob(os.path.join(site_packages, "torch-*dist-info"))


class TestLicense(TestCase):
    # 如果未能导入create_bundled模块，则跳过测试
    @unittest.skipIf(not create_bundled, "can only be run in a source tree")
    def test_license_for_wheel(self):
        # 创建当前状态的文本流对象
        current = io.StringIO()
        # 使用create_bundled函数创建打包文件
        create_bundled("third_party", current)
        # 打开许可证文件并读取内容到src_tree
        with open(license_file) as fid:
            src_tree = fid.read()
        # 如果许可证文件内容与当前状态不一致，则抛出断言错误
        if not src_tree == current.getvalue():
            raise AssertionError(
                f'the contents of "{license_file}" do not '
                "match the current state of the third_party files. Use "
                '"python third_party/build_bundled.py" to regenerate it'
            )

    # 如果site-packages中没有torch-*dist-info目录，则跳过测试
    @unittest.skipIf(len(distinfo) == 0, "no installation in site-package to test")
    def test_distinfo_license(self):
        """If run when pytorch is installed via a wheel, the license will be in
        site-package/torch-*dist-info/LICENSE. Make sure it contains the third
        party bundle of licenses"""

        # 如果找到多于一个torch-*dist-info目录，则抛出断言错误
        if len(distinfo) > 1:
            raise AssertionError(
                'Found too many "torch-*dist-info" directories '
                f'in "{site_packages}, expected only one'
            )
        # 打开第一个torch-*dist-info目录下的LICENSE文件，并读取其内容到txt
        with open(os.path.join(os.path.join(distinfo[0], "LICENSE"))) as fid:
            txt = fid.read()
            # 断言起始文本在LICENSE文件中
            self.assertTrue(starting_txt in txt)


if __name__ == "__main__":
    run_tests()  # 运行单元测试
```