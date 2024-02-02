# `markdown\tests\test_meta.py`

```py

# 导入单元测试模块
import unittest
# 从markdown.__meta__模块中导入_get_version和__version__
from markdown.__meta__ import _get_version, __version__


class TestVersion(unittest.TestCase):

    # 测试_get_version函数
    def test_get_version(self):
        """Test that _get_version formats __version_info__ as required by PEP 440."""

        # 测试_get_version函数返回的版本号格式是否符合PEP 440的要求
        self.assertEqual(_get_version((1, 1, 2, 'dev', 0)), "1.1.2.dev0")
        self.assertEqual(_get_version((1, 1, 2, 'alpha', 1)), "1.1.2a1")
        self.assertEqual(_get_version((1, 2, 0, 'beta', 2)), "1.2b2")
        self.assertEqual(_get_version((1, 2, 0, 'rc', 4)), "1.2rc4")
        self.assertEqual(_get_version((1, 2, 0, 'final', 0)), "1.2")

    # 测试__version__是否有效和是否已标准化
    def test__version__IsValid(self):
        """Test that __version__ is valid and normalized."""

        # 尝试导入packaging.version模块，如果导入失败则跳过测试
        try:
            import packaging.version
        except ImportError:
            self.skipTest('packaging does not appear to be installed')

        # 测试__version__是否有效并且是否已标准化
        self.assertEqual(__version__, str(packaging.version.Version(__version__)))

```