# `.\DB-GPT-src\dbgpt\core\awel\util\tests\test_http_util.py`

```py
# 导入 pytest 模块，用于编写和运行测试用例
import pytest

# 从上级目录的 http_util 模块中导入 join_paths 函数
from ..http_util import join_paths

# 使用 pytest.mark.parametrize 装饰器定义参数化测试
@pytest.mark.parametrize(
    "paths, expected",
    [
        # 测试基本路径连接
        (["/base/path/", "sub/path"], "/base/path/sub/path"),
        # 测试路径末尾和开头都有 '/'
        (["/base/path/", "/sub/path/"], "/base/path/sub/path"),
        # 测试空字符串和普通路径连接
        (["", "/sub/path"], "/sub/path"),
        # 测试多个斜杠的情况
        (["/base///path//", "///sub/path"], "/base/path/sub/path"),
        # 测试只有一个非空路径
        (["/only/path/"], "/only/path"),
        # 测试所有路径都为空
        (["", "", ""], "/"),
        # 测试带有空格的路径
        ([" /base/path/ ", " sub/path "], "/base/path/sub/path"),
        # 测试带有 '.' 和 '..' 的路径
        (["/base/path/..", "sub/./path"], "/base/path/../sub/./path"),
        # 测试数字和特殊字符
        (["/123/", "/$pecial/char&/"], "/123/$pecial/char&"),
    ],
)
# 定义测试函数 test_join_paths，参数为 paths 和 expected
def test_join_paths(paths, expected):
    # 断言 join_paths 函数的返回值与期望值 expected 相等
    assert join_paths(*paths) == expected
```