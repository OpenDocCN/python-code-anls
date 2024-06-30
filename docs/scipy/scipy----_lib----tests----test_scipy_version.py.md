# `D:\src\scipysrc\scipy\scipy\_lib\tests\test_scipy_version.py`

```
import re  # 导入 re 模块，用于正则表达式操作

import scipy  # 导入 scipy 库，科学计算库
from numpy.testing import assert_  # 导入 numpy.testing 模块中的 assert_

def test_valid_scipy_version():
    # 验证 SciPy 版本号是否有效，不含 .post 后缀或其他无效内容。参考 NumPy 问题 gh-6431 引起的无效版本问题。
    version_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+(|a[0-9]|b[0-9]|rc[0-9])"  # 定义版本号的正则表达式模式，允许 alpha、beta、rc 后缀
    dev_suffix = r"(\.dev0\+.+([0-9a-f]{7}|Unknown))"  # 定义开发版后缀的正则表达式模式
    if scipy.version.release:
        res = re.match(version_pattern, scipy.__version__)  # 如果是正式发布版本，使用 version_pattern 匹配版本号
    else:
        res = re.match(version_pattern + dev_suffix, scipy.__version__)  # 否则，使用 version_pattern + dev_suffix 匹配版本号

    assert_(res is not None, scipy.__version__)  # 断言版本号匹配结果不为空，否则抛出错误信息并显示当前版本号
```