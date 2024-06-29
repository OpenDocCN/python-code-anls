# `D:\src\scipysrc\matplotlib\ci\check_version_number.py`

```
#!/usr/bin/env python3
"""
Check that the version number of the installed Matplotlib does not start with 0

To run:
    $ python3 -m build .
    $ pip install dist/matplotlib*.tar.gz for sdist
    $ pip install dist/matplotlib*.whl for wheel
    $ ./ci/check_version_number.py
"""
# 导入 sys 模块，用于处理系统相关的功能
import sys

# 导入 matplotlib 模块
import matplotlib

# 打印当前安装的 Matplotlib 版本号
print(f"Version {matplotlib.__version__} installed")

# 检查 Matplotlib 版本号的第一个字符是否为 "0"
if matplotlib.__version__[0] == "0":
    # 如果版本号以 "0" 开头，输出错误信息并退出程序
    sys.exit("Version incorrectly starts with 0")
```