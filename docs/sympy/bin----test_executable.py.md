# `D:\src\scipysrc\sympy\bin\test_executable.py`

```
#!/usr/bin`
#!/usr/bin/env python
"""
Test that only executable files have an executable bit set
"""
# 引入 print_function，确保在 Python 2 中也能使用 print 函数
from __future__ import print_function

# 引入操作系统相关模块
import os

# 导入路径修正函数
from get_sympy import path_hack

# 获取基础目录路径
base_dir = path_hack()

# 定义测试可执行文件的函数
def test_executable(path):
    # 如果路径不是目录
    if not os.path.isdir(path):
        # 检查文件是否具有执行权限
        if os.access(path, os.X_OK):
            # 打开文件进行读取
            with open(path, 'r') as f:
                # 检查文件的前两个字符是否为 #!
                if f.readline()[:2] != "#!":
                    # 如果不是以 #! 开头，则抛出异常
                    exn_msg = "File at " + path + " either should not be executable or should have a shebang line"
                    raise OSError(exn_msg)
    else:
        # 如果路径是目录，则递归遍历目录中的文件
        for file in os.listdir(path):
            # 跳过特定目录
            if file in ('.git', 'venv_main'):
                continue
            # 递归调用自身，测试子文件或子目录
            test_executable(os.path.join(path, file))

# 在基础目录上执行测试
test_executable(base_dir)
```