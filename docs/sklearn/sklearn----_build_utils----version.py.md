# `D:\src\scipysrc\scikit-learn\sklearn\_build_utils\version.py`

```
#!/usr/bin/env python
"""Extract version number from __init__.py"""

# 导入操作系统相关的功能
import os

# 构造 sklearn_init 变量，指向 __init__.py 文件的路径
sklearn_init = os.path.join(os.path.dirname(__file__), "../__init__.py")

# 打开 __init__.py 文件，读取所有行并存储在 data 列表中
data = open(sklearn_init).readlines()

# 从 data 列表中找到以 "__version__" 开头的行，赋值给 version_line
version_line = next(line for line in data if line.startswith("__version__"))

# 去除 version_line 行两端的空格，并按 " = " 分割，取第二部分，并去除引号，得到版本号字符串
version = version_line.strip().split(" = ")[1].replace('"', "").replace("'", "")

# 输出版本号
print(version)
```