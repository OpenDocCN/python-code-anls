# `D:\src\scipysrc\matplotlib\ci\export_sdist_name.py`

```py
#!/usr/bin/env python3
"""
Determine the name of the sdist and export to GitHub output named SDIST_NAME.

To run:
    $ python3 -m build --sdist
    $ ./ci/determine_sdist_name.py
"""
# 导入必要的模块
import os
from pathlib import Path
import sys

# 获取所有在 dist 目录下的 .tar.gz 文件名
paths = [p.name for p in Path("dist").glob("*.tar.gz")]

# 如果找到的文件数量不为1，则退出程序并打印错误信息
if len(paths) != 1:
    sys.exit(f"Only a single sdist is supported, but found: {paths}")

# 打印找到的唯一的 .tar.gz 文件名
print(paths[0])

# 将文件名写入环境变量 GITHUB_OUTPUT 指定的文件中
with open(os.environ["GITHUB_OUTPUT"], "a") as f:
    f.write(f"SDIST_NAME={paths[0]}\n")
```