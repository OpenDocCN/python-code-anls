# `D:\src\scipysrc\matplotlib\tools\generate_matplotlibrc.py`

```py
#!/usr/bin/env python3
"""
Generate matplotlirc for installs.

If packagers want to change the default backend, insert a `#backend: ...` line.
Otherwise, use the default `##backend: Agg` which has no effect even after
decommenting, which allows _auto_backend_sentinel to be filled in at import
time.
"""

import sys  # 导入sys模块，用于访问命令行参数和系统相关功能
from pathlib import Path  # 导入Path类，用于处理文件路径

# 检查命令行参数的数量，必须为4个，否则退出并显示使用说明
if len(sys.argv) != 4:
    raise SystemExit('usage: {sys.argv[0]} <input> <output> <backend>')

input = Path(sys.argv[1])  # 第一个参数作为输入文件路径
output = Path(sys.argv[2])  # 第二个参数作为输出文件路径
backend = sys.argv[3]  # 第三个参数作为后端设置

# 读取输入文件的内容，并按行分割为列表
template_lines = input.read_text(encoding="utf-8").splitlines(True)

# 找到包含"#backend:"的行的索引，假定只有一行含有此信息，否则会引发异常
backend_line_idx, = [
    idx for idx, line in enumerate(template_lines)
    if "#backend:" in line]

# 替换找到的"#backend:"行，根据命令行参数中的后端值设置新的后端设置行
template_lines[backend_line_idx] = (
    f"#backend: {backend}\n" if backend not in ['', 'auto'] else "##backend: Agg\n")

# 将修改后的内容写入输出文件
output.write_text("".join(template_lines), encoding="utf-8")
```