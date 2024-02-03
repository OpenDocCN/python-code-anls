# `.\AutoGPT\autogpts\autogpt\tests\context.py`

```py
# 导入 os 模块
import os
# 导入 sys 模块
import sys

# 将脚本目录添加到路径中，以便可以导入 browse 模块
# 将脚本目录的绝对路径与上级目录 scripts 拼接，然后插入到路径的最前面
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts"))
)
```