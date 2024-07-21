# `.\pytorch\scripts\get_python_cmake_flags.py`

```
# 导入系统模块
import sys
# 导入系统配置模块
import sysconfig

# 设置一个列表，包含Python可执行文件路径作为CMake的参数
flags = [
    f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
]

# 将参数列表转换为空格分隔的字符串并输出到标准输出，以供外部使用
print(" ".join(flags), end="")
```