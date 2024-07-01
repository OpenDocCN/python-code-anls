# `.\numpy\numpy\typing\tests\data\fail\lib_version.pyi`

```py
# 从 numpy.lib 模块中导入 NumpyVersion 类
from numpy.lib import NumpyVersion

# 声明一个变量 version，类型为 NumpyVersion
version: NumpyVersion

# 创建一个 NumpyVersion 对象，传入字节字符串 b"1.8.0"，表示 numpy 的版本号为 1.8.0
NumpyVersion(b"1.8.0")  # E: incompatible type

# 检查 version 对象是否大于或等于字节字符串 b"1.8.0"，由于版本号与字节字符串比较，导致不支持的操作类型错误
version >= b"1.8.0"  # E: Unsupported operand types
```