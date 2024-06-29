# `.\numpy\numpy\typing\tests\data\pass\lib_version.py`

```py
# 从 numpy 库中导入 NumpyVersion 类
from numpy.lib import NumpyVersion

# 创建一个 NumpyVersion 对象，表示版本号为 "1.8.0"
version = NumpyVersion("1.8.0")

# 获取版本号的字符串表示形式
version.vstring

# 获取版本号的完整列表形式
version.version

# 获取版本号的主要版本号部分
version.major

# 获取版本号的次要版本号部分
version.minor

# 获取版本号的修订版本号部分
version.bugfix

# 获取版本号的预发布版本信息，如果有的话
version.pre_release

# 检查版本号是否为开发版本
version.is_devversion

# 检查两个版本号对象是否相等
version == version

# 检查两个版本号对象是否不相等
version != version

# 检查版本号对象是否小于给定的版本号字符串 "1.8.0"
version < "1.8.0"

# 检查版本号对象是否小于等于另一个版本号对象
version <= version

# 检查版本号对象是否大于另一个版本号对象
version > version

# 检查版本号对象是否大于等于给定的版本号字符串 "1.8.0"
version >= "1.8.0"
```