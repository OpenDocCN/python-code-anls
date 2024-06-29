# `.\numpy\numpy\distutils\numpy_distribution.py`

```py
# XXX: 处理 setuptools ?
# 从 distutils.core 模块导入 Distribution 类
from distutils.core import Distribution

# 这个类用于添加新文件（例如 sconscripts 等）时使用 scons 命令
class NumpyDistribution(Distribution):
    def __init__(self, attrs = None):
        # 存储元组列表，每个元组包含 (sconscripts, pre_hook, post_hook, src, parent_names)
        self.scons_data = []
        # 存储可安装的库列表
        self.installed_libraries = []
        # 存储要生成/安装的 pkg_config 文件的字典
        self.installed_pkg_config = {}
        # 调用父类 Distribution 的初始化方法
        Distribution.__init__(self, attrs)

    # 检查是否存在 scons 脚本
    def has_scons_scripts(self):
        return bool(self.scons_data)
```