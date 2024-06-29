# `.\numpy\numpy\distutils\command\develop.py`

```py
"""
用于重写 setuptools 中的 develop 命令，以确保我们从 build_src 或 build_scripts 生成的文件被正确转换为真实文件并具有文件名。

"""
# 从 setuptools.command.develop 导入 develop 类
from setuptools.command.develop import develop as old_develop

# 创建一个新的 develop 类，继承自 old_develop
class develop(old_develop):
    # 复制旧 develop 类的文档字符串
    __doc__ = old_develop.__doc__
    # 安装用于开发的命令
    def install_for_development(self):
        # 在原地构建源文件
        self.reinitialize_command('build_src', inplace=1)
        # 确保脚本已构建
        self.run_command('build_scripts')
        # 调用旧的 develop 类的 install_for_development 方法
        old_develop.install_for_development(self)
```