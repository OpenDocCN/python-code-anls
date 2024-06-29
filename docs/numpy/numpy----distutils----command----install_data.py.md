# `.\numpy\numpy\distutils\command\install_data.py`

```
# 导入sys模块
import sys
# 检查是否已导入setuptools模块
have_setuptools = ('setuptools' in sys.modules)

# 导入distutils.command.install_data模块的install_data类为old_install_data
from distutils.command.install_data import install_data as old_install_data

# 数据安装程序，比distutils具有更好的智能
# 数据文件被复制到项目目录而不是随意放置
class install_data (old_install_data):

    # 运行安装数据程序
    def run(self):
        # 调用old_install_data的run方法
        old_install_data.run(self)

        # 如果已经导入setuptools模块
        if have_setuptools:
            # 再次运行install_clib，因为setuptools不会自动运行install的子命令
            self.run_command('install_clib')

    # 设置选项的最终方法
    def finalize_options (self):
        # 设置未定义的选项 install_lib, root, force
        self.set_undefined_options('install',
                                   ('install_lib', 'install_dir'),
                                   ('root', 'root'),
                                   ('force', 'force'),
                                  )
```