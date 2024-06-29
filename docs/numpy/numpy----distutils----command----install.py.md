# `.\numpy\numpy\distutils\command\install.py`

```py
# 导入 sys 模块
import sys
# 检查 setuptools 是否在 sys 模块中
if 'setuptools' in sys.modules:
    # 如果是，导入 setuptools.command.install 模块并重命名为 old_install_mod
    import setuptools.command.install as old_install_mod
    # 设置 have_setuptools 为 True
    have_setuptools = True
else:
    # 如果不是，导入 distutils.command.install 模块并重命名为 old_install_mod
    import distutils.command.install as old_install_mod
    # 设置 have_setuptools 为 False
    have_setuptools = False
# 导入 distutils.file_util 模块中的 write_file 方法
from distutils.file_util import write_file

# 将 old_install_mod.install 方法保存到 old_install 变量中
old_install = old_install_mod.install

# 创建名为 install 的类并继承 old_install 类
class install(old_install):

    # 始终运行 install_clib - 这个命令很便宜，所以无需绕过它；
    # 但它不会被 setuptools 运行 - 所以它会在 install_data 中再次运行
    sub_commands = old_install.sub_commands + [
        ('install_clib', lambda x: True)
    ]

    # 完成选项的设置
    def finalize_options (self):
        old_install.finalize_options(self)
        self.install_lib = self.install_libbase

    # setuptools 版本的 .run() 方法
    def setuptools_run(self):
        """ The setuptools version of the .run() method.

    We must pull in the entire code so we can override the level used in the
    _getframe() call since we wrap this call by one more level.
    """
        from distutils.command.install import install as distutils_install

    # 明确请求旧式安装？直接执行
        if self.old_and_unmanageable or self.single_version_externally_managed:
            return distutils_install.run(self)

    # 尝试检测是否被 setup() 或其他命令调用。如果被 setup() 调用，我们的调用者将是 'distutils.dist' 中的 'run_command' 方法，而*其*调用者将是 'run_commands' 方法。如果以其他方式被调用，我们的直接调用者*可能*是 'run_command'，但其不是被 'run_commands' 调用。这有点笨拙，但似乎可行。
        caller = sys._getframe(3)
        caller_module = caller.f_globals.get('__name__', '')
        caller_name = caller.f_code.co_name

        if caller_module != 'distutils.dist' or caller_name!='run_commands':
    # 我们不是从命令行或 setup() 被调用，所以我们应该以向后兼容的方式运行，以支持 bdist_* 命令。
    distutils_install.run(self)
        else:
    self.do_egg_install()
    # 定义一个方法，用于运行安装程序
    def run(self):
        # 如果没有安装 setuptools，则调用旧的安装方法
        if not have_setuptools:
            r = old_install.run(self)
        # 如果有安装 setuptools，则调用 setuptools 的运行方法
        else:
            r = self.setuptools_run()
        # 如果记录安装的文件
        if self.record:
            # 当 INSTALLED_FILES 包含带有空格的路径时，bdist_rpm 会失败。
            # 这样的路径必须用双引号括起来。
            with open(self.record) as f:
                lines = []
                need_rewrite = False
                # 遍历记录中的每一行
                for l in f:
                    l = l.rstrip()
                    # 如果该行包含空格，则需要重写记录
                    if ' ' in l:
                        need_rewrite = True
                        l = '"%s"' % (l)
                    lines.append(l)
            # 如果需要重写记录，则执行写文件方法，将记录覆盖写入
            if need_rewrite:
                self.execute(write_file, (self.record, lines), "re-writing list of installed files to '%s'" % self.record)
        # 返回运行结果
        return r
```