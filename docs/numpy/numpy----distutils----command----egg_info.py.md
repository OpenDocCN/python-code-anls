# `.\numpy\numpy\distutils\command\egg_info.py`

```py
# 导入sys模块
import sys

# 从setuptools.command.egg_info模块中导入egg_info类，重命名为_egg_info
from setuptools.command.egg_info import egg_info as _egg_info

# 定义egg_info类，继承自_egg_info类
class egg_info(_egg_info):
    # 定义run方法
    def run(self):
        # 如果'sdist'在sys.argv中
        if 'sdist' in sys.argv:
            # 导入warnings模块
            import warnings
            # 导入textwrap模块
            import textwrap
            # 定义消息内容
            msg = textwrap.dedent("""
                `build_src` is being run, this may lead to missing
                files in your sdist!  You want to use distutils.sdist
                instead of the setuptools version:

                    from distutils.command.sdist import sdist
                    cmdclass={'sdist': sdist}"

                See numpy's setup.py or gh-7131 for details.""")
            # 发出UserWarning警告，显示消息内容
            warnings.warn(msg, UserWarning, stacklevel=2)

        # 确保build_src已经执行，以便为setuptools的egg_info命令提供真实的文件名，而不是生成文件的函数
        self.run_command("build_src")
        # 调用_egg_info类的run方法
        _egg_info.run(self)
```