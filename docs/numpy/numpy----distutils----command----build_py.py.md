# `.\numpy\numpy\distutils\command\build_py.py`

```
# 从distutils.command.build_py导入build_py作为old_build_py
from distutils.command.build_py import build_py as old_build_py
# 从numpy.distutils.misc_util导入is_string
from numpy.distutils.misc_util import is_string

# 创建一个build_py类，继承自old_build_py
class build_py(old_build_py):

    # 重写run方法
    def run(self):
        # 获取build_src命令的最终化版本
        build_src = self.get_finalized_command('build_src')
        # 如果build_src.py_modules_dict存在并且self.packages为None
        if build_src.py_modules_dict and self.packages is None:
            # 将self.packages设置为build_src.py_modules_dict的键列表
            self.packages = list(build_src.py_modules_dict.keys ())
        # 调用old_build_py的run方法
        old_build_py.run(self)

    # 重写find_package_modules方法
    def find_package_modules(self, package, package_dir):
        # 调用old_build_py的find_package_modules方法, 返回模块列表
        modules = old_build_py.find_package_modules(self, package, package_dir)

        # 查找build_src生成的*.py文件
        build_src = self.get_finalized_command('build_src')
        # 将build_src生成的对应包下的模块添加到modules列表中
        modules += build_src.py_modules_dict.get(package, [])

        # 返回最终的模块列表
        return modules

    # 重写find_modules方法
    def find_modules(self):
        # 复制一份self.py_modules, 并用new_py_modules存储其中的字符串元素
        old_py_modules = self.py_modules[:]
        new_py_modules = [_m for _m in self.py_modules if is_string(_m)]
        # 将self.py_modules替换成字符串元素的列表
        self.py_modules[:] = new_py_modules
        # 调用old_build_py的find_modules方法
        modules = old_build_py.find_modules(self)
        # 恢复原始的self.py_modules
        self.py_modules[:] = old_py_modules

        # 返回模块列表
        return modules

    # 修复find_source_files中对于py_modules列表中元素的处理，希望每个元素是3元组，且元素[2]为源文件。
```