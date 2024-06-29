# `.\numpy\numpy\distutils\command\build_scripts.py`

```
# 修改了从函数构建脚本的build_scripts版本的脚本
from distutils.command.build_scripts import build_scripts as old_build_scripts  # 从distutils.command.build_scripts导入build_scripts
from numpy.distutils import log  # 从numpy.distutils导入log
from numpy.distutils.misc_util import is_string  # 从numpy.distutils.misc_util导入is_string

class build_scripts(old_build_scripts):  # 创建build_scripts类，继承自old_build_scripts类

    def generate_scripts(self, scripts):  # 定义generate_scripts方法，接受参数scripts
        new_scripts = []  # 创建空列表new_scripts
        func_scripts = []  # 创建空列表func_scripts
        for script in scripts:  # 遍历参数scripts
            if is_string(script):  # 判断是否为字符串
                new_scripts.append(script)  # 将script添加到new_scripts
            else:
                func_scripts.append(script)  # 将script添加到func_scripts
        if not func_scripts:  # 如果func_scripts为空
            return new_scripts  # 返回new_scripts

        build_dir = self.build_dir  # 获取build_dir属性
        self.mkpath(build_dir)  # 创建build_dir目录
        for func in func_scripts:  # 遍历func_scripts
            script = func(build_dir)  # 调用func方法，传入build_dir参数，获取返回值赋给script
            if not script:  # 如果没有返回值
                continue  # 继续下一次循环
            if is_string(script):  # 如果返回值是字符串
                log.info("  adding '%s' to scripts" % (script,))  # 记录日志
                new_scripts.append(script)  # 将返回值添加到new_scripts
            else:
                [log.info("  adding '%s' to scripts" % (s,)) for s in script]  # 对每个返回值记录日志
                new_scripts.extend(list(script))  # 将返回值添加到new_scripts
        return new_scripts  # 返回new_scripts

    def run (self):  # 定义run方法
        if not self.scripts:  # 如果没有脚本
            return  # 返回

        self.scripts = self.generate_scripts(self.scripts)  # 调用generate_scripts方法，更新self.scripts
        # 确保distribution对象具有脚本列表。setuptools的develop命令需要这个列表是文件名，而不是函数。
        self.distribution.scripts = self.scripts  # 设置distribution对象的scripts属性为self.scripts

        return old_build_scripts.run(self)  # 调用old_build_scripts的run方法

    def get_source_files(self):  # 定义get_source_files方法
        from numpy.distutils.misc_util import get_script_files  # 从numpy.distutils.misc_util导入get_script_files
        return get_script_files(self.scripts)  # 调用get_script_files方法，传入self.scripts参数，并返回结果
```