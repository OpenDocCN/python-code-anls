# `.\numpy\numpy\distutils\command\bdist_rpm.py`

```
import os
import sys

if 'setuptools' in sys.modules:
    from setuptools.command.bdist_rpm import bdist_rpm as old_bdist_rpm
else:
    from distutils.command.bdist_rpm import bdist_rpm as old_bdist_rpm

class bdist_rpm(old_bdist_rpm):

    def _make_spec_file(self):
        # 调用父类方法以生成原始的 spec 文件内容
        spec_file = old_bdist_rpm._make_spec_file(self)

        # 替换硬编码的 setup.py 脚本名称为实际的 setup 脚本名称
        setup_py = os.path.basename(sys.argv[0])
        if setup_py == 'setup.py':
            # 如果脚本名称为 setup.py，则直接返回原始的 spec 文件内容
            return spec_file

        new_spec_file = []
        # 遍历原始的 spec 文件内容，逐行替换 setup.py 为实际的脚本名称
        for line in spec_file:
            line = line.replace('setup.py', setup_py)
            new_spec_file.append(line)

        # 返回替换后的 spec 文件内容
        return new_spec_file
```