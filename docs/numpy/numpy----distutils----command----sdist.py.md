# `.\numpy\numpy\distutils\command\sdist.py`

```py
# 导入sys模块
import sys
# 检查是否在sys模块中存在setuptools模块
if 'setuptools' in sys.modules:
    # 如果存在，则导入setuptools.command.sdist模块中的sdist类，并命名为old_sdist
    from setuptools.command.sdist import sdist as old_sdist
else:
    # 如果不存在，则导入distutils.command.sdist模块中的sdist类，并命名为old_sdist
    from distutils.command.sdist import sdist as old_sdist

# 导入numpy.distutils.misc_util模块中的get_data_files函数
from numpy.distutils.misc_util import get_data_files

# 创建一个名为sdist的类，继承自old_sdist类
class sdist(old_sdist):

    # 定义add_defaults方法
    def add_defaults (self):
        # 调用old_sdist类的add_defaults方法
        old_sdist.add_defaults(self)

        # 获取self.distribution，并赋值给dist
        dist = self.distribution

        # 如果dist具有数据文件，则遍历dist.data_files并将文件列表添加到filelist中
        if dist.has_data_files():
            for data in dist.data_files:
                self.filelist.extend(get_data_files(data))

        # 如果dist具有头文件，则遍历dist.headers，如果是字符串则直接添加到filelist中，如果是元组则取索引为1的元素添加到filelist中
        if dist.has_headers():
            headers = []
            for h in dist.headers:
                if isinstance(h, str): headers.append(h)
                else: headers.append(h[1])
            self.filelist.extend(headers)

        return
```