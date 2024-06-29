# `.\numpy\numpy\distutils\command\install_headers.py`

```py
# 导入 os 模块
import os
# 从 distutils.command.install_headers 模块导入 install_headers 类，用作老的 install_headers
from distutils.command.install_headers import install_headers as old_install_headers

# 创建一个名为 install_headers 的类，继承自 old_install_headers 类
class install_headers (old_install_headers):

    # 定义 run 方法
    def run (self):
        # 获取分发的头文件列表
        headers = self.distribution.headers
        # 如果没有头文件，则返回
        if not headers:
            return

        # 获取安装目录的上一级目录
        prefix = os.path.dirname(self.install_dir)
        # 遍历头文件列表
        for header in headers:
            # 如果头文件是一个元组
            if isinstance(header, tuple):
                # 一种小技巧，但我不知道在哪里修改这个...
                if header[0] == 'numpy._core':
                    # 将头文件的第一个元素修改为 'numpy'，保留第二个元素不变
                    header = ('numpy', header[1])
                    # 如果头文件的扩展名为 '.inc'，则跳过此头文件
                    if os.path.splitext(header[1])[1] == '.inc':
                        continue
                # 根据头文件的第一个元素创建安装路径
                d = os.path.join(*([prefix]+header[0].split('.')))
                # 获取头文件的第二个元素
                header = header[1]
            else:
                # 如果头文件不是元组，使用默认的安装路径
                d = self.install_dir
            # 创建安装路径
            self.mkpath(d)
            # 复制头文件到安装路径，并将复制的文件名添加到 outfiles 列表
            (out, _) = self.copy_file(header, d)
            self.outfiles.append(out)
```