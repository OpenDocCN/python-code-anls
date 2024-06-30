# `D:\src\scipysrc\scipy\tools\write_release_and_log.py`

```
"""
Standalone script for writing release doc and logs::

    python tools/write_release_and_log.py <LOG_START> <LOG_END>

Example::

    python tools/write_release_and_log.py v1.7.0 v1.8.0

Needs to be run from the root of the repository.

"""

import os  # 导入操作系统相关功能
import sys  # 导入系统相关功能
import subprocess  # 导入子进程管理模块
from hashlib import md5  # 导入 md5 哈希算法
from hashlib import sha256  # 导入 sha256 哈希算法
from pathlib import Path  # 导入路径操作模块

sys.path.insert(0, os.path.dirname(__file__))  # 将当前脚本所在目录添加到模块搜索路径中
sys.path.insert(1, os.path.join(os.path.dirname(__file__), 'tools'))  # 将当前脚本的 'tools' 子目录添加到模块搜索路径中

try:
    version_utils = __import__("version_utils")  # 动态导入名为 'version_utils' 的模块
    FULLVERSION = version_utils.VERSION  # 获取版本号
    # This is duplicated from tools/version_utils.py
    if os.path.exists('.git'):  # 如果存在 '.git' 目录
        GIT_REVISION, _ = version_utils.git_version(
            os.path.join(os.path.dirname(__file__), '..'))  # 获取 git 版本信息
    else:
        GIT_REVISION = "Unknown"  # 否则标记为未知版本

    if not version_utils.ISRELEASED:  # 如果未发布
        if GIT_REVISION == "Unknown":  # 如果 git 版本为未知
            FULLVERSION += '.dev0+Unknown'  # 添加开发版本信息
        else:
            FULLVERSION += '.dev0+' + GIT_REVISION[:7]  # 添加开发版本信息和部分 git 版本号
finally:
    sys.path.pop(1)  # 移除添加的模块搜索路径
    sys.path.pop(0)  # 移除添加的模块搜索路径

try:
    # Ensure sensible file permissions
    os.umask(0o022)  # 设置文件的默认权限掩码
except AttributeError:
    # No umask on non-posix
    pass  # 如果不支持 umask，则忽略

def get_latest_release_doc(path):
    """
    Method to pick the file from 'doc/release' with the highest
    release number (e.g., `1.9.0-notes.rst`).
    """
    file_paths = os.listdir(path)  # 获取指定路径下的文件列表
    file_paths.sort(key=lambda x: list(map(int, (x.split('-')[0].split('.')))))  # 根据文件名中的版本号排序
    return os.path.join(path, file_paths[-1])  # 返回最新版本的文件路径

# ----------------------------
# Release notes and Changelog
# ----------------------------

def compute_md5(idirs):
    released = os.listdir(idirs)  # 获取目录下的文件列表
    checksums = []
    for fn in sorted(released):  # 遍历排序后的文件列表
        fn_updated = os.path.join("release", fn)  # 拼接文件路径
        with open(fn_updated, 'rb') as f:
            m = md5(f.read())  # 计算文件的 MD5 哈希值
        checksums.append(f'{m.hexdigest()}  {os.path.basename(fn)}')  # 将文件名和其 MD5 哈希值添加到列表
    return checksums  # 返回文件名和对应的 MD5 哈希值列表

def compute_sha256(idirs):
    # better checksum so gpg signed README.txt containing the sums can be used
    # to verify the binaries instead of signing all binaries
    released = os.listdir(idirs)  # 获取目录下的文件列表
    checksums = []
    for fn in sorted(released):  # 遍历排序后的文件列表
        fn_updated = os.path.join("release", fn)  # 拼接文件路径
        with open(fn_updated, 'rb') as f:
            m = sha256(f.read())  # 计算文件的 SHA256 哈希值
        checksums.append(f'{m.hexdigest()}  {os.path.basename(fn)}')  # 将文件名和其 SHA256 哈希值添加到列表
    return checksums  # 返回文件名和对应的 SHA256 哈希值列表

def write_release_task(filename='NOTES.txt'):
    idirs = Path('release')  # 创建 Path 对象表示 'release' 目录
    source = Path(get_latest_release_doc('doc/source/release'))  # 获取最新的发布文档路径
    target = Path(filename)  # 创建 Path 对象表示目标文件名
    if target.exists():  # 如果目标文件已存在
        target.remove()  # 删除目标文件

    tmp_target = Path(filename + '.txt')  # 创建临时目标文件的 Path 对象
    os.system(f'cp {source} {tmp_target}')  # 复制最新的发布文档到临时目标文件

    with open(str(tmp_target), 'a') as ftarget:  # 打开临时目标文件进行追加写入
        ftarget.writelines("""
Checksums
=========

MD5
~~~

""")
        ftarget.writelines(['%s\n' % c for c in compute_md5(idirs)])  # 写入 MD5 校验和
        ftarget.writelines("""
SHA256
~~~~~~
```python`
def write_release_task(fname='README'):
    # 打开或创建目标文件，准备写入释放说明
    with open(fname, 'w') as ftarget:
        # 将计算得到的目录的 SHA256 哈希写入目标文件中
        ftarget.writelines(['%s\n' % c for c in compute_sha256(idirs)])
        # 打印成功生成释放说明的消息
        print("Release README generated successfully")


def write_log
```