# `D:\src\scipysrc\scipy\tools\download-wheels.py`

```
#!/usr/bin/env python
"""
Download SciPy wheels from Anaconda staging area.

"""
import os  # 导入操作系统相关的功能
import re  # 导入正则表达式模块
import shutil  # 导入文件操作模块
import argparse  # 导入命令行参数解析模块
import urllib  # 导入 urllib 库
import urllib.request  # 导入 urllib 请求模块

import urllib3  # 导入 urllib3 库
from bs4 import BeautifulSoup  # 从 bs4 库导入 BeautifulSoup 解析 HTML

__version__ = '0.1'

# Edit these for other projects.
STAGING_URL = 'https://anaconda.org/multibuild-wheels-staging/scipy'  # 定义 Anaconda staging 页面的 URL
PREFIX = 'scipy'  # 定义项目的前缀名称

def http_manager():
    """
    Return a urllib3 http request manager, leveraging
    proxy settings when available.
    """
    proxy_dict = urllib.request.getproxies()  # 获取系统的代理设置
    if 'http' in proxy_dict:
        http = urllib3.ProxyManager(proxy_dict['http'])  # 如果有 HTTP 代理，则使用代理创建 urllib3 的 ProxyManager 对象
    elif 'all' in proxy_dict:
        http = urllib3.ProxyManager(proxy_dict['all'])  # 如果有其他代理，也使用代理创建 urllib3 的 ProxyManager 对象
    else:
        http = urllib3.PoolManager(cert_reqs='CERT_REQUIRED')  # 否则创建一个普通的 urllib3 PoolManager 对象
    return http

def get_wheel_names(version):
    """ Get wheel names from Anaconda HTML directory.

    This looks in the Anaconda multibuild-wheels-staging page and
    parses the HTML to get all the wheel names for a release version.

    Parameters
    ----------
    version : str
        The release version. For instance, "1.5.0".

    """
    http = http_manager()  # 获取 HTTP 管理器对象
    tmpl = re.compile(rf"^.*{PREFIX}-{version}-.*\.whl$")  # 编译正则表达式模板，用于匹配特定版本的 wheel 文件名
    index_url = f"{STAGING_URL}/files"  # 构造索引页面的 URL
    index_html = http.request('GET', index_url)  # 发送 HTTP GET 请求获取页面内容
    soup = BeautifulSoup(index_html.data, 'html.parser')  # 使用 BeautifulSoup 解析 HTML 内容
    return soup.findAll(string=tmpl)  # 返回所有匹配正则表达式的字符串列表

def download_wheels(version, wheelhouse):
    """Download release wheels.

    The release wheels for the given SciPy version are downloaded
    into the given directory.

    Parameters
    ----------
    version : str
        The release version. For instance, "1.5.0".
    wheelhouse : str
        Directory in which to download the wheels.

    """
    http = http_manager()  # 获取 HTTP 管理器对象
    wheel_names = get_wheel_names(version)  # 获取特定版本的 wheel 文件名列表

    for i, wheel_name in enumerate(wheel_names):  # 遍历每个 wheel 文件名
        wheel_url = f"{STAGING_URL}/{version}/download/{wheel_name}"  # 构造下载链接
        wheel_path = os.path.join(wheelhouse, wheel_name)  # 构造本地保存路径
        with open(wheel_path, 'wb') as f:  # 打开本地文件，准备写入
            with http.request('GET', wheel_url, preload_content=False) as r:  # 发送 HTTP GET 请求获取文件内容
                print(f"{i + 1:<4}{wheel_name}")  # 打印下载进度信息
                shutil.copyfileobj(r, f)  # 将从网络获取的文件内容写入本地文件
    print(f"\nTotal files downloaded: {len(wheel_names)}")  # 打印总共下载的文件数量

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
    parser.add_argument(
        "version",
        help="SciPy version to download.")  # 添加参数：SciPy 版本号
    parser.add_argument(
        "-w", "--wheelhouse",
        default=os.path.join(os.getcwd(), "release", "installers"),
        help="Directory in which to store downloaded wheels\n"
             "[defaults to <cwd>/release/installers]")  # 添加参数：存放下载文件的目录，默认为当前工作目录下的 release/installers

    args = parser.parse_args()  # 解析命令行参数

    wheelhouse = os.path.expanduser(args.wheelhouse)  # 获取存放文件的目录路径
    if not os.path.isdir(wheelhouse):  # 如果指定的目录不存在
        raise RuntimeError(
            f"{wheelhouse} wheelhouse directory is not present."
            " Perhaps you need to use the '-w' flag to specify one.")  # 抛出运行时错误

    download_wheels(args.version, wheelhouse)  # 调用下载函数，开始下载文件
```