# `arknights-mower\packaging\network.py`

```
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 版权声明和许可证信息
# 导入所需的模块
import os
import sys
import tarfile
import requests
# 导入自定义的日志模块
from ppocr.utils.logging import get_logger

# 定义一个函数，用于显示下载进度
def download_with_progressbar(url, save_path):
    # 获取日志记录器
    logger = get_logger()
    # 发送带有进度条的 GET 请求
    response = requests.get(url, stream=True)
    # 如果请求成功
    if response.status_code == 200:
        # 获取文件总大小
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        block_size = 1024  # 1 Kibibyte
        # 打开文件，写入数据
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
    # 如果请求失败
    else:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)

# 定义一个函数，用于下载模型文件
def maybe_download(model_storage_directory, url):
    # 使用自定义模型
    tar_file_name_list = ['.pdiparams', '.pdiparams.info', '.pdmodel']
    # 如果模型文件不存在
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')
    # 检查模型存储目录是否存在，且不包含'inference.pdmodel'文件
    ) or not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdmodel')):
        # 断言 URL 是否以'.tar'结尾，只支持tar压缩包
        assert url.endswith('.tar'), 'Only supports tar compressed package'
        # 生成临时路径，用于下载模型文件
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        # 打印下载信息
        print('download {} to {}'.format(url, tmp_path))
        # 创建模型存储目录
        os.makedirs(model_storage_directory, exist_ok=True)
        # 使用进度条下载模型文件
        download_with_progressbar(url, tmp_path)
        # 打开tar文件
        with tarfile.open(tmp_path, 'r') as tarObj:
            # 遍历tar文件中的成员
            for member in tarObj.getmembers():
                filename = None
                # 遍历tar文件名列表
                for tar_file_name in tar_file_name_list:
                    # 如果成员名以tar文件名结尾，则设置filename为'inference' + tar_file_name
                    if member.name.endswith(tar_file_name):
                        filename = 'inference' + tar_file_name
                # 如果filename为None，则跳过当前成员
                if filename is None:
                    continue
                # 从tar文件中提取文件
                file = tarObj.extractfile(member)
                # 将提取的文件写入模型存储目录
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        # 删除临时文件
        os.remove(tmp_path)
# 判断字符串是否为链接，如果不是None并且以'http'开头则返回True，否则返回False
def is_link(s):
    return s is not None and s.startswith('http')

# 确认模型目录的URL，如果模型目录为None或者是链接，则使用默认的模型目录和默认的URL
def confirm_model_dir_url(model_dir, default_model_dir, default_url):
    # 默认的URL
    url = default_url
    # 如果模型目录为None或者是链接
    if model_dir is None or is_link(model_dir):
        # 如果模型目录是链接，则使用该链接作为URL
        if is_link(model_dir):
            url = model_dir
        # 从URL中提取文件名
        file_name = url.split('/')[-1][:-4]
        # 使用默认的模型目录
        model_dir = default_model_dir
        # 将文件名添加到模型目录中
        model_dir = os.path.join(model_dir, file_name)
    # 返回模型目录和URL
    return model_dir, url
```