# `.\PaddleOCR\ppocr\utils\network.py`

```
# 版权声明和许可信息
#
# 版权所有 (c) 2020 PaddlePaddle 作者。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。
#

# 导入所需的库
import os
import sys
import tarfile
import requests
from tqdm import tqdm

# 导入自定义的日志记录器
from ppocr.utils.logging import get_logger

# 模型存储目录
MODELS_DIR = os.path.expanduser("~/.paddleocr/models/")

# 下载文件并显示下载进度
def download_with_progressbar(url, save_path):
    # 获取日志记录器
    logger = get_logger()
    # 发送 GET 请求获取文件流
    response = requests.get(url, stream=True)
    # 如果响应状态码为 200
    if response.status_code == 200:
        # 获取文件总大小
        total_size_in_bytes = int(response.headers.get('content-length', 1))
        # 设置块大小为 1 Kibibyte
        block_size = 1024
        # 创建进度条对象
        progress_bar = tqdm(
            total=total_size_in_bytes, unit='iB', unit_scale=True)
        # 打开文件并写入数据
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        # 关闭进度条
        progress_bar.close()
    else:
        # 如果下载出错，记录错误信息
        logger.error("Something went wrong while downloading models")
        # 退出程序
        sys.exit(0)

# 下载模型文件（如果不存在）
def maybe_download(model_storage_directory, url):
    # 使用自定义模型
    tar_file_name_list = ['.pdiparams', '.pdiparams.info', '.pdmodel']
    # 如果推理模型文件不存在
    if not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdiparams')):
    # 检查模型存储目录是否存在，以及是否存在推理模型文件
    ) or not os.path.exists(
            os.path.join(model_storage_directory, 'inference.pdmodel')):
        # 断言 URL 是否以 '.tar' 结尾，只支持 tar 压缩包
        assert url.endswith('.tar'), 'Only supports tar compressed package'
        # 生成临时路径，用于下载模型文件
        tmp_path = os.path.join(model_storage_directory, url.split('/')[-1])
        # 打印下载信息
        print('download {} to {}'.format(url, tmp_path))
        # 创建模型存储目录
        os.makedirs(model_storage_directory, exist_ok=True)
        # 使用进度条下载模型文件
        download_with_progressbar(url, tmp_path)
        # 打开 tar 文件
        with tarfile.open(tmp_path, 'r') as tarObj:
            # 遍历 tar 文件中的成员
            for member in tarObj.getmembers():
                filename = None
                # 遍历 tar 文件名列表
                for tar_file_name in tar_file_name_list:
                    # 如果成员名以指定的 tar 文件名结尾，则设置文件名
                    if member.name.endswith(tar_file_name):
                        filename = 'inference' + tar_file_name
                # 如果没有找到文件名，则继续下一个成员
                if filename is None:
                    continue
                # 提取文件对象
                file = tarObj.extractfile(member)
                # 写入文件数据到模型存储目录
                with open(
                        os.path.join(model_storage_directory, filename),
                        'wb') as f:
                    f.write(file.read())
        # 删除临时文件
        os.remove(tmp_path)
# 检查模型参数文件是否已经存在，如果存在或者不是链接，则直接返回模型路径
def maybe_download_params(model_path):
    if os.path.exists(model_path) or not is_link(model_path):
        return model_path
    else:
        url = model_path
    # 构建临时路径，用于下载模型参数文件
    tmp_path = os.path.join(MODELS_DIR, url.split('/')[-1])
    # 打印下载信息
    print('download {} to {}'.format(url, tmp_path))
    # 确保模型目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    # 使用进度条下载模型参数文件
    download_with_progressbar(url, tmp_path)
    return tmp_path

# 判断字符串是否为链接
def is_link(s):
    return s is not None and s.startswith('http')

# 确认模型目录和链接
def confirm_model_dir_url(model_dir, default_model_dir, default_url):
    url = default_url
    if model_dir is None or is_link(model_dir):
        if is_link(model_dir):
            url = model_dir
        # 获取文件名
        file_name = url.split('/')[-1][:-4]
        model_dir = default_model_dir
        # 构建模型目录
        model_dir = os.path.join(model_dir, file_name)
    return model_dir, url
```