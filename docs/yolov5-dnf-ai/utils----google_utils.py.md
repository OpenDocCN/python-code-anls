# `yolov5-DNF\utils\google_utils.py`

```
# 这个文件包含了谷歌的工具：https://cloud.google.com/storage/docs/reference/libraries
# 通过 pip 安装最新版本的 google-cloud-storage
# from google.cloud import storage

import os
import platform
import subprocess
import time
from pathlib import Path

import torch


def gsutil_getsize(url=''):
    # 使用 gsutil du 命令获取指定 URL 的文件大小，URL 格式为 gs://bucket/file，参考：https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output('gsutil du %s' % url, shell=True).decode('utf-8')
    return eval(s.split(' ')[0]) if len(s) else 0  # 返回文件大小，单位为字节


def attempt_download(weights):
    # 如果本地不存在预训练权重文件，则尝试下载
    weights = weights.strip().replace("'", '')
    file = Path(weights).name

    # 提示预训练权重文件缺失，尝试从 https://github.com/ultralytics/yolov5/releases/ 下载
    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/releases/'
    # 可用的模型列表
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']
    # 如果文件在模型中，并且权重文件不存在
    if file in models and not os.path.isfile(weights):
        try:  # 尝试从 GitHub 下载
            # 构建下载链接
            url = 'https://github.com/ultralytics/yolov5/releases/download/v3.0/' + file
            # 打印下载信息
            print('Downloading %s to %s...' % (url, weights))
            # 使用 torch.hub.download_url_to_file() 方法下载文件
            torch.hub.download_url_to_file(url, weights)
            # 检查文件是否存在并且大小大于 1MB
            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6  # check
        except Exception as e:  # 如果出现异常，尝试从 GCP 下载
            # 打印下载错误信息
            print('Download error: %s' % e)
            # 构建 GCP 下载链接
            url = 'https://storage.googleapis.com/ultralytics/yolov5/ckpt/' + file
            # 打印下载信息
            print('Downloading %s to %s...' % (url, weights))
            # 使用 curl 命令下载文件
            r = os.system('curl -L %s -o %s' % (url, weights))  # torch.hub.download_url_to_file(url, weights)
        finally:
            # 如果文件不存在或者大小不符合要求
            if not (os.path.exists(weights) and os.path.getsize(weights) > 1E6):  # check
                # 如果文件存在，删除文件
                os.remove(weights) if os.path.exists(weights) else None  # remove partial downloads
                # 打印错误信息
                print('ERROR: Download failure: %s' % msg)
            # 打印空行
            print('')
            # 返回
            return
# 从谷歌驱动器下载文件
def gdrive_download(id='1n_oKgR81BJtqk75b00eAjdv03qVCQn2f', name='coco128.zip'):
    # 记录开始时间
    t = time.time()

    # 打印下载链接和文件名
    print('Downloading https://drive.google.com/uc?export=download&id=%s as %s... ' % (id, name), end='')
    # 如果文件存在，则删除
    os.remove(name) if os.path.exists(name) else None  # remove existing
    # 如果 cookie 文件存在，则删除
    os.remove('cookie') if os.path.exists('cookie') else None

    # 尝试下载文件
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system('curl -c ./cookie -s -L "drive.google.com/uc?export=download&id=%s" > %s ' % (id, out))
    # 如果 cookie 文件存在，表示文件较大
    if os.path.exists('cookie'):  # large file
        # 获取 token 并下载文件
        s = 'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm=%s&id=%s" -o %s' % (get_token(), id, name)
    else:  # 文件较小
        # 直接下载文件
        s = 'curl -s -L -o %s "drive.google.com/uc?export=download&id=%s"' % (name, id)
    # 执行命令，捕获返回值
    r = os.system(s)  # execute, capture return
    # 如果 cookie 文件存在，则删除
    os.remove('cookie') if os.path.exists('cookie') else None

    # 检查是否有错误发生
    if r != 0:
        # 如果有错误发生，删除部分下载的文件
        os.remove(name) if os.path.exists(name) else None  # remove partial
        print('Download error ')  # raise Exception('Download error')
        return r

    # 如果文件是压缩文件，则解压缩
    if name.endswith('.zip'):
        print('unzipping... ', end='')
        os.system('unzip -q %s' % name)  # unzip
        os.remove(name)  # remove zip to free space

    # 打印下载完成的信息和所花时间
    print('Done (%.1fs)' % (time.time() - t))
    return r


# 从 cookie 文件中获取 token
def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""

# 上传文件到存储桶
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
# 从本地文件上传到云存储中的指定位置
blob.upload_from_filename(source_file_name)

# 打印上传成功的消息，包括源文件名和目标位置
print('File {} uploaded to {}.'.format(
    source_file_name,
    destination_blob_name))


# 从云存储中下载文件到本地
def download_blob(bucket_name, source_blob_name, destination_file_name):
    # 创建存储客户端对象
    storage_client = storage.Client()
    # 获取指定名称的存储桶
    bucket = storage_client.get_bucket(bucket_name)
    # 获取存储桶中指定的 blob 对象
    blob = bucket.blob(source_blob_name)

    # 将 blob 对象的内容下载到本地文件
    blob.download_to_filename(destination_file_name)

    # 打印下载成功的消息，包括源 blob 名和目标文件名
    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
```