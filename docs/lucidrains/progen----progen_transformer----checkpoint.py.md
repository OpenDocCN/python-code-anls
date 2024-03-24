# `.\lucidrains\progen\progen_transformer\checkpoint.py`

```
# 导入所需的模块
import time
import os, errno
from pathlib import Path
from functools import partial
# 导入 Google Cloud Storage 相关模块
from google.cloud import storage
from cloudpickle import pickle
from progen_transformer.utils import clear_directory_, silentremove

# 文件系统检查点函数

# 重置文件系统检查点
def file_reset_checkpoint(path):
    clear_directory_(path)

# 获取文件系统中最后一个检查点
def file_get_last_checkpoint(path):
    checkpoints = sorted(path.glob('**/ckpt_*'))
    if len(checkpoints) == 0:
        return None

    with open(str(checkpoints[-1]), 'rb') as f:
        package = pickle.load(f)

    return package

# 保存文件系统检查点
def file_save_checkpoint(path, package, keep_last_n = None):
    unix_time = int(time.time())
    checkpoints = sorted(path.glob('**/ckpt_*'))
    num_checkpoints = len(checkpoints)

    with open(str(path / f'ckpt_{unix_time}.pkl'), 'wb') as f:
        pickle.dump(package, f)

    if keep_last_n is None:
        return

    for path_to_rm in checkpoints[:max(0, num_checkpoints - keep_last_n)]:
        silentremove(path_to_rm)

# Google Cloud Storage 检查点函数

GCS_READ_TIMEOUT = 60 * 30
GCS_WRITE_TIMEOUT = 60 * 30

# 重置 Google Cloud Storage 检查点
def gcs_reset_checkpoint(bucket):
    bucket.delete_blobs(list(bucket.list_blobs()))

# 获取 Google Cloud Storage 中最后一个检查点
def gcs_get_last_checkpoint(bucket):
    blobs = sorted(list(bucket.list_blobs()))

    if len(blobs) == 0:
        return None

    last_checkpoint = blobs[-1]

    filename = f'/tmp/{last_checkpoint.name}'
    with open(filename, 'wb') as f:
        last_checkpoint.download_to_file(f, timeout = GCS_READ_TIMEOUT)

    with open(filename, 'rb') as f:
        package = pickle.load(f)

    return package

# 保存 Google Cloud Storage 检查点
def gcs_save_checkpoint(bucket, package, keep_last_n = None):
    unix_time = int(time.time())
    blobs = sorted(list(bucket.list_blobs()))
    num_checkpoints = len(blobs)

    filename = f'ckpt_{unix_time}.pkl'
    tmp_path = f'/tmp/{filename}'

    with open(tmp_path, 'wb') as f:
        pickle.dump(package, f)

    blob = bucket.blob(filename)
    blob.upload_from_filename(tmp_path, timeout = GCS_WRITE_TIMEOUT)

    if keep_last_n is None:
        return

    bucket.delete_blobs(blobs[:max(0, num_checkpoints - keep_last_n)])

# 工厂函数

# 获取检查点函数
def get_checkpoint_fns(path):
    # 判断是否使用 Google Cloud Storage
    use_gcs = path.startswith('gs://')

    if not use_gcs:
        obj = Path(path)
        obj.mkdir(exist_ok = True, parents = True)

        fns = (
            file_reset_checkpoint,
            file_get_last_checkpoint,
            file_save_checkpoint
        )
    else:
        client = storage.Client()
        bucket_name = path[5:]
        obj = client.get_bucket(bucket_name)

        fns = (
            gcs_reset_checkpoint,
            gcs_get_last_checkpoint,
            gcs_save_checkpoint
        )

    # 将函数对象和路径对象绑定，返回函数元组
    fns = tuple(map(lambda fn: partial(fn, obj), fns))
    return fns
```