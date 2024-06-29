# `D:\src\scipysrc\matplotlib\ci\schemas\vendor_schemas.py`

```
#!/usr/bin/env python3
"""
Download YAML Schemas for linting and validation.

Since pre-commit CI doesn't have Internet access, we need to bundle these files
in the repo.
"""

import os                     # 导入操作系统功能模块
import pathlib                # 导入路径操作模块
import urllib.request        # 导入用于网络请求的模块


HERE = pathlib.Path(__file__).parent   # 获取当前脚本文件的父目录路径
SCHEMAS = [
    'https://json.schemastore.org/appveyor.json',
    'https://json.schemastore.org/circleciconfig.json',
    'https://json.schemastore.org/github-funding.json',
    'https://json.schemastore.org/github-issue-config.json',
    'https://json.schemastore.org/github-issue-forms.json',
    'https://json.schemastore.org/codecov.json',
    'https://json.schemastore.org/pull-request-labeler-5.json',
    'https://github.com/microsoft/vscode-python/raw/'
        'main/schemas/conda-environment.json',
]                                 # 定义包含多个 JSON 文件 URL 的列表


def print_progress(block_count, block_size, total_size):
    """
    打印下载进度条的函数。

    Parameters:
    block_count (int): 已下载的数据块数
    block_size (int): 每个数据块的大小
    total_size (int): 要下载的总大小
    """
    size = block_count * block_size   # 计算已下载数据的大小
    if total_size != -1:              # 如果总大小已知
        size = min(size, total_size)  # 取实际下载大小和总大小的较小值
        width = 50                    # 进度条宽度
        percent = size / total_size * 100  # 计算下载进度百分比
        filled = int(percent // (100 // width))  # 计算填充的方块数
        percent_str = '\N{Full Block}' * filled + '\N{Light Shade}' * (width - filled)  # 构建进度条字符串
    print(f'{percent_str} {size:6d} / {total_size:6d}', end='\r')  # 打印进度条信息


# 首先清理现有文件。
for json in HERE.glob('*.json'):    # 遍历当前目录下所有以 .json 结尾的文件
    os.remove(json)                 # 删除这些文件

for schema in SCHEMAS:             # 遍历每个 JSON 文件的 URL
    path = HERE / schema.rsplit('/', 1)[-1]  # 提取 URL 中最后一部分作为文件名，构建本地路径
    print(f'Downloading {schema} to {path}')  # 打印下载信息
    urllib.request.urlretrieve(schema, filename=path, reporthook=print_progress)  # 下载文件，并显示下载进度
    print()                          # 打印空行，提升可读性
    # 这看起来有些奇怪，但它会将文件的换行符标准化为当前平台的格式，
    # 这样 Git 在处理时不会报错。
    path.write_text(path.read_text())  # 将文件内容重新写入文件，以便换行符标准化
```