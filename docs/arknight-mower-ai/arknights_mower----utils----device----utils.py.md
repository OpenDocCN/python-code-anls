# `arknights-mower\arknights_mower\utils\device\utils.py`

```
# 导入必要的模块
from __future__ import annotations
import http
import socket
import tempfile
import requests
from ... import __system__
from ..log import logger

# 定义一个函数，用于下载文件到临时路径，并返回文件路径以供进一步使用
def download_file(target_url: str) -> str:
    # 记录下载的目标 URL
    logger.debug(f'downloading: {target_url}')
    # 发起 HTTP 请求，获取响应
    resp = requests.get(target_url, verify=False)
    # 使用临时文件来保存下载的内容
    with tempfile.NamedTemporaryFile('wb+', delete=False) as f:
        file_name = f.name
        # 将响应内容写入临时文件
        f.write(resp.content)
    # 返回临时文件的文件路径
    return file_name

# 下面的函数被注释掉了，不会执行
# def is_port_using(host: str, port: int) -> bool:
#     """ if port is using by others, return True. else return False """
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.settimeout(1)

#     try:
#         result = s.connect_ex((host, port))
#         # if port is using, return code should be 0. (can be connected)
#         return result == 0
#     finally:
#         s.close()
```