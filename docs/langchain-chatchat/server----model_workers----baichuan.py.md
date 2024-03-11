# `.\Langchain-Chatchat\server\model_workers\baichuan.py`

```py
# 导入所需的模块
import json
import time
import hashlib

from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from fastchat import conversation as conv
import sys
import json
from typing import List, Literal, Dict
from configs import logger, log_verbose

# 计算输入字符串的 MD5 值
def calculate_md5(input_string):
    # 创建 MD5 对象
    md5 = hashlib.md5()
    # 更新 MD5 对象的值
    md5.update(input_string.encode('utf-8'))
    # 获取 MD5 值的十六进制表示
    encrypted = md5.hexdigest()
    return encrypted

# BaiChuanWorker 类，继承自 ApiModelWorker 类
class BaiChuanWorker(ApiModelWorker):
    def __init__(
        self,
        *,
        controller_addr: str = None,
        worker_addr: str = None,
        model_names: List[str] = ["baichuan-api"],
        version: Literal["Baichuan2-53B"] = "Baichuan2-53B",
        **kwargs,
    ):
        # 更新参数
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 32768)
        super().__init__(**kwargs)
        self.version = version

    # 获取嵌入向量
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 创建对话模板
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )

# 主程序入口
if __name__ == "__main__":
    # 导入所需模块
    import uvicorn
    from server.utils import MakeFastAPIOffline
    from fastchat.serve.model_worker import app

    # 创建 BaiChuanWorker 实例
    worker = BaiChuanWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21007",
    )
    # 将 worker 实例添加到模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 将 FastAPI 离线化
    MakeFastAPIOffline(app)
    # 运行 FastAPI 应用
    uvicorn.run(app, port=21007)
    # 执行请求
    # do_request()
```