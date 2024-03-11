# `.\Langchain-Chatchat\server\model_workers\tiangong.py`

```py
# 导入所需的模块
import json
import time
import hashlib

# 从 fastchat.conversation 模块中导入 Conversation 类
from fastchat.conversation import Conversation
# 从 server.model_workers.base 模块中导入所有内容
from server.model_workers.base import *
# 从 server.utils 模块中导入 get_httpx_client 函数
from server.utils import get_httpx_client
# 从 fastchat.conversation 模块中导入 conversation 模块
from fastchat import conversation as conv
# 再次导入 json 模块
import json
# 从 typing 模块中导入 List, Literal, Dict 类型
from typing import List, Literal, Dict
# 导入 requests 模块
import requests

# 定义 TianGongWorker 类，继承自 ApiModelWorker 类
class TianGongWorker(ApiModelWorker):
    # 初始化方法
    def __init__(
            self,
            *,
            controller_addr: str = None,
            worker_addr: str = None,
            model_names: List[str] = ["tiangong-api"],
            version: Literal["SkyChat-MegaVerse"] = "SkyChat-MegaVerse",
            **kwargs,
    ):
        # 更新 kwargs 字典中的内容
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        # 如果 context_len 不存在于 kwargs 中，则设置默认值为 32768
        kwargs.setdefault("context_len", 32768)
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置版本号
        self.version = version
    # 执行聊天功能，传入参数为 ApiChatParams 对象，返回字典类型结果
    def do_chat(self, params: ApiChatParams) -> Dict:
        # 加载配置文件
        params.load_config(self.model_names[0])

        # 定义 API 请求的 URL
        url = 'https://sky-api.singularity-ai.com/saas/api/v4/generate'
        # 准备请求数据
        data = {
            "messages": params.messages,
            "model": "SkyChat-MegaVerse"
        }
        # 生成时间戳
        timestamp = str(int(time.time()))
        # 生成签名内容
        sign_content = params.api_key + params.secret_key + timestamp
        # 计算签名结果
        sign_result = hashlib.md5(sign_content.encode('utf-8')).hexdigest()
        # 设置请求头
        headers = {
            "app_key": params.api_key,
            "timestamp": timestamp,
            "sign": sign_result,
            "Content-Type": "application/json",
            "stream": "true"  # or change to "false" 不处理流式返回内容
        }

        # 发起 POST 请求并获取响应
        response = requests.post(url, headers=headers, json=data, stream=True)

        # 初始化文本变量
        text = ""
        # 处理响应流
        for line in response.iter_lines(chunk_size=None, decode_unicode=True):
            if line:
                # 处理接收到的数据
                resp = json.loads(line)
                if resp["code"] == 200:
                    text += resp['resp_data']['reply']
                    # 返回成功结果
                    yield {
                        "error_code": 0,
                        "text": text
                    }
                else:
                    # 返回错误结果
                    data = {
                        "error_code": resp["code"],
                        "text": resp["code_msg"]
                    }
                    self.logger.error(f"请求天工 API 时出错：{data}")
                    yield data

    # 获取嵌入
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 创建对话模板
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="",
            messages=[],
            roles=["user", "system"],
            sep="\n### ",
            stop_str="###",
        )
```