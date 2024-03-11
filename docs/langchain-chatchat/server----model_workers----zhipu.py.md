# `.\Langchain-Chatchat\server\model_workers\zhipu.py`

```py
# 导入所需模块
from contextlib import contextmanager
import httpx
from fastchat.conversation import Conversation
from httpx_sse import EventSource
from server.model_workers.base import *
from fastchat import conversation as conv
import sys
from typing import List, Dict, Iterator, Literal, Any
import jwt
import time

# 定义上下文管理器，用于建立 SSE 连接
@contextmanager
def connect_sse(client: httpx.Client, method: str, url: str, **kwargs: Any):
    # 使用 HTTPX 客户端建立流式连接
    with client.stream(method, url, **kwargs) as response:
        # 生成 EventSource 对象并返回
        yield EventSource(response)

# 生成 JWT token
def generate_token(apikey: str, exp_seconds: int):
    try:
        # 解析 API key
        id, secret = apikey.split(".")
    except Exception as e:
        # 抛出异常
        raise Exception("invalid apikey", e)

    # 构建 payload
    payload = {
        "api_key": id,
        "exp": int(round(time.time() * 1000)) + exp_seconds * 1000,
        "timestamp": int(round(time.time() * 1000),
    }

    # 使用 secret 对 payload 进行签名，返回 token
    return jwt.encode(
        payload,
        secret,
        algorithm="HS256",
        headers={"alg": "HS256", "sign_type": "SIGN"},
    )

# 定义 ChatGLMWorker 类，继承自 ApiModelWorker
class ChatGLMWorker(ApiModelWorker):
    def __init__(
            self,
            *,
            model_names: List[str] = ["zhipu-api"],
            controller_addr: str = None,
            worker_addr: str = None,
            version: Literal["glm-4"] = "glm-4",
            **kwargs,
    ):
        # 更新 kwargs 中的参数
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 4096)
        # 调用父类的构造函数
        super().__init__(**kwargs)
        # 设置版本号
        self.version = version
    # 定义一个方法用于进行聊天，接收参数为 ApiChatParams 类型，返回一个字典的迭代器
    def do_chat(self, params: ApiChatParams) -> Iterator[Dict]:
        # 加载配置文件
        params.load_config(self.model_names[0])
        # 生成一个 token
        token = generate_token(params.api_key, 60)
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        # 准备请求数据
        data = {
            "model": params.version,
            "messages": params.messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "stream": False
        }

        # 设置请求的 URL
        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        # 使用 httpx 客户端发送 POST 请求
        with httpx.Client(headers=headers) as client:
            response = client.post(url, json=data)
            response.raise_for_status()
            # 获取响应数据
            chunk = response.json()
            print(chunk)
            # 返回结果字典
            yield {"error_code": 0, "text": chunk["choices"][0]["message"]["content"]}

            # 下面是一个注释掉的代码块，暂时不执行
            # with connect_sse(client, "POST", url, json=data) as event_source:
            #     for sse in event_source.iter_sse():
            #         chunk = json.loads(sse.data)
            #         if len(chunk["choices"]) != 0:
            #             text += chunk["choices"][0]["delta"]["content"]
            #             yield {"error_code": 0, "text": text}

    # 定义一个方法用于获取嵌入
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 定义一个方法用于生成对话模板，接收对话模板和模型路径参数，返回一个 Conversation 对象
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是智谱AI小助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n###",
            stop_str="###",
        )
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 导入uvicorn模块
    import uvicorn
    # 从server.utils模块中导入MakeFastAPIOffline函数
    from server.utils import MakeFastAPIOffline
    # 从fastchat.serve.model_worker模块中导入app变量
    from fastchat.serve.model_worker import app

    # 创建ChatGLMWorker对象，指定controller_addr和worker_addr
    worker = ChatGLMWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21001",
    )
    # 将worker对象添加到fastchat.serve.model_worker模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 将FastAPI应用程序app设置为离线模式
    MakeFastAPIOffline(app)
    # 运行FastAPI应用程序app，监听端口21001
    uvicorn.run(app, port=21001)
```