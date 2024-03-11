# `.\Langchain-Chatchat\server\model_workers\xinghuo.py`

```
# 从 fastchat.conversation 模块中导入 Conversation 类
from fastchat.conversation import Conversation
# 从 server.model_workers.base 模块中导入所有内容
from server.model_workers.base import *
# 从 fastchat 模块中导入 conversation 别名为 conv
from fastchat import conversation as conv
# 导入 sys 模块
import sys
# 导入 json 模块
import json
# 从 server.model_workers 模块中导入 SparkApi 类
from server.model_workers import SparkApi
# 导入 websockets 模块
import websockets
# 从 server.utils 模块中导入 iter_over_async 和 asyncio
from server.utils import iter_over_async, asyncio
# 从 typing 模块中导入 List 和 Dict 类型
from typing import List, Dict

# 定义异步函数 request，接收多个参数
async def request(appid, api_key, api_secret, Spark_url, domain, question, temperature, max_token):
    # 创建 SparkApi.Ws_Param 对象
    wsParam = SparkApi.Ws_Param(appid, api_key, api_secret, Spark_url)
    # 生成 WebSocket 连接的 URL
    wsUrl = wsParam.create_url()
    # 生成请求参数数据
    data = SparkApi.gen_params(appid, domain, question, temperature, max_token)
    # 使用 websockets 连接到 wsUrl
    async with websockets.connect(wsUrl) as ws:
        # 发送数据到 WebSocket 服务器
        await ws.send(json.dumps(data, ensure_ascii=False))
        # 初始化 finish 变量为 False
        finish = False
        # 循环接收 WebSocket 服务器返回的数据
        while not finish:
            # 接收数据块
            chunk = await ws.recv()
            # 将接收到的数据解析为 JSON 格式
            response = json.loads(chunk)
            # 如果返回数据中的状态为 2，则设置 finish 为 True
            if response.get("header", {}).get("status") == 2:
                finish = True
            # 如果返回数据中包含文本内容，则生成文本内容的生成器
            if text := response.get("payload", {}).get("choices", {}).get("text"):
                yield text[0]["content"]

# 定义 XingHuoWorker 类，继承自 ApiModelWorker 类
class XingHuoWorker(ApiModelWorker):
    # 初始化方法，接收多个参数
    def __init__(
            self,
            *,
            model_names: List[str] = ["xinghuo-api"],
            controller_addr: str = None,
            worker_addr: str = None,
            version: str = None,
            **kwargs,
    ):
        # 更新 kwargs 字典中的内容
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        # 设置 context_len 默认值为 8000
        kwargs.setdefault("context_len", 8000)
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置版本号
        self.version = version
    # 执行聊天功能，接收参数为 ApiChatParams 对象，返回字典
    def do_chat(self, params: ApiChatParams) -> Dict:
        # 加载配置文件
        params.load_config(self.model_names[0])

        # 定义不同版本的映射关系
        version_mapping = {
            "v1.5": {"domain": "general", "url": "ws://spark-api.xf-yun.com/v1.1/chat", "max_tokens": 4000},
            "v2.0": {"domain": "generalv2", "url": "ws://spark-api.xf-yun.com/v2.1/chat", "max_tokens": 8000},
            "v3.0": {"domain": "generalv3", "url": "ws://spark-api.xf-yun.com/v3.1/chat", "max_tokens": 8000},
            "v3.5": {"domain": "generalv3", "url": "ws://spark-api.xf-yun.com/v3.5/chat", "max_tokens": 16000},
        }

        # 获取指定版本的详细信息
        def get_version_details(version_key):
            return version_mapping.get(version_key, {"domain": None, "url": None})

        # 根据参数版本获取详细信息
        details = get_version_details(params.version)
        domain = details["domain"]
        Spark_url = details["url"]
        text = ""
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()
        
        # 设置最大 tokens 数量
        params.max_tokens = min(details["max_tokens"], params.max_tokens or 0)
        
        # 遍历异步请求的结果
        for chunk in iter_over_async(
                request(params.APPID, params.api_key, params.APISecret, Spark_url, domain, params.messages,
                        params.temperature, params.max_tokens),
                loop=loop,
        ):
            if chunk:
                text += chunk
                # 返回结果字典
                yield {"error_code": 0, "text": text}

    # 获取嵌入信息
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 创建对话模板
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明的助手，请根据用户的提示来完成任务",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )
# 如果当前脚本被直接执行，则执行以下代码块
if __name__ == "__main__":
    # 导入uvicorn模块
    import uvicorn
    # 从server.utils模块中导入MakeFastAPIOffline函数
    from server.utils import MakeFastAPIOffline
    # 从fastchat.serve.model_worker模块中导入app变量
    from fastchat.serve.model_worker import app

    # 创建XingHuoWorker对象，指定controller_addr和worker_addr
    worker = XingHuoWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21003",
    )
    # 将worker对象添加到fastchat.serve.model_worker模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 将FastAPI应用程序app设置为离线模式
    MakeFastAPIOffline(app)
    # 运行FastAPI应用程序app，监听端口21003
    uvicorn.run(app, port=21003)
```