# `.\Langchain-Chatchat\server\model_workers\gemini.py`

```
# 导入所需的模块
import sys
from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from fastchat import conversation as conv
import json, httpx
from typing import List, Dict
from configs import logger, log_verbose

# 定义 GeminiWorker 类，继承自 ApiModelWorker 类
class GeminiWorker(ApiModelWorker):
    # 初始化方法，接收一些参数并设置默认值
    def __init__(
            self,
            *,
            controller_addr: str = None,
            worker_addr: str = None,
            model_names: List[str] = ["gemini-api"],
            **kwargs,
    ):
        # 更新参数字典
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 4096)
        # 调用父类的初始化方法
        super().__init__(**kwargs)

    # 创建 Gemini 消息的方法，接收消息列表并返回 JSON 格式的消息
    def create_gemini_messages(self, messages) -> json:
        # 检查是否有历史消息
        has_history = any(msg['role'] == 'assistant' for msg in messages)
        gemini_msg = []

        # 遍历消息列表
        for msg in messages:
            role = msg['role']
            content = msg['content']
            # 如果消息角色为系统，则跳过
            if role == 'system':
                continue
            # 如果有历史消息
            if has_history:
                if role == 'assistant':
                    role = "model"
                transformed_msg = {"role": role, "parts": [{"text": content}]}
            else:
                if role == 'user':
                    transformed_msg = {"parts": [{"text": content}]}

            gemini_msg.append(transformed_msg)

        # 构建消息字典
        msg = dict(contents=gemini_msg)
        return msg

    # 获取嵌入向量的方法，打印参数
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 创建对话模板的方法，接收模板和模型路径并返回 Conversation 对象
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        return conv.Conversation(
            name=self.model_names[0],
            system_message="You are a helpful, respectful and honest assistant.",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )

# 如果作为主程序运行，则导入 uvicorn 和 MakeFastAPIOffline 方法
if __name__ == "__main__":
    import uvicorn
    from server.utils import MakeFastAPIOffline
    # 导入 fastchat.serve.base_model_worker 模块中的 app 对象
    from fastchat.serve.base_model_worker import app
    
    # 创建 GeminiWorker 对象，指定控制器地址和工作地址
    worker = GeminiWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21012",
    )
    
    # 将 GeminiWorker 对象设置为 fastchat.serve.model_worker 模块中的 worker 对象
    sys.modules["fastchat.serve.model_worker"].worker = worker
    
    # 将 app 对象传入 MakeFastAPIOffline 函数中，使 FastAPI 应用离线
    MakeFastAPIOffline(app)
    
    # 运行 FastAPI 应用，监听端口为 21012
    uvicorn.run(app, port=21012)
```