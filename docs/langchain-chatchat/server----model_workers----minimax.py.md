# `.\Langchain-Chatchat\server\model_workers\minimax.py`

```
# 从fastchat.conversation模块导入Conversation类
from fastchat.conversation import Conversation
# 从server.model_workers.base模块导入所有内容
from server.model_workers.base import *
# 从fastchat模块导入conversation别名为conv
from fastchat import conversation as conv
# 导入sys模块
import sys
# 导入json模块
import json
# 从server.model_workers.base模块导入ApiEmbeddingsParams类
from server.model_workers.base import ApiEmbeddingsParams
# 从server.utils模块导入get_httpx_client函数
from server.utils import get_httpx_client
# 导入List和Dict类型
from typing import List, Dict
# 从configs模块导入logger和log_verbose
from configs import logger, log_verbose

# 定义MiniMaxWorker类，继承自ApiModelWorker类
class MiniMaxWorker(ApiModelWorker):
    # 默认嵌入模型为"embo-01"
    DEFAULT_EMBED_MODEL = "embo-01"

    # 初始化方法，接收一些参数
    def __init__(
        self,
        *,
        model_names: List[str] = ["minimax-api"],
        controller_addr: str = None,
        worker_addr: str = None,
        version: str = "abab5.5-chat",
        **kwargs,
    ):
        # 更新kwargs字典中的model_names、controller_addr和worker_addr键值对
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        # 如果kwargs中没有"context_len"键，则设置默认值为16384
        kwargs.setdefault("context_len", 16384)
        # 调用父类ApiModelWorker的初始化方法，传入kwargs字典
        super().__init__(**kwargs)
        # 设置版本号为传入的version参数
        self.version = version

    # 验证消息的方法，接收消息列表，返回处理后的消息列表
    def validate_messages(self, messages: List[Dict]) -> List[Dict]:
        # 定义角色映射关系，将角色映射为发送者类型
        role_maps = {
            "USER": self.user_role,
            "assistant": self.ai_role,
            "system": "system",
        }
        # 将消息列表中的每个消息转换为包含发送者类型和文本内容的字典
        messages = [{"sender_type": role_maps[x["role"]], "text": x["content"]} for x in messages]
        # 返回处理后的消息列表
        return messages
    # 执行嵌入操作，接收参数并返回字典
    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        # 加载配置文件
        params.load_config(self.model_names[0])
        # 构建 API 请求的 URL
        url = f"https://api.minimax.chat/v1/embeddings?GroupId={params.group_id}"

        # 设置请求头
        headers = {
            "Authorization": f"Bearer {params.api_key}",
            "Content-Type": "application/json",
        }

        # 准备请求数据
        data = {
            "model": params.embed_model or self.DEFAULT_EMBED_MODEL,
            "texts": [],
            "type": "query" if params.to_query else "db",
        }
        # 如果启用了详细日志记录
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:data: {data}')
            logger.info(f'{self.__class__.__name__}:url: {url}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')

        # 使用 HTTPX 客户端发送 POST 请求
        with get_httpx_client() as client:
            result = []
            i = 0
            batch_size = 10
            # 分批处理文本数据
            while i < len(params.texts):
                texts = params.texts[i:i+batch_size]
                data["texts"] = texts
                # 发送 POST 请求并获取响应数据
                r = client.post(url, headers=headers, json=data).json()
                # 处理响应数据中的嵌入向量
                if embeddings := r.get("vectors"):
                    result += embeddings
                # 处理响应数据中的错误信息
                elif error := r.get("base_resp"):
                    data = {
                                "code": error["status_code"],
                                "msg": error["status_msg"],
                                "error": {
                                    "message":  error["status_msg"],
                                    "type": "invalid_request_error",
                                    "param": None,
                                    "code": None,
                                }
                            }
                    self.logger.error(f"请求 MiniMax API 时发生错误：{data}")
                    return data
                i += batch_size
            # 返回处理结果
            return {"code": 200, "data": result}

    # 获取嵌入向量
    def get_embeddings(self, params):
        print("embedding")
        print(params)
    # 定义一个方法用于生成对话模板，接受两个参数：对话模板和模型路径，返回一个Conversation对象
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        # 创建一个Conversation对象，设置名称为模型名称列表中的第一个名称
        # 设置系统消息为固定的提示信息
        # 设置消息列表为空
        # 设置角色为用户和机器人
        # 设置分隔符为换行符后跟"### "
        # 设置停止字符串为"###"
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是MiniMax自主研发的大型语言模型，回答问题简洁有条理。",
            messages=[],
            roles=["USER", "BOT"],
            sep="\n### ",
            stop_str="###",
        )
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 导入uvicorn模块
    import uvicorn
    # 从server.utils中导入MakeFastAPIOffline函数
    from server.utils import MakeFastAPIOffline
    # 从fastchat.serve.model_worker中导入app对象
    from fastchat.serve.model_worker import app

    # 创建MiniMaxWorker对象，指定controller_addr和worker_addr
    worker = MiniMaxWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21002",
    )
    # 将worker对象添加到fastchat.serve.model_worker模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 调用MakeFastAPIOffline函数，将app对象设置为离线状态
    MakeFastAPIOffline(app)
    # 运行FastAPI应用，监听端口21002
    uvicorn.run(app, port=21002)
```