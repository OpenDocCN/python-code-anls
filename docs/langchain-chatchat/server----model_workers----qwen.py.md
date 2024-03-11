# `.\Langchain-Chatchat\server\model_workers\qwen.py`

```py
# 导入所需的模块
import json
import sys

# 导入自定义模块
from fastchat.conversation import Conversation
from configs import TEMPERATURE
from http import HTTPStatus
from typing import List, Literal, Dict

# 导入自定义模块并重命名
from fastchat import conversation as conv
from server.model_workers.base import *
from server.model_workers.base import ApiEmbeddingsParams
from configs import logger, log_verbose

# 定义 QwenWorker 类，继承自 ApiModelWorker 类
class QwenWorker(ApiModelWorker):
    # 默认的嵌入模型
    DEFAULT_EMBED_MODEL = "text-embedding-v1"

    # 初始化方法
    def __init__(
        self,
        *,
        version: Literal["qwen-turbo", "qwen-plus"] = "qwen-turbo",
        model_names: List[str] = ["qwen-api"],
        controller_addr: str = None,
        worker_addr: str = None,
        **kwargs,
    ):
        # 更新 kwargs 字典
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 16384)
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 设置版本号
        self.version = version
    # 定义一个方法用于处理聊天请求，接收一个ApiChatParams类型的参数，并返回一个字典
    def do_chat(self, params: ApiChatParams) -> Dict:
        # 导入dashscope模块
        import dashscope
        # 加载配置文件
        params.load_config(self.model_names[0])
        # 如果日志详细信息开启，则记录参数信息
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:params: {params}')

        # 创建一个Generation对象
        gen = dashscope.Generation()
        # 调用Generation对象的call方法，传入参数并获取响应
        responses = gen.call(
            model=params.version,
            temperature=params.temperature,
            api_key=params.api_key,
            messages=params.messages,
            result_format='message',  # 设置结果为消息格式
            stream=True,
        )

        # 遍历响应列表
        for resp in responses:
            # 如果响应状态码为200
            if resp["status_code"] == 200:
                # 如果存在choices，则返回第一个choice的消息内容
                if choices := resp["output"]["choices"]:
                    yield {
                        "error_code": 0,
                        "text": choices[0]["message"]["content"],
                    }
            else:
                # 构建错误信息字典
                data = {
                    "error_code": resp["status_code"],
                    "text": resp["message"],
                    "error": {
                        "message": resp["message"],
                        "type": "invalid_request_error",
                        "param": None,
                        "code": None,
                    }
                }
                # 记录错误信息
                self.logger.error(f"请求千问 API 时发生错误：{data}")
                yield data
    # 执行嵌入操作，接收参数并返回字典
    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        # 导入dashscope模块
        import dashscope
        # 加载配置文件
        params.load_config(self.model_names[0])
        # 如果日志详细，则记录参数信息
        if log_verbose:
            logger.info(f'{self.__class__.__name__}:params: {params}')
        # 初始化结果列表
        result = []
        # 初始化索引
        i = 0
        # 循环处理文本
        while i < len(params.texts):
            # 每次处理最多25行文本
            texts = params.texts[i:i+25]
            # 调用TextEmbedding的call方法
            resp = dashscope.TextEmbedding.call(
                model=params.embed_model or self.DEFAULT_EMBED_MODEL,
                input=texts, # 最大25行
                api_key=params.api_key,
            )
            # 如果响应状态码不是200，则返回错误信息
            if resp["status_code"] != 200:
                data = {
                            "code": resp["status_code"],
                            "msg": resp.message,
                            "error": {
                                "message": resp["message"],
                                "type": "invalid_request_error",
                                "param": None,
                                "code": None,
                            }
                        }
                self.logger.error(f"请求千问 API 时发生错误：{data}")
                return data
            else:
                # 提取嵌入向量
                embeddings = [x["embedding"] for x in resp["output"]["embeddings"]]
                result += embeddings
            i += 25
        # 返回成功的结果字典
        return {"code": 200, "data": result}

    # 获取嵌入向量
    def get_embeddings(self, params):
        print("embedding")
        print(params)

    # 创建对话模板
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        # 返回一个Conversation对象
        return conv.Conversation(
            name=self.model_names[0],
            system_message="你是一个聪明、对人类有帮助的人工智能，你可以对人类提出的问题给出有用、详细、礼貌的回答。",
            messages=[],
            roles=["user", "assistant", "system"],
            sep="\n### ",
            stop_str="###",
        )
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 导入uvicorn模块
    import uvicorn
    # 从server.utils模块中导入MakeFastAPIOffline函数
    from server.utils import MakeFastAPIOffline
    # 从fastchat.serve.model_worker模块中导入app变量
    from fastchat.serve.model_worker import app

    # 创建QwenWorker对象，指定controller_addr和worker_addr
    worker = QwenWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:20007",
    )
    # 将worker对象添加到fastchat.serve.model_worker模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 将FastAPI应用程序app设置为离线模式
    MakeFastAPIOffline(app)
    # 运行FastAPI应用程序app，监听端口20007
    uvicorn.run(app, port=20007)
```