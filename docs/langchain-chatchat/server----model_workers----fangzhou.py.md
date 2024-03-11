# `.\Langchain-Chatchat\server\model_workers\fangzhou.py`

```
# 从 fastchat.conversation 模块中导入 Conversation 类
from fastchat.conversation import Conversation
# 从 server.model_workers.base 模块中导入所有内容
from server.model_workers.base import *
# 从 fastchat 模块中导入 conversation 模块并重命名为 conv
from fastchat import conversation as conv
# 导入 sys 模块
import sys
# 从 typing 模块中导入 List, Literal, Dict 类型
from typing import List, Literal, Dict
# 从 configs 模块中导入 logger, log_verbose 变量
from configs import logger, log_verbose

# 定义 FangZhouWorker 类，继承自 ApiModelWorker 类
class FangZhouWorker(ApiModelWorker):
    """
    火山方舟
    """

    # 初始化方法
    def __init__(
            self,
            *,
            # 模型名称列表，默认为 ["fangzhou-api"]
            model_names: List[str] = ["fangzhou-api"],
            # 控制器地址，默认为 None
            controller_addr: str = None,
            # 工作地址，默认为 None
            worker_addr: str = None,
            # 版本，类型为 chatglm-6b-model，默认为 "chatglm-6b-model"
            version: Literal["chatglm-6b-model"] = "chatglm-6b-model",
            **kwargs,
    ):
        # 更新 kwargs 字典中的内容
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        # 如果 context_len 不存在于 kwargs 中，则设置默认值为 16384
        kwargs.setdefault("context_len", 16384)
        # 调用父类 ApiModelWorker 的初始化方法
        super().__init__(**kwargs)
        # 设置版本号
        self.version = version
    # 定义一个方法用于处理聊天请求，接收参数为 ApiChatParams 类型，返回一个字典
    def do_chat(self, params: ApiChatParams) -> Dict:
        # 导入 MaasService 类
        from volcengine.maas import MaasService

        # 加载配置文件
        params.load_config(self.model_names[0])
        # 创建 MaasService 实例
        maas = MaasService('maas-api.ml-platform-cn-beijing.volces.com', 'cn-beijing')
        # 设置 API 密钥
        maas.set_ak(params.api_key)
        # 设置密钥
        maas.set_sk(params.secret_key)

        # 准备请求数据
        req = {
            "model": {
                "name": params.version,
            },
            "parameters": {
                # 设置参数，例如最大生成标记数和温度
                "max_new_tokens": params.max_tokens,
                "temperature": params.temperature,
            },
            "messages": params.messages,
        }

        text = ""
        # 如果日志详细，则记录 MaasService 实例
        if log_verbose:
            self.logger.info(f'{self.__class__.__name__}:maas: {maas}')
        # 遍历 MaasService 实例的聊天流
        for resp in maas.stream_chat(req):
            # 如果存在错误
            if error := resp.error:
                # 如果错误代码大于 0
                if error.code_n > 0:
                    # 构建错误信息
                    data = {
                        "error_code": error.code_n,
                        "text": error.message,
                        "error": {
                            "message": error.message,
                            "type": "invalid_request_error",
                            "param": None,
                            "code": None,
                        }
                    }
                    # 记录错误信息
                    self.logger.error(f"请求方舟 API 时发生错误：{data}")
                    # 返回错误信息
                    yield data
                # 如果存在消息内容
                elif chunk := resp.choice.message.content:
                    # 拼接文本内容
                    text += chunk
                    # 返回成功信息
                    yield {"error_code": 0, "text": text}
            else:
                # 构建未知错误信息
                data = {
                    "error_code": 500,
                    "text": f"请求方舟 API 时发生未知的错误: {resp}"
                }
                # 记录未知错误信息
                self.logger.error(data)
                # 返回未知错误信息
                yield data
                break

    # 定义一个方法用于获取嵌入，接收参数 params
    def get_embeddings(self, params):
        # 打印信息
        print("embedding")
        # 打印参数
        print(params)
    # 创建一个对话模板的方法，返回一个对话对象
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        # 创建一个对话对象，设置名称为模型名称列表的第一个元素
        return conv.Conversation(
            name=self.model_names[0],
            # 设置系统消息，提示用户这是一个人工智能对话系统，可以提供有用、详细、礼貌的回答
            system_message="你是一个聪明、对人类有帮助的人工智能，你可以对人类提出的问题给出有用、详细、礼貌的回答。",
            # 初始化消息列表为空
            messages=[],
            # 设置对话角色为用户、助手和系统
            roles=["user", "assistant", "system"],
            # 设置消息分隔符为换行加上"### "
            sep="\n### ",
            # 设置停止字符串为"###"
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

    # 创建FangZhouWorker对象，指定controller_addr和worker_addr
    worker = FangZhouWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21005",
    )
    # 将worker对象添加到fastchat.serve.model_worker模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 将FastAPI应用程序app设置为离线模式
    MakeFastAPIOffline(app)
    # 运行FastAPI应用程序app，监听端口21005
    uvicorn.run(app, port=21005)
```