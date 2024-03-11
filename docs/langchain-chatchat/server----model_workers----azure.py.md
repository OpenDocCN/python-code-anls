# `.\Langchain-Chatchat\server\model_workers\azure.py`

```py
# 导入系统模块
import sys
# 导入操作系统模块
import os
# 从fastchat.conversation模块中导入Conversation类
from fastchat.conversation import Conversation
# 从server.model_workers.base模块中导入所有内容
from server.model_workers.base import *
# 从server.utils模块中导入get_httpx_client函数
from server.utils import get_httpx_client
# 从fastchat.conversation模块中导入conversation模块，并重命名为conv
from fastchat import conversation as conv
# 导入json模块
import json
# 从typing模块中导入List和Dict类
from typing import List, Dict
# 从configs模块中导入logger和log_verbose变量
from configs import logger, log_verbose

# 定义AzureWorker类，继承自ApiModelWorker类
class AzureWorker(ApiModelWorker):
    # 初始化方法
    def __init__(
            self,
            *,
            controller_addr: str = None,
            worker_addr: str = None,
            model_names: List[str] = ["azure-api"],
            version: str = "gpt-35-turbo",
            **kwargs,
    ):
        # 更新kwargs字典中的内容
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        # 调用父类ApiModelWorker的初始化方法
        super().__init__(**kwargs)
        # 设置版本号
        self.version = version
    # 执行聊天操作，接收参数为 ApiChatParams 对象，返回字典类型结果
    def do_chat(self, params: ApiChatParams) -> Dict:
        # 加载配置文件
        params.load_config(self.model_names[0])

        # 构建数据字典
        data = dict(
            messages=params.messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens if params.max_tokens else None,
            stream=True,
        )
        # 构建请求 URL
        url = ("https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}"
               .format(params.resource_name, params.deployment_name, params.api_version))
        # 构建请求头
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'api-key': params.api_key,
        }

        text = ""
        # 如果日志详细信息开启
        if log_verbose:
            # 打印请求 URL、请求头和数据
            logger.info(f'{self.__class__.__name__}:url: {url}')
            logger.info(f'{self.__class__.__name__}:headers: {headers}')
            logger.info(f'{self.__class__.__name__}:data: {data}')

        # 使用 HTTPX 客户端发送 POST 请求
        with get_httpx_client() as client:
            with client.stream("POST", url, headers=headers, json=data) as response:
                # 打印数据
                print(data)
                # 遍历响应内容的每一行
                for line in response.iter_lines():
                    # 如果行为空或包含 "[DONE]"，则跳过
                    if not line.strip() or "[DONE]" in line:
                        continue
                    # 如果行以 "data: " 开头，则去掉前缀
                    if line.startswith("data: "):
                        line = line[6:]
                    # 将行解析为 JSON 格式
                    resp = json.loads(line)
                    # 如果存在 choices 字段
                    if choices := resp["choices"]:
                        # 如果存在 delta 字段和 content 字段
                        if chunk := choices[0].get("delta", {}).get("content"):
                            # 将内容添加到 text 中
                            text += chunk
                            # 返回结果字典
                            yield {
                                    "error_code": 0,
                                    "text": text
                                }
                        # 打印文本内容
                        print(text)
                    else:
                        # 记录错误日志
                        self.logger.error(f"请求 Azure API 时发生错误：{resp}")

    # 获取嵌入向量
    def get_embeddings(self, params):
        # 打印信息
        print("embedding")
        # 打印参数
        print(params)
    # 定义一个方法，用于生成对话模板
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        # 创建一个对话对象，设置对话的名称、系统消息、消息列表、角色、分隔符和停止字符串
        return conv.Conversation(
            name=self.model_names[0],  # 设置对话的名称为模型名称列表的第一个元素
            system_message="You are a helpful, respectful and honest assistant.",  # 设置系统消息
            messages=[],  # 初始化消息列表为空
            roles=["user", "assistant"],  # 设置角色为用户和助手
            sep="\n### ",  # 设置分隔符
            stop_str="###",  # 设置停止字符串
        )
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 导入 uvicorn 模块
    import uvicorn
    # 从 server.utils 模块中导入 MakeFastAPIOffline 函数
    from server.utils import MakeFastAPIOffline
    # 从 fastchat.serve.base_model_worker 模块中导入 app 变量
    from fastchat.serve.base_model_worker import app

    # 创建 AzureWorker 实例，指定控制器地址和工作地址
    worker = AzureWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21008",
    )
    # 将 worker 实例添加到 fastchat.serve.model_worker 模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 将 FastAPI 应用设置为离线模式
    MakeFastAPIOffline(app)
    # 运行 FastAPI 应用，监听端口 21008
    uvicorn.run(app, port=21008)
```