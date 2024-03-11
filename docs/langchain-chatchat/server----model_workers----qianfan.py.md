# `.\Langchain-Chatchat\server\model_workers\qianfan.py`

```
import sys
from fastchat.conversation import Conversation
from server.model_workers.base import *
from server.utils import get_httpx_client
from cachetools import cached, TTLCache
import json
from fastchat import conversation as conv
from sys import path
from server.model_workers.base import ApiEmbeddingsParams
from typing import List, Literal, Dict
from configs import logger, log_verbose

# 定义模型版本与对应的模型名称
MODEL_VERSIONS = {
    "ernie-bot-4": "completions_pro",
    "ernie-bot": "completions",
    "ernie-bot-turbo": "eb-instant",
    "bloomz-7b": "bloomz_7b1",
    "qianfan-bloomz-7b-c": "qianfan_bloomz_7b_compressed",
    "llama2-7b-chat": "llama_2_7b",
    "llama2-13b-chat": "llama_2_13b",
    "llama2-70b-chat": "llama_2_70b",
    "qianfan-llama2-ch-7b": "qianfan_chinese_llama_2_7b",
    "chatglm2-6b-32k": "chatglm2_6b_32k",
    "aquilachat-7b": "aquilachat_7b",
    # 以下模型暂未发布
    # "linly-llama2-ch-7b": "",
    # "linly-llama2-ch-13b": "",
    # "chatglm2-6b": "",
    # "chatglm2-6b-int4": "",
    # "falcon-7b": "",
    # "falcon-180b-chat": "",
    # "falcon-40b": "",
    # "rwkv4-world": "",
    # "rwkv5-world": "",
    # "rwkv4-pile-14b": "",
    # "rwkv4-raven-14b": "",
    # "open-llama-7b": "",
    # "dolly-12b": "",
    # "mpt-7b-instruct": "",
    # "mpt-30b-instruct": "",
    # "OA-Pythia-12B-SFT-4": "",
    # "xverse-13b": "",

    # 以下为企业测试，需要单独申请
    # "flan-ul2": "",
    # "Cerebras-GPT-6.7B": ""
    # "Pythia-6.9B": ""
}

@cached(TTLCache(1, 1800))  # 缓存函数调用结果，每30分钟刷新一次
def get_baidu_access_token(api_key: str, secret_key: str) -> str:
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key}
    # 尝试使用 HTTPX 客户端发送 GET 请求获取数据
    try:
        # 使用 get_httpx_client() 函数获取 HTTPX 客户端，并使用 with 语句管理资源
        with get_httpx_client() as client:
            # 发送 GET 请求到指定 URL，传入参数 params，并将响应转换为 JSON 格式，获取其中的 "access_token" 字段
            return client.get(url, params=params).json().get("access_token")
    # 捕获任何异常并将其存储在变量 e 中
    except Exception as e:
        # 打印错误信息，指示从百度获取令牌失败，并显示具体的异常信息
        print(f"failed to get token from baidu: {e}")
# 定义一个名为 QianFanWorker 的类，继承自 ApiModelWorker 类
class QianFanWorker(ApiModelWorker):
    """
    百度千帆
    """
    # 默认的嵌入模型为 "embedding-v1"
    DEFAULT_EMBED_MODEL = "embedding-v1"

    # 初始化方法，接收一些参数
    def __init__(
            self,
            *,
            version: Literal["ernie-bot", "ernie-bot-turbo"] = "ernie-bot",
            model_names: List[str] = ["qianfan-api"],
            controller_addr: str = None,
            worker_addr: str = None,
            **kwargs,
    ):
        # 更新 kwargs 字典中的 model_names、controller_addr 和 worker_addr 键值对
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        # 如果 kwargs 中没有 "context_len" 键，则设置其默认值为 16384
        kwargs.setdefault("context_len", 16384)
        # 调用父类的初始化方法，传入 kwargs 中的参数
        super().__init__(**kwargs)
        # 设置类属性 version 的值为传入的 version 参数
        self.version = version
    # 执行嵌入操作，返回嵌入结果字典
    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        # 加载配置文件
        params.load_config(self.model_names[0])
        # import qianfan

        # 创建嵌入对象
        # embed = qianfan.Embedding(ak=params.api_key, sk=params.secret_key)
        # 调用嵌入对象的方法，获取嵌入结果
        # resp = embed.do(texts = params.texts, model=params.embed_model or self.DEFAULT_EMBED_MODEL)
        # 如果响应状态码为 200
        # if resp.code == 200:
        #     从响应数据中提取嵌入结果
        #     embeddings = [x.embedding for x in resp.body.get("data", [])]
        #     返回包含状态码和嵌入结果的字典
        #     return {"code": 200, "embeddings": embeddings}
        # 否则
        # else:
        #     返回包含错误码和错误消息的字典
        #     return {"code": resp.code, "msg": str(resp.body)}

        # 获取嵌入模型
        embed_model = params.embed_model or self.DEFAULT_EMBED_MODEL
        # 获取百度访问令牌
        access_token = get_baidu_access_token(params.api_key, params.secret_key)
        # 构建请求 URL
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/{embed_model}?access_token={access_token}"
        # 如果日志详细信息开启
        if log_verbose:
            # 记录请求 URL
            logger.info(f'{self.__class__.__name__}:url: {url}')

        # 使用 HTTPX 客户端发送请求
        with get_httpx_client() as client:
            # 初始化结果列表
            result = []
            # 初始化索引和批处理大小
            i = 0
            batch_size = 10
            # 循环处理文本列表
            while i < len(params.texts):
                # 获取当前批次的文本
                texts = params.texts[i:i + batch_size]
                # 发送 POST 请求，获取响应数据
                resp = client.post(url, json={"input": texts}).json()
                # 如果响应中包含错误码
                if "error_code" in resp:
                    # 构建错误信息字典
                    data = {
                        "code": resp["error_code"],
                        "msg": resp["error_msg"],
                        "error": {
                            "message": resp["error_msg"],
                            "type": "invalid_request_error",
                            "param": None,
                            "code": None,
                        }
                    }
                    # 记录错误信息
                    self.logger.error(f"请求千帆 API 时发生错误：{data}")
                    # 返回错误信息字典
                    return data
                # 否则
                else:
                    # 提取嵌入结果并添加到结果列表中
                    embeddings = [x["embedding"] for x in resp.get("data", [])]
                    result += embeddings
                # 更新索引
                i += batch_size
            # 返回包含状态码和数据结果的字典
            return {"code": 200, "data": result}
    # 定义一个方法用于获取嵌入向量，接受参数params
    def get_embeddings(self, params):
        # 打印信息，表示正在进行嵌入操作
        print("embedding")
        # 打印参数params的值
        print(params)

    # 定义一个方法用于创建对话模板，接受两个参数：conv_template和model_path
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        # 返回一个对话对象，包括名称、系统消息、消息列表、角色、分隔符和停止字符串
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

    # 创建QianFanWorker对象，指定controller_addr和worker_addr
    worker = QianFanWorker(
        controller_addr="http://127.0.0.1:20001",
        worker_addr="http://127.0.0.1:21004"
    )
    # 将worker对象添加到fastchat.serve.model_worker模块中
    sys.modules["fastchat.serve.model_worker"].worker = worker
    # 调用MakeFastAPIOffline函数，将app设置为离线状态
    MakeFastAPIOffline(app)
    # 运行FastAPI应用app，监听端口21004
    uvicorn.run(app, port=21004)
```