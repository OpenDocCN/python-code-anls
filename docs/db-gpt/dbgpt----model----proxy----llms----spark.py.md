# `.\DB-GPT-src\dbgpt\model\proxy\llms\spark.py`

```py
# 导入需要的模块
import base64  # 提供将二进制数据编码为 ASCII 字符串的功能
import hashlib  # 提供常见哈希算法的功能，如 MD5、SHA1 等
import hmac  # 提供消息认证码算法的功能
import json  # 提供 JSON 数据的编解码功能
import os  # 提供与操作系统交互的功能
from concurrent.futures import Executor  # 提供异步执行任务的功能
from datetime import datetime  # 提供处理日期和时间的功能
from time import mktime  # 将时间转换为 POSIX 时间戳的功能
from typing import Iterator, Optional  # 提供类型提示，声明迭代器和可选类型
from urllib.parse import urlencode, urlparse  # 提供 URL 解析和编码的功能

# 导入 dbgpt 模块中的相关功能
from dbgpt.core import MessageConverter, ModelOutput, ModelRequest, ModelRequestContext
from dbgpt.model.parameter import ProxyModelParameters  # 导入代理模型的参数
from dbgpt.model.proxy.base import ProxyLLMClient  # 导入代理长文本模型的客户端
from dbgpt.model.proxy.llms.proxy_model import ProxyModel  # 导入代理模型

# 设置 Spark 默认的 API 版本
SPARK_DEFAULT_API_VERSION = "v3"


def getlength(text):
    # 计算文本列表中所有内容的总长度
    length = 0
    for content in text:
        temp = content["content"]  # 获取每个文本内容
        leng = len(temp)  # 计算每个文本内容的长度
        length += leng  # 累加长度到总长度
    return length


def checklen(text):
    # 当文本列表的总长度超过 8192 时，删除列表中的第一个元素，直到总长度小于等于 8192
    while getlength(text) > 8192:
        del text[0]  # 删除列表的第一个元素
    return text  # 返回处理后的文本列表


def spark_generate_stream(
    model: ProxyModel, tokenizer, params, device, context_len=2048
):
    # 使用 Spark 代理模型生成文本流
    client: SparkLLMClient = model.proxy_llm_client  # 获取 Spark 长文本模型的客户端
    context = ModelRequestContext(
        stream=True,
        user_name=params.get("user_name"),  # 获取请求中的用户名
        request_id=params.get("request_id"),  # 获取请求中的请求 ID
    )
    request = ModelRequest.build_request(
        client.default_model,  # 获取客户端的默认模型
        messages=params["messages"],  # 获取请求中的消息内容
        temperature=params.get("temperature"),  # 获取请求中的温度参数
        context=context,  # 设置请求的上下文
        max_new_tokens=params.get("max_new_tokens"),  # 获取请求中的最大新 token 数量
    )
    # 同步生成文本流，并逐条返回生成的文本
    for r in client.sync_generate_stream(request):
        yield r  # 返回生成的文本流的每一条文本


def get_response(request_url, data):
    # 通过 WebSocket 连接到请求 URL 并发送数据，接收并处理返回的数据
    from websockets.sync.client import connect  # 导入同步 WebSocket 客户端连接模块

    with connect(request_url) as ws:
        ws.send(json.dumps(data, ensure_ascii=False))  # 将数据转换为 JSON 字符串并发送
        result = ""
        while True:
            try:
                chunk = ws.recv()  # 接收返回的数据块
                response = json.loads(chunk)  # 解析 JSON 格式的返回数据
                print("look out the response: ", response)  # 打印返回的响应数据
                choices = response.get("payload", {}).get("choices", {})  # 获取响应中的选择内容
                if text := choices.get("text"):  # 获取选择内容中的文本
                    result += text[0]["content"]  # 将文本内容添加到结果中
                if choices.get("status") == 2:  # 如果响应状态为 2，表示结束
                    break  # 退出循环
            except Exception as e:
                raise e  # 抛出异常
    yield result  # 返回处理后的结果


class SparkAPI:
    def __init__(
        self, appid: str, api_key: str, api_secret: str, spark_url: str
    ) -> None:
        # 初始化 SparkAPI 类，设置所需的属性和 URL 信息
        self.appid = appid  # 设置应用程序 ID
        self.api_key = api_key  # 设置 API 密钥
        self.api_secret = api_secret  # 设置 API 密钥的密钥
        self.host = urlparse(spark_url).netloc  # 获取 Spark URL 的主机部分
        self.path = urlparse(spark_url).path  # 获取 Spark URL 的路径部分

        self.spark_url = spark_url  # 设置 Spark 的完整 URL
    def gen_url(self):
        from wsgiref.handlers import format_date_time

        # 生成当前时间的 RFC1123 格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 构建包含请求信息的原始签名字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 使用 HMAC-SHA256 算法对原始签名字符串进行加密
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        # 将加密后的签名数据转换为 Base64 字符串
        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding="utf-8")

        # 构建授权信息的原始字符串
        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        # 将授权信息的原始字符串转换为 Base64 字符串
        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )

        # 将授权信息和其他请求参数组合成字典
        v = {"authorization": authorization, "date": date, "host": self.host}

        # 构建完整的请求 URL
        url = self.spark_url + "?" + urlencode(v)

        # 返回生成的 URL，用于发起请求
        return url
# 定义 SparkLLMClient 类，继承自 ProxyLLMClient 类
class SparkLLMClient(ProxyLLMClient):
    
    # 初始化方法，设置各种参数
    def __init__(
        self,
        model: Optional[str] = None,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_base: Optional[str] = None,
        api_domain: Optional[str] = None,
        model_version: Optional[str] = None,
        model_alias: Optional[str] = "spark_proxyllm",
        context_length: Optional[int] = 4096,
        executor: Optional[Executor] = None,
    ):
        # 如果未指定 model_version，则尝试从环境变量 XUNFEI_SPARK_API_VERSION 中获取
        if not model_version:
            model_version = model or os.getenv("XUNFEI_SPARK_API_VERSION")
        
        # 如果未指定 api_base，则根据 model_version 设置默认值
        if not api_base:
            if model_version == SPARK_DEFAULT_API_VERSION:
                api_base = "ws://spark-api.xf-yun.com/v3.1/chat"
                domain = "generalv3"
            else:
                api_base = "ws://spark-api.xf-yun.com/v2.1/chat"
                domain = "generalv2"
            
            # 如果未指定 api_domain，则使用根据 model_version 设置的默认 domain
            if not api_domain:
                api_domain = domain
        
        # 设置实例变量
        self._model = model
        self._model_version = model_version
        self._api_base = api_base
        self._domain = api_domain
        self._app_id = app_id or os.getenv("XUNFEI_SPARK_APPID")
        self._api_secret = api_secret or os.getenv("XUNFEI_SPARK_API_SECRET")
        self._api_key = api_key or os.getenv("XUNFEI_SPARK_API_KEY")
        
        # 检查必要的参数是否都已设置，若未设置则抛出 ValueError
        if not self._app_id:
            raise ValueError("app_id can't be empty")
        if not self._api_key:
            raise ValueError("api_key can't be empty")
        if not self._api_secret:
            raise ValueError("api_secret can't be empty")
        
        # 调用父类 ProxyLLMClient 的初始化方法，传入 model_names、context_length 和 executor 参数
        super().__init__(
            model_names=[model, model_alias],
            context_length=context_length,
            executor=executor,
        )
    
    # 类方法，用于创建一个新的 SparkLLMClient 实例
    @classmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "SparkLLMClient":
        # 调用当前类的构造方法，传入 model_params 中的相关参数
        return cls(
            model=model_params.proxyllm_backend,
            app_id=model_params.proxy_api_app_id,
            api_key=model_params.proxy_api_key,
            api_secret=model_params.proxy_api_secret,
            api_base=model_params.proxy_api_base,
            model_alias=model_params.model_name,
            context_length=model_params.max_context_size,
            executor=default_executor,
        )
    
    # 属性方法，返回当前实例的默认模型名称
    @property
    def default_model(self) -> str:
        return self._model
    
    # 同步生成流方法，接受 ModelRequest 类型的 request 参数和可选的 message_converter 参数
    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> Iterator[ModelOutput]:
        # 对请求消息进行本地转换，并使用消息转换器进行转换处理
        request = self.local_covert_message(request, message_converter)
        
        # 将请求消息转换为通用消息，不支持系统角色
        messages = request.to_common_messages(support_system_role=False)
        
        # 获取请求的唯一标识符，如果不存在则使用默认值 "1"
        request_id = request.context.request_id or "1"
        
        # 构建数据字典，包括头部信息和参数信息
        data = {
            "header": {"app_id": self._app_id, "uid": request_id},
            "parameter": {
                "chat": {
                    "domain": self._domain,
                    "random_threshold": 0.5,
                    "max_tokens": request.max_new_tokens,
                    "auditing": "default",
                    "temperature": request.temperature,
                }
            },
            "payload": {"message": {"text": messages}},
        }

        # 创建 SparkAPI 对象，用于与 Spark 服务交互
        spark_api = SparkAPI(
            self._app_id, self._api_key, self._api_secret, self._api_base
        )
        
        # 生成请求的 URL 地址
        request_url = spark_api.gen_url()
        
        # 尝试发送请求并获取响应
        try:
            for text in get_response(request_url, data):
                # 生成模型输出对象，包括返回的文本和错误码
                yield ModelOutput(text=text, error_code=0)
        except Exception as e:
            # 若发生异常，返回带有错误信息的模型输出对象
            return ModelOutput(
                text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )
```