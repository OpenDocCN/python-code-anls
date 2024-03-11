# `.\Langchain-Chatchat\server\model_workers\base.py`

```
# 从fastchat.conversation模块中导入Conversation类
from fastchat.conversation import Conversation
# 从configs模块中导入LOG_PATH和TEMPERATURE常量
from configs import LOG_PATH, TEMPERATURE
# 从fastchat.constants模块中导入LOGDIR常量
import fastchat.constants
# 将fastchat.constants模块中的LOGDIR常量设置为LOG_PATH的值
fastchat.constants.LOGDIR = LOG_PATH
# 从fastchat.serve.base_model_worker模块中导入BaseModelWorker类
from fastchat.serve.base_model_worker import BaseModelWorker
# 导入uuid模块
import uuid
# 导入json模块
import json
# 导入sys模块
import sys
# 从pydantic模块中导入BaseModel和root_validator装饰器
from pydantic import BaseModel, root_validator
# 导入fastchat模块
import fastchat
# 导入asyncio模块
import asyncio
# 从server.utils模块中导入get_model_worker_config函数
from server.utils import get_model_worker_config
# 从typing模块中导入Dict、List和Optional类
from typing import Dict, List, Optional

# 定义模块中可导出的类列表
__all__ = ["ApiModelWorker", "ApiChatParams", "ApiCompletionParams", "ApiEmbeddingsParams"]

# 定义ApiConfigParams类，继承自BaseModel类
class ApiConfigParams(BaseModel):
    '''
    在线API配置参数，未提供的值会自动从model_config.ONLINE_LLM_MODEL中读取
    '''
    # 定义可选的api_base_url参数，默认值为None
    api_base_url: Optional[str] = None
    # 定义可选的api_proxy参数，默认值为None
    api_proxy: Optional[str] = None
    # 定义可选的api_key参数，默认值为None
    api_key: Optional[str] = None
    # 定义可选的secret_key参数，默认值为None
    secret_key: Optional[str] = None
    # 定义可选的group_id参数，默认值为None，用于minimax
    group_id: Optional[str] = None # for minimax
    # 定义布尔类型的is_pro参数，默认值为False，用于minimax
    is_pro: bool = False # for minimax

    # 定义可选的APPID参数，默认值为None，用于xinghuo
    APPID: Optional[str] = None # for xinghuo
    # 定义可选的APISecret参数，默认值为None，用于xinghuo
    APISecret: Optional[str] = None # for xinghuo
    # 定义布尔类型的is_v2参数，默认值为False，用于xinghuo
    is_v2: bool = False # for xinghuo

    # 定义可选的worker_name参数，默认值为None
    worker_name: Optional[str] = None

    # 定义配置类的配置项，允许额外的字段
    class Config:
        extra = "allow"

    # 定义验证配置的根验证器
    @root_validator(pre=True)
    def validate_config(cls, v: Dict) -> Dict:
        # 如果存在worker_name字段对应的配置
        if config := get_model_worker_config(v.get("worker_name")):
            # 遍历类的所有字段
            for n in cls.__fields__:
                # 如果字段在配置中存在，则将配置中的值赋给输入的字典
                if n in config:
                    v[n] = config[n]
        return v

    # 加载配置信息的方法
    def load_config(self, worker_name: str):
        # 设置worker_name字段的值为传入的参数
        self.worker_name = worker_name
        # 如果存在worker_name字段对应的配置
        if config := get_model_worker_config(worker_name):
            # 遍历类的所有字段
            for n in self.__fields__:
                # 如果字段在配置中存在，则设置字段的值为配置中的值
                if n in config:
                    setattr(self, n, config[n])
        return self

# 定义ApiModelParams类，继承自ApiConfigParams类
class ApiModelParams(ApiConfigParams):
    '''
    模型配置参数
    '''
    # 定义可选的version参数，默认值为None
    version: Optional[str] = None
    # 定义可选的version_url参数，默认值为None
    version_url: Optional[str] = None
    # 定义可选的api_version参数，默认值为None，用于azure
    api_version: Optional[str] = None # for azure
    # 定义可选的deployment_name参数，默认值为None，用于azure
    deployment_name: Optional[str] = None # for azure
    # 定义可选的resource_name参数，默认值为None，用于azure

    temperature: float = TEMPERATURE
    # 定义可选的max_tokens参数，默认值为None
    max_tokens: Optional[int] = None
    # 定义一个可选的浮点数类型变量 top_p，并初始化为 1.0
    top_p: Optional[float] = 1.0
class ApiChatParams(ApiModelParams):
    '''
    chat请求参数
    '''
    # 定义包含消息字典的列表
    messages: List[Dict[str, str]]
    # 用于最小最大算法的系统消息
    system_message: Optional[str] = None # for minimax
    # 用于最小最大算法的角色元数据
    role_meta: Dict = {} # for minimax


class ApiCompletionParams(ApiModelParams):
    # 定义提示字符串
    prompt: str


class ApiEmbeddingsParams(ApiConfigParams):
    # 定义包含文本列表
    texts: List[str]
    # 嵌入模型名称，默认为None
    embed_model: Optional[str] = None
    # 用于最小最大算法的查询标志
    to_query: bool = False # for minimax


class ApiModelWorker(BaseModelWorker):
    # 默认嵌入模型名称为None，表示不支持嵌入
    DEFAULT_EMBED_MODEL: str = None

    def __init__(
        self,
        model_names: List[str],
        controller_addr: str = None,
        worker_addr: str = None,
        context_len: int = 2048,
        no_register: bool = False,
        **kwargs,
    ):
        # 设置默认的工作ID和模型路径
        kwargs.setdefault("worker_id", uuid.uuid4().hex[:8])
        kwargs.setdefault("model_path", "")
        kwargs.setdefault("limit_worker_concurrency", 5)
        # 调用父类的初始化方法
        super().__init__(model_names=model_names,
                        controller_addr=controller_addr,
                        worker_addr=worker_addr,
                        **kwargs)
        # 导入必要的模块
        import fastchat.serve.base_model_worker
        import sys
        # 设置日志记录器
        self.logger = fastchat.serve.base_model_worker.logger
        # 恢复被fastchat覆盖的标准输出
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        # 创建新的事件循环
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        # 设置上下文长度和并发限制
        self.context_len = context_len
        self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        self.version = None

        # 如果不禁止注册且有控制器地址，则初始化心跳
        if not no_register and self.controller_addr:
            self.init_heart_beat()


    def count_token(self, params):
        # 获取参数中的提示字符串
        prompt = params["prompt"]
        # 返回标记数和错误代码
        return {"count": len(str(prompt)), "error_code": 0}
    # 生成流式对话的方法，接受参数字典
    def generate_stream_gate(self, params: Dict):
        # 增加调用计数
        self.call_ct += 1

        try:
            # 获取参数中的提示信息
            prompt = params["prompt"]
            # 如果是对话模式，则将提示转换为消息列表，并验证消息
            if self._is_chat(prompt):
                messages = self.prompt_to_messages(prompt)
                messages = self.validate_messages(messages)
            else: # 使用chat模仿续写功能，不支持历史消息
                # 否则，创建一个包含提示信息的消息列表
                messages = [{"role": self.user_role, "content": f"please continue writing from here: {prompt}"}]

            # 构建调用Chat API所需的参数对象
            p = ApiChatParams(
                messages=messages,
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_new_tokens"),
                version=self.version,
            )
            # 调用Chat API，并逐个返回结果
            for resp in self.do_chat(p):
                yield self._jsonify(resp)
        except Exception as e:
            # 发生异常时返回错误信息
            yield self._jsonify({"error_code": 500, "text": f"{self.model_names[0]}请求API时发生错误：{e}"})

    # 生成对话的方法，接受参数
    def generate_gate(self, params):
        try:
            # 调用生成流式对话的方法，并处理结果
            for x in self.generate_stream_gate(params):
                ...
            # 将结果转换为JSON格式并返回
            return json.loads(x[:-1].decode())
        except Exception as e:
            # 发生异常时返回错误信息
            return {"error_code": 500, "text": str(e)}


    # 需要用户自定义的方法

    # 执行Chat的方法，接受参数对象，并返回字典形式的结果
    def do_chat(self, params: ApiChatParams) -> Dict:
        '''
        执行Chat的方法，默认使用模块里面的chat函数。
        要求返回形式：{"error_code": int, "text": str}
        '''
        return {"error_code": 500, "text": f"{self.model_names[0]}未实现chat功能"}

    # def do_completion(self, p: ApiCompletionParams) -> Dict:
    #     '''
    #     执行Completion的方法，默认使用模块里面的completion函数。
    #     要求返回形式：{"error_code": int, "text": str}
    #     '''
    #     return {"error_code": 500, "text": f"{self.model_names[0]}未实现completion功能"}
    # 执行Embeddings的方法，默认使用模块里面的embed_documents函数，返回指定格式的字典
    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        '''
        执行Embeddings的方法，默认使用模块里面的embed_documents函数。
        要求返回形式：{"code": int, "data": List[List[float]], "msg": str}
        '''
        return {"code": 500, "msg": f"{self.model_names[0]}未实现embeddings功能"}

    # 获取Embeddings，fastchat对LLM做Embeddings限制很大，似乎只能使用openai的
    def get_embeddings(self, params):
        # 打印获取Embeddings的信息
        print("get_embedding")
        print(params)

    # 创建对话模板，抛出未实现的错误
    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        raise NotImplementedError

    # 验证消息格式，有些API对messages有特殊格式，可以重写该函数替换默认的messages
    def validate_messages(self, messages: List[Dict]) -> List[Dict]:
        '''
        有些API对mesages有特殊格式，可以重写该函数替换默认的messages。
        之所以跟prompt_to_messages分开，是因为他们应用场景不同、参数不同
        '''
        return messages


    # 帮助方法
    @property
    def user_role(self):
        return self.conv.roles[0]

    @property
    def ai_role(self):
        return self.conv.roles[1]

    # 将chat函数返回的结果按照fastchat openai-api-server的格式返回
    def _jsonify(self, data: Dict) -> str:
        '''
        将chat函数返回的结果按照fastchat openai-api-server的格式返回
        '''
        return json.dumps(data, ensure_ascii=False).encode() + b"\0"

    # 检查prompt是否由chat messages拼接而来
    def _is_chat(self, prompt: str) -> bool:
        '''
        检查prompt是否由chat messages拼接而来
        TODO: 存在误判的可能，也许从fastchat直接传入原始messages是更好的做法
        '''
        key = f"{self.conv.sep}{self.user_role}:"
        return key in prompt
    # 将prompt字符串拆分成messages，并返回一个包含消息字典的列表
    def prompt_to_messages(self, prompt: str) -> List[Dict]:
        # 初始化结果列表
        result = []
        # 获取用户和AI的角色
        user_role = self.user_role
        ai_role = self.ai_role
        # 设置用户和AI消息的起始标记
        user_start = user_role + ":"
        ai_start = ai_role + ":"
        # 遍历prompt字符串中的每个消息
        for msg in prompt.split(self.conv.sep)[1:-1]:
            # 如果消息以用户角色开头
            if msg.startswith(user_start):
                # 提取消息内容并添加到结果列表中
                if content := msg[len(user_start):].strip():
                    result.append({"role": user_role, "content": content})
            # 如果消息以AI角色开头
            elif msg.startswith(ai_start):
                # 提取消息内容并添加到结果列表中
                if content := msg[len(ai_start):].strip():
                    result.append({"role": ai_role, "content": content})
            # 如果消息不以任何已知角色开头，抛出异常
            else:
                raise RuntimeError(f"unknown role in msg: {msg}")
        # 返回处理后的消息列表
        return result

    # 检查是否可以进行嵌入
    @classmethod
    def can_embedding(cls):
        # 返回默认嵌入模型是否存在的布尔值
        return cls.DEFAULT_EMBED_MODEL is not None
```