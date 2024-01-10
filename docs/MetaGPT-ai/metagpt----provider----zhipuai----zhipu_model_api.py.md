# `MetaGPT\metagpt\provider\zhipuai\zhipu_model_api.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : zhipu model api to support sync & async for invoke & sse_invoke

# 导入 json 模块
import json

# 导入 zhipuai 模块
import zhipuai
# 从 zhipuai.model_api.api 模块中导入 InvokeType 和 ModelAPI 类
from zhipuai.model_api.api import InvokeType, ModelAPI
# 从 zhipuai.utils.http_client 模块中导入 headers
from zhipuai.utils.http_client import headers as zhipuai_default_headers
# 从 metagpt.provider.general_api_requestor 模块中导入 GeneralAPIRequestor 类
from metagpt.provider.general_api_requestor import GeneralAPIRequestor
# 从 metagpt.provider.zhipuai.async_sse_client 模块中导入 AsyncSSEClient 类
from metagpt.provider.zhipuai.async_sse_client import AsyncSSEClient

# 定义 ZhiPuModelAPI 类，继承自 ModelAPI 类
class ZhiPuModelAPI(ModelAPI):
    # 定义 get_header 方法，返回请求头字典
    @classmethod
    def get_header(cls) -> dict:
        # 生成 token
        token = cls._generate_token()
        # 更新默认请求头中的 Authorization 字段
        zhipuai_default_headers.update({"Authorization": token})
        return zhipuai_default_headers

    # 定义 get_sse_header 方法，返回 SSE 请求头字典
    @classmethod
    def get_sse_header(cls) -> dict:
        # 生成 token
        token = cls._generate_token()
        # 构建包含 Authorization 字段的请求头字典
        headers = {"Authorization": token}
        return headers

    # 定义 split_zhipu_api_url 方法，拆分 zhipu api 的 URL
    @classmethod
    def split_zhipu_api_url(cls, invoke_type: InvokeType, kwargs):
        # 使用此方法防止 zhipu api 升级到不同版本，并遵循基于 openai sdk 实现的 GeneralAPIRequestor
        zhipu_api_url = cls._build_api_url(kwargs, invoke_type)
        """
            example:
                zhipu_api_url: https://open.bigmodel.cn/api/paas/v3/model-api/{model}/{invoke_method}
        """
        arr = zhipu_api_url.split("/api/")
        # 返回拆分后的 URL 部分
        return f"{arr[0]}/api", f"/{arr[1]}"

    # 定义 arequest 方法，进行异步请求
    @classmethod
    async def arequest(cls, invoke_type: InvokeType, stream: bool, method: str, headers: dict, kwargs):
        # TODO to make the async request to be more generic for models in http mode.
        assert method in ["post", "get"]

        # 拆分 zhipu api 的 URL
        base_url, url = cls.split_zhipu_api_url(invoke_type, kwargs)
        # 创建 GeneralAPIRequestor 实例
        requester = GeneralAPIRequestor(base_url=base_url)
        # 发起异步请求
        result, _, api_key = await requester.arequest(
            method=method,
            url=url,
            headers=headers,
            stream=stream,
            params=kwargs,
            request_timeout=zhipuai.api_timeout_seconds,
        )
        return result

    # 定义 ainvoke 方法，进行异步调用
    @classmethod
    async def ainvoke(cls, **kwargs) -> dict:
        """async invoke different from raw method `async_invoke` which get the final result by task_id"""
        # 获取请求头
        headers = cls.get_header()
        # 发起异步请求
        resp = await cls.arequest(
            invoke_type=InvokeType.SYNC, stream=False, method="post", headers=headers, kwargs=kwargs
        )
        # 解码响应数据
        resp = resp.decode("utf-8")
        # 将响应数据转换为字典
        resp = json.loads(resp)
        return resp

    # 定义 asse_invoke 方法，进行异步 SSE 调用
    @classmethod
    async def asse_invoke(cls, **kwargs) -> AsyncSSEClient:
        """async sse_invoke"""
        # 获取 SSE 请求头
        headers = cls.get_sse_header()
        # 返回异步 SSE 客户端
        return AsyncSSEClient(
            await cls.arequest(invoke_type=InvokeType.SSE, stream=True, method="post", headers=headers, kwargs=kwargs)
        )

```