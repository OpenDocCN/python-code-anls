# `MetaGPT\tests\metagpt\provider\zhipuai\test_zhipu_model_api.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 说明文件的编码格式和描述信息

from typing import Any, Tuple  # 导入类型提示模块

import pytest  # 导入 pytest 模块
import zhipuai  # 导入 zhipuai 模块
from zhipuai.model_api.api import InvokeType  # 从 zhipuai.model_api.api 模块中导入 InvokeType 类
from zhipuai.utils.http_client import headers as zhipuai_default_headers  # 从 zhipuai.utils.http_client 模块中导入 headers 变量

from metagpt.provider.zhipuai.zhipu_model_api import ZhiPuModelAPI  # 从 metagpt.provider.zhipuai.zhipu_model_api 模块中导入 ZhiPuModelAPI 类

api_key = "xxx.xxx"  # 设置 API 密钥
zhipuai.api_key = api_key  # 将 API 密钥设置到 zhipuai 模块中

default_resp = b'{"result": "test response"}'  # 设置默认响应数据

# 模拟请求函数，返回默认响应数据
async def mock_requestor_arequest(self, **kwargs) -> Tuple[Any, Any, str]:
    return default_resp, None, None

# 异步测试函数
@pytest.mark.asyncio
async def test_zhipu_model_api(mocker):
    # 获取 ZhiPuModelAPI 的请求头
    header = ZhiPuModelAPI.get_header()
    zhipuai_default_headers.update({"Authorization": api_key})  # 更新默认请求头中的 Authorization 字段
    assert header == zhipuai_default_headers  # 断言请求头是否相等

    sse_header = ZhiPuModelAPI.get_sse_header()  # 获取 SSE 请求头
    assert len(sse_header["Authorization"]) == 191  # 断言 Authorization 字段的长度是否为 191

    # 分割 zhipu_api_url
    url_prefix, url_suffix = ZhiPuModelAPI.split_zhipu_api_url(InvokeType.SYNC, kwargs={"model": "chatglm_turbo"})
    assert url_prefix == "https://open.bigmodel.cn/api"  # 断言 url_prefix 是否符合预期
    assert url_suffix == "/paas/v3/model-api/chatglm_turbo/invoke"  # 断言 url_suffix 是否符合预期

    # 使用 mocker 模拟请求函数
    mocker.patch("metagpt.provider.general_api_requestor.GeneralAPIRequestor.arequest", mock_requestor_arequest)
    result = await ZhiPuModelAPI.arequest(
        InvokeType.SYNC, stream=False, method="get", headers={}, kwargs={"model": "chatglm_turbo"}
    )
    assert result == default_resp  # 断言请求结果是否符合预期

    result = await ZhiPuModelAPI.ainvoke()  # 调用 ainvoke 方法
    assert result["result"] == "test response"  # 断言返回结果中的 result 字段是否符合预期

```