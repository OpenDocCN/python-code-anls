# `MetaGPT\tests\metagpt\provider\test_spark_api.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : the unittest of spark api
# 导入 pytest 模块
import pytest
# 导入 CONFIG 配置
from metagpt.config import CONFIG
# 导入 spark_api 模块中的 GetMessageFromWeb 和 SparkLLM 类
from metagpt.provider.spark_api import GetMessageFromWeb, SparkLLM

# 设置 CONFIG 中的 spark_appid、spark_api_secret、spark_api_key、domain 和 spark_url
CONFIG.spark_appid = "xxx"
CONFIG.spark_api_secret = "xxx"
CONFIG.spark_api_key = "xxx"
CONFIG.domain = "xxxxxx"
CONFIG.spark_url = "xxxx"

# 设置提示消息和响应内容
prompt_msg = "who are you"
resp_content = "I'm Spark"

# 创建 MockWebSocketApp 类
class MockWebSocketApp(object):
    def __init__(self, ws_url, on_message=None, on_error=None, on_close=None, on_open=None):
        pass

    def run_forever(self, sslopt=None):
        pass

# 测试获取网络消息
def test_get_msg_from_web(mocker):
    # 使用 mocker 模块的 patch 方法模拟 WebSocketApp
    mocker.patch("websocket.WebSocketApp", MockWebSocketApp)

    # 创建 GetMessageFromWeb 对象
    get_msg_from_web = GetMessageFromWeb(text=prompt_msg)
    # 断言获取的参数中的 chat 的 domain 值为 "xxxxxx"
    assert get_msg_from_web.gen_params()["parameter"]["chat"]["domain"] == "xxxxxx"

    # 运行获取消息的方法
    ret = get_msg_from_web.run()
    # 断言返回结果为空字符串
    assert ret == ""

# 模拟 SparkLLM 类中的获取网络消息的方法
def mock_spark_get_msg_from_web_run(self) -> str:
    return resp_content

# 异步测试 SparkLLM 类中的方法
@pytest.mark.asyncio
async def test_spark_acompletion(mocker):
    # 使用 mocker 模块的 patch 方法模拟 GetMessageFromWeb 类中的 run 方法
    mocker.patch("metagpt.provider.spark_api.GetMessageFromWeb.run", mock_spark_get_msg_from_web_run)

    # 创建 SparkLLM 对象
    spark_gpt = SparkLLM()

    # 断言获取网络消息的方法返回结果为 resp_content
    resp = await spark_gpt.acompletion([])
    assert resp == resp_content

    # 断言获取网络消息的方法返回结果为 resp_content
    resp = await spark_gpt.aask(prompt_msg, stream=False)
    assert resp == resp_content

    # 断言获取网络消息的方法返回结果为 resp_content
    resp = await spark_gpt.acompletion_text([], stream=False)
    assert resp == resp_content

    # 断言获取网络消息的方法返回结果为 resp_content
    resp = await spark_gpt.acompletion_text([], stream=True)
    assert resp == resp_content

    # 断言获取网络消息的方法返回结果为 resp_content
    resp = await spark_gpt.aask(prompt_msg)
    assert resp == resp_content

```