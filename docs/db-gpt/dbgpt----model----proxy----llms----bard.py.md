# `.\DB-GPT-src\dbgpt\model\proxy\llms\bard.py`

```py
# 引入需要的模块和类
from typing import List  # 引入类型提示中的 List 类型

import requests  # 引入用于发送 HTTP 请求的 requests 库

# 从 dbgpt.core.interface.message 模块中导入 ModelMessage 和 ModelMessageRoleType 类
from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType

# 从 dbgpt.model.proxy.llms.proxy_model 模块中导入 ProxyModel 类
from dbgpt.model.proxy.llms.proxy_model import ProxyModel


def bard_generate_stream(
    model: ProxyModel,  # 传入的模型对象，类型为 ProxyModel
    tokenizer,  # tokenizer 对象，用于文本分词
    params,  # 参数字典，包含各种配置参数
    device,  # 设备参数，表示运行模型的设备
    context_len=2048  # 上下文长度，默认为 2048
):
    # TODO: 支持新的 bard ProxyLLMClient

    # 获取模型的参数信息
    model_params = model.get_params()
    print(f"Model: {model}, model_params: {model_params}")

    # 获取代理模型的 API 密钥和服务器 URL
    proxy_api_key = model_params.proxy_api_key
    proxy_server_url = model_params.proxy_server_url

    # 获取是否需要转换为兼容格式的配置，默认为 False
    convert_to_compatible_format = params.get("convert_to_compatible_format", False)

    # 初始化一个空的历史消息列表
    history = []

    # 从参数中获取消息列表
    messages: List[ModelMessage] = params["messages"]

    # 遍历消息列表
    for message in messages:
        # 根据消息的角色类型，将消息内容添加到历史记录中
        if message.role == ModelMessageRoleType.HUMAN:
            history.append({"role": "user", "content": message.content})
        elif message.role == ModelMessageRoleType.SYSTEM:
            history.append({"role": "system", "content": message.content})
        elif message.role == ModelMessageRoleType.AI:
            history.append({"role": "assistant", "content": message.content})
        else:
            pass  # 忽略其他类型的消息角色

    # 如果需要转换为兼容格式
    if convert_to_compatible_format:
        # 找到最后一个用户输入的索引
        last_user_input_index = None
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "user":
                last_user_input_index = i
                break
        # 如果找到了最后一个用户输入的索引
        if last_user_input_index:
            # 弹出最后一个用户输入的消息，并将其放回历史记录的最后
            last_user_input = history.pop(last_user_input_index)
            history.append(last_user_input)

    # 提取历史消息中的内容部分，并组成列表
    msgs = []
    for msg in history:
        if msg.get("content"):
            msgs.append(msg["content"])

    # 如果存在代理服务器的 URL
    if proxy_server_url is not None:
        # 设置 HTTP 请求的头部信息
        headers = {"Content-Type": "application/json"}
        # 构造请求的数据载荷
        payloads = {"input": "\n".join(msgs)}
        # 发送 POST 请求到代理服务器
        response = requests.post(
            proxy_server_url, headers=headers, json=payloads, stream=False
        )
        # 如果请求成功，则生成响应的文本内容
        if response.ok:
            yield response.text
        else:
            # 如果请求失败，则生成错误信息
            yield f"bard proxy url request failed!, response = {str(response)}"
    else:
        # 否则，使用默认的 Bard API 客户端
        import bardapi  # 导入 Bard API 客户端库

        # 调用 Bard API 获取回答
        response = bardapi.core.Bard(proxy_api_key).get_answer("\n".join(msgs))

        # 如果响应不为空并且包含内容，则生成内容的字符串表示
        if response is not None and response.get("content") is not None:
            yield str(response["content"])
        else:
            # 否则，生成 Bard 响应错误的信息
            yield f"bard response error: {str(response)}"
```