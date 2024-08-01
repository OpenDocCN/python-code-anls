# `.\DB-GPT-src\dbgpt\model\proxy\llms\baichuan.py`

```py
import json  # 导入处理 JSON 数据的模块
from typing import List  # 引入类型提示模块中的 List 类型

import requests  # 导入用于发送 HTTP 请求的 requests 库

from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType  # 导入模型消息相关的类和枚举
from dbgpt.model.parameter import ProxyModelParameters  # 导入代理模型参数类
from dbgpt.model.proxy.llms.proxy_model import ProxyModel  # 导入代理模型类

BAICHUAN_DEFAULT_MODEL = "Baichuan2-Turbo-192k"  # 定义默认的百川模型名称


def baichuan_generate_stream(
    model: ProxyModel, tokenizer=None, params=None, device=None, context_len=4096
):
    # TODO: Support new Baichuan ProxyLLMClient
    url = "https://api.baichuan-ai.com/v1/chat/completions"  # 设置请求的百川 API URL

    model_params = model.get_params()  # 获取模型的参数
    model_name = model_params.proxyllm_backend or BAICHUAN_DEFAULT_MODEL  # 获取使用的百川模型名称
    proxy_api_key = model_params.proxy_api_key  # 获取百川 API 的密钥

    history = []  # 初始化一个空列表，用于存储历史消息
    messages: List[ModelMessage] = params["messages"]  # 获取传入参数中的消息列表

    # 添加历史对话记录
    for message in messages:
        if message.role == ModelMessageRoleType.HUMAN:
            history.append({"role": "user", "content": message.content})  # 用户消息
        elif message.role == ModelMessageRoleType.SYSTEM:
            # 今天系统消息暂不支持
            history.append({"role": "user", "content": message.content})  # 系统消息
        elif message.role == ModelMessageRoleType.AI:
            history.append({"role": "assistant", "content": message.content})  # 助手消息
        else:
            pass  # 其他角色的消息（如未知角色）

    payload = {
        "model": model_name,  # 指定使用的模型名称
        "messages": history,  # 传入的历史消息
        "temperature": params.get("temperature", 0.3),  # 获取温度参数，默认为0.3
        "top_k": params.get("top_k", 5),  # 获取 top_k 参数，默认为5
        "top_p": params.get("top_p", 0.85),  # 获取 top_p 参数，默认为0.85
        "stream": True,  # 使用流式生成
    }

    headers = {
        "Content-Type": "application/json",  # 指定请求内容类型为 JSON
        "Authorization": "Bearer " + proxy_api_key,  # 设置授权头部，使用百川 API 密钥
    }

    print(f"Sending request to {url} with model {model_name}")  # 打印发送请求的 URL 和模型名称
    res = requests.post(url=url, json=payload, headers=headers)  # 发送 POST 请求

    text = ""  # 初始化空字符串，用于存储结果文本
    for line in res.iter_lines():  # 遍历响应的每一行
        if line:  # 如果行不为空
            if not line.startswith(b"data: "):  # 如果不是以"data: "开头的行
                error_message = line.decode("utf-8")  # 解码错误消息
                yield error_message  # 返回错误消息
            else:  # 如果是以"data: "开头的行
                json_data = line.split(b": ", 1)[1]  # 切分出 JSON 数据部分
                decoded_line = json_data.decode("utf-8")  # 解码 JSON 数据
                if decoded_line.lower() != "[DONE]".lower():  # 如果不是"[DONE]"
                    obj = json.loads(json_data)  # 解析 JSON 数据
                    if obj["choices"][0]["delta"].get("content") is not None:
                        content = obj["choices"][0]["delta"].get("content")  # 获取内容
                        text += content  # 将内容追加到文本中
                yield text  # 返回生成的文本


def main():
    model_params = ProxyModelParameters(
        model_name="not-used",
        model_path="not-used",
        proxy_server_url="not-used",
        proxy_api_key="YOUR_BAICHUAN_API_KEY",  # 设置百川 API 密钥
        proxyllm_backend="Baichuan2-Turbo-192k",  # 设置百川模型名称
    )
    final_text = ""  # 初始化最终文本为空字符串
    for part in baichuan_generate_stream(
        model=ProxyModel(model_params=model_params),  # 使用代理模型初始化
        params={
            "messages": [
                ModelMessage(role=ModelMessageRoleType.HUMAN, content="背诵《论语》第一章")
            ]  # 设置消息参数，包含用户消息
        },
    ):
        final_text = part  # 遍历生成的文本部分，存储到最终文本中
    # 输出最终文本内容到控制台
    print(final_text)
# 如果当前脚本作为主程序运行，则执行 main() 函数
if __name__ == "__main__":
    main()
```