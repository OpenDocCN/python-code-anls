# `.\ollama_knowlage_graph\ollama\client.py`

```
# 导入必要的库
import os  # 用于操作系统相关功能
import json  # 用于 JSON 数据的解析和处理
import requests  # 用于发送 HTTP 请求

# 从环境变量中获取基本 URL，如果没有设置则默认为 http://localhost:11434
BASE_URL = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

# 使用提供的模型为给定的提示生成响应。这是一个流式端点，因此会产生一系列响应。
# 最终的响应对象将包括来自请求的统计数据和附加数据。使用回调函数覆盖
# 默认处理函数。
def generate(model_name, prompt, system=None, template=None, context=None, options=None, callback=None):
    try:
        # 构建 API 请求的 URL
        url = f"{BASE_URL}/api/generate"
        
        # 构建 API 请求的 payload
        payload = {
            "model": model_name, 
            "prompt": prompt, 
            "system": system, 
            "template": template, 
            "context": context, 
            "options": options
        }
        
        # 删除 payload 中值为 None 的键值对
        payload = {k: v for k, v in payload.items() if v is not None}
        
        # 发送 POST 请求到指定 URL，接收流式响应
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            
            # 创建一个变量来保存最终块的上下文历史记录
            final_context = None
            
            # 如无回调，则用于保存串联响应字符串的变量
            full_response = ""

            # 逐行遍历回复并显示详情
            for line in response.iter_lines():
                if line:
                    # 解析每一行（JSON 块）并提取详细信息
                    chunk = json.loads(line)
                    
                    # 如果提供了回调函数，则使用块作为参数调用回调函数
                    if callback:
                        callback(chunk)
                    else:
                        # 如果这不是最后一个块，将“response”字段值添加到 full_response 并打印它
                        if not chunk.get("done"):
                            response_piece = chunk.get("response", "")
                            full_response += response_piece
                            print(response_piece, end="", flush=True)
                    
                    # 检查是否为最后一块（done 为 true）
                    if chunk.get("done"):
                        final_context = chunk.get("context")
            
            # 返回完整的响应和最终上下文
            return full_response, final_context
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

# 从 Modelfile 创建模型。使用回调函数覆盖默认处理程序。
def create(model_name, model_path, callback=None):
    try:
        # 构建 API 请求的 URL
        url = f"{BASE_URL}/api/create"
        
        # 构建 API 请求的 payload
        payload = {"name": model_name, "path": model_path}
        
        # 发出 POST 请求，并将流参数设置为 True，以处理流式响应
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # 逐行遍历回复并显示状态
            for line in response.iter_lines():
                if line:
                    # 解析每一行（JSON 块）并提取状态
                    chunk = json.loads(line)

                    if callback:
                        callback(chunk)
                    else:
                        print(f"Status: {chunk.get('status')}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# 从模型注册表中提取模型。取消的拉取将从原处恢复，并且多个
# 调用将共享相同的下载进度。使用回调函数覆盖默认处理程序。
def pull(model_name, insecure=False, callback=None):
    try:
        # 构建完整的 URL，用于向基础 URL 发送 API 拉取请求
        url = f"{BASE_URL}/api/pull"
        # 准备要发送的数据负载，包括模型名称和是否不安全标志
        payload = {
            "name": model_name,
            "insecure": insecure
        }

        # 发出 POST 请求，并将流参数设置为 True，以处理流式响应
        with requests.post(url, json=payload, stream=True) as response:
            # 检查响应的状态，如果不是 200 OK 将抛出异常
            response.raise_for_status()

            # 逐行遍历响应内容，处理流式 JSON 响应
            for line in response.iter_lines():
                if line:
                    # 解析每一行 JSON 块并转换为 Python 对象
                    chunk = json.loads(line)

                    # 如果提供了回调函数，则将每个块作为参数调用回调函数
                    if callback:
                        callback(chunk)
                    else:
                        # 否则直接将状态信息打印到控制台
                        print(chunk.get('status', ''), end='', flush=True)
                    
                    # 如果响应中包含图层数据，可能需要显示更多细节（根据需要调整）
                    if 'digest' in chunk:
                        print(f" - Digest: {chunk['digest']}", end='', flush=True)
                        print(f" - Total: {chunk['total']}", end='', flush=True)
                        print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                    else:
                        print()
    except requests.exceptions.RequestException as e:
        # 捕获请求异常并将异常信息打印到控制台
        print(f"An error occurred: {e}")
# 将模型推送到模型注册表。使用回调函数覆盖默认处理程序。
def push(model_name, insecure=False, callback=None):
    try:
        # 构建推送请求的 URL
        url = f"{BASE_URL}/api/push"
        # 构建请求的有效负载，包括模型名称和是否使用不安全选项
        payload = {
            "name": model_name,
            "insecure": insecure
        }

        # 发出 POST 请求，并将流参数设置为 True，以处理流式响应
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()

            # 逐行遍历响应内容
            for line in response.iter_lines():
                if line:
                    # 解析每一行（JSON 块）并提取详细信息
                    chunk = json.loads(line)

                    # 如果提供了回调函数，则使用块作为参数调用回调函数
                    if callback:
                        callback(chunk)
                    else:
                        # 将状态信息直接打印到控制台
                        print(chunk.get('status', ''), end='', flush=True)
                    
                    # 如果块中包含图层数据，可能还需要打印（根据需要进行调整）
                    if 'digest' in chunk:
                        print(f" - Digest: {chunk['digest']}", end='', flush=True)
                        print(f" - Total: {chunk['total']}", end='', flush=True)
                        print(f" - Completed: {chunk['completed']}", end='\n', flush=True)
                    else:
                        print()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

# 列出本地可提供的型号。
def list():
    try:
        # 发出 GET 请求获取可用型号列表
        response = requests.get(f"{BASE_URL}/api/tags")
        response.raise_for_status()
        # 解析 JSON 响应并提取型号信息
        data = response.json()
        models = data.get('models', [])
        return models

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# 复制模型 从现有模型创建另一个名称的模型。
def copy(source, destination):
    try:
        # 创建 JSON 有效负载，包括源模型和目标模型名称
        payload = {
            "source": source,
            "destination": destination
        }
        
        # 发出 POST 请求执行复制操作
        response = requests.post(f"{BASE_URL}/api/copy", json=payload)
        response.raise_for_status()
        
        # 如果请求成功，则返回一条信息，说明复制成功
        return "Copy successful"

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# 删除模型及其数据。
def delete(model_name):
    try:
        # 构建删除请求的 URL
        url = f"{BASE_URL}/api/delete"
        # 构建请求的有效负载，包括要删除的模型名称
        payload = {"name": model_name}
        # 发出 DELETE 请求执行删除操作
        response = requests.delete(url, json=payload)
        response.raise_for_status()
        return "Delete successful"
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# 显示有关模型的信息。
def show(model_name):
    try:
        # 构建显示请求的 URL
        url = f"{BASE_URL}/api/show"
        # 构建请求的有效负载，包括要显示信息的模型名称
        payload = {"name": model_name}
        # 发出 POST 请求执行显示操作
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # 解析 JSON 响应并返回
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# 心跳功能（待补充具体实现）
def heartbeat():
    # 构建 URL，基于 BASE_URL 变量，并在结尾加上斜杠
    url = f"{BASE_URL}/"
    # 发送 HTTP HEAD 请求到构建好的 URL，获取响应
    response = requests.head(url)
    # 如果响应状态码不是 2xx，则抛出异常
    response.raise_for_status()
    # 如果上述步骤没有异常，说明服务正在运行，返回成功消息
    return "Ollama is running"
except requests.exceptions.RequestException as e:
    # 捕获请求过程中的异常，并打印出错信息
    print(f"An error occurred: {e}")
    # 返回服务不在运行的消息
    return "Ollama is not running"
```