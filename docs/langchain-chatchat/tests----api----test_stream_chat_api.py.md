# `.\Langchain-Chatchat\tests\api\test_stream_chat_api.py`

```py
# 导入requests库，用于发送HTTP请求
import requests
# 导入json库，用于处理JSON数据
import json
# 导入sys库，用于访问Python解释器的变量和函数
import sys
# 从pathlib库中导入Path类，用于处理文件路径
from pathlib import Path

# 将上级目录的路径添加到sys.path中
sys.path.append(str(Path(__file__).parent.parent.parent))
# 从configs模块中导入BING_SUBSCRIPTION_KEY变量
from configs import BING_SUBSCRIPTION_KEY
# 从server.utils模块中导入api_address函数
from server.utils import api_address

# 从pprint库中导入pprint函数，用于美化打印输出
from pprint import pprint

# 获取API基础URL
api_base_url = api_address()

# 定义函数dump_input，用于打印输入数据
def dump_input(d, title):
    print("\n")
    print("=" * 30 + title + "  input " + "="*30)
    pprint(d)

# 定义函数dump_output，用于打印输出数据
def dump_output(r, title):
    print("\n")
    print("=" * 30 + title + "  output" + "="*30)
    # 遍历响应内容，打印每一行
    for line in r.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)

# 定义请求头信息
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

# 定义数据
data = {
    "query": "请用100字左右的文字介绍自己",
    "history": [
        {
            "role": "user",
            "content": "你好"
        },
        {
            "role": "assistant",
            "content": "你好，我是人工智能大模型"
        }
    ],
    "stream": True,
    "temperature": 0.7,
}

# 定义测试聊天API函数
def test_chat_chat(api="/chat/chat"):
    # 构建完整的URL
    url = f"{api_base_url}{api}"
    # 打印输入数据
    dump_input(data, api)
    # 发送POST请求，获取响应
    response = requests.post(url, headers=headers, json=data, stream=True)
    # 打印输出数据
    dump_output(response, api)
    # 断言响应状态码为200
    assert response.status_code == 200

# 定义测试知识库聊天API函数
def test_knowledge_chat(api="/chat/knowledge_base_chat"):
    # 构建完整的URL
    url = f"{api_base_url}{api}"
    # 定义新的数据
    data = {
        "query": "如何提问以获得高质量答案",
        "knowledge_base_name": "samples",
        "history": [
            {
                "role": "user",
                "content": "你好"
            },
            {
                "role": "assistant",
                "content": "你好，我是 ChatGLM"
            }
        ],
        "stream": True
    }
    # 打印输入数据
    dump_input(data, api)
    # 发送POST请求，获取响应
    response = requests.post(url, headers=headers, json=data, stream=True)
    print("\n")
    print("=" * 30 + api + "  output" + "="*30)
    # 遍历响应内容，解析JSON数据并打印
    for line in response.iter_content(None, decode_unicode=True):
        data = json.loads(line[6:])
        if "answer" in data:
            print(data["answer"], end="", flush=True)
    # 打印最终数据
    pprint(data)
    # 断言"data"字典中包含"docs"键并且其值是一个非空列表
    assert "docs" in data and len(data["docs"]) > 0
    # 断言响应状态码为200
    assert response.status_code == 200
# 定义一个测试搜索引擎聊天功能的函数，可以传入API路径，默认为"/chat/search_engine_chat"
def test_search_engine_chat(api="/chat/search_engine_chat"):
    # 声明全局变量data
    global data

    # 设置查询内容为"室温超导最新进展是什么样？"
    data["query"] = "室温超导最新进展是什么样？"

    # 构建完整的API请求URL
    url = f"{api_base_url}{api}"
    
    # 遍历搜索引擎列表["bing", "duckduckgo"]
    for se in ["bing", "duckduckgo"]:
        # 设置搜索引擎名称为当前遍历到的搜索引擎
        data["search_engine_name"] = se
        
        # 打印输入数据
        dump_input(data, api + f" by {se}")
        
        # 发送POST请求到API，并获取响应
        response = requests.post(url, json=data, stream=True)
        
        # 如果当前搜索引擎为bing且未设置BING_SUBSCRIPTION_KEY，则断言返回的数据中包含特定信息
        if se == "bing" and not BING_SUBSCRIPTION_KEY:
            data = response.json()
            assert data["code"] == 404
            assert data["msg"] == f"要使用Bing搜索引擎，需要设置 `BING_SUBSCRIPTION_KEY`"

        # 打印API输出分隔线和API输出信息
        print("\n")
        print("=" * 30 + api + f" by {se}  output" + "="*30)
        
        # 遍历响应内容的每一行，解析JSON数据并打印答案
        for line in response.iter_content(None, decode_unicode=True):
            data = json.loads(line[6:])
            if "answer" in data:
                print(data["answer"], end="", flush=True)
        
        # 断言返回的数据中包含"docs"字段且"docs"字段不为空
        assert "docs" in data and len(data["docs"]) > 0
        
        # 打印"docs"字段内容
        pprint(data["docs"])
        
        # 断言响应状态码为200
        assert response.status_code == 200
```