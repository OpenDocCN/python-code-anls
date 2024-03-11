# `.\Langchain-Chatchat\tests\api\test_kb_summary_api.py`

```
# 导入requests库，用于发送HTTP请求
import requests
# 导入json库，用于处理JSON数据
import json
# 导入sys库，用于访问Python解释器的变量和函数
import sys
# 从pathlib库中导入Path类，用于处理文件路径
from pathlib import Path

# 获取当前文件的父目录的父目录的父目录作为根路径
root_path = Path(__file__).parent.parent.parent
# 将根路径添加到Python解释器的搜索路径中
sys.path.append(str(root_path))
# 从server.utils模块中导入api_address函数
from server.utils import api_address

# 获取API的基础URL
api_base_url = api_address()

# 知识库名称
kb = "samples"
# 文件名
file_name = "/media/gpt4-pdf-chatbot-langchain/langchain-ChatGLM/knowledge_base/samples/content/llm/大模型技术栈-实战与应用.md"
# 文档ID列表
doc_ids = [
    "357d580f-fdf7-495c-b58b-595a398284e8",
    "c7338773-2e83-4671-b237-1ad20335b0f0",
    "6da613d1-327d-466f-8c1a-b32e6f461f47"
]

# 定义测试函数，将文件摘要存储为向量
def test_summary_file_to_vector_store(api="/knowledge_base/kb_summary_api/summary_file_to_vector_store"):
    # 构建完整的API URL
    url = api_base_url + api
    # 打印提示信息
    print("\n文件摘要：")
    # 发送POST请求，传递知识库名称和文件名
    r = requests.post(url, json={"knowledge_base_name": kb,
                                 "file_name": file_name
                                 }, stream=True)
    # 遍历响应内容的数据块
    for chunk in r.iter_content(None):
        # 解析JSON数据
        data = json.loads(chunk[6:])
        # 断言数据类型为字典
        assert isinstance(data, dict)
        # 断言返回的状态码为200
        assert data["code"] == 200
        # 打印消息内容
        print(data["msg"])

# 定义测试函数，将文档ID列表的摘要存储为向量
def test_summary_doc_ids_to_vector_store(api="/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store"):
    # 构建完整的API URL
    url = api_base_url + api
    # 打印提示信息
    print("\n文件摘要：")
    # 发送POST请求，传递知识库名称和文档ID列表
    r = requests.post(url, json={"knowledge_base_name": kb,
                                 "doc_ids": doc_ids
                                 }, stream=True)
    # 遍历响应内容的数据块
    for chunk in r.iter_content(None):
        # 解析JSON数据
        data = json.loads(chunk[6:])
        # 断言数据类型为字典
        assert isinstance(data, dict)
        # 断言返回的状态码为200
        assert data["code"] == 200
        # 打印数据内容
        print(data)
```