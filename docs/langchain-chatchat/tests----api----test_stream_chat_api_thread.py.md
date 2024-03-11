# `.\Langchain-Chatchat\tests\api\test_stream_chat_api_thread.py`

```py
# 导入requests库，用于发送HTTP请求
import requests
# 导入json库，用于处理JSON数据
import json
# 导入sys库，用于访问Python解释器的变量和函数
import sys
# 从pathlib模块中导入Path类，用于处理文件路径
from pathlib import Path

# 将上级目录的路径添加到sys.path中
sys.path.append(str(Path(__file__).parent.parent.parent))
# 从configs模块中导入BING_SUBSCRIPTION_KEY变量
from configs import BING_SUBSCRIPTION_KEY
# 从server.utils模块中导入api_address函数
from server.utils import api_address

# 从pprint模块中导入pprint函数，用于美化打印输出
from pprint import pprint
# 从concurrent.futures模块中导入ThreadPoolExecutor和as_completed函数
from concurrent.futures import ThreadPoolExecutor, as_completed
# 导入time模块，用于处理时间相关操作
import time

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
    # 遍历响应内容，打印每行内容
    for line in r.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)

# 定义请求头信息
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

# 定义知识库聊天函数knowledge_chat
def knowledge_chat(api="/chat/knowledge_base_chat"):
    # 构建完整的URL
    url = f"{api_base_url}{api}"
    # 构建请求数据
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
    result = []
    # 发送POST请求
    response = requests.post(url, headers=headers, json=data, stream=True)

    # 遍历响应内容，解析JSON数据并添加到结果列表中
    for line in response.iter_content(None, decode_unicode=True):
        data = json.loads(line[6:])
        result.append(data)
    
    return result

# 定义测试多线程函数test_thread
def test_thread():
    threads = []
    times = []
    pool = ThreadPoolExecutor()
    start = time.time()
    # 提交多个任务到线程池
    for i in range(10):
        t = pool.submit(knowledge_chat)
        threads.append(t)
    
    # 等待所有任务完成
    for r in as_completed(threads):
        end = time.time()
        times.append(end - start)
        print("\nResult:\n")
        pprint(r.result())

    print("\nTime used:\n")
    # 打印每个任务的执行时间
    for x in times:
        print(f"{x}")
```