# `.\Langchain-Chatchat\tests\api\test_kb_api.py`

```py
# 导入需要的模块
import requests
import json
import sys
from pathlib import Path

# 获取当前文件的父目录的父目录的父目录作为根路径
root_path = Path(__file__).parent.parent.parent
# 将根路径添加到系统路径中
sys.path.append(str(root_path))
# 导入自定义模块
from server.utils import api_address
from configs import VECTOR_SEARCH_TOP_K
from server.knowledge_base.utils import get_kb_path, get_file_path

# 导入 pprint 模块
from pprint import pprint

# 获取 API 基础地址
api_base_url = api_address()

# 定义知识库名称和测试文件路径的字典
kb = "kb_for_api_test"
test_files = {
    "wiki/Home.MD": get_file_path("samples", "wiki/Home.md"),
    "wiki/开发环境部署.MD": get_file_path("samples", "wiki/开发环境部署.md"),
    "test_files/test.txt": get_file_path("samples", "test_files/test.txt"),
}

# 打印提示信息
print("\n\n直接url访问\n")

# 定义测试删除知识库前的函数
def test_delete_kb_before(api="/knowledge_base/delete_knowledge_base"):
    # 如果知识库文件路径不存在，则直接返回
    if not Path(get_kb_path(kb)).exists():
        return

    # 构建 API 请求的 URL
    url = api_base_url + api
    print("\n测试知识库存在，需要删除")
    # 发送 POST 请求删除知识库
    r = requests.post(url, json=kb)
    data = r.json()
    pprint(data)

    # 检查知识库是否已删除
    url = api_base_url + "/knowledge_base/list_knowledge_bases"
    print("\n获取知识库列表：")
    r = requests.get(url)
    data = r.json()
    pprint(data)
    assert data["code"] == 200
    assert isinstance(data["data"], list) and len(data["data"]) > 0
    assert kb not in data["data"]

# 定义测试创建知识库的函数
def test_create_kb(api="/knowledge_base/create_knowledge_base"):
    url = api_base_url + api

    # 尝试用空名称创建知识库
    print(f"\n尝试用空名称创建知识库：")
    r = requests.post(url, json={"knowledge_base_name": " "})
    data = r.json()
    pprint(data)
    assert data["code"] == 404
    assert data["msg"] == "知识库名称不能为空，请重新填写知识库名称"

    # 创建新知识库
    print(f"\n创建新知识库： {kb}")
    r = requests.post(url, json={"knowledge_base_name": kb})
    data = r.json()
    pprint(data)
    assert data["code"] == 200
    assert data["msg"] == f"已新增知识库 {kb}"

    # 尝试创建同名知识库
    print(f"\n尝试创建同名知识库： {kb}")
    r = requests.post(url, json={"knowledge_base_name": kb})
    data = r.json()
    pprint(data)
    assert data["code"] == 404
    assert data["msg"] == f"已存在同名知识库 {kb}"

# 定义测试列出知识库的函数
def test_list_kbs(api="/knowledge_base/list_knowledge_bases"):
    # 拼接 API 的完整 URL
    url = api_base_url + api
    # 打印提示信息
    print("\n获取知识库列表：")
    # 发起 GET 请求获取数据
    r = requests.get(url)
    # 将响应数据解析为 JSON 格式
    data = r.json()
    # 打印数据
    pprint(data)
    # 断言返回的数据中 code 字段的值为 200
    assert data["code"] == 200
    # 断言返回的数据中 data 字段是一个列表且长度大于 0
    assert isinstance(data["data"], list) and len(data["data"]) > 0
    # 断言返回的数据中包含指定的知识库
    assert kb in data["data"]
# 测试上传文档功能
def test_upload_docs(api="/knowledge_base/upload_docs"):
    # 构建完整的 API URL
    url = api_base_url + api
    # 准备上传的文件列表
    files = [("files", (name, open(path, "rb"))) for name, path in test_files.items()]

    # 打印提示信息
    print(f"\n上传知识文件")
    # 准备上传的数据
    data = {"knowledge_base_name": kb, "override": True}
    # 发起 POST 请求上传文件
    r = requests.post(url, data=data, files=files)
    # 解析响应数据
    data = r.json()
    pprint(data)
    # 断言响应状态码为 200
    assert data["code"] == 200
    # 断言上传失败的文件数为 0
    assert len(data["data"]["failed_files"]) == 0

    # 打印提示信息
    print(f"\n尝试重新上传知识文件， 不覆盖")
    # 准备上传的数据
    data = {"knowledge_base_name": kb, "override": False}
    # 准备上传的文件列表
    files = [("files", (name, open(path, "rb"))) for name, path in test_files.items()]
    # 发起 POST 请求重新上传文件
    r = requests.post(url, data=data, files=files)
    # 解析响应数据
    data = r.json()
    pprint(data)
    # 断言响应状态码为 200
    assert data["code"] == 200
    # 断言上传失败的文件数与测试文件数相同
    assert len(data["data"]["failed_files"]) == len(test_files)

    # 打印提示信息
    print(f"\n尝试重新上传知识文件， 覆盖，自定义docs")
    # 准备自定义的文档数据
    docs = {"FAQ.MD": [{"page_content": "custom docs", "metadata": {}}]}
    # 准备上传的数据
    data = {"knowledge_base_name": kb, "override": True, "docs": json.dumps(docs)}
    # 准备上传的文件列表
    files = [("files", (name, open(path, "rb"))) for name, path in test_files.items()]
    # 发起 POST 请求重新上传文件
    r = requests.post(url, data=data, files=files)
    # 解析响应数据
    data = r.json()
    pprint(data)
    # 断言响应状态码为 200
    assert data["code"] == 200
    # 断言上传失败的文件数为 0
    assert len(data["data"]["failed_files"]) == 0


# 测试获取文件列表功能
def test_list_files(api="/knowledge_base/list_files"):
    # 构建完整的 API URL
    url = api_base_url + api
    # 打印提示信息
    print("\n获取知识库中文件列表：")
    # 发起 GET 请求获取文件列表
    r = requests.get(url, params={"knowledge_base_name": kb})
    # 解析响应数据
    data = r.json()
    pprint(data)
    # 断言响应状态码为 200
    assert data["code"] == 200
    # 断言返回的数据是列表类型
    assert isinstance(data["data"], list)
    # 断言测试文件名在返回的文件列表中
    for name in test_files:
        assert name in data["data"]


# 测试搜索文档功能
def test_search_docs(api="/knowledge_base/search_docs"):
    # 构建完整的 API URL
    url = api_base_url + api
    # 设置搜索关键词
    query = "介绍一下langchain-chatchat项目"
    # 打印提示信息
    print("\n检索知识库：")
    print(query)
    # 发起 POST 请求搜索文档
    r = requests.post(url, json={"knowledge_base_name": kb, "query": query})
    # 解析响应数据
    data = r.json()
    pprint(data)
    # 断言返回的数据是列表类型且长度为 VECTOR_SEARCH_TOP_K
    assert isinstance(data, list) and len(data) == VECTOR_SEARCH_TOP_K


# 测试更新信息功能
def test_update_info(api="/knowledge_base/update_info"):
    # 拼接 API 的完整 URL
    url = api_base_url + api
    # 打印提示信息
    print("\n更新知识库介绍")
    # 发起 POST 请求，传递 JSON 数据
    r = requests.post(url, json={"knowledge_base_name": "samples", "kb_info": "你好"})
    # 解析响应数据为 JSON 格式
    data = r.json()
    # 格式化输出响应数据
    pprint(data)
    # 断言响应数据中的 code 字段为 200
    assert data["code"] == 200
# 测试更新知识库文档的函数
def test_update_docs(api="/knowledge_base/update_docs"):
    # 构建完整的 API 地址
    url = api_base_url + api

    # 打印提示信息
    print(f"\n更新知识文件")
    # 发送 POST 请求，更新知识库文档
    r = requests.post(url, json={"knowledge_base_name": kb, "file_names": list(test_files)})
    # 解析响应数据
    data = r.json()
    # 打印响应数据
    pprint(data)
    # 断言响应状态码为 200
    assert data["code"] == 200
    # 断言失败文件列表为空
    assert len(data["data"]["failed_files"]) == 0


# 测试删除知识库文档的函数
def test_delete_docs(api="/knowledge_base/delete_docs"):
    # 构建完整的 API 地址
    url = api_base_url + api

    # 打印提示信息
    print(f"\n删除知识文件")
    # 发送 POST 请求，删除知识库文档
    r = requests.post(url, json={"knowledge_base_name": kb, "file_names": list(test_files)})
    # 解析响应数据
    data = r.json()
    # 打印响应数据
    pprint(data)
    # 断言响应状态码为 200
    assert data["code"] == 200
    # 断言失败文件列表为空
    assert len(data["data"]["failed_files"]) == 0

    # 构建搜索知识库文档的 API 地址
    url = api_base_url + "/knowledge_base/search_docs"
    query = "介绍一下langchain-chatchat项目"
    # 打印提示信息
    print("\n尝试检索删除后的检索知识库：")
    print(query)
    # 发送 POST 请求，搜索知识库文档
    r = requests.post(url, json={"knowledge_base_name": kb, "query": query})
    # 解析响应数据
    data = r.json()
    # 打印响应数据
    pprint(data)
    # 断言返回数据为列表且长度为 0
    assert isinstance(data, list) and len(data) == 0


# 测试重建向量存储的函数
def test_recreate_vs(api="/knowledge_base/recreate_vector_store"):
    # 构建完整的 API 地址
    url = api_base_url + api
    # 打印提示信息
    print("\n重建知识库：")
    # 发送 POST 请求，重建向量存储
    r = requests.post(url, json={"knowledge_base_name": kb}, stream=True)
    # 遍历响应内容
    for chunk in r.iter_content(None):
        data = json.loads(chunk[6:])
        # 断言返回数据为字典
        assert isinstance(data, dict)
        # 断言响应状态码为 200
        assert data["code"] == 200
        print(data["msg"])

    # 构建搜索知识库文档的 API 地址
    url = api_base_url + "/knowledge_base/search_docs"
    query = "本项目支持哪些文件格式?"
    # 打印提示信息
    print("\n尝试检索重建后的检索知识库：")
    print(query)
    # 发送 POST 请求，搜索知识库文档
    r = requests.post(url, json={"knowledge_base_name": kb, "query": query})
    # 解析响应数据
    data = r.json()
    # 打印响应数据
    pprint(data)
    # 断言返回数据为列表且长度为 VECTOR_SEARCH_TOP_K

# 测试删除知识库的函数
def test_delete_kb_after(api="/knowledge_base/delete_knowledge_base"):
    # 构建完整的 API 地址
    url = api_base_url + api
    # 打印提示信息
    print("\n删除知识库")
    # 发送 POST 请求，删除知识库
    r = requests.post(url, json=kb)
    # 解析响应数据
    data = r.json()
    # 打印响应数据
    pprint(data)

    # 检查知识库是否已删除
    url = api_base_url + "/knowledge_base/list_knowledge_bases"
    # 打印提示信息
    print("\n获取知识库列表：")
    # 发送 GET 请求，获取知识库列表
    r = requests.get(url)
    # 从响应中获取 JSON 数据
    data = r.json()
    # 使用 pprint 函数打印数据，方便查看
    pprint(data)
    # 断言数据中的 code 键的值为 200
    assert data["code"] == 200
    # 断言数据中的 data 键的值是一个列表且长度大于 0
    assert isinstance(data["data"], list) and len(data["data"]) > 0
    # 断言 kb 不在 data["data"] 中
    assert kb not in data["data"]
```