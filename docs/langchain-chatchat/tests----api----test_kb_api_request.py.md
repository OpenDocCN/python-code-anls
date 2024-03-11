# `.\Langchain-Chatchat\tests\api\test_kb_api_request.py`

```py
# 导入requests、json、sys模块以及Path类
import requests
import json
import sys
from pathlib import Path

# 获取当前文件的父目录的父目录的父目录作为根路径
root_path = Path(__file__).parent.parent.parent
# 将根路径添加到系统路径中
sys.path.append(str(root_path))
# 导入api_address函数和VECTOR_SEARCH_TOP_K常量
from server.utils import api_address
from configs import VECTOR_SEARCH_TOP_K
# 导入get_kb_path、get_file_path函数
from server.knowledge_base.utils import get_kb_path, get_file_path
# 导入ApiRequest类
from webui_pages.utils import ApiRequest

# 导入pprint函数
from pprint import pprint

# 获取API基础地址
api_base_url = api_address()
# 创建ApiRequest对象
api: ApiRequest = ApiRequest(api_base_url)

# 定义知识库名称
kb = "kb_for_api_test"
# 定义测试文件字典
test_files = {
    "FAQ.MD": str(root_path / "docs" / "FAQ.MD"),
    "README.MD": str(root_path / "README.MD"),
    "test.txt": get_file_path("samples", "test.txt"),
}

# 打印信息
print("\n\nApiRquest调用\n")

# 测试删除知识库前的函数
def test_delete_kb_before():
    # 如果知识库路径不存在，则返回
    if not Path(get_kb_path(kb)).exists():
        return

    # 调用API删除知识库
    data = api.delete_knowledge_base(kb)
    pprint(data)
    assert data["code"] == 200
    assert isinstance(data["data"], list) and len(data["data"]) > 0
    assert kb not in data["data"]

# 测试创建知识库的函数
def test_create_kb():
    # 尝试用空名称创建知识库
    print(f"\n尝试用空名称创建知识库：")
    data = api.create_knowledge_base(" ")
    pprint(data)
    assert data["code"] == 404
    assert data["msg"] == "知识库名称不能为空，请重新填写知识库名称"

    # 创建新知识库
    print(f"\n创建新知识库： {kb}")
    data = api.create_knowledge_base(kb)
    pprint(data)
    assert data["code"] == 200
    assert data["msg"] == f"已新增知识库 {kb}"

    # 尝试创建同名知识库
    print(f"\n尝试创建同名知识库： {kb}")
    data = api.create_knowledge_base(kb)
    pprint(data)
    assert data["code"] == 404
    assert data["msg"] == f"已存在同名知识库 {kb}"

# 测试列出知识库的函数
def test_list_kbs():
    data = api.list_knowledge_bases()
    pprint(data)
    assert isinstance(data, list) and len(data) > 0
    assert kb in data

# 测试上传文档的函数
def test_upload_docs():
    files = list(test_files.values())

    # 上传知识文件
    print(f"\n上传知识文件")
    data = {"knowledge_base_name": kb, "override": True}
    data = api.upload_kb_docs(files, **data)
    pprint(data)
    assert data["code"] == 200
    assert len(data["data"]["failed_files"]) == 0

    # 尝试重新上传知识文件，不覆盖
    print(f"\n尝试重新上传知识文件， 不覆盖")
    data = {"knowledge_base_name": kb, "override": False}
    # 使用 API 上传知识库文档文件，并返回上传结果
    data = api.upload_kb_docs(files, **data)
    # 打印上传结果
    pprint(data)
    # 断言上传结果中的状态码为 200
    assert data["code"] == 200
    # 断言上传失败的文件数量与测试文件数量相同
    assert len(data["data"]["failed_files"]) == len(test_files)

    # 打印提示信息，重新上传知识文件，覆盖已存在的文件，使用自定义文档
    print(f"\n尝试重新上传知识文件， 覆盖，自定义docs")
    # 定义自定义文档内容
    docs = {"FAQ.MD": [{"page_content": "custom docs", "metadata": {}}]}
    # 构建上传知识库文档的参数
    data = {"knowledge_base_name": kb, "override": True, "docs": docs}
    # 使用 API 上传知识库文档文件，并返回上传结果
    data = api.upload_kb_docs(files, **data)
    # 打印上传结果
    pprint(data)
    # 断言上传结果中的状态码为 200
    assert data["code"] == 200
    # 断言上传失败的文件数量为 0
    assert len(data["data"]["failed_files"]) == 0
# 测试函数：列出知识库中的文件列表
def test_list_files():
    # 打印提示信息
    print("\n获取知识库中文件列表：")
    # 调用API函数列出知识库文档
    data = api.list_kb_docs(knowledge_base_name=kb)
    # 打印返回数据
    pprint(data)
    # 断言返回数据类型为列表
    assert isinstance(data, list)
    # 遍历测试文件列表，确保文件名在返回数据中
    for name in test_files:
        assert name in data

# 测试函数：搜索知识库文档
def test_search_docs():
    # 设置搜索关键词
    query = "介绍一下langchain-chatchat项目"
    # 打印提示信息
    print("\n检索知识库：")
    print(query)
    # 调用API函数搜索知识库文档
    data = api.search_kb_docs(query, kb)
    # 打印返回数据
    pprint(data)
    # 断言返回数据类型为列表且长度为预设值
    assert isinstance(data, list) and len(data) == VECTOR_SEARCH_TOP_K

# 测试函数：更新知识库文档
def test_update_docs():
    # 打印提示信息
    print(f"\n更新知识文件")
    # 调用API函数更新知识库文档
    data = api.update_kb_docs(knowledge_base_name=kb, file_names=list(test_files))
    # 打印返回数据
    pprint(data)
    # 断言返回数据中状态码为200
    assert data["code"] == 200
    # 断言返回数据中无失败文件
    assert len(data["data"]["failed_files"]) == 0

# 测试函数：删除知识库文档
def test_delete_docs():
    # 打印提示信息
    print(f"\n删除知识文件")
    # 调用API函数删除知识库文档
    data = api.delete_kb_docs(knowledge_base_name=kb, file_names=list(test_files))
    # 打印返回数据
    pprint(data)
    # 断言返回数据中状态码为200
    assert data["code"] == 200
    # 断言返回数据中无失败文件
    assert len(data["data"]["failed_files"]) == 0

    # 再次搜索已删除的文件
    query = "介绍一下langchain-chatchat项目"
    print("\n尝试检索删除后的检索知识库：")
    print(query)
    # 调用API函数搜索知识库文档
    data = api.search_kb_docs(query, kb)
    # 打印返回数据
    pprint(data)
    # 断言返回数据类型为列表且长度为0
    assert isinstance(data, list) and len(data) == 0

# 测试函数：重建知识库
def test_recreate_vs():
    # 打印提示信息
    print("\n重建知识库：")
    # 调用API函数重建向量存储
    r = api.recreate_vector_store(kb)
    # 遍历返回数据
    for data in r:
        # 断言返回数据类型为字典
        assert isinstance(data, dict)
        # 断言返回数据中状态码为200
        assert data["code"] == 200
        print(data["msg"])

    # 再次搜索重建后的文件
    query = "本项目支持哪些文件格式?"
    print("\n尝试检索重建后的检索知识库：")
    print(query)
    # 调用API函数搜索知识库文档
    data = api.search_kb_docs(query, kb)
    # 打印返回数据
    pprint(data)
    # 断言返回数据类型为列表且长度为预设值
    assert isinstance(data, list) and len(data) == VECTOR_SEARCH_TOP_K

# 测试函数：删除知识库
def test_delete_kb_after():
    # 打印提示信息
    print("\n删除知识库")
    # 调用API函数删除知识库
    data = api.delete_knowledge_base(kb)
    # 打印返回数据
    pprint(data)

    # 检查知识库是否已删除
    print("\n获取知识库列表：")
    # 调用API函数列出知识库列表
    data = api.list_knowledge_bases()
    # 打印返回数据
    pprint(data)
    # 断言返回数据类型为列表且长度大于0
    assert isinstance(data, list) and len(data) > 0
    # 断言知识库不在返回数据中
    assert kb not in data
```