# `.\AutoGPT\autogpts\forge\forge\memory\memstore_test.py`

```py
# 导入 hashlib 和 shutil 模块
import hashlib
import shutil

# 导入 pytest 模块
import pytest

# 从 forge.sdk.memory.memstore 模块中导入 ChromaMemStore 类
from forge.sdk.memory.memstore import ChromaMemStore

# 使用 pytest.fixture 装饰器定义 memstore 函数
@pytest.fixture
def memstore():
    # 创建 ChromaMemStore 对象，存储在指定目录 ".test_mem_store" 中
    mem = ChromaMemStore(".test_mem_store")
    # 使用 yield 关键字将 mem 对象作为生成器的返回值
    yield mem
    # 在测试结束后删除指定目录 ".test_mem_store"
    shutil.rmtree(".test_mem_store")

# 定义测试函数 test_add，接受 memstore 参数
def test_add(memstore):
    # 定义任务 ID 和文档内容
    task_id = "test_task"
    document = "This is a test document."
    # 定义元数据
    metadatas = {"metadata": "test_metadata"}
    # 向 memstore 中添加文档
    memstore.add(task_id, document, metadatas)
    # 计算文档内容的 SHA256 哈希值的前 20 位作为文档 ID
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    # 断言 memstore 中指定任务 ID 的文档数量为 1
    assert memstore.client.get_or_create_collection(task_id).count() == 1

# 定义测试函数 test_query，接受 memstore 参数
def test_query(memstore):
    # 定义任务 ID 和文档内容
    task_id = "test_task"
    document = "This is a test document."
    # 定义元数据
    metadatas = {"metadata": "test_metadata"}
    # 向 memstore 中添加文档
    memstore.add(task_id, document, metadatas)
    # 定义查询关键词
    query = "test"
    # 断言 memstore 中指定任务 ID 和查询关键词的文档数量为 1
    assert len(memstore.query(task_id, query)["documents"]) == 1

# 定义测试函数 test_update，接受 memstore 参数
def test_update(memstore):
    # 定义任务 ID 和文档内容
    task_id = "test_task"
    document = "This is a test document."
    # 定义元数据
    metadatas = {"metadata": "test_metadata"}
    # 向 memstore 中添加文档
    memstore.add(task_id, document, metadatas)
    # 计算文档内容的 SHA256 哈希值的前 20 位作为文档 ID
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    # 定义更新后的文档内容和元数据
    updated_document = "This is an updated test document."
    updated_metadatas = {"metadata": "updated_test_metadata"}
    # 更新 memstore 中指定任务 ID 和文档 ID 的文档内容和元数据
    memstore.update(task_id, [doc_id], [updated_document], [updated_metadatas])
    # 断言 memstore 中指定任务 ID 和文档 ID 的文档内容和元数据更新成功
    assert memstore.get(task_id, [doc_id]) == {
        "documents": [updated_document],
        "metadatas": [updated_metadatas],
        "embeddings": None,
        "ids": [doc_id],
    }

# 定义测试函数 test_delete，接受 memstore 参数
def test_delete(memstore):
    # 定义任务 ID 和文档内容
    task_id = "test_task"
    document = "This is a test document."
    # 定义元数据
    metadatas = {"metadata": "test_metadata"}
    # 向 memstore 中添加文档
    memstore.add(task_id, document, metadatas)
    # 计算文档内容的 SHA256 哈希值的前 20 位作为文档 ID
    doc_id = hashlib.sha256(document.encode()).hexdigest()[:20]
    # 删除 memstore 中指定任务 ID 和文档 ID 的文档
    memstore.delete(task_id, doc_id)
    # 断言 memstore 中指定任务 ID 的文档数量为 0
    assert memstore.client.get_or_create_collection(task_id).count() == 0
```