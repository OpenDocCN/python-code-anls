# `.\Langchain-Chatchat\tests\kb_vector_db\test_milvus_db.py`

```py
# 从不同的模块中导入不同的知识库服务类
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
from server.knowledge_base.kb_service.pg_kb_service import PGKBService
# 从知识库迁移模块中导入创建表的函数
from server.knowledge_base.migrate import create_tables
# 从知识库工具模块中导入知识文件类
from server.knowledge_base.utils import KnowledgeFile

# 创建一个 Milvus 知识库服务对象，传入参数为 "test"
kbService = MilvusKBService("test")

# 定义测试用的知识库名称和文件名
test_kb_name = "test"
test_file_name = "README.md"
# 创建一个知识文件对象，传入参数为测试文件名和知识库名称
testKnowledgeFile = KnowledgeFile(test_file_name, test_kb_name)
# 定义搜索内容
search_content = "如何启动api服务"

# 定义测试初始化函数
def test_init():
    create_tables()

# 定义测试创建知识库函数
def test_create_db():
    # 断言知识库服务对象创建成功
    assert kbService.create_kb()

# 定义测试添加文档函数
def test_add_doc():
    # 断言知识库服务对象成功添加知识文件
    assert kbService.add_doc(testKnowledgeFile)

# 定义测试搜索知识库函数
def test_search_db():
    # 使用知识库服务对象搜索文档，将结果存储在 result 中
    result = kbService.search_docs(search_content)
    # 断言搜索结果数量大于 0
    assert len(result) > 0

# 定义测试删除文档函数
def test_delete_doc():
    # 断言知识库服务对象成功删除知识文件
    assert kbService.delete_doc(testKnowledgeFile)
```