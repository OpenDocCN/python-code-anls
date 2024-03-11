# `.\Langchain-Chatchat\tests\kb_vector_db\test_pg_db.py`

```
# 从指定路径导入FaissKBService类
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
# 从指定路径导入PGKBService类
from server.knowledge_base.kb_service.pg_kb_service import PGKBService
# 从指定路径导入create_tables函数
from server.knowledge_base.migrate import create_tables
# 从指定路径导入KnowledgeFile类
from server.knowledge_base.utils import KnowledgeFile

# 创建一个PGKBService对象，传入参数"test"
kbService = PGKBService("test")

# 定义测试用的知识库名称和文件名
test_kb_name = "test"
test_file_name = "README.md"
# 创建一个KnowledgeFile对象，传入文件名和知识库名称
testKnowledgeFile = KnowledgeFile(test_file_name, test_kb_name)
# 定义搜索内容
search_content = "如何启动api服务"

# 定义测试初始化函数
def test_init():
    # 调用create_tables函数
    create_tables()

# 定义测试创建数据库函数
def test_create_db():
    # 断言调用kbService对象的create_kb方法
    assert kbService.create_kb()

# 定义测试添加文档函数
def test_add_doc():
    # 断言调用kbService对象的add_doc方法，传入testKnowledgeFile对象
    assert kbService.add_doc(testKnowledgeFile)

# 定义测试搜索数据库函数
def test_search_db():
    # 调用kbService对象的search_docs方法，传入搜索内容
    result = kbService.search_docs(search_content)
    # 断言搜索结果长度大于0
    assert len(result) > 0

# 定义测试删除文档函数
def test_delete_doc():
    # 断言调用kbService对象的delete_doc方法，传入testKnowledgeFile对象
    assert kbService.delete_doc(testKnowledgeFile)
```