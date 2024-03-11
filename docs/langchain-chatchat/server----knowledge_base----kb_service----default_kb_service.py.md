# `.\Langchain-Chatchat\server\knowledge_base\kb_service\default_kb_service.py`

```py
# 从 typing 模块导入 List 类型
from typing import List

# 从 langchain.embeddings.base 模块导入 Embeddings 类
from langchain.embeddings.base import Embeddings
# 从 langchain.schema 模块导入 Document 类
from langchain.schema import Document

# 从 server.knowledge_base.kb_service.base 模块导入 KBService 类
from server.knowledge_base.kb_service.base import KBService

# 定义 DefaultKBService 类，继承自 KBService 类
class DefaultKBService(KBService):
    # 定义创建知识库的方法
    def do_create_kb(self):
        pass

    # 定义删除知识库的方法
    def do_drop_kb(self):
        pass

    # 定义添加文档的方法，参数为文档列表
    def do_add_doc(self, docs: List[Document]):
        pass

    # 定义清空向量空间的方法
    def do_clear_vs(self):
        pass

    # 定义返回向量空间类型的方法
    def vs_type(self) -> str:
        return "default"

    # 定义初始化方法
    def do_init(self):
        pass

    # 定义搜索方法
    def do_search(self):
        pass

    # 定义插入多个知识的方法
    def do_insert_multi_knowledge(self):
        pass

    # 定义插入单个知识的方法
    def do_insert_one_knowledge(self):
        pass

    # 定义删除文档的方法
    def do_delete_doc(self):
        pass
```