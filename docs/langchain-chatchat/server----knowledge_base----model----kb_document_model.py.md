# `.\Langchain-Chatchat\server\knowledge_base\model\kb_document_model.py`

```py
# 导入 Document 类
from langchain.docstore.document import Document

# 定义一个名为 DocumentWithVSId 的类，继承自 Document 类
class DocumentWithVSId(Document):
    """
    矢量化后的文档
    """
    # 定义类属性 id，类型为字符串，默认值为 None
    id: str = None
    # 定义类属性 score，类型为浮点数，默认值为 3.0
    score: float = 3.0
```