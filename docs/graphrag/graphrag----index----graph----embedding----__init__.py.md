# `.\graphrag\graphrag\index\graph\embedding\__init__.py`

```py
# 版权声明，标明版权归 2024 年 Microsoft 公司所有，使用 MIT 许可证授权
# 导入当前目录下的 embedding 模块中的 NodeEmbeddings 类和 embed_nod2vec 函数
"""The Indexing Engine graph embedding package root."""
# 当前文件为索引引擎图嵌入包的根文件，说明此模块的作用和范围

from .embedding import NodeEmbeddings, embed_nod2vec
# 导入 embedding 模块中的 NodeEmbeddings 类和 embed_nod2vec 函数

__all__ = ["NodeEmbeddings", "embed_nod2vec"]
# 定义模块的公开接口，即外部可访问的模块成员列表
```