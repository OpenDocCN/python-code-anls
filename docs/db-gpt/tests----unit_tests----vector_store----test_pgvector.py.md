# `.\DB-GPT-src\tests\unit_tests\vector_store\test_pgvector.py`

```py
from dbgpt.rag.index.base import IndexStoreConfig
from dbgpt.storage import vector_store

def test_vetorestore_imports() -> None:
    """Simple test to make sure all things can be imported."""

    # 遍历 vector_store 模块中声明的所有类名
    for cls in vector_store.__all__:
        # 获取 vector_store 模块中对应类名的类对象和配置类对象
        store_cls, config_cls = getattr(vector_store, cls)
        
        # 从 dbgpt.rag.index.base 模块中导入 IndexStoreBase 类
        from dbgpt.rag.index.base import IndexStoreBase
        
        # 断言 store_cls 是 IndexStoreBase 的子类
        assert issubclass(store_cls, IndexStoreBase)
        
        # 断言 config_cls 是 IndexStoreConfig 的子类
        assert issubclass(config_cls, IndexStoreConfig)
```