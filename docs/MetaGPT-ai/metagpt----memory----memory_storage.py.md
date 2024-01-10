# `MetaGPT\metagpt\memory\memory_storage.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Desc   : the implement of memory storage
@Modified By: mashenquan, 2023/8/20. Remove global configuration `CONFIG`, enable configuration support for business isolation.
"""

from pathlib import Path  # 导入 Path 模块，用于处理文件路径
from typing import Optional  # 导入 Optional 类型提示，用于指定可选类型

from langchain.embeddings import OpenAIEmbeddings  # 导入 OpenAIEmbeddings 类
from langchain.vectorstores.faiss import FAISS  # 导入 FAISS 类
from langchain_core.embeddings import Embeddings  # 导入 Embeddings 类

from metagpt.const import DATA_PATH, MEM_TTL  # 从 metagpt.const 模块导入 DATA_PATH 和 MEM_TTL 常量
from metagpt.document_store.faiss_store import FaissStore  # 从 metagpt.document_store.faiss_store 模块导入 FaissStore 类
from metagpt.logs import logger  # 从 metagpt.logs 模块导入 logger 对象
from metagpt.schema import Message  # 从 metagpt.schema 模块导入 Message 类
from metagpt.utils.serialize import deserialize_message, serialize_message  # 从 metagpt.utils.serialize 模块导入 deserialize_message 和 serialize_message 函数


class MemoryStorage(FaissStore):
    """
    The memory storage with Faiss as ANN search engine
    """

    def __init__(self, mem_ttl: int = MEM_TTL, embedding: Embeddings = None):
        self.role_id: str = None  # 初始化 role_id 属性为 None
        self.role_mem_path: str = None  # 初始化 role_mem_path 属性为 None
        self.mem_ttl: int = mem_ttl  # 初始化 mem_ttl 属性为传入的 mem_ttl 参数
        self.threshold: float = 0.1  # 初始化 threshold 属性为 0.1
        self._initialized: bool = False  # 初始化 _initialized 属性为 False

        self.embedding = embedding or OpenAIEmbeddings()  # 如果传入的 embedding 参数为 None，则使用 OpenAIEmbeddings，否则使用传入的 embedding
        self.store: FAISS = None  # 初始化 store 属性为 None，类型为 FAISS

    @property
    def is_initialized(self) -> bool:
        return self._initialized  # 返回 _initialized 属性的值

    def _load(self) -> Optional["FaissStore"]:
        index_file, store_file = self._get_index_and_store_fname(index_ext=".faiss")  # 调用 _get_index_and_store_fname 方法获取索引文件和存储文件的路径

        if not (index_file.exists() and store_file.exists()):  # 如果索引文件和存储文件都不存在
            logger.info("Missing at least one of index_file/store_file, load failed and return None")  # 记录日志信息
            return None  # 返回 None

        return FAISS.load_local(self.role_mem_path, self.embedding, self.role_id)  # 调用 FAISS 类的 load_local 方法加载本地存储的数据

    def recover_memory(self, role_id: str) -> list[Message]:
        self.role_id = role_id  # 设置 role_id 属性为传入的 role_id 参数
        self.role_mem_path = Path(DATA_PATH / f"role_mem/{self.role_id}/")  # 设置 role_mem_path 属性为指定路径
        self.role_mem_path.mkdir(parents=True, exist_ok=True)  # 创建指定路径的文件夹，如果存在则不报错

        self.store = self._load()  # 调用 _load 方法加载存储数据
        messages = []  # 初始化 messages 列表
        if not self.store:  # 如果存储数据不存在
            # TODO init `self.store` under here with raw faiss api instead under `add`
            pass  # 什么也不做
        else:  # 如果存储数据存在
            for _id, document in self.store.docstore._dict.items():  # 遍历存储数据的文档
                messages.append(deserialize_message(document.metadata.get("message_ser")))  # 将反序列化后的消息添加到 messages 列表中
            self._initialized = True  # 设置 _initialized 属性为 True

        return messages  # 返回 messages 列表

    # ...（以下部分代码同理，根据代码逐行添加注释）

```