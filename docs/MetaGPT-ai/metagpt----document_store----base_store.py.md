# `MetaGPT\metagpt\document_store\base_store.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/28 00:01
@Author  : alexanderwu
@File    : base_store.py
"""
# 导入必要的模块
from abc import ABC, abstractmethod
from pathlib import Path

from metagpt.config import Config

# 定义一个抽象基类 BaseStore
class BaseStore(ABC):
    """FIXME: consider add_index, set_index and think about granularity."""
    
    # 定义抽象方法 search
    @abstractmethod
    def search(self, *args, **kwargs):
        raise NotImplementedError

    # 定义抽象方法 write
    @abstractmethod
    def write(self, *args, **kwargs):
        raise NotImplementedError

    # 定义抽象方法 add
    @abstractmethod
    def add(self, *args, **kwargs):
        raise NotImplementedError

# 定义一个 LocalStore 类，继承自 BaseStore 和 ABC
class LocalStore(BaseStore, ABC):
    # 初始化方法
    def __init__(self, raw_data_path: Path, cache_dir: Path = None):
        # 如果 raw_data_path 不存在，则抛出 FileNotFoundError
        if not raw_data_path:
            raise FileNotFoundError
        # 实例化 Config 类
        self.config = Config()
        # 设置 raw_data_path 和 fname 属性
        self.raw_data_path = raw_data_path
        self.fname = self.raw_data_path.stem
        # 如果 cache_dir 不存在，则设置为 raw_data_path 的父目录
        if not cache_dir:
            cache_dir = raw_data_path.parent
        self.cache_dir = cache_dir
        # 调用 _load 方法加载数据
        self.store = self._load()
        # 如果加载的数据为空，则调用 write 方法写入数据
        if not self.store:
            self.store = self.write()

    # 定义一个内部方法，用于生成索引文件名和存储文件名
    def _get_index_and_store_fname(self, index_ext=".index", pkl_ext=".pkl"):
        index_file = self.cache_dir / f"{self.fname}{index_ext}"
        store_file = self.cache_dir / f"{self.fname}{pkl_ext}"
        return index_file, store_file

    # 定义抽象方法 _load
    @abstractmethod
    def _load(self):
        raise NotImplementedError

    # 定义抽象方法 _write
    @abstractmethod
    def _write(self, docs, metadatas):
        raise NotImplementedError

```