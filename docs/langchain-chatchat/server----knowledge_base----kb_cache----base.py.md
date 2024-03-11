# `.\Langchain-Chatchat\server\knowledge_base\kb_cache\base.py`

```
# 导入所需的模块
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS
import threading
from configs import (EMBEDDING_MODEL, CHUNK_SIZE,
                     logger, log_verbose)
from server.utils import embedding_device, get_model_path, list_online_embed_models
from contextlib import contextmanager
from collections import OrderedDict
from typing import List, Any, Union, Tuple

# 定义一个线程安全的对象类
class ThreadSafeObject:
    def __init__(self, key: Union[str, Tuple], obj: Any = None, pool: "CachePool" = None):
        self._obj = obj
        self._key = key
        self._pool = pool
        self._lock = threading.RLock()
        self._loaded = threading.Event()

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}>"

    @property
    def key(self):
        return self._key

    # 定义上下文管理器，用于获取对象并执行操作
    @contextmanager
    def acquire(self, owner: str = "", msg: str = "") -> FAISS:
        owner = owner or f"thread {threading.get_native_id()}"
        try:
            self._lock.acquire()
            if self._pool is not None:
                self._pool._cache.move_to_end(self.key)
            if log_verbose:
                logger.info(f"{owner} 开始操作：{self.key}。{msg}")
            yield self._obj
        finally:
            if log_verbose:
                logger.info(f"{owner} 结束操作：{self.key}。{msg}")
            self._lock.release()

    # 标记对象开始加载
    def start_loading(self):
        self._loaded.clear()

    # 标记对象加载完成
    def finish_loading(self):
        self._loaded.set()

    # 等待对象加载完成
    def wait_for_loading(self):
        self._loaded.wait()

    @property
    def obj(self):
        return self._obj

    # 设置对象属性
    @obj.setter
    def obj(self, val: Any):
        self._obj = val

# 定义一个缓存池类
class CachePool:
    def __init__(self, cache_num: int = -1):
        self._cache_num = cache_num
        self._cache = OrderedDict()
        self.atomic = threading.RLock()

    # 返回缓存池中的所有键
    def keys(self) -> List[str]:
        return list(self._cache.keys())
    # 检查缓存数量是否合法，如果是整数且大于0，则保持缓存数量不超过设定值
    def _check_count(self):
        if isinstance(self._cache_num, int) and self._cache_num > 0:
            while len(self._cache) > self._cache_num:
                self._cache.popitem(last=False)

    # 获取缓存中指定键对应的对象，如果存在则等待加载完成后返回对象
    def get(self, key: str) -> ThreadSafeObject:
        if cache := self._cache.get(key):
            cache.wait_for_loading()
            return cache

    # 设置缓存中指定键对应的对象，并检查缓存数量是否超过设定值
    def set(self, key: str, obj: ThreadSafeObject) -> ThreadSafeObject:
        self._cache[key] = obj
        self._check_count()
        return obj

    # 从缓存中弹出指定键对应的对象，如果未指定键则弹出最旧的对象
    def pop(self, key: str = None) -> ThreadSafeObject:
        if key is None:
            return self._cache.popitem(last=False)
        else:
            return self._cache.pop(key, None)

    # 获取指定键对应的对象，并尝试获取对象的锁，如果对象不存在或不可加锁则抛出异常
    def acquire(self, key: Union[str, Tuple], owner: str = "", msg: str = ""):
        cache = self.get(key)
        if cache is None:
            raise RuntimeError(f"请求的资源 {key} 不存在")
        elif isinstance(cache, ThreadSafeObject):
            self._cache.move_to_end(key)
            return cache.acquire(owner=owner, msg=msg)
        else:
            return cache

    # 加载知识库的嵌入向量，根据指定的知识库名称和设备加载对应的嵌入向量
    def load_kb_embeddings(
            self,
            kb_name: str,
            embed_device: str = embedding_device(),
            default_embed_model: str = EMBEDDING_MODEL,
    ) -> Embeddings:
        from server.db.repository.knowledge_base_repository import get_kb_detail
        from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter

        # 获取知识库的详细信息和嵌入模型
        kb_detail = get_kb_detail(kb_name)
        embed_model = kb_detail.get("embed_model", default_embed_model)

        # 如果嵌入模型在在线嵌入模型列表中，则返回适配器对象
        if embed_model in list_online_embed_models():
            return EmbeddingsFunAdapter(embed_model)
        else:
            # 否则加载指定模型的嵌入向量
            return embeddings_pool.load_embeddings(model=embed_model, device=embed_device)
# 定义一个名为EmbeddingsPool的类，继承自CachePool类
class EmbeddingsPool(CachePool):
# 创建一个名为embeddings_pool的实例，使用EmbeddingsPool类，设置缓存数量为1
embeddings_pool = EmbeddingsPool(cache_num=1)
```