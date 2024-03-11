# `.\Langchain-Chatchat\server\knowledge_base\kb_cache\faiss_cache.py`

```
# 从configs模块中导入CACHED_VS_NUM和CACHED_MEMO_VS_NUM
from configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM
# 从server.knowledge_base.kb_cache.base模块中导入所有内容
from server.knowledge_base.kb_cache.base import *
# 从server.knowledge_base.kb_service.base模块中导入所有内容
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
# 从server.utils模块中导入load_local_embeddings函数
from server.utils import load_local_embeddings
# 从server.knowledge_base.utils模块中导入get_vs_path函数
from server.knowledge_base.utils import get_vs_path
# 从langchain.vectorstores.faiss模块中导入FAISS类
from langchain.vectorstores.faiss import FAISS
# 从langchain.docstore.in_memory模块中导入InMemoryDocstore类
from langchain.docstore.in_memory import InMemoryDocstore
# 从langchain.schema模块中导入Document类
from langchain.schema import Document
# 导入os模块
import os

# 定义一个新的函数_new_ds_search，用于在Document.metadata中包含文档id
def _new_ds_search(self, search: str) -> Union[str, Document]:
    # 如果搜索的id不在字典中，则返回提示信息
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        # 否则获取对应的文档对象
        doc = self._dict[search]
        # 如果文档对象是Document类型，则在metadata中添加id字段
        if isinstance(doc, Document):
            doc.metadata["id"] = search
        return doc
# 将新定义的函数_new_ds_search绑定到InMemoryDocstore类的search方法上
InMemoryDocstore.search = _new_ds_search

# 定义一个ThreadSafeFaiss类，继承自ThreadSafeObject类
class ThreadSafeFaiss(ThreadSafeObject):
    # 重写__repr__方法，返回对象的字符串表示形式
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    # 定义docs_count方法，返回文档数量
    def docs_count(self) -> int:
        return len(self._obj.docstore._dict)

    # 定义save方法，将向量库保存到指定路径
    def save(self, path: str, create_path: bool = True):
        with self.acquire():
            # 如果路径不存在且create_path为True，则创建路径
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            # 调用_obj的save_local方法保存向量库
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    # 定义clear方法，清空向量库
    def clear(self):
        ret = []
        with self.acquire():
            # 获取所有文档id
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                # 删除所有文档
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret

# 定义一个_FaissPool类，继承自CachePool类
class _FaissPool(CachePool):
    # 定义new_vector_store方法，用于创建新的向量库
    def new_vector_store(
        self,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    # 定义一个方法，用于创建并返回一个FAISS向量存储对象
    def create_vector_store(self, embed_model: EmbedModel) -> FAISS:
        # 创建一个嵌入适配器对象
        embeddings = EmbeddingsFunAdapter(embed_model)
        # 创建一个文档对象
        doc = Document(page_content="init", metadata={})
        # 使用文档对象和嵌入适配器对象创建一个FAISS向量存储对象
        vector_store = FAISS.from_documents([doc], embeddings, normalize_L2=True, distance_strategy="METRIC_INNER_PRODUCT")
        # 获取向量存储对象中的文档ID列表
        ids = list(vector_store.docstore._dict.keys())
        # 删除向量存储对象中的文档
        vector_store.delete(ids)
        # 返回创建的向量存储对象
        return vector_store

    # 定义一个方法，用于保存向量存储对象到指定路径
    def save_vector_store(self, kb_name: str, path: str=None):
        # 如果缓存中存在指定名称的向量存储对象，则保存到指定路径
        if cache := self.get(kb_name):
            return cache.save(path)

    # 定义一个方法，用于卸载指定名称的向量存储对象
    def unload_vector_store(self, kb_name: str):
        # 如果缓存中存在指定名称的向量存储对象，则从缓存中移除
        if cache := self.get(kb_name):
            self.pop(kb_name)
            # 记录成功释放向量库的日志信息
            logger.info(f"成功释放向量库：{kb_name}")
# 定义一个类 KBFaissPool，继承自 _FaissPool 类
class KBFaissPool(_FaissPool):
    # 加载向量存储
    def load_vector_store(
            self,
            kb_name: str,
            vector_name: str = None,
            create: bool = True,
            embed_model: str = EMBEDDING_MODEL,
            embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        # 获取锁，确保原子操作
        self.atomic.acquire()
        # 如果 vector_name 为空，则使用 embed_model
        vector_name = vector_name or embed_model
        # 获取缓存
        cache = self.get((kb_name, vector_name)) # 用元组比拼接字符串好一些
        # 如果缓存为空
        if cache is None:
            # 创建一个 ThreadSafeFaiss 对象
            item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
            # 将 item 存入缓存
            self.set((kb_name, vector_name), item)
            # 获取 item 的锁
            with item.acquire(msg="初始化"):
                # 释放原子锁
                self.atomic.release()
                # 记录日志，加载向量存储
                logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                # 获取向量存储路径
                vs_path = get_vs_path(kb_name, vector_name)

                # 如果 index.faiss 文件存在
                if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                    # 加载知识库嵌入
                    embeddings = self.load_kb_embeddings(kb_name=kb_name, embed_device=embed_device, default_embed_model=embed_model)
                    # 加载本地向量存储
                    vector_store = FAISS.load_local(vs_path, embeddings, normalize_L2=True,distance_strategy="METRIC_INNER_PRODUCT")
                # 如果 create 为真
                elif create:
                    # 创建一个空的向量存储
                    if not os.path.exists(vs_path):
                        os.makedirs(vs_path)
                    # 创建新的向量存储
                    vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                    # 保存向量存储到本地
                    vector_store.save_local(vs_path)
                else:
                    # 抛出运行时错误，知识库不存在
                    raise RuntimeError(f"knowledge base {kb_name} not exist.")
                # 将向量存储赋值给 item 的 obj 属性
                item.obj = vector_store
                # 完成加载
                item.finish_loading()
        else:
            # 释放原子锁
            self.atomic.release()
        # 返回缓存中的向量存储
        return self.get((kb_name, vector_name))


# 定义一个类 MemoFaissPool，继承自 _FaissPool 类
class MemoFaissPool(_FaissPool):
    # 加载向量存储
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    # 定义一个方法，接受一个键名参数并返回一个ThreadSafeFaiss对象
    ) -> ThreadSafeFaiss:
        # 获取锁对象
        self.atomic.acquire()
        # 获取缓存数据
        cache = self.get(kb_name)
        # 如果缓存为空
        if cache is None:
            # 创建一个ThreadSafeFaiss对象
            item = ThreadSafeFaiss(kb_name, pool=self)
            # 将对象存入缓存
            self.set(kb_name, item)
            # 获取对象锁
            with item.acquire(msg="初始化"):
                # 释放锁
                self.atomic.release()
                # 记录日志信息
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                # 创建一个空的向量存储
                vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                # 将向量存储对象赋值给item的obj属性
                item.obj = vector_store
                # 完成加载
                item.finish_loading()
        else:
            # 释放锁
            self.atomic.release()
        # 返回缓存中的对象
        return self.get(kb_name)
# 创建一个 KBFaissPool 对象，用于缓存向量检索结果，设置缓存数量为 CACHED_VS_NUM
kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
# 创建一个 MemoFaissPool 对象，用于缓存备忘向量检索结果，设置缓存数量为 CACHED_MEMO_VS_NUM
memo_faiss_pool = MemoFaissPool(cache_num=CACHED_MEMO_VS_NUM)

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    import time, random
    from pprint import pprint

    # 定义一个包含向量存储名称的列表
    kb_names = ["vs1", "vs2", "vs3"]
    
    # 定义一个 worker 函数，用于模拟处理向量存储的操作
    def worker(vs_name: str, name: str):
        # 将 vs_name 设置为 "samples"
        vs_name = "samples"
        # 随机休眠1到5秒
        time.sleep(random.randint(1, 5))
        # 加载本地嵌入向量
        embeddings = load_local_embeddings()
        # 随机生成一个1到3的整数
        r = random.randint(1, 3)

        # 使用 kb_faiss_pool 加载指定名称的向量存储，并获取其访问权限
        with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
            # 如果 r 等于 1，执行添加文档操作
            if r == 1:
                # 向向量存储中添加文本，并返回添加的文档 ID
                ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
                pprint(ids)
            # 如果 r 等于 2，执行搜索文档操作
            elif r == 2:
                # 使用相似度阈值为1.0进行搜索，并返回匹配的文档列表
                docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
                pprint(docs)
        # 如果 r 等于 3，执行删除文档操作
        if r == 3:
            logger.warning(f"清除 {vs_name} by {name}")
            # 清空指定向量存储中的所有文档
            kb_faiss_pool.get(vs_name).clear()

    # 创建线程列表
    threads = []
    # 循环创建30个线程
    for n in range(1, 30):
        # 创建一个线程，目标函数为 worker，传入随机选择的向量存储名称和工作名称
        t = threading.Thread(target=worker,
                             kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
                             daemon=True)
        # 启动线程
        t.start()
        threads.append(t)

    # 等待所有线程执行完毕
    for t in threads:
        t.join()
```