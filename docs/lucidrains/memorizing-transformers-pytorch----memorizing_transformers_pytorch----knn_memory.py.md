# `.\lucidrains\memorizing-transformers-pytorch\memorizing_transformers_pytorch\knn_memory.py`

```py
# 导入必要的库
import os
import math
import torch
import faiss
import numpy as np
from pathlib import Path
from functools import wraps

# 导入上下文管理器相关的库
from contextlib import ExitStack, contextmanager

# 导入 einops 库
from einops import rearrange, pack, unpack

# 导入 multiprocessing 相关库
from joblib import Parallel, delayed, cpu_count

# 定义常量
FAISS_INDEX_GPU_ID = int(os.getenv('FAISS_INDEX_GPU_ID', 0))
DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY = './.tmp/knn.memories'

# 定义一些辅助函数

# 检查变量是否存在
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将变量转换为列表
def cast_list(val):
    return val if isinstance(val, list) else [val]

# 检查数组中的元素是否全部唯一
def all_el_unique(arr):
    return len(set(arr)) == len(arr)

# 定义一个多上下文管理器
@contextmanager
def multi_context(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]

# 计算两个数组的交集
def count_intersect(x, y):
    return np.sum(rearrange(x, 'i -> i 1') == rearrange(y, 'j -> 1 j'), axis = -1)

# 检查张量的形状是否符合指定的模式
def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)

# 定义一个 KNN 类，封装了 faiss IndexIVFFlat，并自动处理旧键的过期
class KNN():
    def __init__(
        self,
        dim,
        max_num_entries,
        cap_num_entries = False,
        M = 15,
        keep_stats = False
    ):
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index = index
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries
        self.is_trained = False
        self.keep_stats = keep_stats

        self.reset()

    def __del__(self):
        if hasattr(self, 'index'):
            del self.index

    def reset(self):
        self.ids = np.empty((0,), dtype = np.int32)

        if self.keep_stats:
            self.hits = np.empty((0,), dtype = np.int32)
            self.age_num_iterations = np.empty((0,), dtype = np.int32)
            self.ages_since_last_hit = np.empty((0,), dtype = np.int32)

        self.index.reset()
        self.is_trained = False

    def train(self, x):
        self.index.train(x)
        self.is_trained = True

    def add(self, x, ids):
        if not self.is_trained:
            self.train(x)

        self.ids = np.concatenate((ids, self.ids))

        if self.keep_stats:
            self.hits = np.concatenate((np.zeros_like(ids), self.hits))
            self.age_num_iterations = np.concatenate((np.zeros_like(ids), self.age_num_iterations))
            self.ages_since_last_hit = np.concatenate((np.zeros_like(ids), self.ages_since_last_hit))

        if self.cap_num_entries and len(self.ids) > self.max_num_entries:
            self.reset()

        return self.index.add(x)

    def search(
        self,
        x,
        topk,
        nprobe = 8,
        return_distances = False,
        increment_hits = False,
        increment_age = True
    ):
        if not self.is_trained:
            return np.full((x.shape[0], topk), -1)

        distances, indices = self.index.search(x, k = topk)

        if increment_hits and self.keep_stats:
            hits = count_intersect(self.ids, rearrange(indices, '... -> (...)'))
            self.hits += hits

            self.ages_since_last_hit += 1
            self.ages_since_last_hit *= (hits == 0)

        if increment_age and self.keep_stats:
            self.age_num_iterations += 1

        if return_distances:
            return indices, distances

        return indices

# 定义一个 KNNMemory 类，用于存储键/值记忆，可以自动处理一组 faiss 索引（跨批次维度）
class KNNMemory():
    def __init__(
        self,
        dim,
        max_memories = 16000,
        num_indices = 1,
        memmap_filename = './knn.memory.memmap',
        multiprocessing = True
    # 初始化方法，设置对象的维度、索引数量、索引范围、最大内存、形状、数据库偏移量等属性
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.scoped_indices = list(range(num_indices))

        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype = np.int32)

        # 创建一个内存映射对象，用于存储数据
        self.db = np.memmap(memmap_filename, mode = 'w+', dtype = np.float32, shape = self.shape)
        # 创建一个 KNN 对象列表
        self.knns = [KNN(dim = dim, max_num_entries = max_memories, cap_num_entries = True) for _ in range(num_indices)]
    
        # 根据是否使用多进程设置并行任务数
        self.n_jobs = cpu_count() if multiprocessing else 1

    # 设置作用域索引
    def set_scoped_indices(self, indices):
        indices = list(indices)
        # 检查索引是否唯一
        assert all_el_unique(indices), f'all scoped batch indices must be unique, received: {indices}'
        # 检查索引范围是否在有效范围内
        assert all([0 <= i < self.num_indices for i in indices]), f'each batch index must be between 0 and less than {self.num_indices}: received {indices}'
        self.scoped_indices = indices

    # 上下文管理器，用于设置作用域索引
    @contextmanager
    def at_batch_indices(self, indices):
        prev_indices = self.scoped_indices
        self.set_scoped_indices(indices)
        yield self
        self.set_scoped_indices(prev_indices)

    # 清空指定批次的数据
    def clear(self, batch_indices = None):
        if not exists(batch_indices):
            batch_indices = list(range(self.num_indices))

        batch_indices = cast_list(batch_indices)

        # 重置指定批次的 KNN 对象
        for index in batch_indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[batch_indices] = 0

    # 添加新的记忆数据
    def add(self, memories):
        # 检查记忆数据的形状
        check_shape(memories, 'b n kv d', d = self.dim, kv = 2, b = len(self.scoped_indices))

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1]

        knn_insert_ids = np.arange(num_memories)

        keys = np.ascontiguousarray(memories[..., 0, :])
        knns = [self.knns[i] for i in self.scoped_indices]
        db_offsets = [self.db_offsets[i] for i in self.scoped_indices]

        # 使用 joblib 将新的键/值记忆插入到 faiss 索引中

        @delayed
        def knn_add(knn, key, db_offset):
            knn.add(key, ids = knn_insert_ids + db_offset)
            return knn

        updated_knns = Parallel(n_jobs = self.n_jobs)(knn_add(*args) for args in zip(knns, keys, db_offsets))
        for knn_idx, scoped_idx in enumerate(self.scoped_indices):
            self.knns[scoped_idx] = updated_knns[knn_idx]

        # 将新的记忆数据添加到内存映射的数据库中

        add_indices = (rearrange(np.arange(num_memories), 'j -> 1 j') + rearrange(self.db_offsets[list(self.scoped_indices)], 'i -> i 1')) % self.max_memories
        self.db[rearrange(np.array(self.scoped_indices), 'i -> i 1'), add_indices] = memories
        self.db.flush()

        self.db_offsets += num_memories

    # 搜索方法，用于查询最近邻
    def search(
        self,
        queries,
        topk,
        nprobe = 8,
        increment_hits = True,
        increment_age = True
        ):
        # 检查查询数据的形状是否符合要求
        check_shape(queries, 'b ... d', d = self.dim, b = len(self.scoped_indices))
        # 将查询数据打包成指定格式
        queries, ps = pack([queries], 'b * d')

        # 获取查询数据的设备信息
        device = queries.device
        # 将查询数据转换为 numpy 数组
        queries = queries.detach().cpu().numpy()

        # 初始化空列表用于存储掩码和键值对
        all_masks = []
        all_key_values = []

        # 获取指定索引处的 knn 对象
        knns = [self.knns[i] for i in self.scoped_indices]

        # 并行化 faiss 搜索

        @delayed
        def knn_search(knn, query):
            return knn.search(query, topk, nprobe, increment_hits = increment_hits, increment_age = increment_age)

        # 并行执行 knn_search 函数，获取搜索结果
        fetched_indices = Parallel(n_jobs = self.n_jobs)(knn_search(*args) for args in zip(knns, queries))

        # 从内存映射 'database' 中获取所有的键/值对
        # 待办事项 - 移除下面的 for 循环

        for batch_index, indices in zip(self.scoped_indices, fetched_indices):
            # 创建掩码，将无效索引替换为 0
            mask = indices !=  -1
            db_indices = np.where(mask, indices, 0)

            # 将掩码转换为 PyTorch 张量并添加到列表中
            all_masks.append(torch.from_numpy(mask))

            # 获取键值对并添加到列表中
            key_values = self.db[batch_index, db_indices % self.max_memories]
            all_key_values.append(torch.from_numpy(key_values))

        # 将所有掩码和键值对堆叠成张量
        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        # 使用掩码填充键值对中的无效值为 0
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        # 拆分键值对张量
        all_key_values, = unpack(all_key_values, ps, 'b * n kv d')
        all_masks, = unpack(all_masks, ps, 'b * n')

        # 返回结果并将其发送到指定设备
        return all_key_values.to(device), all_masks.to(device)

    def __del__(self):
        # 在对象销毁时，删除 knns 和 db 属性
        if hasattr(self, 'knns'):
            for knn in self.knns:
                del knn
        del self.db
# 为 KNN 记忆集合扩展了一些额外的方法

class KNNMemoryList(list):
    # 清理方法，用于清理所有记忆
    def cleanup(self):
        for memory in self:
            del memory

    # 创建记忆方法，用于创建多个记忆对象
    @classmethod
    def create_memories(
        self,
        *,
        batch_size,
        num_memory_layers,
        memories_directory = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY
    ):
        # 设置记忆路径
        memories_path = Path(memories_directory)
        memories_path.mkdir(exist_ok = True, parents = True)

        # 内部方法，用于创建多个记忆对象
        def inner(*args, **kwargs):
            return self([KNNMemory(*args, num_indices = batch_size, memmap_filename = str(memories_path / f'knn.memory.layer.{ind + 1}.memmap'), **kwargs) for ind in range(num_memory_layers)])
        return inner

    # 批量索引上下文管理器，用于在多个记忆对象上进行批量索引
    @contextmanager
    def at_batch_indices(
        self,
        indices
    ):
        knn_batch_indices_contexts = [memory.at_batch_indices(indices) for memory in self]
        with multi_context(*knn_batch_indices_contexts):
            yield

    # 清除记忆方法，用于清除指定的记忆对象
    def clear_memory(
        self,
        batch_indices = None,
        memory_indices = None
    ):
        # 默认情况下清除所有记忆对象
        memory_indices = default(memory_indices, tuple(range(len(self)))

        # 遍历指定的记忆对象，清除指定的批次索引
        for memory_index in memory_indices:
            memory = self[memory_index]
            memory.clear(batch_indices)
```