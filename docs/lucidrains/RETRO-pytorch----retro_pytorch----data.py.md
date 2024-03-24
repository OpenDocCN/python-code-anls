# `.\lucidrains\RETRO-pytorch\retro_pytorch\data.py`

```py
# 导入所需的库
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset

# 导入自定义的模块
from retro_pytorch.retrieval import EOS_ID
from retro_pytorch.utils import memmap

# 定义函数 knn_to_retrieved_chunks，用于将 KNN 结果转换为检索到的块
def knn_to_retrieved_chunks(
    knns,
    chunks_memmap,
    *,
    add_continuations,
    num_chunks,
    pad_id = 0,
    eos_id = EOS_ID,
):

    # 推导出没有找到邻居的掩码
    no_neighbor_mask = knns == -1
    knns = np.maximum(knns, 0)

    # 获取邻居和连续块
    knn_chunks = chunks_memmap[knns]
    is_last_document_chunk = np.any(knn_chunks == eos_id, axis = -1, keepdims = True)

    # 使用 [EOS] 在块中的存在作为检测文档边界的方式
    retrieved = knn_chunks[..., :-1]

    if add_continuations:
        continuation_indices = np.clip(knns + 1, 0, num_chunks - 1) # 块是连续存储的
        continuation_chunks = chunks_memmap[continuation_indices][..., :-1]
        continuation_chunks *= ~is_last_document_chunk

        # 将邻居与连续块合并
        retrieved = np.concatenate((retrieved, continuation_chunks), axis = -1)

    # 将任何最近邻块为 -1（在索引时未找到）的掩码为填充 ID
    retrieved = np.where(~no_neighbor_mask[..., None], retrieved, pad_id)
    return retrieved

# 定义类 RETRODataset，继承自 Dataset 类
class RETRODataset(Dataset):
    def __init__(
        self,
        *,
        num_chunks,
        chunk_size,
        seq_len,
        num_sequences,
        num_neighbors,
        chunk_memmap_path,
        chunk_nn_memmap_path,
        seq_memmap_path,
        eos_id = EOS_ID,
        pad_id = 0.,
        add_continuations = True
    ):
        super().__init__()
        self.num_chunks = num_chunks
        self.num_sequences = num_sequences
        self.seq_num_chunks = seq_len // chunk_size
        self.eos_id = eos_id
        self.pad_id = pad_id

        num_chunks_with_padding = num_chunks + self.seq_num_chunks

        chunks_shape = (num_chunks_with_padding, chunk_size + 1)
        knn_shape = (num_chunks_with_padding, num_neighbors)

        self.add_continuations = add_continuations
        self.get_chunks = partial(memmap, chunk_memmap_path, dtype = np.int32, shape = chunks_shape)
        self.get_knns = partial(memmap, chunk_nn_memmap_path, dtype = np.int32, shape = knn_shape)
        self.get_seqs = partial(memmap, seq_memmap_path, dtype = np.int32, shape = (num_sequences,))

    # 返回数据集的长度
    def __len__(self):
        return self.num_sequences

    # 获取数据集中指定索引的数据
    def __getitem__(self, ind):
        with self.get_chunks() as chunks_memmap, self.get_knns() as knns_memmap, self.get_seqs() as seqs_memmap:
            begin_chunk_index = seqs_memmap[ind]
            chunk_range = slice(begin_chunk_index, (begin_chunk_index + self.seq_num_chunks))

            chunks = chunks_memmap[chunk_range]

            # 剪切最后一个标记，除了最后一个块的最后一个标记
            seq_tokens = np.concatenate((chunks[:, :-1].flatten(), chunks[-1, -1:]))

            # 掩码掉（使用填充标记）任何跟在 <eos> 后的标记 | 不允许一个序列中有多个文档，因为这会破坏 RETRO 的 CCA
            seq_mask = np.cumsum(seq_tokens == self.eos_id, axis = 0)
            seq_mask = np.pad(seq_mask, (1, 0))[:-1] == 0.
            seq_tokens = np.where(seq_mask, seq_tokens, 0.)

            # 推导出检索到的标记
            knns = knns_memmap[chunk_range]

            retrieved = knn_to_retrieved_chunks(
                knns,
                chunks_memmap,
                add_continuations = self.add_continuations,
                eos_id = self.eos_id,
                num_chunks = self.num_chunks
            )

        seq_tokens_torch = torch.from_numpy(seq_tokens).long()
        retrieved_torch = torch.from_numpy(retrieved).long()
        return seq_tokens_torch, retrieved_torch
```