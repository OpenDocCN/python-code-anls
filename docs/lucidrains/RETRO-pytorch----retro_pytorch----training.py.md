# `.\lucidrains\RETRO-pytorch\retro_pytorch\training.py`

```py
import numpy as np
from functools import partial
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from retro_pytorch import RETRO, RETRODataset
from retro_pytorch.data import knn_to_retrieved_chunks
from retro_pytorch.optimizer import get_optimizer
from retro_pytorch.retrieval import text_folder_to_chunks_, chunks_to_precalculated_knn_, bert_embed, SOS_ID, EOS_ID
from retro_pytorch.utils import memmap, is_true_env_flag

from einops import rearrange

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 评估装饰器，用于在评估时切换模型状态
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 安全拼接张量
def safe_cat(accum, t, dim = -1):
    if not exists(accum):
        return t
    return torch.cat((accum, t), dim = dim)

# sampling helpers

# 对数函数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 从 Gumbel 噪声中采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

# Top-K 采样
def top_k(logits, thres = 0.9):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# Top-P 采样
def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# 从序列块获取 KNN 块的函数
def knn_chunks_from_seq_chunks(
    seq_chunks,
    *,
    knn,
    faiss_index,
    num_chunks,
    chunk_size,
    chunks_memmap_path,
):
    b, device = seq_chunks.shape[0], seq_chunks.device

    # 为 BERT 嵌入准备带有 SOS 和 EOS 标记的最后一个块

    ones = torch.ones((b, 1), dtype = torch.bool, device = device)
    sos = ones * SOS_ID
    eos = ones * EOS_ID

    seq_chunks = torch.cat((sos, seq_chunks, eos), dim = 1)

    # 使用冻结的 BERT 进行嵌入

    embeds = bert_embed(seq_chunks.cpu()) # 暂时在 CPU 上获取嵌入

    # 使用 faiss 检索 KNN

    _, knn_indices = faiss_index.search(embeds.cpu().numpy(), k = knn)

    # numpy 转换为 torch

    with memmap(chunks_memmap_path, dtype = np.int32, shape = (num_chunks + 1, chunk_size + 1)) as chunk_memmap:
        knn_chunks = knn_to_retrieved_chunks(
            knn_indices,
            chunk_memmap,
            add_continuations = True,
            num_chunks = num_chunks
        )

        knn_chunks_torch = torch.from_numpy(knn_chunks).to(device)

    return knn_chunks_torch

# 训练包装类
class TrainingWrapper(nn.Module):
    def __init__(
        self,
        *,
        retro,
        chunk_size,
        documents_path,
        knn,
        glob = '**/*.txt',
        chunks_memmap_path = './train.chunks.dat',
        seqs_memmap_path = './train.seq.dat',
        doc_ids_memmap_path = './train.doc_ids.dat',
        max_chunks = 1_000_000,
        max_seqs = 100_000,
        knn_extra_neighbors = 100,
        processed_stats_json_path = './processed-stats.json',
        faiss_index_filename = 'knn.index',
        **index_kwargs
    # 初始化 RETROGenerator 类
    def __init__(
        self,
        retro: RETRO,
        processed_stats_json_path: str,
        documents_path: str,
        chunks_memmap_path: str,
        seqs_memmap_path: str,
        doc_ids_memmap_path: str,
        chunk_size: int,
        max_chunks: int,
        max_seqs: int,
        knn: int,
        knn_extra_neighbors: int,
        faiss_index_filename: str,
        **index_kwargs
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 断言 retro 必须是 RETRO 类的实例
        assert isinstance(retro, RETRO), 'retro must be instance of RETRO'
        # 将 retro 赋值给 self.retro
        self.retro = retro

        # 检查是否需要强制重新处理数据
        force_reprocess = is_true_env_flag('REPROCESS')

        # 存储处理后的训练数据统计信息，如块数和序列数
        stats_path = Path(processed_stats_json_path)

        # 如果统计文件不存在或需要强制重新处理，则处理文本文件夹
        if not stats_path.exists() or force_reprocess:
            # 调用函数处理文本文件夹，返回统计信息
            self.stats = text_folder_to_chunks_(
                folder = documents_path,
                glob = glob,
                chunks_memmap_path = chunks_memmap_path,
                seqs_memmap_path = seqs_memmap_path,
                doc_ids_memmap_path = doc_ids_memmap_path,
                chunk_size = chunk_size,
                seq_len = retro.seq_len,
                max_chunks = max_chunks,
                max_seqs = max_seqs
            )
            # 将统计信息写入 JSON 文件
            with open(processed_stats_json_path, 'w') as f:
                json.dump(self.stats, f)
        else:
            # 如果统计文件已经存在，则加载已处理的统计信息
            print(f'found to be previously processed at {str(stats_path)}')
            self.stats = json.loads(stats_path.read_text())

        # 获取块数和序列数
        num_chunks = self.stats['chunks']
        num_seqs = self.stats['seqs']

        # 计算 knn 的内存映射路径并获取 faiss 索引
        knn_memmap_path, faiss_index = chunks_to_precalculated_knn_(
            num_chunks = num_chunks,
            chunk_size = chunk_size,
            chunk_memmap_path = chunks_memmap_path,
            doc_ids_memmap_path = doc_ids_memmap_path,
            num_nearest_neighbors = knn,
            num_extra_neighbors = knn_extra_neighbors,
            index_file = faiss_index_filename,
            force_reprocess = force_reprocess,
            **index_kwargs
        )

        # 初始化 RETRODataset 类
        self.ds = RETRODataset(
            num_sequences = num_seqs,
            num_chunks = num_chunks,
            num_neighbors = knn,
            chunk_size = chunk_size,
            seq_len = retro.seq_len,
            chunk_memmap_path = chunks_memmap_path,
            chunk_nn_memmap_path = knn_memmap_path,
            seq_memmap_path = seqs_memmap_path
        )

        # 生成所需的参数
        self.chunk_size = chunk_size
        self.max_seq_len = self.retro.seq_len

        # 部分函数，用于从序列块中获取 knn 块
        self.fetch_knn_chunks_fn = partial(
            knn_chunks_from_seq_chunks,
            knn = knn,
            chunk_size = chunk_size,
            num_chunks = num_chunks,
            chunks_memmap_path = chunks_memmap_path,
            faiss_index = faiss_index
        )

    # 生成文本的方法，使用装饰器进行评估
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start = None,
        retrieved = None,
        filter_fn = top_k,
        filter_thres = 0.9,
        temperature = 1.0,
    ):
        # 断言过滤函数必须是top-k或nucleus
        assert filter_fn in {top_k, top_p}, 'filter function must be either top-k or nucleus'

        # 获取设备信息
        device = next(self.retro.parameters()).device

        # 如果没有给定起始标记，则假设从SOS标记开始，批量大小为1
        if not exists(start):
            start = torch.full((1, 1), SOS_ID, device=device).long()

        b, start_seq_len = start.shape

        # 将起始标记移动到与RETRO相同的设备上
        start = start.to(device)

        # 准备检索相关变量
        if start_seq_len >= self.chunk_size:
            seq_index = (start_seq_len // self.chunk_size) * self.chunk_size
            past_seq_chunks = rearrange(start[:, :seq_index], 'b (n c) -> (b n) c', c=self.chunk_size)

            # 获取KNN块
            retrieved = self.fetch_knn_chunks_fn(past_seq_chunks)
            retrieved = rearrange(retrieved, '(b n) k c -> b n k c', b=b)

        # 获取起始序列索引
        out = start

        # 采样循环
        for i in range(start_seq_len - 1, self.max_seq_len):

            logits = self.retro(out, retrieved=retrieved)
            logits = logits[:, i]

            logits = filter_fn(logits, thres=filter_thres)
            sampled = gumbel_sample(logits, temperature=temperature, dim=-1)
            sampled = rearrange(sampled, 'b -> b 1')

            out = torch.cat((out, sampled), dim=1)

            # 如果全部是EOS标记，则提前终止
            is_eos_tokens = (out == EOS_ID)

            if is_eos_tokens.any(dim=-1).all():

                # 在EOS标记后屏蔽所有内容
                shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                mask = shifted_is_eos_tokens.float().cumsum(dim=-1) >= 1
                out = out.masked_fill(mask, self.retro.pad_id)
                break

            # 当序列长度是块大小的倍数时，检索下一组KNN
            curr_seq_len = out.shape[-1]

            if (curr_seq_len % self.chunk_size) == 0:
                last_chunk = rearrange(out, 'b (c n) -> b c n', n=self.chunk_size)[:, -1]

                knn_chunks = self.fetch_knn_chunks_fn(last_chunk)

                # 将检索到的KNN块连接到所有检索到的内容中
                # 以便在下一次迭代中发送到Retro进行块交叉注意力
                knn_chunks = rearrange(knn_chunks, 'b k r -> b 1 k r')
                retrieved = safe_cat(retrieved, knn_chunks, dim=1)

                print(f'retrieved at {curr_seq_len} / {self.max_seq_len}')

        return out

    # 获取数据加载器
    def get_dataloader(self, **kwargs):
        return DataLoader(self.ds, **kwargs)

    # 获取优化器
    def get_optimizer(self, **kwargs):
        return get_optimizer(self.retro.parameters(), **kwargs)

    # 前向传播函数
    def forward(self):
        raise NotImplemented
```