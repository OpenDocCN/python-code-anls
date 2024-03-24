# `.\lucidrains\RETRO-pytorch\retro_pytorch\retrieval.py`

```py
# 导入所需的模块
from pathlib import Path
from math import ceil

import torch
import torch.nn.functional as F
import logging
import numpy as np
from einops import rearrange

import faiss
from autofaiss import build_index

from retro_pytorch.utils import memmap, reset_folder_

# 常量定义

SOS_ID = 101
EOS_ID = 102
BERT_MODEL_DIM = 768
BERT_VOCAB_SIZE = 28996

TMP_PATH = Path('./.tmp')
INDEX_FOLDER_PATH = TMP_PATH / '.index'
EMBEDDING_TMP_SUBFOLDER = 'embeddings'

# 辅助函数

def exists(val):
    return val is not None

def range_chunked(max_value, *, batch_size):
    counter = 0
    while counter < max_value:
        curr = counter + batch_size
        curr = min(curr, max_value)
        yield slice(counter, curr)
        counter = curr

# 索引辅助函数

def faiss_read_index(path):
    return faiss.read_index(str(path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)

# 单例全局变量

MODEL = None
TOKENIZER = None

def get_tokenizer():
    global TOKENIZER
    if not exists(TOKENIZER):
        TOKENIZER = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
    return TOKENIZER

def get_bert():
    global MODEL
    if not exists(MODEL):
        MODEL = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')
        if torch.cuda.is_available():
            MODEL = MODEL.cuda()

    return MODEL

# 分词

def tokenize(texts, add_special_tokens = True):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]

    tokenizer = get_tokenizer()

    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens = add_special_tokens,
        padding = True,
        return_tensors = 'pt'
    )

    token_ids = encoding.input_ids
    return token_ids

# 文本转换为块和序列索引

def doc_text_to_chunks_and_seq_indices(
    *,
    doc_text,
    chunk_size = 64,
    seq_len = 2048,
    pad_id = 0
):
    assert (seq_len % chunk_size) == 0, 'sequence length must be divisible by chunk size'

    ids = tokenize(doc_text)
    ids = rearrange(ids, '1 ... -> ...')

    text_len = ids.shape[-1]

    # 用额外的标记填充到块大小的倍数

    padding = chunk_size - ((text_len - 1) % chunk_size)
    ids = F.pad(ids, (0, padding))

    # 分离最后一个标记

    ids, last_token = ids[:-1], ids[-1:]
    ids = rearrange(ids, '(n c) -> n c', c = chunk_size)

    # 块的第一个标记[2:]及之后的标记将成为块[1:]的最后一个标记

    last_token_per_chunk = ids[1:, 0]
    all_last_tokens = torch.cat((last_token_per_chunk, last_token), dim = 0)
    all_last_tokens = rearrange(all_last_tokens, 'n -> n 1')

    # 将所有最后一个标记附加到块中，形成(num_chunks, chunk_size + 1)

    chunks_with_extra_token = torch.cat((ids, all_last_tokens), dim = -1)

    # 计算从0开始的块索引，间隔为序列长度的块数

    total_chunks = ids.shape[0]
    num_chunks_per_seq = seq_len // chunk_size
    seq = torch.arange(0, total_chunks, num_chunks_per_seq)

    return chunks_with_extra_token, seq

def text_folder_to_chunks_(
    *,
    folder,
    chunks_memmap_path,
    seqs_memmap_path,
    doc_ids_memmap_path,
    chunk_size = 64,
    seq_len = 2048,
    glob = '**/*.txt',
    max_chunks = 1_000_000,
    max_seqs = 100_000
):
    paths = sorted([*Path(folder).glob(glob)])

    total_chunks = 0
    total_docs = 0
    total_seqs = 0

    chunks_shape = (max_chunks, chunk_size + 1)
    seqs_shape = (max_seqs,)
    doc_ids_shape = (max_chunks,)
    # 使用上下文管理器打开三个内存映射文件，分别用于存储chunks、seqs和doc_ids
    with memmap(chunks_memmap_path, shape = chunks_shape, dtype = np.int32, mode = 'w+') as chunks_memmap\
        , memmap(seqs_memmap_path, shape = seqs_shape, dtype = np.int32, mode = 'w+') as seqs_memmap\
        , memmap(doc_ids_memmap_path, shape = doc_ids_shape, dtype = np.int32, mode = 'w+') as doc_ids_memmap:

        # 遍历所有路径
        for path in paths:
            # 打印当前处理的路径
            print(f'processing {path}')

            # 将文档文本转换为chunks和seq的索引
            chunks, seq = doc_text_to_chunks_and_seq_indices(
                doc_text = path.read_text(),
                chunk_size = chunk_size,
                seq_len = seq_len
            )

            # 获取当前文档的chunks和seq的长度
            doc_chunk_len = chunks.shape[0]
            doc_seq_len = seq.shape[0]

            # 将当前文档的chunks写入chunks内存映射文件
            chunks_memmap[total_chunks:(total_chunks + doc_chunk_len)] = chunks.numpy()
            # 将当前文档的seq索引写入seqs内存映射文件，并加上之前文档的总chunks数
            seqs_memmap[total_seqs:(total_seqs + doc_seq_len)] = seq.numpy() + total_chunks
            # 将当前文档的doc_ids写入doc_ids内存映射文件，使用当前文档的总chunks数填充
            doc_ids_memmap[total_chunks:(total_chunks + doc_chunk_len)] = np.full((doc_chunk_len,), total_docs)

            # 更新总chunks、总seqs和总docs数
            total_chunks += doc_chunk_len
            total_seqs += doc_seq_len
            total_docs += 1

    # 返回包含总chunks、总docs和总seqs数的字典
    return dict(
        chunks = total_chunks,
        docs = total_docs,
        seqs = total_seqs
    )
# 嵌入函数

@torch.no_grad()
def bert_embed(
    token_ids,
    return_cls_repr = False,
    eps = 1e-8,
    pad_id = 0.
):
    # 获取 BERT 模型
    model = get_bert()
    # 创建掩码，标记不是填充符的位置
    mask = token_ids != pad_id

    # 如果有可用的 GPU，则将数据移至 GPU
    if torch.cuda.is_available():
        token_ids = token_ids.cuda()
        mask = mask.cuda()

    # 使用 BERT 模型进行前向传播
    outputs = model(
        input_ids = token_ids,
        attention_mask = mask,
        output_hidden_states = True
    )

    # 获取最后一个隐藏状态
    hidden_state = outputs.hidden_states[-1]

    # 如果需要返回 [cls] 作为表示，则返回 [cls] 的隐藏状态
    if return_cls_repr:
        return hidden_state[:, 0]

    # 如果没有掩码存在，则计算所有 token 的平均值
    if not exists(mask):
        return hidden_state.mean(dim = 1)

    # 更新掩码，排除 [cls]，考虑长度
    mask = mask[:, 1:]
    mask = rearrange(mask, 'b n -> b n 1')

    # 计算加权平均值
    numer = (hidden_state[:, 1:] * mask).sum(dim = 1)
    denom = mask.sum(dim = 1)
    masked_mean =  numer / (denom + eps)
    return masked_mean

# 将块转换为 KNN

def chunks_to_embeddings_(
    *,
    num_chunks,
    chunks_memmap_path,
    embeddings_memmap_path,
    chunk_size = 64,
    embed_dim = BERT_MODEL_DIM,
    batch_size = 16,
    use_cls_repr = False,
    pad_id = 0.
):
    chunks_shape = (num_chunks, chunk_size + 1)
    embed_shape = (num_chunks, embed_dim)

    # 使用内存映射加载块和嵌入
    with memmap(chunks_memmap_path, shape = chunks_shape, dtype = np.int32) as chunks\
        , memmap(embeddings_memmap_path, shape = embed_shape, dtype = np.float32, mode = 'w+') as embeddings:

        # 对块进行分批处理
        for dim_slice in range_chunked(num_chunks, batch_size = batch_size):
            batch_chunk_npy = chunks[dim_slice]

            batch_chunk = torch.from_numpy(batch_chunk_npy)

            cls_tokens = torch.full((batch_chunk.shape[0], 1), SOS_ID)
            batch_chunk = torch.cat((cls_tokens, batch_chunk), dim = 1)

            batch_chunk = batch_chunk[:, :-1] # 省略最后一个 token，下一个块的第一个 token，用于自回归训练

            # 获取块的嵌入
            batch_embed = bert_embed(
                batch_chunk,
                return_cls_repr = use_cls_repr
            )

            # 将嵌入写入内存映射
            embeddings[dim_slice] = batch_embed.detach().cpu().numpy()
            print(f'embedded {dim_slice.stop} / {num_chunks}')


def memmap_file_to_chunks_(
    memmap_path,
    *,
    folder,
    shape,
    dtype,
    max_rows_per_file = 500
):
    rows, _ = shape

    # 使用内存映射将文件分割为块并保存
    with memmap(memmap_path, shape = shape, dtype = dtype, mode = 'r') as f:
        root_path = TMP_PATH / folder
        reset_folder_(root_path)

        for ind, dim_slice in enumerate(range_chunked(rows, batch_size = max_rows_per_file)):
            filename = root_path / f'{ind:05d}.npy'
            data_slice = f[dim_slice]

            np.save(str(filename), f[dim_slice])
            print(f'saved {str(filename)}')

def index_embeddings(
    embeddings_folder,
    *,
    index_file = 'knn.index',
    index_infos_file = 'index_infos.json',
    max_index_memory_usage = '100m',
    current_memory_available = '1G'
):
    embeddings_path = TMP_PATH / embeddings_folder
    index_path = INDEX_FOLDER_PATH / index_file

    reset_folder_(INDEX_FOLDER_PATH)

    # 构建索引
    build_index(
        embeddings = str(embeddings_path),
        index_path = str(index_path),
        index_infos_path = str(INDEX_FOLDER_PATH / index_infos_file),
        metric_type = "l2",
        max_index_memory_usage = max_index_memory_usage,
        current_memory_available = current_memory_available,
        make_direct_map = True,
        should_be_memory_mappable = False,
        use_gpu = torch.cuda.is_available(),
    )

    # 读取索引
    index = faiss_read_index(index_path)
    return index

def chunks_to_index_and_embed(
    *,
    num_chunks,
    chunk_size,
    chunk_memmap_path,
    use_cls_repr = False,
    max_rows_per_file = 500,
    chunks_to_embeddings_batch_size = 16,
    embed_dim = BERT_MODEL_DIM,
    index_file = 'knn.index',
    **index_kwargs
):
    embedding_path = f'{chunk_memmap_path}.embedded'
    embed_shape = (num_chunks, embed_dim)
    # 将数据分块转换为嵌入向量
    chunks_to_embeddings_(
        num_chunks = num_chunks,  # 数据分块的数量
        chunk_size = chunk_size,  # 每个数据分块的大小
        chunks_memmap_path = chunk_memmap_path,  # 数据分块的内存映射路径
        embeddings_memmap_path = embedding_path,  # 嵌入向量的内存映射路径
        use_cls_repr = use_cls_repr,  # 是否使用分类表示
        batch_size = chunks_to_embeddings_batch_size,  # 转换为嵌入向量的批处理大小
        embed_dim = embed_dim  # 嵌入向量的维度
    )

    # 将内存映射文件转换为数据分块
    memmap_file_to_chunks_(
        embedding_path,  # 嵌入向量的内存映射路径
        shape = embed_shape,  # 嵌入向量的形状
        dtype = np.float32,  # 数据类型为32位浮点数
        folder = EMBEDDING_TMP_SUBFOLDER,  # 数据分块存储的文件夹
        max_rows_per_file = max_rows_per_file  # 每个文件的最大行数
    )

    # 对嵌入向量进行索引
    index = index_embeddings(
        embeddings_folder = EMBEDDING_TMP_SUBFOLDER,  # 嵌入向量存储的文件夹
        index_file = index_file,  # 索引文件
        **index_kwargs  # 其他索引参数
    )

    # 从内存映射文件中读取嵌入向量
    embeddings = np.memmap(embedding_path, shape = embed_shape, dtype = np.float32, mode = 'r')
    # 返回索引和嵌入向量
    return index, embeddings
# 定义一个函数，用于将数据划分为预先计算的 KNN（K-Nearest Neighbors）索引
def chunks_to_precalculated_knn_(
    *,
    num_nearest_neighbors,  # 最近邻居的数量
    num_chunks,  # 数据块的数量
    chunk_size,  # 数据块的大小
    chunk_memmap_path,  # 数据块的内存映射路径
    doc_ids_memmap_path,  # 文档 ID 的内存映射路径
    use_cls_repr = False,  # 是否使用分类表示
    max_rows_per_file = 500,  # 每个文件的最大行数
    chunks_to_embeddings_batch_size = 16,  # 数据块到嵌入的批处理大小
    embed_dim = BERT_MODEL_DIM,  # 嵌入维度
    num_extra_neighbors = 10,  # 额外的邻居数量
    force_reprocess = False,  # 是否强制重新处理
    index_file = 'knn.index',  # 索引文件名
    **index_kwargs  # 其他索引参数
):
    # 获取数据块的路径
    chunk_path = Path(chunk_memmap_path)
    # 获取 KNN 文件的路径
    knn_path = chunk_path.parents[0] / f'{chunk_path.stem}.knn{chunk_path.suffix}'
    # 获取索引文件的路径
    index_path = INDEX_FOLDER_PATH / index_file

    # 如果索引文件和 KNN 文件存在且不需要强制重新处理，则直接返回 KNN 文件路径和 Faiss 索引
    if index_path.exists() and knn_path.exists() and not force_reprocess:
        print(f'preprocessed knn found at {str(knn_path)}, faiss index reconstituted from {str(index_path)}')
        index = faiss_read_index(index_path)
        return knn_path, index

    # 获取 Faiss 索引和数据块的嵌入
    index, embeddings = chunks_to_index_and_embed(
        num_chunks = num_chunks,
        chunk_size = chunk_size,
        chunk_memmap_path = chunk_memmap_path,
        index_file = index_file,
        **index_kwargs
    )

    # 计算需要获取的总邻居数
    total_neighbors_to_fetch = num_extra_neighbors + num_nearest_neighbors + 1

    # 使用内存映射创建 KNN 和文档 ID 的数组
    with memmap(knn_path, shape = (num_chunks, num_nearest_neighbors), dtype = np.int32, mode = 'w+') as knns\
        , memmap(doc_ids_memmap_path, shape = (num_chunks,), dtype = np.int32, mode = 'r') as doc_ids:

        # 对数据块进行分片处理
        for dim_slice in range_chunked(num_chunks, batch_size = max_rows_per_file):
            # 获取查询向量
            query_vector = embeddings[dim_slice]

            # 使用索引查找最近邻居
            distances, indices = index.search(query_vector, k = total_neighbors_to_fetch)

            # 移除自身作为邻居
            distances = distances[:, 1:]
            indices = indices[:, 1:]

            # 将属于同一文档的邻居标记为 -1
            query_doc_ids = doc_ids[dim_slice]
            neighbor_doc_ids = doc_ids[indices]
            neighbor_from_same_doc = query_doc_ids[..., None] == neighbor_doc_ids

            indices = np.where(neighbor_from_same_doc, -1, indices)
            distances = np.where(neighbor_from_same_doc, 1e3, distances)

            # 根据更新后的距离重新排序索引
            indices = np.take_along_axis(indices, np.argsort(distances, axis = 1), axis = 1)

            # 将最近邻居存储到 KNN 内存映射中
            knns[dim_slice] = indices[:, :num_nearest_neighbors]

            print(f'knns calculated for {dim_slice.stop} / {num_chunks}')

    # 打印 KNN 文件保存路径
    print(f'knn saved to {knn_path}')
    return knn_path, index
```