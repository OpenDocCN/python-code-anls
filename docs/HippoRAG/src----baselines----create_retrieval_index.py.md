# `.\HippoRAG\src\baselines\create_retrieval_index.py`

```py
# 导入操作系统接口模块
import os
# 导入系统特定参数和功能模块
import sys

# 将当前目录添加到系统路径中，以便可以导入本地模块
sys.path.append('.')

# 导入命令行参数解析模块
import argparse
# 导入 JSON 操作模块
import json
# 导入时间处理模块
import time

# 导入 Faiss 库，用于高效相似度搜索
import faiss
# 导入 NumPy 库，用于数值计算
import numpy as np
# 导入 PyTorch 库，用于深度学习
import torch
# 导入 Elasticsearch 库，用于与 Elasticsearch 进行交互
from elasticsearch import Elasticsearch
# 导入 SentenceTransformers 库，用于句子嵌入生成
from sentence_transformers import SentenceTransformer
# 导入进度条模块，用于显示处理进度
from tqdm import tqdm

# 从本地模块中导入创建和索引的函数
from src.elastic_search_tool import create_and_index

# 主程序入口点
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数，用于指定语料库的名称
    parser.add_argument('--corpus', type=str)
    # 添加参数，用于指定数据集的名称
    parser.add_argument('--dataset', type=str)
    # 添加参数，用于指定检索器的类型
    parser.add_argument('--retriever', type=str)
    # 添加参数，用于指定嵌入的维度
    parser.add_argument('--dim', type=int)
    # 添加参数，用于指定单位类型，并设置默认值为 'hippo'
    parser.add_argument('--unit', type=str, choices=['hippo', 'proposition'], default='hippo')
    # 解析命令行参数
    args = parser.parse_args()

    # 设置是否归一化的标志
    norm = True

    # 根据指定的语料库名称加载对应的语料库数据
    if args.corpus == 'hotpotqa_1000':
        # 从 JSON 文件中加载语料库数据
        corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        # 根据语料库大小生成索引名称
        index_name = f'hotpotqa_{len(corpus)}_bm25'
        # 处理语料库内容为每条记录的标题和文本的拼接
        corpus_contents = [title + '\n' + ''.join(text) for title, text in corpus.items()]
    elif args.corpus == 'hotpotqa_1000_proposition':
        # 从 JSON 文件中加载语料库数据
        corpus = json.load(open('data/dense_x_retrieval/hotpotqa_proposition_corpus.json', 'r'))
        # 根据语料库大小生成索引名称
        index_name = f'hotpotqa_{len(corpus)}_proposition_bm25'
        # 处理语料库内容为每条记录的标题和提案的拼接
        corpus_contents = [item['title'] + '\n' + item['propositions'] for item in corpus]
    elif args.corpus == 'musique_1000':
        # 从 JSON 文件中加载语料库数据
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        # 根据语料库大小生成索引名称
        index_name = f'musique_{len(corpus)}_bm25'
        # 处理语料库内容为每条记录的标题和文本的拼接
        corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
    elif args.corpus == 'musique_1000_proposition':
        # 从 JSON 文件中加载语料库数据
        corpus = json.load(open('data/dense_x_retrieval/musique_proposition_corpus.json', 'r'))
        # 根据语料库大小生成索引名称
        index_name = f'musique_{len(corpus)}_proposition_bm25'
        # 处理语料库内容为每条记录的标题和提案的拼接
        corpus_contents = [item['title'] + '\n' + item['propositions'] for item in corpus]
    elif args.corpus == '2wikimultihopqa_1000':
        # 从 JSON 文件中加载语料库数据
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        # 根据语料库大小生成索引名称
        index_name = f'2wikimultihopqa_{len(corpus)}_bm25'
        # 处理语料库内容为每条记录的标题和文本的拼接
        corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
    elif args.corpus == '2wikimultihopqa_1000_proposition':
        # 从 JSON 文件中加载语料库数据
        corpus = json.load(open('data/dense_x_retrieval/2wikimultihopqa_proposition_corpus.json', 'r'))
        # 根据语料库大小生成索引名称
        index_name = f'2wikimultihopqa_{len(corpus)}_proposition_bm25'
        # 处理语料库内容为每条记录的标题和提案的拼接
        corpus_contents = [item['title'] + '\n' + item['propositions'] for item in corpus]
    else:
        # 如果语料库名称不匹配，则抛出未实现错误
        raise NotImplementedError('Invalid corpus name')

    # 如果检索器类型为 'bm25'
    if args.retriever == 'bm25':
        # 记录开始时间
        start_time = time.time()
        # 打印正在创建索引的消息
        print('Creating index', index_name)
        # 创建 Elasticsearch 实例，连接到本地 Elasticsearch 服务
        es = Elasticsearch([{"host": "localhost", "port": 9200, "scheme": "http"}])
        # 使用创建和索引函数将语料库数据索引到 Elasticsearch
        create_and_index(es, index_name, corpus_contents, 'BM25')
        # 打印索引创建完成的消息及消耗的时间
        print('BM25 index created, consumed time:', round(time.time() - start_time, 2))
    # 如果 retriever 的前缀是 'sentence-transformers/'
    elif args.retriever.startswith('sentence-transformers/'):
        # 将 retriever 的路径分隔符 '/' 和 '.' 替换为 '_'
        retriever_label = args.retriever.replace('/', '_').replace('.', '_')
        # 根据 norm 标志决定文件路径，包含归一化与否
        if norm:
            vector_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_vectors_norm.npy'
            index_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_ip_norm.index'
        else:
            vector_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_vectors.npy'
            index_path = f'data/{args.dataset}/{args.dataset}_{retriever_label}_{args.unit}_ip.index'

        # 使用 SentenceTransformer 加载模型，并将其移至 GPU
        model = SentenceTransformer(args.retriever).to('cuda')

        # 设置批处理大小，以提高处理效率
        batch_size = 16 * torch.cuda.device_count()
        # 初始化一个零矩阵用于存储向量
        vectors = np.zeros((len(corpus_contents), args.dim))
        # 逐批处理语料库
        for start_idx in tqdm(range(0, len(corpus_contents), batch_size), desc='encoding corpus'):
            end_idx = min(start_idx + batch_size, len(corpus_contents))
            batch_passages = corpus_contents[start_idx:end_idx]

            try:
                # 编码当前批次的文本
                batch_embeddings = model.encode(batch_passages)
                if norm:
                    # 计算并应用归一化
                    norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                    batch_embeddings = batch_embeddings / norms
            except Exception as e:
                # 如果发生错误，记录错误并设置当前批次的向量为零
                batch_embeddings = torch.zeros((len(batch_passages), args.dim))
                print(f'Error at {start_idx}:', e)

            # 将批处理结果存储到总向量矩阵中
            vectors[start_idx:end_idx] = batch_embeddings

        # 将向量保存到文件
        np.save(vector_path, vectors)
        print('vectors saved to {}'.format(vector_path))

        # 为语料库构建内积索引
        if os.path.isfile(index_path):
            # 如果索引文件已存在，则读取并报告索引大小
            print('index file already exists:', index_path)
            print('index size: {}'.format(faiss.read_index(index_path).ntotal))
        else:
            print('Building index...')
            # 创建一个内积索引，并将向量添加到索引中
            index = faiss.IndexFlatIP(args.dim)
            vectors = vectors.astype('float32')
            index.add(vectors)

            # 将索引保存到文件
            faiss.write_index(index, index_path)
            print('index saved to {}'.format(index_path))
            print('index size: {}'.format(index.ntotal))

    # 如果 retriever 的前缀不是指定的值，抛出未实现的错误
    else:
        raise NotImplementedError('Invalid retriever name')
```