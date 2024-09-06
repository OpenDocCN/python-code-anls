# `.\HippoRAG\src\baselines\mean_pooling_ip_faiss.py`

```py
# 导入操作系统模块
import os
# 导入系统模块
import sys

# 从 src.processing 导入 mean_pooling 和 mean_pooling_embedding_with_normalization 函数
from src.processing import mean_pooling, mean_pooling_embedding_with_normalization

# 将当前目录添加到系统路径中，以便导入模块
sys.path.append('.')

# 导入命令行参数解析模块
import argparse
# 导入 JSON 处理模块
import json
# 导入 NumPy 模块
import numpy as np

# 导入 FAISS 模块（用于高效相似性搜索）
import faiss
# 导入 PyTorch 模块
import torch
# 导入进度条模块
from tqdm import tqdm
# 导入 Hugging Face 的 Transformer 库中的模型和分词器
from transformers import AutoTokenizer, AutoModel

# 如果是主程序运行
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数：模型名称
    parser.add_argument('--model', type=str, default='facebook/contriever')
    # 添加参数：向量维度
    parser.add_argument('--dim', type=int, default=768)
    # 添加参数：数据集名称
    parser.add_argument('--dataset', type=str)
    # 添加参数：单位类型（'hippo' 或 'proposition'）
    parser.add_argument('--unit', type=str, default='hippo', choices=['hippo', 'proposition'])
    # 解析命令行参数
    args = parser.parse_args()

    # 获取向量维度
    dim = args.dim
    # 设置是否进行归一化
    norm = True
    # 根据模型名称生成标签
    model_label = args.model.replace('/', '_').replace('.', '_')
    # 根据是否归一化设置向量和索引文件路径
    if norm:
        vector_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_vectors_norm.npy'
        index_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_ip_norm.index'
    else:
        vector_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_vectors.npy'
        index_path = f'data/{args.dataset}/{args.dataset}_{model_label}_{args.unit}_ip.index'

    # 初始化语料库内容列表
    corpus_contents = []
    # 根据数据集和单位类型加载对应的语料库
    if args.dataset == 'hotpotqa':
        if args.unit == 'hippo':
            # 读取 'hotpotqa_corpus.json' 文件
            corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
            # 拼接标题和文本，添加到语料库内容列表
            for title, text in corpus.items():
                corpus_contents.append(title + '\n' + '\n'.join(text))
        elif args.unit == 'proposition':
            # 读取 'hotpotqa_proposition_corpus.json' 文件
            corpus = json.load(open('data/dense_x_retrieval/hotpotqa_proposition_corpus.json', 'r'))
    elif args.dataset == 'musique':
        if args.unit == 'hippo':
            # 读取 'musique_corpus.json' 文件
            corpus = json.load(open('data/musique_corpus.json', 'r'))
            # 拼接标题和文本，添加到语料库内容列表
            corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
        elif args.unit == 'proposition':
            # 读取 'musique_proposition_corpus.json' 文件
            corpus = json.load(open('data/dense_x_retrieval/musique_proposition_corpus.json', 'r'))
    elif args.dataset == '2wikimultihopqa':
        if args.unit == 'hippo':
            # 读取 '2wikimultihopqa_corpus.json' 文件
            corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
            # 拼接标题和文本，添加到语料库内容列表
            corpus_contents = [item['title'] + '\n' + item['text'] for item in corpus]
        elif args.unit == 'proposition':
            # 读取 '2wikimultihopqa_proposition_corpus.json' 文件
            corpus = json.load(open('data/dense_x_retrieval/2wikimultihopqa_proposition_corpus.json', 'r'))

    # 如果单位类型是 'proposition'，拼接标题和提案，添加到语料库内容列表
    if args.unit == 'proposition':
        for item in corpus:
            corpus_contents.append(item['title'] + '\n' + item['propositions'])

    # 打印语料库大小
    print('corpus size: {}'.format(len(corpus_contents)))

    # 如果向量文件存在
    if os.path.isfile(vector_path):
        # 打印正在加载的向量文件路径
        print('Loading existing vectors:', vector_path)
        # 加载现有向量
        vectors = np.load(vector_path)
        # 打印已加载向量的数量
        print('Vectors loaded:', len(vectors))
    else:
        # 加载模型
        tokenizer = AutoTokenizer.from_pretrained(args.model)  # 根据预训练模型参数实例化分词器
        model = AutoModel.from_pretrained(args.model)  # 根据预训练模型参数实例化模型

        # 检查是否有多个 GPU 可用，若有，则全部使用
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")  # 打印使用的 GPU 数量
            model = torch.nn.DataParallel(model)  # 多 GPU 并行处理
        model.to('cuda')  # 将模型移至 GPU

        # 为了提高效率，对文本进行批处理编码
        batch_size = 16 * torch.cuda.device_count()  # 定义批处理大小
        vectors = np.zeros((len(corpus_contents), dim))  # 初始化存储编码向量的数组
        for start_idx in tqdm(range(0, len(corpus_contents), batch_size), desc='encoding corpus'):
            end_idx = min(start_idx + batch_size, len(corpus_contents))  # 计算当前批次的结束索引
            batch_passages = corpus_contents[start_idx:end_idx]  # 获取当前批次的文本内容

            if norm:  # 如果需要进行标准化
                batch_embeddings = mean_pooling_embedding_with_normalization(batch_passages, tokenizer, model)
            else:  # 否则
                try:
                    inputs = tokenizer(batch_passages, padding=True, truncation=True, return_tensors='pt', max_length=384).to('cuda')  # 使用分词器对文本进行处理
                    outputs = model(**inputs)  # 模型推理
                    batch_embeddings = mean_pooling(outputs[0], inputs['attention_mask'])  # 计算平均池化的编码向量

                except Exception as e:  # 捕获异常
                    batch_embeddings = torch.zeros((len(batch_passages), dim))  # 初始化为零的编码向量
                    print(f'Error at {start_idx}:', e)  # 打印错误信息

            vectors[start_idx:end_idx] = batch_embeddings.to('cpu').detach().numpy()  # 将批次的编码向量存储到数组中

        # 将编码向量保存到文件
        np.save(vector_path, vectors)  # 保存编码向量
        print('vectors saved to {}'.format(vector_path))  # 打印保存的位置

    # 为语料库构建内积索引
    if os.path.isfile(index_path):  # 如果索引文件已经存在
        print('index file already exists:', index_path)  # 打印索引文件已存在的提示信息
        print('index size: {}'.format(faiss.read_index(index_path).ntotal))  # 打印索引大小
    else:  # 否则
        print('Building index...')  # 打印构建索引的提示信息
        index = faiss.IndexFlatIP(dim)  # 创建内积索引
        vectors = vectors.astype('float32')  # 将向量数据类型转换为 float32
        index.add(vectors)  # 向索引中添加向量

        # 将 faiss 索引保存到文件
        faiss.write_index(index, index_path)  # 保存 faiss 索引
        print('index saved to {}'.format(index_path))  # 打印保存索引的位置
        print('index size: {}'.format(index.ntotal))  # 打印索引大小
```