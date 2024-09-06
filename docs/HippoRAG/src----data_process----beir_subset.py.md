# `.\HippoRAG\src\data_process\beir_subset.py`

```py
# 导入需要的模块
import argparse
import json
import os.path
# 从 src.data_process.util 导入 chunk_corpus 函数
from src.data_process.util import chunk_corpus

# 定义一个函数用于生成包含相关语料库的数据集
def generate_dataset_with_relevant_corpus(split: str, qrels_path: str, chunk=False):
    """

    @param split: split name, e.g., 'train', 'test'
    @param qrels_path: the path to BEIR subset's qrels file
    @return: None
    """
    # 根据 chunk 参数确定文件名
    chunk_state = '_chunk' if chunk else ''
    # 打开 qrels 文件，读取数据
    with open(qrels_path) as f:
        qrels = f.readlines()
    # 通过空格分割每条数据，组成列表
    qrels = [q.split() for q in qrels[1:]]
    # 打印 split 和相关数据的数量
    print(f'#{split}', len(qrels))
    # 初始化相关变量
    split_corpus = []
    split_corpus_ids = set()
    split_queries = []
    split_query_ids = set()
    query_to_corpus = {}  # query_id -> [corpus_id]

    # 遍历 qrels 列表中的每个元素
    for idx, item in enumerate(qrels):
        query_id = item[0]
        corpus_id = item[1]
        score = item[2]
        # 如果分数等于 0，则跳过
        if int(score) == 0:
            continue
        # 尝试从语料库中获取相应的数据
        try:
            corpus_item = corpus[corpus_id]
        except KeyError:
            print(f'corpus_id {corpus_id} not found')
            continue
        query_item = queries[query_id]

        # 如果语料库中的 id 不在 split_corpus_ids 中，则添加到 split_corpus 中
        if corpus_item['_id'] not in split_corpus_ids:
            split_corpus.append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})
            split_corpus_ids.add(corpus_item['_id'])
        # 如果查询中的 id 不在 split_query_ids 中，则添加到 split_queries 中
        if query_item['_id'] not in split_query_ids:
            split_queries.append({**query_item, 'id': query_item['_id'], 'question': query_item['text']})
            split_query_ids.add(query_item['_id'])
        # 如果 query_id 不存在于 query_to_corpus 中，则添加到 query_to_corpus 中
        if query_id not in query_to_corpus:
            query_to_corpus[query_id] = {}
        query_to_corpus[query_id][corpus_id] = int(score)

    # 为查询信息添加支持段落
    for query in split_queries:
        query['paragraphs'] = []
        for c in query_to_corpus[query['_id']]:
            corpus_item = corpus[c]
            query['paragraphs'].append({'title': corpus_item['title'], 'text': corpus_item['text'], 'idx': corpus_item['_id']})

    # 如果 chunk 为真，则对 split_corpus 进行分块处理
    if chunk:
        split_corpus = chunk_corpus(split_corpus)
    # 保存 split_corpus
    corpus_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_corpus.json'
    with open(corpus_output_path, 'w') as f:
        json.dump(split_corpus, f)
        print(f'{split} corpus saved to {corpus_output_path}, len: {len(split_corpus)}')

    # 保存 split_queries
    queries_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_queries.json'
    with open(queries_output_path, 'w') as f:
        json.dump(split_queries, f)
        print(f'{split} queries saved to {queries_output_path}, len: {len(split_queries)}')

    # 保存处理后的 qrel json 文件
    qrels_output_path = f'data/beir_{subset_name}_{split}{chunk_state}_qrel.json'
    with open(qrels_output_path, 'w') as f:
        json.dump(query_to_corpus, f)
        print(f'{split} qrels saved to {qrels_output_path}, len: {len(query_to_corpus)}')


def generate_dataest_with_full_corpus(split, qrels_path: str, chunk=False):
    chunk_state = '_chunk' if chunk else ''
    # 打开 qrels_path 文件，并将其内容按行读取到 qrels 列表中
# 检查脚本是否作为主程序运行
if __name__ == '__main__':
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个命令行参数，用于指定 BEIR 子集的目录路径
    parser.add_argument('--data', type=str, help='directory path to a BEIR subset')
    # 添加一个命令行参数，用于指定语料库类型，取值为 'full' 或 'relevant'
    parser.add_argument('--corpus', type=str, choices=['full', 'relevant'], help='full or relevant corpus', default='full')
    # 添加一个布尔值命令行参数，用于指定是否分块处理数据
    parser.add_argument('--chunk', action='store_true')
    # 解析命令行参数并将结果存储在 args 中
    args = parser.parse_args()

    # 打印解析后的命令行参数
    print(args)
    # 从数据目录路径中提取子集名称
    subset_name = args.data.split('/')[-1]
    # 打开并读取 queries.jsonl 文件中的每一行
    with open(os.path.join(args.data, 'queries.jsonl')) as f:
        queries = f.readlines()
    # 将每一行 JSON 格式的数据解析成 Python 字典，并创建一个以 '_id' 为键的字典
    queries = [json.loads(q) for q in queries]
    queries = {q['_id']: q for q in queries}

    # 打开并读取 corpus.jsonl 文件中的每一行
    with open(os.path.join(args.data, 'corpus.jsonl')) as f:
        corpus = f.readlines()
    # 将每一行 JSON 格式的数据解析成 Python 字典，并创建一个以 '_id' 为键的字典
    corpus = [json.loads(c) for c in corpus]
    corpus = {c['_id']: c for c in corpus}

    # 遍历数据集中的 'train', 'dev', 'test' 三个部分
    for split in ['train', 'dev', 'test']:
        # 检查 qrels 目录下是否存在对应的 .tsv 文件
        if os.path.isfile(os.path.join(args.data, f'qrels/{split}.tsv')):
            # 根据命令行参数指定的语料库类型，调用相应的函数生成数据集
            if args.corpus == 'relevant':
                generate_dataset_with_relevant_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), args.chunk)
            elif args.corpus == 'full':
                generate_dataest_with_full_corpus(split, os.path.join(args.data, f'qrels/{split}.tsv'), args.chunk)
        else:
            # 如果 .tsv 文件不存在，打印提示信息
            print(f'{split} not found, skipped')
```