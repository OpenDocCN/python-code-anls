# `.\HippoRAG\src\colbertv2_knn.py`

```py
import argparse  # 导入用于解析命令行参数的模块
import json  # 导入用于处理 JSON 数据的模块
import os.path  # 导入用于处理路径的模块

import ipdb  # 导入 IPython 调试器
import pandas as pd  # 导入 pandas 数据处理库
from colbert import Indexer, Searcher  # 从 colbert 包导入索引器和搜索器
from colbert.infra import Run, RunConfig, ColBERTConfig  # 从 colbert.infra 导入运行和配置相关的类
from colbert.data import Queries  # 从 colbert.data 导入查询类
from processing import *  # 从 processing 模块导入所有功能
import pickle  # 导入用于序列化和反序列化 Python 对象的模块


def retrieve_knn(kb, queries, duplicate=True, nns=100):
    checkpoint_path = 'exp/colbertv2.0'  # 定义检查点路径

    if duplicate:
        kb = list(set(list(kb) + list(queries)))  # 如果 duplicate 为真，合并 kb 和 queries 并去重，准备进行查询和查询之间的评分

    with open('data/lm_vectors/colbert/corpus.tsv', 'w') as f:  # 打开并准备写入 corpus.tsv 文件
        for pid, p in enumerate(kb):
            f.write(f"{pid}\t\"{p}\"" + '\n')  # 将每个 kb 元素以 "ID\t内容" 格式写入文件

    with open('data/lm_vectors/colbert/queries.tsv', 'w') as f:  # 打开并准备写入 queries.tsv 文件
        for qid, q in enumerate(queries):
            f.write(f"{qid}\t{q}" + '\n')  # 将每个查询以 "ID\t查询内容" 格式写入文件

    ranking_output_path = 'nbits_2_ranking.tsv'  # 定义排名输出路径
    # index
    with Run().context(RunConfig(nranks=1, experiment="colbert", root="")):  # 使用 ColBERT 配置上下文管理器
        config = ColBERTConfig(
            nbits=2,  # 设置位数为 2
            root="data/lm_vectors/colbert"  # 指定数据根路径
        )
        indexer = Indexer(checkpoint=checkpoint_path, config=config)  # 创建 Indexer 实例
        indexer.index(name="nbits_2", collection="data/lm_vectors/colbert/corpus.tsv", overwrite=True)  # 创建索引

    # retrieval
    with Run().context(RunConfig(nranks=1, experiment="colbert", root="")):  # 使用 ColBERT 配置上下文管理器
        config = ColBERTConfig(
            root="data/lm_vectors/colbert",  # 指定数据根路径
        )
        searcher = Searcher(index="nbits_2", config=config)  # 创建 Searcher 实例
        queries = Queries("data/lm_vectors/colbert/queries.tsv")  # 加载查询数据
        ranking = searcher.search_all(queries, k=nns)  # 执行查询，获取最近邻排名

    ranking_dict = {}  # 创建一个空字典来存储排名结果

    for i in range(len(queries)):  # 遍历每个查询
        query = queries[i]  # 获取查询内容
        rank = ranking.data[i]  # 获取该查询的排名数据
        max_score = rank[0][2]  # 获取最高分数以便进行归一化
        if duplicate:
            rank = rank[1:]  # 如果需要重复，去掉第一个（查询对查询的评分）
        ranking_dict[query] = ([kb[r[0]] for r in rank], [r[2] / max_score for r in rank])  # 将结果存储到字典中

    return ranking_dict  # 返回排名结果字典


if __name__ == '__main__':  # 如果这个文件作为主程序运行
    parser = argparse.ArgumentParser()  # 创建 ArgumentParser 实例
    parser.add_argument('--filename', type=str)  # 添加命令行参数 --filename，类型为字符串
    args = parser.parse_args()  # 解析命令行参数

    string_filename = args.filename  # 获取文件名参数

    # prepare tsv data
    string_df = pd.read_csv(string_filename, sep='\t')  # 从 TSV 文件读取数据到 DataFrame
    string_df.strings = [processing_phrases(str(s)) for s in string_df.strings]  # 对 DataFrame 中的每个字符串进行处理

    queries = string_df[string_df.type == 'query']  # 从 DataFrame 中提取查询数据
    kb = string_df[string_df.type == 'kb']  # 从 DataFrame 中提取知识库数据

    nearest_neighbors = retrieve_knn(kb.strings.values, queries.strings.values)  # 调用 retrieve_knn 函数获取最近邻
    output_path = 'data/lm_vectors/colbert/nearest_neighbor_{}.p'.format(string_filename.split('/')[1].split('.')[0])  # 定义输出路径
    pickle.dump(nearest_neighbors, open(output_path, 'wb'))  # 将最近邻结果序列化并保存到文件
    print('Saved nearest neighbors to {}'.format(output_path))  # 打印保存路径
```