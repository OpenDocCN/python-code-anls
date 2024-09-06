# `.\HippoRAG\src\data_process\beir.py`

```py
import argparse  # 导入用于处理命令行参数的 argparse 模块
import json  # 导入用于处理 JSON 数据的 json 模块
import os  # 导入用于操作文件和目录的 os 模块


def subset_relevant_corpus_statistics(subset_path: str, split: str):
    relevant_corpus_ids = set()  # 初始化一个集合，用于存储相关的 corpus_id
    # 构造包含 'qrels/{split}.tsv' 文件的路径，并检查文件是否存在
    if os.path.isfile(os.path.join(subset_path, f'qrels/{split}.tsv')):
        # 打开文件并读取每一行
        with open(os.path.join(subset_path, f'qrels/{split}.tsv')) as f:
            for line in f:
                # 跳过 TSV 文件的第一行（标题行）
                if line.startswith('query-id'):
                    continue
                # 将每行数据拆分成 query_id, corpus_id 和 score
                query_id, corpus_id, score = line.strip().split('\t')
                # 如果 score 为 0，跳过此条记录
                if int(score) == 0:
                    continue
                # 将相关的 corpus_id 添加到集合中
                relevant_corpus_ids.add(corpus_id)
        # 返回相关 corpus_id 的数量
        return len(relevant_corpus_ids)
    else:
        # 如果文件不存在，则返回 None
        return None


def subset_statistics(subset_path: str):
    # 读取 corpus.jsonl 文件中的所有数据
    full_corpus = []  # 初始化一个列表，用于存储完整的 corpus 数据
    # 打开 corpus.jsonl 文件并读取每一行
    with open(os.path.join(subset_path, 'corpus.jsonl')) as f:
        for line in f:
            item = json.loads(line)  # 将 JSON 字符串解析为字典
            # 提取需要的字段并将其添加到 full_corpus 列表中
            full_corpus.append({'title': item['title'], 'text': item['text'], 'idx': item['_id']})

    # 计算每个分割集的相关 corpus 的数量
    len_train_corpus = subset_relevant_corpus_statistics(subset_path, 'train')
    len_dev_corpus = subset_relevant_corpus_statistics(subset_path, 'dev')
    len_test_corpus = subset_relevant_corpus_statistics(subset_path, 'test')

    # 打印统计信息，包括路径、全 corpus 的长度及各分割集的相关 corpus 数量
    print(f'{subset_path[subset_path.find("beir/"):]}\t{len(full_corpus)}\t{len_train_corpus}\t{len_dev_corpus}\t{len_test_corpus}')


if __name__ == '__main__':
    # 创建 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加命令行参数 --data，指定 BEIR 根目录路径
    parser.add_argument('--data', type=str, help='directory path to BEIR root')
    # 解析命令行参数
    args = parser.parse_args()

    # 确保提供的目录路径是一个有效的目录
    assert os.path.isdir(args.data)
    # 遍历根目录下的每个子目录
    for subset_name in os.listdir(args.data):
        # 检查子目录是否为目录且不为空
        if not os.path.isdir(os.path.join(args.data, subset_name)):
            continue
        if len(os.listdir(os.path.join(args.data, subset_name))) == 0:
            continue
        # 如果子目录中包含 'corpus.jsonl' 文件，计算统计信息
        if 'corpus.jsonl' in os.listdir(os.path.join(args.data, subset_name)):
            subset_statistics(os.path.join(args.data, subset_name))
        else:
            # 否则，检查子目录中的第二级目录
            for second_subset_name in os.listdir(os.path.join(args.data, subset_name)):
                subset_statistics(os.path.join(args.data, subset_name, second_subset_name))
```