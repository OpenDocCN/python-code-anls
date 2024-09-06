# `.\HippoRAG\src\create_graph.py`

```py
# 导入模块
import copy  # 用于创建对象的深拷贝
import pandas as pd  # 用于数据处理和分析
from scipy.sparse import csr_array  # 用于处理稀疏矩阵
from processing import *  # 导入 processing 模块中的所有内容
from glob import glob  # 用于文件模式匹配

import os  # 用于操作系统相关功能
import json  # 用于处理 JSON 数据
from tqdm import tqdm  # 用于显示进度条
import pickle  # 用于序列化和反序列化对象
import argparse  # 用于处理命令行参数

# 设置环境变量以禁用并行化的 tokenizers
os.environ['TOKENIZERS_PARALLELISM'] = 'FALSE'

# 定义函数来创建图
def create_graph(dataset: str, extraction_type: str, extraction_model: str, retriever_name: str, processed_retriever_name: str, threshold: float = 0.9,
                 create_graph_flag: bool = False, cosine_sim_edges: bool = False):
    # 设置版本号
    version = 'v3'
    # 设置三元组的权重
    inter_triple_weight = 1.0
    # 设置相似度的最大值
    similarity_max = 1.0
    # 根据给定的参数格式化并找到所有符合条件的 JSON 文件
    possible_files = glob('output/openie_{}_results_{}_{}_*.json'.format(dataset, extraction_type, extraction_model))
    # 从找到的文件中提取最大的样本数
    max_samples = np.max([int(file.split('{}_'.format(extraction_model))[1].split('.json')[0]) for file in possible_files])
    # 读取指定文件中的 JSON 数据
    extracted_file = json.load(open('output/openie_{}_results_{}_{}_{}.json'.format(dataset, extraction_type, extraction_model, max_samples), 'r'))

    # 从 JSON 数据中提取三元组
    extracted_triples = extracted_file['docs']
    # 如果使用的模型不是 'gpt-3.5-turbo-1106'，则更新提取类型
    if extraction_model != 'gpt-3.5-turbo-1106':
        extraction_type = extraction_type + '_' + extraction_model
    # 设置短语类型为仅实体、小写、预处理
    phrase_type = 'ents_only_lower_preprocess'
    # 根据是否使用余弦相似度边来决定图的类型
    if cosine_sim_edges:
        graph_type = 'facts_and_sim'  # 包含提取的事实和相似短语
    else:
        graph_type = 'facts'  # 仅包含提取的事实

    # 初始化各种用于存储数据的列表和字典
    passage_json = []  # 用于存储文章的 JSON 数据
    phrases = []  # 用于存储短语
    entities = []  # 用于存储实体
    relations = {}  # 用于存储关系
    incorrectly_formatted_triples = []  # 用于存储格式不正确的三元组
    triples_wo_ner_entity = []  # 用于存储没有命名实体的三元组
    triple_tuples = []  # 用于存储三元组的元组
    full_neighborhoods = {}  # 用于存储完整的邻域
    correct_wiki_format = 0  # 记录正确的维基格式数量
    # 遍历提取的三元组，并显示进度条
    for i, row in tqdm(enumerate(extracted_triples), total=len(extracted_triples)):
        # 从行中提取文本内容
        document = row['passage']
        # 从行中提取原始实体
        raw_ner_entities = row['extracted_entities']
        # 处理提取的实体短语
        ner_entities = [processing_phrases(p) for p in row['extracted_entities']]

        # 从行中提取三元组
        triples = row['extracted_triples']

        # 将当前行数据保存到 JSON 对象中
        doc_json = row

        # 初始化干净和不干净的三元组列表，以及实体集合
        clean_triples = []
        unclean_triples = []
        doc_entities = set()

        # 从 OpenIE 生成三元组并处理
        for triple in triples:

            # 将三元组中的每个元素转换为字符串
            triple = [str(s) for s in triple]

            # 检查三元组的长度
            if len(triple) > 1:
                # 如果三元组长度不为 3，则为不正确格式
                if len(triple) != 3:
                    # 处理三元组的短语
                    clean_triple = [processing_phrases(p) for p in triple]

                    # 将不正确格式的三元组添加到列表
                    incorrectly_formatted_triples.append(triple)
                    unclean_triples.append(triple)
                else:
                    # 处理三元组的短语
                    clean_triple = [processing_phrases(p) for p in triple]

                    # 将干净的三元组添加到列表
                    clean_triples.append(clean_triple)
                    phrases.extend(clean_triple)

                    # 获取三元组的头部和尾部实体
                    head_ent = clean_triple[0]
                    tail_ent = clean_triple[2]

                    # 如果头部和尾部实体都不在NER实体中，则添加到无NER实体三元组列表
                    if head_ent not in ner_entities and tail_ent not in ner_entities:
                        triples_wo_ner_entity.append(triple)

                    # 将头部和尾部实体及关系添加到关系字典中
                    relations[(head_ent, tail_ent)] = clean_triple[1]

                    # 获取原始头部和尾部实体
                    raw_head_ent = triple[0]
                    raw_tail_ent = triple[2]

                    # 更新头部实体的邻域
                    entity_neighborhood = full_neighborhoods.get(raw_head_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_head_ent] = entity_neighborhood

                    # 更新尾部实体的邻域
                    entity_neighborhood = full_neighborhoods.get(raw_tail_ent, set())
                    entity_neighborhood.add((raw_head_ent, triple[1], raw_tail_ent))
                    full_neighborhoods[raw_tail_ent] = entity_neighborhood

                    # 将三元组中的实体添加到实体列表和文档实体集合中
                    for triple_entity in [clean_triple[0], clean_triple[2]]:
                        entities.append(triple_entity)
                        doc_entities.add(triple_entity)

        # 将实体、干净三元组和不干净三元组添加到文档 JSON 中
        doc_json['entities'] = list(set(doc_entities))
        doc_json['clean_triples'] = clean_triples
        doc_json['noisy_triples'] = unclean_triples
        # 将干净的三元组添加到三元组元组列表中
        triple_tuples.append(clean_triples)

        # 将文档 JSON 添加到通道 JSON 列表中
        passage_json.append(doc_json)

    # 打印正确 Wiki 格式的三元组数量
    print('Correct Wiki Format: {} out of {}'.format(correct_wiki_format, len(extracted_triples)))
    # 尝试读取名为指定数据集的 TSV 文件
    try:
        queries_full = pd.read_csv('output/{}_queries.named_entity_output.tsv'.format(dataset), sep='\t')

        # 如果数据集包含 'hotpotqa'，则读取相应的 JSON 文件
        if 'hotpotqa' in args.dataset:
            queries = json.load(open(f'data/{args.dataset}.json', 'r'))
            # 提取问题列表
            questions = [q['question'] for q in queries]
            # 将 '0' 列设置为索引
            queries_full = queries_full.set_index('0', drop=False)
        else:
            # 对于其他数据集，从 JSON 文件读取数据
            queries_df = pd.read_json(f'data/{args.dataset}.json')
            # 提取问题列表
            questions = queries_df['question'].values
            # 将 'question' 列设置为索引
            queries_full = queries_full.set_index('question', drop=False)
            # 根据问题筛选查询数据
            queries_full = queries_full.loc[questions]

        # 依据问题筛选查询数据
        queries_full = queries_full.loc[questions]
    except:
        # 如果读取文件失败，则创建一个空的 DataFrame
        queries_full = pd.DataFrame([], columns=['question', 'triples'])
    # 初始化实体列表和每文档实体列表
    q_entities = []
    q_entities_by_doc = []
    # 遍历查询数据中的 triples 列
    for doc_ents in tqdm(queries_full.triples):
        # 解析 JSON 字符串并提取命名实体
        doc_ents = eval(doc_ents)['named_entities']
        try:
            # 清洗每个实体
            clean_doc_ents = [processing_phrases(p) for p in doc_ents]
        except:
            # 如果清洗失败，则使用空列表
            clean_doc_ents = []
        # 扩展实体列表和每文档实体列表
        q_entities.extend(clean_doc_ents)
        q_entities_by_doc.append(clean_doc_ents)
    # 获取唯一的短语和关系
    unique_phrases = list(np.unique(entities))
    unique_relations = np.unique(list(relations.values()) + ['equivalent'])
    q_phrases = list(np.unique(q_entities))
    # 复制唯一短语列表并扩展查询短语
    all_phrases = copy.deepcopy(unique_phrases)
    all_phrases.extend(q_phrases)
    # 将唯一短语列表转为 DataFrame
    kb = pd.DataFrame(unique_phrases, columns=['strings'])
    kb2 = copy.deepcopy(kb)
    # 为 DataFrame 添加类型标签
    kb['type'] = 'query'
    kb2['type'] = 'kb'
    # 合并 DataFrame 并保存为 TSV 文件
    kb_full = pd.concat([kb, kb2])
    kb_full.to_csv('output/kb_to_kb.tsv', sep='\t')
    # 将唯一关系列表转为 DataFrame
    rel_kb = pd.DataFrame(unique_relations, columns=['strings'])
    rel_kb2 = copy.deepcopy(rel_kb)
    # 为 DataFrame 添加类型标签
    rel_kb['type'] = 'query'
    rel_kb2['type'] = 'kb'
    # 合并 DataFrame 并保存为 TSV 文件
    rel_kb_full = pd.concat([rel_kb, rel_kb2])
    rel_kb_full.to_csv('output/rel_kb_to_kb.tsv', sep='\t')
    # 将查询短语转为 DataFrame
    query_df = pd.DataFrame(q_phrases, columns=['strings'])
    query_df['type'] = 'query'
    kb['type'] = 'kb'
    # 合并知识库和查询短语 DataFrame 并保存为 TSV 文件
    kb_query = pd.concat([kb, query_df])
    kb_query.to_csv('output/query_to_kb.tsv', sep='\t')
# 当脚本作为主程序运行时执行以下代码
if __name__ == '__main__':
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个字符串类型的参数，用于输入数据集的路径或名称
    parser.add_argument('--dataset', type=str)
    # 添加一个字符串类型的参数，用于输入模型名称
    parser.add_argument('--model_name', type=str)
    # 添加一个字符串类型的参数，用于输入提取模型的名称
    parser.add_argument('--extraction_model', type=str)
    # 添加一个浮点类型的参数，用于输入阈值
    parser.add_argument('--threshold', type=float)
    # 添加一个布尔型的开关参数，用于决定是否创建图
    parser.add_argument('--create_graph', action='store_true')
    # 添加一个字符串类型的参数，用于输入提取类型
    parser.add_argument('--extraction_type', type=str)
    # 添加一个布尔型的开关参数，用于决定是否使用余弦相似度边
    parser.add_argument('--cosine_sim_edges', action='store_true')

    # 解析命令行参数
    args = parser.parse_args()
    # 获取数据集参数的值
    dataset = args.dataset
    # 获取模型名称参数的值
    retriever_name = args.model_name
    # 处理模型名称，替换斜杠和点，以便文件系统兼容
    processed_retriever_name = retriever_name.replace('/', '_').replace('.', '')
    # 处理提取模型名称，替换斜杠，以便文件系统兼容
    extraction_model = args.extraction_model.replace('/', '_')
    # 获取阈值参数的值
    threshold = args.threshold
    # 获取是否创建图的标志
    create_graph_flag = args.create_graph
    # 获取提取类型参数的值
    extraction_type = args.extraction_type
    # 获取是否使用余弦相似度边的标志
    cosine_sim_edges = args.cosine_sim_edges

    # 调用创建图的函数，传入所有解析得到的参数
    create_graph(dataset, extraction_type, extraction_model, retriever_name, processed_retriever_name, threshold, create_graph_flag, cosine_sim_edges)
```