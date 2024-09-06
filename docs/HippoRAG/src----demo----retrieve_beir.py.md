# `.\HippoRAG\src\demo\retrieve_beir.py`

```py
# 注意 BEIR 使用 https://github.com/cvangysel/pytrec_eval 来评估检索结果。
import sys

# 将当前目录添加到系统路径中，以便可以导入当前目录下的模块
sys.path.append('.')

# 从 src.data_process.util 导入 merge_chunk_scores 和 merge_chunks 函数
from src.data_process.util import merge_chunk_scores, merge_chunks
# 从 src.hipporag 导入 HippoRAG 类
from src.hipporag import HippoRAG
import os
# 导入 pytrec_eval 库用于评估检索结果
import pytrec_eval
import argparse
# 导入 json 库用于处理 JSON 数据
import json
# 导入 tqdm 库用于显示进度条
from tqdm import tqdm


def detailed_log(queries, run_dict, eval_res, chunk=False, threshold=None, dpr_only=False):
    # 初始化一个空的日志列表
    logs = []
    # 遍历 run_dict['retrieved'] 中的每一个条目，显示进度条和错误分析描述
    for idx, query_id in tqdm(enumerate(run_dict['retrieved']), desc='Error analysis', total=len(run_dict['retrieved'])):
        # 根据当前索引获取查询项
        query_item = queries[idx]
        # 如果有阈值且当前查询的 NDCG 分数达到阈值，则跳过当前循环
        if threshold is not None and eval_res[query_id]['ndcg'] >= threshold:
            continue
        # 获取正确的段落（gold passages）
        gold_passages = queries[idx]['paragraphs']
        # 获取正确段落的 ID 列表
        gold_passage_ids = [p['idx'] for p in query_item['paragraphs']]

        # 初始化距离、计数和提取实体、三元组的列表
        distances = []
        num_dis = 0
        gold_passage_extracted_entities = []
        gold_passage_extracted_triples = []
        # 如果不是仅 DPR 模式
        if not dpr_only:
            # 根据段落 ID 获取提取的实体和三元组
            gold_passage_extractions = [hipporag.get_extraction_by_passage_idx(p_idx, chunk) for p_idx in gold_passage_ids]
            gold_passage_extracted_entities = [e for extr in gold_passage_extractions for e in extr['extracted_entities']]
            gold_passage_extracted_triples = [t for extr in gold_passage_extractions for t in extr['extracted_triples']]

            # 如果日志中包含 linked_node_scores
            if 'linked_node_scores' in run_dict['log'][query_id]:
                # 遍历每个节点链接
                for node_linking in run_dict['log'][query_id]['linked_node_scores']:
                    # 获取链接节点的短语
                    linked_node_phrase = node_linking[1]
                    # 初始化距离列表
                    distance = []
                    # 计算与提取的实体之间的距离
                    for e in gold_passage_extracted_entities:
                        if e == linked_node_phrase:
                            distance.append(0)
                            num_dis += 1
                        d = hipporag.get_shortest_distance_between_nodes(linked_node_phrase, e)
                        if d > 0:
                            distance.append(d)
                            num_dis += 1
                    # 将计算的距离添加到距离列表
                    distances.append(distance)

        # 初始化预测段落列表
        pred_passages = []
        # 根据预测的语料 ID 获取预测的段落
        for pred_corpus_id in run_dict['retrieved'][query_id]:
            if not chunk:
                # 如果不是 chunk 模式，则直接匹配 ID
                for corpus_item in corpus:
                    if corpus_item['idx'] == pred_corpus_id:
                        pred_passages.append(corpus_item)
            else:
                # 如果是 chunk 模式，则匹配 ID 或以 ID 开头的 ID
                for corpus_item in corpus:
                    if corpus_item['idx'] == pred_corpus_id or corpus_item['idx'].startswith(f'{pred_corpus_id}_'):
                        pred_passages.append(corpus_item)
        # 如果是 chunk 模式，则合并段落块
        if chunk:
            pred_passages = merge_chunks(pred_passages)

        # 将当前查询的结果记录到日志中
        logs.append({
            'query': queries[idx]['text'],  # 查询文本
            'ndcg': eval_res[query_id]['ndcg'],  # NDCG 分数
            'gold_passages': gold_passages,  # 正确段落
            'pred_passages': pred_passages,  # 预测段落
            'log': run_dict['log'][query_id],  # 运行日志
            'distances': distances,  # 距离列表
            # 计算平均距离，如果有计算的距离，则计算平均值，否则为 None
            'avg_distance': sum([sum(d) for d in distances]) / num_dis if num_dis > 0 else None,
            'entities_in_supporting_passage': gold_passage_extracted_entities,  # 提取的实体
            'triples_in_supporting_passage': gold_passage_extracted_triples,  # 提取的三元组
        })
    # 返回包含所有日志的列表
    return logs
# 当脚本作为主程序运行时执行以下代码
if __name__ == '__main__':
    # 创建 ArgumentParser 对象用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加数据集参数，指定数据集名称及其拆分信息
    parser.add_argument('--dataset', type=str, help='dataset name and split, e.g., `sci_fact_test`, `fiqa_dev`.')
    # 添加 chunk 参数，作为布尔开关
    parser.add_argument('--chunk', action='store_true')
    # 添加 extraction_model 参数，指定提取模型，默认值为 'gpt-3.5-turbo-1106'
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo-1106')
    # 添加 retrieval_model 参数，指定图创建检索器名称
    parser.add_argument('--retrieval_model', type=str, help="Graph creating retriever name, e.g., 'facebook/contriever', 'colbertv2'")
    # 添加 linking_model 参数，指定节点链接模型名称
    parser.add_argument('--linking_model', type=str, help="Node linking model name, e.g., 'facebook/contriever', 'colbertv2'")
    # 添加 doc_ensemble 参数，作为布尔开关
    parser.add_argument('--doc_ensemble', action='store_true')
    # 添加 dpr_only 参数，作为布尔开关
    parser.add_argument('--dpr_only', action='store_true')
    # 解析命令行参数
    args = parser.parse_args()

    # 如果没有指定 chunk 参数但数据集中包含 'chunk'，则自动启用 chunk 参数
    if args.chunk is False and 'chunk' in args.dataset:
        args.chunk = True
    # 确保 doc_ensemble 和 dpr_only 至多只有一个为 True
    assert not (args.doc_ensemble and args.dpr_only)
    # 从 JSON 文件中加载语料库数据
    corpus = json.load(open(f'data/{args.dataset}_corpus.json'))
    # 从 JSON 文件中加载 Qrel 数据，用于 pytrec_eval
    qrel = json.load(open(f'data/{args.dataset}_qrel.json'))  # note that this is json file processed from tsv file, used for pytrec_eval
    # 创建 HippoRAG 对象，传入数据集、提取模型、检索模型及其他参数
    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble, dpr_only=args.dpr_only,
                        linking_retriever_name=args.linking_model)

    # 从 JSON 文件中加载查询数据
    with open(f'data/{args.dataset}_queries.json') as f:
        queries = json.load(f)

    # 生成 doc_ensemble 和模型字符串，用于构建运行输出路径
    doc_ensemble_str = 'doc_ensemble' if args.doc_ensemble else 'no_ensemble'
    extraction_str = args.extraction_model.replace('/', '_').replace('.', '_')
    graph_creating_str = args.retrieval_model.replace('/', '_').replace('.', '_')
    if args.linking_model is None:
        args.linking_model = args.retrieval_model
    linking_str = args.linking_model.replace('/', '_').replace('.', '_')
    dpr_only_str = '_dpr_only' if args.dpr_only else ''
    run_output_path = f'exp/{args.dataset}_run_{doc_ensemble_str}_{extraction_str}_{graph_creating_str}_{linking_str}{dpr_only_str}.json'

    # 定义评价指标
    metrics = {'map', 'ndcg'}
    # 创建 pytrec_eval 的 RelevanceEvaluator 对象
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)

    # 如果运行输出文件存在，则加载其内容并打印日志文件的信息
    if os.path.isfile(run_output_path):
        run_dict = json.load(open(run_output_path))
        print(f'Log file found at {run_output_path}, len: {len(run_dict["retrieved"])}')
    else:
        # 如果文件不存在，则初始化一个包含空数据的字典
        run_dict = {'retrieved': {}, 'log': {}}  # for pytrec_eval

    # 初始化标记是否需要更新运行数据
    to_update_run = False
    # 遍历查询数据
    for query in tqdm(queries):
        query_text = query['text']
        query_id = query['id']
        # 如果查询 ID 已经存在于 retrieved 字典中，则跳过
        if query_id in run_dict['retrieved']:
            continue
        # 使用 HippoRAG 对象对文档进行排名
        ranks, scores, log = hipporag.rank_docs(query_text, top_k=10)

        # 根据排名提取检索到的文档
        retrieved_docs = [corpus[r] for r in ranks]
        # 更新 run_dict 中的 retrieved 和 log 数据
        run_dict['retrieved'][query_id] = {doc['idx']: score for doc, score in zip(retrieved_docs, scores)}
        run_dict['log'][query_id] = log
        # 设置标记以更新运行数据
        to_update_run = True
    # 如果需要更新运行结果
    if to_update_run:
        # 打开运行结果输出路径，并以写入模式创建文件对象
        with open(run_output_path, 'w') as f:
            # 将运行结果字典以 JSON 格式写入文件
            json.dump(run_dict, f)
            # 打印保存成功信息，包括保存路径和数据长度
            print(f'Run saved to {run_output_path}, len: {len(run_dict["retrieved"])}')

    # 如果需要对检索到的结果进行后处理（如果语料库被分块）
    if args.chunk:
        # 遍历检索到的结果列表中的每个索引
        for idx in run_dict['retrieved']:
            # 调用函数合并分块的分数，并更新检索到的结果
            run_dict['retrieved'][idx] = merge_chunk_scores(run_dict['retrieved'][idx])

    # 对检索结果进行评估，得到评估结果
    eval_res = evaluator.evaluate(run_dict['retrieved'])

    # 计算各指标的平均分数
    avg_scores = {}
    for metric in metrics:
        # 计算每个指标的平均分数，并保留三位小数
        avg_scores[metric] = round(sum([v[metric] for v in eval_res.values()]) / len(eval_res), 3)
    # 打印评估结果的平均分数
    print(f'Evaluation results: {avg_scores}')

    # 生成详细日志
    logs = detailed_log(queries, run_dict, eval_res, args.chunk, dpr_only=args.dpr_only)
    # 构建详细日志输出路径
    detailed_log_output_path = f'exp/{args.dataset}_log_{doc_ensemble_str}_{extraction_str}_{graph_creating_str}{linking_str}{dpr_only_str}.json'
    # 打开详细日志输出文件，并以写入模式创建文件对象
    with open(detailed_log_output_path, 'w') as f:
        # 将详细日志以 JSON 格式写入文件
        json.dump(logs, f)
    # 打印详细日志保存成功信息，包括保存路径
    print(f'Detailed log saved to {detailed_log_output_path}')
```