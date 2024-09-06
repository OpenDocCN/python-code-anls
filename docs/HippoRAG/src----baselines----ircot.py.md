# `.\HippoRAG\src\baselines\ircot.py`

```py
# 导入系统模块
import sys

# 将当前目录添加到模块搜索路径中
sys.path.append('.')

# 从 langchain_core.messages 模块中导入 SystemMessage 和 HumanMessage 类
from langchain_core.messages import SystemMessage, HumanMessage
# 从 langchain_core.prompts 模块中导入 ChatPromptTemplate 类
from langchain_core.prompts import ChatPromptTemplate

# 从 src.langchain_util 模块中导入 init_langchain_model 函数和 num_tokens_by_tiktoken 函数
from src.langchain_util import init_langchain_model, num_tokens_by_tiktoken
# 从 src.processing 模块中导入 mean_pooling_embedding_with_normalization 函数
from src.processing import mean_pooling_embedding_with_normalization
# 从 src.elastic_search_tool 模块中导入 search_with_score 函数
from src.elastic_search_tool import search_with_score
# 导入 numpy 模块，并简写为 np
import numpy as np
# 从 sentence_transformers 模块中导入 SentenceTransformer 类
from sentence_transformers import SentenceTransformer

# 导入 torch 模块
import torch
# 导入并简写 concurrent 模块
import concurrent
# 从 abc 模块中导入 abstractmethod 装饰器
from abc import abstractmethod
# 从 concurrent.futures 模块中导入 ThreadPoolExecutor 类
from concurrent.futures import ThreadPoolExecutor

# 导入 faiss 模块
import faiss
# 从 elasticsearch 模块中导入 Elasticsearch 类
from elasticsearch import Elasticsearch
# 从 transformers 模块中导入 AutoTokenizer 和 AutoModel 类
from transformers import AutoTokenizer, AutoModel

# 导入 argparse、json 和 os 模块
import argparse
import json
import os

# 从 tqdm 模块中导入 tqdm 函数，用于显示进度条
from tqdm import tqdm

# 定义用于生成多步推理的指令
ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'

# 定义抽象类 DocumentRetriever
class DocumentRetriever:
    # 定义抽象方法 rank_docs
    @abstractmethod
    def rank_docs(self, query: str, top_k: int):
        """
        根据给定的查询对语料库中的文档进行排序
        :param query: 查询字符串
        :param top_k: 返回的文档数量
        :return: 检索到的文档的排名和分数
        """

# 定义 BM25Retriever 类，继承自 DocumentRetriever
class BM25Retriever(DocumentRetriever):
    # 初始化方法，设置 Elasticsearch 实例及索引名称
    def __init__(self, index_name: str, host: str = 'localhost', port: int = 9200):
        # 创建 Elasticsearch 实例，并设置连接参数
        self.es = Elasticsearch([{"host": host, "port": port, "scheme": "http"}], max_retries=5, retry_on_timeout=True, request_timeout=30)
        # 设置索引名称
        self.index_name = index_name

    # 实现 rank_docs 方法
    def rank_docs(self, query: str, top_k: int):
        # 调用 search_with_score 函数检索并评分文档
        results = search_with_score(self.es, self.index_name, query, top_k)
        # 返回文档索引和评分
        return [int(result[0]) for result in results], [result[1] for result in results]

# 定义 DPRRetriever 类，继承自 DocumentRetriever
class DPRRetriever(DocumentRetriever):
    # 初始化方法，设置模型、faiss 索引、语料库和设备
    def __init__(self, model_name: str, faiss_index: str, corpus, device='cuda'):
        """
        :param model_name: 模型名称
        :param faiss_index: faiss 索引路径
        """
        # 从预训练模型加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 从预训练模型加载模型并转移到指定设备
        self.model = AutoModel.from_pretrained(model_name).to(device)
        # 设置 faiss 索引和语料库
        self.faiss_index = faiss_index
        self.corpus = corpus
        self.device = device

    # 实现 rank_docs 方法
    def rank_docs(self, query: str, top_k: int):
        # 禁用梯度计算，以提高推理效率
        with torch.no_grad():
            # 计算查询的嵌入表示，并将其转换为 numpy 数组
            query_embedding = mean_pooling_embedding_with_normalization(query, self.tokenizer, self.model, self.device).detach().cpu().numpy()
        # 使用 faiss 索引进行检索，获取相似度和文档索引
        inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)
        # 返回文档索引和相似度分数
        return corpus_idx.tolist()[0], inner_product.tolist()[0]
class SentenceTransformersRetriever(DocumentRetriever):
    # 初始化 SentenceTransformersRetriever 类
    def __init__(self, model_name: str, faiss_index: str, corpus, device='cuda', norm=True):
        """
        初始化 SentenceTransformersRetriever 类的构造函数
        
        :param model_name: 用于生成嵌入的模型名称
        :param faiss_index: FAISS 索引的路径
        """
        # 创建 SentenceTransformer 模型实例
        self.model = SentenceTransformer(model_name)
        # 保存 FAISS 索引的路径
        self.faiss_index = faiss_index
        # 保存语料库
        self.corpus = corpus
        # 设备选择（默认为 'cuda'）
        self.device = device
        # 是否对嵌入进行归一化（默认为 True）
        self.norm = norm

    # 排名文档的方法
    def rank_docs(self, query: str, top_k: int):
        # 将查询文本编码成嵌入向量
        query_embedding = self.model.encode(query)
        # 为嵌入向量添加一个新的维度
        query_embedding = np.expand_dims(query_embedding, axis=0)
        # 如果需要，计算嵌入向量的范数并进行归一化
        if self.norm:
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding / norm
        # 使用 FAISS 索引进行搜索，获取内积和文档索引
        inner_product, corpus_idx = self.faiss_index.search(query_embedding, top_k)
        # 返回文档索引和内积的列表
        return corpus_idx.tolist()[0], inner_product.tolist()[0]


class Colbertv2Retriever(DocumentRetriever):
    # 初始化 Colbertv2Retriever 类
    def __init__(self, root: str, index_name: str):
        # 保存根目录路径
        self.root = root
        # 保存索引名称
        self.index_name = index_name
        # 从 ColBERT 库中导入必要的模块
        from colbert.infra import Run
        from colbert.infra import RunConfig
        from colbert import Searcher
        from colbert.infra import ColBERTConfig

        # 使用 Run 上下文管理器配置 ColBERT 环境
        with Run().context(RunConfig(nranks=1, experiment="colbert", root=self.root)):
            # 配置 ColBERT 参数
            config = ColBERTConfig(
                root=self.root.rstrip('/') + '/colbert',
            )
            # 创建 Searcher 实例
            self.searcher = Searcher(index=self.index_name, config=config)

    # 排名文档的方法
    def rank_docs(self, query: str, top_k: int):
        # 从 ColBERT 库中导入 Queries 模块
        from colbert.data import Queries

        # 初始化 ID 和分数列表
        ids = []
        scores = []
        # 创建 Queries 实例
        queries = Queries(path=None, data={0: query})
        # 执行搜索，获取排名结果
        ranking = self.searcher.search_all(queries, k=top_k)

        # 遍历排名结果，提取文档 ID 和分数
        for docid, rank, score in ranking.data[0]:
            ids.append(docid)
            scores.append(score)
        # 返回文档 ID 和分数的列表
        return ids, scores


# 解析提示内容的函数
def parse_prompt(file_path: str, has_context=True):
    # 以 UTF-8 编码打开文件，并读取其内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 按照元数据模式分割内容
    parts = content.split('# METADATA: ')
    # 初始化解析后的数据列表
    parsed_data = []
    if has_context:
        # 如果存在上下文
        for part in parts[1:]:  # 跳过第一个分割项，因为它会是空的
            # 将当前部分按行分割为元数据部分和其余数据部分
            metadata_section, rest_of_data = part.split('\n', 1)
            # 解析 JSON 格式的元数据
            metadata = json.loads(metadata_section)
            # 将其余数据部分按 '\n\nQ: ' 分割为文档部分和 QA 对
            document_sections = rest_of_data.strip().split('\n\nQ: ')
            # 提取文档部分并去掉两边的空白字符
            document_text = document_sections[0].strip()
            # 将 QA 对部分按 '\nA: ' 分割为问题和答案
            qa_pair = document_sections[1].split('\nA: ')
            # 提取问题并去掉两边的空白字符
            question = qa_pair[0].strip()
            # 将思考和答案部分按 'So the answer is: ' 分割为思考和答案
            thought_and_answer = qa_pair[1].strip().split('So the answer is: ')
            # 提取思考部分并去掉两边的空白字符
            thought = thought_and_answer[0].strip()
            # 提取答案部分并去掉两边的空白字符
            answer = thought_and_answer[1].strip()

            # 将解析后的数据添加到结果列表中
            parsed_data.append({
                'metadata': metadata,
                'document': document_text,
                'question': question,
                'thought_and_answer': qa_pair[1].strip(),
                'thought': thought,
                'answer': answer
            })
    else:
        # 如果不存在上下文
        for part in parts[1:]:
            # 将当前部分按行分割为元数据部分和其余数据部分
            metadata_section, rest_of_data = part.split('\n', 1)
            # 解析 JSON 格式的元数据
            metadata = json.loads(metadata_section)
            # 将其余数据部分按行分割
            s = rest_of_data.split('\n')
            # 提取问题部分，并去掉前缀 'Q: ' 和两边的空白字符
            question = s[0][3:].strip()
            # 将思考和答案部分按 'So the answer is: ' 分割为思考和答案
            thought_and_answer = s[1][3:].strip().split('So the answer is: ')
            # 提取思考部分并去掉两边的空白字符
            thought = thought_and_answer[0].strip()
            # 提取答案部分并去掉两边的空白字符
            answer = thought_and_answer[1].strip()

            # 将解析后的数据添加到结果列表中
            parsed_data.append({
                'metadata': metadata,
                'question': question,
                'thought_and_answer': s[1][3:].strip(),
                'thought': thought,
                'answer': answer
            })

    # 返回解析后的数据列表
    return parsed_data
# 定义一个函数用于检索相关文档并返回结果
def retrieve_step(query: str, corpus, top_k: int, retriever: DocumentRetriever, dataset: str):
    # 使用 DocumentRetriever 对象对文档进行排序，并返回文档 ID 和对应的得分
    doc_ids, scores = retriever.rank_docs(query, top_k=top_k)
    # 根据数据集类型处理不同的文档格式
    if dataset in ['hotpotqa']:
        # 初始化一个空列表来存储检索到的段落
        retrieved_passages = []
        # 遍历文档 ID 列表
        for doc_id in doc_ids:
            # 获取对应的文档键，并读取文档内容
            key = list(corpus.keys())[doc_id]
            # 将文档标题和内容添加到检索段落列表
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    elif dataset in ['musique', '2wikimultihopqa']:
        # 对于其他数据集，直接按格式从 corpus 中提取标题和文本
        retrieved_passages = [corpus[doc_id]['title'] + '\n' + corpus[doc_id]['text'] for doc_id in doc_ids]
    else:
        # 如果数据集不在已实现的数据集中，抛出未实现错误
        raise NotImplementedError(f'Dataset {dataset} not implemented')
    # 返回检索到的段落和得分
    return retrieved_passages, scores


# 定义一个函数用于合并具有相同首行的元素
def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    # 初始化一个空字典来存储合并后的结果
    merged_dict = {}

    # 遍历元素列表中的每个元素
    for element in elements:
        # 将元素按行分割，并获取第一行
        lines = element.split('\n')
        first_line = lines[0]

        # 检查字典中是否已有该首行作为键
        if first_line in merged_dict:
            # 如果有，将当前元素追加到现有值中
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # 如果没有，将当前元素作为新条目添加到字典中
            merged_dict[first_line] = prefix + element

    # 从字典中提取合并后的元素
    merged_elements = list(merged_dict.values())
    # 返回合并后的元素列表
    return merged_elements


# 定义一个函数用于生成下一步的想法
def reason_step(dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
    """
    给定少量示例、查询、之前检索的段落和想法，使用 LangChain LLM 生成下一个想法。
    生成的想法将用于进一步的检索步骤。
    :return: 下一个想法
    """
    prompt_demo = ''

    prompt_user = ''
    # 针对特定数据集处理检索到的段落
    if dataset in ['hotpotqa']:
        # 合并具有相同首行的段落
        passages = merge_elements_with_same_first_line(passages)
    # 生成用户提示文本
    for passage in passages:
        prompt_user += f'Wikipedia Title: {passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    # 根据少量示例生成演示提示
    for sample in few_shot:
        cur_sample = f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["thought_and_answer"]}\n\n'
        # 检查生成的提示文本是否在令牌限制内
        if num_tokens_by_tiktoken(ircot_reason_instruction + prompt_demo + cur_sample + prompt_user) < 15000:
            prompt_demo += cur_sample

    # 创建聊天提示模板并格式化提示
    messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                 HumanMessage(prompt_user)]).format_prompt()

    try:
        # 使用客户端生成聊天完成结果
        chat_completion = client.invoke(messages.to_messages())
        response_content = chat_completion.content
    except Exception as e:
        # 处理异常并返回空字符串
        print(e)
        return ''
    # 返回生成的想法
    return response_content


# 定义一个函数用于处理样本
def process_sample(idx, sample, args, corpus, retriever, client, processed_ids):
    # 检查样本是否已被处理
    # 根据数据集类型提取样本的 ID
    if args.dataset in ['hotpotqa', '2wikimultihopqa']:
        sample_id = sample['_id']
    elif args.dataset in ['musique']:
        sample_id = sample['id']
    else:
        # 数据集类型未实现，抛出异常
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    # 如果样本 ID 已处理，跳过此样本
    if sample_id in processed_ids:
        return  # Skip already processed samples

    # 执行检索和推理步骤
    query = sample['question']
    # 调用检索步骤，获取相关段落及其分数
    retrieved_passages, scores = retrieve_step(query, corpus, args.top_k, retriever, args.dataset)

    thoughts = []
    # 创建段落与分数的字典
    retrieved_passages_dict = {passage: score for passage, score in zip(retrieved_passages, scores)}
    it = 1
    # 迭代进行推理步骤
    for it in range(1, max_steps):
        # 调用推理步骤，获取新的思路
        new_thought = reason_step(args.dataset, few_shot_samples, query, retrieved_passages[:args.top_k], thoughts, client)
        thoughts.append(new_thought)
        # 如果新思路包含答案，结束迭代
        if 'So the answer is:' in new_thought:
            break
        # 调用检索步骤，获取新的段落及其分数
        new_retrieved_passages, new_scores = retrieve_step(new_thought, corpus, args.top_k, retriever, args.dataset)

        # 更新段落字典中的分数
        for passage, score in zip(new_retrieved_passages, new_scores):
            if passage in retrieved_passages_dict:
                retrieved_passages_dict[passage] = max(retrieved_passages_dict[passage], score)
            else:
                retrieved_passages_dict[passage] = score

        # 解压字典中的段落和分数
        retrieved_passages, scores = zip(*retrieved_passages_dict.items())

        # 根据分数对段落进行排序
        sorted_passages_scores = sorted(zip(retrieved_passages, scores), key=lambda x: x[1], reverse=True)
        retrieved_passages, scores = zip(*sorted_passages_scores)
    # end iteration

    # 计算召回率
    if args.dataset in ['hotpotqa']:
        # 提取金标准段落
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    elif args.dataset in ['musique']:
        # 提取金标准段落
        gold_passages = [item for item in sample['paragraphs'] if item['is_supporting']]
        gold_items = set([item['title'] + '\n' + item['paragraph_text'] for item in gold_passages])
        retrieved_items = retrieved_passages
    elif args.dataset in ['2wikimultihopqa']:
        # 提取金标准段落
        gold_passages = [item for item in sample['supporting_facts']]
        gold_items = set([item[0] for item in gold_passages])
        retrieved_items = [passage.split('\n')[0].strip() for passage in retrieved_passages]
    else:
        # 数据集类型未实现，抛出异常
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    recall = dict()
    # 打印索引
    print(f'idx: {idx + 1} ', end='')
    for k in k_list:
        # 计算召回率
        recall[k] = sum(1 for t in gold_items if t in retrieved_items[:k]) / len(gold_items)
    # 返回索引、召回率、检索到的段落、思路列表和迭代次数
    return idx, recall, retrieved_passages, thoughts, it
# 当脚本作为主程序运行时，执行以下代码
if __name__ == '__main__':
    # 创建一个命令行参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个必需的命令行参数 '--dataset'，用于指定数据集的名称
    parser.add_argument('--dataset', type=str, choices=['hotpotqa', 'musique', '2wikimultihopqa'], required=True)
    # 添加一个可选的命令行参数 '--llm'，指定使用的语言模型，默认值为 'openai'
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    # 添加一个可选的命令行参数 '--llm_model'，指定语言模型的具体版本，默认值为 'gpt-3.5-turbo-1106'
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106')
    # 添加一个可选的命令行参数 '--retriever'，指定检索器的名称，默认值为 'facebook/contriever'
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    # 添加一个可选的命令行参数 '--prompt'，用于指定提示文件的路径
    parser.add_argument('--prompt', type=str)
    # 添加一个可选的命令行参数 '--unit'，用于指定单位类型，取值范围为 ['hippo', 'proposition']，默认值为 'hippo'
    parser.add_argument('--unit', type=str, choices=['hippo', 'proposition'], default='hippo')
    # 添加一个必需的命令行参数 '--num_demo'，指定示例的数量
    parser.add_argument('--num_demo', type=int, default=1, help='the number of documents in the demonstration', required=True)
    # 添加一个可选的命令行参数 '--max_steps'，用于指定最大步骤数
    parser.add_argument('--max_steps', type=int)
    # 添加一个可选的命令行参数 '--top_k'，指定每一步检索的文档数量，默认值为 8
    parser.add_argument('--top_k', type=int, default=8, help='retrieving k documents at each step')
    # 添加一个可选的命令行参数 '--thread'，指定并行处理的线程数，默认值为 6
    parser.add_argument('--thread', type=int, default=6, help='number of threads for parallel processing, 1 for sequential processing')
    # 解析命令行参数
    args = parser.parse_args()

    # 将检索器名称中的 '/' 和 '.' 替换为 '_'，生成新的检索器名称
    retriever_name = args.retriever.replace('/', '_').replace('.', '_')
    # 初始化语言模型
    client = init_langchain_model(args.llm, args.llm_model)
    # 创建包含 ColBERT 配置的字典
    colbert_configs = {'root': f'data/lm_vectors/colbertv2/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}

    # 根据数据集的名称加载数据集和语料库
    if args.dataset == 'hotpotqa':
        # 加载 'hotpotqa' 数据集
        data = json.load(open('data/hotpotqa.json', 'r'))
        # 加载 'hotpotqa' 语料库
        corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        # 设置与 'hotpotqa' 数据集相关的提示文件路径
        prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        # 如果未指定 max_steps，则默认为 2
        max_steps = args.max_steps if args.max_steps is not None else 2
    elif args.dataset == 'musique':
        # 加载 'musique' 数据集
        data = json.load(open('data/musique.json', 'r'))
        # 加载 'musique' 语料库
        corpus = json.load(open('data/musique_corpus.json', 'r'))
        # 设置与 'musique' 数据集相关的提示文件路径
        prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        # 如果未指定 max_steps，则默认为 4
        max_steps = args.max_steps if args.max_steps is not None else 4
    elif args.dataset == '2wikimultihopqa':
        # 加载 '2wikimultihopqa' 数据集
        data = json.load(open('data/2wikimultihopqa.json', 'r'))
        # 加载 '2wikimultihopqa' 语料库
        corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        # 设置与 '2wikimultihopqa' 数据集相关的提示文件路径
        prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        # 如果未指定 max_steps，则默认为 2
        max_steps = args.max_steps if args.max_steps is not None else 2
    else:
        # 如果数据集名称不在预期的列表中，抛出未实现的异常
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')
    # 解析提示文件，获取少量示例
    few_shot_samples = parse_prompt(prompt_path)
    # 根据指定的 num_demo 参数限制少量示例的数量
    few_shot_samples = few_shot_samples[:args.num_demo]
    # 打印少量示例的数量
    print('num of demo:', len(few_shot_samples))

    # 根据是否使用文档集成决定文档集成的字符串表示（目前此变量为空）
    doc_ensemble_str = ''
    # 根据 max_steps 的值构建输出文件路径
    if max_steps > 1:
        output_path = f'output/ircot/{args.dataset}_{retriever_name}_demo_{args.num_demo}_{args.llm_model}_{doc_ensemble_str}_step_{max_steps}_top_{args.top_k}.json'
    else:  # 处理步骤只剩下一个
        # 设置 top_k 参数为 100
        args.top_k = 100
        # 根据指定的参数生成输出文件的路径
        output_path = f'output/{args.unit}_{args.dataset}_{retriever_name}_{doc_ensemble_str}.json'

    # 根据选择的检索器类型进行条件判断
    if args.retriever == 'bm25':
        # 如果单位是 'hippo'，创建 BM25Retriever 对象，并指定索引名称
        if args.unit == 'hippo':
            retriever = BM25Retriever(index_name=f'{args.dataset}_{len(corpus)}_bm25')
        # 如果单位是 'proposition'，创建 BM25Retriever 对象，并指定索引名称
        elif args.unit == 'proposition':
            retriever = BM25Retriever(index_name=f'{args.dataset}_{len(corpus)}_proposition_bm25')
    # 如果检索器是 'colbertv2'
    elif args.retriever == 'colbertv2':
        # 根据数据集名称选择根目录
        if args.dataset == 'hotpotqa':
            root = 'exp/hotpotqa'
            # 根据单位选择索引名称
            if args.unit == 'hippo':
                index_name = 'hotpotqa_1000_nbits_2'
            elif args.unit == 'proposition':
                index_name = 'hotpotqa_1000_proposition_nbits_2'
        elif args.dataset == 'musique':
            root = 'exp/musique'
            if args.unit == 'hippo':
                index_name = 'musique_1000_nbits_2'
            elif args.unit == 'proposition':
                index_name = 'musique_1000_proposition_nbits_2'
        elif args.dataset == '2wikimultihopqa':
            root = 'exp/2wikimultihopqa'
            if args.unit == 'hippo':
                index_name = '2wikimultihopqa_1000_nbits_2'
            elif args.unit == 'proposition':
                index_name = '2wikimultihopqa_1000_proposition_nbits_2'
        # 创建 Colbertv2Retriever 对象，并指定根目录和索引名称
        retriever = Colbertv2Retriever(root, index_name)
    # 如果检索器是 'facebook/contriever'
    elif args.retriever == 'facebook/contriever':
        # 根据数据集名称选择并读取相应的 FAISS 索引文件
        if args.dataset == 'hotpotqa':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_facebook_contriever_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_proposition_ip_norm.index')
        elif args.dataset == 'musique':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/musique/musique_facebook_contriever_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/musique/musique_proposition_ip_norm.index')
        elif args.dataset == '2wikimultihopqa':
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_facebook_contriever_hippo_ip_norm.index')
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_proposition_ip_norm.index')
        # 创建 DPRRetriever 对象，并指定检索器、FAISS 索引和语料库
        retriever = DPRRetriever(args.retriever, faiss_index, corpus)
    # 如果 retriever 的字符串以 'sentence-transformers/gtr-t5' 开头
    elif args.retriever.startswith('sentence-transformers/gtr-t5'):
        # 如果 dataset 参数为 'hotpotqa'
        if args.dataset == 'hotpotqa':
            # 如果 unit 参数为 'hippo'，读取对应的 FAISS 索引文件
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_sentence-transformers_gtr-t5-base_hippo_ip_norm.index')
            # 如果 unit 参数为 'proposition'，读取对应的 FAISS 索引文件
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/hotpotqa/hotpotqa_sentence-transformers_gtr-t5-base_proposition_ip_norm.index')
        # 如果 dataset 参数为 'musique'
        elif args.dataset == 'musique':
            # 如果 unit 参数为 'hippo'，读取对应的 FAISS 索引文件
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/musique/musique_sentence-transformers_gtr-t5-base_hippo_ip_norm.index')
            # 如果 unit 参数为 'proposition'，读取对应的 FAISS 索引文件
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/musique/musique_sentence-transformers_gtr-t5-base_proposition_ip_norm.index')
        # 如果 dataset 参数为 '2wikimultihopqa'
        elif args.dataset == '2wikimultihopqa':
            # 如果 unit 参数为 'hippo'，读取对应的 FAISS 索引文件
            if args.unit == 'hippo':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_sentence-transformers_gtr-t5-base_hippo_ip_norm.index')
            # 如果 unit 参数为 'proposition'，读取对应的 FAISS 索引文件
            elif args.unit == 'proposition':
                faiss_index = faiss.read_index('data/2wikimultihopqa/2wikimultihopqa_sentence-transformers_gtr-t5-base_proposition_ip_norm.index')
        # 使用指定的 retriever、FAISS 索引和 corpus 创建 SentenceTransformersRetriever 对象
        retriever = SentenceTransformersRetriever(args.retriever, faiss_index, corpus)

    # 定义一组 k 值，用于后续计算召回率
    k_list = [1, 2, 5, 10, 15, 20, 30, 50, 100]
    # 初始化一个字典 total_recall，用于存储每个 k 值的召回率总和
    total_recall = {k: 0 for k in k_list}

    # 从 data 变量中读取先前的结果
    results = data
    # 初始化标志，表示是否读取了现有的数据
    read_existing_data = False
    try:
        # 如果 output_path 指定的文件存在
        if os.path.isfile(output_path):
            # 以读取模式打开文件
            with open(output_path, 'r') as f:
                # 从文件中加载 JSON 格式的结果数据
                results = json.load(f)
                # 打印加载结果的条数
                print(f'Loaded {len(results)} results from {output_path}')
                # 如果结果数据不为空，设置标志为 True
                if len(results):
                    read_existing_data = True
        # 根据 dataset 的类型，提取已处理的样本 ID
        if args.dataset in ['hotpotqa', '2wikimultihopqa']:
            processed_ids = {sample['_id'] for sample in results if 'retrieved' in sample}
        elif args.dataset in ['musique']:
            processed_ids = {sample['id'] for sample in results if 'retrieved' in sample}
        else:
            # 如果 dataset 类型未实现，则抛出 NotImplementedError 异常
            raise NotImplementedError(f'Dataset {args.dataset} not implemented')
        # 计算每个样本的召回率，并更新 total_recall 字典
        for sample in results:
            if 'recall' in sample:
                total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
    except Exception as e:
        # 捕获异常，打印异常信息和结果文件可能为空的提示
        print('loading results exception', e)
        print(f'Results file {output_path} maybe empty, cannot be loaded.')
        # 如果发生异常，初始化 processed_ids 为空集合
        processed_ids = set()

    # 如果 results 列表不为空
    if len(results) > 0:
        # 打印每个 k 值的平均召回率
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        # 打印换行符
        print()
    # 如果已经读取了现有的数据
    if read_existing_data:
        # 打印提示信息，并退出程序
        print(f'All samples have been already in the result file ({output_path}), exit.')
        exit(0)
    # 使用线程池执行任务，最大线程数为 args.thread
    with ThreadPoolExecutor(max_workers=args.thread) as executor:
        # 将任务提交到线程池中，并保存 Future 对象
        futures = [executor.submit(process_sample, idx, sample, args, corpus, retriever, client, processed_ids) for idx, sample in enumerate(data)]

        # 遍历已完成的 Future 对象
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(data), desc='Parallel IRCoT'):
            # 获取任务结果
            idx, recall, retrieved_passages, thoughts, it = future.result()

            # 打印每个 k 值的 recall 结果
            for k in k_list:
                total_recall[k] += recall[k]
                print(f'R@{k}: {total_recall[k] / (idx + 1):.4f} ', end='')
            print()
            # 如果 max_steps 大于 1，打印当前迭代信息
            if args.max_steps > 1:
                print('[ITERATION]', it, '[PASSAGE]', len(retrieved_passages), '[THOUGHT]', thoughts)

            # 记录结果到结果字典
            results[idx]['retrieved'] = retrieved_passages
            results[idx]['recall'] = recall
            results[idx]['thoughts'] = thoughts

            # 每 50 个任务保存一次结果
            if idx % 50 == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f)

    # 在所有任务完成后保存最终结果
    with open(output_path, 'w') as f:
        json.dump(results, f)
    # 打印最终的 recall 结果
    print(f'Saved results to {output_path}')
    for k in k_list:
        print(f'R@{k}: {total_recall[k] / len(data):.4f} ', end='')
```