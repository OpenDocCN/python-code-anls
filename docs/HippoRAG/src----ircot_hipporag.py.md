# `.\HippoRAG\src\ircot_hipporag.py`

```py
# 导入系统模块，用于操作 Python 解释器及其环境
import sys

# 将当前目录添加到系统路径中，以便可以导入当前目录中的模块
sys.path.append('.')

# 导入调试工具
import ipdb
# 从 langchain_core 模块导入 SystemMessage 和 HumanMessage 类，用于消息处理
from langchain_core.messages import SystemMessage, HumanMessage
# 从 langchain_core 模块导入 ChatPromptTemplate 类，用于创建聊天提示模板
from langchain_core.prompts import ChatPromptTemplate

# 从 src.langchain_util 模块导入 init_langchain_model 函数，用于初始化语言模型
from src.langchain_util import init_langchain_model
# 从 transformers.hf_argparser 模块导入 string_to_bool 函数，用于将字符串转换为布尔值
from transformers.hf_argparser import string_to_bool
# 导入 argparse 模块，用于处理命令行参数
import argparse
# 导入 json 模块，用于解析 JSON 数据
import json

# 从 tqdm 模块导入 tqdm 类，用于显示进度条
from tqdm import tqdm

# 从 hipporag 模块导入 HippoRAG 类，用于处理 RAG（Retrieval-Augmented Generation）任务
from hipporag import HippoRAG

# 定义一个用于说明智能助手角色和任务的指令
ircot_reason_instruction = 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, start with "So the answer is:".'

# 定义一个解析提示文件的函数
def parse_prompt(file_path):
    # 打开文件并以 UTF-8 编码读取内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 按照元数据模式分割内容
    parts = content.split('# METADATA: ')
    parsed_data = []

    # 遍历分割后的每个部分，跳过第一个部分
    for part in parts[1:]:  # Skip the first split as it will be empty
        # 将元数据部分和其余数据部分分开
        metadata_section, rest_of_data = part.split('\n', 1)
        # 解析元数据
        metadata = json.loads(metadata_section)
        # 将其余数据部分按 'Q: ' 分割，得到文档和问答对
        document_sections = rest_of_data.strip().split('\n\nQ: ')
        document_text = document_sections[0].strip()
        qa_pair = document_sections[1].split('\nA: ')
        question = qa_pair[0].strip()
        answer = qa_pair[1].strip()

        # 将解析后的数据添加到结果列表中
        parsed_data.append({
            'metadata': metadata,
            'document': document_text,
            'question': question,
            'answer': answer
        })

    # 返回解析后的数据列表
    return parsed_data

# 定义一个检索步骤的函数
def retrieve_step(query: str, corpus, top_k: int, rag: HippoRAG, dataset: str):
    # 使用 RAG 对象对文档进行排名，并获取排名、分数和日志
    ranks, scores, logs = rag.rank_docs(query, top_k=top_k)
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        retrieved_passages = []
        # 根据排名从 corpus 中提取检索到的段落
        for rank in ranks:
            key = list(corpus.keys())[rank]
            retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
    else:
        # 根据排名从 corpus 中提取标题和文本
        retrieved_passages = [corpus[rank]['title'] + '\n' + corpus[rank]['text'] for rank in ranks]
    # 返回检索到的段落、分数和日志
    return retrieved_passages, scores, logs

# 定义一个合并具有相同首行的元素的函数
def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
    merged_dict = {}

    # 遍历元素列表中的每个元素
    for element in elements:
        # 将元素按行分割，并获取首行
        lines = element.split('\n')
        first_line = lines[0]

        # 检查首行是否已经是字典中的一个键
        if first_line in merged_dict:
            # 如果是，则将当前元素追加到已有的值中
            merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
        else:
            # 如果不是，则将当前元素作为新条目添加到字典中
            merged_dict[first_line] = prefix + element

    # 从字典中提取合并后的元素
    # 将 merged_dict 字典中的所有值提取到一个列表中
    merged_elements = list(merged_dict.values())
    # 返回包含所有值的列表
    return merged_elements
# 定义根据样本、查询、检索的段落和思考生成下一个思考的函数
def reason_step(dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
    """
    根据少量样本、查询、先前检索的段落和思考，使用 OpenAI 模型生成下一个思考。生成的思考用于进一步的检索步骤。
    :return: 下一个思考
    """
    # 初始化示例提示字符串
    prompt_demo = ''
    # 遍历所有示例，构建提示字符串
    for sample in few_shot:
        prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

    # 初始化用户提示字符串
    prompt_user = ''
    # 如果数据集是 hotpotqa 或 hotpotqa_train，合并相同第一行的段落
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    # 遍历段落，构建用户提示字符串
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    # 添加查询和当前思考到用户提示字符串中
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

    # 根据系统消息和用户消息创建聊天提示模板，并格式化提示
    messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                 HumanMessage(prompt_user)]).format_prompt()

    try:
        # 调用客户端执行消息，获取聊天完成结果
        chat_completion = client.invoke(messages.to_messages())
        # 获取聊天完成的内容
        response_content = chat_completion.content
    except Exception as e:
        # 捕获异常并打印错误，返回空字符串
        print(e)
        return ''
    # 返回聊天完成的内容
    return response_content


# 主程序入口
if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加数据集参数
    parser.add_argument('--dataset', type=str)
    # 添加 LLM 参数，默认为 'openai'
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    # 添加 LLM 模型名称参数，默认为 'gpt-3.5-turbo-1106'
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    # 添加检索器参数，默认为 'facebook/contriever'
    parser.add_argument('--retriever', type=str, default='facebook/contriever')
    # 添加提示参数
    parser.add_argument('--prompt', type=str)
    # 添加示例数量参数，默认为 1
    parser.add_argument('--num_demo', type=int, default=1, help='the number of demo samples')
    # 添加最大步骤数参数
    parser.add_argument('--max_steps', type=int)
    # 添加检索的文档数量参数，默认为 8
    parser.add_argument('--top_k', type=int, default=8, help='retrieving k documents at each step')
    # 添加文档集合参数，默认为 't'
    parser.add_argument('--doc_ensemble', type=str, default='t')
    # 添加是否仅使用 DPR 参数，默认为 'f'
    parser.add_argument('--dpr_only', type=str, default='f')
    # 添加图算法参数，默认为 'ppr'
    parser.add_argument('--graph_alg', type=str, default='ppr')
    # 添加是否不使用节点特征参数
    parser.add_argument('--wo_node_spec', action='store_true')
    # 添加相似度阈值参数，默认为 0.8
    parser.add_argument('--sim_threshold', type=float, default=0.8)
    # 添加识别阈值参数，默认为 0.9
    parser.add_argument('--recognition_threshold', type=float, default=0.9)
    # 添加阻尼系数参数，默认为 0.1
    parser.add_argument('--damping', type=float, default=0.1)
    # 添加强制重试参数
    parser.add_argument('--force_retry', action='store_true')
    # 解析命令行参数
    args = parser.parse_args()

    # 设置环境变量 OPENAI_API_KEY
    doc_ensemble = string_to_bool(args.doc_ensemble)
    dpr_only = string_to_bool(args.dpr_only)

    # 初始化语言模型客户端
    client = init_langchain_model(args.llm, args.llm_model)
    # 处理 LLM 模型名称
    llm_model_name_processed = args.llm_model.replace('/', '_').replace('.', '_')
    # 如果模型名称是 'gpt-3.5-turbo-1106'（默认 OpenIE 系统），则设置相关配置
    # if args.llm_model == 'gpt-3.5-turbo-1106':  # Default OpenIE system
    #     colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}
    # else:
    # 设置 ColBERT 配置，包括数据路径、文档索引名称和短语索引名称
    colbert_configs = {'root': f'data/lm_vectors/colbert/{args.dataset}', 'doc_index_name': 'nbits_2', 'phrase_index_name': 'nbits_2'}

    # 创建 HippoRAG 实例，初始化参数包括数据集、LLM、检索器、文档集成和各种算法参数
    rag = HippoRAG(args.dataset, args.llm, args.llm_model, args.retriever, doc_ensemble=doc_ensemble, node_specificity=not (args.wo_node_spec), sim_threshold=args.sim_threshold,
                   colbert_config=colbert_configs, dpr_only=dpr_only, graph_alg=args.graph_alg, damping=args.damping, recognition_threshold=args.recognition_threshold)

    # 从 JSON 文件中加载数据和语料库
    data = json.load(open(f'data/{args.dataset}.json', 'r'))
    corpus = json.load(open(f'data/{args.dataset}_corpus.json', 'r'))
    # 获取最大步骤数
    max_steps = args.max_steps

    # 根据数据集的类型选择适当的提示路径
    if max_steps > 1:
        if 'hotpotqa' in args.dataset:
            prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif 'musique' in args.dataset:
            prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
        elif '2wikimultihopqa' in args.dataset:
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
        else:
            prompt_path = f'data/ircot_prompts/{args.dataset}/gold_with_3_distractors_context_cot_qa_codex.txt'

        # 解析提示文件并获取少量示例
        few_shot_samples = parse_prompt(prompt_path)[:args.num_demo]

    # 根据是否使用文档集成和识别阈值生成文档集成描述字符串
    doc_ensemble_str = f'doc_ensemble_{args.recognition_threshold}' if doc_ensemble else 'no_ensemble'

    # 根据是否只使用 DPR 生成相应的字符串
    if dpr_only:
        dpr_only_str = 'dpr_only'
    else:
        dpr_only_str = 'hipporag'

    # 根据图算法的不同生成输出路径
    if args.graph_alg == 'ppr':
        output_path = f'output/ircot/ircot_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_demo_{args.num_demo}_{llm_model_name_processed}_{doc_ensemble_str}_step_{max_steps}_top_{args.top_k}_sim_thresh_{args.sim_threshold}'
        if args.damping != 0.1:
            # 如果阻尼系数不是默认值，则将其添加到输出路径
            output_path += f'_damping_{args.damping}'
    else:
        output_path = f'output/ircot/ircot_results_{args.dataset}_{dpr_only_str}_{rag.graph_creating_retriever_name_processed}_demo_{args.num_demo}_{llm_model_name_processed}_{doc_ensemble_str}_step_{max_steps}_top_{args.top_k}_{args.graph_alg}_sim_thresh_{args.sim_threshold}'

    # 如果不使用节点特异性，将相关标志添加到输出路径
    if args.wo_node_spec:
        output_path += 'wo_node_spec'

    # 最终的输出路径添加文件扩展名
    output_path += '.json'

    # 定义一组 K 值，并初始化总召回字典
    k_list = [1, 2, 5, 10, 15, 20, 30, 40, 50, 80, 100]
    total_recall = {k: 0 for k in k_list}

    # 强制重试标志
    force_retry = args.force_retry

    # 如果需要强制重试，则初始化结果列表和已处理 ID 集合
    if force_retry:
        results = []
        processed_ids = set()
    else:
        try:
            # 尝试以读取模式打开指定路径的文件
            with open(output_path, 'r') as f:
                # 从文件中加载 JSON 数据
                results = json.load(f)
            # 根据数据集类型提取已处理的样本 ID
            if args.dataset in ['hotpotqa', '2wikimultihopqa', 'hotpotqa_train']:
                processed_ids = {sample['_id'] for sample in results}
            else:
                processed_ids = {sample['id'] for sample in results}

            # 计算总召回率
            for sample in results:
                total_recall = {k: total_recall[k] + sample['recall'][str(k)] for k in k_list}
        except Exception as e:
            # 捕捉异常并打印错误信息
            print(e)
            print('Results file maybe empty, cannot be loaded.')
            # 初始化空结果和已处理 ID 集合
            results = []
            processed_ids = set()
            total_recall = {k: 0 for k in k_list}

    # 打印从文件加载的结果数量
    print(f'Loaded {len(results)} results from {output_path}')
    if len(results) > 0:
        # 打印每个 k 的平均召回率
        for k in k_list:
            print(f'R@{k}: {total_recall[k] / len(results):.4f} ', end='')
        print()

    # 保存结果到文件
    with open(output_path, 'w') as f:
        json.dump(results, f)
    # 打印保存的结果数量
    print(f'Saved {len(results)} results to {output_path}')
```