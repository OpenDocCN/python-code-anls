# `.\HippoRAG\src\named_entity_extraction_parallel.py`

```py
# 导入系统相关模块
import sys
# 从 functools 导入 partial 函数
from functools import partial

# 将当前目录添加到系统路径中，以便导入自定义模块
sys.path.append('.')

# 从 src.processing 导入 extract_json_dict 函数
from src.processing import extract_json_dict
# 从 langchain_community 导入 ChatOllama 和 ChatLlamaCpp 模型
from langchain_community.chat_models import ChatOllama, ChatLlamaCpp

# 再次将当前目录添加到系统路径中
sys.path.append('.')
# 从 argparse 导入 ArgumentParser 类
import argparse
# 从 multiprocessing 导入 Pool 类以支持多进程
from multiprocessing import Pool

# 导入 numpy 和 pandas 库用于数据处理
import numpy as np
import pandas as pd
# 从 langchain_core 导入消息相关的类
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
# 从 langchain_core 导入 ChatPromptTemplate 类
from langchain_core.prompts import ChatPromptTemplate
# 从 langchain_openai 导入 ChatOpenAI 类
from langchain_openai import ChatOpenAI

# 从 tqdm 导入 tqdm 类用于显示进度条
from tqdm import tqdm

# 从 src.langchain_util 导入初始化 langchain 模型的函数
from src.langchain_util import init_langchain_model

# 定义用于提取命名实体的查询提示
query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?

"""
# 定义期望的命名实体提取输出的示例
query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

# 定义查询提示模板
query_prompt_template = """
Question: {}

"""

# 定义命名实体识别函数
def named_entity_recognition(client, text: str):
    # 创建查询的消息模板
    query_ner_prompts = ChatPromptTemplate.from_messages([SystemMessage("You're a very effective entity extraction system."),
                                                          HumanMessage(query_prompt_one_shot_input),
                                                          AIMessage(query_prompt_one_shot_output),
                                                          HumanMessage(query_prompt_template.format(text))])
    # 格式化提示消息
    query_ner_messages = query_ner_prompts.format_prompt()

    # 初始化 JSON 模式标志
    json_mode = False
    # 如果客户端是 ChatOpenAI，则使用 JSON 格式
    if isinstance(client, ChatOpenAI):  # JSON mode
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'], response_format={"type": "json_object"})
        response_content = chat_completion.content
        total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
        json_mode = True
    # 如果客户端是 ChatOllama 或 ChatLlamaCpp
    elif isinstance(client, ChatOllama) or isinstance(client, ChatLlamaCpp):
        response_content = client.invoke(query_ner_messages.to_messages())
        response_content = extract_json_dict(response_content)
        total_tokens = len(response_content.split())
    else:  # 其他情况
        chat_completion = client.invoke(query_ner_messages.to_messages(), temperature=0, max_tokens=300, stop=['\n\n'])
        response_content = chat_completion.content
        response_content = extract_json_dict(response_content)
        total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']

    # 如果不是 JSON 模式，进行额外的处理
    if not json_mode:
        try:
            # 确保响应内容包含 'named_entities'
            assert 'named_entities' in response_content
            response_content = str(response_content)
        except Exception as e:
            # 打印异常信息并设置空的命名实体列表
            print('Query NER exception', e)
            response_content = {'named_entities': []}

    # 返回提取的命名实体和总 token 数量
    return response_content, total_tokens

# 定义在文本上运行 NER 的函数（目前尚未完成）
def run_ner_on_texts(client, texts):
    ner_output = []
    total_cost = 0
    # 遍历文本列表，使用 tqdm 显示进度条
    for text in tqdm(texts):
        # 对每个文本进行命名实体识别，并计算处理成本
        ner, cost = named_entity_recognition(client, text)
        # 将命名实体识别的结果添加到结果列表中
        ner_output.append(ner)
        # 累加处理成本
        total_cost += cost

    # 返回命名实体识别结果和总处理成本
    return ner_output, total_cost
# 仅在脚本作为主程序运行时执行下面的代码
if __name__ == '__main__':
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加数据集参数
    parser.add_argument('--dataset', type=str)
    # 添加语言模型参数，默认为 'openai'，并提供帮助信息
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    # 添加模型名称参数，默认为 'gpt-3.5-turbo-1106'，并提供帮助信息
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    # 添加进程数量参数，默认为10，并提供帮助信息
    parser.add_argument('--num_processes', type=int, default=10, help='Number of processes')

    # 解析命令行参数
    args = parser.parse_args()

    # 从解析结果中提取数据集名称和模型名称
    dataset = args.dataset
    model_name = args.model_name

    # 生成输出文件路径，格式化为 'output/数据集名称_queries.named_entity_output.tsv'
    output_file = 'output/{}_queries.named_entity_output.tsv'.format(dataset)

    # 初始化语言模型
    client = init_langchain_model(args.llm, model_name)  # LangChain model

    try:
        # 从 JSON 文件中读取查询数据，路径为 'data/数据集名称.json'
        queries_df = pd.read_json(f'data/{dataset}.json')

        # 检查数据集名称是否包含 'hotpotqa'
        if 'hotpotqa' in dataset:
            # 仅保留 'question' 列，并重命名列
            queries_df = queries_df[['question']]
            queries_df['0'] = queries_df['question']
            queries_df['query'] = queries_df['question']
            query_name = 'query'
        else:
            query_name = 'question'

        try:
            # 尝试读取输出文件，分隔符为制表符
            output_df = pd.read_csv(output_file, sep='\t')
        except:
            # 如果输出文件不存在，初始化为空列表
            output_df = []

        # 检查查询数据和输出数据的行数是否匹配
        if len(queries_df) != len(output_df):
            # 获取查询文本
            queries = queries_df[query_name].values

            # 获取进程数量
            num_processes = args.num_processes

            # 将查询文本分割为多个子集，每个子集分配给一个进程
            splits = np.array_split(range(len(queries)), num_processes)

            # 为每个子集创建参数列表
            args = []

            for split in splits:
                args.append([queries[i] for i in split])

            # 如果进程数量为1，直接处理查询
            if num_processes == 1:
                outputs = [run_ner_on_texts(client, args[0])]
            else:
                # 使用多进程处理查询
                partial_func = partial(run_ner_on_texts, client)
                with Pool(processes=num_processes) as pool:
                    outputs = pool.map(partial_func, args)

            # 初始化总的 ChatGPT 消耗的 token 数量
            chatgpt_total_tokens = 0

            # 初始化查询三元组列表
            query_triples = []

            # 处理每个输出，汇总查询三元组和 token 数量
            for output in outputs:
                query_triples.extend(output[0])
                chatgpt_total_tokens += output[1]

            # 计算当前的费用
            current_cost = 0.002 * chatgpt_total_tokens / 1000

            # 将查询三元组添加到数据框，并保存到输出文件
            queries_df['triples'] = query_triples
            queries_df.to_csv(output_file, sep='\t')
            print('Passage NER saved to', output_file)
        else:
            # 如果输出文件已经存在，打印相关信息
            print('Passage NER already saved to', output_file)
    except Exception as e:
        # 捕获异常，打印错误信息
        print('No queries will be processed for later retrieval.', e)
```