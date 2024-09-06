# `.\HippoRAG\src\openie_with_retrieval_option_parallel.py`

```py
import sys  # 导入 sys 模块以便访问系统相关的功能

sys.path.append('.')  # 将当前目录添加到模块搜索路径中

from langchain_community.chat_models import ChatOllama, ChatLlamaCpp  # 从 langchain_community 模块中导入 ChatOllama 和 ChatLlamaCpp 类

import argparse  # 导入 argparse 模块用于解析命令行参数
import json  # 导入 json 模块用于处理 JSON 数据
from glob import glob  # 从 glob 模块导入 glob 函数，用于文件路径匹配

import numpy as np  # 导入 numpy 库并以 np 作为别名
from langchain_openai import ChatOpenAI  # 从 langchain_openai 模块中导入 ChatOpenAI 类

import ipdb  # 导入 ipdb 模块用于调试
from multiprocessing import Pool  # 从 multiprocessing 模块导入 Pool 类用于多进程处理
from tqdm import tqdm  # 从 tqdm 模块导入 tqdm 函数用于显示进度条

from src.langchain_util import init_langchain_model  # 从 src.langchain_util 模块中导入 init_langchain_model 函数
from src.openie_extraction_instructions import ner_prompts, openie_post_ner_prompts  # 从 src.openie_extraction_instructions 模块中导入 ner_prompts 和 openie_post_ner_prompts
from src.processing import extract_json_dict  # 从 src.processing 模块中导入 extract_json_dict 函数


def print_messages(messages):  # 定义 print_messages 函数，接收消息列表作为参数
    for message in messages:  # 遍历每条消息
        print(message['content'])  # 打印消息内容


def named_entity_recognition(passage: str):  # 定义 named_entity_recognition 函数，接收一个字符串参数 passage
    ner_messages = ner_prompts.format_prompt(user_input=passage)  # 使用 ner_prompts 模板格式化消息

    not_done = True  # 初始化 not_done 标志为 True

    total_tokens = 0  # 初始化 total_tokens 为 0
    response_content = '{}'  # 初始化 response_content 为一个空的 JSON 字符串

    while not_done:  # 当 not_done 为 True 时循环
        try:
            if isinstance(client, ChatOpenAI):  # 如果 client 是 ChatOpenAI 实例
                chat_completion = client.invoke(ner_messages.to_messages(), temperature=0, response_format={"type": "json_object"})  # 调用 ChatOpenAI 的 invoke 方法
                response_content = chat_completion.content  # 获取响应内容
                response_content = eval(response_content)  # 将响应内容解析为字典
                total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']  # 累加 token 使用量
            elif isinstance(client, ChatOllama) or isinstance(client, ChatLlamaCpp):  # 如果 client 是 ChatOllama 或 ChatLlamaCpp 实例
                response_content = client.invoke(ner_messages.to_messages())  # 调用 client 的 invoke 方法
                response_content = extract_json_dict(response_content)  # 解析响应内容为字典
                total_tokens += len(response_content.split())  # 统计并累加 token 数量
            else:  # 如果 client 不是上述类型
                chat_completion = client.invoke(ner_messages.to_messages(), temperature=0)  # 调用 client 的 invoke 方法
                response_content = chat_completion.content  # 获取响应内容
                response_content = extract_json_dict(response_content)  # 解析响应内容为字典
                total_tokens += chat_completion.response_metadata['token_usage']['total_tokens']  # 累加 token 使用量

            if 'named_entities' not in response_content:  # 如果响应内容中没有 'named_entities' 键
                response_content = []  # 设置响应内容为空列表
            else:
                response_content = response_content['named_entities']  # 从响应内容中提取 'named_entities'

            not_done = False  # 将 not_done 标志设置为 False，退出循环
        except Exception as e:  # 捕获所有异常
            print('Passage NER exception')  # 打印异常信息
            print(e)  # 打印具体异常内容

    return response_content, total_tokens  # 返回命名实体和总 token 数量


def openie_post_ner_extract(passage: str, entities: list, model: str):  # 定义 openie_post_ner_extract 函数，接收 passage、entities 和 model 参数
    named_entity_json = {"named_entities": entities}  # 创建一个包含命名实体的字典
    openie_messages = openie_post_ner_prompts.format_prompt(passage=passage, named_entity_json=json.dumps(named_entity_json))  # 使用 openie_post_ner_prompts 模板格式化消息
    try:
        # 根据不同客户端类型处理响应
        if isinstance(client, ChatOpenAI):  # 如果客户端是 ChatOpenAI，使用 JSON 模式
            # 调用 ChatOpenAI 接口，获取 JSON 格式响应
            chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096, response_format={"type": "json_object"})
            # 获取响应内容
            response_content = chat_completion.content
            # 获取总使用的 tokens 数量
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']
        elif isinstance(client, ChatOllama) or isinstance(client, ChatLlamaCpp):  # 如果客户端是 ChatOllama 或 ChatLlamaCpp
            # 调用接口，获取原始响应
            response_content = client.invoke(openie_messages.to_messages())
            # 提取 JSON 字典
            response_content = extract_json_dict(response_content)
            # 转换为字符串
            response_content = str(response_content)
            # 计算 tokens 数量
            total_tokens = len(response_content.split())
        else:  # 如果客户端不支持 JSON 模式
            # 调用接口，获取响应内容
            chat_completion = client.invoke(openie_messages.to_messages(), temperature=0, max_tokens=4096)
            # 获取响应内容
            response_content = chat_completion.content
            # 提取 JSON 字典
            response_content = extract_json_dict(response_content)
            # 转换为字符串
            response_content = str(response_content)
            # 获取总使用的 tokens 数量
            total_tokens = chat_completion.response_metadata['token_usage']['total_tokens']

    except Exception as e:
        # 捕获异常并打印错误信息，返回空字符串和零
        print('OpenIE exception', e)
        return '', 0

    # 返回响应内容和 tokens 数量
    return response_content, total_tokens
# 如果这个脚本是直接运行的入口
if __name__ == '__main__':
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个字符串类型的参数 --dataset
    parser.add_argument('--dataset', type=str)
    # 添加一个布尔值参数 --run_ner
    parser.add_argument('--run_ner', action='store_true')
    # 添加一个字符串类型的参数 --num_passages，默认为 '10'
    parser.add_argument('--num_passages', type=str, default='10')
    # 添加一个字符串类型的参数 --llm，默认为 'openai'，提供帮助信息
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    # 添加一个字符串类型的参数 --model_name，默认为 'gpt-3.5-turbo-1106'，提供帮助信息
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    # 添加一个整数类型的参数 --num_processes，默认为 10
    parser.add_argument('--num_processes', type=int, default=10)

    # 解析命令行参数
    args = parser.parse_args()

    # 将参数赋值给对应的变量
    dataset = args.dataset
    run_ner = args.run_ner
    num_passages = args.num_passages
    model_name = args.model_name

    # 从 JSON 文件中加载语料库
    corpus = json.load(open(f'data/{dataset}_corpus.json', 'r'))

    # 根据数据集名称的不同格式化语料库
    if 'hotpotqa' in dataset:
        keys = list(corpus.keys())
        # 生成检索语料库，每个条目包含索引和拼接后的段落
        retrieval_corpus = [{'idx': i, 'passage': key + '\n' + ''.join(corpus[key])} for i, key in enumerate(keys)]
    else:
        retrieval_corpus = corpus
        # 更新每个文档的段落格式
        for document in retrieval_corpus:
            document['passage'] = document['title'] + '\n' + document['text']

    # 更新数据集名称
    dataset = '_' + dataset

    # 处理 num_passages 参数，设置为 'all' 时表示所有条目
    if num_passages == 'all':
        num_passages = len(retrieval_corpus)
    else:
        try:
            num_passages = int(num_passages)
        except:
            # 确保 num_passages 是整数或 'all'
            assert False, "Set 'num_passages' to an integer or 'all'"

    # 根据是否启用 NER 生成标志列表
    flag_names = ['ner']
    flags_present = [flag_names[i] for i, flag in enumerate([run_ner]) if flag]
    # 根据参数生成最终的参数字符串
    if len(flags_present) > 0:
        arg_str = '_'.join(flags_present) + '_' + model_name.replace('/', '_') + f'_{num_passages}'
    else:
        arg_str = model_name.replace('/', '_') + f'_{num_passages}'

    # 打印生成的参数字符串
    print(arg_str)

    # 初始化 LangChain 模型
    client = init_langchain_model(args.llm, model_name)  # LangChain model
    already_done = False

    try:
        # 获取与当前设置相同的未完成提取输出
        arg_str_regex = arg_str.replace(str(num_passages), '*')

        prev_num_passages = 0
        new_json_temp = None

        # 查找符合条件的输出 JSON 文件
        for file in glob('output/openie{}_results_{}.json'.format(dataset, arg_str_regex)):
            possible_json = json.load(open(file, 'r'))
            if prev_num_passages < len(possible_json['docs']):
                prev_num_passages = len(possible_json['docs'])
                new_json_temp = possible_json

        # 从新 JSON 文件中提取文档和实体数据
        existing_json = new_json_temp['docs']
        if 'ents_by_doc' in new_json_temp:
            ents_by_doc = new_json_temp['ents_by_doc']
        elif 'non_dedup_ents_by_doc' in new_json_temp:
            ents_by_doc = new_json_temp['non_dedup_ents_by_doc']
        else:
            ents_by_doc = []

        # 判断是否已经完成处理
        if num_passages < len(existing_json):
            already_done = True
    except:
        # 如果出现异常，初始化为空数据
        existing_json = []
        ents_by_doc = []

    # 生成辅助文件的文件名模式
    aux_file_str = '_'.join(flags_present) + '*_' + model_name + f'_{args.num_passages}'
    aux_file_str = aux_file_str.replace('{}'.format(num_passages), '*')
    # 根据指定模式和数据集参数，查找文件名匹配的 JSON 文件列表
    auxiliary_files = glob('output/openie{}_results_{}.json'.format(dataset, aux_file_str))

    # 初始化辅助文件存在标志为 False
    auxiliary_file_exists = False

    # 如果找到的辅助文件列表长度大于 0，则进行处理
    if len(auxiliary_files) > 0:
        # 遍历每个辅助文件
        for auxiliary_file in auxiliary_files:
            # 打开并加载 JSON 文件中的内容
            aux_info_json = json.load(open(auxiliary_file, 'r'))
            # 如果 JSON 文件中的文档数量大于等于指定的段落数量
            if len(aux_info_json['docs']) >= num_passages:
                # 获取文档实体信息
                ents_by_doc = aux_info_json["ents_by_doc"]
                # 将辅助文件存在标志设置为 True
                auxiliary_file_exists = True
                # 打印正在使用的辅助文件名
                print('Using Auxiliary File: {}'.format(auxiliary_file))
                # 跳出循环
                break

    # 定义函数：从三元组 JSON 中提取 OpenIE 结果
    def extract_openie_from_triples(triple_json):

        # 初始化新 JSON 结果列表和所有实体列表
        new_json = []
        all_entities = []

        # 初始化 ChatGPT 总标记数
        chatgpt_total_tokens = 0

        # 遍历三元组 JSON
        for i, r in tqdm(triple_json, total=len(triple_json)):

            # 获取段落文本
            passage = r['passage']

            # 如果索引 i 小于现有 JSON 长度，直接添加到新 JSON 结果中
            if i < len(existing_json):
                new_json.append(existing_json[i])
            else:
                # 如果辅助文件存在，则使用预先提取的实体信息
                if auxiliary_file_exists:
                    doc_entities = ents_by_doc[i]
                else:
                    # 否则，进行命名实体识别，并获取文档实体列表及总 NER 标记数
                    doc_entities, total_ner_tokens = named_entity_recognition(passage)

                    # 去重实体列表
                    doc_entities = list(np.unique(doc_entities))
                    # 累加 ChatGPT 总标记数
                    chatgpt_total_tokens += total_ner_tokens

                    # 将实体列表添加到 ents_by_doc 中
                    ents_by_doc.append(doc_entities)

                # 执行 OpenIE 提取，获取三元组及总标记数
                triples, total_tokens = openie_post_ner_extract(passage, doc_entities, model_name)

                # 累加 ChatGPT 总标记数
                chatgpt_total_tokens += total_tokens

                # 将提取的实体列表和三元组添加到当前结果 r 中
                r['extracted_entities'] = doc_entities

                try:
                    # 尝试解析三元组字符串并存入 r 中的提取三元组字段
                    r['extracted_triples'] = eval(triples)["triples"]
                except:
                    # 若解析失败，打印错误信息，并将提取三元组字段置空列表
                    print('ERROR')
                    print(triples)
                    r['extracted_triples'] = []

                # 将 r 添加到新 JSON 结果列表中
                new_json.append(r)

        # 返回新 JSON 结果列表、所有实体列表及 ChatGPT 总标记数
        return (new_json, all_entities, chatgpt_total_tokens)

    # 提取检索语料的子集，长度为 num_passages
    extracted_triples_subset = retrieval_corpus[:num_passages]

    # 设置进程数为 args.num_processes
    num_processes = args.num_processes

    # 将 extracted_triples_subset 按进程数分割
    splits = np.array_split(range(len(extracted_triples_subset)), num_processes)

    # 初始化参数列表
    args = []

    # 遍历分割后的索引范围，为每个分段创建参数列表
    for split in splits:
        args.append([(i, extracted_triples_subset[i]) for i in split])

    # 如果进程数大于 1，则使用多进程池执行 extract_openie_from_triples 函数
    if num_processes > 1:
        with Pool(processes=num_processes) as pool:
            outputs = pool.map(extract_openie_from_triples, args)
    else:
        # 否则，顺序执行 extract_openie_from_triples 函数
        outputs = [extract_openie_from_triples(arg) for arg in args]

    # 初始化新 JSON 结果列表、所有实体列表及 LM 总标记数
    new_json = []
    all_entities = []
    lm_total_tokens = 0

    # 遍历每个进程的输出结果
    for output in outputs:
        # 扩展新 JSON 结果列表、所有实体列表，并累加 LM 总标记数
        new_json.extend(output[0])
        all_entities.extend(output[1])
        lm_total_tokens += output[2]
    # 检查是否已经完成某个操作，如果没有完成则继续执行以下代码
    if not (already_done):
        # 计算所有实体的平均字符数
        avg_ent_chars = np.mean([len(e) for e in all_entities])
        # 计算所有实体的平均单词数
        avg_ent_words = np.mean([len(e.split()) for e in all_entities])

        # 计算当前的总令牌数
        approx_total_tokens = (len(retrieval_corpus) / num_passages) * lm_total_tokens

        # 创建一个包含额外信息的字典
        extra_info_json = {"docs": new_json,
                           "ents_by_doc": ents_by_doc,
                           "avg_ent_chars": avg_ent_chars,
                           "avg_ent_words": avg_ent_words,
                           "num_tokens": lm_total_tokens,
                           "approx_total_tokens": approx_total_tokens,
                           }
        # 格式化输出文件路径
        output_path = 'output/openie{}_results_{}.json'.format(dataset, arg_str)
        # 将字典转换为 JSON 格式并写入文件
        json.dump(extra_info_json, open(output_path, 'w'))
        # 打印提示信息，告知文件保存位置
        print('OpenIE saved to', output_path)
```