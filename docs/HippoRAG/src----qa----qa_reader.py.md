# `.\HippoRAG\src\qa\qa_reader.py`

```py
# 导入系统模块
import sys

# 将上级目录添加到模块搜索路径
sys.path.append('..')

# 从 langchain_core.messages 导入 HumanMessage, SystemMessage 和 AIMessage 类
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# 从 langchain_core.prompts 导入 ChatPromptTemplate 类
from langchain_core.prompts import ChatPromptTemplate

# 从 src.langchain_util 导入 init_langchain_model 函数
from src.langchain_util import init_langchain_model

# 从 src.baselines.ircot 导入 parse_prompt 函数
from src.baselines.ircot import parse_prompt
# 从 src.qa.hotpotqa_evaluation 导入 update_answer 函数
from src.qa.hotpotqa_evaluation import update_answer
# 从 src.qa.musique_evaluation 导入 evaluate 函数
from src.qa.musique_evaluation import evaluate
# 从 src.qa.twowikimultihopqa_evaluation 导入 exact_match_score 和 f1_score 函数
from src.qa.twowikimultihopqa_evaluation import exact_match_score, f1_score

# 导入操作路径的模块
import os.path
# 从 concurrent.futures 导入 ThreadPoolExecutor 和 as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
# 导入命令行参数解析模块
import argparse
# 导入 JSON 处理模块
import json
# 导入进度条库
from tqdm import tqdm


# 定义函数，去除文本中第一个换行符后的所有换行符
def remove_newlines_after_first(s):
    # 查找第一个换行符的位置
    first_newline_pos = s.find('\n')
    # 如果没有换行符，直接返回原文本
    if first_newline_pos == -1:
        return s
    # 获取第一个换行符前的部分
    part_before_first_newline = s[:first_newline_pos + 1]
    # 获取第一个换行符后的部分，并去除所有换行符
    part_after_first_newline = s[first_newline_pos + 1:].replace('\n', '')
    # 返回合并后的文本
    return part_before_first_newline + part_after_first_newline


# 定义带有文档的系统指令
cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')
# 定义不带文档的系统指令
cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.')


# 定义一个函数来处理查询、文档和示例，返回答案
def qa_read(query: str, passages: list, few_shot: list, client):
    """
    # 函数文档字符串，描述函数的参数和返回值
    @param query: 查询字符串
    @param passages: 文档列表
    @param few_shot: few-shot 上下文示例列表
    @param client: Langchain 客户端
    @return: 从文档中得到的答案
    """

    # 根据是否有文档选择系统指令
    instruction = cot_system_instruction if len(passages) else cot_system_instruction_no_doc
    # 创建系统消息
    messages = [SystemMessage(instruction)]
    # 遍历 few-shot 示例
    if few_shot:
        for sample in few_shot:
            # 如果示例包含文档，构造包含文档和问题的文本
            if 'document' in sample:
                cur_sample = f'{sample["document"]}\n\nQuestion: {sample["question"]}'
            # 否则，只构造包含问题的文本
            else:
                cur_sample = f'Question: {sample["question"]}'
            # 如果示例包含 Chain-of-Thought，添加相应的消息
            if 'thought' in sample:
                messages.append(HumanMessage(cur_sample + '\nThought: '))
                messages.append(AIMessage(f'{sample["thought"]}\nAnswer: {sample["answer"]}'))
            # 否则，直接回答问题
            else:
                messages.append(HumanMessage(cur_sample + '\nAnswer: '))
                messages.append(AIMessage(f'Answer: {sample["answer"]}'))
    # 初始化用户提示的字符串为空
    user_prompt = ''
    # 遍历所有的 passages，将每个 passage 加入到用户提示字符串中
    for passage in passages:
        user_prompt += f'Wikipedia Title: {passage}\n\n'
    # 添加问题和思考提示到用户提示字符串
    user_prompt += 'Question: ' + query + '\nThought: '
    # 将用户提示字符串封装成 HumanMessage 对象并添加到消息列表中
    messages.append(HumanMessage(user_prompt))

    # 如果提供了少量示例，则检查消息列表的长度是否正确
    if few_shot:
        assert len(messages) == len(few_shot) * 2 + 2
    # 否则检查消息列表的长度是否为 2
    else:
        assert len(messages) == 2
    # 使用消息列表创建 ChatPromptTemplate 对象并格式化提示
    messages = ChatPromptTemplate.from_messages(messages).format_prompt()
    try:
        # 调用客户端的 invoke 方法生成聊天完成内容
        chat_completion = client.invoke(messages.to_messages())
        # 提取聊天完成内容的实际响应
        response_content = chat_completion.content
    except Exception as e:
        # 捕获异常并输出错误信息，返回空字符串
        print('QA read exception', e)
        return ''
    # 返回聊天生成的响应内容
    return response_content
# 定义一个处理并行 QA 任务的函数，接收多个参数
def parallel_qa_read(data: list, demos: list, args, client, output_path: str, total_metrics: dict, sample_id_set: set):
    # 定义处理单个样本的内部函数
    def process_sample(sample):
        # 解包样本索引和样本数据
        sample_idx, sample = sample
        # 从样本中提取 ID，如果没有，则使用 'id'
        sample_id = sample['_id'] if '_id' in sample else sample['id']
        # 如果样本 ID 已经处理过，则跳过
        if sample_id in sample_id_set:
            return None  # Skip processing if sample already processed
        # 提取查询问题
        query = sample['question']
        # 根据样本是否包含 'retrieved' 或 'retrieved_id' 选择相应的数据
        if 'retrieved' in sample:
            retrieved = sample['retrieved'][:args.num_doc]
        elif 'retrieved_id' in sample:
            retrieved = [corpus[doc_id] for doc_id in sample['retrieved_id']][:args.num_doc]
        else:
            retrieved = []
        # 确保检索到的文档数量与要求一致
        assert len(retrieved) == args.num_doc, f'sample {sample_id}: #retrieved {len(retrieved)} != args.num_doc {args.num_doc}'
        # 如果有检索结果，根据数据类型格式化
        if len(retrieved):
            if isinstance(retrieved[0], dict):
                retrieved = [item['title'] + '\n' + item['text'] for item in retrieved]
            elif isinstance(retrieved[0], list):
                retrieved = ['\n'.join(item) for item in retrieved]

        # 如果数据集是 'hotpotqa'，去除每个结果中第一个换行后的其他换行
        if args.dataset == 'hotpotqa':
            retrieved = [remove_newlines_after_first(item) for item in retrieved]

        # 调用 QA 函数获取答案
        response = qa_read(query, retrieved, demos, client)
        try:
            # 尝试从响应中提取答案
            pred_ans = response.split('Answer:')[1].strip()
        except Exception as e:
            # 如果提取失败，输出错误信息并返回原响应
            print('Parsing prediction:', e, response)
            pred_ans = response

        # 提取样本的正确答案
        gold_ans = sample['answer']
        # 根据数据集类型计算评价指标
        if args.dataset == 'hotpotqa':
            em, f1, precision, recall = update_answer({'em': 0, 'f1': 0, 'precision': 0, 'recall': 0}, pred_ans, gold_ans)
            return sample_idx, sample_id, retrieved, pred_ans, {'em': em, 'f1': f1, 'precision': precision, 'recall': recall}
        elif args.dataset == 'musique':
            em, f1 = evaluate({'predicted_answer': pred_ans}, sample)
            return sample_idx, sample_id, retrieved, pred_ans, {'em': em, 'f1': f1}
        elif args.dataset == '2wikimultihopqa':
            em = 1 if exact_match_score(pred_ans, gold_ans) else 0
            f1, precision, recall = f1_score(pred_ans, gold_ans)
            return sample_idx, sample_id, retrieved, pred_ans, {'em': em, 'f1': f1, 'precision': precision, 'recall': recall}
    # 使用线程池执行器来并行处理任务，最大线程数由参数指定
    with ThreadPoolExecutor(max_workers=args.thread) as executor:
        # 提交每个样本处理任务到线程池，并生成未来对象的列表
        futures = [executor.submit(process_sample, (sample_idx, sample)) for sample_idx, sample in enumerate(data)]
        # 遍历已完成的任务，并显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc='QA read'):
            # 获取任务的结果
            result = future.result()
            # 如果结果不为 None，则处理结果
            if result is not None:
                sample_idx, sample_id, retrieved, pred_ans, metrics = result
                # 将样本 ID 添加到集合中
                sample_id_set.add(sample_id)
                # 获取原始样本数据
                sample = data[sample_idx]
                # 更新样本数据中的检索结果和预测答案
                sample['retrieved'] = retrieved
                sample['prediction'] = pred_ans
                # 将每个指标的值添加到样本中，并更新总指标
                for key in metrics:
                    sample['qa_' + key] = metrics[key]
                    total_metrics['qa_' + key] += metrics[key]

                # 每处理 50 个样本，保存当前数据到文件
                if sample_idx % 50 == 0:
                    with open(output_path, 'w') as f:
                        json.dump(data, f)
# 当脚本作为主程序运行时执行以下代码
if __name__ == '__main__':
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加 '--dataset' 参数，指定数据集类型，并定义可选值
    parser.add_argument('--dataset', type=str, help='retrieval results or QA reading results', choices=['hotpotqa', 'musique', '2wikimultihopqa'], required=True)
    # 添加 '--data' 参数，指定数据文件路径
    parser.add_argument('--data', type=str, help='retrieval results or QA reading results')
    # 添加 '--retriever' 参数，指定检索器名称
    parser.add_argument('--retriever', type=str, help='retriever name to distinguish different experiments')
    # 添加 '--llm' 参数，指定语言模型名称，默认值为 'openai'
    parser.add_argument('--llm', type=str, default='openai', help="LLM, e.g., 'openai' or 'together'")
    # 添加 '--llm_model' 参数，指定具体的模型名称，默认值为 'gpt-3.5-turbo-1106'
    parser.add_argument('--llm_model', type=str, default='gpt-3.5-turbo-1106', help='Specific model name')
    # 添加 '--num_demo' 参数，指定少量示例的数量，默认值为 1
    parser.add_argument('--num_demo', type=int, default=1, help='the number of few-shot examples')
    # 添加 '--num_doc' 参数，指定上下文文档的数量，默认值为 5
    parser.add_argument('--num_doc', type=int, default=5, help='the number of in-context documents')
    # 添加 '--thread' 参数，指定并行处理的工作线程数，默认值为 8
    parser.add_argument('--thread', type=int, default=8, help='the number of workers for parallel processing')
    # 解析传入的命令行参数
    args = parser.parse_args()

    # 构建输出文件路径，包含数据集、检索器、模型和示例数量
    output_path = f'exp/qa_{args.dataset}_{args.retriever}_{args.llm_model}_demo_{args.num_demo}_doc_{args.num_doc}.json'
    # 初始化已处理 ID 集合
    processed_id_set = set()
    # 初始化总度量字典
    total_metrics = {'qa_em': 0, 'qa_f1': 0, 'qa_precision': 0, 'qa_recall': 0}
    # 如果提供了数据文件路径，则加载数据
    if args.data:
        data = json.load(open(args.data, 'r'))
    else:
        # 如果没有提供数据文件路径，打印提示并退出程序
        print('Please provide the retrieval results')
        exit(1)

    # 如果检索器名称为 'none'，则将文档数量设置为 0
    if args.retriever == 'none':
        args.num_doc = 0

    # 如果文档数量为 0，根据数据集类型加载相应的提示和数据
    if args.num_doc == 0:
        if args.dataset == 'hotpotqa':
            prompt_path = 'data/ircot_prompts/hotpotqa/no_context_cot_qa_codex.txt'
            data = json.load(open('data/hotpotqa.json', 'r'))
        elif args.dataset == 'musique':
            prompt_path = 'data/ircot_prompts/musique/no_context_cot_qa_codex.txt'
            data = json.load(open('data/musique.json', 'r'))
        elif args.dataset == '2wikimultihopqa':
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/no_context_cot_qa_codex.txt'
            data = json.load(open('data/2wikimultihopqa.json', 'r'))
        # 解析提示文件
        demos = parse_prompt(prompt_path, False)
    else:
        # 如果 output_path 指定的文件存在，恢复之前的结果
        if os.path.isfile(output_path):  # resume from previous results
            # 从 output_path 文件中加载 JSON 数据
            data = json.load(open(output_path, 'r'))
            # 遍历 total_metrics 字典的每个键，计算该键的总值
            for key in total_metrics.keys():
                total_metrics[key] = sum([sample[key] for sample in data if key in sample])
        # 根据 args.dataset 的值选择不同的数据和提示文件
        if args.dataset == 'hotpotqa':
            # 设置提示文件路径和加载热锅 QA 数据
            prompt_path = 'data/ircot_prompts/hotpotqa/gold_with_3_distractors_context_cot_qa_codex.txt'
            corpus = json.load(open('data/hotpotqa_corpus.json', 'r'))
        elif args.dataset == 'musique':
            # 设置提示文件路径和加载音乐数据
            prompt_path = 'data/ircot_prompts/musique/gold_with_3_distractors_context_cot_qa_codex.txt'
            corpus = json.load(open('data/musique_corpus.json', 'r'))
        elif args.dataset == '2wikimultihopqa':
            # 设置提示文件路径和加载 2wikimultihopqa 数据
            prompt_path = 'data/ircot_prompts/2wikimultihopqa/gold_with_3_distractors_context_cot_qa_codex.txt'
            corpus = json.load(open('data/2wikimultihopqa_corpus.json', 'r'))
        # 解析提示文件
        demos = parse_prompt(prompt_path)

    # 根据数据集选择已处理的 ID 集合
    if args.dataset in ['hotpotqa', '2wikimultihopqa']:
        # 对于 'hotpotqa' 和 '2wikimultihopqa' 数据集，提取 '_id'
        processed_id_set = {sample['_id'] for sample in data if 'prediction' in sample}
    elif args.dataset in ['musique']:
        # 对于 'musique' 数据集，提取 'id'
        processed_id_set = {sample['id'] for sample in data if 'prediction' in sample}

    # 确保数据非空且有至少一个元素
    assert data and len(data)
    # 截取前 num_demo 个演示数据
    demos = demos[:args.num_demo]
    # 初始化语言模型客户端
    client = init_langchain_model(args.llm, args.llm_model)
    # 执行并行的 QA 读取操作
    parallel_qa_read(data, demos, args, client, output_path, total_metrics, processed_id_set)
    # 将更新后的数据保存到 output_path 文件
    with open(output_path, 'w') as f:
        json.dump(data, f)
    # 打印保存结果的消息
    print('QA results saved to', output_path)

    # 计算并打印每个指标的平均值
    metric_str = ' '.join([f'{key}: {total_metrics[key] / len(data):.4f}' for key in total_metrics.keys()])
    print(metric_str)
```