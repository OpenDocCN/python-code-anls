# `.\evaluation.py`

```py
import sys  # 导入sys模块，用于系统相关操作
import io, os  # 导入io和os模块，用于输入输出和操作系统相关功能
import numpy as np  # 导入NumPy库，用于数值计算
import logging  # 导入logging模块，用于日志记录
import argparse  # 导入argparse模块，用于命令行参数解析
from prettytable import PrettyTable  # 从prettytable库中导入PrettyTable类，用于创建漂亮的表格
import torch  # 导入PyTorch库
import transformers  # 导入transformers库
from transformers import AutoModel, AutoTokenizer  # 从transformers库中导入AutoModel和AutoTokenizer类

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
# 配置日志记录格式和级别，这里设置为DEBUG级别

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'  # 设置SentEval库的路径
PATH_TO_DATA = './SentEval/data'  # 设置SentEval数据的路径

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)  # 将SentEval库路径添加到系统路径中，以便导入

import senteval  # 导入SentEval库，用于评估句子表示模型

def print_table(task_names, scores):
    tb = PrettyTable()  # 创建一个PrettyTable对象
    tb.field_names = task_names  # 设置表格的列名
    tb.add_row(scores)  # 添加一行得分数据
    print(tb)  # 打印表格

def main():
    parser = argparse.ArgumentParser()  # 创建一个命令行参数解析器对象
    parser.add_argument("--model_name_or_path", type=str, 
            help="Transformers' model name or path")  # 添加命令行参数，指定模型名称或路径
    parser.add_argument("--pooler", type=str, 
            choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], 
            default='cls', 
            help="Which pooler to use")  # 添加命令行参数，指定使用哪种池化方法
    parser.add_argument("--mode", type=str, 
            choices=['dev', 'test', 'fasttest'],
            default='test', 
            help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")  # 添加命令行参数，指定评估模式
    parser.add_argument("--task_set", type=str, 
            choices=['sts', 'transfer', 'full', 'na'],
            default='sts',
            help="What set of tasks to evaluate on. If not 'na', this will override '--tasks'")  # 添加命令行参数，指定要评估的任务集合
    parser.add_argument("--tasks", type=str, nargs='+', 
            default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                     'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC',
                     'SICKRelatedness', 'STSBenchmark'], 
            help="Tasks to evaluate on. If '--task_set' is specified, this will be overridden")  # 添加命令行参数，指定要评估的具体任务列表
    
    args = parser.parse_args()  # 解析命令行参数，并返回一个包含各参数值的命名空间

    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)  # 根据指定的模型名称或路径加载预训练模型
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)  # 根据指定的模型名称或路径加载模型的分词器
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 根据CUDA是否可用选择设备
    model = model.to(device)  # 将模型移动到相应的设备上
    
    # Set up the tasks
    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}  # 设置SentEval评估的参数
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}  # 设置分类器的参数
    elif args.mode == 'test':
        # 如果参数 mode 是 'test'，则进行完整模式设置
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}
    else:
        # 如果参数 mode 不是 'test'，则抛出未实现错误
        raise NotImplementedError

    # SentEval prepare and batcher
    # 定义 SentEval 的 prepare 函数，但未实现具体功能
    def prepare(params, samples):
        return
    
    # 定义 SentEval 的 batcher 函数，用于处理批处理数据
    def batcher(params, batch, max_length=None):
        # 处理数据集中的稀有标记编码问题
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        # 将批次中的句子列表转换为字符串形式
        sentences = [' '.join(s) for s in batch]

        # 根据 max_length 参数进行标记化处理
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
            )

        # 将处理后的 batch 数据移动到指定设备（通常是 GPU）
        for k in batch:
            batch[k] = batch[k].to(device)
        
        # 获取模型的原始嵌入
        with torch.no_grad():
            # 调用模型进行推理，同时输出隐藏状态
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states

        # 根据不同的池化方式选择返回结果
        if args.pooler == 'cls':
            # 如果选择了 'cls' 池化方式，则返回池化后的输出（CPU 上）
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            # 如果选择了 'cls_before_pooler' 池化方式，则返回 CLS 表示之后的输出（CPU 上）
            return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            # 如果选择了 'avg' 池化方式，则返回平均池化后的输出（CPU 上）
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            # 如果选择了 'avg_first_last' 池化方式，则返回首尾隐藏状态的平均值池化后的输出（CPU 上）
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            # 如果选择了 'avg_top2' 池化方式，则返回倒数第二和最后一个隐藏状态的平均值池化后的输出（CPU 上）
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            # 如果未实现其他池化方式，则抛出未实现错误
            raise NotImplementedError

    results = {}

    # 对每一个任务进行 SentEval 评估
    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result
    
    # 打印评估结果
    # 如果运行模式是开发模式
    if args.mode == 'dev':
        # 打印模式信息
        print("------ %s ------" % (args.mode))
    
        # 初始化任务名称和分数列表
        task_names = []
        scores = []
    
        # 遍历需要评估的开发任务列表
        for task in ['STSBenchmark', 'SICKRelatedness']:
            # 添加任务名称到列表
            task_names.append(task)
            # 如果结果中包含当前任务
            if task in results:
                # 提取并格式化任务的评估分数（Spearman 相关系数乘以 100）
                scores.append("%.2f" % (results[task]['dev']['spearman'][0] * 100))
            else:
                # 如果结果中没有当前任务，分数为 0.00
                scores.append("0.00")
        
        # 打印任务名称和对应分数的表格
        print_table(task_names, scores)
    
        # 初始化任务名称和分数列表（用于下一个任务组）
        task_names = []
        scores = []
    
        # 遍历需要评估的更多开发任务列表
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            # 添加任务名称到列表
            task_names.append(task)
            # 如果结果中包含当前任务
            if task in results:
                # 提取并格式化任务的准确率（accuracy）
                scores.append("%.2f" % (results[task]['devacc']))
            else:
                # 如果结果中没有当前任务，分数为 0.00
                scores.append("0.00")
        
        # 添加平均分的任务名称和对应平均分数到列表末尾
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        
        # 打印任务名称和对应分数的表格
        print_table(task_names, scores)
    
    # 如果运行模式是测试模式或快速测试模式
    elif args.mode == 'test' or args.mode == 'fasttest':
        # 打印模式信息
        print("------ %s ------" % (args.mode))
    
        # 初始化任务名称和分数列表
        task_names = []
        scores = []
    
        # 遍历需要评估的测试任务列表
        for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
            # 添加任务名称到列表
            task_names.append(task)
            # 如果结果中包含当前任务
            if task in results:
                # 如果是STS任务，则提取并格式化任务的Spearman相关系数（乘以100）
                if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                    scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
                # 否则提取并格式化任务的Spearman相关系数
                else:
                    scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
            else:
                # 如果结果中没有当前任务，分数为 0.00
                scores.append("0.00")
        
        # 添加平均分的任务名称和对应平均分数到列表末尾
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        
        # 打印任务名称和对应分数的表格
        print_table(task_names, scores)
    
        # 初始化任务名称和分数列表（用于下一个任务组）
        task_names = []
        scores = []
    
        # 遍历需要评估的更多测试任务列表
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            # 添加任务名称到列表
            task_names.append(task)
            # 如果结果中包含当前任务
            if task in results:
                # 提取并格式化任务的准确率（accuracy）
                scores.append("%.2f" % (results[task]['acc']))
            else:
                # 如果结果中没有当前任务，分数为 0.00
                scores.append("0.00")
        
        # 添加平均分的任务名称和对应平均分数到列表末尾
        task_names.append("Avg.")
        scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
        
        # 打印任务名称和对应分数的表格
        print_table(task_names, scores)
# 如果当前脚本被直接执行而非被导入，则执行下面的代码块
if __name__ == "__main__":
    # 调用主函数执行程序的主要逻辑
    main()
```