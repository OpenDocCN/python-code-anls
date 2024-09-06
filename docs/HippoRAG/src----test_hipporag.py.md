# `.\HippoRAG\src\test_hipporag.py`

```py
# 导入 sys 模块
import sys

# 将当前目录添加到模块搜索路径中
sys.path.append('.')

# 从 src.langchain_util 模块导入 LangChainModel 类
from src.langchain_util import LangChainModel
# 从 src.qa.qa_reader 模块导入 qa_read 函数
from src.qa.qa_reader import qa_read

# 导入 argparse 模块，用于解析命令行参数
import argparse

# 从 src.hipporag 模块导入 HippoRAG 类
from src.hipporag import HippoRAG

# 如果作为主程序运行
if __name__ == '__main__':
    # 创建 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加 --dataset 参数，类型为 str，必填，默认值为 'sample'
    parser.add_argument('--dataset', type=str, required=True, default='sample')
    # 添加 --extraction_model 参数，类型为 str，默认值为 'gpt-3.5-turbo-1106'
    parser.add_argument('--extraction_model', type=str, default='gpt-3.5-turbo-1106')
    # 添加 --retrieval_model 参数，类型为 str，必填，帮助信息说明
    parser.add_argument('--retrieval_model', type=str, required=True, help='e.g., "facebook/contriever", "colbertv2"')
    # 添加 --doc_ensemble 参数，作为布尔值开关
    parser.add_argument('--doc_ensemble', action='store_true')
    # 添加 --dpr_only 参数，作为布尔值开关
    parser.add_argument('--dpr_only', action='store_true')
    # 解析命令行参数
    args = parser.parse_args()

    # 确保不能同时使用 --doc_ensemble 和 --dpr_only
    assert not (args.doc_ensemble and args.dpr_only)
    # 创建 HippoRAG 对象，传入命令行参数和默认值
    hipporag = HippoRAG(args.dataset, 'openai', args.extraction_model, args.retrieval_model, doc_ensemble=args.doc_ensemble, dpr_only=args.dpr_only,
                        qa_model=LangChainModel('openai', 'gpt-3.5-turbo'))

    # 定义查询列表
    queries = ["Which Stanford University professor works on Alzheimer's"]
    # qa_few_shot_samples 初始化为 None，预备用于少量样本 QA
    qa_few_shot_samples = None

    # 遍历查询列表
    for query in queries:
        # 排序文档，获取排名、分数和日志
        ranks, scores, logs = hipporag.rank_docs(query, top_k=10)
        # 根据排名从 HippoRAG 对象中获取检索到的段落
        retrieved_passages = [hipporag.get_passage_by_idx(rank) for rank in ranks]

        # 调用 qa_read 函数获取回答
        response = qa_read(query, retrieved_passages, qa_few_shot_samples, hipporag.qa_model)
        # 打印排名
        print(ranks)
        # 打印分数
        print(scores)
        # 打印回答
        print(response)
```