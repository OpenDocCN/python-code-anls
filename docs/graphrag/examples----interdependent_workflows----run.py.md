# `.\graphrag\examples\interdependent_workflows\run.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入 asyncio 模块，支持异步编程
import asyncio
# 引入操作系统功能的 os 模块
import os

# 引入 pandas 库，用于数据处理
import pandas as pd

# 从 graphrag.index 中引入 run_pipeline 和 run_pipeline_with_config 函数
from graphrag.index import run_pipeline, run_pipeline_with_config
# 从 graphrag.index.config 中引入 PipelineWorkflowReference 类
from graphrag.index.config import PipelineWorkflowReference

# 创建一个虚拟的数据集 DataFrame
dataset = pd.DataFrame([
    {"type": "A", "col1": 2, "col2": 4},
    {"type": "A", "col1": 5, "col2": 10},
    {"type": "A", "col1": 15, "col2": 26},
    {"type": "B", "col1": 6, "col2": 15},
])

# 异步函数：使用配置文件运行流水线
async def run_with_config():
    """Run a pipeline with a config file"""
    # 获取 pipeline.yml 文件的完整路径，该文件位于当前脚本所在目录下
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # 初始化一个空列表用于存储表格数据
    tables = []
    # 使用 run_pipeline_with_config 函数异步迭代运行流水线
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        # 将每个表格数据追加到 tables 列表中
        tables.append(table)
    
    # 获取流水线的最后一个结果
    pipeline_result = tables[-1]

    # 如果流水线的结果不为 None
    if pipeline_result.result is not None:
        # 打印流水线的结果，期望格式如下，并与 Python 示例完全一致：
        #     type  aggregated_output
        # 0    A                448
        # 1    B                 90
        print(pipeline_result.result)
    else:
        # 如果流水线没有产生结果，则打印 "No results!"
        print("No results!")


# 定义一个异步函数，用 Python 代码运行流水线
async def run_python():
    # 定义一个列表，包含 PipelineWorkflowReference 对象
    workflows: list[PipelineWorkflowReference] = [
        # 第一个流水线参考，名称为 "aggregate_workflow"
        PipelineWorkflowReference(
            name="aggregate_workflow",
            steps=[
                {
                    "verb": "aggregate",  # 使用聚合操作
                    "args": {
                        "groupby": "type",  # 按 "type" 列分组
                        "column": "col_multiplied",  # 新生成的列名为 "col_multiplied"
                        "to": "aggregated_output",  # 输出列名为 "aggregated_output"
                        "operation": "sum",  # 执行求和操作
                    },
                    "input": {
                        "source": "workflow:derive_workflow",  # 引用 "derive_workflow"，需要先执行该流水线
                        # 注意，这些步骤的顺序可能不同，索引引擎会确定正确的运行顺序
                    },
                }
            ],
        ),
        # 第二个流水线参考，名称为 "derive_workflow"
        PipelineWorkflowReference(
            name="derive_workflow",
            steps=[
                {
                    "verb": "derive",  # 使用派生操作
                    "args": {
                        "column1": "col1",  # 使用 "col1" 列作为第一个操作数
                        "column2": "col2",  # 使用 "col2" 列作为第二个操作数
                        "to": "col_multiplied",  # 新生成的列名为 "col_multiplied"
                        "operator": "*",  # 执行乘法运算
                    },
                    # 由于我们要在默认输入上执行操作，因此不需要显式指定输入
                }
            ],
        ),
    ]
    # 从管道中获取最后一个结果，应该是我们的 aggregate_workflow，因为它应该是最后一个运行的
    tables = []
    # 使用异步迭代来运行管道，获取每个表格并添加到列表中
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)
    # 获取管道运行的最后一个结果
    pipeline_result = tables[-1]

    # 如果管道结果不为 None
    if pipeline_result.result is not None:
        # 输出结果，通常类似于以下格式：
        #     type  aggregated_output
        # 0    A                448
        # 1    B                 90

        # 这是因为在 "derive_workflow" 中我们首先将 col1 和 col2 相乘，然后在 "aggregate_workflow" 中按 type 进行求和
        print(pipeline_result.result)
    else:
        # 如果结果为 None，则输出 "No results!"
        print("No results!")
# 如果这个模块是直接被执行的主程序（而不是被导入到其他模块中执行）
if __name__ == "__main__":
    # 使用 asyncio 模块的 run 函数来运行 run_python 协程
    asyncio.run(run_python())
    # 使用 asyncio 模块的 run 函数来运行 run_with_config 协程
    asyncio.run(run_with_config())
```