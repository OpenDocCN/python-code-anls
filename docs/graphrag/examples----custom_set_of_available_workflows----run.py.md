# `.\graphrag\examples\custom_set_of_available_workflows\run.py`

```py
# 导入必要的库和模块
import asyncio  # 异步编程库
import os  # 系统操作库

import pandas as pd  # 数据处理库

# 导入自定义工作流定义
from examples.custom_set_of_available_workflows.custom_workflow_definitions import (
    custom_workflows,
)
# 导入图数据库索引相关模块和函数
from graphrag.index import run_pipeline, run_pipeline_with_config
from graphrag.index.config import PipelineWorkflowReference

# 设置样本数据目录路径
sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../_sample_data/"
)

# 定义我们的虚拟数据集
dataset = pd.DataFrame([{"col1": 2, "col2": 4}, {"col1": 5, "col2": 10}])


async def run_with_config():
    """使用配置文件运行流水线"""
    # 加载当前目录下的 pipeline.yml 配置文件
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # 从流水线中获取最后一个结果，应为实体抽取结果
    tables = []
    async for table in run_pipeline_with_config(
        config_or_path=config_path,
        dataset=dataset,
        additional_workflows=custom_workflows,
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        # 结果应该类似于以下内容:
        #    col1  col2  col_1_multiplied
        # 0     2     4                 8
        # 1     5    10                50
        print(pipeline_result.result)
    else:
        print("No results!")


async def run_python():
    """使用 Python API 运行流水线"""
    # 定义要运行的实际工作流，与 Python API 相同
    workflows: list[PipelineWorkflowReference] = [
        # 运行名为 "my_workflow" 的工作流，注意我们仅使用了 custom_workflows 字典中的 "my_workflow" 而不使用 "my_unused_workflow"
        PipelineWorkflowReference(
            name="my_workflow",  # 与 custom_workflows 中工作流的名称对应
            config={  # 传入配置
                "derive_output_column": "col_1_multiplied"  # 设置输出列为 "col_1_multiplied"，将传递给上面定义的工作流定义
            },
        ),
    ]

    # 从流水线中获取最后一个结果，应为实体抽取结果
    tables = []
    async for table in run_pipeline(
        workflows, dataset=dataset, additional_workflows=custom_workflows
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    if pipeline_result.result is not None:
        # 结果应该类似于以下内容:
        #    col1  col2  col_1_multiplied
        # 0     2     4                 8
        # 1     5    10                50
        print(pipeline_result.result)
    else:
        print("No results!")


if __name__ == "__main__":
    # 运行使用 Python API 的流水线
    asyncio.run(run_python())
    # 运行使用配置文件的流水线
    asyncio.run(run_with_config())
```