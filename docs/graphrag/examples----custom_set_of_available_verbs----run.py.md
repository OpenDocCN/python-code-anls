# `.\graphrag\examples\custom_set_of_available_verbs\run.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入 asyncio 和 os 模块
import asyncio
import os

# 引入 pandas 库，并将示例数据集定义为 DataFrame
import pandas as pd

# 从自定义模块中导入自定义动词定义
from examples.custom_set_of_available_verbs.custom_verb_definitions import custom_verbs

# 从 graphrag 模块中导入索引运行函数
from graphrag.index import run_pipeline, run_pipeline_with_config
# 从 graphrag.index.config 中导入流水线工作流参考类
from graphrag.index.config import PipelineWorkflowReference

# 我们的虚拟数据集
dataset = pd.DataFrame([{"col1": 2, "col2": 4}, {"col1": 5, "col2": 10}])


async def run_with_config():
    """使用配置文件运行流水线"""
    # 加载当前目录下的 pipeline.yml 配置文件
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    outputs = []
    # 使用配置文件运行流水线，并异步获取输出结果
    async for output in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        outputs.append(output)
    # 获取流水线运行的最后一个结果
    pipeline_result = outputs[-1]

    if pipeline_result.result is not None:
        # 输出结果，展示包含自定义动词处理后的数据
        # 结果类似如下格式，应该与 Python 示例完全一致：
        #    col1  col2  col_1_custom
        # 0     2     4  2 - custom verb
        # 1     5    10  5 - custom verb
        print(pipeline_result.result)
    else:
        print("No results!")


async def run_python():
    # 定义工作流列表，包含一个流水线工作流参考对象
    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            name="my_workflow",
            steps=[
                {
                    "verb": "str_append",  # 应当是传递给 custom_verbs 字典的键
                    "args": {
                        "source_column": "col1",  # 来自上述定义的列
                        "target_column": "col_1_custom",  # 新列名称
                        "string_to_append": " - custom verb",  # 要附加到列的字符串
                    },
                    # 由于我们尝试对默认输入进行操作，因此不需要显式指定输入
                }
            ],
        ),
    ]

    outputs = []
    # 运行流水线，异步获取输出结果
    async for output in run_pipeline(
        dataset=dataset,
        workflows=workflows,
        additional_verbs=custom_verbs,
    ):
        outputs.append(output)

    # 找到我们关心的工作流结果
    pipeline_result = next(
        (output for output in outputs if output.workflow == "my_workflow"), None
    )

    if pipeline_result is not None and pipeline_result.result is not None:
        # 输出结果，展示包含自定义动词处理后的数据
        # 结果类似如下格式：
        #    col1  col2     col_1_custom
        # 0     2     4  2 - custom verb
        # 1     5    10  5 - custom verb
        print(pipeline_result.result)
    else:
        print("No results!")


if __name__ == "__main__":
    # 运行 Python 示例
    asyncio.run(run_python())
    # 使用配置文件运行流水线
    asyncio.run(run_with_config())
```