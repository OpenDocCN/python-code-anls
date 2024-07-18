# `.\graphrag\examples\single_verb\run.py`

```py
# 导入 asyncio 和 os 模块
import asyncio
import os

# 导入 pandas 库并重命名为 pd
import pandas as pd

# 导入自定义模块中的函数和类
from graphrag.index import run_pipeline, run_pipeline_with_config
from graphrag.index.config import PipelineWorkflowReference

# 创建一个虚拟数据集 DataFrame
dataset = pd.DataFrame([{"col1": 2, "col2": 4}, {"col1": 5, "col2": 10}])


async def run_with_config():
    """使用配置文件运行流水线"""
    # 获取当前脚本所在目录，并拼接 pipeline.yml 的路径
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # 用配置文件运行流水线，并异步获取生成的数据表
    tables = []
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        tables.append(table)
    # 获取流水线的最后结果
    pipeline_result = tables[-1]

    # 如果流水线有结果，则打印结果
    if pipeline_result.result is not None:
        # 结果应该类似于以下内容，与 Python 示例应该完全一致：
        #    col1  col2  col_multiplied
        # 0     2     4               8
        # 1     5    10              50
        print(pipeline_result.result)
    else:
        print("No results!")


async def run_python():
    """使用 Python API 运行流水线"""
    # 定义流程引用列表，包含一个 PipelineWorkflowReference 对象
    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            steps=[
                {
                    # 内置动词
                    "verb": "derive",  # https://github.com/microsoft/datashaper/blob/main/python/datashaper/datashaper/engine/verbs/derive.py
                    "args": {
                        "column1": "col1",  # 使用上面定义的列名
                        "column2": "col2",  # 使用上面定义的列名
                        "to": "col_multiplied",  # 新列的名称
                        "operator": "*",  # 对两列进行乘法操作
                    },
                    # 由于我们在默认输入上操作，无需显式指定输入
                }
            ]
        ),
    ]

    # 使用 Python API 运行流水线，并异步获取生成的数据表
    tables = []
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)
    # 获取流水线的最后结果
    pipeline_result = tables[-1]

    # 如果流水线有结果，则打印结果
    if pipeline_result.result is not None:
        # 结果应该类似于以下内容：
        #    col1  col2  col_multiplied
        # 0     2     4               8
        # 1     5    10              50
        print(pipeline_result.result)
    else:
        print("No results!")


if __name__ == "__main__":
    # 运行带有配置的流水线
    asyncio.run(run_with_config())
    # 使用 Python API 运行流水线
    asyncio.run(run_python())
```