# `.\graphrag\examples\entity_extraction\with_nltk\run.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入 asyncio 和 os 库
import asyncio
import os

# 从 graphrag.index 中引入 run_pipeline 和 run_pipeline_with_config 函数
from graphrag.index import run_pipeline, run_pipeline_with_config
# 从 graphrag.index.config 中引入 PipelineCSVInputConfig 和 PipelineWorkflowReference 类
from graphrag.index.config import PipelineCSVInputConfig, PipelineWorkflowReference
# 从 graphrag.index.input 中引入 load_input 函数
from graphrag.index.input import load_input

# 设置示例数据目录为当前文件所在目录的上两级目录下的 _sample_data/ 目录
sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../_sample_data/"
)

# 使用 asyncio.run 来加载输入数据集
shared_dataset = asyncio.run(
    load_input(
        PipelineCSVInputConfig(
            file_pattern=".*\\.csv$",
            base_dir=sample_data_dir,  # 基础目录为示例数据目录
            source_column="author",  # 指定来源列为 'author'
            text_column="message",  # 指定文本列为 'message'
            timestamp_column="date(yyyyMMddHHmmss)",  # 指定时间戳列
            timestamp_format="%Y%m%d%H%M%S",  # 时间戳格式为年月日时分秒
            title_column="message",  # 指定标题列为 'message'
        ),
    )
)


async def run_with_config():
    """Run a pipeline with a config file"""
    # 获取共享数据集的前10条记录作为示例数据集
    dataset = shared_dataset.head(10)

    # 构建 pipeline.yml 文件的完整路径
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # 通过配置文件运行 pipeline，并收集结果表
    tables = []
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    # 打印实体信息，每个文本单元会对应一个实体列表
    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


async def run_python():
    """Run a pipeline programmatically"""
    # 获取共享数据集的前10条记录作为示例数据集
    dataset = shared_dataset.head(10)

    # 定义工作流列表，指定工作流名称为 'entity_extraction'，并设置配置项为 nltk 策略
    workflows: list[PipelineWorkflowReference] = [
        PipelineWorkflowReference(
            name="entity_extraction",
            config={"entity_extract": {"strategy": {"type": "nltk"}}},
        )
    ]

    # 通过程序化方式运行 pipeline，并收集结果表
    tables = []
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)
    pipeline_result = tables[-1]

    # 打印实体信息，每个文本单元会对应一个实体列表
    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


if __name__ == "__main__":
    # 运行 run_python() 函数
    asyncio.run(run_python())
    # 运行 run_with_config() 函数
    asyncio.run(run_with_config())
```