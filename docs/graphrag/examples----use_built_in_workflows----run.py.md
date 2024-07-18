# `.\graphrag\examples\use_built_in_workflows\run.py`

```py
# 导入必要的模块和库
import asyncio
import os

# 导入图形数据库索引相关的模块和函数
from graphrag.index import run_pipeline, run_pipeline_with_config
from graphrag.index.config import PipelineCSVInputConfig, PipelineWorkflowReference
from graphrag.index.input import load_input

# 定义示例数据目录路径
sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../_sample_data/"
)

# 一次性加载数据集
shared_dataset = asyncio.run(
    load_input(
        PipelineCSVInputConfig(
            file_pattern=".*\\.csv$",
            base_dir=sample_data_dir,
            source_column="author",
            text_column="message",
            timestamp_column="date(yyyyMMddHHmmss)",
            timestamp_format="%Y%m%d%H%M%S",
            title_column="message",
        ),
    )
)

async def run_with_config():
    """使用配置文件运行流水线"""
    # 从共享数据集中取出前 10 行数据
    dataset = shared_dataset.head(10)

    # 加载当前目录下的 pipeline.yml 文件
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # 执行带有配置的流水线，并收集结果表格
    tables = []
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        tables.append(table)
    pipeline_result = tables[-1]

    # 如果流水线结果不为空，则输出相关信息
    if pipeline_result.result is not None:
        # 输出第一个结果的级别信息
        print(f"level: {first_result['level'][0]}")
        # 输出第一个结果的嵌入信息
        print(f"embeddings: {first_result['embeddings'][0]}")
        # 输出第一个结果的实体图节点位置信息
        print(f"entity_graph_positions: {first_result['node_positions'][0]}")
    else:
        # 如果流水线结果为空，则打印无结果的消息
        print("No results!")


async def run_python():
    # 从共享数据集中取出前 10 行数据
    dataset = shared_dataset.head(10)
    workflows: list[PipelineWorkflowReference] = [
        # 定义流水线工作流引用列表，包含两个 PipelineWorkflowReference 对象

        # 定义实体抽取工作流引用，配置使用 NLTK 策略
        PipelineWorkflowReference(
            name="entity_extraction",
            config={
                "entity_extract": {
                    "strategy": {
                        "type": "nltk",
                    }
                }
            },
        ),

        # 定义实体图工作流引用，配置包括聚类图、嵌入图和布局图的策略
        PipelineWorkflowReference(
            name="entity_graph",
            config={
                "cluster_graph": {"strategy": {"type": "leiden"}},
                "embed_graph": {
                    "strategy": {
                        "type": "node2vec",
                        "num_walks": 10,
                        "walk_length": 40,
                        "window_size": 2,
                        "iterations": 3,
                        "random_seed": 597832,
                    }
                },
                "layout_graph": {
                    "strategy": {
                        "type": "umap",
                    },
                },
            },
        ),
    ]

    # 从流水线中获取最后一个结果，这应该是我们的实体抽取结果
    tables = []
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)
    pipeline_result = tables[-1]

    # 输出将包含每个层次级别的实体图，每个实体的嵌入
    if pipeline_result.result is not None:
        first_result = pipeline_result.result.head(1)
        print(f"level: {first_result['level'][0]}")
        print(f"embeddings: {first_result['embeddings'][0]}")
        print(f"entity_graph_positions: {first_result['node_positions'][0]}")
    else:
        print("No results!")
# 如果这个模块是直接运行的主程序入口
if __name__ == "__main__":
    # 运行 asyncio 事件循环，并执行 run_python() 函数
    asyncio.run(run_python())
    # 运行 asyncio 事件循环，并执行 run_with_config() 函数
    asyncio.run(run_with_config())
```