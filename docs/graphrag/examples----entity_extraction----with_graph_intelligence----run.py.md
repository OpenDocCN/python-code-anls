# `.\graphrag\examples\entity_extraction\with_graph_intelligence\run.py`

```py
# 引入 asyncio 和 os 模块
import asyncio
import os

# 从 graphrag.index 中引入 run_pipeline 和 run_pipeline_with_config 函数
# 以及 PipelineCSVInputConfig 和 PipelineWorkflowReference 类
from graphrag.index import run_pipeline, run_pipeline_with_config
from graphrag.index.config import PipelineCSVInputConfig, PipelineWorkflowReference
# 从 graphrag.index.input 中引入 load_input 函数
from graphrag.index.input import load_input

# 定义示例数据目录路径，使用当前文件所在目录的父目录下的 _sample_data 子目录
sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../_sample_data/"
)

# 使用 asyncio.run 来加载输入数据集
shared_dataset = asyncio.run(
    load_input(
        PipelineCSVInputConfig(
            file_pattern=".*\\.csv$",  # 匹配以 .csv 结尾的文件
            base_dir=sample_data_dir,  # 基础目录为示例数据目录
            source_column="author",  # 源列为 "author"
            text_column="message",  # 文本列为 "message"
            timestamp_column="date(yyyyMMddHHmmss)",  # 时间戳列为 "date(yyyyMMddHHmmss)"
            timestamp_format="%Y%m%d%H%M%S",  # 时间戳格式为 "%Y%m%d%H%M%S"
            title_column="message",  # 标题列为 "message"
        ),
    )
)


async def run_with_config():
    """使用配置文件运行流水线"""
    # 从共享数据集中取出前 10 行数据
    dataset = shared_dataset.head(10)

    # 加载当前目录下的 pipeline.yml 文件路径
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # 创建一个空列表 tables 来存储流水线运行结果的表格
    tables = []
    # 使用 run_pipeline_with_config 函数异步运行流水线
    async for table in run_pipeline_with_config(
        config_or_path=config_path, dataset=dataset
    ):
        tables.append(table)
    # 获取最后一个表格作为流水线运行的结果
    pipeline_result = tables[-1]

    # 打印实体结果。每个文本单元将包含一个实体列表
    # 结果将类似于 Python 版本，但由于我们使用的是 LLM（语言生成模型）
    # 打印结果可能会根据模型对文本的处理方式有所不同
    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")


async def run_python():
    # 检查环境变量中是否包含必要的 API 密钥
    if (
        "EXAMPLE_OPENAI_API_KEY" not in os.environ
        and "OPENAI_API_KEY" not in os.environ
    ):
        # 如果环境变量中没有相应的 API 密钥，抛出异常并提示设置环境变量
        msg = "Please set EXAMPLE_OPENAI_API_KEY or OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

    # 从共享数据集中取出前 10 行数据
    dataset = shared_dataset.head(10)
    # 定义一个工作流列表，每个工作流参考一个PipelineWorkflowReference对象
    workflows: list[PipelineWorkflowReference] = [
        # 创建一个名为"entity_extraction"的PipelineWorkflowReference对象，配置包含实体提取的参数
        PipelineWorkflowReference(
            name="entity_extraction",
            config={
                "entity_extract": {
                    "strategy": {
                        "type": "graph_intelligence",
                        "llm": {
                            "type": "openai_chat",
                            "api_key": os.environ.get(
                                "EXAMPLE_OPENAI_API_KEY",
                                os.environ.get("OPENAI_API_KEY", None),
                            ),  # 使用环境变量中的API密钥，若无则为None
                            "model": os.environ.get(
                                "EXAMPLE_OPENAI_MODEL", "gpt-3.5-turbo"
                            ),  # 使用环境变量中的模型名称，若无则默认为"gpt-3.5-turbo"
                            "max_tokens": os.environ.get(
                                "EXAMPLE_OPENAI_MAX_TOKENS", 2500
                            ),  # 使用环境变量中的最大token数，若无则默认为2500
                            "temperature": os.environ.get(
                                "EXAMPLE_OPENAI_TEMPERATURE", 0
                            ),  # 使用环境变量中的温度参数，若无则默认为0
                        },
                    }
                }
            },
        )
    ]

    # 从管道中获取最后一个结果，预期是实体提取的结果
    tables = []
    async for table in run_pipeline(dataset=dataset, workflows=workflows):
        tables.append(table)
    pipeline_result = tables[-1]

    # 打印实体结果，每个文本单元会有一个实体列表
    if pipeline_result.result is not None:
        print(pipeline_result.result["entities"].to_list())
    else:
        print("No results!")  # 如果结果为空，则打印"无结果！"
# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 使用 asyncio 库运行 run_python 函数，直到其完成
    asyncio.run(run_python())
    # 使用 asyncio 库运行 run_with_config 函数，直到其完成
    asyncio.run(run_with_config())
```