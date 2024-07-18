# `.\graphrag\examples\various_levels_of_configs\workflows_only.py`

```py
# 导入 asyncio 和 os 模块
import asyncio
import os

# 导入 graphrag 库中的相关模块和函数
from graphrag.index import run_pipeline_with_config
from graphrag.index.config import PipelineCSVInputConfig
from graphrag.index.input import load_input

# 定义示例数据目录路径为当前文件目录下的 _sample_data 文件夹
sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../_sample_data/"
)

# 异步函数，主程序入口
async def main():
    # 检查环境变量中是否设置了 OpenAI API key
    if (
        "EXAMPLE_OPENAI_API_KEY" not in os.environ
        and "OPENAI_API_KEY" not in os.environ
    ):
        msg = "Please set EXAMPLE_OPENAI_API_KEY or OPENAI_API_KEY environment variable to run this example"
        # 抛出异常，提示用户设置 API key
        raise Exception(msg)

    # 使用 PipelineCSVInputConfig 加载数据集，设置相关参数
    dataset = await load_input(
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

    # 仅保留数据集中的前 10 条记录
    dataset = dataset.head(10)

    # 指定流水线配置文件路径
    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipelines/workflows_only.yml"
    )
    tables = []
    # 使用给定的配置文件路径和数据集运行流水线，并将结果表追加到 tables 列表中
    async for table in run_pipeline_with_config(pipeline_path, dataset=dataset):
        tables.append(table)
    # 获取流水线运行的最后一个结果
    pipeline_result = tables[-1]

    # 输出流水线运行结果的列信息和前 10 条数据
    if pipeline_result.result is not None:
        top_nodes = pipeline_result.result.head(10)
        print(
            "pipeline result\ncols: ", pipeline_result.result.columns, "\n", top_nodes
        )
    else:
        # 如果结果为空，则输出 "No results!"
        print("No results!")


# 程序入口，运行主程序 main()
if __name__ == "__main__":
    asyncio.run(main())
```