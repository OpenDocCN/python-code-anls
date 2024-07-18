# `.\graphrag\examples\multiple_workflows\run.py`

```py
# 引入 asyncio 模块，用于异步编程
# 引入 os 模块，提供与操作系统相关的功能
import asyncio
import os

# 从 graphrag.index 模块中导入 run_pipeline_with_config 函数
# 从 graphrag.index.config 模块中导入 PipelineCSVInputConfig 类
# 从 graphrag.index.input 模块中导入 load_input 函数
from graphrag.index import run_pipeline_with_config
from graphrag.index.config import PipelineCSVInputConfig
from graphrag.index.input import load_input

# 设置样本数据目录为当前脚本所在目录的相对路径 _sample_data/
sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./../_sample_data/"
)

# 定义异步函数 run_with_config，用于执行带有配置的数据处理流程
async def run_with_config():
    # 使用 load_input 函数加载数据集，传入 PipelineCSVInputConfig 的实例作为参数
    dataset = await load_input(
        PipelineCSVInputConfig(
            file_pattern=".*\\.csv$",  # 匹配以 .csv 结尾的文件
            base_dir=sample_data_dir,   # 指定数据文件的基础目录
            source_column="author",     # 指定源列为 "author"
            text_column="message",      # 指定文本列为 "message"
            timestamp_column="date(yyyyMMddHHmmss)",  # 指定时间戳列
            timestamp_format="%Y%m%d%H%M%S",          # 指定时间戳格式
            title_column="message",     # 指定标题列为 "message"
        ),
    )

    # 对数据集进行处理，仅保留前两行数据
    dataset = dataset.head(2)

    # 构建流水线配置文件的路径，该路径为当前脚本所在目录的 pipeline.yml 文件
    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
    )

    # 使用 run_pipeline_with_config 函数执行配置文件中定义的数据处理流水线
    # 并通过 async for 循环异步处理每一个流水线运行的结果
    async for result in run_pipeline_with_config(pipeline_path, dataset=dataset):
        # 打印每个工作流程的结果信息
        print(f"Workflow {result.workflow} result\n: ")
        print(result.result)


# 如果脚本直接执行（而不是被导入到其他模块中），则调用 asyncio.run() 运行 run_with_config 函数
if __name__ == "__main__":
    asyncio.run(run_with_config())
```