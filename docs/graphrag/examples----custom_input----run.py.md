# `.\graphrag\examples\custom_input\run.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 引入 asyncio 模块，用于支持异步操作
import asyncio
# 引入 os 模块，用于处理文件路径
import os

# 引入 pandas 库，用于处理数据
import pandas as pd

# 从 graphrag.index 中引入 run_pipeline_with_config 函数
from graphrag.index import run_pipeline_with_config

# 设置 pipeline_file 变量为当前文件路径下的 pipeline.yml 文件的完整路径
pipeline_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./pipeline.yml"
)

# 异步函数定义，用于运行整个流程
async def run():
    # 加载数据集，具体加载方式由 _load_dataset_some_unique_way 函数实现
    dataset = _load_dataset_some_unique_way()

    # 设置 config 变量为 pipeline_file，即 pipeline.yml 的路径
    config = pipeline_file

    # 初始化空列表 outputs，用于存储管道运行的输出结果
    outputs = []
    # 异步迭代 run_pipeline_with_config 函数的输出结果
    async for output in run_pipeline_with_config(
        config_or_path=config, dataset=dataset
    ):
        outputs.append(output)
    # 获取管道运行的最后一个结果，通常是实体抽取的结果
    pipeline_result = outputs[-1]

    # 检查管道运行的最后结果是否为空
    if pipeline_result.result is not None:
        # 输出最后结果，假设格式如下：
        #            col1  col2 filled_column
        # 0     2     4  Filled Value
        # 1     5    10  Filled Value
        print(pipeline_result.result)
    else:
        # 如果结果为空，则输出 "No results!"
        print("No results!")


# 定义函数 _load_dataset_some_unique_way，返回一个包含数据的 pandas DataFrame
def _load_dataset_some_unique_way() -> pd.DataFrame:
    # 这里的数据集加载方式是从其他地方加载
    return pd.DataFrame([{"col1": 2, "col2": 4}, {"col1": 5, "col2": 10}])


# 如果当前脚本作为主程序运行，则调用 asyncio.run() 执行异步函数 run()
if __name__ == "__main__":
    asyncio.run(run())
```