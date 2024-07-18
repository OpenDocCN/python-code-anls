# `.\graphrag\examples\various_levels_of_configs\workflows_and_inputs.py`

```py
# 导入 asyncio 模块，用于支持异步操作
import asyncio
# 导入 os 模块，用于访问操作系统功能
import os

# 从 graphrag.index 模块中导入 run_pipeline_with_config 函数
from graphrag.index import run_pipeline_with_config


# 异步函数 main，程序的入口点
async def main():
    # 检查环境变量中是否包含必要的 API 密钥
    if (
        "EXAMPLE_OPENAI_API_KEY" not in os.environ
        and "OPENAI_API_KEY" not in os.environ
    ):
        # 如果缺少 API 密钥，则抛出异常并显示相关提示信息
        msg = "Please set EXAMPLE_OPENAI_API_KEY or OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)

    # 构建 pipeline_path 变量，指定 pipeline 配置文件的路径
    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./pipelines/workflows_and_inputs.yml",
    )

    # 创建空列表 tables 用于存储从 pipeline 返回的表格数据
    tables = []
    # 使用 run_pipeline_with_config 函数运行 pipeline，并异步迭代返回的每个表格
    async for table in run_pipeline_with_config(pipeline_path):
        tables.append(table)
    # 获取 pipeline 返回的最后一个表格，假定这是最后一个运行的工作流的结果（我们的节点）
    pipeline_result = tables[-1]

    # 如果 pipeline_result 不为空，则获取结果的前10个节点并打印
    if pipeline_result.result is not None:
        top_nodes = pipeline_result.result.head(10)
        print("pipeline result", top_nodes)
    else:
        # 如果 pipeline_result 为空，则打印无结果的消息
        print("No results!")


# 如果脚本作为主程序运行，则调用 asyncio.run() 执行主函数 main()
if __name__ == "__main__":
    asyncio.run(main())
```