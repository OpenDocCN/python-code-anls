# `.\graphrag\examples\various_levels_of_configs\workflows_and_inputs_with_custom_handlers.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
import asyncio  # 引入异步I/O支持
import os  # 引入操作系统接口
from typing import Any  # 引入类型提示

from datashaper import NoopWorkflowCallbacks, Progress  # 从datashaper模块导入NoopWorkflowCallbacks和Progress类

from graphrag.index import run_pipeline_with_config  # 从graphrag.index模块导入run_pipeline_with_config函数
from graphrag.index.cache import InMemoryCache, PipelineCache  # 从graphrag.index.cache模块导入InMemoryCache和PipelineCache类
from graphrag.index.storage import MemoryPipelineStorage  # 从graphrag.index.storage模块导入MemoryPipelineStorage类


async def main():
    if (
        "EXAMPLE_OPENAI_API_KEY" not in os.environ  # 检查环境变量是否存在EXAMPLE_OPENAI_API_KEY
        and "OPENAI_API_KEY" not in os.environ  # 检查环境变量是否存在OPENAI_API_KEY
    ):
        msg = "Please set EXAMPLE_OPENAI_API_KEY or OPENAI_API_KEY environment variable to run this example"
        raise Exception(msg)  # 如果环境变量不存在，则抛出异常

    # run the pipeline with the config, and override the dataset with the one we just created
    # and grab the last result from the pipeline, should be our entity extraction
    pipeline_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),  # 获取当前脚本文件所在目录的绝对路径
        "./pipelines/workflows_and_inputs.yml",  # 构建流水线配置文件的路径
    )

    # Create our custom storage
    custom_storage = ExampleStorage()  # 创建自定义存储实例

    # Create our custom reporter
    custom_reporter = ExampleReporter()  # 创建自定义报告实例

    # Create our custom cache
    custom_cache = ExampleCache()  # 创建自定义缓存实例

    # run the pipeline with the config, and override the dataset with the one we just created
    # and grab the last result from the pipeline, should be the last workflow that was run (our nodes)
    pipeline_result = []
    async for result in run_pipeline_with_config(
        pipeline_path,
        storage=custom_storage,
        callbacks=custom_reporter,
        cache=custom_cache,
    ):
        pipeline_result.append(result)  # 将每次流水线运行的结果添加到列表中
    pipeline_result = pipeline_result[-1]  # 获取最后一个流水线运行的结果

    # The output will contain a list of positioned nodes
    if pipeline_result.result is not None:
        top_nodes = pipeline_result.result.head(10)  # 获取结果中的前10个节点
        print("pipeline result", top_nodes)  # 打印流水线结果的前10个节点
    else:
        print("No results!")  # 如果没有结果，则打印无结果


class ExampleStorage(MemoryPipelineStorage):
    """Example of a custom storage handler"""

    async def get(
        self, key: str, as_bytes: bool | None = None, encoding: str | None = None
    ) -> Any:
        print(f"ExampleStorage.get {key}")  # 打印获取操作的键值
        return await super().get(key, as_bytes)  # 调用父类的get方法进行异步获取

    async def set(
        self, key: str, value: str | bytes | None, encoding: str | None = None
    ) -> None:
        print(f"ExampleStorage.set {key}")  # 打印设置操作的键值
        return await super().set(key, value)  # 调用父类的set方法进行异步设置

    async def has(self, key: str) -> bool:
        print(f"ExampleStorage.has {key}")  # 打印检查键值是否存在的操作
        return await super().has(key)  # 调用父类的has方法进行异步检查

    async def delete(self, key: str) -> None:
        print(f"ExampleStorage.delete {key}")  # 打印删除操作的键值
        return await super().delete(key)  # 调用父类的delete方法进行异步删除

    async def clear(self) -> None:
        print("ExampleStorage.clear")  # 打印清空操作
        return await super().clear()  # 调用父类的clear方法进行异步清空


class ExampleCache(InMemoryCache):
    """Example of a custom cache handler"""

    async def get(self, key: str) -> Any:
        print(f"ExampleCache.get {key}")  # 打印获取操作的键值
        return await super().get(key)  # 调用父类的get方法进行异步获取
    # 异步方法：设置缓存项，打印设置的键名，然后调用父类的设置方法
    async def set(self, key: str, value: Any, debug_data: dict | None = None) -> None:
        print(f"ExampleCache.set {key}")
        return await super().set(key, value, debug_data)

    # 异步方法：检查缓存是否存在指定键，打印检查的键名，然后调用父类的检查方法
    async def has(self, key: str) -> bool:
        print(f"ExampleCache.has {key}")
        return await super().has(key)

    # 异步方法：删除缓存中指定键的项，打印删除的键名，然后调用父类的删除方法
    async def delete(self, key: str) -> None:
        print(f"ExampleCache.delete {key}")
        return await super().delete(key)

    # 异步方法：清空缓存，打印清空操作信息，然后调用父类的清空方法
    async def clear(self) -> None:
        print("ExampleCache.clear")
        return await super().clear()

    # 实例方法：创建并返回一个新的 PipelineCache 对象，打印创建的子缓存的名称
    def child(self, name: str) -> PipelineCache:
        print(f"ExampleCache.child {name}")
        return ExampleCache(name)
class ExampleReporter(NoopWorkflowCallbacks):
    """自定义报告器的示例。这将打印出流水线中的所有状态更新。"""

    # 打印进度更新信息
    def progress(self, progress: Progress):
        print("ExampleReporter.progress: ", progress)

    # 打印错误消息和相关细节（如果提供）
    def error(self, message: str, details: dict[str, Any] | None = None):
        print("ExampleReporter.error: ", message)

    # 打印警告消息和相关细节（如果提供）
    def warning(self, message: str, details: dict[str, Any] | None = None):
        print("ExampleReporter.warning: ", message)

    # 打印一般日志消息和相关细节（如果提供）
    def log(self, message: str, details: dict[str, Any] | None = None):
        print("ExampleReporter.log: ", message)


if __name__ == "__main__":
    # 运行主程序的入口点
    asyncio.run(main())
```