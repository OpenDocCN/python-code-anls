# `.\graphrag\tests\integration\_pipeline\test_run.py`

```py
# 导入所需的模块和库
import logging  # 导入日志记录模块
import os  # 导入操作系统相关的功能
import unittest  # 导入单元测试框架

from graphrag.index.run import run_pipeline_with_config  # 从指定路径导入运行管道配置函数
from graphrag.index.typing import PipelineRunResult  # 从指定路径导入管道运行结果类型

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class TestRun(unittest.IsolatedAsyncioTestCase):
    async def test_megapipeline(self):
        # 构建管道配置文件路径
        pipeline_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "./megapipeline.yml",
        )
        # 运行管道配置，并生成一个异步生成器，获取管道运行结果列表
        pipeline_result = [gen async for gen in run_pipeline_with_config(pipeline_path)]

        errors = []
        # 遍历管道运行结果列表，收集所有错误信息
        for result in pipeline_result:
            if result.errors is not None and len(result.errors) > 0:
                errors.extend(result.errors)

        # 如果有错误信息，则打印错误信息列表
        if len(errors) > 0:
            print("Errors: ", errors)
        # 断言没有错误信息，否则抛出异常并输出所有错误信息
        assert len(errors) == 0, "received errors\n!" + "\n".join(errors)

        # 调用私有方法验证文本单元和实体是否相互引用
        self._assert_text_units_and_entities_reference_each_other(pipeline_result)

    def _assert_text_units_and_entities_reference_each_other(
        self, pipeline_result: list[PipelineRunResult]
    ):
        # 获取 "create_final_text_units" 工作流的结果数据框
        text_unit_df = next(
            filter(lambda x: x.workflow == "create_final_text_units", pipeline_result)
        ).result
        # 获取 "create_final_entities" 工作流的结果数据框
        entity_df = next(
            filter(lambda x: x.workflow == "create_final_entities", pipeline_result)
        ).result

        # 断言文本单元数据框不为空
        assert text_unit_df is not None, "Text unit dataframe should not be None"
        # 断言实体数据框不为空
        assert entity_df is not None, "Entity dataframe should not be None"

        # 处理类型问题
        if text_unit_df is None or entity_df is None:
            return

        # 断言文本单元数据框和实体数据框不为空
        assert len(text_unit_df) > 0, "Text unit dataframe should not be empty"
        assert len(entity_df) > 0, "Entity dataframe should not be empty"

        # 创建文本单元和实体之间的映射关系字典
        text_unit_entity_map = {}
        log.info("text_unit_df %s", text_unit_df.columns)

        # 遍历文本单元数据框的每一行，获取实体 ID 列的值，并映射到文本单元 ID
        for _, row in text_unit_df.iterrows():
            values = row.get("entity_ids", [])
            text_unit_entity_map[row["id"]] = set([] if values is None else values)

        # 创建实体和文本单元之间的映射关系字典
        entity_text_unit_map = {}
        # 遍历实体数据框的每一行，获取文本单元 ID 列的值，并映射到实体 ID
        for _, row in entity_df.iterrows():
            values = row.get("text_unit_ids", [])
            entity_text_unit_map[row["id"]] = set([] if values is None else values)

        # 获取文本单元 ID 和实体 ID 的集合
        text_unit_ids = set(text_unit_entity_map.keys())
        entity_ids = set(entity_text_unit_map.keys())

        # 验证每个文本单元引用的实体 ID 都在实体 ID 集合中存在
        for text_unit_id, text_unit_entities in text_unit_entity_map.items():
            assert text_unit_entities.issubset(
                entity_ids
            ), f"Text unit {text_unit_id} has entities {text_unit_entities} that are not in the entity set"
        
        # 验证每个实体引用的文本单元 ID 都在文本单元 ID 集合中存在
        for entity_id, entity_text_units in entity_text_unit_map.items():
            assert entity_text_units.issubset(
                text_unit_ids
            ), f"Entity {entity_id} has text units {entity_text_units} that are not in the text unit set"
```