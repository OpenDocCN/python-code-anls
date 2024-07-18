# `.\graphrag\tests\unit\indexing\config\helpers.py`

```py
# 导入所需的模块和类
import json
import unittest
from typing import Any

# 从外部库导入函数和类
from graphrag.config import create_graphrag_config
from graphrag.index import PipelineConfig, create_pipeline_config


def assert_contains_default_config(
    test_case: unittest.TestCase,
    config: Any,
    check_input=True,
    check_storage=True,
    check_reporting=True,
    check_cache=True,
    check_workflows=True,
):
    """Asserts that the config contains the default config."""
    # 断言配置对象不为空，并且是 PipelineConfig 类型的实例
    assert config is not None
    assert isinstance(config, PipelineConfig)

    # 将配置对象转换为不包含默认值和未设置项的 JSON 字典
    checked_config = json.loads(
        config.model_dump_json(exclude_defaults=True, exclude_unset=True)
    )

    # 创建默认配置对象，并转换为不包含默认值和未设置项的 JSON 字典
    actual_default_config = json.loads(
        create_pipeline_config(create_graphrag_config()).model_dump_json(
            exclude_defaults=True, exclude_unset=True
        )
    )

    # 需要忽略的属性列表
    props_to_ignore = ["root_dir", "extends"]

    # 如果不需要检查工作流程，则忽略 "workflows" 属性
    if not check_workflows:
        props_to_ignore.append("workflows")

    # 如果不需要检查输入，则忽略 "input" 属性
    if not check_input:
        props_to_ignore.append("input")

    # 如果不需要检查存储，则忽略 "storage" 属性
    if not check_storage:
        props_to_ignore.append("storage")

    # 如果不需要检查报告，则忽略 "reporting" 属性
    if not check_reporting:
        props_to_ignore.append("reporting")

    # 如果不需要检查缓存，则忽略 "cache" 属性
    if not check_cache:
        props_to_ignore.append("cache")

    # 遍历需要忽略的属性列表，从两个配置字典中删除对应的属性项
    for prop in props_to_ignore:
        checked_config.pop(prop, None)
        actual_default_config.pop(prop, None)

    # 断言实际的默认配置和检查后的配置字典相等
    assert actual_default_config == checked_config
```