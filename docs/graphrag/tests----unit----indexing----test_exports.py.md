# `.\graphrag\tests\unit\indexing\test_exports.py`

```py
# 导入语句，从 graphrag.index 模块导入以下函数：
# - create_pipeline_config
# - run_pipeline
# - run_pipeline_with_config
from graphrag.index import (
    create_pipeline_config,
    run_pipeline,
    run_pipeline_with_config,
)


# 定义测试函数 test_exported_functions
def test_exported_functions():
    # 断言 create_pipeline_config 是可调用的函数
    assert callable(create_pipeline_config)
    # 断言 run_pipeline_with_config 是可调用的函数
    assert callable(run_pipeline_with_config)
    # 断言 run_pipeline 是可调用的函数
    assert callable(run_pipeline)
```