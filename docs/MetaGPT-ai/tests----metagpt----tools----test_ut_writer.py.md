# `MetaGPT\tests\metagpt\tools\test_ut_writer.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/4/30 21:44
@Author  : alexanderwu
@File    : test_ut_writer.py
"""
# 导入模块
from pathlib import Path
import pytest
# 导入自定义模块
from metagpt.config import CONFIG
from metagpt.const import API_QUESTIONS_PATH, UT_PY_PATH
from metagpt.tools.ut_writer import YFT_PROMPT_PREFIX, UTGenerator

# 定义测试类
class TestUTWriter:
    # 异步测试方法
    @pytest.mark.asyncio
    async def test_api_to_ut_sample(self):
        # Prerequisites
        # 获取swagger文件路径
        swagger_file = Path(__file__).parent / "../../data/ut_writer/yft_swaggerApi.json"
        # 断言swagger文件存在
        assert swagger_file.exists()
        # 断言配置中存在OPENAI_API_KEY且不为"YOUR_API_KEY"
        assert CONFIG.OPENAI_API_KEY and CONFIG.OPENAI_API_KEY != "YOUR_API_KEY"
        # 断言配置中不存在OPENAI_API_TYPE
        assert not CONFIG.OPENAI_API_TYPE
        # 断言配置中存在OPENAI_API_MODEL
        assert CONFIG.OPENAI_API_MODEL

        # 定义测试标签
        tags = ["测试", "作业"]
        # 创建UTGenerator对象
        utg = UTGenerator(
            swagger_file=str(swagger_file),
            ut_py_path=UT_PY_PATH,
            questions_path=API_QUESTIONS_PATH,
            template_prefix=YFT_PROMPT_PREFIX,
        )
        # 生成UT
        ret = await utg.generate_ut(include_tags=tags)
        # 断言生成结果
        assert ret

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```