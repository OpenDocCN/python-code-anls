# `MetaGPT\tests\metagpt\test_startup.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/15 11:40
@Author  : alexanderwu
@File    : test_startup.py
"""
# 导入 pytest 模块
import pytest
# 从 typer.testing 模块中导入 CliRunner 类
from typer.testing import CliRunner
# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger
# 从 metagpt.startup 模块中导入 app 对象
from metagpt.startup import app
# 从 metagpt.team 模块中导入 Team 类
from metagpt.team import Team

# 创建一个 CliRunner 实例
runner = CliRunner()

# 标记该测试函数为异步函数
@pytest.mark.asyncio
async def test_empty_team(new_filename):
    # FIXME: we're now using "metagpt" cli, so the entrance should be replaced instead.
    # 创建一个 Team 实例
    company = Team()
    # 调用 run 方法，传入参数 idea，获取返回结果并记录到 history 中
    history = await company.run(idea="Build a simple search system. I will upload my files later.")
    # 记录 history 到日志中
    logger.info(history)

# 定义一个测试函数 test_startup，传入参数 new_filename
def test_startup(new_filename):
    # 定义参数 args 为 ["Make a cli snake game"]
    args = ["Make a cli snake game"]
    # 调用 app 对象的 invoke 方法，传入参数 args，获取返回结果并记录到 result 中
    result = runner.invoke(app, args)
    # 记录 result 到日志中
    logger.info(result)
    # 记录 result 的输出到日志中
    logger.info(result.output)

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行 pytest 测试，传入参数 [__file__, "-s"]
    pytest.main([__file__, "-s"])

```