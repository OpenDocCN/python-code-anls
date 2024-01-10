# `MetaGPT\tests\metagpt\utils\test_mermaid.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/27
@Author  : mashenquan
@File    : test_mermaid.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.config 模块中导入 CONFIG 对象
from metagpt.config import CONFIG
# 从 metagpt.utils.common 模块中导入 check_cmd_exists 函数
from metagpt.utils.common import check_cmd_exists
# 从 metagpt.utils.mermaid 模块中导入 MMC1 类和 mermaid_to_file 函数

# 使用 pytest.mark.asyncio 装饰器标记异步测试
@pytest.mark.asyncio
# 使用 pytest.mark.parametrize 装饰器定义参数化测试，参数为 "engine"，值为 ["nodejs", "ink"]
async def test_mermaid(engine):
    # 检查 npm 命令是否存在
    assert check_cmd_exists("npm") == 0
    # 断言 CONFIG.PYPPETEER_EXECUTABLE_PATH 是否存在

    # 将 CONFIG.mermaid_engine 设置为参数化的 engine
    CONFIG.mermaid_engine = engine
    # 将结果保存到指定路径
    save_to = CONFIG.git_repo.workdir / f"{CONFIG.mermaid_engine}/1"
    # 调用 mermaid_to_file 函数，将 MMC1 转换为文件保存到指定路径
    await mermaid_to_file(MMC1, save_to)

    # 如果引擎为 "ink"，则执行以下操作
    if engine == "ink":
        # 遍历文件后缀名列表
        for ext in [".svg", ".png"]:
            # 断言指定后缀名的文件是否存在，存在则删除
            assert save_to.with_suffix(ext).exists()
            save_to.with_suffix(ext).unlink(missing_ok=True)
    # 如果引擎不为 "ink"，则执行以下操作
    else:
        # 遍历文件后缀名列表
        for ext in [".pdf", ".svg", ".png"]:
            # 断言指定后缀名的文件是否存在，存在则删除
            assert save_to.with_suffix(ext).exists()
            save_to.with_suffix(ext).unlink(missing_ok=True)

# 如果当前模块为主模块，则执行测试
if __name__ == "__main__":
    # 运行 pytest 测试，并输出结果
    pytest.main([__file__, "-s"])

```