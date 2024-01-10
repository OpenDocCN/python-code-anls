# `MetaGPT\tests\metagpt\actions\test_run_code.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:46
@Author  : alexanderwu
@File    : test_run_code.py
@Modifiled By: mashenquan, 2023-12-6. According to RFC 135
"""
# 导入 pytest 模块
import pytest

# 导入 RunCode 类和 RunCodeContext 类
from metagpt.actions.run_code import RunCode
from metagpt.schema import RunCodeContext

# 异步测试函数，测试 run_text 方法
@pytest.mark.asyncio
async def test_run_text():
    # 测试代码 "result = 1 + 1"
    out, err = await RunCode.run_text("result = 1 + 1")
    assert out == 2
    assert err == ""

    # 测试代码 "result = 1 / 0"
    out, err = await RunCode.run_text("result = 1 / 0")
    assert out == ""
    assert "division by zero" in err

# 异步测试函数，测试 run_script 方法
@pytest.mark.asyncio
async def test_run_script():
    # 成功的命令
    out, err = await RunCode.run_script(".", command=["echo", "Hello World"])
    assert out.strip() == "Hello World"
    assert err == ""

    # 失败的命令
    out, err = await RunCode.run_script(".", command=["python", "-c", "print(1/0)"])
    assert "ZeroDivisionError" in err

# 异步测试函数，测试 run 方法
@pytest.mark.asyncio
async def test_run():
    # 输入测试数据
    inputs = [
        (RunCodeContext(mode="text", code_filename="a.txt", code="print('Hello, World')"), "PASS"),
        (
            RunCodeContext(
                mode="script",
                code_filename="a.sh",
                code="echo 'Hello World'",
                command=["echo", "Hello World"],
                working_directory=".",
            ),
            "PASS",
        ),
        (
            RunCodeContext(
                mode="script",
                code_filename="a.py",
                code='python -c "print(1/0)"',
                command=["python", "-c", "print(1/0)"],
                working_directory=".",
            ),
            "FAIL",
        ),
    ]
    # 遍历测试数据
    for ctx, result in inputs:
        # 运行测试
        rsp = await RunCode(context=ctx).run()
        assert result in rsp.summary

```