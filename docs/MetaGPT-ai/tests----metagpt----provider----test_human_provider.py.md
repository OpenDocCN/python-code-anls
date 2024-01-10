# `MetaGPT\tests\metagpt\provider\test_human_provider.py`

```

#!/usr/bin/env python
# 指定解释器为 python
# -*- coding: utf-8 -*-
# 指定编码格式为 UTF-8
# @Desc   : the unittest of HumanProvider
# 描述：HumanProvider 的单元测试

import pytest
# 导入 pytest 模块

from metagpt.provider.human_provider import HumanProvider
# 从 metagpt.provider.human_provider 模块中导入 HumanProvider 类

resp_content = "test"
# 定义变量 resp_content，赋值为 "test"
resp_exit = "exit"
# 定义变量 resp_exit，赋值为 "exit"

@pytest.mark.asyncio
# 使用 pytest.mark.asyncio 装饰器标记为异步测试
async def test_async_human_provider(mocker):
    # 定义异步测试函数 test_async_human_provider，接受 mocker 参数
    mocker.patch("builtins.input", lambda _: resp_content)
    # 使用 mocker.patch 方法模拟内置函数 input，使其返回 resp_content
    human_provider = HumanProvider()
    # 创建 HumanProvider 实例对象

    resp = human_provider.ask(resp_content)
    # 调用 ask 方法，传入 resp_content 参数，将结果赋值给 resp
    assert resp == resp_content
    # 断言 resp 等于 resp_content
    resp = await human_provider.aask(None)
    # 调用 aask 方法，传入 None 参数，将结果赋值给 resp
    assert resp_content == resp
    # 断言 resp_content 等于 resp

    mocker.patch("builtins.input", lambda _: resp_exit)
    # 使用 mocker.patch 方法模拟内置函数 input，使其返回 resp_exit
    with pytest.raises(SystemExit):
        # 使用 pytest.raises 断言捕获 SystemExit 异常
        human_provider.ask(resp_exit)

    resp = await human_provider.acompletion([])
    # 调用 acompletion 方法，传入空列表参数，将结果赋值给 resp
    assert not resp
    # 断言 resp 为假值

    resp = await human_provider.acompletion_text([])
    # 调用 acompletion_text 方法，传入空列表参数，将结果赋值给 resp
    assert resp == ""
    # 断言 resp 等于空字符串

```