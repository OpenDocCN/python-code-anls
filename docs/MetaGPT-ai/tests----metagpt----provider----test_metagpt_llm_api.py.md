# `MetaGPT\tests\metagpt\provider\test_metagpt_llm_api.py`

```

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定编码格式为 UTF-8
"""
@Time    : 2023/8/30
@Author  : mashenquan
@File    : test_metagpt_llm_api.py
"""
# 文件的时间、作者和文件名信息
from metagpt.provider.metagpt_api import MetaGPTLLM
# 导入 MetaGPTLLM 类

def test_metagpt():
    # 创建 MetaGPTLLM 实例
    llm = MetaGPTLLM()
    # 断言 llm 存在
    assert llm

if __name__ == "__main__":
    # 如果是主程序，则运行 test_metagpt 函数
    test_metagpt()

```