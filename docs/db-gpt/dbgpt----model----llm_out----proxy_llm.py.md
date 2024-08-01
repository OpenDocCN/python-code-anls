# `.\DB-GPT-src\dbgpt\model\llm_out\proxy_llm.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

# 导入各个模型的数据流生成函数
from dbgpt.model.proxy.llms.baichuan import baichuan_generate_stream
from dbgpt.model.proxy.llms.bard import bard_generate_stream
from dbgpt.model.proxy.llms.chatgpt import chatgpt_generate_stream
from dbgpt.model.proxy.llms.claude import claude_generate_stream
from dbgpt.model.proxy.llms.gemini import gemini_generate_stream
from dbgpt.model.proxy.llms.proxy_model import ProxyModel
from dbgpt.model.proxy.llms.spark import spark_generate_stream
from dbgpt.model.proxy.llms.tongyi import tongyi_generate_stream
from dbgpt.model.proxy.llms.wenxin import wenxin_generate_stream
from dbgpt.model.proxy.llms.zhipu import zhipu_generate_stream

# This has been moved to dbgpt/model/adapter/proxy_adapter.py
# def proxyllm_generate_stream(
#     model: ProxyModel, tokenizer, params, device, context_len=2048
# ):
#     generator_mapping = {
#         "proxyllm": chatgpt_generate_stream,  # 映射 "proxyllm" 到 chatgpt_generate_stream 函数
#         "chatgpt_proxyllm": chatgpt_generate_stream,  # 映射 "chatgpt_proxyllm" 到 chatgpt_generate_stream 函数
#         "bard_proxyllm": bard_generate_stream,  # 映射 "bard_proxyllm" 到 bard_generate_stream 函数
#         "claude_proxyllm": claude_generate_stream,  # 映射 "claude_proxyllm" 到 claude_generate_stream 函数
#         # "gpt4_proxyllm": gpt4_generate_stream, move to chatgpt_generate_stream
#         "wenxin_proxyllm": wenxin_generate_stream,  # 映射 "wenxin_proxyllm" 到 wenxin_generate_stream 函数
#         "tongyi_proxyllm": tongyi_generate_stream,  # 映射 "tongyi_proxyllm" 到 tongyi_generate_stream 函数
#         "zhipu_proxyllm": zhipu_generate_stream,  # 映射 "zhipu_proxyllm" 到 zhipu_generate_stream 函数
#         "gemini_proxyllm": gemini_generate_stream,  # 映射 "gemini_proxyllm" 到 gemini_generate_stream 函数
#         "bc_proxyllm": baichuan_generate_stream,  # 映射 "bc_proxyllm" 到 baichuan_generate_stream 函数
#         "spark_proxyllm": spark_generate_stream,  # 映射 "spark_proxyllm" 到 spark_generate_stream 函数
#     }
#     获取模型的参数
#     model_params = model.get_params()
#     获取模型的名称
#     model_name = model_params.model_name
#     定义默认错误信息
#     default_error_message = f"{model_name} LLM is not supported"
#     根据模型名称获取相应的生成函数，如果未定义则返回默认错误信息函数
#     generator_function = generator_mapping.get(
#         model_name, lambda *args: [default_error_message]
#     )
#
#     使用生成函数生成数据流
#     yield from generator_function(model, tokenizer, params, device, context_len)
```