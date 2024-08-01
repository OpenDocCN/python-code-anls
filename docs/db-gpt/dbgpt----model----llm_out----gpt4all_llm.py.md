# `.\DB-GPT-src\dbgpt\model\llm_out\gpt4all_llm.py`

```py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 定义一个生成器函数，用于生成文本流
def gpt4all_generate_stream(model, tokenizer, params, device, max_position_embeddings):
    # 从参数中获取停止符，默认为"###"
    stop = params.get("stop", "###")
    # 从参数中获取提示文本
    prompt = params["prompt"]
    # 从提示文本中提取角色和查询内容，使用停止符分割后再使用冒号分割
    role, query = prompt.split(stop)[0].split(":")
    # 打印角色和查询内容
    print(f"gpt4all, role: {role}, query: {query}")
    # 生成文本，使用模型生成给定查询内容的文本，启用流式生成
    yield model.generate(prompt=query, streaming=True)
```