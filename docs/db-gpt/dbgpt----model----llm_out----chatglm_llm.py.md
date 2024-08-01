# `.\DB-GPT-src\dbgpt\model\llm_out\chatglm_llm.py`

```py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
from typing import List

import torch

from dbgpt.app.scene import ModelMessage, _parse_model_messages

# 用于分隔消息的常量定义
_CHATGLM_SEP = "\n"
_CHATGLM2_SEP = "\n\n"


@torch.inference_mode()
def chatglm_generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    """Generate text using chatglm model's chat api_v1"""
    # 从参数中获取生成文本所需的提示信息
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    stop = params.get("stop", "###")
    echo = params.get("echo", False)

    generate_kwargs = {
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": 1.0,
        "logits_processor": None,
    }

    if temperature > 1e-5:
        generate_kwargs["temperature"] = temperature

    # TODO, Fix this
    # 打印生成文本的提示信息
    # print(prompt)
    # 按照指定分隔符拆分提示信息，生成消息列表
    messages: List[ModelMessage] = params["messages"]
    query, system_messages, hist = _parse_model_messages(messages)
    system_messages_str = "".join(system_messages)
    if not hist:
        # 若无历史对话记录，但有系统消息，则将系统消息合并到用户查询中
        query = prompt_adaptation(system_messages_str, query)
    else:
        # 若存在历史记录，则将系统消息添加到历史记录的开头
        hist[0][0] = system_messages_str + _CHATGLM2_SEP + hist[0][0]

    # 打印查询消息和历史记录
    print("Query Message: ", query)
    print("hist: ", hist)

    # 使用模型流式生成聊天文本
    for i, (response, new_hist) in enumerate(
        model.stream_chat(tokenizer, query, hist, **generate_kwargs)
    ):
        if echo:
            output = query + " " + response
        else:
            output = response

        yield output

    yield output


class HistoryEntry:
    def __init__(self, question: str = "", answer: str = ""):
        self.question = question
        self.answer = answer

    def add_question(self, question: str):
        self.question += question

    def add_answer(self, answer: str):
        self.answer += answer

    def to_list(self):
        # 将问题和答案转换为列表形式，如果问题或答案为空则返回None
        if self.question == "" or self.answer == "":
            return None
        return [self.question, self.answer]


def build_history(hist: List[HistoryEntry]) -> List[List[str]]:
    # 构建历史记录的列表，过滤掉空记录
    return list(filter(lambda hl: hl is not None, map(lambda h: h.to_list(), hist)))


def prompt_adaptation(system_messages_str: str, human_message: str) -> str:
    if not system_messages_str or system_messages_str == "":
        return human_message
    # TODO Multi-model prompt adaptation
    # 根据系统消息调整用户消息的适应性
    adaptation_rules = [
        r"Question:\s*{}\s*",  # chat_db scene
        r"Goals:\s*{}\s*",  # chat_execution
        r"问题:\s*{}\s*",  # chat_knowledge zh
        r"question:\s*{}\s*",  # chat_knowledge en
    ]
    # 如果系统消息已包含用户问题，则直接返回用户消息
    # 遍历适应规则列表中的每个规则
    for rule in adaptation_rules:
        # 使用人类消息进行格式化，并转义特殊字符，创建正则表达式模式
        pattern = re.compile(rule.format(re.escape(human_message)))
        # 在系统消息字符串中搜索匹配该模式的文本
        if re.search(pattern, system_messages_str):
            # 如果找到匹配的文本，则返回系统消息字符串本身
            return system_messages_str
    # 若未找到任何匹配的规则，则返回系统消息字符串与特定分隔符和人类消息的组合
    # 参考链接：https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
    return system_messages_str + _CHATGLM2_SEP + human_message
```