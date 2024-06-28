# `.\agent\GenAINewsAgent\server\llms\ctx.py`

```
from typing import List, Dict, Literal, Union
from transformers import AutoTokenizer

class ContextManagement:

    def __init__(self):
        # 初始化对象时，使用预训练的AutoTokenizer从'meta-llama/Meta-Llama-3-8B'加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    def __count_tokens__(self, content: str):
        # 使用tokenizer对内容进行分词，并返回分词后的数量加上2（预留特殊标记）
        tokens = self.tokenizer.tokenize(content)
        return len(tokens) + 2

    def __pad_content__(self, content: str, num_tokens: int):
        # 使用tokenizer对内容进行编码，并根据num_tokens进行截断或者填充，然后解码为文本
        return self.tokenizer.decode(self.tokenizer.encode(content, max_length=num_tokens))

    def __call__(self, messages: List[Dict], max_length: int = 28_000):
        managed_messages = []  # 存放处理后的消息列表
        current_length = 0  # 当前已处理的消息长度
        current_message_role = None  # 当前消息的角色

        # 反向遍历消息列表
        for ix, message in enumerate(messages[::-1]):
            content = message.get("content")  # 获取消息内容
            message_tokens = self.__count_tokens__(message.get("content"))  # 获取消息内容的token数量

            if ix > 0:
                # 如果加入当前消息后长度超过最大长度
                if current_length + message_tokens >= max_length:
                    tokens_to_keep = max_length - current_length
                    if tokens_to_keep > 0:
                        # 对消息内容进行截断或填充，使其不超过最大长度
                        content = self.__pad_content__(content, tokens_to_keep)
                        current_length += tokens_to_keep
                    else:
                        break  # 如果无法保留任何token，则跳出循环

                # 如果当前消息的角色与上一条相同，则合并内容
                if message.get("role") == current_message_role:
                    managed_messages[-1]["content"] += f"\n\n{content}"
                else:
                    # 否则将新消息添加到管理消息列表中，并更新当前消息的角色和长度
                    managed_messages.append({
                        "role": message.get("role"),
                        "content": content
                    })
                    current_message_role = message.get("role")
                    current_length += message_tokens
            else:
                # 处理第一条消息的情况
                if current_length + message_tokens >= max_length:
                    tokens_to_keep = max_length - current_length
                    if tokens_to_keep > 0:
                        content = self.__pad_content__(content, tokens_to_keep)
                        current_length += tokens_to_keep
                        managed_messages.append({
                            "role": message.get("role"),
                            "content": content
                        })
                    else:
                        break  # 如果无法保留任何token，则跳出循环
                else:
                    managed_messages.append({
                        "role": message.get("role"),
                        "content": content
                    })
                    current_length += message_tokens
                current_message_role = message.get("role")
        
        # 输出处理后消息的总token数量
        print(f"TOTAL TOKENS: ", current_length)
        
        # 返回处理后的消息列表，反向输出以保持消息顺序
        return managed_messages[::-1]

if __name__ == "__main__":
    import json
    messages = [{
        "role": "user",
        "content": "What is your favourite condiment?"
    }, {
        "role":
        "assistant",
        "content":
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"
    }, {
        "role": "user",
        "content": "Do you have mayonnaise recipes?"
    }, {
        "role": "user",
        "content": "Do you have mayonnaise recipes? - 2"
    }]

这部分是一个包含多个字典的列表，每个字典描述了一个消息的角色和内容。


    ctxmgmt = ContextManagement()
    print(json.dumps(ctxmgmt(messages, 45), indent=4))

创建一个 `ContextManagement` 的实例 `ctxmgmt`，然后调用其方法 `ctxmgmt(messages, 45)`，并将返回的结果使用 `json.dumps` 格式化为 JSON 格式并打印出来，缩进为4个空格。
```