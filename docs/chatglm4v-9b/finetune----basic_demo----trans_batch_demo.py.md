# `.\chatglm4-finetune\basic_demo\trans_batch_demo.py`

```
"""
# 示例文档，说明如何使用批量请求 glm-4-9b，
# 这里需要自己构建对话格式，然后调用批量函数以进行批量请求。
# 请注意，此演示中内存消耗显著增加。
"""

# 导入所需的类型和库
from typing import Optional, Union
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList

# 定义模型的路径
MODEL_PATH = 'THUDM/glm-4-9b-chat'

# 从预训练模型加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,  # 信任远程代码
    encode_special_tokens=True  # 编码特殊标记
)

# 从预训练模型加载模型，并设置为评估模式
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

# 定义处理模型输出的函数
def process_model_outputs(inputs, outputs, tokenizer):
    responses = []  # 存储响应列表
    # 遍历输入 ID 和输出 ID
    for input_ids, output_ids in zip(inputs.input_ids, outputs):
        # 解码输出 ID 为响应文本，去掉特殊标记
        response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
        responses.append(response)  # 添加响应到列表
    return responses  # 返回所有响应

# 定义批量处理函数
def batch(
        model,
        tokenizer,
        messages: Union[str, list[str]],  # 输入消息可以是字符串或字符串列表
        max_input_tokens: int = 8192,  # 最大输入标记数
        max_new_tokens: int = 8192,  # 最大生成的新标记数
        num_beams: int = 1,  # 光束搜索的数量
        do_sample: bool = True,  # 是否进行采样
        top_p: float = 0.8,  # Top-p 采样的阈值
        temperature: float = 0.8,  # 温度控制生成的多样性
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),  # 日志处理器
):
    # 将字符串消息转换为列表格式
    messages = [messages] if isinstance(messages, str) else messages
    # 使用分词器编码消息，并返回张量，填充到最大长度
    batched_inputs = tokenizer(messages, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=max_input_tokens).to(model.device)

    # 定义生成的参数
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        "eos_token_id": model.config.eos_token_id  # 获取模型的结束标记 ID
    }
    # 生成模型输出
    batched_outputs = model.generate(**batched_inputs, **gen_kwargs)
    # 处理模型输出以获取响应
    batched_response = process_model_outputs(batched_inputs, batched_outputs, tokenizer)
    return batched_response  # 返回批量响应

# 主程序入口
if __name__ == "__main__":

    # 定义批量消息的示例
    batch_message = [
        [
            {"role": "user", "content": "我的爸爸和妈妈结婚为什么不能带我去"},  # 用户提问
            {"role": "assistant", "content": "因为他们结婚时你还没有出生"},  # 助手回答
            {"role": "user", "content": "我刚才的提问是"}  # 用户提问
        ],
        [
            {"role": "user", "content": "你好，你是谁"}  # 用户提问
        ]
    ]

    batch_inputs = []  # 存储批量输入
    max_input_tokens = 1024  # 初始化最大输入标记数
    # 遍历批量消息，构建输入
    for i, messages in enumerate(batch_message):
        # 使用分词器应用聊天模板，添加生成提示
        new_batch_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # 更新最大输入标记数
        max_input_tokens = max(max_input_tokens, len(new_batch_input))
        batch_inputs.append(new_batch_input)  # 添加到批量输入列表
    # 定义生成的参数
    gen_kwargs = {
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": 8192,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "num_beams": 1,
    }

    # 调用批量处理函数，生成响应
    batch_responses = batch(model, tokenizer, batch_inputs, **gen_kwargs)
    # 遍历批量响应列表中的每个响应
        for response in batch_responses:
            # 打印十个等号，作为分隔符
            print("=" * 10)
            # 打印当前响应的内容
            print(response)
```