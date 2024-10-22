# `.\chatglm4-finetune\finetune_demo\inference.py`

```py
# 从 pathlib 库导入 Path 类，用于路径操作
from pathlib import Path
# 导入用于类型注解的 Annotated 和 Union
from typing import Annotated, Union
# 导入 typer 库，用于创建命令行界面
import typer
# 从 peft 库导入 PeftModelForCausalLM 模型
from peft import PeftModelForCausalLM
# 从 transformers 库导入自动模型和自动标记器
from transformers import (
    AutoModel,
    AutoTokenizer,
)
# 从 PIL 库导入 Image，用于图像处理
from PIL import Image
# 导入 PyTorch 库
import torch

# 创建一个 typer 应用，设置不显示局部变量的异常信息
app = typer.Typer(pretty_exceptions_show_locals=False)


# 定义加载模型和标记器的函数，接收模型目录和信任远程代码的标志
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
):
    # 解析并规范化模型目录路径
    model_dir = Path(model_dir).expanduser().resolve()
    # 检查 adapter_config.json 是否存在于模型目录
    if (model_dir / 'adapter_config.json').exists():
        # 导入 JSON 库用于解析配置文件
        import json
        # 打开并读取 adapter_config.json 配置文件
        with open(model_dir / 'adapter_config.json', 'r', encoding='utf-8') as file:
            config = json.load(file)
        # 根据配置文件加载基础模型
        model = AutoModel.from_pretrained(
            config.get('base_model_name_or_path'),
            trust_remote_code=trust_remote_code,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        # 从预训练模型加载 Peft 模型
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
            trust_remote_code=trust_remote_code,
        )
        # 获取标记器目录
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        # 如果没有 adapter_config.json，直接根据模型目录加载基础模型
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        # 设置标记器目录为模型目录
        tokenizer_dir = model_dir
    # 从预训练目录加载标记器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=trust_remote_code,
        encode_special_tokens=True,
        use_fast=False
    )
    # 返回加载的模型和标记器
    return model, tokenizer


# 定义主命令函数，接收模型目录参数
@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
):
    # 为 GLM-4 进行无工具微调的消息示例
    messages = [
        {
            "role": "user", "content": "#裙子#夏天",
        }
    ]

    # 为 GLM-4 进行有工具微调的消息示例
    # messages = [
    #     {
    #         "role": "system", "content": "",
    #         "tools":
    #             [
    #                 {
    #                     "type": "function",
    #                     "function": {
    #                         "name": "create_calendar_event",
    #                         "description": "Create a new calendar event",
    #                         "parameters": {
    #                             "type": "object",
    #                             "properties": {
    #                                 "title": {
    #                                     "type": "string",
    #                                     "description": "The title of the event"
    #                                 },
    #                                 "start_time": {
    #                                     "type": "string",
    #                                     "description": "The start time of the event in the format YYYY-MM-DD HH:MM"
    #                                 },
    #                                 "end_time": {
    #                                     "type": "string",
    #                                     "description": "事件结束时间，格式为 YYYY-MM-DD HH:MM"
    #                                 }
    #                             },
    #                             "required": [
    #                                 "title",  # 事件的标题是必填项
    #                                 "start_time",  # 事件的开始时间是必填项
    #                                 "end_time"  # 事件的结束时间是必填项
    #                             ]
    #                         }
    #                     }
    #                 }
    #             ]
    #
    #     },
    #     {
    #         "role": "user",  # 消息的角色为用户
    #         "content": "能帮我创建一个明天会议的日历事件吗？标题是\"团队会议\"，开始时间是上午10:00，结束时间是上午11:00。"  # 用户请求创建日历事件的内容
    #     },
    # ]
    
    # 为 GLM-4V 微调准备消息
    # messages = [
    #     {
    #         "role": "user",  # 消息的角色为用户
    #         "content": "女孩可能希望观众做什么？",  # 用户的问题内容
    #         "image": Image.open("your Image").convert("RGB")  # 打开图像文件并转换为 RGB 格式
    #     }
    # ]

    model, tokenizer = load_model_and_tokenizer(model_dir)  # 加载模型和分词器
    inputs = tokenizer.apply_chat_template(  # 应用聊天模板格式化输入消息
        messages,  # 传入的消息列表
        add_generation_prompt=True,  # 添加生成提示
        tokenize=True,  # 对输入进行分词
        return_tensors="pt",  # 返回 PyTorch 张量
        return_dict=True  # 返回字典格式
    ).to(model.device)  # 将输入张量转移到模型的设备上
    generate_kwargs = {  # 定义生成时的参数
        "max_new_tokens": 1024,  # 生成的最大新标记数
        "do_sample": True,  # 允许随机采样
        "top_p": 0.8,  # 采样时的累积概率阈值
        "temperature": 0.8,  # 控制生成文本的随机性
        "repetition_penalty": 1.2,  # 重复惩罚因子
        "eos_token_id": model.config.eos_token_id,  # 结束标记的 ID
    }
    outputs = model.generate(**inputs, **generate_kwargs)  # 生成模型的输出
    response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True).strip()  # 解码生成的输出并去除特殊标记
    print("=========")  # 打印分隔符
    print(response)  # 输出生成的响应
# 如果当前脚本是主程序，则执行下面的代码
if __name__ == '__main__':
    # 调用应用程序的主函数
    app()
```