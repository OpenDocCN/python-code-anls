# `.\chatglm4-finetune\basic_demo\vllm_cli_demo.py`

```
"""
这个脚本创建了一个命令行界面（CLI）示例，使用 vllm 后端和 glm-4-9b 模型，
允许用户通过命令行接口与模型进行交互。

用法：
- 运行脚本以启动 CLI 演示。
- 通过输入问题与模型进行交互，并接收回答。

注意：该脚本包含一个修改，以处理 Markdown 到纯文本的转换，
确保 CLI 界面正确显示格式化文本。
"""
# 导入时间模块
import time
# 导入异步编程模块
import asyncio
# 从 transformers 库导入 AutoTokenizer
from transformers import AutoTokenizer
# 从 vllm 库导入相关的采样参数和引擎
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine
# 导入类型注解
from typing import List, Dict
# 从 vllm.lora.request 导入 LoRARequest
from vllm.lora.request import LoRARequest

# 定义模型路径
MODEL_PATH = 'THUDM/glm-4-9b-chat'
# 初始化 LoRA 路径为空
LORA_PATH = ''

# 定义加载模型和分词器的函数
def load_model_and_tokenizer(model_dir: str, enable_lora: bool):
    # 创建异步引擎参数的实例
    engine_args = AsyncEngineArgs(
        model=model_dir,  # 设置模型路径
        tokenizer=model_dir,  # 设置分词器路径
        enable_lora=enable_lora,  # 是否启用 LoRA
        tensor_parallel_size=1,  # 设置张量并行大小
        dtype="bfloat16",  # 设置数据类型
        trust_remote_code=True,  # 允许远程代码
        gpu_memory_utilization=0.9,  # GPU 内存利用率
        enforce_eager=True,  # 强制使用急切执行
        worker_use_ray=True,  # 使用 Ray 来处理工作
        disable_log_requests=True  # 禁用日志请求
        # 如果遇见 OOM 现象，建议开启下述参数
        # enable_chunked_prefill=True,
        # max_num_batched_tokens=8192
    )
    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,  # 模型路径
        trust_remote_code=True,  # 允许远程代码
        encode_special_tokens=True  # 编码特殊符号
    )
    # 从引擎参数创建异步 LLM 引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    # 返回引擎和分词器
    return engine, tokenizer

# 初始化 LoRA 启用标志为 False
enable_lora = False
# 如果有 LoRA 路径，则启用 LoRA
if LORA_PATH:
    enable_lora = True

# 加载模型和分词器
engine, tokenizer = load_model_and_tokenizer(MODEL_PATH, enable_lora)

# 定义异步生成函数
async def vllm_gen(lora_path: str, enable_lora: bool, messages: List[Dict[str, str]], top_p: float, temperature: float, max_dec_len: int):
    # 应用聊天模板处理输入消息
    inputs = tokenizer.apply_chat_template(
        messages,  # 输入的消息
        add_generation_prompt=True,  # 添加生成提示
        tokenize=False  # 不进行标记化
    )
    # 定义采样参数的字典
    params_dict = {
        "n": 1,  # 生成的响应数量
        "best_of": 1,  # 从中选择最佳响应
        "presence_penalty": 1.0,  # 存在惩罚
        "frequency_penalty": 0.0,  # 频率惩罚
        "temperature": temperature,  # 温度参数
        "top_p": top_p,  # 样本的累积概率阈值
        "top_k": -1,  # 前 K 个采样
        "use_beam_search": False,  # 不使用束搜索
        "length_penalty": 1,  # 长度惩罚
        "early_stopping": False,  # 不提前停止
        "ignore_eos": False,  # 不忽略结束符
        "max_tokens": max_dec_len,  # 最大生成长度
        "logprobs": None,  # 日志概率
        "prompt_logprobs": None,  # 提示日志概率
        "skip_special_tokens": True,  # 跳过特殊符号
    }
    # 创建采样参数实例
    sampling_params = SamplingParams(**params_dict)
    # 如果启用了 LoRA，则使用 LoRA 请求生成输出
    if enable_lora:
        async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}", lora_request=LoRARequest("glm-4-lora", 1, lora_path=lora_path)):
            # 生成输出文本
            yield output.outputs[0].text
    # 否则，直接生成输出
    else:
        async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
            # 生成输出文本
            yield output.outputs[0].text

# 定义聊天的异步函数
async def chat():
    # 初始化聊天历史记录
    history = []
    # 设置最大长度
    max_length = 8192
    # 设置 top_p 参数
    top_p = 0.8
    # 设置温度参数
    temperature = 0.6

    # 打印欢迎消息
    print("欢迎来到 GLM-4-9B CLI 聊天。请在下面输入您的消息。")
    # 无限循环，直到用户选择退出
        while True:
            # 提示用户输入
            user_input = input("\nYou: ")
            # 检查用户输入是否为退出命令
            if user_input.lower() in ["exit", "quit"]:
                break
            # 将用户输入添加到历史记录中，初始助手回复为空
            history.append([user_input, ""])
    
            # 初始化消息列表
            messages = []
            # 遍历历史记录，构建消息列表
            for idx, (user_msg, model_msg) in enumerate(history):
                # 如果是最后一条用户消息且没有助手回复，则只添加用户消息
                if idx == len(history) - 1 and not model_msg:
                    messages.append({"role": "user", "content": user_msg})
                    break
                # 如果有用户消息，则添加到消息列表
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                # 如果有助手回复，则添加到消息列表
                if model_msg:
                    messages.append({"role": "assistant", "content": model_msg})
    
            # 打印助手的响应前缀
            print("\nGLM-4: ", end="")
            # 当前输出长度初始化为0
            current_length = 0
            # 初始化输出字符串
            output = ""
            # 异步生成助手的响应
            async for output in vllm_gen(LORA_PATH, enable_lora, messages, top_p, temperature, max_length):
                # 打印输出中从当前长度开始的新内容
                print(output[current_length:], end="", flush=True)
                # 更新当前输出长度
                current_length = len(output)
            # 更新历史记录中最后一条消息的助手回复
            history[-1][1] = output
# 当脚本直接运行时，以下代码将被执行
if __name__ == "__main__":
    # 使用 asyncio 运行 chat() 协程
    asyncio.run(chat())
```