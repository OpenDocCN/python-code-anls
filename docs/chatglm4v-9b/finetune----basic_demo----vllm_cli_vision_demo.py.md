# `.\chatglm4-finetune\basic_demo\vllm_cli_vision_demo.py`

```
# 该脚本创建一个 CLI 演示，使用 vllm 后端支持 glm-4v-9b 模型，
# 允许用户通过命令行界面与模型互动。

# 使用说明：
# - 运行脚本以启动 CLI 演示。
# - 输入问题与模型互动，获取响应。

# 注意：该脚本包含修改，以处理 markdown 到纯文本的转换，
# 确保 CLI 接口正确显示格式化文本。
"""
import time  # 导入时间模块，用于时间相关功能
import asyncio  # 导入异步模块，以支持异步编程
from PIL import Image  # 从 PIL 库导入 Image 类，用于图像处理
from typing import List, Dict  # 从 typing 导入 List 和 Dict 类型，用于类型注释
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine  # 从 vllm 导入相关类和参数

MODEL_PATH = 'THUDM/glm-4v-9b'  # 定义模型路径常量

# 定义函数以加载模型和分词器
def load_model_and_tokenizer(model_dir: str):
    # 设置异步引擎参数
    engine_args = AsyncEngineArgs(
        model=model_dir,  # 指定模型目录
        tensor_parallel_size=1,  # 设置张量并行大小
        dtype="bfloat16",  # 指定数据类型为 bfloat16
        trust_remote_code=True,  # 信任远程代码执行
        gpu_memory_utilization=0.9,  # 设置 GPU 内存利用率
        enforce_eager=True,  # 强制使用急切执行
        worker_use_ray=True,  # 启用 Ray 进行工作者管理
        disable_log_requests=True,  # 禁用日志请求
        # 如果遇见 OOM 现象，建议开启下述参数
        # enable_chunked_prefill=True,  # 启用分块预填充
        # max_num_batched_tokens=8192  # 设置最大批处理令牌数
    )
    # 从引擎参数创建异步 LLM 引擎
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine  # 返回创建的引擎

# 调用函数以加载模型和分词器
engine = load_model_and_tokenizer(MODEL_PATH)

# 定义异步生成函数
async def vllm_gen(messages: List[Dict[str, str]], top_p: float, temperature: float, max_dec_len: int):
    inputs = messages[-1]  # 获取消息列表中的最后一条消息作为输入
    params_dict = {
        "n": 1,  # 设置生成数量为 1
        "best_of": 1,  # 设置最佳选择数量为 1
        "presence_penalty": 1.0,  # 设置出现惩罚为 1.0
        "frequency_penalty": 0.0,  # 设置频率惩罚为 0.0
        "temperature": temperature,  # 设置生成温度
        "top_p": top_p,  # 设置 top_p 参数
        "top_k": -1,  # 设置 top_k 参数为 -1，表示不使用
        "use_beam_search": False,  # 不使用束搜索
        "length_penalty": 1,  # 设置长度惩罚为 1
        "early_stopping": False,  # 不启用早停
        "ignore_eos": False,  # 不忽略结束标记
        "max_tokens": max_dec_len,  # 设置最大令牌数
        "logprobs": None,  # 日志概率设置为 None
        "prompt_logprobs": None,  # 提示日志概率设置为 None
        "skip_special_tokens": True,  # 跳过特殊令牌
        "stop_token_ids" :[151329, 151336, 151338]  # 设置停止令牌 ID
    }
    # 使用参数字典创建采样参数
    sampling_params = SamplingParams(**params_dict)

    # 异步生成输出
    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        yield output.outputs[0].text  # 生成输出文本


# 定义异步聊天函数
async def chat():
    history = []  # 初始化聊天历史
    max_length = 8192  # 设置最大长度为 8192
    top_p = 0.8  # 设置 top_p 参数为 0.8
    temperature = 0.6  # 设置温度参数为 0.6
    image = None  # 初始化图像变量

    print("Welcome to the GLM-4v-9B CLI chat. Type your messages below.")  # 输出欢迎信息
    image_path = input("Image Path:")  # 提示用户输入图像路径
    try:
        # 尝试打开并转换图像为 RGB 格式
        image = Image.open(image_path).convert("RGB")
    except:
        # 捕获异常并提示用户路径无效，继续文本对话
        print("Invalid image path. Continuing with text conversation.")
    # 无限循环，直到用户选择退出
        while True:
            # 获取用户输入
            user_input = input("\nYou: ")
            # 检查用户输入是否为退出命令
            if user_input.lower() in ["exit", "quit"]:
                break
            # 将用户输入添加到历史记录，初始化模型响应为空
            history.append([user_input, ""])
    
            # 初始化消息列表
            messages = []
            # 遍历历史记录中的消息
            for idx, (user_msg, model_msg) in enumerate(history):
                # 如果是最新的用户消息且没有模型响应，构造包含图像的消息
                if idx == len(history) - 1 and not model_msg:
                    messages.append({
                        "prompt": user_msg,
                        "multi_modal_data": {
                            "image": image
                            },})
                    break
                # 如果存在用户消息，添加到消息列表
                if user_msg:
                    messages.append({"role": "user", "prompt": user_msg})
                # 如果存在模型消息，添加到消息列表
                if model_msg:
                    messages.append({"role": "assistant", "prompt": model_msg})
    
            # 打印模型的响应，准备输出
            print("\nGLM-4v: ", end="")
            # 当前输出长度初始化为0
            current_length = 0
            # 初始化输出字符串
            output = ""
            # 异步生成模型输出
            async for output in vllm_gen(messages, top_p, temperature, max_length):
                # 输出当前生成的内容，保持在同一行
                print(output[current_length:], end="", flush=True)
                # 更新当前输出长度
                current_length = len(output)
            # 更新历史记录中最新消息的模型响应
            history[-1][1] = output
# 当脚本被直接运行时，执行以下代码
if __name__ == "__main__":
    # 启动异步事件循环并运行 chat 函数
    asyncio.run(chat())
```