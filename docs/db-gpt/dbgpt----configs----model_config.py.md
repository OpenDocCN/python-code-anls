# `.\DB-GPT-src\dbgpt\configs\model_config.py`

```py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# 导入必要的模块
import os
from functools import cache

# 根据当前文件的路径获取根目录路径
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 设置模型存储路径
MODEL_PATH = os.path.join(ROOT_PATH, "models")
# 设置项目 pilot 子目录路径
PILOT_PATH = os.path.join(ROOT_PATH, "pilot")
# 根据环境变量或默认路径设置日志目录路径
LOGDIR = os.getenv("DBGPT_LOG_DIR", os.path.join(ROOT_PATH, "logs"))
# 设置静态消息图片路径
STATIC_MESSAGE_IMG_PATH = os.path.join(PILOT_PATH, "message/img")

# 设置数据集目录路径
DATASETS_DIR = os.path.join(PILOT_PATH, "datasets")
# 设置数据目录路径
DATA_DIR = os.path.join(PILOT_PATH, "data")
# 设置插件目录路径
PLUGINS_DIR = os.path.join(ROOT_PATH, "plugins")
# 设置模型磁盘缓存目录路径
MODEL_DISK_CACHE_DIR = os.path.join(DATA_DIR, "model_cache")
# 设置 DAG 定义目录路径
_DAG_DEFINITION_DIR = os.path.join(ROOT_PATH, "examples/awel")
# 设置全局语言设置目录路径
LOCALES_DIR = os.path.join(ROOT_PATH, "i18n/locales")

# 获取当前工作目录
current_directory = os.getcwd()

# 定义获取设备类型的函数，使用缓存以提高效率
@cache
def get_device() -> str:
    try:
        import torch

        # 检查是否有可用的 CUDA 设备，若有返回 "cuda"，否则返回 "mps" 或 "cpu"
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    except ModuleNotFoundError:
        # 若未找到 torch 模块，则默认返回 "cpu"
        return "cpu"

# 设置不同模型的配置路径字典
LLM_MODEL_CONFIG = {
    "flan-t5-base": os.path.join(MODEL_PATH, "flan-t5-base"),
    "vicuna-13b": os.path.join(MODEL_PATH, "vicuna-13b"),
    "vicuna-7b": os.path.join(MODEL_PATH, "vicuna-7b"),
    # 基于 Llama2 的模型路径，详见链接
    "vicuna-13b-v1.5": os.path.join(MODEL_PATH, "vicuna-13b-v1.5"),
    "vicuna-7b-v1.5": os.path.join(MODEL_PATH, "vicuna-7b-v1.5"),
    "codegen2-1b": os.path.join(MODEL_PATH, "codegen2-1B"),
    "codet5p-2b": os.path.join(MODEL_PATH, "codet5p-2b"),
    "chatglm-6b-int4": os.path.join(MODEL_PATH, "chatglm-6b-int4"),
    "chatglm-6b": os.path.join(MODEL_PATH, "chatglm-6b"),
    "chatglm2-6b": os.path.join(MODEL_PATH, "chatglm2-6b"),
    "chatglm2-6b-int4": os.path.join(MODEL_PATH, "chatglm2-6b-int4"),
    # 详见链接
    "chatglm3-6b": os.path.join(MODEL_PATH, "chatglm3-6b"),
    # 详见链接
    "glm-4-9b-chat": os.path.join(MODEL_PATH, "glm-4-9b-chat"),
    "glm-4-9b-chat-1m": os.path.join(MODEL_PATH, "glm-4-9b-chat-1m"),
    # 详见链接
    "codegeex4-all-9b": os.path.join(MODEL_PATH, "codegeex4-all-9b"),
    "guanaco-33b-merged": os.path.join(MODEL_PATH, "guanaco-33b-merged"),
    "falcon-40b": os.path.join(MODEL_PATH, "falcon-40b"),
    "gorilla-7b": os.path.join(MODEL_PATH, "gorilla-7b"),
    "gptj-6b": os.path.join(MODEL_PATH, "ggml-gpt4all-j-v1.3-groovy.bin"),
    "proxyllm": "chatgpt_proxyllm",
    "chatgpt_proxyllm": "chatgpt_proxyllm",
    "bard_proxyllm": "bard_proxyllm",
    "claude_proxyllm": "claude_proxyllm",
    "wenxin_proxyllm": "wenxin_proxyllm",
    "tongyi_proxyllm": "tongyi_proxyllm",
    "zhipu_proxyllm": "zhipu_proxyllm",
    "gemini_proxyllm": "gemini_proxyllm",
    "bc_proxyllm": "bc_proxyllm",
    "spark_proxyllm": "spark_proxyllm",
    # 详见链接
}
    "yi_proxyllm": "yi_proxyllm",
    # 用于指定 yi_proxyllm 的路径

    # https://platform.moonshot.cn/docs/
    "moonshot_proxyllm": "moonshot_proxyllm",
    # 用于指定 moonshot_proxyllm 的路径，参考 Moonshot 平台文档

    "ollama_proxyllm": "ollama_proxyllm",
    # 用于指定 ollama_proxyllm 的路径

    # https://platform.deepseek.com/api-docs/
    "deepseek_proxyllm": "deepseek_proxyllm",
    # 用于指定 deepseek_proxyllm 的路径，参考 Deepseek 平台 API 文档

    "llama-2-7b": os.path.join(MODEL_PATH, "Llama-2-7b-chat-hf"),
    # 指定 llama-2-7b 模型的路径，使用 MODEL_PATH 变量与 "Llama-2-7b-chat-hf" 拼接

    "llama-2-13b": os.path.join(MODEL_PATH, "Llama-2-13b-chat-hf"),
    # 指定 llama-2-13b 模型的路径，使用 MODEL_PATH 变量与 "Llama-2-13b-chat-hf" 拼接

    "llama-2-70b": os.path.join(MODEL_PATH, "Llama-2-70b-chat-hf"),
    # 指定 llama-2-70b 模型的路径，使用 MODEL_PATH 变量与 "Llama-2-70b-chat-hf" 拼接

    # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    "meta-llama-3-8b-instruct": os.path.join(MODEL_PATH, "Meta-Llama-3-8B-Instruct"),
    # 指定 meta-llama-3-8b-instruct 模型的路径，使用 MODEL_PATH 变量与 "Meta-Llama-3-8B-Instruct" 拼接

    # https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct
    "meta-llama-3-70b-instruct": os.path.join(MODEL_PATH, "Meta-Llama-3-70B-Instruct"),
    # 指定 meta-llama-3-70b-instruct 模型的路径，使用 MODEL_PATH 变量与 "Meta-Llama-3-70B-Instruct" 拼接

    "meta-llama-3.1-8b-instruct": os.path.join(
        MODEL_PATH, "Meta-Llama-3.1-8B-Instruct"
    ),
    # 指定 meta-llama-3.1-8b-instruct 模型的路径，使用 MODEL_PATH 变量与 "Meta-Llama-3.1-8B-Instruct" 拼接

    "meta-llama-3.1-70b-instruct": os.path.join(
        MODEL_PATH, "Meta-Llama-3.1-70B-Instruct"
    ),
    # 指定 meta-llama-3.1-70b-instruct 模型的路径，使用 MODEL_PATH 变量与 "Meta-Llama-3.1-70B-Instruct" 拼接

    "meta-llama-3.1-405b-instruct": os.path.join(
        MODEL_PATH, "Meta-Llama-3.1-405B-Instruct"
    ),
    # 指定 meta-llama-3.1-405b-instruct 模型的路径，使用 MODEL_PATH 变量与 "Meta-Llama-3.1-405B-Instruct" 拼接

    "baichuan-13b": os.path.join(MODEL_PATH, "Baichuan-13B-Chat"),
    # 指定 baichuan-13b 模型的路径，使用 MODEL_PATH 变量与 "Baichuan-13B-Chat" 拼接

    # please rename "fireballoon/baichuan-vicuna-chinese-7b" to "baichuan-7b"
    "baichuan-7b": os.path.join(MODEL_PATH, "baichuan-7b"),
    # 指定 baichuan-7b 模型的路径，使用 MODEL_PATH 变量与 "baichuan-7b" 拼接

    "baichuan2-7b": os.path.join(MODEL_PATH, "Baichuan2-7B-Chat"),
    # 指定 baichuan2-7b 模型的路径，使用 MODEL_PATH 变量与 "Baichuan2-7B-Chat" 拼接

    "baichuan2-13b": os.path.join(MODEL_PATH, "Baichuan2-13B-Chat"),
    # 指定 baichuan2-13b 模型的路径，使用 MODEL_PATH 变量与 "Baichuan2-13B-Chat" 拼接

    # https://huggingface.co/Qwen/Qwen-7B-Chat
    "qwen-7b-chat": os.path.join(MODEL_PATH, "Qwen-7B-Chat"),
    # 指定 qwen-7b-chat 模型的路径，使用 MODEL_PATH 变量与 "Qwen-7B-Chat" 拼接

    # https://huggingface.co/Qwen/Qwen-7B-Chat-Int8
    "qwen-7b-chat-int8": os.path.join(MODEL_PATH, "Qwen-7B-Chat-Int8"),
    # 指定 qwen-7b-chat-int8 模型的路径，使用 MODEL_PATH 变量与 "Qwen-7B-Chat-Int8" 拼接

    # https://huggingface.co/Qwen/Qwen-7B-Chat-Int4
    "qwen-7b-chat-int4": os.path.join(MODEL_PATH, "Qwen-7B-Chat-Int4"),
    # 指定 qwen-7b-chat-int4 模型的路径，使用 MODEL_PATH 变量与 "Qwen-7B-Chat-Int4" 拼接

    # https://huggingface.co/Qwen/Qwen-14B-Chat
    "qwen-14b-chat": os.path.join(MODEL_PATH, "Qwen-14B-Chat"),
    # 指定 qwen-14b-chat 模型的路径，使用 MODEL_PATH 变量与 "Qwen-14B-Chat" 拼接

    # https://huggingface.co/Qwen/Qwen-14B-Chat-Int8
    "qwen-14b-chat-int8": os.path.join(MODEL_PATH, "Qwen-14B-Chat-Int8"),
    # 指定 qwen-14b-chat-int8 模型的路径，使用 MODEL_PATH 变量与 "Qwen-14B-Chat-Int8" 拼接

    # https://huggingface.co/Qwen/Qwen-14B-Chat-Int4
    "qwen-14b-chat-int4": os.path.join(MODEL_PATH, "Qwen-14B-Chat-Int4"),
    # 指定 qwen-14b-chat-int4 模型的路径，使用 MODEL_PATH 变量与 "Qwen-14B-Chat-Int4" 拼接

    # https://huggingface.co/Qwen/Qwen-72B-Chat
    "qwen-72b-chat": os.path.join(MODEL_PATH, "Qwen-72B-Chat"),
    # 指定 qwen-72b-chat 模型的路径，使用 MODEL_PATH 变量与 "Qwen-72B-Chat" 拼接

    # https://huggingface.co/Qwen/Qwen-72B-Chat-Int8
    "qwen-72b-chat-int8": os.path.join(MODEL_PATH, "Qwen-72B-Chat-Int8"),
    # 指定 qwen-72b-chat-int8 模型的路径，使用 MODEL_PATH 变量与 "Qwen-72B-Chat-Int8" 拼接

    # https://huggingface.co/Qwen/Qwen-72B-Chat-Int4
    "qwen-72b-chat-int4": os.path.join(MODEL_PATH, "Qwen-72B-Chat-Int4"),
    # 指定 qwen-72b-chat-int4 模型的路径，使用 MODEL_PATH 变量与 "Qwen-72B-Chat-Int4" 拼接

    # https://huggingface.co/Qwen/Qwen-1_8B-Chat
    "qwen-1.8b-chat": os.path.join(MODEL_PATH, "Qwen-1_8B-Chat"),
    # 指
    # 定义模型路径中各个模型的名称及其对应路径
    "qwen1.5-1.8b-chat": os.path.join(MODEL_PATH, "Qwen1.5-1.8B-Chat"),
    "qwen1.5-7b-chat": os.path.join(MODEL_PATH, "Qwen1.5-7B-Chat"),
    "qwen1.5-14b-chat": os.path.join(MODEL_PATH, "Qwen1.5-14B-Chat"),
    # 引用 huggingface.co 网站上的 Qwen1.5-32B-Chat 模型链接
    "qwen1.5-32b-chat": os.path.join(MODEL_PATH, "Qwen1.5-32B-Chat"),
    "qwen1.5-72b-chat": os.path.join(MODEL_PATH, "Qwen1.5-72B-Chat"),
    # 引用 huggingface.co 网站上的 Qwen1.5-110B-Chat 模型链接
    "qwen1.5-110b-chat": os.path.join(MODEL_PATH, "Qwen1.5-110B-Chat"),
    # 引用 huggingface.co 网站上的 CodeQwen1.5-7B-Chat 模型链接
    "codeqwen1.5-7b-chat": os.path.join(MODEL_PATH, "CodeQwen1.5-7B-Chat"),
    # 引用 huggingface.co 网站上的 Qwen1.5-MoE-A2.7B-Chat 模型链接
    "qwen1.5-moe-a2.7b-chat": os.path.join(MODEL_PATH, "Qwen1.5-MoE-A2.7B-Chat"),
    "qwen2-57b-a14b-instruct": os.path.join(MODEL_PATH, "Qwen2-57B-A14B-Instruct"),
    "qwen2-57b-a14b-instruct-gptq-int4": os.path.join(
        MODEL_PATH, "Qwen2-57B-A14B-Instruct-GPTQ-Int4"
    ),
    "qwen2-72b-instruct": os.path.join(MODEL_PATH, "Qwen2-72B-Instruct"),
    "qwen2-72b-instruct-awq": os.path.join(MODEL_PATH, "Qwen2-72B-Instruct-AWQ"),
    "qwen2-72b-instruct-gptq-int8": os.path.join(
        MODEL_PATH, "Qwen2-72B-Instruct-GPTQ-Int8"
    ),
    "qwen2-72b-instruct-gptq-int4": os.path.join(
        MODEL_PATH, "Qwen2-72B-Instruct-GPTQ-Int4"
    ),
    "qwen2-7b-instruct": os.path.join(MODEL_PATH, "Qwen2-7B-Instruct"),
    "qwen2-7b-instruct-awq": os.path.join(MODEL_PATH, "Qwen2-7B-Instruct-AWQ"),
    "qwen2-7b-instruct-gptq-int8": os.path.join(
        MODEL_PATH, "Qwen2-7B-Instruct-GPTQ-Int8"
    ),
    "qwen2-7b-instruct-gptq-int4": os.path.join(
        MODEL_PATH, "Qwen2-7B-Instruct-GPTQ-Int4"
    ),
    "qwen2-1.5b-instruct": os.path.join(MODEL_PATH, "Qwen2-1.5B-Instruct"),
    "qwen2-1.5b-instruct-awq": os.path.join(MODEL_PATH, "Qwen2-1.5B-Instruct-AWQ"),
    "qwen2-1.5b-instruct-gptq-int8": os.path.join(
        MODEL_PATH, "Qwen2-1.5B-Instruct-GPTQ-Int8"
    ),
    "qwen2-1.5b-instruct-gptq-int4": os.path.join(
        MODEL_PATH, "Qwen2-1.5B-Instruct-GPTQ-Int4"
    ),
    "qwen2-0.5b-instruct": os.path.join(MODEL_PATH, "Qwen2-0.5B-Instruct"),
    "qwen2-0.5b-instruct-awq": os.path.join(MODEL_PATH, "Qwen2-0.5B-Instruct-AWQ"),
    "qwen2-0.5b-instruct-gptq-int8": os.path.join(
        MODEL_PATH, "Qwen2-0.5B-Instruct-GPTQ-Int8"
    ),
    "qwen2-0.5b-instruct-gptq-int4": os.path.join(
        MODEL_PATH, "Qwen2-0.5B-Instruct-GPTQ-Int4"
    ),
    # 引用 huggingface.co 网站上的 WizardLM-13B-V1.2 模型链接，基于 Llama-2 的 Llama2 版本
    "wizardlm-13b": os.path.join(MODEL_PATH, "WizardLM-13B-V1.2"),
    # wget 命令下载链接的说明，下载 Vicuna-13B-V1.5-GGUF 模型并重命名为 ggml-model-q4_0.gguf
    "llama-cpp": os.path.join(MODEL_PATH, "ggml-model-q4_0.gguf"),
    # 定义了多个模型名称到路径的映射，每个键值对代表一个模型名称和其对应的文件路径
    "internlm-7b": os.path.join(MODEL_PATH, "internlm-chat-7b"),
    "internlm-7b-8k": os.path.join(MODEL_PATH, "internlm-chat-7b-8k"),
    "internlm-20b": os.path.join(MODEL_PATH, "internlm-chat-20b"),
    "internlm2_5-7b-chat": os.path.join(MODEL_PATH, "internlm2_5-7b-chat"),
    "internlm2_5-7b-chat-1m": os.path.join(MODEL_PATH, "internlm2_5-7b-chat-1m"),
    "codellama-7b": os.path.join(MODEL_PATH, "CodeLlama-7b-Instruct-hf"),
    "codellama-7b-sql-sft": os.path.join(MODEL_PATH, "codellama-7b-sql-sft"),
    "codellama-13b": os.path.join(MODEL_PATH, "CodeLlama-13b-Instruct-hf"),
    "codellama-13b-sql-sft": os.path.join(MODEL_PATH, "codellama-13b-sql-sft"),
    # 用于测试的模型路径
    "opt-125m": os.path.join(MODEL_PATH, "opt-125m"),
    # 指向Orca-2-7b模型的路径
    "orca-2-7b": os.path.join(MODEL_PATH, "Orca-2-7b"),
    # 指向Orca-2-13b模型的路径
    "orca-2-13b": os.path.join(MODEL_PATH, "Orca-2-13b"),
    # 指向openchat-3.5模型的路径
    "openchat-3.5": os.path.join(MODEL_PATH, "openchat_3.5"),
    # 指向openchat-3.5-1210模型的路径
    "openchat-3.5-1210": os.path.join(MODEL_PATH, "openchat-3.5-1210"),
    # 指向openchat-3.6-8b-20240522模型的路径
    "openchat-3.6-8b-20240522": os.path.join(MODEL_PATH, "openchat-3.6-8b-20240522"),
    # 指向chinese-alpaca-2-7b模型的路径
    "chinese-alpaca-2-7b": os.path.join(MODEL_PATH, "chinese-alpaca-2-7b"),
    # 指向chinese-alpaca-2-13b模型的路径
    "chinese-alpaca-2-13b": os.path.join(MODEL_PATH, "chinese-alpaca-2-13b"),
    # 指向codegeex2-6b模型的路径
    "codegeex2-6b": os.path.join(MODEL_PATH, "codegeex2-6b"),
    # 指向zephyr-7b-alpha模型的路径
    "zephyr-7b-alpha": os.path.join(MODEL_PATH, "zephyr-7b-alpha"),
    # 指向mistral-7b-instruct-v0.1模型的路径
    "mistral-7b-instruct-v0.1": os.path.join(MODEL_PATH, "Mistral-7B-Instruct-v0.1"),
    # 指向mixtral-8x7b-instruct-v0.1模型的路径
    "mixtral-8x7b-instruct-v0.1": os.path.join(MODEL_PATH, "Mixtral-8x7B-Instruct-v0.1"),
    # 指向solar-10.7b-instruct-v1.0模型的路径
    "solar-10.7b-instruct-v1.0": os.path.join(MODEL_PATH, "SOLAR-10.7B-Instruct-v1.0"),
    # 指向mistral-7b-openorca模型的路径
    "mistral-7b-openorca": os.path.join(MODEL_PATH, "Mistral-7B-OpenOrca"),
    # 指向xwin-lm-7b-v0.1模型的路径
    "xwin-lm-7b-v0.1": os.path.join(MODEL_PATH, "Xwin-LM-7B-V0.1"),
    # 指向xwin-lm-13b-v0.1模型的路径
    "xwin-lm-13b-v0.1": os.path.join(MODEL_PATH, "Xwin-LM-13B-V0.1"),
    # 指向xwin-lm-70b-v0.1模型的路径
    "xwin-lm-70b-v0.1": os.path.join(MODEL_PATH, "Xwin-LM-70B-V0.1"),
    # 指向yi-34b-chat模型的路径
    "yi-34b-chat": os.path.join(MODEL_PATH, "Yi-34B-Chat"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-34b-chat-8bits": os.path.join(MODEL_PATH, "Yi-34B-Chat-8bits"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-34b-chat-4bits": os.path.join(MODEL_PATH, "Yi-34B-Chat-4bits"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-6b-chat": os.path.join(MODEL_PATH, "Yi-6B-Chat"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-1.5-6b-chat": os.path.join(MODEL_PATH, "Yi-1.5-6B-Chat"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-1.5-9b-chat": os.path.join(MODEL_PATH, "Yi-1.5-9B-Chat"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-1.5-9b-chat-16k": os.path.join(MODEL_PATH, "Yi-1.5-9B-Chat-16K"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-1.5-34b-chat": os.path.join(MODEL_PATH, "Yi-1.5-34B-Chat"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "yi-1.5-34b-chat-16k": os.path.join(MODEL_PATH, "Yi-1.5-34B-Chat-16K"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "gemma-7b-it": os.path.join(MODEL_PATH, "gemma-7b-it"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "gemma-2b-it": os.path.join(MODEL_PATH, "gemma-2b-it"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "gemma-2-9b-it": os.path.join(MODEL_PATH, "gemma-2-9b-it"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "gemma-2-27b-it": os.path.join(MODEL_PATH, "gemma-2-27b-it"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "starling-lm-7b-beta": os.path.join(MODEL_PATH, "Starling-LM-7B-beta"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "deepseek-v2-lite-chat": os.path.join(MODEL_PATH, "DeepSeek-V2-Lite-Chat"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "deepseek-coder-v2-instruct": os.path.join(MODEL_PATH, "DeepSeek-Coder-V2-Instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "deepseek-coder-v2-lite-instruct": os.path.join(MODEL_PATH, "DeepSeek-Coder-V2-Lite-Instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "sailor-14b-chat": os.path.join(MODEL_PATH, "Sailor-14B-Chat"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "phi-3-medium-128k-instruct": os.path.join(MODEL_PATH, "Phi-3-medium-128k-instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "phi-3-medium-4k-instruct": os.path.join(MODEL_PATH, "Phi-3-medium-4k-instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "phi-3-small-128k-instruct": os.path.join(MODEL_PATH, "Phi-3-small-128k-instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "phi-3-small-8k-instruct": os.path.join(MODEL_PATH, "Phi-3-small-8k-instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "phi-3-mini-128k-instruct": os.path.join(MODEL_PATH, "Phi-3-mini-128k-instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "phi-3-mini-4k-instruct": os.path.join(MODEL_PATH, "Phi-3-mini-4k-instruct"),
    # 定义模型名称与对应的路径，将模型名称映射到其在文件系统中的位置
    "llama-3-sqlcoder-8b": os.path.join(MODEL_PATH, "llama-3-sqlcoder-8b"),
# 结束定义
}

EMBEDDING_MODEL_CONFIG = {
    # 指定中文大型文本嵌入模型的路径
    "text2vec": os.path.join(MODEL_PATH, "text2vec-large-chinese"),
    # 指定中文基础文本嵌入模型的路径
    "text2vec-base": os.path.join(MODEL_PATH, "text2vec-base-chinese"),
    # 模型 m3e-base 的路径，可从链接 https://huggingface.co/moka-ai/m3e-base 获取
    "m3e-base": os.path.join(MODEL_PATH, "m3e-base"),
    # 模型 m3e-large 的路径，可从链接 https://huggingface.co/moka-ai/m3e-large 获取
    "m3e-large": os.path.join(MODEL_PATH, "m3e-large"),
    # 英文大型 BGE 模型的路径，可从链接 https://huggingface.co/BAAI/bge-large-en 获取
    "bge-large-en": os.path.join(MODEL_PATH, "bge-large-en"),
    # 英文基础 BGE 模型的路径，可从链接 https://huggingface.co/BAAI/bge-base-en 获取
    "bge-base-en": os.path.join(MODEL_PATH, "bge-base-en"),
    # 中文大型 BGE 模型的路径，可从链接 https://huggingface.co/BAAI/bge-large-zh 获取
    "bge-large-zh": os.path.join(MODEL_PATH, "bge-large-zh"),
    # 中文基础 BGE 模型的路径，可从链接 https://huggingface.co/BAAI/bge-base-zh 获取
    "bge-base-zh": os.path.join(MODEL_PATH, "bge-base-zh"),
    # 模型 bge-m3 的路径，需满足 beg need normalize_embeddings=True 的要求，可从链接 https://huggingface.co/BAAI/bge-m3 获取
    "bge-m3": os.path.join(MODEL_PATH, "bge-m3"),
    # 中文大型 GTE 模型的路径
    "gte-large-zh": os.path.join(MODEL_PATH, "gte-large-zh"),
    # 中文基础 GTE 模型的路径
    "gte-base-zh": os.path.join(MODEL_PATH, "gte-base-zh"),
    # 所有 MiniLM-L6-v2 模型的路径
    "sentence-transforms": os.path.join(MODEL_PATH, "all-MiniLM-L6-v2"),
    # OpenAI 的代理模型
    "proxy_openai": "proxy_openai",
    # Azure 的代理模型
    "proxy_azure": "proxy_azure",
    # 通用的 HTTP 嵌入模型
    "proxy_http_openapi": "proxy_http_openapi",
    # Ollama 的代理模型
    "proxy_ollama": "proxy_ollama",
    # 同义词的代理模型
    "proxy_tongyi": "proxy_tongyi",
    # 重新排名模型，rerank mode 是一个特殊的嵌入模型
    "bge-reranker-base": os.path.join(MODEL_PATH, "bge-reranker-base"),
    # 大型的重新排名模型
    "bge-reranker-large": os.path.join(MODEL_PATH, "bge-reranker-large"),
    # HTTP OpenAPI 的重新排名代理模型
    "rerank_proxy_http_openapi": "rerank_proxy_http_openapi",
}

# 知识上传的根路径设置为 DATA_DIR
KNOWLEDGE_UPLOAD_ROOT_PATH = DATA_DIR
```