# `.\DB-GPT-src\dbgpt\model\llm\llama_cpp\llama_cpp.py`

```py
"""
Fork from text-generation-webui https://github.com/oobabooga/text-generation-webui/blob/main/modules/llamacpp_model.py
"""
# 导入必要的库和模块
import logging  # 导入日志模块
import re  # 导入正则表达式模块
from typing import Dict  # 导入类型提示模块

import llama_cpp  # 导入 llama_cpp 库
import torch  # 导入 PyTorch 库

from dbgpt.model.parameter import LlamaCppModelParameters  # 从 dbgpt 模块导入 LlamaCppModelParameters 类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

# 检查是否支持 CUDA，并根据情况导入 llama_cpp_cuda 模块
if torch.cuda.is_available() and not torch.version.hip:
    try:
        import llama_cpp_cuda
    except:
        llama_cpp_cuda = None
else:
    llama_cpp_cuda = None

# 返回适合的 llama_cpp 库，根据是否偏好 CPU 或 CUDA
def llama_cpp_lib(prefer_cpu: bool = False):
    if prefer_cpu or llama_cpp_cuda is None:
        logger.info(f"Llama.cpp use cpu")
        return llama_cpp
    else:
        return llama_cpp_cuda

# 处理禁止 EOS 标志的 logits 处理器
def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float("inf")  # 将 EOS 标志对应的 logits 设置为负无穷
    return logits

# 获取模型参数字典
def get_params(model_path: str, model_params: LlamaCppModelParameters) -> Dict:
    return {
        "model_path": model_path,  # 模型路径
        "n_ctx": model_params.max_context_size,  # 最大上下文大小
        "seed": model_params.seed,  # 随机种子
        "n_threads": model_params.n_threads,  # 线程数
        "n_batch": model_params.n_batch,  # 批量大小
        "use_mmap": True,  # 使用 mmap
        "use_mlock": False,  # 不使用 mlock
        "low_vram": False,  # 不低 VRAM
        "n_gpu_layers": 0 if model_params.prefer_cpu else model_params.n_gpu_layers,  # GPU 层数
        "n_gqa": model_params.n_gqa,  # GQA 数量
        "logits_all": True,  # 所有 logits
        "rms_norm_eps": model_params.rms_norm_eps,  # RMS 标准化参数
    }

# LlamaCppModel 类
class LlamaCppModel:
    def __init__(self):
        self.initialized = False  # 初始化状态
        self.model = None  # 模型对象
        self.verbose = True  # 是否详细输出日志

    def __del__(self):
        if self.model:
            self.model.__del__()  # 删除模型对象

    # 从预训练加载模型
    @classmethod
    def from_pretrained(self, model_path, model_params: LlamaCppModelParameters):
        Llama = llama_cpp_lib(prefer_cpu=model_params.prefer_cpu).Llama  # 获取适合的 Llama 对象
        LlamaCache = llama_cpp_lib(prefer_cpu=model_params.prefer_cpu).LlamaCache  # 获取适合的 LlamaCache 对象

        result = self()  # 创建当前类的实例

        cache_capacity = 0  # 缓存容量初始化为 0
        cache_capacity_str = model_params.cache_capacity  # 缓存容量字符串
        if cache_capacity_str is not None:
            if "GiB" in cache_capacity_str:
                cache_capacity = (
                    int(re.sub("[a-zA-Z]", "", cache_capacity_str)) * 1000 * 1000 * 1000
                )  # 根据单位计算缓存容量（以字节为单位）
            elif "MiB" in cache_capacity_str:
                cache_capacity = (
                    int(re.sub("[a-zA-Z]", "", cache_capacity_str)) * 1000 * 1000
                )  # 根据单位计算缓存容量（以字节为单位）
            else:
                cache_capacity = int(cache_capacity_str)  # 将缓存容量转换为整数

        params = get_params(model_path, model_params)  # 获取模型参数字典
        logger.info("Cache capacity is " + str(cache_capacity) + " bytes")  # 记录缓存容量日志信息
        logger.info(f"Load LLama model with params: {params}")  # 记录加载 LLama 模型的参数日志信息

        result.model = Llama(**params)  # 使用参数初始化模型
        result.verbose = model_params.verbose  # 是否详细输出日志
        if cache_capacity > 0:
            result.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))  # 设置模型缓存

        # 这是丑陋的写法，但在这个库中模型和分词器是同一个对象。
        return result, result  # 返回初始化后的结果对象和它自身
    def encode(self, string):
        # 如果输入是字符串，将其转换为字节流
        if type(string) is str:
            string = string.encode()

        # 使用模型的标记化方法处理输入字符串并返回结果
        return self.model.tokenize(string)

    def decode(self, tokens):
        # 使用模型的去标记化方法处理输入的标记化序列并返回字符串
        return self.model.detokenize(tokens)

    def generate_streaming(self, params, context_len: int):
        # LogitsProcessorList = llama_cpp_lib().LogitsProcessorList

        # 读取参数
        prompt = params["prompt"]
        if self.verbose:
            # 如果设置了详细模式，打印模型的提示信息
            print(f"Prompt of model: \n{prompt}")

        # 从参数中获取生成的温度，重复惩罚，top-p和top-k等设置
        temperature = float(params.get("temperature", 1.0))
        repetition_penalty = float(params.get("repetition_penalty", 1.1))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", -1))  # -1表示禁用
        max_new_tokens = int(params.get("max_new_tokens", 2048))
        echo = bool(params.get("echo", True))

        # 根据上下文长度和最大新token数计算允许的最大源长度
        max_src_len = context_len - max_new_tokens

        # 编码提示文本为模型可处理的标记化输入
        prompt = self.encode(prompt)
        # 根据最大源长度截断标记化后的输入
        prompt = prompt[-max_src_len:]
        # 将截断后的标记化序列解码为UTF-8格式的字符串
        prompt = self.decode(prompt).decode("utf-8")

        # 使用模型的创建完成方法生成流式输出的文本块
        completion_chunks = self.model.create_completion(
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repetition_penalty,
            stream=True,
            echo=echo,
            logits_processor=None,  # 使用空的logits处理器
        )

        output = ""
        # 逐个处理生成的文本块
        for completion_chunk in completion_chunks:
            text = completion_chunk["choices"][0]["text"]
            output += text
            # 生成输出文本
            yield output
```