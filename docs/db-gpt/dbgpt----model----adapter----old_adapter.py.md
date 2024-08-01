# `.\DB-GPT-src\dbgpt\model\adapter\old_adapter.py`

```py
"""
This code file will be deprecated in the future. 
We have integrated fastchat. For details, see: dbgpt/model/model_adapter.py
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import re
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

from dbgpt._private.config import Config
from dbgpt.configs.model_config import get_device
from dbgpt.model.adapter.base import LLMModelAdapter
from dbgpt.model.adapter.template import ConversationAdapter, PromptType
from dbgpt.model.base import ModelType
from dbgpt.model.llm.conversation import Conversation
from dbgpt.model.parameter import (
    LlamaCppModelParameters,
    ModelParameters,
    ProxyModelParameters,
)

if TYPE_CHECKING:
    from dbgpt.app.chat_adapter import BaseChatAdpter

logger = logging.getLogger(__name__)

CFG = Config()


class BaseLLMAdaper:
    """The Base class for multi model, in our project.
    We will support those model, which performance resemble ChatGPT"""

    def use_fast_tokenizer(self) -> bool:
        return False

    def model_type(self) -> str:
        return ModelType.HF

    def model_param_class(self, model_type: str = None) -> ModelParameters:
        model_type = model_type if model_type else self.model_type()
        if model_type == ModelType.LLAMA_CPP:
            return LlamaCppModelParameters
        elif model_type == ModelType.PROXY:
            return ProxyModelParameters
        return ModelParameters

    def match(self, model_path: str):
        return False

    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer


llm_model_adapters: List[BaseLLMAdaper] = []


# Register llm models to adapters, by this we can use multi models.
def register_llm_model_adapters(cls):
    """Register a llm model adapter."""
    llm_model_adapters.append(cls())


@cache
def get_llm_model_adapter(model_name: str, model_path: str) -> BaseLLMAdaper:
    # Prefer using model name matching
    for adapter in llm_model_adapters:
        if adapter.match(model_name):
            logger.info(
                f"Found llm model adapter with model name: {model_name}, {adapter}"
            )
            return adapter

    for adapter in llm_model_adapters:
        if model_path and adapter.match(model_path):
            logger.info(
                f"Found llm model adapter with model path: {model_path}, {adapter}"
            )
            return adapter

    raise ValueError(
        f"Invalid model adapter for model name {model_name} and model path {model_path}"
    )


# TODO support cpu? for practise we support gpt4all or chatglm-6b-int4?
class VicunaLLMAdapater(BaseLLMAdaper):
    """Vicuna Adapter"""

    # 判断模型路径中是否包含"vicuna"关键字
    def match(self, model_path: str):
        return "vicuna" in model_path

    # 根据模型路径加载模型和分词器
    def loader(self, model_path: str, from_pretrained_kwagrs: dict):
        # 根据模型路径加载分词器，关闭快速模式（use_fast=False）
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # 根据模型路径加载条件语言模型，设置低CPU内存使用（low_cpu_mem_usage=True），并传入额外参数
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwagrs
        )
        return model, tokenizer


def auto_configure_device_map(num_gpus):
    """handling multi gpu calls"""
    # transformer.word_embeddings 占用 1 层
    # transformer.final_layernorm 和 lm_head 各占用 1 层
    # transformer.layers 占用 28 层
    # 将总共的 30 层分配到各个 GPU 卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # 初始化设备映射字典，将相关层放置在第一张 GPU 卡上
    device_map = {
        "transformer.embedding.word_embeddings": 0,
        "transformer.encoder.final_layernorm": 0,
        "transformer.output_layer": 0,
        "transformer.rotary_pos_emb": 0,
        "lm_head": 0,
    }

    used = 2
    gpu_target = 0

    # 遍历每一层的编码器，按照设备映射将层分配到不同的 GPU 卡上
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f"transformer.encoder.layers.{i}"] = gpu_target
        used += 1

    return device_map


class ChatGLMAdapater(BaseLLMAdaper):
    """LLM Adatpter for THUDM/chatglm-6b"""

    # 判断模型路径中是否包含"chatglm"关键字
    def match(self, model_path: str):
        return "chatglm" in model_path
    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        # 导入PyTorch库
        import torch

        # 使用AutoTokenizer加载预训练模型的分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # 检查当前设备是否非GPU（即CPU）
        if get_device() != "cuda":
            # 在CPU上加载预训练模型
            model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, **from_pretrained_kwargs
            ).float()
            return model, tokenizer
        else:
            # 初始化设备映射为None
            device_map = None
            # 获取当前系统中的GPU数量
            num_gpus = torch.cuda.device_count()
            # 在GPU上加载预训练模型，并将其转换为半精度（float16）
            model = (
                AutoModel.from_pretrained(
                    model_path, trust_remote_code=True, **from_pretrained_kwargs
                ).half()
                # .cuda()  # 此处原本是将模型移到GPU，但已被注释掉
            )
            # 导入加速库中的模型分发函数
            from accelerate import dispatch_model

            # 如果设备映射为None，则自动配置设备映射
            if device_map is None:
                device_map = auto_configure_device_map(num_gpus)

            # 使用分发模型函数将模型分发到指定的设备映射上
            model = dispatch_model(model, device_map=device_map)

            return model, tokenizer
class GuanacoAdapter(BaseLLMAdaper):
    """TODO Support guanaco"""

    def match(self, model_path: str):
        # 检查模型路径是否包含 'guanaco'
        return "guanaco" in model_path

    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        # 从预训练模型路径中创建 tokenizer 对象
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        # 根据预训练模型路径加载模型，使用 4 位加载
        model = AutoModelForCausalLM.from_pretrained(
            model_path, load_in_4bit=True, **from_pretrained_kwargs
        )
        return model, tokenizer


class FalconAdapater(BaseLLMAdaper):
    """falcon Adapter"""

    def match(self, model_path: str):
        # 检查模型路径是否包含 'falcon'
        return "falcon" in model_path

    def loader(self, model_path: str, from_pretrained_kwagrs: dict):
        # 根据预训练模型路径创建 tokenizer 对象，不使用快速模式
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if CFG.QLoRA:
            from transformers import BitsAndBytesConfig

            # 配置 BitsAndBytesConfig 用于量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=False,
            )
            # 根据预训练模型路径加载模型，使用 4 位加载和指定的量化配置
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,  # quantize
                quantization_config=bnb_config,
                trust_remote_code=True,
                **from_pretrained_kwagrs,
            )
        else:
            # 根据预训练模型路径加载模型，信任远程代码
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **from_pretrained_kwagrs,
            )
        return model, tokenizer


class GorillaAdapter(BaseLLMAdaper):
    """TODO Support gorilla"""

    def match(self, model_path: str):
        # 检查模型路径是否包含 'gorilla'
        return "gorilla" in model_path

    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        # 根据预训练模型路径创建 tokenizer 对象，不使用快速模式
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        # 根据预训练模型路径加载模型，启用低 CPU 内存使用
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
        )
        return model, tokenizer


class StarCoderAdapter(BaseLLMAdaper):
    pass


class KoalaLLMAdapter(BaseLLMAdaper):
    """Koala LLM Adapter which Based LLaMA"""

    def match(self, model_path: str):
        # 检查模型路径是否包含 'koala'
        return "koala" in model_path


class RWKV4LLMAdapter(BaseLLMAdaper):
    """LLM Adapter for RwKv4"""

    def match(self, model_path: str):
        # 检查模型路径是否包含 'RWKV-4'
        return "RWKV-4" in model_path

    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        # TODO: 实现加载器的具体功能，当前函数体为空
        pass


class GPT4AllAdapter(BaseLLMAdaper):
    """
    A light version for someone who want practise LLM use laptop.
    All model names see: https://gpt4all.io/models/models.json
    """

    def match(self, model_path: str):
        # 检查模型路径是否包含 'gptj-6b'
        return "gptj-6b" in model_path
    # 定义一个方法 loader，接收模型路径和预训练参数的字典作为参数
    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        # 导入 gpt4all 模块
        import gpt4all
        
        # 如果模型路径为 None 并且 from_pretrained_kwargs 中没有指定 model_name
        if model_path is None and from_pretrained_kwargs.get("model_name") is None:
            # 使用默认模型名称创建一个 GPT4All 模型对象
            model = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy")
        else:
            # 否则，从 model_path 中分离出路径和文件名
            path, file = os.path.split(model_path)
            # 使用指定的路径和模型文件名创建一个 GPT4All 模型对象
            model = gpt4all.GPT4All(model_path=path, model_name=file)
        
        # 返回创建的模型对象和 None（没有额外的返回值）
        return model, None
class ProxyllmAdapter(BaseLLMAdaper):
    """The model adapter for local proxy"""

    # 返回模型类型字符串为 "PROXY"
    def model_type(self) -> str:
        return ModelType.PROXY

    # 检查模型路径是否包含字符串 "proxyllm"
    def match(self, model_path: str):
        return "proxyllm" in model_path

    # 加载模型，对于代理模型直接返回 "proxyllm" 和 None
    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        return "proxyllm", None


class Llama2Adapter(BaseLLMAdaper):
    """The model adapter for llama-2"""

    # 检查模型路径是否包含字符串 "llama-2"（不区分大小写）
    def match(self, model_path: str):
        return "llama-2" in model_path.lower()

    # 加载模型，调用父类方法获取模型和分词器，然后设置配置的结束标记和填充标记
    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().loader(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer


class CodeLlamaAdapter(BaseLLMAdaper):
    """The model adapter for codellama"""

    # 检查模型路径是否包含字符串 "codellama"（不区分大小写）
    def match(self, model_path: str):
        return "codellama" in model_path.lower()

    # 加载模型，调用父类方法获取模型和分词器，然后设置配置的结束标记和填充标记
    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        model, tokenizer = super().loader(model_path, from_pretrained_kwargs)
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer


class BaichuanAdapter(BaseLLMAdaper):
    """The model adapter for Baichuan models (e.g., baichuan-inc/Baichuan-13B-Chat)"""

    # 检查模型路径是否包含字符串 "baichuan"（不区分大小写）
    def match(self, model_path: str):
        return "baichuan" in model_path.lower()

    # 加载模型，使用AutoTokenizer和AutoModelForCausalLM从预训练模型路径加载模型和分词器
    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer


class WizardLMAdapter(BaseLLMAdaper):
    # 检查模型路径是否包含字符串 "wizardlm"（不区分大小写）
    def match(self, model_path: str):
        return "wizardlm" in model_path.lower()


class LlamaCppAdapater(BaseLLMAdaper):
    @staticmethod
    # 解析模型路径，返回是否支持以及模型路径
    def _parse_model_path(model_path: str) -> Tuple[bool, str]:
        path = Path(model_path)
        if not path.exists():
            # 只支持本地模型，路径不存在则返回False和None
            return False, None
        if not path.is_file():
            # 如果路径不是文件，则寻找符合 *ggml*.gguf 格式的模型文件
            model_paths = list(path.glob("*ggml*.gguf"))
            if not model_paths:
                return False, None
            model_path = str(model_paths[0])
            logger.warn(
                f"Model path {model_path} is not single file, use first *gglm*.gguf model file: {model_path}"
            )
        if not re.fullmatch(r".*ggml.*\.gguf", model_path):
            return False, None
        return True, model_path

    # 返回模型类型为 ModelType.LLAMA_CPP
    def model_type(self) -> ModelType:
        return ModelType.LLAMA_CPP
    def match(self, model_path: str):
        """
        判断给定的模型路径是否匹配特定条件
        https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
        """
        # 检查是否是特定的模型路径字符串
        if "llama-cpp" == model_path:
            return True
        # 调用内部方法解析模型路径，并返回解析结果中的匹配布尔值
        is_match, _ = LlamaCppAdapater._parse_model_path(model_path)
        return is_match

    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        """
        加载模型和分词器
        TODO 暂时不支持
        """
        # 调用内部方法解析模型路径，并忽略解析结果的第一个返回值
        _, model_path = LlamaCppAdapater._parse_model_path(model_path)
        # 使用指定的模型路径加载分词器，允许信任远程代码，但不使用快速模式
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        # 使用指定的模型路径加载模型，允许信任远程代码，启用低CPU内存使用模式，并应用其他从参数传入的加载参数
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **from_pretrained_kwargs,
        )
        return model, tokenizer
class InternLMAdapter(BaseLLMAdaper):
    """The model adapter for internlm/internlm-chat-7b"""

    # 判断给定的模型路径是否包含 "internlm"，用于匹配模型适配器
    def match(self, model_path: str):
        return "internlm" in model_path.lower()

    # 加载模型和分词器，返回加载后的模型和分词器对象
    def loader(self, model_path: str, from_pretrained_kwargs: dict):
        # 获取从预训练关键字参数中的修订版本（默认为 "main"）
        revision = from_pretrained_kwargs.get("revision", "main")
        
        # 使用AutoModelForCausalLM从预训练模型路径加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage=True,  # 设置低CPU内存使用
            trust_remote_code=True,  # 信任远程代码
            **from_pretrained_kwargs,
        )
        model = model.eval()  # 将模型设置为评估模式

        # 如果模型路径中包含 "8k"，则设置模型配置的最大序列长度为 8192
        if "8k" in model_path.lower():
            model.config.max_sequence_length = 8192
        
        # 使用AutoTokenizer从预训练模型路径加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,  # 禁用快速模式分词器
            trust_remote_code=True,  # 信任远程代码
            revision=revision
        )
        
        return model, tokenizer


class OldLLMModelAdapterWrapper(LLMModelAdapter):
    """Wrapping old adapter, which may be removed later"""

    # 初始化方法，接受一个基础LLM适配器和一个基础聊天适配器
    def __init__(self, adapter: BaseLLMAdaper, chat_adapter: "BaseChatAdpter") -> None:
        self._adapter = adapter  # 存储基础LLM适配器对象
        self._chat_adapter = chat_adapter  # 存储基础聊天适配器对象

    # 创建并返回一个旧的适配器包装器对象
    def new_adapter(self, **kwargs) -> "LLMModelAdapter":
        return OldLLMModelAdapterWrapper(self._adapter, self._chat_adapter)

    # 返回基础LLM适配器的使用快速分词器的状态
    def use_fast_tokenizer(self) -> bool:
        return self._adapter.use_fast_tokenizer()

    # 返回基础LLM适配器的模型类型
    def model_type(self) -> str:
        return self._adapter.model_type()

    # 根据模型类型返回基础LLM适配器的模型参数类
    def model_param_class(self, model_type: str = None) -> ModelParameters:
        return self._adapter.model_param_class(model_type)

    # 获取默认的会话模板适配器对象，如果不存在则返回None
    def get_default_conv_template(
        self, model_name: str, model_path: str
    ) -> Optional[ConversationAdapter]:
        # 从基础聊天适配器中获取会话模板对象
        conv_template = self._chat_adapter.get_conv_template(model_path)
        return OldConversationAdapter(conv_template) if conv_template else None

    # 调用基础LLM适配器的加载方法
    def load(self, model_path: str, from_pretrained_kwargs: dict):
        return self._adapter.loader(model_path, from_pretrained_kwargs)

    # 获取基础聊天适配器的生成流函数
    def get_generate_stream_function(self, model, model_path: str):
        return self._chat_adapter.get_generate_stream_func(model_path)

    # 返回包含类名和基础LLM适配器类信息的字符串表示形式
    def __str__(self) -> str:
        return "{}({}.{})".format(
            self.__class__.__name__,
            self._adapter.__class__.__module__,
            self._adapter.__class__.__name__,
        )


class OldConversationAdapter(ConversationAdapter):
    """Wrapping old Conversation, which may be removed later"""

    # 初始化方法，接受一个旧的Conversation对象作为参数
    def __init__(self, conv: Conversation) -> None:
        self._conv = conv  # 存储旧的Conversation对象

    # 返回当前会话适配器的提示类型（DBGPT）
    @property
    def prompt_type(self) -> PromptType:
        return PromptType.DBGPT

    # 返回当前会话适配器的角色元组
    @property
    def roles(self) -> Tuple[str]:
        return self._conv.roles

    # 返回当前会话适配器的分隔字符串（如果有的话）
    @property
    def sep(self) -> Optional[str]:
        return self._conv.sep

    # 返回当前会话适配器的停止字符串
    @property
    def stop_str(self) -> str:
        return self._conv.stop_str

    # 返回当前会话适配器的停止令牌ID列表（如果有的话）
    @property
    def stop_token_ids(self) -> Optional[List[int]]:
        return self._conv.stop_token_ids

    # 获取当前会话适配器的提示信息字符串
    def get_prompt(self) -> str:
        return self._conv.get_prompt()
    # 设置系统消息到对话对象中
    def set_system_message(self, system_message: str) -> None:
        self._conv.update_system_message(system_message)

    # 向对话中追加消息
    def append_message(self, role: str, message: str) -> None:
        self._conv.append_message(role, message)

    # 更新对话中的最后一条消息
    def update_last_message(self, message: str) -> None:
        self._conv.update_last_message(message)

    # 创建当前对话适配器的副本并返回
    def copy(self) -> "ConversationAdapter":
        return OldConversationAdapter(self._conv.copy())
# 使用函数 register_llm_model_adapters 注册 VicunaLLMAdapater 适配器
register_llm_model_adapters(VicunaLLMAdapater)
# 使用函数 register_llm_model_adapters 注册 ChatGLMAdapater 适配器
register_llm_model_adapters(ChatGLMAdapater)
# 使用函数 register_llm_model_adapters 注册 GuanacoAdapter 适配器
register_llm_model_adapters(GuanacoAdapter)
# 使用函数 register_llm_model_adapters 注册 FalconAdapater 适配器
register_llm_model_adapters(FalconAdapater)
# 使用函数 register_llm_model_adapters 注册 GorillaAdapter 适配器
register_llm_model_adapters(GorillaAdapter)
# 使用函数 register_llm_model_adapters 注册 GPT4AllAdapter 适配器
register_llm_model_adapters(GPT4AllAdapter)
# 使用函数 register_llm_model_adapters 注册 Llama2Adapter 适配器
register_llm_model_adapters(Llama2Adapter)
# 使用函数 register_llm_model_adapters 注册 CodeLlamaAdapter 适配器
register_llm_model_adapters(CodeLlamaAdapter)
# 使用函数 register_llm_model_adapters 注册 BaichuanAdapter 适配器
register_llm_model_adapters(BaichuanAdapter)
# 使用函数 register_llm_model_adapters 注册 WizardLMAdapter 适配器
register_llm_model_adapters(WizardLMAdapter)
# 使用函数 register_llm_model_adapters 注册 LlamaCppAdapater 适配器
register_llm_model_adapters(LlamaCppAdapater)
# 使用函数 register_llm_model_adapters 注册 InternLMAdapter 适配器
register_llm_model_adapters(InternLMAdapter)
# TODO 默认支持 vicuna，其他模型需要测试和评估

# 仅用于 test_py 测试用途，稍后移除此行
register_llm_model_adapters(ProxyllmAdapter)
# 仅用于 test_py 测试用途，稍后移除此行
register_llm_model_adapters(BaseLLMAdaper)
```