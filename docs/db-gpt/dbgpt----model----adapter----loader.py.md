# `.\DB-GPT-src\dbgpt\model\adapter\loader.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Any, Dict, Optional

from dbgpt.configs.model_config import get_device
from dbgpt.model.adapter.base import LLMModelAdapter
from dbgpt.model.adapter.model_adapter import get_llm_model_adapter
from dbgpt.model.base import ModelType
from dbgpt.model.parameter import (
    LlamaCppModelParameters,
    ModelParameters,
    ProxyModelParameters,
)
from dbgpt.util import get_gpu_memory
from dbgpt.util.parameter_utils import EnvArgumentParser, _genenv_ignoring_key_case

logger = logging.getLogger(__name__)


def _check_multi_gpu_or_4bit_quantization(model_params: ModelParameters):
    # TODO: vicuna-v1.5 8-bit quantization info is slow
    # TODO: support wizardlm quantization, see: https://huggingface.co/WizardLM/WizardLM-13B-V1.2/discussions/5
    # TODO: support internlm quantization
    # 检查模型是否支持多GPU或4位量化
    model_name = model_params.model_name.lower()
    supported_models = ["llama", "baichuan", "vicuna"]
    return any(m in model_name for m in supported_models)


def _check_quantization(model_params: ModelParameters):
    # 检查模型是否支持量化
    model_name = model_params.model_name.lower()
    has_quantization = any([model_params.load_8bit or model_params.load_4bit])
    if has_quantization:
        if model_params.device != "cuda":
            logger.warn(
                "8-bit quantization and 4-bit quantization just supported by cuda"
            )
            return False
        elif "chatglm" in model_name:
            if "int4" not in model_name:
                logger.warn(
                    "chatglm or chatglm2 not support quantization now, see: https://github.com/huggingface/transformers/issues/25228"
                )
            return False
    return has_quantization


def _get_model_real_path(model_name, default_model_path) -> str:
    """Get model real path by model name
    priority from high to low:
    1. environment variable with key: {model_name}_model_path
    2. environment variable with key: model_path
    3. default_model_path
    """
    # 根据模型名称获取模型的真实路径
    env_prefix = model_name + "_"
    env_prefix = env_prefix.replace("-", "_")
    env_model_path = _genenv_ignoring_key_case("model_path", env_prefix=env_prefix)
    if env_model_path:
        return env_model_path
    return _genenv_ignoring_key_case("model_path", default_value=default_model_path)


class ModelLoader:
    """Model loader is a class for model load

      Args: model_path

    TODO: multi model support.
    """

    def __init__(self, model_path: str, model_name: str = None) -> None:
        # 初始化模型加载器，设定设备和模型路径
        self.device = get_device()
        self.model_path = model_path
        self.model_name = model_name
        self.prompt_template: str = None

    # TODO multi gpu support
    def loader(
        self,
        load_8bit=False,
        load_4bit=False,
        debug=False,
        cpu_offloading=False,
        max_gpu_memory: Optional[str] = None,
        ):
        # 获取指定模型的适配器
        llm_adapter = get_llm_model_adapter(self.model_name, self.model_path)
        # 获取模型类型
        model_type = llm_adapter.model_type()
        # 获取模型参数类
        param_cls = llm_adapter.model_param_class(model_type)

        # 创建环境参数解析器
        args_parser = EnvArgumentParser()
        # 根据模型名称前缀从环境变量中读取模型参数，当前具有最高优先级
        # vicuna_13b_max_gpu_memory=13Gib 或 VICUNA_13B_MAX_GPU_MEMORY=13Gib
        env_prefix = self.model_name + "_"
        env_prefix = env_prefix.replace("-", "_")
        # 解析参数并转换为数据类
        model_params = args_parser.parse_args_into_dataclass(
            param_cls,
            env_prefixes=[env_prefix],
            device=self.device,
            model_path=self.model_path,
            model_name=self.model_name,
            max_gpu_memory=max_gpu_memory,
            cpu_offloading=cpu_offloading,
            load_8bit=load_8bit,
            load_4bit=load_4bit,
            verbose=debug,
        )
        # 设置提示模板
        self.prompt_template = model_params.prompt_template

        # 记录模型参数信息
        logger.info(f"model_params:\n{model_params}")

        # 根据模型类型选择加载器
        if model_type == ModelType.HF:
            return huggingface_loader(llm_adapter, model_params)
        elif model_type == ModelType.LLAMA_CPP:
            return llamacpp_loader(llm_adapter, model_params)
        else:
            raise Exception(f"Unkown model type {model_type}")

    def loader_with_params(
        self, model_params: ModelParameters, llm_adapter: LLMModelAdapter
    ):
        # 获取模型类型
        model_type = llm_adapter.model_type()
        # 设置提示模板
        self.prompt_template = model_params.prompt_template
        # 根据模型类型选择加载器
        if model_type == ModelType.HF:
            return huggingface_loader(llm_adapter, model_params)
        elif model_type == ModelType.LLAMA_CPP:
            return llamacpp_loader(llm_adapter, model_params)
        elif model_type == ModelType.PROXY:
            # 返回代理加载器
            # return proxyllm_loader(llm_adapter, model_params)
            return llm_adapter.load_from_params(model_params)
        elif model_type == ModelType.VLLM:
            return llm_adapter.load_from_params(model_params)
        else:
            raise Exception(f"Unkown model type {model_type}")
def huggingface_loader(llm_adapter: LLMModelAdapter, model_params: ModelParameters):
    import torch  # 导入PyTorch库

    from dbgpt.model.llm.compression import compress_module  # 导入压缩模块

    device = model_params.device  # 从参数中获取设备信息
    max_memory = None  # 初始化最大内存为None

    # if device is cpu or mps. gpu need to be zero
    num_gpus = 0  # 初始化GPU数量为0

    if device == "cpu":
        kwargs = {"torch_dtype": torch.float32}  # 如果设备是CPU，设置Torch的数据类型为float32
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}  # 如果设备是CUDA，设置Torch的数据类型为float16
        num_gpus = torch.cuda.device_count()  # 获取CUDA设备数量
        available_gpu_memory = get_gpu_memory(num_gpus)  # 获取可用GPU内存信息
        max_memory = {
            i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(num_gpus)
        }  # 计算每个GPU的最大内存，并存储为字典
        if num_gpus != 1:
            kwargs["device_map"] = "auto"  # 如果有多个GPU，自动映射设备
            if model_params.max_gpu_memory:
                logger.info(
                    f"There has max_gpu_memory from config: {model_params.max_gpu_memory}"
                )  # 记录配置中的最大GPU内存信息
                max_memory = {i: model_params.max_gpu_memory for i in range(num_gpus)}  # 使用配置的最大GPU内存
                kwargs["max_memory"] = max_memory
            else:
                kwargs["max_memory"] = max_memory  # 使用计算得到的最大GPU内存
        logger.debug(f"max_memory: {max_memory}")  # 记录调试信息：最大内存情况

    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}  # 如果设备是MPS，设置Torch的数据类型为float16

        import transformers  # 导入transformers库

        version = tuple(int(v) for v in transformers.__version__.split("."))  # 获取transformers库的版本信息
        if version < (4, 35, 0):
            from dbgpt.model.llm.monkey_patch import (
                replace_llama_attn_with_non_inplace_operations,
            )  # 导入monkey_patch模块中的替换函数

            # NOTE: Recent transformers library seems to fix the mps issue, also
            # it has made some changes causing compatibility issues with our
            # original patch. So we only apply the patch for older versions.
            # Avoid bugs in mps backend by not using in-place operations.
            replace_llama_attn_with_non_inplace_operations()  # 替换注意力机制以避免MPS后端的问题

    else:
        raise ValueError(f"Invalid device: {device}")  # 抛出异常：无效的设备类型

    model, tokenizer = _try_load_default_quantization_model(
        llm_adapter, device, num_gpus, model_params, kwargs
    )  # 尝试加载默认的量化模型

    if model:
        return model, tokenizer  # 如果成功加载模型，则返回模型和分词器

    can_quantization = _check_quantization(model_params)  # 检查是否可以量化

    if can_quantization and (num_gpus > 1 or model_params.load_4bit):
        if _check_multi_gpu_or_4bit_quantization(model_params):
            return load_huggingface_quantization_model(
                llm_adapter, model_params, kwargs, max_memory
            )  # 加载Hugging Face量化模型
        else:
            logger.warn(
                f"Current model {model_params.model_name} not supported quantization"
            )  # 记录警告信息：当前模型不支持量化

    # default loader
    model, tokenizer = llm_adapter.load(model_params.model_path, kwargs)  # 加载默认模型和分词器

    if model_params.load_8bit and num_gpus == 1 and tokenizer:
        # TODO merge current code into `load_huggingface_quantization_model`
        compress_module(model, model_params.device)  # 压缩模型

    return _handle_model_and_tokenizer(model, tokenizer, device, num_gpus, model_params)  # 处理模型和分词器的返回
def _try_load_default_quantization_model(
    llm_adapter: LLMModelAdapter,
    device: str,
    num_gpus: int,
    model_params: ModelParameters,
    kwargs: Dict[str, Any],
):
    """Try load default quantization model(Support by huggingface default)"""
    # 克隆 kwargs 字典，以防止修改原始参数
    cloned_kwargs = {k: v for k, v in kwargs.items()}
    try:
        model, tokenizer = None, None
        # 如果设备不是 "cuda"，则返回 None
        if device != "cuda":
            return None, None
        # 如果要加载 8 位模型并且 llm_adapter 支持 8 位模型
        elif model_params.load_8bit and llm_adapter.support_8bit:
            cloned_kwargs["load_in_8bit"] = True
            # 使用 llm_adapter 加载模型和分词器
            model, tokenizer = llm_adapter.load(model_params.model_path, cloned_kwargs)
        # 如果要加载 4 位模型并且 llm_adapter 支持 4 位模型
        elif model_params.load_4bit and llm_adapter.support_4bit:
            cloned_kwargs["load_in_4bit"] = True
            # 使用 llm_adapter 加载模型和分词器
            model, tokenizer = llm_adapter.load(model_params.model_path, cloned_kwargs)
        # 如果成功加载了模型
        if model:
            # 记录加载成功的信息
            logger.info(
                f"Load default quantization model {model_params.model_name} success"
            )
            # 处理模型和分词器，返回处理后的结果
            return _handle_model_and_tokenizer(
                model, tokenizer, device, num_gpus, model_params
            )
        # 如果未能加载模型，则返回 None
        return None, None
    # 捕获可能的异常
    except Exception as e:
        # 记录加载失败的信息和异常详情
        logger.warning(
            f"Load default quantization model {model_params.model_name} failed, error: {str(e)}"
        )
        # 返回 None
        return None, None


def _handle_model_and_tokenizer(
    model, tokenizer, device: str, num_gpus: int, model_params: ModelParameters
):
    # 如果满足条件，将模型移动到指定设备上
    if (
        (device == "cuda" and num_gpus == 1 and not model_params.cpu_offloading)
        or device == "mps"
        and tokenizer
    ):
        try:
            model.to(device)
        # 捕获可能的错误类型
        except ValueError:
            pass
        except AttributeError:
            pass
    # 如果 model_params.verbose 为 True，则打印模型信息
    if model_params.verbose:
        print(model)
    # 返回模型和分词器
    return model, tokenizer


def load_huggingface_quantization_model(
    llm_adapter: LLMModelAdapter,
    model_params: ModelParameters,
    kwargs: Dict,
    max_memory: Dict[int, str],
):
    import torch

    try:
        import transformers
        from accelerate import init_empty_weights
        from accelerate.utils import infer_auto_device_map
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            LlamaForCausalLM,
            LlamaTokenizer,
        )
    # 捕获可能的 ImportError 异常
    except ImportError as exc:
        # 抛出 ValueError 异常，并提供安装相应依赖的指导信息
        raise ValueError(
            "Could not import depend python package "
            "Please install it with `pip install transformers` "
            "`pip install bitsandbytes``pip install accelerate`."
        ) from exc
    # 如果模型名称中包含 "llama-2"，并且 transformers 版本低于 "4.31.0"
    if (
        "llama-2" in model_params.model_name.lower()
        and not transformers.__version__ >= "4.31.0"
    ):
        # 抛出 ValueError 异常，指示需要的 transformers 版本不满足要求
        raise ValueError(
            "Llama-2 quantization require transformers.__version__>=4.31.0"
        )
    # 初始化参数字典，并设置相关选项
    params = {"low_cpu_mem_usage": True}
    params["low_cpu_mem_usage"] = True
    params["device_map"] = "auto"
    # 从kwargs参数中获取torch_dtype
    torch_dtype = kwargs.get("torch_dtype")

    # 如果模型参数指定了加载4位量化模型
    if model_params.load_4bit:
        compute_dtype = None
        # 如果模型参数指定了计算精度，并且在允许的精度列表中
        if model_params.compute_dtype and model_params.compute_dtype in [
            "bfloat16",
            "float16",
            "float32",
        ]:
            # 根据指定的计算精度字符串，通过eval动态获取对应的torch数据类型对象
            compute_dtype = eval("torch.{}".format(model_params.compute_dtype))

        # 设置4位量化配置参数
        quantization_config_params = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_quant_type": model_params.quant_type,
            "bnb_4bit_use_double_quant": model_params.use_double_quant,
        }
        # 记录警告信息，显示使用的4位量化参数配置
        logger.warn(
            "Using the following 4-bit params: " + str(quantization_config_params)
        )
        # 在params字典中设置quantization_config键对应的值，初始化为BitsAndBytesConfig对象
        params["quantization_config"] = BitsAndBytesConfig(**quantization_config_params)

    # 如果模型参数指定了加载8位量化模型，并且有max_memory值
    elif model_params.load_8bit and max_memory:
        # 在params字典中设置quantization_config键对应的值，初始化为BitsAndBytesConfig对象
        # 同时设置load_in_8bit和llm_int8_enable_fp32_cpu_offload参数
        params["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
        )

    # 如果模型参数指定了直接加载8位量化模型
    elif model_params.load_in_8bit:
        # 在params字典中设置quantization_config键对应的值，初始化为BitsAndBytesConfig对象
        params["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # 如果torch_dtype存在，则在params字典中设置torch_dtype键对应的值
    params["torch_dtype"] = torch_dtype if torch_dtype else torch.float16
    # 在params字典中设置max_memory键对应的值
    params["max_memory"] = max_memory

    # 如果模型名包含"chatglm"
    if "chatglm" in model_params.model_name.lower():
        # 设置LoaderClass为AutoModel
        LoaderClass = AutoModel
    else:
        # 从预训练模型路径加载配置信息
        config = AutoConfig.from_pretrained(
            model_params.model_path, trust_remote_code=model_params.trust_remote_code
        )
        # 如果配置信息表明是编码-解码模型
        if config.to_dict().get("is_encoder_decoder", False):
            # 设置LoaderClass为AutoModelForSeq2SeqLM
            LoaderClass = AutoModelForSeq2SeqLM
        else:
            # 否则设置LoaderClass为AutoModelForCausalLM
            LoaderClass = AutoModelForCausalLM

    # 如果模型参数指定了加载8位量化模型，并且max_memory不为空
    if model_params.load_8bit and max_memory is not None:
        # 从预训练模型路径加载配置信息
        config = AutoConfig.from_pretrained(
            model_params.model_path, trust_remote_code=model_params.trust_remote_code
        )
        # 使用init_empty_weights函数初始化模型权重
        with init_empty_weights():
            # 根据配置和模型加载类创建模型对象
            model = LoaderClass.from_config(
                config, trust_remote_code=model_params.trust_remote_code
            )

        # 绑定模型的权重
        model.tie_weights()
        # 推断出自动设备映射
        params["device_map"] = infer_auto_device_map(
            model,
            dtype=torch.int8,
            max_memory=params["max_memory"].copy(),
            no_split_module_classes=model._no_split_modules,
        )

    try:
        # 如果允许信任远程代码，则在params中设置trust_remote_code为True
        if model_params.trust_remote_code:
            params["trust_remote_code"] = True
        # 记录信息日志，显示params的内容
        logger.info(f"params: {params}")
        # 根据模型路径和params加载模型
        model = LoaderClass.from_pretrained(model_params.model_path, **params)

    # 捕获异常情况
    except Exception as e:
        # 记录错误日志，显示加载量化模型失败的错误信息和params的内容
        logger.error(
            f"Load quantization model failed, error: {str(e)}, params: {params}"
        )
        # 抛出异常
        raise e

    # 加载分词器部分
    # 如果模型类型是 LlamaForCausalLM，则执行以下操作
    if type(model) is LlamaForCausalLM:
        # 记录日志，指示当前模型类型为 LlamaForCausalLM，使用 LlamaTokenizer 加载分词器
        logger.info(
            f"Current model is type of: LlamaForCausalLM, load tokenizer by LlamaTokenizer"
        )
        # 使用预训练模型路径加载 LlamaTokenizer，并确保清除分词空格
        tokenizer = LlamaTokenizer.from_pretrained(
            model_params.model_path, clean_up_tokenization_spaces=True
        )
        # 尝试设置特定的特殊标记 ID，这段代码是为了解决一些用户遇到的问题，但可能会引发错误
        try:
            tokenizer.eos_token_id = 2  # 设置结束标记的 ID
            tokenizer.bos_token_id = 1  # 设置开始标记的 ID
            tokenizer.pad_token_id = 0  # 设置填充标记的 ID
        except Exception as e:
            # 如果设置特殊标记 ID 出错，记录警告日志并继续
            logger.warn(f"{str(e)}")
    else:
        # 如果模型类型不是 LlamaForCausalLM，则执行以下操作
        logger.info(
            f"Current model type is not LlamaForCausalLM, load tokenizer by AutoTokenizer"
        )
        # 使用预训练模型路径加载 AutoTokenizer，并根据设置使用远程代码信任和快速分词器选项
        tokenizer = AutoTokenizer.from_pretrained(
            model_params.model_path,
            trust_remote_code=model_params.trust_remote_code,
            use_fast=llm_adapter.use_fast_tokenizer(),
        )

    # 返回加载的模型和分词器对象
    return model, tokenizer
# 定义函数，用于加载 LLM 适配器和 LlamaCpp 模型参数
def llamacpp_loader(
    llm_adapter: LLMModelAdapter, model_params: LlamaCppModelParameters
):
    try:
        # 尝试导入 LlamaCppModel 类
        from dbgpt.model.llm.llama_cpp.llama_cpp import LlamaCppModel
    except ImportError as exc:
        # 如果导入失败，抛出 ValueError 异常，提示用户安装必要的 Python 包
        raise ValueError(
            "Could not import python package: llama-cpp-python "
            "Please install db-gpt llama support with `cd $DB-GPT-DIR && pip install .[llama_cpp]` "
            "or install llama-cpp-python with `pip install llama-cpp-python`"
        ) from exc
    # 获取模型路径
    model_path = model_params.model_path
    # 使用预训练模型路径和模型参数创建 LlamaCppModel 实例
    model, tokenizer = LlamaCppModel.from_pretrained(model_path, model_params)
    # 返回创建的模型实例和其对应的 tokenizer 实例
    return model, tokenizer


# 定义函数，用于加载 LLM 适配器和代理模型参数
def proxyllm_loader(llm_adapter: LLMModelAdapter, model_params: ProxyModelParameters):
    # 导入 ProxyModel 类
    from dbgpt.model.proxy.llms.proxy_model import ProxyModel

    # 记录信息：加载代理 LLM 模型
    logger.info("Load proxyllm")
    # 使用给定的模型参数创建 ProxyModel 实例
    model = ProxyModel(model_params)
    # 返回创建的模型实例和另一个模型实例（通常是同一个对象的引用）
    return model, model
```