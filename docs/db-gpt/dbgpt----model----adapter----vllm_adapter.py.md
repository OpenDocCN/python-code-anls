# `.\DB-GPT-src\dbgpt\model\adapter\vllm_adapter.py`

```py
import dataclasses
import logging

# 导入自定义模块和类
from dbgpt.model.adapter.base import LLMModelAdapter
from dbgpt.model.adapter.template import ConversationAdapter, ConversationAdapterFactory
from dbgpt.model.base import ModelType
from dbgpt.model.parameter import BaseModelParameters
from dbgpt.util.parameter_utils import (
    _build_parameter_class,
    _extract_parameter_details,
    _get_dataclass_print_str,
)

# 设置日志记录器
logger = logging.getLogger(__name__)


class VLLMModelAdapterWrapper(LLMModelAdapter):
    """Wrapping vllm engine"""

    def __init__(self, conv_factory: ConversationAdapterFactory):
        self.conv_factory = conv_factory

    def new_adapter(self, **kwargs) -> "VLLMModelAdapterWrapper":
        # 创建新的 VLLMModelAdapterWrapper 实例并返回
        return VLLMModelAdapterWrapper(self.conv_factory)

    def model_type(self) -> str:
        # 返回模型类型字符串 "VLLM"
        return ModelType.VLLM

    def model_param_class(self, model_type: str = None) -> BaseModelParameters:
        # 导入必要的模块和类
        import argparse
        from vllm.engine.arg_utils import AsyncEngineArgs

        # 创建参数解析器
        parser = argparse.ArgumentParser()
        # 添加 AsyncEngineArgs 的命令行参数
        parser = AsyncEngineArgs.add_cli_args(parser)
        # 添加自定义的命令行参数
        parser.add_argument("--model_name", type=str, help="model name")
        parser.add_argument(
            "--model_path",
            type=str,
            help="local model path of the huggingface model to use",
        )
        parser.add_argument("--model_type", type=str, help="model type")
        # parser.add_argument("--device", type=str, default=None, help="device")
        # TODO parse prompt templete from `model_name` and `model_path`
        parser.add_argument(
            "--prompt_template",
            type=str,
            default=None,
            help="Prompt template. If None, the prompt template is automatically determined from model path",
        )

        # 提取参数的详细信息，并进行定制化设置
        descs = _extract_parameter_details(
            parser,
            "dbgpt.model.parameter.VLLMModelParameters",
            skip_names=["model"],
            overwrite_default_values={"trust_remote_code": True},
        )
        # 构建参数类并返回
        return _build_parameter_class(descs)
    # 加载模型参数并初始化 AsyncLLMEngine 实例
    def load_from_params(self, params):
        import torch  # 导入 torch 库
        from vllm import AsyncLLMEngine  # 导入 AsyncLLMEngine 类
        from vllm.engine.arg_utils import AsyncEngineArgs  # 导入 AsyncEngineArgs 类

        # 获取当前系统中的 GPU 数量
        num_gpus = torch.cuda.device_count()
        # 如果有多个 GPU 并且 params 对象有 tensor_parallel_size 属性，则设置其值为 GPU 数量
        if num_gpus > 1 and hasattr(params, "tensor_parallel_size"):
            setattr(params, "tensor_parallel_size", num_gpus)
        
        # 记录日志，显示启动 vllm AsyncLLMEngine 的参数
        logger.info(
            f"Start vllm AsyncLLMEngine with args: {_get_dataclass_print_str(params)}"
        )

        # 将 params 对象转换为字典
        params = dataclasses.asdict(params)
        # 将 model_path 键的值赋给 params 字典中的 model 键
        params["model"] = params["model_path"]
        # 获取 AsyncEngineArgs 类中所有属性名列表
        attrs = [attr.name for attr in dataclasses.fields(AsyncEngineArgs)]
        # 根据 attrs 中的属性名从 params 字典中获取对应的值，构建 vllm_engine_args_dict 字典
        vllm_engine_args_dict = {attr: params.get(attr) for attr in attrs}
        # 使用 vllm_engine_args_dict 中的参数创建 AsyncEngineArgs 实例
        engine_args = AsyncEngineArgs(**vllm_engine_args_dict)
        # 根据 engine_args 创建 AsyncLLMEngine 实例
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        # 获取 engine 实例的 tokenizer 属性
        tokenizer = engine.engine.tokenizer
        # 如果 tokenizer 对象有 "tokenizer" 属性，则赋值为该属性值（用于 vllm >= 0.2.7 版本）
        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer
        # 返回 engine 实例和 tokenizer 实例
        return engine, tokenizer

    # 返回是否支持异步生成流的布尔值
    def support_async(self) -> bool:
        return True

    # 获取异步生成流函数
    def get_async_generate_stream_function(self, model, model_path: str):
        from dbgpt.model.llm_out.vllm_llm import generate_stream  # 导入 generate_stream 函数

        # 返回 generate_stream 函数
        return generate_stream

    # 获取默认的对话适配器模板
    def get_default_conv_template(
        self, model_name: str, model_path: str
    ) -> ConversationAdapter:
        # 返回由 conv_factory 根据 model_name 和 model_path 获取的 ConversationAdapter 实例
        return self.conv_factory.get_by_model(model_name, model_path)

    # 返回对象的字符串表示形式，格式为 模块名.类名
    def __str__(self) -> str:
        return "{}.{}".format(self.__class__.__module__, self.__class__.__name__)
```