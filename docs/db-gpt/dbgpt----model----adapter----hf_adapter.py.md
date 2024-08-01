# `.\DB-GPT-src\dbgpt\model\adapter\hf_adapter.py`

```py
import logging  # 导入日志模块，用于记录程序运行信息
from abc import ABC, abstractmethod  # 导入抽象基类和抽象方法装饰器
from typing import Any, Dict, List, Optional  # 导入类型提示相关模块

from dbgpt.core import ModelMessage  # 导入自定义的模型消息类
from dbgpt.model.adapter.base import LLMModelAdapter, register_model_adapter  # 导入模型适配器基类和注册装饰器
from dbgpt.model.base import ModelType  # 导入模型类型枚举

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class NewHFChatModelAdapter(LLMModelAdapter, ABC):
    """Model adapter for new huggingface chat models

    See https://huggingface.co/docs/transformers/main/en/chat_templating

    We can transform the inference chat messages to chat model instead of create a
    prompt template for this model
    """

    trust_remote_code: bool = True  # 布尔类型属性，表示是否信任远程代码，默认为True

    def new_adapter(self, **kwargs) -> "NewHFChatModelAdapter":
        """Create a new instance of the current adapter."""
        return self.__class__()

    def match(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        """Check if the provided model type, name, or path matches the adapter's criteria.

        Args:
            model_type (str): Type of the model (e.g., HF for Hugging Face models).
            model_name (Optional[str]): Name of the model (optional).
            model_path (Optional[str]): Path to the model (optional).

        Returns:
            bool: True if the model matches the criteria, False otherwise.
        """
        if model_type != ModelType.HF:
            return False
        if model_name is None and model_path is None:
            return False
        model_name = model_name.lower() if model_name else None
        model_path = model_path.lower() if model_path else None
        return self.do_match(model_name) or self.do_match(model_path)

    @abstractmethod
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        """Abstract method to be implemented by subclasses.

        Args:
            lower_model_name_or_path (Optional[str]): Lowercased model name or path.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def check_dependencies(self) -> None:
        """Check if the dependencies required for the adapter are installed.

        Raises:
            ValueError: If the required dependencies are not installed.
        """
        try:
            import transformers
        except ImportError as exc:
            raise ValueError(
                "Could not import depend python package "
                "Please install it with `pip install transformers`."
            ) from exc
        self.check_transformer_version(transformers.__version__)

    def check_transformer_version(self, current_version: str) -> None:
        """Check if the current installed version of transformers meets the required version.

        Args:
            current_version (str): Current version of transformers.

        Raises:
            ValueError: If the installed version is lower than required.
        """
        if not current_version >= "4.34.0":
            raise ValueError(
                "Current model (Load by NewHFChatModelAdapter) require transformers.__version__>=4.34.0"
            )
    def load(self, model_path: str, from_pretrained_kwargs: dict):
        try:
            import transformers  # 导入transformers库
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # 导入模型和分词器相关类
        except ImportError as exc:
            raise ValueError(
                "Could not import depend python package "
                "Please install it with `pip install transformers`."
            ) from exc  # 抛出导入失败异常并提示安装transformers库
        self.check_dependencies()  # 检查依赖项是否满足

        logger.info(
            f"Load model from {model_path}, from_pretrained_kwargs: {from_pretrained_kwargs}"
        )  # 记录日志，指示从指定路径加载模型和预训练参数

        revision = from_pretrained_kwargs.get("revision", "main")  # 获取预训练参数中的revision，默认为"main"
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=self.use_fast_tokenizer(),  # 根据配置决定是否使用快速分词器
                revision=revision,
                trust_remote_code=self.trust_remote_code,  # 根据配置决定是否信任远程代码
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,  # 强制禁用快速分词器
                revision=revision,
                trust_remote_code=self.trust_remote_code,
            )
        try:
            if "trust_remote_code" not in from_pretrained_kwargs:
                from_pretrained_kwargs["trust_remote_code"] = self.trust_remote_code
            model = AutoModelForCausalLM.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )  # 使用预训练模型加载CausalLM模型
        except NameError:
            model = AutoModel.from_pretrained(
                model_path, low_cpu_mem_usage=True, **from_pretrained_kwargs
            )  # 使用预训练模型加载一般模型
        # tokenizer.use_default_system_prompt = False  # 禁用默认系统提示
        return model, tokenizer  # 返回加载的模型和分词器对象

    def get_generate_stream_function(self, model, model_path: str):
        """Get the generate stream function of the model"""
        from dbgpt.model.llm_out.hf_chat_llm import huggingface_chat_generate_stream  # 导入生成流函数

        return huggingface_chat_generate_stream  # 返回生成流函数

    def get_str_prompt(
        self,
        params: Dict,
        messages: List[ModelMessage],
        tokenizer: Any,
        prompt_template: str = None,
        convert_to_compatible_format: bool = False,
    ) -> Optional[str]:
        from transformers import AutoTokenizer  # 导入分词器类

        if not tokenizer:
            raise ValueError("tokenizer is is None")  # 如果分词器为空，则抛出数值错误异常
        tokenizer: AutoTokenizer = tokenizer  # 标记分词器的类型为AutoTokenizer

        messages = self.transform_model_messages(messages, convert_to_compatible_format)  # 转换模型消息为兼容格式
        logger.debug(f"The messages after transform: \n{messages}")  # 记录调试信息，显示转换后的消息
        str_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )  # 应用聊天模板到消息，生成字符串提示
        return str_prompt  # 返回生成的字符串提示
class YiAdapter(NewHFChatModelAdapter):
    # 定义一个名为YiAdapter的类，继承自NewHFChatModelAdapter类
    support_4bit: bool = True
    # 定义一个布尔类型的属性support_4bit，并赋值为True
    support_8bit: bool = True
    # 定义一个布尔类型的属性support_8bit，并赋值为True
    support_system_message: bool = True
    # 定义一个布尔类型的属性support_system_message，并赋值为True

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 定义一个方法do_match，接受lower_model_name_or_path参数，默认值为None
        return (
            lower_model_name_or_path
            and "yi-" in lower_model_name_or_path
            and "chat" in lower_model_name_or_path
        )
        # 返回一个布尔值，判断lower_model_name_or_path是否包含"yi-"和"chat"

class Yi15Adapter(YiAdapter):
    # 定义一个名为Yi15Adapter的类，继承自YiAdapter类
    """Yi 1.5 model adapter."""
    # 类的文档字符串

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 定义一个方法do_match，接受lower_model_name_or_path参数，默认值为None
        return (
            lower_model_name_or_path
            and "yi-" in lower_model_name_or_path
            and "1.5" in lower_model_name_or_path
            and "chat" in lower_model_name_or_path
        )
        # 返回一个布尔值，判断lower_model_name_or_path是否包含"yi-", "1.5"和"chat"

    def get_str_prompt(
        self,
        params: Dict,
        messages: List[ModelMessage],
        tokenizer: Any,
        prompt_template: str = None,
        convert_to_compatible_format: bool = False,
    ) -> Optional[str]:
        # 定义一个方法get_str_prompt，接受多个参数，并返回一个可选的字符串
        str_prompt = super().get_str_prompt(
            params,
            messages,
            tokenizer,
            prompt_template,
            convert_to_compatible_format,
        )
        # 调用父类的get_str_prompt方法，并赋值给str_prompt
        terminators = [
            tokenizer.eos_token_id,
        ]
        # 创建一个包含tokenizer.eos_token_id的列表terminators
        exist_token_ids = params.get("stop_token_ids", [])
        # 获取params中"stop_token_ids"对应的值，如果不存在则返回空列表
        terminators.extend(exist_token_ids)
        # 将exist_token_ids中的元素添加到terminators列表中
        params["stop_token_ids"] = terminators
        # 将terminators列表赋值给params中"stop_token_ids"
        return str_prompt
        # 返回str_prompt

class Mixtral8x7BAdapter(NewHFChatModelAdapter):
    # 定义一个名为Mixtral8x7BAdapter的类，继承自NewHFChatModelAdapter类
    """
    https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1
    """
    # 类的文档字符串

    support_4bit: bool = True
    # 定义一个布尔类型的属性support_4bit，并赋值为True
    support_8bit: bool = True
    # 定义一个布尔类型的属性support_8bit，并赋值为True
    support_system_message: bool = False
    # 定义一个布尔类型的属性support_system_message，并赋值为False

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 定义一个方法do_match，接受lower_model_name_or_path参数，默认值为None
        return (
            lower_model_name_or_path
            and "mixtral" in lower_model_name_or_path
            and "8x7b" in lower_model_name_or_path
        )
        # 返回一个布尔值，判断lower_model_name_or_path是否包含"mixtral"和"8x7b"

class SOLARAdapter(NewHFChatModelAdapter):
    # 定义一个名为SOLARAdapter的类，继承自NewHFChatModelAdapter类
    """
    https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0
    """
    # 类的文档字符串

    support_4bit: bool = True
    # 定义一个布尔类型的属性support_4bit，并赋值为True
    support_8bit: bool = False
    # 定义一个布尔类型的属性support_8bit，并赋值为False

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 定义一个方法do_match，接受lower_model_name_or_path参数，默认值为None
        return (
            lower_model_name_or_path
            and "solar-" in lower_model_name_or_path
            and "instruct" in lower_model_name_or_path
        )
        # 返回一个布尔值，判断lower_model_name_or_path是否包含"solar-"和"instruct"

class GemmaAdapter(NewHFChatModelAdapter):
    # 定义一个名为GemmaAdapter的类，继承自NewHFChatModelAdapter类
    """
    https://huggingface.co/google/gemma-7b-it

    TODO: There are problems with quantization.
    """
    # 类的文档字符串

    support_4bit: bool = False
    # 定义一个布尔类型的属性support_4bit，并赋值为False
    support_8bit: bool = False
    # 定义一个布尔类型的属性support_8bit，并赋值为False
    support_system_message: bool = False
    # 定义一个布尔类型的属性support_system_message，并赋值为False

    def check_transformer_version(self, current_version: str) -> None:
        # 定义一个方法check_transformer_version，接受current_version参数，并返回None
        if not current_version >= "4.38.0":
            # 如果current_version小于"4.38.0"
            raise ValueError(
                "Gemma require transformers.__version__>=4.38.0, please upgrade your transformers package."
            )
            # 抛出一个值错误异常，提示需要升级transformers包
    # 定义一个方法 `do_match`，接受一个可选的字符串参数 `lower_model_name_or_path`
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 返回一个布尔值，判断参数 `lower_model_name_or_path` 是否满足以下条件：
        # 1. 参数不为 None
        # 2. 字符串 "gemma-" 存在于参数中
        # 3. 字符串 "it" 存在于参数中
        return (
            lower_model_name_or_path
            and "gemma-" in lower_model_name_or_path
            and "it" in lower_model_name_or_path
        )
class Gemma2Adapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/google/gemma-2-27b-it
    https://huggingface.co/google/gemma-2-9b-it
    """

    # 是否支持4位精度模型
    support_4bit: bool = True
    # 是否支持8位精度模型
    support_8bit: bool = True
    # 是否支持系统消息
    support_system_message: bool = False

    # 返回使用快速分词器的布尔值
    def use_fast_tokenizer(self) -> bool:
        return True

    # 检查变压器版本是否符合要求
    def check_transformer_version(self, current_version: str) -> None:
        if not current_version >= "4.42.1":
            raise ValueError(
                "Gemma2 require transformers.__version__>=4.42.1, please upgrade your transformers package."
            )

    # 检查模型名或路径是否与适配器匹配
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "gemma-2-" in lower_model_name_or_path
            and "it" in lower_model_name_or_path
        )

    # 加载模型
    def load(self, model_path: str, from_pretrained_kwargs: dict):
        import torch

        # 如果未提供from_pretrained_kwargs，则初始化为空字典
        if not from_pretrained_kwargs:
            from_pretrained_kwargs = {}
        # 设置torch_dtype参数为torch.bfloat16
        from_pretrained_kwargs["torch_dtype"] = torch.bfloat16
        # 调用父类方法加载模型和分词器
        model, tokenizer = super().load(model_path, from_pretrained_kwargs)
        return model, tokenizer


class StarlingLMAdapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/Nexusflow/Starling-LM-7B-beta
    """

    # 是否支持4位精度模型
    support_4bit: bool = True
    # 是否支持8位精度模型
    support_8bit: bool = True
    # 是否支持系统消息
    support_system_message: bool = False

    # 检查模型名或路径是否与适配器匹配
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "starling-" in lower_model_name_or_path
            and "lm" in lower_model_name_or_path
        )

    # 获取字符串提示信息
    def get_str_prompt(
        self,
        params: Dict,
        messages: List[ModelMessage],
        tokenizer: Any,
        prompt_template: str = None,
        convert_to_compatible_format: bool = False,
    ):
        # 实现获取字符串提示信息的方法，这里没有具体实现内容
        pass
    ) -> Optional[str]:
        # 调用父类方法，获取字符串提示信息
        str_prompt = super().get_str_prompt(
            params,
            messages,
            tokenizer,
            prompt_template,
            convert_to_compatible_format,
        )
        # 初始化对话模式变量为 None
        chat_mode = None
        # 检查参数中是否包含上下文，并且上下文中是否包含对话模式信息
        if params and "context" in params and "chat_mode" in params["context"]:
            # 获取对话模式信息
            chat_mode = params["context"].get("chat_mode")
        # 如果对话模式属于以下任一类型
        if chat_mode in [
            "chat_dashboard",
            "chat_with_db_execute",
            "excel_learning",
            "chat_excel",
        ]:
            # 使用代码提示作为对话内容
            # 这是一个临时解决方案，应该使用更好的方法来区分对话类型
            # 参考链接：https://huggingface.co/Nexusflow/Starling-LM-7B-beta#code-examples
            str_prompt = str_prompt.replace("GPT4 Correct User:", "Code User:").replace(
                "GPT4 Correct Assistant:", "Code Assistant:"
            )
            # 记录日志信息，指示使用代码提示进行对话，并转换提示文本
            logger.info(
                f"Use code prompt for chat_mode: {chat_mode}, transform 'GPT4 Correct User:' to 'Code User:' "
                "and 'GPT4 Correct Assistant:' to 'Code Assistant:'"
            )
        # 返回最终的字符串提示信息
        return str_prompt
class QwenAdapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/Qwen/Qwen1.5-32B-Chat

    TODO: There are problems with quantization.
    """

    # 是否支持4位量化
    support_4bit: bool = True
    # 是否支持8位量化
    support_8bit: bool = False  # TODO: Support 8bit quantization

    # 检查Transformer版本是否符合要求
    def check_transformer_version(self, current_version: str) -> None:
        # 如果当前版本小于4.37.0，抛出数值错误异常
        if not current_version >= "4.37.0":
            raise ValueError(
                "Qwen 1.5 require transformers.__version__>=4.37.0, please upgrade your transformers package."
            )

    # 判断是否匹配特定模型名或路径
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "qwen" in lower_model_name_or_path
            and "1.5" in lower_model_name_or_path
            and "moe" not in lower_model_name_or_path
            and "qwen2" not in lower_model_name_or_path
        )


class Qwen2Adapter(QwenAdapter):
    # 是否支持4位量化
    support_4bit: bool = True
    # 是否支持8位量化
    support_8bit: bool = True

    # 判断是否匹配特定模型名或路径
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "qwen2" in lower_model_name_or_path
            and "instruct" in lower_model_name_or_path
        )


class QwenMoeAdapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B

    TODO: There are problems with quantization.
    """

    # 是否支持4位量化
    support_4bit: bool = False
    # 是否支持8位量化
    support_8bit: bool = False

    # 检查Transformer版本是否符合要求
    def check_transformer_version(self, current_version: str) -> None:
        # 打印当前版本信息
        print(f"Checking version: Current version {current_version}")
        # 如果当前版本小于4.40.0，抛出数值错误异常
        if not current_version >= "4.40.0":
            raise ValueError(
                "Qwen 1.5 Moe require transformers.__version__>=4.40.0, please upgrade your transformers package."
            )

    # 判断是否匹配特定模型名或路径
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "qwen" in lower_model_name_or_path
            and "1.5" in lower_model_name_or_path
            and "moe" in lower_model_name_or_path
        )


class Llama3Adapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
    https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct
    """

    # 是否支持4位量化
    support_4bit: bool = True
    # 是否支持8位量化
    support_8bit: bool = True

    # 判断是否匹配特定模型名或路径
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "llama-3" in lower_model_name_or_path
            and "instruct" in lower_model_name_or_path
            and "3.1" not in lower_model_name_or_path
        )

    # 获取字符串提示信息
    def get_str_prompt(
        self,
        params: Dict,
        messages: List[ModelMessage],
        tokenizer: Any,
        prompt_template: str = None,
        convert_to_compatible_format: bool = False,
    ):
    ) -> Optional[str]:
        # 调用父类方法获取字符串提示信息，传入参数、消息、分词器、提示模板及格式转换选项
        str_prompt = super().get_str_prompt(
            params,
            messages,
            tokenizer,
            prompt_template,
            convert_to_compatible_format,
        )
        # 设置终止符号列表，包括分词器的结束符号和指定的特殊结束符号
        terminators = [
            tokenizer.eos_token_id,  # 分词器的结束符号的 ID
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # 将特殊结束标记转换为 ID
        ]
        exist_token_ids = params.get("stop_token_ids", [])
        terminators.extend(exist_token_ids)  # 将已存在的终止标记 ID 加入终止符号列表
        # TODO(fangyinc): 将来应修改参数
        params["stop_token_ids"] = terminators  # 将更新后的终止符号列表写入参数中
        # 返回生成的字符串提示信息
        return str_prompt
class Llama31Adapter(Llama3Adapter):
    # 继承自 Llama3Adapter 的 Llama3.1 适配器类

    def check_transformer_version(self, current_version: str) -> None:
        # 检查 transformer 版本，记录日志当前版本信息
        logger.info(f"Checking transformers version: Current version {current_version}")
        # 如果当前版本小于 "4.43.0"，抛出数值错误异常
        if not current_version >= "4.43.0":
            raise ValueError(
                "Llama-3.1 require transformers.__version__>=4.43.0, please upgrade your transformers package."
            )

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查是否匹配 Llama-3.1 模型名和 instruct 关键字
        return (
            lower_model_name_or_path
            and "llama-3.1" in lower_model_name_or_path
            and "instruct" in lower_model_name_or_path
        )


class DeepseekV2Adapter(NewHFChatModelAdapter):
    # 继承自 NewHFChatModelAdapter 的 DeepseekV2 适配器类
    support_4bit: bool = False
    support_8bit: bool = False

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查是否匹配 deepseek v2 chat 模型名
        return (
            lower_model_name_or_path
            and "deepseek" in lower_model_name_or_path
            and "v2" in lower_model_name_or_path
            and "chat" in lower_model_name_or_path
        )

    def load(self, model_path: str, from_pretrained_kwargs: dict):
        # 加载模型和 tokenizer，如果 from_pretrained_kwargs 为空则初始化为空字典
        if not from_pretrained_kwargs:
            from_pretrained_kwargs = {}
        # 如果 from_pretrained_kwargs 中没有 trust_remote_code 关键字，则设置为 True
        if "trust_remote_code" not in from_pretrained_kwargs:
            from_pretrained_kwargs["trust_remote_code"] = True
        # 调用父类的 load 方法加载模型和 tokenizer
        model, tokenizer = super().load(model_path, from_pretrained_kwargs)

        # 导入 GenerationConfig 类
        from transformers import GenerationConfig

        # 使用模型路径初始化 GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        # 将 pad_token_id 设置为 eos_token_id
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model, tokenizer


class DeepseekCoderV2Adapter(DeepseekV2Adapter):
    # 继承自 DeepseekV2Adapter 的 Deepseek Coder V2 适配器类

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查是否匹配 deepseek coder v2 instruct 模型名
        return (
            lower_model_name_or_path
            and "deepseek" in lower_model_name_or_path
            and "coder" in lower_model_name_or_path
            and "v2" in lower_model_name_or_path
            and "instruct" in lower_model_name_or_path
        )


class SailorAdapter(QwenAdapter):
    """
    https://huggingface.co/sail/Sailor-14B-Chat
    """
    # SailorAdapter 类，继承自 QwenAdapter

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查是否匹配 sailor chat 模型名
        return (
            lower_model_name_or_path
            and "sailor" in lower_model_name_or_path
            and "chat" in lower_model_name_or_path
        )


class PhiAdapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/microsoft/Phi-3-medium-128k-instruct
    """
    # PhiAdapter 类，继承自 NewHFChatModelAdapter

    support_4bit: bool = True
    support_8bit: bool = True
    support_system_message: bool = False

    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        # 检查是否匹配 phi-3 instruct 模型名
        return (
            lower_model_name_or_path
            and "phi-3" in lower_model_name_or_path
            and "instruct" in lower_model_name_or_path
        )
    # 载入模型的方法，从指定路径加载模型，并使用预设参数字典中的参数
    def load(self, model_path: str, from_pretrained_kwargs: dict):
        # 如果未提供预设参数字典，则初始化为空字典
        if not from_pretrained_kwargs:
            from_pretrained_kwargs = {}
        # 如果预设参数字典中不包含"trust_remote_code"参数，则设置为True
        if "trust_remote_code" not in from_pretrained_kwargs:
            from_pretrained_kwargs["trust_remote_code"] = True
        # 调用父类的load方法，加载模型并返回
        return super().load(model_path, from_pretrained_kwargs)

    # 获取字符串提示信息的方法，返回可选的字符串提示
    def get_str_prompt(
        self,
        params: Dict,
        messages: List[ModelMessage],
        tokenizer: Any,
        prompt_template: str = None,
        convert_to_compatible_format: bool = False,
    ) -> Optional[str]:
        # 调用父类的get_str_prompt方法，获取基础的字符串提示信息
        str_prompt = super().get_str_prompt(
            params,
            messages,
            tokenizer,
            prompt_template,
            convert_to_compatible_format,
        )
        # 向params参数字典中添加自定义停用词列表
        params["custom_stop_words"] = ["<|end|>"]
        # 返回获取的字符串提示信息
        return str_prompt
class SQLCoderAdapter(Llama3Adapter):
    """
    https://huggingface.co/defog/llama-3-sqlcoder-8b
    """

    # 检查给定的模型名或路径是否包含 "llama-3" 和 "sqlcoder"
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "llama-3" in lower_model_name_or_path
            and "sqlcoder" in lower_model_name_or_path
        )


class OpenChatAdapter(Llama3Adapter):
    """
    https://huggingface.co/openchat/openchat-3.6-8b-20240522
    """

    support_4bit: bool = True
    support_8bit: bool = True

    # 检查给定的模型名或路径是否包含 "openchat" 和 "3.6"
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "openchat" in lower_model_name_or_path
            and "3.6" in lower_model_name_or_path
        )


class GLM4Adapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/THUDM/glm-4-9b-chat
    """

    # 检查给定的模型名或路径是否包含 "glm-4" 和 "chat"
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "glm-4" in lower_model_name_or_path
            and "chat" in lower_model_name_or_path
        )


class Codegeex4Adapter(GLM4Adapter):
    """
    https://huggingface.co/THUDM/codegeex4-all-9b
    """

    # 检查给定的模型名或路径是否包含 "codegeex4"
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return lower_model_name_or_path and "codegeex4" in lower_model_name_or_path

    # 加载指定路径的模型，并设置从预训练参数中获取的选项
    def load(self, model_path: str, from_pretrained_kwargs: dict):
        if not from_pretrained_kwargs:
            from_pretrained_kwargs = {}
        # 如果未设置 "trust_remote_code" 参数，则默认设置为 True
        if "trust_remote_code" not in from_pretrained_kwargs:
            from_pretrained_kwargs["trust_remote_code"] = True
        # 调用父类的 load 方法加载模型
        return super().load(model_path, from_pretrained_kwargs)


class Internlm2Adapter(NewHFChatModelAdapter):
    """
    https://huggingface.co/internlm/internlm2_5-7b-chat
    """

    # 检查给定的模型名或路径是否包含 "internlm2" 和 "chat"
    def do_match(self, lower_model_name_or_path: Optional[str] = None):
        return (
            lower_model_name_or_path
            and "internlm2" in lower_model_name_or_path
            and "chat" in lower_model_name_or_path
        )

    # 加载指定路径的模型，并设置从预训练参数中获取的选项
    def load(self, model_path: str, from_pretrained_kwargs: dict):
        if not from_pretrained_kwargs:
            from_pretrained_kwargs = {}
        # 如果未设置 "trust_remote_code" 参数，则默认设置为 True
        if "trust_remote_code" not in from_pretrained_kwargs:
            from_pretrained_kwargs["trust_remote_code"] = True
        # 调用父类的 load 方法加载模型
        return super().load(model_path, from_pretrained_kwargs)


# 以下代码用于注册模型适配器
# 最后注册的模型适配器首先匹配
register_model_adapter(YiAdapter)
register_model_adapter(Yi15Adapter)
register_model_adapter(Mixtral8x7BAdapter)
register_model_adapter(SOLARAdapter)
register_model_adapter(GemmaAdapter)
register_model_adapter(Gemma2Adapter)
register_model_adapter(StarlingLMAdapter)
register_model_adapter(QwenAdapter)
register_model_adapter(QwenMoeAdapter)
register_model_adapter(Llama3Adapter)
register_model_adapter(Llama31Adapter)
register_model_adapter(DeepseekV2Adapter)
# 注册不同的模型适配器，用于特定模型的适配处理
register_model_adapter(DeepseekCoderV2Adapter)
register_model_adapter(SailorAdapter)
register_model_adapter(PhiAdapter)
register_model_adapter(SQLCoderAdapter)
register_model_adapter(OpenChatAdapter)
register_model_adapter(GLM4Adapter)
register_model_adapter(Codegeex4Adapter)
register_model_adapter(Qwen2Adapter)
register_model_adapter(Internlm2Adapter)
```