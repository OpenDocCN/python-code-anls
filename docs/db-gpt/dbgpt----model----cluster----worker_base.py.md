# `.\DB-GPT-src\dbgpt\model\cluster\worker_base.py`

```py
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Type

from dbgpt.core import ModelMetadata, ModelOutput  # 导入模型元数据和模型输出相关的模块
from dbgpt.model.parameter import ModelParameters, WorkerType  # 导入模型参数和工作类型相关的模块
from dbgpt.util.parameter_utils import ParameterDescription, _get_parameter_descriptions  # 导入参数描述和获取参数描述相关的模块


class ModelWorker(ABC):
    """
    Abstract representation of a Model Worker responsible for model interaction, startup, and shutdown. Supports 'llm' and 'text2vec' models.
    """

    def worker_type(self) -> WorkerType:
        """Return the type of worker as LLM."""
        return WorkerType.LLM  # 返回工作类型为LLM

    def model_param_class(self) -> Type:
        """Return the class representing model parameters."""
        return ModelParameters  # 返回表示模型参数的类

    def support_async(self) -> bool:
        """
        Whether support async, if True, invoke async_generate_stream, async_generate and async_embeddings instead of generate_stream, generate and embeddings
        """
        return False  # 不支持异步操作，返回False

    @abstractmethod
    def parse_parameters(self, command_args: List[str] = None) -> ModelParameters:
        """
        Parse the parameters using the provided command arguments.

        Args:
            command_args (List[str]): The command-line arguments. Default is sys.argv[1:].
        """
        pass  # 解析命令行参数以获取模型参数，子类需要实现具体逻辑

    @abstractmethod
    def load_worker(self, model_name: str, model_path: str, **kwargs) -> None:
        """Load the worker with the specified model name and path."""
        pass  # 使用指定的模型名称和路径加载工作器，子类需要实现具体逻辑

    @abstractmethod
    def start(
        self, model_params: ModelParameters = None, command_args: List[str] = None
    ) -> None:
        """Start the model worker"""
        pass  # 启动模型工作者，子类需要实现具体逻辑

    @abstractmethod
    def stop(self) -> None:
        """Stop the model worker and clean up all the resources used."""
        pass  # 停止模型工作者并清理所有使用的资源，子类需要实现具体逻辑

    def restart(
        self, model_params: ModelParameters = None, command_args: List[str] = None
    ) -> None:
        """Restart the model worker."""
        self.stop()  # 调用停止方法停止模型工作者
        self.start(model_params, command_args)  # 调用启动方法重新启动模型工作者

    def parameter_descriptions(self) -> List[ParameterDescription]:
        """Fetch the parameter configuration information for the current model."""
        param_cls = self.model_param_class()  # 获取当前模型参数类
        return _get_parameter_descriptions(param_cls)  # 调用获取参数描述信息的函数并返回结果

    @abstractmethod
    def generate_stream(self, params: Dict) -> Iterator[ModelOutput]:
        """Generate a stream based on provided parameters.

        Args:
            params (Dict): Parameters matching the PromptRequest data class format. Example:
                {
                    "messages": [{"role": "user", "content": "Hello world"}],  # List of ModelMessage objects
                    "model": "vicuna-13b-v1.5",
                    "prompt": "Hello world",
                    "temperature": 0.7,  # Optional; float value between 0 and 1
                    "max_new_tokens": 2048,  # Optional; max number of new tokens for the output
                    "stop": "#",  # Optional; stopping condition for the output
                    "echo": True  # Optional; whether to echo the input in the output
                }

        Returns:
            Iterator[ModelOutput]: Stream of model outputs.
        """
        
    async def async_generate_stream(self, params: Dict) -> Iterator[ModelOutput]:
        """Asynchronously generate a stream based on provided parameters."""
        raise NotImplementedError
        # 抛出未实现错误，因为该方法应在子类中实现

    @abstractmethod
    def generate(self, params: Dict) -> ModelOutput:
        """Generate output (non-stream) based on provided parameters."""
        # 抽象方法声明，需要在子类中实现

    async def async_generate(self, params: Dict) -> ModelOutput:
        """Asynchronously generate output (non-stream) based on provided parameters."""
        raise NotImplementedError
        # 抛出未实现错误，因为该方法应在子类中实现

    @abstractmethod
    def count_token(self, prompt: str) -> int:
        """Count token of prompt
        Args:
            prompt (str): prompt

        Returns:
            int: token count
        """
        # 抽象方法声明，需要在子类中实现

    async def async_count_token(self, prompt: str) -> int:
        """Asynchronously count token of prompt
        Args:
            prompt (str): prompt

        Returns:
            int: token count
        """
        raise NotImplementedError
        # 抛出未实现错误，因为该方法应在子类中实现

    @abstractmethod
    def get_model_metadata(self, params: Dict) -> ModelMetadata:
        """Get model metadata

        Args:
            params (Dict): parameters, eg. {"model": "vicuna-13b-v1.5"}
        """
        # 抽象方法声明，需要在子类中实现

    async def async_get_model_metadata(self, params: Dict) -> ModelMetadata:
        """Asynchronously get model metadata

        Args:
            params (Dict): parameters, eg. {"model": "vicuna-13b-v1.5"}
        """
        raise NotImplementedError
        # 抛出未实现错误，因为该方法应在子类中实现

    @abstractmethod
    def embeddings(self, params: Dict) -> List[List[float]]:
        """
        Return embeddings for the given input parameters.

        Args:
            params (Dict): Parameters matching the EmbeddingsRequest data class format. Example:
                {
                    "model": "text2vec-large-chinese",
                    "input": ["Hello world", "DB-GPT is amazing"]
                }

        Returns:
            List[List[float]]: List of embeddings corresponding to each input string.
        """
        # 抽象方法声明，需要在子类中实现
    # 定义一个异步方法 async_embeddings，接收一个名为 params 的字典类型参数，
    # 返回一个包含列表的列表，每个内部列表包含浮点数作为嵌入向量。
    async def async_embeddings(self, params: Dict) -> List[List[float]]:
        """Return embeddings asynchronously for the given input parameters."""
        # 抛出 NotImplementedError 异常，表明该方法需要在子类中实现具体逻辑。
        raise NotImplementedError
```