# `.\DB-GPT-src\dbgpt\storage\cache\operators.py`

```py
"""Operators for processing model outputs with caching support."""
# 导入日志模块
import logging
# 导入类型提示模块
from typing import AsyncIterator, Dict, List, Optional, Union, cast
# 导入自定义模块
from dbgpt.core import ModelOutput, ModelRequest
from dbgpt.core.awel import (
    BaseOperator,
    BranchFunc,
    BranchOperator,
    MapOperator,
    StreamifyAbsOperator,
    TransformStreamAbsOperator,
)
# 导入缓存相关模块
from .llm_cache import LLMCacheClient, LLMCacheKey, LLMCacheValue
from .manager import CacheManager

# 获取日志记录器
logger = logging.getLogger(__name__)

# 定义常量
_LLM_MODEL_INPUT_VALUE_KEY = "llm_model_input_value"
_LLM_MODEL_OUTPUT_CACHE_KEY = "llm_model_output_cache"

# 定义流式处理带缓存支持的操作符
class CachedModelStreamOperator(StreamifyAbsOperator[ModelRequest, ModelOutput]):
    """Operator for streaming processing of model outputs with caching.

    Args:
        cache_manager (CacheManager): The cache manager to handle caching operations.
        **kwargs: Additional keyword arguments.

    Methods:
        streamify: Processes a stream of inputs with cache support, yielding model outputs.
    """

    def __init__(self, cache_manager: CacheManager, **kwargs) -> None:
        """Create a new instance of CachedModelStreamOperator."""
        super().__init__(**kwargs)
        self._cache_manager = cache_manager
        self._client = LLMCacheClient(cache_manager)

    async def streamify(self, input_value: ModelRequest):
        """Process inputs as a stream with cache support and yield model outputs.

        Args:
            input_value (ModelRequest): The input value for the model.

        Returns:
            AsyncIterator[ModelOutput]: An asynchronous iterator of model outputs.
        """
        # 解析缓存键字典
        cache_dict = _parse_cache_key_dict(input_value)
        # 创建新的缓存键
        llm_cache_key: LLMCacheKey = self._client.new_key(**cache_dict)
        # 获取缓存值
        llm_cache_value = await self._client.get(llm_cache_key)
        logger.info(f"llm_cache_value: {llm_cache_value}")
        if not llm_cache_value:
            raise ValueError(f"Cache value not found for key: {llm_cache_key}")
        outputs = cast(List[ModelOutput], llm_cache_value.get_value().output)
        for out in outputs:
            yield cast(ModelOutput, out)

# 定义映射处理带缓存支持的操作符
class CachedModelOperator(MapOperator[ModelRequest, ModelOutput]):
    """Operator for map-based processing of model outputs with caching.

    Args:
        cache_manager (CacheManager): Manager for caching operations.
        **kwargs: Additional keyword arguments.

    Methods:
        map: Processes a single input with cache support and returns the model output.
    """

    def __init__(self, cache_manager: CacheManager, **kwargs) -> None:
        """Create a new instance of CachedModelOperator."""
        super().__init__(**kwargs)
        self._cache_manager = cache_manager
        self._client = LLMCacheClient(cache_manager)
    # 异步方法，用于处理单个输入值并支持缓存，返回模型输出
    async def map(self, input_value: ModelRequest) -> ModelOutput:
        """Process a single input with cache support and return the model output.

        Args:
            input_value (ModelRequest): The input value for the model.

        Returns:
            ModelOutput: The output from the model.
        """
        # 解析输入值生成缓存键的字典
        cache_dict = _parse_cache_key_dict(input_value)
        # 使用缓存键字典创建新的缓存键对象
        llm_cache_key: LLMCacheKey = self._client.new_key(**cache_dict)
        # 异步获取缓存中的值
        llm_cache_value = await self._client.get(llm_cache_key)
        # 如果缓存中没有找到值，则抛出异常
        if not llm_cache_value:
            raise ValueError(f"Cache value not found for key: {llm_cache_key}")
        # 记录日志，显示缓存值信息
        logger.info(f"llm_cache_value: {llm_cache_value}")
        # 返回模型输出，强制类型转换为 ModelOutput 类型
        return cast(ModelOutput, llm_cache_value.get_value().output)
class ModelCacheBranchOperator(BranchOperator[ModelRequest, Dict]):
    """Branch operator for model processing with cache support.

    A branch operator that decides whether to use cached data or to process data using
    the model.

    Args:
        cache_manager (CacheManager): The cache manager for managing cache operations.
        model_task_name (str): The name of the task to process data using the model.
        cache_task_name (str): The name of the task to process data using the cache.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        model_task_name: str,
        cache_task_name: str,
        **kwargs,
    ):
        """Create a new instance of ModelCacheBranchOperator."""
        # 调用父类的构造函数，初始化分支操作符
        super().__init__(branches=None, **kwargs)
        # 设置缓存管理器
        self._cache_manager = cache_manager
        # 创建缓存客户端
        self._client = LLMCacheClient(cache_manager)
        # 设置模型任务名称
        self._model_task_name = model_task_name
        # 设置缓存任务名称
        self._cache_task_name = cache_task_name

    async def branches(
        self,
    ) -> Dict[BranchFunc[ModelRequest], Union[BaseOperator, str]]:
        """Branch logic based on cache availability.

        Defines branch logic based on cache availability.

        Returns:
            Dict[BranchFunc[Dict], Union[BaseOperator, str]]: A dictionary mapping
                branch functions to task names.
        """

        async def check_cache_true(input_value: ModelRequest) -> bool:
            # Check if the cache contains the result for the given input
            if input_value.context and not input_value.context.cache_enable:
                return False
            # 解析缓存键的字典
            cache_dict = _parse_cache_key_dict(input_value)
            # 创建缓存键对象
            cache_key: LLMCacheKey = self._client.new_key(**cache_dict)
            # 从缓存中获取值
            cache_value = await self._client.get(cache_key)
            # 记录调试日志，包括缓存键、哈希键和缓存值
            logger.debug(
                f"cache_key: {cache_key}, hash key: {hash(cache_key)}, cache_value: "
                f"{cache_value}"
            )
            # 将输入值的缓存键保存到共享数据中
            await self.current_dag_context.save_to_share_data(
                _LLM_MODEL_INPUT_VALUE_KEY, cache_key, overwrite=True
            )
            return bool(cache_value)

        async def check_cache_false(input_value: ModelRequest):
            # 反向检查，如果缓存未命中则返回真
            return not await check_cache_true(input_value)

        # 返回一个字典，将检查函数映射到任务名称
        return {
            check_cache_true: self._cache_task_name,
            check_cache_false: self._model_task_name,
        }


class ModelStreamSaveCacheOperator(
    TransformStreamAbsOperator[ModelOutput, ModelOutput]
):
    """An operator to save the stream of model outputs to cache.

    Args:
        cache_manager (CacheManager): The cache manager for handling cache operations.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, cache_manager: CacheManager, **kwargs):
        """Create a new instance of ModelStreamSaveCacheOperator."""
        # 初始化方法，创建一个新的 ModelStreamSaveCacheOperator 实例。
        self._cache_manager = cache_manager
        # 设置缓存管理器属性
        self._client = LLMCacheClient(cache_manager)
        # 使用缓存管理器创建一个 LLMCacheClient 客户端
        super().__init__(**kwargs)
        # 调用父类的初始化方法，传递任意关键字参数

    async def transform_stream(self, input_value: AsyncIterator[ModelOutput]):
        """Save the stream of model outputs to cache.

        Transforms the input stream by saving the outputs to cache.

        Args:
            input_value (AsyncIterator[ModelOutput]): An asynchronous iterator of model
                outputs.

        Returns:
            AsyncIterator[ModelOutput]: The same input iterator, but the outputs are
                saved to cache.
        """
        llm_cache_key: Optional[LLMCacheKey] = None
        # 初始化 LLMCacheKey 变量，用于存储缓存键
        outputs = []
        # 创建一个空列表用于存储模型输出
        async for out in input_value:
            # 异步迭代输入的模型输出
            if not llm_cache_key:
                llm_cache_key = await self.current_dag_context.get_from_share_data(
                    _LLM_MODEL_INPUT_VALUE_KEY
                )
                # 如果缓存键为空，则从共享数据中获取当前 DAG 上下文的输入模型值键
            outputs.append(out)
            # 将当前输出添加到输出列表中
            yield out
            # 返回当前输出
        if llm_cache_key and _is_success_model_output(outputs):
            # 如果存在缓存键且模型输出成功
            llm_cache_value: LLMCacheValue = self._client.new_value(output=outputs)
            # 创建新的 LLMCacheValue 对象，用于存储输出
            await self._client.set(llm_cache_key, llm_cache_value)
            # 将缓存键和缓存值存储到缓存客户端中
class ModelSaveCacheOperator(MapOperator[ModelOutput, ModelOutput]):
    """An operator to save a single model output to cache.

    Args:
        cache_manager (CacheManager): The cache manager for handling cache operations.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, cache_manager: CacheManager, **kwargs):
        """Create a new instance of ModelSaveCacheOperator."""
        self._cache_manager = cache_manager  # 设置缓存管理器实例变量
        self._client = LLMCacheClient(cache_manager)  # 初始化缓存客户端
        super().__init__(**kwargs)  # 调用父类构造函数初始化

    async def map(self, input_value: ModelOutput) -> ModelOutput:
        """Save model output to cache.

        Args:
            input_value (ModelOutput): The output from the model to be cached.

        Returns:
            ModelOutput: The same input model output.
        """
        llm_cache_key: LLMCacheKey = await self.current_dag_context.get_from_share_data(
            _LLM_MODEL_INPUT_VALUE_KEY
        )  # 从共享数据中获取缓存键
        llm_cache_value: LLMCacheValue = self._client.new_value(output=input_value)  # 创建新的缓存数值对象
        if llm_cache_key and _is_success_model_output(input_value):
            await self._client.set(llm_cache_key, llm_cache_value)  # 将缓存键值对设置到缓存中
        return input_value


def _parse_cache_key_dict(input_value: ModelRequest) -> Dict:
    """Parse and extract relevant fields from input to form a cache key dictionary.

    Args:
        input_value (Dict): The input dictionary containing model and prompt parameters.

    Returns:
        Dict: A dictionary used for generating cache keys.
    """
    prompt: str = input_value.messages_to_string().strip()  # 将输入的消息转换为字符串并去除两端空白
    return {
        "prompt": prompt,  # 缓存键字典中的提示字段
        "model_name": input_value.model,  # 缓存键字典中的模型名称字段
        "temperature": input_value.temperature,  # 缓存键字典中的温度字段
        "max_new_tokens": input_value.max_new_tokens,  # 缓存键字典中的最大新标记数字段
        # "top_p": input_value.get("top_p", "1.0"),  # 可选字段，缓存键字典中的top_p字段，默认为"1.0"
        # TODO pass model_type
        # "model_type": input_value.get("model_type", "huggingface"),  # 可选字段，缓存键字典中的模型类型字段，默认为"huggingface"
    }


def _is_success_model_output(out: Union[Dict, ModelOutput, List[ModelOutput]]) -> bool:
    if not out:
        return False  # 如果输出为空，则返回False
    if isinstance(out, list):
        # check last model output
        out = out[-1]  # 获取列表中的最后一个模型输出
    error_code = 0
    if isinstance(out, ModelOutput):
        error_code = out.error_code  # 获取模型输出对象的错误代码
    else:
        error_code = int(out.get("error_code", 0))  # 获取字典形式的输出的错误代码，转换为整数
    return error_code == 0  # 如果错误代码为0，则认为模型输出成功，返回True
```