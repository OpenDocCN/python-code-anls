# `.\DB-GPT-src\dbgpt\model\proxy\base.py`

```py
        # 导入所需模块和类
        from __future__ import annotations

        import logging
        from abc import ABC, abstractmethod
        from concurrent.futures import Executor, ThreadPoolExecutor
        from functools import cache
        from typing import TYPE_CHECKING, AsyncIterator, Iterator, List, Optional
        
        # 导入 DBGPT 相关模块和类
        from dbgpt.core import (
            LLMClient,
            MessageConverter,
            ModelMetadata,
            ModelOutput,
            ModelRequest,
        )
        from dbgpt.model.parameter import ProxyModelParameters
        from dbgpt.util.executor_utils import blocking_func_to_async
        
        # 如果在类型检查环境中，则导入 tiktoken
    ):
        self.model_names = model_names
        self.context_length = context_length
        self.executor = executor or ThreadPoolExecutor()
        self.proxy_tokenizer = proxy_tokenizer or TiktokenProxyTokenizer()


        # 初始化对象的属性：模型名称列表和上下文长度
        self.model_names = model_names
        self.context_length = context_length
        # 初始化线程池执行器，如果未提供则使用默认线程池
        self.executor = executor or ThreadPoolExecutor()
        # 初始化代理分词器，如果未提供则使用默认的 TiktokenProxyTokenizer
        self.proxy_tokenizer = proxy_tokenizer or TiktokenProxyTokenizer()



    @classmethod
    @abstractmethod
    def new_client(
        cls,
        model_params: ProxyModelParameters,
        default_executor: Optional[Executor] = None,
    ) -> "ProxyLLMClient":
        """Create a new client instance from model parameters.

        Args:
            model_params (ProxyModelParameters): model parameters
            default_executor (Executor): default executor, If your model is blocking,
                you should pass a ThreadPoolExecutor.
        """


        # 抽象类方法：创建一个新的客户端实例，使用给定的模型参数

        Args:
            # 模型参数对象，类型为 ProxyModelParameters
            model_params (ProxyModelParameters): model parameters
            # 默认的执行器，如果您的模型是阻塞的，应传入一个 ThreadPoolExecutor
            default_executor (Executor): default executor, If your model is blocking,
                you should pass a ThreadPoolExecutor.



    async def generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        """Generate model output from model request.

        We strongly recommend you to implement this method instead of sync_generate for high performance.

        Args:
            request (ModelRequest): model request
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            ModelOutput: model output
        """


        # 异步方法：从模型请求生成模型输出

        Args:
            # 模型请求对象，类型为 ModelRequest
            request (ModelRequest): model request
            # 消息转换器，可选参数，用于将消息转换为特定格式。默认为 None
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            # 模型输出对象，类型为 ModelOutput
            ModelOutput: model output



        return await blocking_func_to_async(
            self.executor, self.sync_generate, request, message_converter
        )


        # 调用异步函数，将同步生成方法转为异步执行

        self.executor: 执行器实例，用于异步执行任务
        self.sync_generate: 同步生成方法，将在异步环境中执行
        request: 调用生成的请求对象
        message_converter: 可选的消息转换器，如果未提供则为 None



    def sync_generate(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelOutput:
        """Generate model output from model request.

        Args:
            request (ModelRequest): model request
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            ModelOutput: model output
        """


        # 同步方法：从模型请求生成模型输出

        Args:
            # 模型请求对象，类型为 ModelRequest
            request (ModelRequest): model request
            # 消息转换器，可选参数，用于将消息转换为特定格式。默认为 None
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            # 模型输出对象，类型为 ModelOutput
            ModelOutput: model output



        output = None
        for out in self.sync_generate_stream(request, message_converter):
            output = out
        return output


        # 初始化输出为 None
        output = None
        # 遍历同步生成流方法的输出，更新输出对象
        for out in self.sync_generate_stream(request, message_converter):
            output = out
        # 返回最终输出对象
        return output



    async def generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> AsyncIterator[ModelOutput]:
        """Generate model output stream from model request.

        We strongly recommend you to implement this method instead of sync_generate_stream for high performance.

        Args:
            request (ModelRequest): model request
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            AsyncIterator[ModelOutput]: model output stream
        """


        # 异步方法：从模型请求生成模型输出流

        Args:
            # 模型请求对象，类型为 ModelRequest
            request (ModelRequest): model request
            # 消息转换器，可选参数，用于将消息转换为特定格式。默认为 None
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            # 异步迭代器，生成模型输出流
            AsyncIterator[ModelOutput]: model output stream



        from starlette.concurrency import iterate_in_threadpool

        async for output in iterate_in_threadpool(
            self.sync_generate_stream(request, message_converter)
        ):
            yield output


        # 导入 iterate_in_threadpool 方法，用于在线程池中迭代执行同步生成流方法

        async for output in iterate_in_threadpool(
            self.sync_generate_stream(request, message_converter)
        ):
            yield output



    def sync_generate_stream(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,


        # 同步方法：从模型请求生成模型输出流

        Args:
            # 模型请求对象，类型为 ModelRequest
            request (ModelRequest): model request
            # 消息转换器，可选参数，用于将消息转换为特定格式。默认为 None
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.
    @abstractmethod
    def generate_outputs(
        self, request: ModelRequest, message_converter: Optional[MessageConverter] = None
    ) -> Iterator[ModelOutput]:
        """Generate model output stream from model request.

        Args:
            request (ModelRequest): model request
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            Iterator[ModelOutput]: model output stream
        """

        raise NotImplementedError()

    async def models(self) -> List[ModelMetadata]:
        """Get model metadata list

        Returns:
            List[ModelMetadata]: model metadata list
        """
        return self._models()

    @property
    def default_model(self) -> str:
        """Get default model name

        Returns:
            str: default model name
        """
        return self.model_names[0]

    @cache
    def _models(self) -> List[ModelMetadata]:
        """Cache and return model metadata list.

        Returns:
            List[ModelMetadata]: cached model metadata list
        """
        results = []
        for model in self.model_names:
            results.append(
                ModelMetadata(model=model, context_length=self.context_length)
            )
        return results

    def local_covert_message(
        self,
        request: ModelRequest,
        message_converter: Optional[MessageConverter] = None,
    ) -> ModelRequest:
        """Convert message locally using a message converter if provided.

        Args:
            request (ModelRequest): model request
            message_converter (Optional[MessageConverter], optional): message converter. Defaults to None.

        Returns:
            ModelRequest: converted model request
        """
        if not message_converter:
            return request
        metadata = self._models[0].ext_metadata
        new_request = request.copy()
        new_messages = message_converter.convert(request.messages, metadata)
        new_request.messages = new_messages
        return new_request

    async def count_token(self, model: str, prompt: str) -> int:
        """Count tokens in the provided prompt using the specified model.

        Args:
            model (str): model name
            prompt (str): prompt to count tokens

        Returns:
            int: token count, -1 if counting fails
        """
        counts = await blocking_func_to_async(
            self.executor, self.proxy_tokenizer.count_token, model, [prompt]
        )
        return counts[0]
```