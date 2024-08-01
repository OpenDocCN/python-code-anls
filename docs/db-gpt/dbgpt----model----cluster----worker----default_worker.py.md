# `.\DB-GPT-src\dbgpt\model\cluster\worker\default_worker.py`

```py
# 导入必要的模块
import logging  # 导入日志记录模块
import os  # 导入操作系统相关模块
import time  # 导入时间模块
import traceback  # 导入异常跟踪模块
from typing import Dict, Iterator, List, Optional  # 导入类型提示相关模块

# 导入项目内部模块
from dbgpt.configs.model_config import get_device  # 从项目配置中获取设备信息函数
from dbgpt.core import (  # 导入核心功能模块
    ModelExtraMedata,  # 模型额外元数据
    ModelInferenceMetrics,  # 模型推理指标
    ModelMetadata,  # 模型元数据
    ModelOutput,  # 模型输出
)
from dbgpt.model.adapter.base import LLMModelAdapter  # 导入语言模型适配器基类
from dbgpt.model.adapter.loader import ModelLoader, _get_model_real_path  # 导入模型加载器和获取模型真实路径函数
from dbgpt.model.adapter.model_adapter import get_llm_model_adapter  # 导入获取语言模型适配器函数
from dbgpt.model.cluster.worker_base import ModelWorker  # 导入模型工作器基类
from dbgpt.model.parameter import ModelParameters  # 导入模型参数类
from dbgpt.util.model_utils import _clear_model_cache, _get_current_cuda_memory  # 导入模型工具函数：清理模型缓存和获取当前 CUDA 内存使用情况
from dbgpt.util.parameter_utils import EnvArgumentParser, _get_dict_from_obj  # 导入参数工具函数：环境参数解析器和从对象获取字典
from dbgpt.util.system_utils import get_system_info  # 导入系统信息获取函数
from dbgpt.util.tracer import SpanType, SpanTypeRunName, root_tracer  # 导入跟踪器相关模块

# 设置日志记录器
logger = logging.getLogger(__name__)

# 初始化 Torch 相关变量
_torch_imported = False
torch = None


class DefaultModelWorker(ModelWorker):
    def __init__(self) -> None:
        self.model = None  # 模型对象初始化为空
        self.tokenizer = None  # 分词器对象初始化为空
        self._model_params = None  # 模型参数对象初始化为空
        self.llm_adapter: LLMModelAdapter = None  # 语言模型适配器对象初始化为空
        self._support_async = False  # 默认不支持异步操作

    def load_worker(self, model_name: str, model_path: str, **kwargs) -> None:
        if model_path.endswith("/"):
            model_path = model_path[:-1]  # 去除路径结尾的斜杠
        model_path = _get_model_real_path(model_name, model_path)  # 获取模型的真实路径
        self.model_name = model_name  # 记录模型名称
        self.model_path = model_path  # 记录模型路径

        model_type = kwargs.get("model_type")  # 获取模型类型参数
        ### Temporary configuration, fastchat will be used by default in the future.
        use_fastchat = os.getenv("USE_FASTCHAT", "True").lower() == "true"  # 检查是否启用快速聊天模式

        # 获取语言模型适配器
        self.llm_adapter = get_llm_model_adapter(
            self.model_name,
            self.model_path,
            use_fastchat=use_fastchat,
            model_type=model_type,
        )
        model_type = self.llm_adapter.model_type()  # 获取适配器的模型类型
        self.param_cls = self.llm_adapter.model_param_class(model_type)  # 获取模型参数类
        self._support_async = self.llm_adapter.support_async()  # 检查适配器是否支持异步操作

        logger.info(
            f"model_name: {self.model_name}, model_path: {self.model_path}, model_param_class: {self.param_cls}"
        )  # 记录模型加载信息

        self.ml: ModelLoader = ModelLoader(
            model_path=self.model_path, model_name=self.model_name
        )  # 创建模型加载器实例
        # 设置默认模型上下文长度
        self.context_len = 2048

    def model_param_class(self) -> ModelParameters:
        return self.param_cls  # 返回模型参数类实例

    def support_async(self) -> bool:
        return self._support_async  # 返回是否支持异步操作的标志
    # 解析命令行参数并返回模型参数对象
    def parse_parameters(self, command_args: List[str] = None) -> ModelParameters:
        # 获取模型参数类
        param_cls = self.model_param_class()
        # 创建环境参数解析器对象
        model_args = EnvArgumentParser()
        # 获取环境变量前缀
        env_prefix = EnvArgumentParser.get_env_prefix(self.model_name)
        # 获取模型类型
        model_type = self.llm_adapter.model_type()
        # 使用参数解析器对象解析参数，并封装成数据类对象
        model_params: ModelParameters = model_args.parse_args_into_dataclass(
            param_cls,
            env_prefixes=[env_prefix, "LLM_"],
            command_args=command_args,
            model_name=self.model_name,
            model_path=self.model_path,
            model_type=model_type,
        )
        # 如果模型设备未指定，则获取默认设备并记录日志
        if not model_params.device:
            model_params.device = get_device()
            logger.info(
                f"[DefaultModelWorker] Parameters of device is None, use {model_params.device}"
            )
        # 返回解析后的模型参数对象
        return model_params

    # 启动模型加载过程
    def start(
        self, model_params: ModelParameters = None, command_args: List[str] = None
    ) -> None:
        # 懒加载 torch 库
        _try_import_torch()
        # 如果未提供模型参数，则解析命令行参数获取模型参数
        if not model_params:
            model_params = self.parse_parameters(command_args)
        # 记录模型参数
        self._model_params = model_params
        logger.info(f"Begin load model, model params: {model_params}")
        # 设置元数据信息
        metadata = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "model_type": self.llm_adapter.model_type(),
            "llm_adapter": str(self.llm_adapter),
            "run_service": SpanTypeRunName.MODEL_WORKER,
            "params": _get_dict_from_obj(model_params),
            "sys_infos": _get_dict_from_obj(get_system_info()),
        }
        # 使用根跟踪器开始一个跟踪 span
        with root_tracer.start_span(
            "DefaultModelWorker.start", span_type=SpanType.RUN, metadata=metadata
        ):
            # 使用指定参数加载模型和分词器
            self.model, self.tokenizer = self.ml.loader_with_params(
                model_params, self.llm_adapter
            )
            # 解析模型的最大长度
            model_max_length = self.llm_adapter.parse_max_length(
                self.model, self.tokenizer
            )
            # 如果成功解析到模型的最大长度，则记录日志并设置上下文长度
            if model_max_length:
                logger.info(
                    f"Parse model max length {model_max_length} from model {self.model_name}."
                )
                self.context_len = model_max_length
            # 否则，如果模型参数对象具有最大上下文大小属性，则设置上下文长度
            elif hasattr(model_params, "max_context_size"):
                self.context_len = model_params.max_context_size

    # 停止模型服务
    def stop(self) -> None:
        # 如果模型不存在，则记录警告并直接返回
        if not self.model:
            logger.warn("Model has been stopped!!")
            return
        # 清理模型和分词器对象
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        # 清理模型缓存
        _clear_model_cache(self._model_params.device)
    # 生成数据流的方法，返回一个生成器，生成模型输出的迭代器
    def generate_stream(self, params: Dict) -> Iterator[ModelOutput]:
        # 开始一个新的跟踪 span，命名为"DefaultModelWorker.generate_stream"，使用给定的 span_id
        span = root_tracer.start_span(
            "DefaultModelWorker.generate_stream", params.get("span_id")
        )
        try:
            # 准备生成数据流所需的参数和上下文
            (
                params,
                model_context,
                generate_stream_func,
                model_span,
            ) = self._prepare_generate_stream(
                params,
                span_operation_name="DefaultModelWorker_call.generate_stream_func",
            )

            # 初始化前一次响应的字符串
            previous_response = ""
            # 创建模型推断的度量指标对象
            last_metrics = ModelInferenceMetrics.create_metrics()
            # 标记是否是第一次生成数据
            is_first_generate = True

            # 获取上下文长度，如果未指定则使用默认值
            context_len = params.get("context_len") or self.context_len
            # 使用 generate_stream_func 生成数据流，每次迭代产生一个输出
            for output in generate_stream_func(
                self.model, self.tokenizer, params, get_device(), context_len
            ):
                # 处理模型的输出结果
                (
                    model_output,
                    incremental_output,
                    output_str,
                    current_metrics,
                ) = self._handle_output(
                    output,
                    previous_response,
                    model_context,
                    last_metrics,
                    is_first_generate,
                )
                # 更新前一次响应字符串
                previous_response = output_str
                # 更新最新的度量指标
                last_metrics = current_metrics
                # 生成当前模型输出
                yield model_output

            # 打印完整的数据流输出和生成参数信息
            print(
                f"\n\nfull stream output:\n{previous_response}\n\nmodel generate_stream params:\n{params}"
            )
            # 结束当前的 model_span 跟踪，记录输出信息
            model_span.end(metadata={"output": previous_response})
            # 结束当前的 span 跟踪
            span.end()
        except Exception as e:
            # 处理异常情况，并记录错误信息到跟踪
            output = self._handle_exception(e)
            yield output
            span.end(metadata={"error": output.to_dict()})

    # 生成非数据流结果的方法
    def generate(self, params: Dict) -> ModelOutput:
        """Generate non stream result"""
        output = None
        # 调用 generate_stream 方法来生成结果，返回最后一个输出
        for out in self.generate_stream(params):
            output = out
        return output

    # 计算给定提示文本中的标记数量
    def count_token(self, prompt: str) -> int:
        return _try_to_count_token(prompt, self.tokenizer, self.model)

    # 异步计算给定提示文本中的标记数量
    async def async_count_token(self, prompt: str) -> int:
        # 如果模型是 ProxyModel 类型并且有 proxy_llm_client 属性，则调用其异步计算方法
        from dbgpt.model.proxy.llms.proxy_model import ProxyModel

        if isinstance(self.model, ProxyModel) and self.model.proxy_llm_client:
            return await self.model.proxy_llm_client.count_token(
                self.model.proxy_llm_client.default_model, prompt
            )
        # 否则抛出未实现错误
        raise NotImplementedError
    # 返回模型的元数据，包括模型名称、上下文长度和额外元数据
    def get_model_metadata(self, params: Dict) -> ModelMetadata:
        # 获取语言模型适配器的提示角色和默认消息分隔符
        ext_metadata = ModelExtraMedata(
            prompt_roles=self.llm_adapter.get_prompt_roles(),
            prompt_sep=self.llm_adapter.get_default_message_separator(),
        )
        # 创建并返回模型元数据对象
        return ModelMetadata(
            model=self.model_name,
            context_length=self.context_len,
            ext_metadata=ext_metadata,
        )

    # 异步获取模型元数据
    async def async_get_model_metadata(self, params: Dict) -> ModelMetadata:
        # 调用同步方法获取模型元数据
        return self.get_model_metadata(params)

    # 返回嵌入向量，但此处抛出了未实现错误
    def embeddings(self, params: Dict) -> List[List[float]]:
        raise NotImplementedError

    # 异步生成流式输出
    async def async_generate_stream(self, params: Dict) -> Iterator[ModelOutput]:
        # 在分布式追踪中创建一个新的跟踪器 span
        span = root_tracer.start_span(
            "DefaultModelWorker.async_generate_stream", params.get("span_id")
        )
        try:
            # 准备生成流的参数和相关操作
            (
                params,
                model_context,
                generate_stream_func,
                model_span,
            ) = self._prepare_generate_stream(
                params,
                span_operation_name="DefaultModelWorker_call.generate_stream_func",
            )

            # 获取或设置上下文长度
            context_len = params.get("context_len") or self.context_len

            # 创建模型推断的指标对象
            last_metrics = ModelInferenceMetrics.create_metrics()
            is_first_generate = True

            # 异步迭代生成流的输出
            async for output in generate_stream_func(
                self.model, self.tokenizer, params, get_device(), context_len
            ):
                # 处理每一步的模型输出
                (
                    model_output,
                    incremental_output,
                    output_str,
                    current_metrics,
                ) = self._handle_output(
                    output,
                    previous_response,
                    model_context,
                    last_metrics,
                    is_first_generate,
                )
                if is_first_generate:
                    is_first_generate = False

                # 更新前一次的输出字符串
                previous_response = output_str
                # 更新最新的推断指标
                last_metrics = current_metrics

                # 返回模型输出
                yield model_output

            # 打印完整的流式输出和生成流的参数
            print(
                f"\n\nfull stream output:\n{previous_response}\n\nmodel generate_stream params:\n{params}"
            )

            # 结束模型跟踪器 span，并记录输出元数据
            model_span.end(metadata={"output": previous_response})
            span.end()

        except Exception as e:
            # 处理异常情况并记录错误信息到跟踪器 span
            output = self._handle_exception(e)
            yield output
            span.end(metadata={"error": output.to_dict()})

    # 异步生成模型输出
    async def async_generate(self, params: Dict) -> ModelOutput:
        output = None
        # 异步迭代生成流的输出，返回最终的输出结果
        async for out in self.async_generate_stream(params):
            output = out
        return output
    # 准备生成数据流的操作，接受参数字典和操作名作为输入
    def _prepare_generate_stream(self, params: Dict, span_operation_name: str):
        # 调用语言模型适配器的模型适应方法，返回适配后的参数和模型上下文
        params, model_context = self.llm_adapter.model_adaptation(
            params,
            self.model_name,
            self.model_path,
            self.tokenizer,
            prompt_template=self.ml.prompt_template,
        )
        # 初始化数据流类型为空字符串
        stream_type = ""
        # 如果支持异步操作
        if self.support_async():
            # 获取异步生成数据流的函数
            generate_stream_func = self.llm_adapter.get_async_generate_stream_function(
                self.model, self.model_path
            )
            # 设置数据流类型为异步
            stream_type = "async "
            # 记录日志，指示当前使用的生成数据流函数是异步函数
            logger.info(
                "current generate stream function is asynchronous stream function"
            )
        else:
            # 否则获取同步生成数据流的函数
            generate_stream_func = self.llm_adapter.get_generate_stream_function(
                self.model, self.model_path
            )
        # 获取参数中的提示字符串，如果不存在则尝试获取字符串型提示
        str_prompt = params.get("prompt")
        if not str_prompt:
            str_prompt = params.get("string_prompt")
        # 打印语言模型适配器信息、模型提示和数据流类型
        print(
            f"llm_adapter: {str(self.llm_adapter)}\n\nmodel prompt: \n\n{str_prompt}\n\n{stream_type}stream output:\n"
        )

        # 生成数据流函数的完全限定名
        generate_stream_func_str_name = "{}.{}".format(
            generate_stream_func.__module__, generate_stream_func.__name__
        )

        # 复制参数字典以备后用，如果存在 "messages" 键则将其转换为字典列表
        span_params = {k: v for k, v in params.items()}
        if "messages" in span_params:
            span_params["messages"] = list(
                map(lambda m: m.dict(), span_params["messages"])
            )

        # 构建元数据字典，包含是否为异步函数、语言模型适配器信息、生成数据流函数的名称等
        metadata = {
            "is_async_func": self.support_async(),
            "llm_adapter": str(self.llm_adapter),
            "generate_stream_func": generate_stream_func_str_name,
        }
        # 更新元数据字典，加入参数字典、模型上下文信息和提示字符串
        metadata.update(span_params)
        metadata.update(model_context)
        metadata["prompt"] = str_prompt

        # 使用分布式跟踪系统启动一个新的跟踪操作，记录模型操作名和元数据
        model_span = root_tracer.start_span(span_operation_name, metadata=metadata)

        # 返回适配后的参数字典、模型上下文、生成数据流函数和跟踪操作对象
        return params, model_context, generate_stream_func, model_span
        ):
        # 初始化变量
        finish_reason = None
        usage = None
        error_code = 0
        # 如果输出是一个字典，则从中提取完成原因和使用情况，并获取文本内容
        if isinstance(output, dict):
            finish_reason = output.get("finish_reason")
            usage = output.get("usage")
            output = output["text"]
            # 如果有完成原因，则记录日志信息
            if finish_reason is not None:
                logger.info(f"finish_reason: {finish_reason}")
        # 如果输出是一个 ModelOutput 对象，则直接从中提取相应字段
        elif isinstance(output, ModelOutput):
            finish_reason = output.finish_reason
            usage = output.usage
            error_code = output.error_code
            output = output.text
        # 计算增量输出，从上一个响应结束位置开始
        incremental_output = output[len(previous_response) :]
        # 打印增量输出，并保持输出不缓冲
        print(incremental_output, end="", flush=True)

        # 根据模型输出生成新的指标(metrics)
        metrics = _new_metrics_from_model_output(last_metrics, is_first_generate, usage)
        # 创建模型输出对象
        model_output = ModelOutput(
            text=output,
            error_code=error_code,
            model_context=model_context,
            finish_reason=finish_reason,
            usage=usage,
            metrics=metrics,
        )
        # 返回模型输出对象、增量输出、原始输出及指标(metrics)
        return model_output, incremental_output, output, metrics

    def _handle_exception(self, e):
        # 检查异常是否为 torch.cuda.CudaError，并且确保 torch 已经导入
        if _torch_imported and isinstance(e, torch.cuda.CudaError):
            # 创建模型输出对象，指示 GPU 内存不足
            model_output = ModelOutput(
                text="**GPU OutOfMemory, Please Refresh.**", error_code=1
            )
        else:
            # 获取详细的异常信息并记录日志
            msg = traceback.format_exc()
            logger.error(f"Model inference error, detail: {msg}")
            # 创建模型输出对象，指示模型生成错误
            model_output = ModelOutput(
                text=f"**LLMServer Generate Error, Please CheckErrorInfo.**: {e}",
                error_code=1,
            )
        # 返回处理后的模型输出对象
        return model_output
# 根据给定的模型和分词器对象，返回模型的最大长度，如果找不到则返回 None
def _parse_model_max_length(model, tokenizer) -> Optional[int]:
    # 如果 tokenizer 或 model 为空，则直接返回 None
    if not (tokenizer or model):
        return None
    try:
        # 如果 tokenizer 存在并且有 model_max_length 属性，则返回其值
        if tokenizer and hasattr(tokenizer, "model_max_length"):
            return tokenizer.model_max_length
        # 如果 model 存在并且有 config 属性
        if model and hasattr(model, "config"):
            model_config = model.config
            # 如果 config 对象有 max_sequence_length 属性，则返回其值
            if hasattr(model_config, "max_sequence_length"):
                return model_config.max_sequence_length
            # 如果 config 对象有 max_position_embeddings 属性，则返回其值
            if hasattr(model_config, "max_position_embeddings"):
                return model_config.max_position_embeddings
    except Exception:
        # 发生任何异常时返回 None
        return None


# 根据模型输出和相关信息生成新的推断指标对象
def _new_metrics_from_model_output(
    last_metric: ModelInferenceMetrics,
    is_first_generate: bool,
    usage: Optional[Dict] = None,
) -> ModelInferenceMetrics:
    # 根据上一个指标对象创建一个新的指标对象
    metrics = ModelInferenceMetrics.create_metrics(last_metric)
    # 将收集索引递增
    metrics.collect_index = last_metric.collect_index + 1
    # 如果是第一次生成，则记录第一次生成的时间和日志信息
    if is_first_generate:
        logger.info(f"is_first_generate, usage: {usage}")
        metrics.first_completion_time_ms = time.time_ns() // 1_000_000

    # 如果没有 usage 或者 usage 不是字典类型，则直接返回 metrics
    if not usage or not isinstance(usage, dict):
        return metrics

    # 从 usage 字典中获取 prompt_tokens、completion_tokens 和 total_tokens
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # 如果 prompt_tokens 为 None，则使用 metrics 中的值
    if prompt_tokens is None:
        prompt_tokens = metrics.prompt_tokens
    # 如果 completion_tokens 为 None，则使用 metrics 中的值
    if completion_tokens is None:
        completion_tokens = metrics.completion_tokens
    # 如果 total_tokens 为 None，则计算并赋值给 metrics
    if total_tokens is None:
        total_tokens = metrics.total_tokens

    # 如果是第一次生成且 completion_tokens 不为 None，则记录首次生成的完成 token 数量和时间
    if is_first_generate and (completion_tokens is not None):
        # completion_tokens == 0 表示预填充情况
        metrics.first_completion_tokens = completion_tokens
        if completion_tokens == 1:
            metrics.first_token_time_ms = metrics.first_completion_time_ms

    # 如果不是第一次生成且第一个 token 时间为 None 且 completion_tokens 为 1，则记录第一个 token 时间
    if (
        not is_first_generate
        and metrics.first_token_time_ms is None
        and completion_tokens == 1
    ):
        # 情况：第一次生成没有 token，第二次生成有一个 token
        metrics.first_token_time_ms = time.time_ns() // 1_000_000

    # 如果 prompt_tokens 存在，则更新 metrics 中的值
    if prompt_tokens:
        metrics.prompt_tokens = prompt_tokens
    # 如果 completion_tokens 存在，则更新 metrics 中的值
    if completion_tokens:
        metrics.completion_tokens = completion_tokens
    # 如果 total_tokens 存在，则更新 metrics 中的值；如果 prompt_tokens 和 completion_tokens 都存在，则计算并更新 total_tokens
    if total_tokens:
        total_tokens = prompt_tokens + completion_tokens
        metrics.total_tokens = total_tokens

    # 如果 total_tokens 存在，则计算每秒处理的 token 数量
    if total_tokens:
        # 时间消耗（秒）
        duration = (metrics.current_time_ms - metrics.start_time_ms) / 1000.0
        metrics.speed_per_second = total_tokens / duration

    # 获取当前 GPU 信息并更新 metrics 中的值
    current_gpu_infos = _get_current_cuda_memory()
    metrics.current_gpu_infos = current_gpu_infos
    # 如果平均 GPU 信息为空，则更新为当前 GPU 信息
    if not metrics.avg_gpu_infos:
        metrics.avg_gpu_infos = current_gpu_infos
    # 如果当前 GPU 信息列表不为空，则执行以下操作
    elif current_gpu_infos:
        # 遍历 metrics.avg_gpu_infos 列表，同时追踪索引 i 和 last_avg 对象
        for i, last_avg in enumerate(metrics.avg_gpu_infos):
            # 计算已分配内存的累计量，考虑前 metrics.collect_index-1 次的平均值
            allocated_memory_gb = (
                last_avg.allocated_memory_gb * (metrics.collect_index - 1)
                + current_gpu_infos[i].allocated_memory_gb
            )
            # 更新 metrics.avg_gpu_infos[i] 的平均已分配内存
            metrics.avg_gpu_infos[i].allocated_memory_gb = (
                allocated_memory_gb / metrics.collect_index
            )
            # 更新 metrics.avg_gpu_infos[i] 的总内存
            metrics.avg_gpu_infos[i].total_memory_gb = current_gpu_infos[
                i
            ].total_memory_gb
            # 更新 metrics.avg_gpu_infos[i] 的缓存内存
            metrics.avg_gpu_infos[i].cached_memory_gb = current_gpu_infos[
                i
            ].cached_memory_gb
            # 更新 metrics.avg_gpu_infos[i] 的可用内存
            metrics.avg_gpu_infos[i].available_memory_gb = current_gpu_infos[
                i
            ].available_memory_gb

    # 返回更新后的 metrics 对象
    return metrics
# 尝试计算提示语中的标记数量
def _try_to_count_token(prompt: str, tokenizer, model) -> int:
    """Try to count token of prompt

    Args:
        prompt (str): 提示语
        tokenizer ([type]): 分词器
        model ([type]): 模型

    Returns:
        int: 标记数量，如果出错则返回 -1

    TODO: More implementation
    """
    try:
        # 导入代理模型的模块
        from dbgpt.model.proxy.llms.proxy_model import ProxyModel

        # 如果模型是代理模型实例，调用其 count_token 方法
        if isinstance(model, ProxyModel):
            return model.count_token(prompt)
        
        # 如果模型不是代理模型，假定是Huggingface模型，计算输入prompt的标记数量
        return len(tokenizer(prompt).input_ids[0])
    except Exception as e:
        # 捕获异常并记录警告日志，返回 -1
        logger.warning(f"Count token error, detail: {e}, return -1")
        return -1


# 尝试导入 Torch 库
def _try_import_torch():
    global torch
    global _torch_imported
    try:
        # 尝试导入 Torch 库
        import torch

        _torch_imported = True
    except ImportError:
        # 如果导入失败，则忽略错误
        pass
```