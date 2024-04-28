# `.\transformers\commands\serving.py`

```
# 导入必要的模块和类
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional

from ..pipelines import Pipeline, get_supported_tasks, pipeline
from ..utils import logging
from . import BaseTransformersCLICommand

# 尝试导入 FastAPI 和相关模块
try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.routing import APIRoute
    from pydantic import BaseModel
    from starlette.responses import JSONResponse
    from uvicorn import run

    # 标记已成功导入 FastAPI 相关模块
    _serve_dependencies_installed = True
# 处理导入异常情况
except (ImportError, AttributeError):
    # 如果导入失败，则定义一个空的基本模型
    BaseModel = object
    # 定义一个空的函数，模拟 Body 函数
    def Body(*x, **y):
        pass
    # 标记未成功导入 FastAPI 相关模块
    _serve_dependencies_installed = False

# 获取 logger 对象
logger = logging.get_logger("transformers-cli/serving")

# 定义服务命令的工厂函数，根据命令行参数实例化服务服务器
def serve_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.

    Returns: ServeCommand
    """
    # 创建一个 pipeline 对象，用于处理自然语言处理任务
    nlp = pipeline(
        task=args.task,
        model=args.model if args.model else None,
        config=args.config,
        tokenizer=args.tokenizer,
        device=args.device,
    )
    # 返回 ServeCommand 实例
    return ServeCommand(nlp, args.host, args.port, args.workers)

# 定义 ServeModelInfoResult 类，用于暴露模型信息
class ServeModelInfoResult(BaseModel):
    """
    Expose model information
    """

    infos: dict

# 定义 ServeTokenizeResult 类，用于表示分词结果
class ServeTokenizeResult(BaseModel):
    """
    Tokenize result model
    """

    tokens: List[str]  # 分词后的 token 列表
    tokens_ids: Optional[List[int]]  # 分词后的 token ID 列表（可选）

# 定义 ServeDeTokenizeResult 类，用于表示反分词结果
class ServeDeTokenizeResult(BaseModel):
    """
    DeTokenize result model
    """

    text: str  # 反分词后的文本字符串

# 定义 ServeForwardResult 类，用于表示转发结果
class ServeForwardResult(BaseModel):
    """
    Forward result model
    """

    output: Any  # 转发的输出结果

# 定义 ServeCommand 类，用于处理服务命令
class ServeCommand(BaseTransformersCLICommand):
    @staticmethod
    # 将该命令注册到 argparse，以便在 transformer-cli 中可用
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        # 添加名为 "serve" 的子命令解析器，用于运行 REST 和 GraphQL 端点的推理请求
        serve_parser = parser.add_parser(
            "serve", help="CLI tool to run inference requests through REST and GraphQL endpoints."
        )
        # 添加命令行参数，用于指定要在管道上运行的任务
        serve_parser.add_argument(
            "--task",
            type=str,
            choices=get_supported_tasks(),
            help="The task to run the pipeline on",
        )
        # 添加命令行参数，用于指定服务器将监听的接口
        serve_parser.add_argument("--host", type=str, default="localhost", help="Interface the server will listen on.")
        # 添加命令行参数，用于指定服务将监听的端口
        serve_parser.add_argument("--port", type=int, default=8888, help="Port the serving will listen to.")
        # 添加命令行参数，用于指定 HTTP 工作线程的数量
        serve_parser.add_argument("--workers", type=int, default=1, help="Number of http workers")
        # 添加命令行参数，用于指定模型的名称或存储模型的路径
        serve_parser.add_argument("--model", type=str, help="Model's name or path to stored model.")
        # 添加命令行参数，用于指定模型配置的名称或存储模型配置的路径
        serve_parser.add_argument("--config", type=str, help="Model's config name or path to stored model.")
        # 添加命令行参数，用于指定要使用的分词器名称
        serve_parser.add_argument("--tokenizer", type=str, help="Tokenizer name to use.")
        # 添加命令行参数，用于指定要运行的设备，-1 表示 CPU，>= 0 表示 GPU（默认值：-1）
        serve_parser.add_argument(
            "--device",
            type=int,
            default=-1,
            help="Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)",
        )
        # 设置默认函数，将其指向 serve_command_factory
        serve_parser.set_defaults(func=serve_command_factory)
    # 初始化方法，接受一个管道对象、主机地址、端口号和工作进程数作为参数
    def __init__(self, pipeline: Pipeline, host: str, port: int, workers: int):
        # 将管道对象保存在实例属性中
        self._pipeline = pipeline

        # 将主机地址、端口号和工作进程数保存在实例属性中
        self.host = host
        self.port = port
        self.workers = workers

        # 检查是否安装了 serve 的依赖
        if not _serve_dependencies_installed:
            # 如果没有安装依赖，则引发运行时错误并提示安装相关依赖
            raise RuntimeError(
                "Using serve command requires FastAPI and uvicorn. "
                'Please install transformers with [serving]: pip install "transformers[serving]". '
                "Or install FastAPI and uvicorn separately."
            )
        else:
            # 如果安装了依赖，则记录日志显示模型服务的主机和端口
            logger.info(f"Serving model over {host}:{port}")
            # 创建 FastAPI 应用实例
            self._app = FastAPI(
                routes=[
                    # 定义根路径的路由，调用 model_info 方法，返回 ServeModelInfoResult 类型的结果
                    APIRoute(
                        "/",
                        self.model_info,
                        response_model=ServeModelInfoResult,
                        response_class=JSONResponse,
                        methods=["GET"],
                    ),
                    # 定义 tokenize 路径的路由，调用 tokenize 方法，返回 ServeTokenizeResult 类型的结果
                    APIRoute(
                        "/tokenize",
                        self.tokenize,
                        response_model=ServeTokenizeResult,
                        response_class=JSONResponse,
                        methods=["POST"],
                    ),
                    # 定义 detokenize 路径的路由，调用 detokenize 方法，返回 ServeDeTokenizeResult 类型的结果
                    APIRoute(
                        "/detokenize",
                        self.detokenize,
                        response_model=ServeDeTokenizeResult,
                        response_class=JSONResponse,
                        methods=["POST"],
                    ),
                    # 定义 forward 路径的路由，调用 forward 方法，返回 ServeForwardResult 类型的结果
                    APIRoute(
                        "/forward",
                        self.forward,
                        response_model=ServeForwardResult,
                        response_class=JSONResponse,
                        methods=["POST"],
                    ),
                ],
                # 设置请求超时时间为 600 秒
                timeout=600,
            )

    # 运行方法，启动 FastAPI 应用
    def run(self):
        # 调用 uvicorn 的 run 方法启动 FastAPI 应用
        run(self._app, host=self.host, port=self.port, workers=self.workers)

    # 返回模型信息的方法
    def model_info(self):
        # 返回 ServeModelInfoResult 类型的结果，包含管道模型的配置信息
        return ServeModelInfoResult(infos=vars(self._pipeline.model.config))

    # 对输入文本进行分词处理的方法
    def tokenize(self, text_input: str = Body(None, embed=True), return_ids: bool = Body(False, embed=True)):
        """
        Tokenize the provided input and eventually returns corresponding tokens id: - **text_input**: String to
        tokenize - **return_ids**: Boolean flags indicating if the tokens have to be converted to their integer
        mapping.
        """
        try:
            # 使用管道中的分词器对输入文本进行分词处理
            tokens_txt = self._pipeline.tokenizer.tokenize(text_input)

            # 判断是否需要返回分词对应的 ID
            if return_ids:
                # 将分词转换为对应的 ID
                tokens_ids = self._pipeline.tokenizer.convert_tokens_to_ids(tokens_txt)
                # 返回 ServeTokenizeResult 类型的结果，包含分词结果和对应的 ID
                return ServeTokenizeResult(tokens=tokens_txt, tokens_ids=tokens_ids)
            else:
                # 返回 ServeTokenizeResult 类型的结果，仅包含分词结果
                return ServeTokenizeResult(tokens=tokens_txt)

        except Exception as e:
            # 捕获异常并返回 HTTP 状态码 500，以及相关错误信息
            raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})
    # 定义一个方法用于将 tokens ids 解码为可读文本
    def detokenize(
        self,
        tokens_ids: List[int] = Body(None, embed=True),  # tokens ids 列表，默认为 None
        skip_special_tokens: bool = Body(False, embed=True),  # 是否跳过特殊 tokens，默认为 False
        cleanup_tokenization_spaces: bool = Body(True, embed=True),  # 是否清除 tokenization 中的空格，默认为 True
    ):
        """
        Detokenize the provided tokens ids to readable text: - **tokens_ids**: List of tokens ids -
        **skip_special_tokens**: Flag indicating to not try to decode special tokens - **cleanup_tokenization_spaces**:
        Flag indicating to remove all leading/trailing spaces and intermediate ones.
        """
        try:
            # 使用 tokenizer 对 tokens ids 进行解码
            decoded_str = self._pipeline.tokenizer.decode(tokens_ids, skip_special_tokens, cleanup_tokenization_spaces)
            # 返回解码后的文本结果
            return ServeDeTokenizeResult(model="", text=decoded_str)
        except Exception as e:
            # 抛出异常
            raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})

    # 异步方法，用于前向传播
    async def forward(self, inputs=Body(None, embed=True)):
        """
        **inputs**: **attention_mask**: **tokens_type_ids**:
        """

        # 检查输入是否为空字符串
        if len(inputs) == 0:
            # 返回空输出和注意力列表
            return ServeForwardResult(output=[], attention=[])

        try:
            # 通过模型进行前向传播
            output = self._pipeline(inputs)
            # 返回前向传播结果
            return ServeForwardResult(output=output)
        except Exception as e:
            # 抛出异常
            raise HTTPException(500, {"error": str(e)})
```