# `.\commands\serving.py`

```
# 版权声明和许可证信息，指明此代码受 Apache License, Version 2.0 保护，禁止未经许可使用
#
# from ...pipelines 导入需要的模块和函数
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional

from ..pipelines import Pipeline, get_supported_tasks, pipeline
# 导入日志模块
from ..utils import logging
# 导入基础命令行接口类
from . import BaseTransformersCLICommand

try:
    # 尝试导入 FastAPI 和相关依赖
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.routing import APIRoute
    from pydantic import BaseModel
    from starlette.responses import JSONResponse
    from uvicorn import run

    # 标记服务依赖已安装
    _serve_dependencies_installed = True
except (ImportError, AttributeError):
    # 如果导入错误或属性错误，将 BaseModel 设为 object，并定义 Body 函数为空函数
    BaseModel = object

    def Body(*x, **y):
        pass

    # 标记服务依赖未安装
    _serve_dependencies_installed = False

# 获取名为 "transformers-cli/serving" 的日志记录器对象
logger = logging.get_logger("transformers-cli/serving")


def serve_command_factory(args: Namespace):
    """
    从提供的命令行参数实例化服务服务器的工厂函数。

    Returns: ServeCommand 实例
    """
    # 调用 pipeline 函数创建 NLP 管道对象 nlp
    nlp = pipeline(
        task=args.task,
        model=args.model if args.model else None,
        config=args.config,
        tokenizer=args.tokenizer,
        device=args.device,
    )
    # 返回 ServeCommand 的实例，传递 nlp 对象、主机地址、端口和工作进程数作为参数
    return ServeCommand(nlp, args.host, args.port, args.workers)


class ServeModelInfoResult(BaseModel):
    """
    暴露模型信息的数据模型
    """

    infos: dict


class ServeTokenizeResult(BaseModel):
    """
    分词结果数据模型
    """

    tokens: List[str]
    tokens_ids: Optional[List[int]]


class ServeDeTokenizeResult(BaseModel):
    """
    反分词结果数据模型
    """

    text: str


class ServeForwardResult(BaseModel):
    """
    前向传播结果数据模型
    """

    output: Any


class ServeCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        # 创建一个子命令解析器 'serve'，用于运行 REST 和 GraphQL 端点的推理请求
        serve_parser = parser.add_parser(
            "serve", help="CLI tool to run inference requests through REST and GraphQL endpoints."
        )
        # 添加 '--task' 参数，指定要在管道上运行的任务，从支持的任务列表中选择
        serve_parser.add_argument(
            "--task",
            type=str,
            choices=get_supported_tasks(),
            help="The task to run the pipeline on",
        )
        # 添加 '--host' 参数，指定服务器监听的接口，默认为 localhost
        serve_parser.add_argument("--host", type=str, default="localhost", help="Interface the server will listen on.")
        # 添加 '--port' 参数，指定服务器监听的端口，默认为 8888
        serve_parser.add_argument("--port", type=int, default=8888, help="Port the serving will listen to.")
        # 添加 '--workers' 参数，指定 HTTP 服务器的工作线程数，默认为 1
        serve_parser.add_argument("--workers", type=int, default=1, help="Number of http workers")
        # 添加 '--model' 参数，指定模型的名称或存储路径
        serve_parser.add_argument("--model", type=str, help="Model's name or path to stored model.")
        # 添加 '--config' 参数，指定模型配置的名称或存储路径
        serve_parser.add_argument("--config", type=str, help="Model's config name or path to stored model.")
        # 添加 '--tokenizer' 参数，指定要使用的分词器的名称
        serve_parser.add_argument("--tokenizer", type=str, help="Tokenizer name to use.")
        # 添加 '--device' 参数，指定运行的设备，-1 表示 CPU，>= 0 表示 GPU，默认为 -1
        serve_parser.add_argument(
            "--device",
            type=int,
            default=-1,
            help="Indicate the device to run onto, -1 indicates CPU, >= 0 indicates GPU (default: -1)",
        )
        # 将函数 'serve_command_factory' 设置为默认处理函数
        serve_parser.set_defaults(func=serve_command_factory)
    # 初始化方法，接受 Pipeline 对象、主机名、端口号和工作进程数作为参数
    def __init__(self, pipeline: Pipeline, host: str, port: int, workers: int):
        # 将传入的 Pipeline 对象赋值给实例变量 _pipeline
        self._pipeline = pipeline

        # 将传入的主机名赋值给实例变量 host
        self.host = host
        # 将传入的端口号赋值给实例变量 port
        self.port = port
        # 将传入的工作进程数赋值给实例变量 workers
        self.workers = workers

        # 检查是否已安装 serve 所需的依赖，如果未安装则抛出运行时错误
        if not _serve_dependencies_installed:
            raise RuntimeError(
                "Using serve command requires FastAPI and uvicorn. "
                'Please install transformers with [serving]: pip install "transformers[serving]". '
                "Or install FastAPI and uvicorn separately."
            )
        else:
            # 若依赖已安装，则记录信息，指示模型正在指定的主机和端口上提供服务
            logger.info(f"Serving model over {host}:{port}")
            # 创建 FastAPI 应用实例 _app，并设置路由和超时时间
            self._app = FastAPI(
                routes=[
                    APIRoute(
                        "/",
                        self.model_info,
                        response_model=ServeModelInfoResult,
                        response_class=JSONResponse,
                        methods=["GET"],
                    ),
                    APIRoute(
                        "/tokenize",
                        self.tokenize,
                        response_model=ServeTokenizeResult,
                        response_class=JSONResponse,
                        methods=["POST"],
                    ),
                    APIRoute(
                        "/detokenize",
                        self.detokenize,
                        response_model=ServeDeTokenizeResult,
                        response_class=JSONResponse,
                        methods=["POST"],
                    ),
                    APIRoute(
                        "/forward",
                        self.forward,
                        response_model=ServeForwardResult,
                        response_class=JSONResponse,
                        methods=["POST"],
                    ),
                ],
                timeout=600,
            )

    # 启动服务的方法，运行 FastAPI 应用
    def run(self):
        run(self._app, host=self.host, port=self.port, workers=self.workers)

    # 返回模型信息的方法，以 ServeModelInfoResult 对象的形式返回 Pipeline 模型的配置信息
    def model_info(self):
        return ServeModelInfoResult(infos=vars(self._pipeline.model.config))

    # 对输入文本进行标记化处理的方法，接受 text_input 和 return_ids 两个参数
    def tokenize(self, text_input: str = Body(None, embed=True), return_ids: bool = Body(False, embed=True)):
        """
        Tokenize the provided input and eventually returns corresponding tokens id: - **text_input**: String to
        tokenize - **return_ids**: Boolean flags indicating if the tokens have to be converted to their integer
        mapping.
        """
        try:
            # 使用 Pipeline 对象的 tokenizer 对输入文本进行标记化处理
            tokens_txt = self._pipeline.tokenizer.tokenize(text_input)

            # 如果 return_ids 为 True，则将标记化后的文本转换为对应的整数标识
            if return_ids:
                tokens_ids = self._pipeline.tokenizer.convert_tokens_to_ids(tokens_txt)
                return ServeTokenizeResult(tokens=tokens_txt, tokens_ids=tokens_ids)
            else:
                # 否则，返回标记化后的文本
                return ServeTokenizeResult(tokens=tokens_txt)

        # 捕获异常，并返回 HTTP 错误码 500 及错误详情
        except Exception as e:
            raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})
    # Detokenize函数将token ids转换为可读文本
    def detokenize(
        self,
        tokens_ids: List[int] = Body(None, embed=True),  # 输入参数：token ids列表
        skip_special_tokens: bool = Body(False, embed=True),  # 是否跳过特殊token的标志
        cleanup_tokenization_spaces: bool = Body(True, embed=True),  # 是否清除token化空格的标志
    ):
        """
        Detokenize the provided tokens ids to readable text:
        - **tokens_ids**: List of tokens ids
        - **skip_special_tokens**: Flag indicating to not try to decode special tokens
        - **cleanup_tokenization_spaces**: Flag indicating to remove all leading/trailing spaces and intermediate ones.
        """
        try:
            # 使用tokenizer对象解码tokens_ids，根据skip_special_tokens和cleanup_tokenization_spaces的设置进行处理
            decoded_str = self._pipeline.tokenizer.decode(tokens_ids, skip_special_tokens, cleanup_tokenization_spaces)
            # 返回解码后的结果，作为ServeDeTokenizeResult的一部分
            return ServeDeTokenizeResult(model="", text=decoded_str)
        except Exception as e:
            # 如果出现异常，抛出HTTP异常，返回500状态码和错误详情
            raise HTTPException(status_code=500, detail={"model": "", "error": str(e)})

    async def forward(self, inputs=Body(None, embed=True)):
        """
        **inputs**: **attention_mask**: **tokens_type_ids**:
        """

        # 检查输入是否为空字符串
        if len(inputs) == 0:
            # 如果为空，则返回空的ServeForwardResult对象，输出和attention都为空列表
            return ServeForwardResult(output=[], attention=[])

        try:
            # 通过模型处理输入数据
            output = self._pipeline(inputs)
            # 返回ServeForwardResult对象，包含模型输出的结果
            return ServeForwardResult(output=output)
        except Exception as e:
            # 如果出现异常，抛出HTTP异常，返回500状态码和错误详情
            raise HTTPException(500, {"error": str(e)})
```