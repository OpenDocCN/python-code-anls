# `.\DB-GPT-src\dbgpt\util\tracer\tracer_middleware.py`

```py
import logging  # 导入日志模块
from contextvars import ContextVar  # 导入上下文变量相关模块

from starlette.middleware.base import BaseHTTPMiddleware  # 导入 Starlette 的基础 HTTP 中间件
from starlette.requests import Request  # 导入 Starlette 的请求类
from starlette.types import ASGIApp  # 导入 Starlette 的 ASGI 应用类型

from dbgpt.util.tracer import Tracer, TracerContext  # 导入自定义的追踪器和追踪上下文

from .base import _parse_span_id  # 从当前目录下的 base 模块导入 _parse_span_id 函数

_DEFAULT_EXCLUDE_PATHS = ["/api/controller/heartbeat", "/api/health"]  # 默认要排除的路径列表

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class TraceIDMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        trace_context_var: ContextVar[TracerContext],
        tracer: Tracer,
        root_operation_name: str = "DB-GPT-Web-Entry",
        include_prefix: str = "/api",
        exclude_paths=_DEFAULT_EXCLUDE_PATHS,
    ):
        super().__init__(app)
        self.trace_context_var = trace_context_var  # 设置追踪上下文变量
        self.tracer = tracer  # 设置追踪器对象
        self.root_operation_name = root_operation_name  # 设置根操作名称，默认为 "DB-GPT-Web-Entry"
        self.include_prefix = include_prefix  # 设置包含路径的前缀，默认为 "/api"
        self.exclude_paths = exclude_paths  # 设置要排除的路径列表，默认为 _DEFAULT_EXCLUDE_PATHS

    async def dispatch(self, request: Request, call_next):
        if (
            request.url.path in self.exclude_paths
            or not request.url.path.startswith(self.include_prefix)
        ):
            return await call_next(request)

        # 从请求头中读取 trace_id
        span_id = _parse_span_id(request)
        logger.debug(
            f"TraceIDMiddleware: span_id={span_id}, path={request.url.path}, "
            f"headers={request.headers}"
        )
        # 使用追踪器开始一个新的 span，并传入相关的元数据
        with self.tracer.start_span(
            self.root_operation_name, span_id, metadata={"path": request.url.path}
        ):
            response = await call_next(request)
        return response
```