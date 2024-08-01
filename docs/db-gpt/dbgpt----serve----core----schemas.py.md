# `.\DB-GPT-src\dbgpt\serve\core\schemas.py`

```py
import logging  # 导入 logging 模块，用于记录日志
import sys  # 导入 sys 模块，用于访问与 Python 解释器相关的变量和函数
from typing import TYPE_CHECKING  # 导入 TYPE_CHECKING，用于类型检查

from fastapi import HTTPException, Request  # 导入 fastapi 中的 HTTPException 和 Request 类
from fastapi.exceptions import RequestValidationError  # 导入 fastapi 中的 RequestValidationError 异常
from fastapi.responses import JSONResponse  # 导入 fastapi 中的 JSONResponse 类

from dbgpt.core.schema.api import Result  # 从 dbgpt.core.schema.api 模块导入 Result 类

if sys.version_info < (3, 11):  # 如果 Python 版本低于 3.11
    try:
        from exceptiongroup import ExceptionGroup  # 尝试导入 exceptiongroup 模块中的 ExceptionGroup 类
    except ImportError:
        ExceptionGroup = None  # 如果导入失败，将 ExceptionGroup 设置为 None

if TYPE_CHECKING:  # 如果在类型检查模式下
    from fastapi import FastAPI  # 从 fastapi 中导入 FastAPI 类

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 对象


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Validation exception handler"""
    message = ""
    for error in exc.errors():  # 遍历异常中的错误列表
        loc = ".".join(list(map(str, error.get("loc"))))  # 获取错误位置，并转换为字符串形式
        message += loc + ":" + error.get("msg") + ";"  # 拼接错误位置和错误消息
    res = Result.failed(msg=message, err_code="E0001")  # 构建一个失败的 Result 对象
    logger.error(f"validation_exception_handler catch RequestValidationError: {res}")  # 记录错误日志
    return JSONResponse(status_code=400, content=res.to_dict())  # 返回 JSON 响应


async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler"""
    res = Result.failed(
        msg=str(exc.detail),  # 获取异常的详细信息作为消息
        err_code=str(exc.status_code),  # 获取异常的状态码作为错误码
    )
    logger.error(f"http_exception_handler catch HTTPException: {res}")  # 记录错误日志
    return JSONResponse(status_code=exc.status_code, content=res.to_dict())  # 返回 JSON 响应


async def common_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Common exception handler"""
    
    if ExceptionGroup and isinstance(exc, ExceptionGroup):  # 如果异常是 ExceptionGroup 的实例
        err_strs = []
        for e in exc.exceptions:  # 遍历 ExceptionGroup 中的各个异常
            err_strs.append(str(e))  # 将异常转换为字符串并添加到列表中
        err_msg = ";".join(err_strs)  # 使用 ';' 连接所有异常字符串
    else:
        err_msg = str(exc)  # 否则，直接使用异常的字符串表示
    
    res = Result.failed(
        msg=err_msg,  # 使用错误消息初始化 Result 对象
        err_code="E0003",  # 设置错误码为 E0003
    )
    logger.error(f"common_exception_handler catch Exception: {res}")  # 记录错误日志
    return JSONResponse(status_code=400, content=res.to_dict())  # 返回 JSON 响应


def add_exception_handler(app: "FastAPI"):
    """Add exception handler"""
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # 添加 RequestValidationError 异常处理器
    app.add_exception_handler(HTTPException, http_exception_handler)  # 添加 HTTPException 异常处理器
    app.add_exception_handler(Exception, common_exception_handler)  # 添加通用 Exception 异常处理器
```