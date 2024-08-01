# `.\DB-GPT-src\dbgpt\core\awel\trigger\ext_http_trigger.py`

```py
"""Extends HTTP Triggers.

Supports more trigger types, such as RequestHttpTrigger.
"""
# 导入所需模块和类
from enum import Enum
from typing import Dict, List, Optional, Type, Union

# 导入 Starlette 请求对象
from starlette.requests import Request

# 导入国际化翻译函数
from dbgpt.util.i18n_utils import _

# 导入 AWEL 框架中的相关类和模块
from ..flow import IOField, OperatorCategory, OperatorType, Parameter, ViewMetadata
from ..operators.common_operator import MapOperator

# 导入 HTTP 触发器相关类
from .http_trigger import (
    _PARAMETER_ENDPOINT,
    _PARAMETER_MEDIA_TYPE,
    _PARAMETER_METHODS_ALL,
    _PARAMETER_RESPONSE_BODY,
    _PARAMETER_STATUS_CODE,
    _PARAMETER_STREAMING_RESPONSE,
    BaseHttpBody,
    HttpTrigger,
)

# 定义 RequestHttpTrigger 类，继承自 HttpTrigger 类
class RequestHttpTrigger(HttpTrigger):
    """Request http trigger for AWEL."""

    # 视图元数据，描述 RequestHttpTrigger 类的信息
    metadata = ViewMetadata(
        label=_("Request Http Trigger"),
        name="request_http_trigger",
        category=OperatorCategory.TRIGGER,
        operator_type=OperatorType.INPUT,
        description=_(
            "Trigger your workflow by http request, and parse the request body"
            " as a starlette Request"
        ),
        inputs=[],
        outputs=[
            IOField.build_from(
                _("Request Body"),
                "request_body",
                Request,
                description=_(
                    "The request body of the API endpoint, parse as a starlette"
                    " Request"
                ),
            ),
        ],
        parameters=[
            _PARAMETER_ENDPOINT.new(),
            _PARAMETER_METHODS_ALL.new(),
            _PARAMETER_STREAMING_RESPONSE.new(),
            _PARAMETER_RESPONSE_BODY.new(),
            _PARAMETER_MEDIA_TYPE.new(),
            _PARAMETER_STATUS_CODE.new(),
        ],
    )

    # 初始化方法，设置 RequestHttpTrigger 实例的属性
    def __init__(
        self,
        endpoint: str,
        methods: Optional[Union[str, List[str]]] = "POST",
        streaming_response: bool = False,
        http_response_body: Optional[Type[BaseHttpBody]] = None,
        response_media_type: Optional[str] = None,
        status_code: Optional[int] = 200,
        router_tags: Optional[List[str | Enum]] = None,
        **kwargs,
    ):
        """Initialize a RequestHttpTrigger."""
        # 如果未指定 router_tags，则默认为 ["AWEL RequestHttpTrigger"]
        if not router_tags:
            router_tags = ["AWEL RequestHttpTrigger"]
        
        # 调用父类构造函数，初始化 HttpTrigger 类的实例
        super().__init__(
            endpoint,
            methods,
            streaming_response=streaming_response,
            request_body=Request,
            http_response_body=http_response_body,
            response_media_type=response_media_type,
            status_code=status_code,
            router_tags=router_tags,
            register_to_app=True,
            **kwargs,
        )


# 定义 DictHTTPSender 类，继承自 MapOperator，用于 AWEL 框架中的 HTTP 发送操作
class DictHTTPSender(MapOperator[Dict, Dict]):
    """HTTP Sender operator for AWEL."""
    metadata = ViewMetadata(
        label=_("HTTP Sender"),  # 设置视图标签为"HTTP Sender"
        name="awel_dict_http_sender",  # 设置视图名称为"awel_dict_http_sender"
        category=OperatorCategory.SENDER,  # 设置视图类别为SENDER
        description=_("Send a HTTP request to a specified endpoint"),  # 设置视图描述为"Send a HTTP request to a specified endpoint"
        inputs=[  # 定义输入参数列表
            IOField.build_from(
                _("Request Body"),  # 输入字段标签为"Request Body"
                "request_body",  # 输入字段名称为"request_body"
                dict,  # 输入字段类型为字典
                description=_("The request body to send"),  # 输入字段描述为"The request body to send"
            )
        ],
        outputs=[  # 定义输出参数列表
            IOField.build_from(
                _("Response Body"),  # 输出字段标签为"Response Body"
                "response_body",  # 输出字段名称为"response_body"
                dict,  # 输出字段类型为字典
                description=_("The response body of the HTTP request"),  # 输出字段描述为"The response body of the HTTP request"
            )
        ],
        parameters=[  # 定义视图参数列表
            Parameter.build_from(
                _("HTTP Address"),  # 参数标签为"HTTP Address"
                _("address"),  # 参数名称为"address"
                type=str,  # 参数类型为字符串
                description=_("The address to send the HTTP request to"),  # 参数描述为"The address to send the HTTP request to"
            ),
            _PARAMETER_METHODS_ALL.new(),  # 引用预定义的参数方法
            _PARAMETER_STATUS_CODE.new(),  # 引用预定义的状态码参数
            Parameter.build_from(
                _("Timeout"),  # 参数标签为"Timeout"
                "timeout",  # 参数名称为"timeout"
                type=int,  # 参数类型为整数
                optional=True,  # 参数可选
                default=60,  # 默认值为60
                description=_("The timeout of the HTTP request in seconds"),  # 参数描述为"The timeout of the HTTP request in seconds"
            ),
            Parameter.build_from(
                _("Token"),  # 参数标签为"Token"
                "token",  # 参数名称为"token"
                type=str,  # 参数类型为字符串
                optional=True,  # 参数可选
                default=None,  # 默认值为None
                description=_("The token to use for the HTTP request"),  # 参数描述为"The token to use for the HTTP request"
            ),
            Parameter.build_from(
                _("Cookies"),  # 参数标签为"Cookies"
                "cookies",  # 参数名称为"cookies"
                type=str,  # 参数类型为字符串
                optional=True,  # 参数可选
                default=None,  # 默认值为None
                description=_("The cookies to use for the HTTP request"),  # 参数描述为"The cookies to use for the HTTP request"
            ),
        ],
    )
    
    def __init__(
        self,
        address: str,
        methods: Optional[str] = "GET",
        status_code: Optional[int] = 200,
        timeout: Optional[int] = 60,
        token: Optional[str] = None,
        cookies: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """Initialize a HTTPSender."""
        try:
            import aiohttp  # 导入aiohttp库，如果导入失败抛出ImportError异常
        except ImportError:
            raise ImportError(
                "aiohttp is required for HTTPSender, please install it with "
                "`pip install aiohttp`"
            )
        self._address = address  # 初始化HTTP请求地址
        self._methods = methods  # 初始化HTTP请求方法，默认为"GET"
        self._status_code = status_code  # 初始化期望的HTTP响应状态码，默认为200
        self._timeout = timeout  # 初始化HTTP请求超时时间，默认为60秒
        self._token = token  # 初始化用于HTTP请求的令牌
        self._cookies = cookies  # 初始化用于HTTP请求的cookies
        super().__init__(**kwargs)  # 调用父类初始化方法，传递额外的关键字参数
    async def map(self, request_body: Dict) -> Dict:
        """Send the request body to the specified address."""
        # 导入 aiohttp 库，用于发送异步 HTTP 请求
        import aiohttp

        # 根据请求方法选择不同的请求参数
        if self._methods in ["POST", "PUT"]:
            req_kwargs = {"json": request_body}  # 使用 JSON 格式发送请求体
        else:
            req_kwargs = {"params": request_body}  # 使用 URL 参数发送请求体
        method = self._methods or "GET"  # 确定请求方法，默认为 GET

        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"  # 添加授权头部信息

        # 使用 aiohttp 客户端会话发送请求
        async with aiohttp.ClientSession(
            headers=headers,
            cookies=self._cookies,
            timeout=aiohttp.ClientTimeout(total=self._timeout),  # 设置超时时间
        ) as session:
            async with session.request(
                method,
                self._address,  # 请求地址
                raise_for_status=False,  # 不抛出异常处理状态码
                **req_kwargs,  # 传递请求参数
            ) as response:
                status_code = response.status  # 获取响应状态码
                if status_code != self._status_code:
                    raise ValueError(
                        f"HTTP request failed with status code {status_code}"
                    )  # 如果状态码不符合预期，抛出异常

                response_body = await response.json()  # 解析 JSON 格式的响应体
                return response_body  # 返回响应体作为字典
```