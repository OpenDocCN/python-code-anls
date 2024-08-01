# `.\DB-GPT-src\dbgpt\serve\prompt\service\service.py`

```py
from typing import List, Optional

from dbgpt.component import BaseComponent, SystemApp
from dbgpt.serve.core import BaseService
from dbgpt.storage.metadata import BaseDao
from dbgpt.util.pagination_utils import PaginationResult

from ..api.schemas import ServeRequest, ServerResponse
from ..config import SERVE_CONFIG_KEY_PREFIX, SERVE_SERVICE_COMPONENT_NAME, ServeConfig
from ..models.models import ServeDao, ServeEntity


class Service(BaseService[ServeEntity, ServeRequest, ServerResponse]):
    """The service class for Prompt"""

    name = SERVE_SERVICE_COMPONENT_NAME

    def __init__(self, system_app: SystemApp, dao: Optional[ServeDao] = None):
        # 初始化 Service 类的实例
        self._system_app = None  # 初始化系统应用为 None
        self._serve_config: ServeConfig = None  # 初始化 ServeConfig 为 None
        self._dao: ServeDao = dao  # 初始化 DAO 对象
        super().__init__(system_app)

    def init_app(self, system_app: SystemApp) -> None:
        """Initialize the service

        Args:
            system_app (SystemApp): The system app
        """
        super().init_app(system_app)

        self._serve_config = ServeConfig.from_app_config(
            system_app.config, SERVE_CONFIG_KEY_PREFIX
        )  # 从系统配置中加载 ServeConfig 对象
        self._dao = self._dao or ServeDao(self._serve_config)  # 如果未提供 DAO 对象则创建一个新的 ServeDao
        self._system_app = system_app  # 设置系统应用

    @property
    def dao(self) -> BaseDao[ServeEntity, ServeRequest, ServerResponse]:
        """Returns the internal DAO."""
        return self._dao  # 返回内部 DAO 对象

    @property
    def config(self) -> ServeConfig:
        """Returns the internal ServeConfig."""
        return self._serve_config  # 返回内部 ServeConfig 对象

    def create(self, request: ServeRequest) -> ServerResponse:
        """Create a new Prompt entity

        Args:
            request (ServeRequest): The request

        Returns:
            ServerResponse: The response
        """

        if not request.user_name:
            request.user_name = self.config.default_user  # 如果请求中未提供用户名，则使用默认用户名
        if not request.sys_code:
            request.sys_code = self.config.default_sys_code  # 如果请求中未提供系统代码，则使用默认系统代码
        return super().create(request)  # 调用父类的 create 方法处理请求

    def update(self, request: ServeRequest) -> ServerResponse:
        """Update a Prompt entity

        Args:
            request (ServeRequest): The request

        Returns:
            ServerResponse: The response
        """
        # Build the query request from the request
        query_request = {
            "prompt_name": request.prompt_name,
            "sys_code": request.sys_code,
        }  # 构建查询请求对象

        return self.dao.update(query_request, update_request=request)  # 调用 DAO 对象的 update 方法更新数据

    def get(self, request: ServeRequest) -> Optional[ServerResponse]:
        """Get a Prompt entity

        Args:
            request (ServeRequest): The request

        Returns:
            ServerResponse: The response
        """
        # TODO: implement your own logic here
        # Build the query request from the request
        query_request = request  # 使用请求对象构建查询请求
        return self.dao.get_one(query_request)  # 调用 DAO 对象的 get_one 方法获取单个实体对象
    def delete(self, request: ServeRequest) -> None:
        """
        Delete a Prompt entity

        Args:
            request (ServeRequest): The request object containing prompt_name and sys_code
        """

        # TODO: implement your own logic here
        # Build the query request dictionary from the request object
        query_request = {
            "prompt_name": request.prompt_name,
            "sys_code": request.sys_code,
        }
        # Call the DAO's delete method with the constructed query_request
        self.dao.delete(query_request)

    def get_list(self, request: ServeRequest) -> List[ServerResponse]:
        """
        Get a list of Prompt entities

        Args:
            request (ServeRequest): The request object

        Returns:
            List[ServerResponse]: The list of server responses
        """
        # TODO: implement your own logic here
        # Directly use the provided request object as the query request
        query_request = request
        # Call the DAO's get_list method with the query_request and return its result
        return self.dao.get_list(query_request)

    def get_list_by_page(
        self, request: ServeRequest, page: int, page_size: int
    ) -> PaginationResult[ServerResponse]:
        """
        Get a list of Prompt entities by page

        Args:
            request (ServeRequest): The request object
            page (int): The page number
            page_size (int): The size of each page

        Returns:
            PaginationResult[ServerResponse]: The paginated result of server responses
        """
        # Use the provided request object as the query request
        query_request = request
        # Call the DAO's get_list_page method with query_request, page, and page_size
        return self.dao.get_list_page(query_request, page, page_size)
```