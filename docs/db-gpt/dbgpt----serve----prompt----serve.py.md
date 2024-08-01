# `.\DB-GPT-src\dbgpt\serve\prompt\serve.py`

```py
# 导入 logging 模块，用于日志记录
import logging
# 导入类型提示相关的模块
from typing import List, Optional, Union
# 导入 SQLAlchemy 中的 URL 类
from sqlalchemy import URL

# 导入 dbgpt.component 中的 SystemApp 类
from dbgpt.component import SystemApp
# 导入 dbgpt.core 中的 PromptManager 类
from dbgpt.core import PromptManager
# 导入 dbgpt.serve.core 中的 BaseServe 类
from dbgpt.serve.core import BaseServe
# 导入 dbgpt.storage.metadata 中的 DatabaseManager 类
from dbgpt.storage.metadata import DatabaseManager

# 从当前包中导入的模块和变量
from .api.endpoints import init_endpoints, router
from .config import (
    APP_NAME,
    SERVE_APP_NAME,
    SERVE_APP_NAME_HUMP,
    SERVE_CONFIG_KEY_PREFIX,
    ServeConfig,
)
# 导入 prompt_template_adapter 模块中的 PromptTemplateAdapter 类
from .models.prompt_template_adapter import PromptTemplateAdapter

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)


class Serve(BaseServe):
    """Serve component

    Examples:

        Register the serve component to the system app

        .. code-block:: python

            from fastapi import FastAPI
            from dbgpt import SystemApp
            from dbgpt.core import PromptTemplate
            from dbgpt.serve.prompt.serve import Serve, SERVE_APP_NAME

            app = FastAPI()
            system_app = SystemApp(app)
            system_app.register(Serve, api_prefix="/api/v1/prompt")
            system_app.on_init()
            # Run before start hook
            system_app.before_start()

            prompt_serve = system_app.get_component(SERVE_APP_NAME, Serve)

            # Get the prompt manager
            prompt_manager = prompt_serve.prompt_manager
            prompt_manager.save(
                PromptTemplate(template="Hello {name}", input_variables=["name"]),
                prompt_name="prompt_name",
            )

        With your database url

        .. code-block:: python

            from fastapi import FastAPI
            from dbgpt import SystemApp
            from dbgpt.core import PromptTemplate
            from dbgpt.serve.prompt.serve import Serve, SERVE_APP_NAME

            app = FastAPI()
            system_app = SystemApp(app)
            system_app.register(
                Serve,
                api_prefix="/api/v1/prompt",
                db_url_or_db="sqlite:///:memory:",
                try_create_tables=True,
            )
            system_app.on_init()
            # Run before start hook
            system_app.before_start()

            prompt_serve = system_app.get_component(SERVE_APP_NAME, Serve)

            # Get the prompt manager
            prompt_manager = prompt_serve.prompt_manager
            prompt_manager.save(
                PromptTemplate(template="Hello {name}", input_variables=["name"]),
                prompt_name="prompt_name",
            )

    """

    # 设置组件的名称为 SERVE_APP_NAME
    name = SERVE_APP_NAME

    # Serve 类的初始化方法
    def __init__(
        self,
        system_app: SystemApp,
        api_prefix: Optional[str] = f"/api/v1/serve/{APP_NAME}",
        api_tags: Optional[List[str]] = None,
        db_url_or_db: Union[str, URL, DatabaseManager] = None,
        try_create_tables: Optional[bool] = False,
    ):
        # 如果未提供 api_tags 参数，则默认使用 [SERVE_APP_NAME_HUMP] 作为 api_tags
        if api_tags is None:
            api_tags = [SERVE_APP_NAME_HUMP]
        # 调用父类的初始化方法，传入系统应用、API 前缀、API 标签、数据库 URL 或数据库对象、是否尝试创建表
        super().__init__(
            system_app, api_prefix, api_tags, db_url_or_db, try_create_tables
        )
        # 初始化时将 _prompt_manager 设置为 None
        self._prompt_manager = None
        # 将 _db_manager 类型标注为可选的 DatabaseManager，初始值为 None
        self._db_manager: Optional[DatabaseManager] = None

    def init_app(self, system_app: SystemApp):
        # 如果应用已经初始化过，则直接返回
        if self._app_has_initiated:
            return
        # 将系统应用保存到 self._system_app 中
        self._system_app = system_app
        # 在系统应用中包含路由器，使用指定的 API 前缀和 API 标签
        self._system_app.app.include_router(
            router, prefix=self._api_prefix, tags=self._api_tags
        )
        # 初始化端点
        init_endpoints(self._system_app)
        # 标记应用已经初始化
        self._app_has_initiated = True

    @property
    def prompt_manager(self) -> PromptManager:
        """获取带有数据库存储的 Serve 应用的提示管理器"""
        return self._prompt_manager

    def on_init(self):
        """在应用启动前调用。

        可以在这里进行一些初始化操作。
        """
        # 在应用启动前导入自己的模块，确保模块在应用启动前已加载
        from .models.models import ServeEntity

    def before_start(self):
        """在应用启动前调用。

        可以在这里进行一些初始化操作。
        """
        # 在应用启动前导入自己的模块，确保模块在应用启动前已加载
        from dbgpt.core.interface.prompt import PromptManager
        from dbgpt.storage.metadata.db_storage import SQLAlchemyStorage
        from dbgpt.util.serialization.json_serialization import JsonSerializer

        from .models.models import ServeEntity

        # 创建或获取数据库管理器
        self._db_manager = self.create_or_get_db_manager()
        # 创建 Prompt 模板适配器
        storage_adapter = PromptTemplateAdapter()
        # 创建 JSON 序列化器
        serializer = JsonSerializer()
        # 创建 SQLAlchemy 存储对象，传入数据库管理器、实体类、存储适配器、序列化器
        storage = SQLAlchemyStorage(
            self._db_manager,
            ServeEntity,
            storage_adapter,
            serializer,
        )
        # 创建 PromptManager，并将其存储在 _prompt_manager 中
        self._prompt_manager = PromptManager(storage)
```