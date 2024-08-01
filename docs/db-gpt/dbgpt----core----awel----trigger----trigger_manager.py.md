# `.\DB-GPT-src\dbgpt\core\awel\trigger\trigger_manager.py`

```py
    def register_trigger(
        self, trigger: Any, system_app: SystemApp
    ) -> Optional[TriggerMetadata]:
        """Register a trigger to the HTTP trigger manager.

        Args:
            trigger (Any): The trigger object to register.
            system_app (SystemApp): The system application associated with the trigger.

        Returns:
            Optional[TriggerMetadata]: Metadata associated with the registered trigger, 
            or None if registration fails.
        """
    ) -> Optional[TriggerMetadata]:
        """Register a trigger to current manager."""
        from .http_trigger import HttpTrigger  # 导入HttpTrigger类

        if not isinstance(trigger, HttpTrigger):  # 检查trigger是否为HttpTrigger的实例
            raise ValueError(f"Current trigger {trigger} not an object of HttpTrigger")
        trigger_id = trigger.node_id  # 获取trigger的node_id作为唯一标识符
        if trigger_id not in self._trigger_map:  # 如果trigger_id不在_trigger_map中
            path = join_paths(self._router_prefix, trigger._endpoint)  # 构建路由路径
            methods = trigger._methods  # 获取trigger支持的HTTP方法列表
            # 检查路由是否已注册
            self._register_route_tables(path, methods)  # 注册路由到路由表
            try:
                if trigger.register_to_app():  # 尝试将触发器注册到应用程序
                    app = system_app.app  # 获取系统应用程序对象
                    if not app:
                        raise ValueError("System app not initialized")
                    # 将触发器挂载到应用程序，支持动态路由
                    trigger_metadata = trigger.mount_to_app(app, self._router_prefix)
                else:
                    trigger_metadata = trigger.mount_to_router(
                        self._router, self._router_prefix
                    )  # 将触发器挂载到路由器
                self._trigger_map[trigger_id] = (trigger, trigger_metadata)  # 将触发器和元数据添加到_trigger_map中
                return trigger_metadata  # 返回触发器元数据
            except Exception as e:
                self._unregister_route_tables(path, methods)  # 注销路由表中的路由
                raise e  # 抛出异常
        return None  # 如果trigger_id已经在_trigger_map中，返回None

    def unregister_trigger(self, trigger: Any, system_app: SystemApp) -> None:
        """Unregister a trigger to current manager."""
        from .http_trigger import HttpTrigger  # 导入HttpTrigger类

        if not isinstance(trigger, HttpTrigger):  # 检查trigger是否为HttpTrigger的实例
            raise ValueError(f"Current trigger {trigger} not an object of Trigger")
        trigger_id = trigger.node_id  # 获取trigger的node_id作为唯一标识符
        if trigger_id in self._trigger_map:  # 如果trigger_id在_trigger_map中
            if trigger.register_to_app():  # 如果触发器已注册到应用程序
                app = system_app.app  # 获取系统应用程序对象
                if not app:
                    raise ValueError("System app not initialized")
                trigger.remove_from_app(app, self._router_prefix)  # 从应用程序中移除触发器
                self._unregister_route_tables(
                    join_paths(self._router_prefix, trigger._endpoint), trigger._methods
                )  # 注销路由表中的路由
            del self._trigger_map[trigger_id]  # 从_trigger_map中删除该触发器

    def _init_app(self, system_app: SystemApp):
        # if self._inited:
        #     return None
        if not self.keep_running():  # 如果不应继续运行
            return
        logger.info(
            f"Include router {self._router} to prefix path {self._router_prefix}"
        )  # 记录信息，包含路由器和前缀路径
        app = system_app.app  # 获取系统应用程序对象
        if not app:
            raise RuntimeError("System app not initialized")
        app.include_router(self._router, prefix=self._router_prefix, tags=["AWEL"])  # 将路由器包含到应用程序中，使用指定的前缀路径和标签

    def keep_running(self) -> bool:
        """Whether keep running.

        Returns:
            bool: Whether keep running, True means keep running, False means stop.
        """
        return len(self._trigger_map) > 0  # 返回_trigger_map中是否有触发器，决定是否继续运行
    # 注册指定路径的路由表和方法
    def _register_route_tables(
        self, path: str, methods: Optional[Union[str, List[str]]]
    ):
        # 解析方法参数，确保方法以列表形式返回
        methods = self._parse_methods(methods)
        # 获取指定路径的路由表
        tables = self._router_tables[path]
        # 遍历方法列表，检查是否已注册，如已注册则抛出异常
        for m in methods:
            if m in tables:
                raise ValueError(f"Route {path} method {m} already registered")
            # 添加方法到路由表中
            tables.add(m)
        # 更新路由表
        self._router_tables[path] = tables

    # 取消注册指定路径的路由表和方法
    def _unregister_route_tables(
        self, path: str, methods: Optional[Union[str, List[str]]]
    ):
        # 解析方法参数，确保方法以列表形式返回
        methods = self._parse_methods(methods)
        # 获取指定路径的路由表
        tables = self._router_tables[path]
        # 遍历方法列表，从路由表中移除对应方法
        for m in methods:
            if m in tables:
                tables.remove(m)
        # 更新路由表
        self._router_tables[path] = tables

    # 解析方法参数，将其规范化为大写字母形式的方法列表
    def _parse_methods(self, methods: Optional[Union[str, List[str]]]) -> List[str]:
        # 如果方法参数为 None，则默认返回包含 "GET" 的列表
        if not methods:
            return ["GET"]
        # 如果方法参数为字符串，则将其转换为单元素列表
        elif isinstance(methods, str):
            return [methods]
        # 如果方法参数为列表，则将列表中的方法名都转换为大写形式
        return [m.upper() for m in methods]
class DefaultTriggerManager(TriggerManager, BaseComponent):
    """Default trigger manager for AWEL.

    Manage all trigger managers. Just support http trigger now.
    """

    name = ComponentType.AWEL_TRIGGER_MANAGER

    def __init__(self, system_app: SystemApp | None = None):
        """Initialize a DefaultTriggerManager."""
        self.system_app = system_app  # 初始化系统应用对象
        self._http_trigger: Optional[HttpTriggerManager] = None  # 初始化为可选的 HttpTriggerManager 对象
        super().__init__()  # 调用父类的初始化方法

    def init_app(self, system_app: SystemApp):
        """Initialize the trigger manager."""
        self.system_app = system_app  # 设置系统应用对象
        if system_app and self.system_app.app:
            self._http_trigger = HttpTriggerManager()  # 如果系统应用和其应用对象存在，则初始化 _http_trigger 为 HttpTriggerManager

    def register_trigger(
        self, trigger: Any, system_app: SystemApp
    ) -> Optional[TriggerMetadata]:
        """Register a trigger to current manager."""
        from .http_trigger import HttpTrigger

        if isinstance(trigger, HttpTrigger):  # 检查触发器类型是否为 HttpTrigger
            logger.info(f"Register trigger {trigger}")  # 记录注册的触发器信息
            if not self._http_trigger:
                raise ValueError("Http trigger manager not initialized")  # 如果 _http_trigger 未初始化则抛出异常
            return self._http_trigger.register_trigger(trigger, system_app)  # 调用 _http_trigger 的注册方法
        else:
            return None

    def unregister_trigger(self, trigger: Any, system_app: SystemApp) -> None:
        """Unregister a trigger to current manager."""
        from .http_trigger import HttpTrigger

        if isinstance(trigger, HttpTrigger):  # 检查触发器类型是否为 HttpTrigger
            logger.info(f"Unregister trigger {trigger}")  # 记录取消注册的触发器信息
            if not self._http_trigger:
                raise ValueError("Http trigger manager not initialized")  # 如果 _http_trigger 未初始化则抛出异常
            self._http_trigger.unregister_trigger(trigger, system_app)  # 调用 _http_trigger 的取消注册方法

    def after_register(self) -> None:
        """After register, init the trigger manager."""
        if self.system_app and self._http_trigger:
            self._http_trigger._init_app(self.system_app)  # 如果系统应用和 _http_trigger 都存在，则调用 _init_app 方法

    def keep_running(self) -> bool:
        """Whether keep running.

        Returns:
            bool: Whether keep running, True means keep running, False means stop.
        """
        if not self._http_trigger:
            return False  # 如果 _http_trigger 未初始化，则返回 False
        return self._http_trigger.keep_running()  # 调用 _http_trigger 的 keep_running 方法，并返回其结果
```