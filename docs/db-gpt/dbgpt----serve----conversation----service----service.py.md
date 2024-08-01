# `.\DB-GPT-src\dbgpt\serve\conversation\service\service.py`

```py
    def create_storage_conv(
    ) -> StorageConversation:
        # 获取会话存储和消息存储对象的引用
        conv_storage = self.conv_storage
        message_storage = self.message_storage
        # 如果会话存储或消息存储对象不存在，抛出运行时错误
        if not conv_storage or not message_storage:
            raise RuntimeError(
                "Can't get the conversation storage or message storage from current serve component."
            )
        # 如果请求是字典类型，转换为ServeRequest对象
        if isinstance(request, dict):
            request = ServeRequest(**request)
        # 创建StorageConversation对象并初始化
        storage_conv: StorageConversation = StorageConversation(
            conv_uid=request.conv_uid,
            chat_mode=request.chat_mode,
            user_name=request.user_name,
            sys_code=request.sys_code,
            conv_storage=conv_storage,
            message_storage=message_storage,
            load_message=load_message,
        )
        # 返回创建的StorageConversation对象
        return storage_conv

    def update(self, request: ServeRequest) -> ServerResponse:
        """Update a Conversation entity

        Args:
            request (ServeRequest): The request

        Returns:
            ServerResponse: The response
        """
        # TODO: implement your own logic here
        # 构建查询请求对象
        query_request = {
            # "id": request.id
        }
        # 调用DAO层的update方法执行更新操作
        return self.dao.update(query_request, update_request=request)

    def get(self, request: ServeRequest) -> Optional[ServerResponse]:
        """Get a Conversation entity

        Args:
            request (ServeRequest): The request

        Returns:
            ServerResponse: The response
        """
        # TODO: implement your own logic here
        # 直接使用请求对象作为查询请求
        query_request = request
        # 调用DAO层的get_one方法获取单个对象
        return self.dao.get_one(query_request)

    def delete(self, request: ServeRequest) -> None:
        """Delete current conversation and its messages

        Args:
            request (ServeRequest): The request
        """
        # 创建StorageConversation对象并执行删除操作
        conv: StorageConversation = self.create_storage_conv(request)
        conv.delete()

    def get_list(self, request: ServeRequest) -> List[ServerResponse]:
        """Get a list of Conversation entities

        Args:
            request (ServeRequest): The request

        Returns:
            List[ServerResponse]: The response
        """
        # TODO: implement your own logic here
        # 直接使用请求对象作为查询请求
        query_request = request
        # 调用DAO层的get_list方法获取对象列表
        return self.dao.get_list(query_request)

    def get_list_by_page(
        self, request: ServeRequest, page: int, page_size: int
    ) -> PaginationResult[ServerResponse]:
        """Get a list of Conversation entities by page

        Args:
            request (ServeRequest): The request
            page (int): The page number
            page_size (int): The page size

        Returns:
            List[ServerResponse]: The response
        """
        # 调用DAO层的get_conv_by_page方法获取分页结果
        return self.dao.get_conv_by_page(request, page, page_size)

    def get_history_messages(
        self, request: Union[ServeRequest, Dict[str, Any]]
    ) -> List[MessageVo]:
        """获取会话实体列表

        Args:
            request (ServeRequest): 请求对象

        Returns:
            List[ServerResponse]: 响应对象列表
        """
        # 创建存储会话对象
        conv: StorageConversation = self.create_storage_conv(request)
        # 初始化结果列表
        result = []
        # 获取视图消息列表
        messages = _append_view_messages(conv.messages)
        # 遍历消息列表
        for msg in messages:
            # 构造消息值对象
            result.append(
                MessageVo(
                    role=msg.type,               # 消息角色
                    context=msg.content,         # 消息内容
                    order=msg.round_index,       # 消息顺序索引
                    model_name=self.config.default_model,  # 默认模型名称
                )
            )
        # 返回结果列表
        return result
```