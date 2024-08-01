# `.\DB-GPT-src\dbgpt\model\cluster\registry_impl\db_storage.py`

```py
# 从 datetime 模
    # 定义函数，返回一个 ModelInstanceStorageItem 实例，根据传入的 model 对象的属性
    # 构造一个 ModelInstanceStorageItem 实例，包括 model_name, host, port, weight,
    # check_healthy, healthy, enabled, prompt_template, last_heartbeat 等属性
    ) -> ModelInstanceStorageItem:
        return ModelInstanceStorageItem(
            model_name=model.model_name,
            host=model.host,
            port=model.port,
            weight=model.weight,
            check_healthy=model.check_healthy,
            healthy=model.healthy,
            enabled=model.enabled,
            prompt_template=model.prompt_template,
            last_heartbeat=model.last_heartbeat,
        )

    # 定义方法，根据给定的 storage_format 和 resource_id 构造查询对象
    def get_query_for_identifier(
        self,
        storage_format: ModelInstanceEntity,
        resource_id: ResourceIdentifier,
        **kwargs,
    ):
        # 从 kwargs 中获取 session 对象，若未提供则抛出异常
        session: Session = kwargs.get("session")
        if session is None:
            raise Exception("session is None")
        
        # 使用 session 构造查询对象，初始为 ModelInstanceEntity 表的查询对象
        query_obj = session.query(ModelInstanceEntity)
        
        # 遍历 resource_id 的属性字典，根据非空值筛选对应的查询条件并添加到 query_obj 中
        for key, value in resource_id.to_dict().items():
            if value is None:
                continue
            query_obj = query_obj.filter(getattr(ModelInstanceEntity, key) == value)
        
        # 返回构建好的查询对象
        return query_obj
```