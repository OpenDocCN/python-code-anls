# `.\DB-GPT-src\dbgpt\serve\datasource\service\service.py`

```py
# 导入必要的模块和类
import logging  # 导入日志记录模块
from typing import List, Optional  # 引入类型提示相关模块

from fastapi import HTTPException  # 导入FastAPI的HTTP异常类

# 导入私有配置和工具
from dbgpt._private.config import Config  # 导入配置模块
from dbgpt._private.pydantic import model_to_dict  #
    # 创建新的数据源实体
    def create(self, request: DatasourceServeRequest) -> DatasourceServeResponse:
        """Create a new Datasource entity

        Args:
            request (DatasourceServeRequest): The request

        Returns:
            DatasourceServeResponse: The response
        """
        # 根据数据源名称从 DAO 中获取数据源信息
        datasource = self._dao.get_by_names(request.db_name)
        # 如果数据源已存在，则抛出 HTTP 异常
        if datasource:
            raise HTTPException(
                status_code=400,
                detail=f"datasource name:{request.db_name} already exists",
            )
        try:
            # 获取请求中的数据库类型
            db_type = DBType.of_db_type(request.db_type)
            # 如果数据库类型不支持，则抛出 HTTP 异常
            if not db_type:
                raise HTTPException(
                    status_code=400, detail=f"Unsupported Db Type, {request.db_type}"
                )
            # 在 DAO 中创建新的数据源
            res = self._dao.create(request)

            # 异步执行嵌入操作
            executor = self._system_app.get_component(
                ComponentType.EXECUTOR_DEFAULT, ExecutorFactory
            ).create()  # type: ignore
            executor.submit(
                self._db_summary_client.db_summary_embedding,
                request.db_name,
                request.db_type,
            )
        except Exception as e:
            # 如果发生异常，则抛出带有错误信息的 ValueError
            raise ValueError("Add db connect info error!" + str(e))
        # 返回 DAO 创建数据源的结果
        return res

    # 更新数据源实体
    def update(self, request: DatasourceServeRequest) -> DatasourceServeResponse:
        """Create a new Datasource entity

        Args:
            request (DatasourceServeRequest): The request

        Returns:
            DatasourceServeResponse: The response
        """
        # 根据数据源名称从 DAO 中获取数据源信息
        datasources = self._dao.get_by_names(request.db_name)
        # 如果数据源不存在，则抛出 HTTP 异常
        if datasources is None:
            raise HTTPException(
                status_code=400,
                detail=f"there is no datasource name:{request.db_name} exists",
            )
        # 将请求对象转换为数据库配置对象
        db_config = DBConfig(**model_to_dict(request))
        # 调用本地数据库管理器编辑数据库配置信息
        if CFG.local_db_manager.edit_db(db_config):
            return DatasourceServeResponse(**model_to_dict(db_config))
        else:
            # 如果编辑失败，则抛出 HTTP 异常
            raise HTTPException(
                status_code=400,
                detail=f"update datasource name:{request.db_name} failed",
            )

    # 获取数据源实体
    def get(self, datasource_id: str) -> Optional[DatasourceServeResponse]:
        """Get a Flow entity

        Args:
            request (DatasourceServeRequest): The request

        Returns:
            DatasourceServeResponse: The response
        """
        # 根据数据源 ID 从 DAO 中获取单个数据源信息
        return self._dao.get_one({"id": datasource_id})
    # 删除数据源实体
    def delete(self, datasource_id: str) -> Optional[DatasourceServeResponse]:
        """Delete a Flow entity

        Args:
            datasource_id (str): The datasource_id

        Returns:
            DatasourceServeResponse: The data after deletion
        """
        # 获取数据源配置信息
        db_config = self._dao.get_one({"id": datasource_id})
        # 根据数据库名称生成向量名称
        vector_name = db_config.db_name + "_profile"
        # 创建向量存储配置对象
        vector_store_config = VectorStoreConfig(name=vector_name)
        # 创建向量存储连接器对象
        self._vector_connector = VectorStoreConnector(
            vector_store_type=CFG.VECTOR_STORE_TYPE,
            vector_store_config=vector_store_config,
        )
        # 删除指定向量名称的向量
        self._vector_connector.delete_vector_name(vector_name)
        # 如果存在数据源配置信息，则删除该数据源
        if db_config:
            self._dao.delete({"id": datasource_id})
        # 返回被删除的数据源配置信息
        return db_config

    # 列出数据源实体
    def list(self) -> List[DatasourceServeResponse]:
        """List the Flow entities.

        Returns:
            List[DatasourceServeResponse]: The list of responses
        """
        # 获取本地数据库管理器中的数据库列表
        db_list = CFG.local_db_manager.get_db_list()
        # 返回数据源实体的响应列表
        return [DatasourceServeResponse(**db) for db in db_list]
```