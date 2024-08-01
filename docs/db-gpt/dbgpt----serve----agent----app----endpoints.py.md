# `.\DB-GPT-src\dbgpt\serve\agent\app\endpoints.py`

```py
# 引入必要的类型提示
from typing import Optional

# 引入FastAPI框架中的APIRouter和Query类
from fastapi import APIRouter, Query

# 从dbgpt.serve.agent.db.gpts_app模块中导入需要的类
from dbgpt.serve.agent.db.gpts_app import (
    GptsApp,
    GptsAppCollectionDao,
    GptsAppDao,
    GptsAppQuery,
)

# 从dbgpt.serve.core模块中导入Result类
from dbgpt.serve.core import Result

# 创建一个APIRouter实例，用于定义路由
router = APIRouter()

# 创建GptsAppDao实例，用于处理应用程序的数据访问
gpts_dao = GptsAppDao()

# 创建GptsAppCollectionDao实例，用于处理应用程序集合的数据访问
collection_dao = GptsAppCollectionDao()

# 定义GET方法的路由'/v2/serve/apps'，用于获取应用程序列表
@router.get("/v2/serve/apps")
async def app_list(
    user_name: Optional[str] = Query(default=None, description="user name"),
    sys_code: Optional[str] = Query(default=None, description="system code"),
    is_collected: Optional[str] = Query(default=None, description="system code"),
    page: int = Query(default=1, description="current page"),
    page_size: int = Query(default=20, description="page size"),
):
    try:
        # 创建GptsAppQuery对象，用于查询参数的封装
        query = GptsAppQuery(
            page_no=page, page_size=page_size, is_collected=is_collected
        )
        # 调用gpts_dao的app_list方法，查询应用程序列表，并返回成功的Result对象
        return Result.succ(gpts_dao.app_list(query, True))
    except Exception as ex:
        # 捕获异常，并返回失败的Result对象，包含错误代码和异常信息
        return Result.failed(err_code="E000X", msg=f"query app error: {ex}")

# 定义GET方法的路由'/v2/serve/apps/{app_id}'，用于获取特定应用程序的详细信息
@router.get("/v2/serve/apps/{app_id}")
async def app_detail(app_id: str):
    try:
        # 调用gpts_dao的app_detail方法，查询特定应用程序的详细信息，并返回成功的Result对象
        return Result.succ(gpts_dao.app_detail(app_id))
    except Exception as ex:
        # 捕获异常，并返回失败的Result对象，包含错误代码和异常信息
        return Result.failed(err_code="E000X", msg=f"query app error: {ex}")

# 定义PUT方法的路由'/v2/serve/apps/{app_id}'，用于更新特定应用程序的信息
@router.put("/v2/serve/apps/{app_id}")
async def app_update(app_id: str, gpts_app: GptsApp):
    try:
        # 调用gpts_dao的edit方法，编辑特定应用程序的信息，并返回成功的Result对象
        return Result.succ(gpts_dao.edit(gpts_app))
    except Exception as ex:
        # 捕获异常，并返回失败的Result对象，包含错误代码和异常信息
        return Result.failed(err_code="E000X", msg=f"edit app error: {ex}")

# 定义POST方法的路由'/v2/serve/apps'，用于创建新的应用程序
@router.post("/v2/serve/apps")
async def app_create(gpts_app: GptsApp):
    try:
        # 调用gpts_dao的create方法，创建新的应用程序，并返回成功的Result对象
        return Result.succ(gpts_dao.create(gpts_app))
    except Exception as ex:
        # 捕获异常，并返回失败的Result对象，包含错误代码和异常信息
        return Result.failed(err_code="E000X", msg=f"edit app error: {ex}")

# 定义DELETE方法的路由'/v2/serve/apps/{app_id}'，用于删除特定应用程序
@router.delete("/v2/serve/apps/{app_id}")
async def app_delete(app_id: str, user_code: Optional[str], sys_code: Optional[str]):
    try:
        # 调用gpts_dao的delete方法，删除特定应用程序，并返回成功的Result对象
        gpts_dao.delete(app_id, user_code, sys_code)
        return Result.succ([])
    except Exception as ex:
        # 捕获异常，并返回失败的Result对象，包含错误代码和异常信息
        return Result.failed(err_code="E000X", msg=f"delete app error: {ex}")
```