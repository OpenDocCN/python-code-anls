# `.\DB-GPT-src\dbgpt\serve\agent\app\controller.py`

```py
# 导入日志模块
import logging

# 导入FastAPI的APIRouter用于定义路由
from fastapi import APIRouter

# 导入私有配置模块Config
from dbgpt._private.config import Config

# 导入agent管理相关模块
from dbgpt.agent.core.agent_manage import get_agent_manager
from dbgpt.agent.resource.manage import get_resource_manager

# 导入LLM策略类型枚举
from dbgpt.agent.util.llm.llm import LLMStrategyType

# 导入结果模型
from dbgpt.app.openapi.api_view_model import Result

# 导入GPTS服务器相关模块
from dbgpt.serve.agent.app.gpts_server import available_llms

# 导入GPTS应用数据库相关模块
from dbgpt.serve.agent.db.gpts_app import (
    GptsApp,
    GptsAppCollectionDao,
    GptsAppDao,
    GptsAppQuery,
    GptsAppResponse,
)

# 导入团队模式相关模块
from dbgpt.serve.agent.team.base import TeamMode

# 创建Config对象实例
CFG = Config()

# 创建APIRouter对象实例
router = APIRouter()

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 创建GptsAppDao对象实例
gpts_dao = GptsAppDao()

# 创建GptsAppCollectionDao对象实例
collection_dao = GptsAppCollectionDao()


# 定义创建应用的API接口
@router.post("/v1/app/create")
async def create(gpts_app: GptsApp):
    try:
        return Result.succ(gpts_dao.create(gpts_app))
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"create app error: {ex}")


# 定义查询应用列表的API接口
@router.post("/v1/app/list", response_model=Result[GptsAppResponse])
async def app_list(query: GptsAppQuery):
    try:
        return Result.succ(gpts_dao.app_list(query, True))
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"query app error: {ex}")


# 定义查询应用详情的API接口
@router.post("/v1/app/detail")
async def app_list(gpts_app: GptsApp):
    try:
        return Result.succ(gpts_dao.app_detail(gpts_app.app_code))
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"query app error: {ex}")


# 定义编辑应用的API接口
@router.post("/v1/app/edit")
async def edit(gpts_app: GptsApp):
    try:
        return Result.succ(gpts_dao.edit(gpts_app))
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"edit app error: {ex}")


# 定义查询所有代理的API接口
@router.get("/v1/agents/list")
async def all_agents():
    try:
        return Result.succ(get_agent_manager().list_agents())
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"query agents error: {ex}")


# 定义删除应用的API接口
@router.post("/v1/app/remove")
async def delete(gpts_app: GptsApp):
    try:
        # 调用GptsAppDao的delete方法删除应用
        gpts_dao.delete(gpts_app.app_code, gpts_app.user_code, gpts_app.sys_code)
        return Result.succ(None)
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"delete app error: {ex}")


# 定义收藏应用的API接口
@router.post("/v1/app/collect", response_model=Result[str])
async def collect(gpts_app: GptsApp):
    try:
        # 调用GptsAppCollectionDao的collect方法收藏应用
        collection_dao.collect(gpts_app.app_code, gpts_app.user_code, gpts_app.sys_code)
        return Result.succ([])
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"collect app error: {ex}")


# 定义取消收藏应用的API接口
@router.post("/v1/app/uncollect", response_model=Result[str])
async def uncollect(gpts_app: GptsApp):
    try:
        # 调用GptsAppCollectionDao的uncollect方法取消收藏应用
        collection_dao.uncollect(
            gpts_app.app_code, gpts_app.user_code, gpts_app.sys_code
        )
        return Result.succ([])
    except Exception as ex:
        return Result.failed(code="E000X", msg=f"uncollect app error: {ex}")


# 定义查询团队模式列表的API接口
@router.get("/v1/team-mode/list")
async def team_mode_list():
    # 尝试执行以下代码块
    try:
        # 返回一个成功的结果对象，包含 TeamMode 中每个枚举值的列表
        return Result.succ([mode.value for mode in TeamMode])
    # 如果出现任何异常，捕获并赋给变量 ex
    except Exception as ex:
        # 返回一个失败的结果对象，指定错误代码和消息，包含异常信息
        return Result.failed(code="E000X", msg=f"query team mode list error: {ex}")
@router.get("/v1/resource-type/list")
async def team_mode_list():
    try:
        # 获取资源管理器并从中获取指定版本（v1）支持的所有资源
        resources = get_resource_manager().get_supported_resources(version="v1")
        # 将资源的键（资源类型）转换为列表并返回成功的结果
        return Result.succ(list(resources.keys()))
    except Exception as ex:
        # 如果出现异常，返回失败的结果，包括错误代码和具体错误消息
        return Result.failed(code="E000X", msg=f"query resource type list error: {ex}")


@router.get("/v1/llm-strategy/list")
async def llm_strategies():
    try:
        # 返回所有的 LLM 策略类型的值列表作为成功的结果
        return Result.succ([type.value for type in LLMStrategyType])
    except Exception as ex:
        # 如果出现异常，返回失败的结果，包括错误代码和具体错误消息
        return Result.failed(
            code="E000X", msg=f"query llm strategy type list error: {ex}"
        )


@router.get("/v1/llm-strategy/value/list")
async def llm_strategy_values(type: str):
    try:
        results = []
        # 根据传入的类型值匹配 LLMStrategyType 枚举，获取相应的可用 LLM 策略
        match type:
            case LLMStrategyType.Priority.value:
                results = await available_llms()
        # 返回获取的结果作为成功的结果
        return Result.succ(results)
    except Exception as ex:
        # 如果出现异常，返回失败的结果，包括错误代码和具体错误消息
        return Result.failed(
            code="E000X", msg=f"query llm strategy type list error: {ex}"
        )


@router.get("/v1/app/resources/list", response_model=Result[list[str]])
async def app_resources(
    type: str, name: str = None, user_code: str = None, sys_code: str = None
):
    """
    Get agent resources, such as db, knowledge, internet, plugin.
    """
    try:
        # 获取资源管理器并从中获取指定版本（v1）支持的所有资源
        resources = get_resource_manager().get_supported_resources("v1")
        # 根据传入的资源类型（type）获取相应的资源列表
        results = resources.get(type, [])
        # 返回获取的资源列表作为成功的结果
        return Result.succ(results)
    except Exception as ex:
        # 如果出现异常，返回失败的结果，包括错误代码和具体错误消息
        return Result.failed(code="E000X", msg=f"query app resources error: {ex}")
```