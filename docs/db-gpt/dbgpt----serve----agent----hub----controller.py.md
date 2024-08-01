# `.\DB-GPT-src\dbgpt\serve\agent\hub\controller.py`

```py
import logging  # 导入日志模块，用于记录程序运行时的信息
from abc import ABC  # 导入抽象基类 ABC，用于定义抽象类
from typing import List  # 导入类型提示 List，用于声明列表类型

from fastapi import APIRouter, Body, File, UploadFile  # 导入 FastAPI 相关模块

from dbgpt.agent.resource.tool.autogpt.plugins_util import scan_plugins  # 导入插件扫描函数
from dbgpt.agent.resource.tool.pack import AutoGPTPluginToolPack  # 导入插件工具包类
from dbgpt.app.openapi.api_view_model import Result  # 导入结果模型类
from dbgpt.component import BaseComponent, ComponentType, SystemApp  # 导入组件基类、组件类型枚举、系统应用类
from dbgpt.configs.model_config import PLUGINS_DIR  # 导入插件目录配置
from dbgpt.serve.agent.db.plugin_hub_db import PluginHubEntity  # 导入插件集线器实体类
from dbgpt.serve.agent.hub.plugin_hub import plugin_hub  # 导入插件集线器实例
from dbgpt.serve.agent.model import (  # 导入模型相关类
    PagenationFilter,
    PagenationResult,
    PluginHubFilter,
    PluginHubParam,
)

from ..db import MyPluginEntity  # 导入自定义插件实体类
from ..model import MyPluginVO, PluginHubVO  # 导入自定义插件视图对象类

router = APIRouter()  # 创建 FastAPI 的路由器对象
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


class ModulePlugin(BaseComponent, ABC):
    name = ComponentType.PLUGIN_HUB  # 定义组件名称为 PLUGIN_HUB

    def __init__(self):
        # 加载插件
        self.refresh_plugins()

    def init_app(self, system_app: SystemApp):
        system_app.app.include_router(router, prefix="/api", tags=["Agent"])  # 将路由器添加到系统应用中

    def refresh_plugins(self):
        self.plugins = scan_plugins(PLUGINS_DIR)  # 扫描指定目录下的插件
        self.tools = AutoGPTPluginToolPack(PLUGINS_DIR)  # 初始化插件工具包
        self.tools.preload_resource()  # 预加载资源


module_plugin = ModulePlugin()  # 创建模块插件实例


@router.post("/v1/agent/hub/update", response_model=Result[str])
async def plugin_hub_update(update_param: PluginHubParam = Body()):
    logger.info(f"plugin_hub_update:{update_param.__dict__}")  # 记录更新插件集线器的日志信息
    try:
        branch = (
            update_param.branch
            if update_param.branch is not None and len(update_param.branch) > 0
            else "main"
        )
        authorization = (
            update_param.authorization
            if update_param.authorization is not None and len(update_param.authorization) > 0
            else None
        )
        # TODO change it to async
        plugin_hub.refresh_hub_from_git(update_param.url, branch, authorization)  # 从 Git 刷新插件集线器
        return Result.succ(None)  # 返回成功的结果对象
    except Exception as e:
        logger.error("Agent Hub Update Error!", e)  # 记录更新插件集线器出错时的日志信息
        return Result.failed(code="E0020", msg=f"Agent Hub Update Error! {e}")  # 返回失败的结果对象


@router.post("/v1/agent/query", response_model=Result[dict])
async def get_agent_list(filter: PagenationFilter[PluginHubFilter] = Body()):
    logger.info(f"get_agent_list:{filter.__dict__}")  # 记录获取代理列表的日志信息
    filter_enetity: PluginHubEntity = PluginHubEntity()  # 创建插件集线器实体对象
    if filter.filter:
        attrs = vars(filter.filter)  # 获取原始对象的属性字典
        for attr, value in attrs.items():
            setattr(filter_enetity, attr, value)  # 设置拷贝对象的属性值

    datas, total_pages, total_count = plugin_hub.hub_dao.list(
        filter_enetity, filter.page_index, filter.page_size
    )  # 查询插件集线器中的数据
    result: PagenationResult[PluginHubVO] = PagenationResult[PluginHubVO]()  # 创建分页结果对象
    result.page_index = filter.page_index  # 设置分页索引
    result.page_size = filter.page_size  # 设置分页大小
    result.total_page = total_pages  # 设置总页数
    result.total_row_count = total_count  # 设置总行数
    result.datas = PluginHubEntity.to_vo(datas)  # 将数据转换为视图对象
    # 调用 result 对象的 to_dic() 方法，将其转换为字典格式
    return Result.succ(result.to_dic())
# 定义一个路由，处理 POST 请求，路径为 "/v1/agent/my"，返回类型为 Result[List[MyPluginVO]]
async def my_agents(user: str = None):
    # 记录信息日志，指示正在处理 my_agents 请求，并打印用户信息
    logger.info(f"my_agents:{user}")
    # 通过 plugin_hub 获取特定用户的插件信息
    agents = plugin_hub.get_my_plugin(user)
    # 将 MyPluginEntity 对象转换为 MyPluginVO 对象列表
    agent_dicts = MyPluginEntity.to_vo(agents)
    # 返回成功的 Result 对象，包含插件信息的 VO 列表
    return Result.succ(agent_dicts)


# 定义一个路由，处理 POST 请求，路径为 "/v1/agent/install"，返回类型为 Result[str]
async def agent_install(plugin_name: str, user: str = None):
    # 记录信息日志，指示正在安装插件，并打印插件名称和用户信息
    logger.info(f"agent_install:{plugin_name},{user}")
    try:
        # 调用 plugin_hub 安装指定插件到指定用户
        plugin_hub.install_plugin(plugin_name, user)
        # 刷新模块的插件列表
        module_plugin.refresh_plugins()
        # 返回成功的 Result 对象，不包含具体数据
        return Result.succ(None)
    except Exception as e:
        # 记录错误日志，指示插件安装过程中出错，并打印异常信息
        logger.error("Plugin Install Error!", e)
        # 返回失败的 Result 对象，包含错误代码和具体错误信息
        return Result.failed(code="E0021", msg=f"Plugin Install Error {e}")


# 定义一个路由，处理 POST 请求，路径为 "/v1/agent/uninstall"，返回类型为 Result[str]
async def agent_uninstall(plugin_name: str, user: str = None):
    # 记录信息日志，指示正在卸载插件，并打印插件名称和用户信息
    logger.info(f"agent_uninstall:{plugin_name},{user}")
    try:
        # 调用 plugin_hub 卸载指定插件从指定用户
        plugin_hub.uninstall_plugin(plugin_name, user)
        # 刷新模块的插件列表
        module_plugin.refresh_plugins()
        # 返回成功的 Result 对象，不包含具体数据
        return Result.succ(None)
    except Exception as e:
        # 记录错误日志，指示插件卸载过程中出错，并打印异常信息
        logger.error("Plugin Uninstall Error!", e)
        # 返回失败的 Result 对象，包含错误代码和具体错误信息
        return Result.failed(code="E0022", msg=f"Plugin Uninstall Error {e}")


# 定义一个路由，处理 POST 请求，路径为 "/v1/personal/agent/upload"，返回类型为 Result[str]
async def personal_agent_upload(doc_file: UploadFile = File(...), user: str = None):
    # 记录信息日志，指示正在上传个人插件，并打印文件名和用户信息
    logger.info(f"personal_agent_upload:{doc_file.filename},{user}")
    try:
        # 调用 plugin_hub 上传用户的个人插件文件
        await plugin_hub.upload_my_plugin(doc_file, user)
        # 刷新模块的插件列表
        module_plugin.refresh_plugins()
        # 返回成功的 Result 对象，不包含具体数据
        return Result.succ(None)
    except Exception as e:
        # 记录错误日志，指示上传个人插件过程中出错，并打印异常信息
        logger.error("Upload Personal Plugin Error!", e)
        # 返回失败的 Result 对象，包含错误代码和具体错误信息
        return Result.failed(code="E0023", msg=f"Upload Personal Plugin Error {e}")
```