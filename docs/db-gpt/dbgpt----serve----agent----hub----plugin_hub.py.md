# `.\DB-GPT-src\dbgpt\serve\agent\hub\plugin_hub.py`

```py
import glob
import json
import logging
import os
import shutil
import tempfile
from typing import Any

from fastapi import UploadFile

from dbgpt.agent.core.schema import PluginStorageType
from dbgpt.agent.resource.tool.autogpt.plugins_util import scan_plugins, update_from_git
from dbgpt.configs.model_config import PLUGINS_DIR

from ..db.my_plugin_db import MyPluginDao, MyPluginEntity
from ..db.plugin_hub_db import PluginHubDao, PluginHubEntity

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象
Default_User = "default"  # 默认用户名称
DEFAULT_PLUGIN_REPO = "https://github.com/eosphoros-ai/DB-GPT-Plugins.git"  # 默认插件仓库的 URL
TEMP_PLUGIN_PATH = ""  # 临时插件路径


class PluginHub:
    def __init__(self, plugin_dir) -> None:
        self.hub_dao = PluginHubDao()  # 初始化插件中心数据访问对象
        self.my_plugin_dao = MyPluginDao()  # 初始化我的插件数据访问对象
        os.makedirs(plugin_dir, exist_ok=True)  # 创建插件目录，如果目录不存在则创建
        self.plugin_dir = plugin_dir  # 设置插件目录
        self.temp_hub_file_path = os.path.join(plugin_dir, "temp")  # 设置临时插件中心文件路径
    # 安装插件的方法，接受插件名称和可选的用户名作为参数
    def install_plugin(self, plugin_name: str, user_name: str = None):
        # 记录安装插件的信息到日志中
        logger.info(f"install_plugin {plugin_name}")
        
        # 根据插件名称从数据库中获取插件实体对象
        plugin_entity = self.hub_dao.get_by_name(plugin_name)
        
        # 如果获取到了插件实体对象
        if plugin_entity:
            # 检查插件的存储通道类型是否为 Git
            if plugin_entity.storage_channel == PluginStorageType.Git.value:
                try:
                    branch_name = None
                    authorization = None
                    
                    # 如果插件有下载参数，则解析参数中的分支名和授权信息
                    if plugin_entity.download_param:
                        download_param = json.loads(plugin_entity.download_param)
                        branch_name = download_param.get("branch_name")
                        authorization = download_param.get("authorization")
                    
                    # 从 Git 中下载插件文件，并返回文件名
                    file_name = self.__download_from_git(
                        plugin_entity.storage_url, branch_name, authorization
                    )

                    # 将插件的安装次数加一
                    plugin_entity.installed = plugin_entity.installed + 1

                    # 根据用户和插件名称从数据库中获取用户插件实体对象
                    my_plugin_entity = self.my_plugin_dao.get_by_user_and_plugin(
                        user_name, plugin_name
                    )
                    
                    # 如果没有找到用户插件实体对象，则创建一个新的
                    if my_plugin_entity is None:
                        my_plugin_entity = self.__build_my_plugin(plugin_entity)
                    
                    # 设置用户插件实体对象的文件名
                    my_plugin_entity.file_name = file_name
                    
                    # 如果提供了用户名，则设置用户相关信息到插件实体对象中
                    if user_name:
                        # TODO: 使用用户相关信息
                        my_plugin_entity.user_code = user_name
                        my_plugin_entity.user_name = user_name
                        my_plugin_entity.tenant = ""
                    else:
                        # 否则设置为默认用户
                        my_plugin_entity.user_code = Default_User

                    # 使用 hub_dao 创建会话对象
                    with self.hub_dao.session() as session:
                        # 如果用户插件实体对象的 id 为空，则添加到数据库中
                        if my_plugin_entity.id is None:
                            session.add(my_plugin_entity)
                        else:
                            # 否则更新到数据库中
                            session.merge(my_plugin_entity)
                        
                        # 更新插件实体对象到数据库中
                        session.merge(plugin_entity)
                
                # 捕获任何异常，并记录到日志中，然后重新抛出 ValueError 异常
                except Exception as e:
                    logger.error("install pluguin exception!", e)
                    raise ValueError(f"Install Plugin {plugin_name} Faild! {str(e)}")
            
            # 如果插件的存储通道类型不是 Git，则抛出 ValueError 异常
            else:
                raise ValueError(
                    f"Unsupport Storage Channel {plugin_entity.storage_channel}!"
                )
        
        # 如果没有获取到插件实体对象，则抛出 ValueError 异常
        else:
            raise ValueError(f"Can't Find Plugin {plugin_name}!")
    # 卸载指定插件，更新相关信息
    def uninstall_plugin(self, plugin_name, user):
        # 记录卸载插件的操作日志
        logger.info(f"uninstall_plugin:{plugin_name},{user}")
        # 获取插件实体对象
        plugin_entity = self.hub_dao.get_by_name(plugin_name)
        # 获取用户特定插件实体对象
        my_plugin_entity = self.my_plugin_dao.get_by_user_and_plugin(user, plugin_name)
        
        # 如果存在该插件实体对象
        if plugin_entity is not None:
            # 更新已安装插件计数
            plugin_entity.installed = plugin_entity.installed - 1
        
        # 使用数据库会话进行操作
        with self.hub_dao.session() as session:
            # 查询用户特定插件实体对象
            my_plugin_q = session.query(MyPluginEntity).filter(
                MyPluginEntity.name == plugin_name
            )
            # 如果提供了用户信息，进一步筛选
            if user:
                my_plugin_q.filter(MyPluginEntity.user_code == user)
            
            # 删除匹配条件的所有查询结果
            my_plugin_q.delete()
            
            # 如果存在插件实体对象，则合并到数据库会话中
            if plugin_entity is not None:
                session.merge(plugin_entity)
        
        # 如果存在插件实体对象
        if plugin_entity is not None:
            # 检查是否有其他插件仍在使用，决定是否删除插件文件
            plugin_infos = self.hub_dao.get_by_storage_url(plugin_entity.storage_url)
            have_installed = False
            for plugin_info in plugin_infos:
                if plugin_info.installed > 0:
                    have_installed = True
                    break
            # 如果没有其他插件在使用，则删除相关文件
            if not have_installed:
                plugin_repo_name = (
                    plugin_entity.storage_url.replace(".git", "")
                    .strip("/")
                    .split("/")[-1]
                )
                files = glob.glob(os.path.join(self.plugin_dir, f"{plugin_repo_name}*"))
                for file in files:
                    os.remove(file)
        else:
            # 如果不存在插件实体对象，则删除用户特定插件文件
            files = glob.glob(
                os.path.join(self.plugin_dir, f"{my_plugin_entity.file_name}")
            )
            for file in files:
                os.remove(file)

    # 从指定的 GitHub 仓库下载更新插件
    def __download_from_git(self, github_repo, branch_name, authorization):
        return update_from_git(self.plugin_dir, github_repo, branch_name, authorization)

    # 根据插件中心的插件构建用户的插件实体对象
    def __build_my_plugin(self, hub_plugin: PluginHubEntity) -> MyPluginEntity:
        my_plugin_entity = MyPluginEntity()
        my_plugin_entity.name = hub_plugin.name
        my_plugin_entity.type = hub_plugin.type
        my_plugin_entity.version = hub_plugin.version
        return my_plugin_entity

    # 从指定的 GitHub 仓库刷新插件中心的数据
    def refresh_hub_from_git(
        self,
        github_repo: str = None,
        branch_name: str = "main",
        authorization: str = None,
        ):
            # 记录日志信息，指示开始通过 Git 刷新插件中心
            logger.info("refresh_hub_by_git start!")
            # 从指定的 GitHub 仓库更新插件信息到临时文件路径
            update_from_git(
                self.temp_hub_file_path, github_repo, branch_name, authorization
            )
            # 扫描临时文件路径中的插件，返回插件对象列表
            git_plugins = scan_plugins(self.temp_hub_file_path)
            try:
                # 遍历临时文件路径中的每个插件对象
                for git_plugin in git_plugins:
                    # 根据插件名称从数据库中获取旧的插件信息
                    old_hub_info = self.hub_dao.get_by_name(git_plugin._name)
                    if old_hub_info:
                        # 如果数据库中已存在该插件信息，则使用旧的信息
                        plugin_hub_info = old_hub_info
                    else:
                        # 如果数据库中不存在该插件信息，则创建新的插件信息对象
                        plugin_hub_info = PluginHubEntity()
                        plugin_hub_info.type = ""
                        plugin_hub_info.storage_channel = PluginStorageType.Git.value
                        plugin_hub_info.storage_url = DEFAULT_PLUGIN_REPO
                        plugin_hub_info.author = getattr(git_plugin, "_author", "DB-GPT")
                        plugin_hub_info.email = getattr(git_plugin, "_email", "")
                        download_param = {}
                        # 如果指定了分支名称，则设置下载参数中的分支名称
                        if branch_name:
                            download_param["branch_name"] = branch_name
                        # 如果指定了授权信息且长度大于 0，则设置下载参数中的授权信息
                        if authorization and len(authorization) > 0:
                            download_param["authorization"] = authorization
                        # 将下载参数序列化为 JSON 字符串，并保存到插件信息对象中
                        plugin_hub_info.download_param = json.dumps(download_param)
                        # 设置插件安装状态为未安装
                        plugin_hub_info.installed = 0

                    # 设置插件信息对象的名称、版本号和描述信息
                    plugin_hub_info.name = git_plugin._name
                    plugin_hub_info.version = git_plugin._version
                    plugin_hub_info.description = git_plugin._description
                    # 将插件信息对象更新到数据库中
                    self.hub_dao.raw_update(plugin_hub_info)
            except Exception as e:
                # 捕获异常并抛出带有详细错误信息的 ValueError 异常
                raise ValueError(f"Update Agent Hub Db Info Faild!{str(e)}")
    # 异步函数，用于上传插件文件到指定目录，并扫描插件信息进行处理
    async def upload_my_plugin(self, doc_file: UploadFile, user: Any = Default_User):
        # 构建插件文件的完整路径
        file_path = os.path.join(self.plugin_dir, doc_file.filename)
        # 如果文件已存在，则删除之前的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # 创建临时文件并写入上传的文件内容
        tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.join(self.plugin_dir))
        with os.fdopen(tmp_fd, "wb") as tmp:
            tmp.write(await doc_file.read())
        
        # 将临时文件移动到插件目录下，覆盖同名文件
        shutil.move(
            tmp_path,
            os.path.join(self.plugin_dir, doc_file.filename),
        )

        # 扫描插件目录下的所有插件文件信息
        my_plugins = scan_plugins(self.plugin_dir, doc_file.filename)

        # 如果用户未指定或为空，则使用默认用户
        if user is None or len(user) <= 0:
            user = Default_User

        # 遍历扫描到的插件信息列表，更新或新增到数据库中
        for my_plugin in my_plugins:
            my_plugin_entity = self.my_plugin_dao.get_by_user_and_plugin(
                user, my_plugin._name
            )
            if my_plugin_entity is None:
                my_plugin_entity = MyPluginEntity()
            my_plugin_entity.name = my_plugin._name
            my_plugin_entity.version = my_plugin._version
            my_plugin_entity.type = "Personal"
            my_plugin_entity.user_code = user
            my_plugin_entity.user_name = user
            my_plugin_entity.tenant = ""
            my_plugin_entity.file_name = doc_file.filename
            self.my_plugin_dao.raw_update(my_plugin_entity)

    # 重新加载插件信息并返回扫描到的插件列表
    def reload_my_plugins(self):
        logger.info(f"load_plugins start!")
        return scan_plugins(self.plugin_dir)

    # 根据用户获取其插件信息
    def get_my_plugin(self, user: str):
        logger.info(f"get_my_plugin:{user}")
        # 如果用户为空，则使用默认用户
        if not user:
            user = Default_User
        # 根据用户查询数据库返回其插件信息
        return self.my_plugin_dao.get_by_user(user)
# 使用给定的插件目录创建一个 PluginHub 对象实例
plugin_hub = PluginHub(PLUGINS_DIR)
```