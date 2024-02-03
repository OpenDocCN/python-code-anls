# `.\AutoGPT\autogpts\autogpt\autogpt\core\plugin\simple.py`

```py
from importlib import import_module
from typing import TYPE_CHECKING

from autogpt.core.plugin.base import (
    PluginLocation,
    PluginService,
    PluginStorageFormat,
    PluginStorageRoute,
)

if TYPE_CHECKING:
    from autogpt.core.plugin.base import PluginType

# 定义一个简单的插件服务类，继承自插件服务基类
class SimplePluginService(PluginService):
    # 静态方法：根据插件位置获取插件对象
    @staticmethod
    def get_plugin(plugin_location: dict | PluginLocation) -> "PluginType":
        """Get a plugin from a plugin location."""
        # 如果插件位置是字典类型，则转换为插件位置对象
        if isinstance(plugin_location, dict):
            plugin_location = PluginLocation.parse_obj(plugin_location)
        # 根据存储格式加载插件
        if plugin_location.storage_format == PluginStorageFormat.WORKSPACE:
            return SimplePluginService.load_from_workspace(
                plugin_location.storage_route
            )
        elif plugin_location.storage_format == PluginStorageFormat.INSTALLED_PACKAGE:
            return SimplePluginService.load_from_installed_package(
                plugin_location.storage_route
            )
        else:
            raise NotImplementedError(
                "Plugin storage format %s is not implemented."
                % plugin_location.storage_format
            )

    ####################################
    # Low-level storage format loaders #
    ####################################
    # 静态方法：从文件路径加载插件
    @staticmethod
    def load_from_file_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from a file path."""
        # TODO: Define an on disk storage format and implement this.
        #   Can pull from existing zip file loading implementation
        raise NotImplementedError("Loading from file path is not implemented.")

    # 静态方法：从导入路径加载插件
    @staticmethod
    def load_from_import_path(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from an import path."""
        module_path, _, class_name = plugin_route.rpartition(".")
        return getattr(import_module(module_path), class_name)

    @staticmethod
    def resolve_name_to_path(
        plugin_route: PluginStorageRoute, path_type: str
    ) -> PluginStorageRoute:
        """Resolve a plugin name to a plugin path."""
        # 解析插件名称到插件路径的函数
        # TODO: Implement a discovery system for finding plugins by name from known
        #   storage locations. E.g. if we know that path_type is a file path, we can
        #   search the workspace for it. If it's an import path, we can check the core
        #   system and the auto_gpt_plugins package.
        # TODO: 实现一个发现系统，通过已知的存储位置按名称查找插件。例如，如果我们知道 path_type 是文件路径，我们可以在工作区中搜索它。如果是导入路径，我们可以检查核心系统和 auto_gpt_plugins 包。
        raise NotImplementedError("Resolving plugin name to path is not implemented.")
        # 抛出未实现错误，提示插件名称到路径的解析功能尚未实现

    #####################################
    # High-level storage format loaders #
    #####################################

    @staticmethod
    def load_from_workspace(plugin_route: PluginStorageRoute) -> "PluginType":
        """Load a plugin from the workspace."""
        # 从工作区加载插件的静态方法
        plugin = SimplePluginService.load_from_file_path(plugin_route)
        # 调用 SimplePluginService 类的 load_from_file_path 方法加载插件
        return plugin
        # 返回加载的插件对象

    @staticmethod
    def load_from_installed_package(plugin_route: PluginStorageRoute) -> "PluginType":
        # 从已安装包加载插件的静态方法
        plugin = SimplePluginService.load_from_import_path(plugin_route)
        # 调用 SimplePluginService 类的 load_from_import_path 方法加载插件
        return plugin
        # 返回加载的插件对象
```