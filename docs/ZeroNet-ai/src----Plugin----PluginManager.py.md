# `ZeroNet\src\Plugin\PluginManager.py`

```py
import logging  # 导入日志模块
import os  # 导入操作系统模块
import sys  # 导入系统模块
import shutil  # 导入文件操作模块
import time  # 导入时间模块
from collections import defaultdict  # 导入默认字典模块

import importlib  # 导入动态加载模块
import json  # 导入 JSON 模块

from Debug import Debug  # 从 Debug 模块中导入 Debug 类
from Config import config  # 从 Config 模块中导入 config 对象
import plugins  # 导入插件模块


class PluginManager:
    def __init__(self):
        self.log = logging.getLogger("PluginManager")  # 初始化日志记录器
        self.path_plugins = os.path.abspath(os.path.dirname(plugins.__file__))  # 获取插件路径
        self.path_installed_plugins = config.data_dir + "/__plugins__"  # 获取已安装插件路径
        self.plugins = defaultdict(list)  # 注册的插件（键：类名，值：类的插件列表）
        self.subclass_order = {}  # 记录插件的加载顺序，以便在重新加载后保持顺序
        self.pluggable = {}  # 可插拔的
        self.plugin_names = []  # 已加载的插件名称
        self.plugins_updated = {}  # 自重启以来更新的插件列表
        self.plugins_rev = {}  # 已安装插件的修订号
        self.after_load = []  # 加载插件后执行的函数
        self.function_flags = {}  # 权限标志函数
        self.reloading = False  # 是否正在重新加载
        self.config_path = config.data_dir + "/plugins.json"  # 插件配置文件路径
        self.loadConfig()  # 加载配置文件

        self.config.setdefault("builtin", {})  # 设置默认内置配置

        sys.path.append(os.path.join(os.getcwd(), self.path_plugins))  # 将插件路径添加到系统路径中
        self.migratePlugins()  # 迁移插件

        if config.debug:  # 在文件更改时自动重新加载插件
            from Debug import DebugReloader
            DebugReloader.watcher.addCallback(self.reloadPlugins)

    def loadConfig(self):
        if os.path.isfile(self.config_path):  # 如果配置文件存在
            try:
                self.config = json.load(open(self.config_path, encoding="utf8"))  # 加载配置文件
            except Exception as err:
                self.log.error("Error loading %s: %s" % (self.config_path, err))  # 记录加载配置文件时的错误
                self.config = {}  # 设置配置为空
        else:
            self.config = {}  # 设置配置为空

    def saveConfig(self):
        f = open(self.config_path, "w", encoding="utf8")  # 打开配置文件
        json.dump(self.config, f, ensure_ascii=False, sort_keys=True, indent=2)  # 将配置写入文件
    # 迁移插件
    def migratePlugins(self):
        # 遍历插件路径下的所有文件夹
        for dir_name in os.listdir(self.path_plugins):
            # 如果文件夹名为"Mute"
            if dir_name == "Mute":
                # 输出日志信息，删除已弃用/重命名的插件
                self.log.info("Deleting deprecated/renamed plugin: %s" % dir_name)
                # 递归删除文件夹及其内容
                shutil.rmtree("%s/%s" % (self.path_plugins, dir_name))

    # -- 加载/卸载 --

    # 列出插件
    def listPlugins(self, list_disabled=False):
        # 初始化插件列表
        plugins = []
        # 遍历插件路径下的所有文件夹，并按名称排序
        for dir_name in sorted(os.listdir(self.path_plugins)):
            # 获取文件夹的完整路径
            dir_path = os.path.join(self.path_plugins, dir_name)
            # 获取插件名称（去除"disabled-"前缀）
            plugin_name = dir_name.replace("disabled-", "")
            # 如果文件夹名以"disabled"开头
            if dir_name.startswith("disabled"):
                # 设置插件为禁用状态
                is_enabled = False
            else:
                # 设置插件为启用状态
                is_enabled = True

            # 获取插件配置信息
            plugin_config = self.config["builtin"].get(plugin_name, {})
            # 如果配置中包含"enabled"字段
            if "enabled" in plugin_config:
                # 根据配置设置插件的启用状态
                is_enabled = plugin_config["enabled"]

            # 如果文件夹名为"__pycache__"或者不是文件夹
            if dir_name == "__pycache__" or not os.path.isdir(dir_path):
                continue  # 跳过
            # 如果文件夹名以"Debug"开头且不是调试模式
            if dir_name.startswith("Debug") and not config.debug:
                continue  # 只在调试模式下加载以"Debug"开头的模块
            # 如果插件未启用且不需要列出禁用的插件
            if not is_enabled and not list_disabled:
                continue  # 如果禁用则不加载

            # 构建插件信息字典
            plugin = {}
            plugin["source"] = "builtin"
            plugin["name"] = plugin_name
            plugin["dir_name"] = dir_name
            plugin["dir_path"] = dir_path
            plugin["inner_path"] = plugin_name
            plugin["enabled"] = is_enabled
            plugin["rev"] = config.rev
            plugin["loaded"] = plugin_name in self.plugin_names
            # 将插件信息字典添加到插件列表中
            plugins.append(plugin)

        # 获取已安装插件列表，并添加到插件列表中
        plugins += self.listInstalledPlugins(list_disabled)
        # 返回插件列表
        return plugins
    # 列出已安装的插件信息，可以选择是否列出已禁用的插件
    def listInstalledPlugins(self, list_disabled=False):
        # 初始化插件列表
        plugins = []

        # 遍历配置项中的地址和站点插件
        for address, site_plugins in sorted(self.config.items()):
            # 如果地址是 "builtin"，则跳过
            if address == "builtin":
                continue
            # 遍历站点插件中的插件路径和插件配置
            for plugin_inner_path, plugin_config in sorted(site_plugins.items()):
                # 获取插件是否启用的信息
                is_enabled = plugin_config.get("enabled", False)
                # 如果插件未启用且不需要列出已禁用的插件，则跳过
                if not is_enabled and not list_disabled:
                    continue
                # 获取插件名
                plugin_name = os.path.basename(plugin_inner_path)

                # 构建插件目录路径
                dir_path = "%s/%s/%s" % (self.path_installed_plugins, address, plugin_inner_path)

                # 创建插件信息字典
                plugin = {}
                plugin["source"] = address
                plugin["name"] = plugin_name
                plugin["dir_name"] = plugin_name
                plugin["dir_path"] = dir_path
                plugin["inner_path"] = plugin_inner_path
                plugin["enabled"] = is_enabled
                plugin["rev"] = plugin_config.get("rev", 0)
                plugin["loaded"] = plugin_name in self.plugin_names
                # 将插件信息添加到插件列表中
                plugins.append(plugin)

        # 返回插件列表
        return plugins

    # 加载所有插件
    # 加载所有插件
    def loadPlugins(self):
        # 初始化所有插件都已加载标志为 True
        all_loaded = True
        # 记录开始加载插件的时间
        s = time.time()
        # 遍历所有插件
        for plugin in self.listPlugins():
            # 记录调试信息，显示正在加载的插件名称和来源
            self.log.debug("Loading plugin: %s (%s)" % (plugin["name"], plugin["source"]))
            # 如果插件来源不是内置的
            if plugin["source"] != "builtin":
                # 将插件的名称和版本号添加到插件反向字典中
                self.plugins_rev[plugin["name"]] = plugin["rev"]
                # 获取插件所在目录的路径
                site_plugin_dir = os.path.dirname(plugin["dir_path"])
                # 如果插件所在目录不在系统路径中
                if site_plugin_dir not in sys.path:
                    # 将插件所在目录添加到系统路径中
                    sys.path.append(site_plugin_dir)
            try:
                # 尝试导入插件模块
                sys.modules[plugin["name"]] = __import__(plugin["dir_name"])
            except Exception as err:
                # 记录错误日志，显示插件加载错误的插件名称和错误信息
                self.log.error("Plugin %s load error: %s" % (plugin["name"], Debug.formatException(err)))
                # 将所有插件加载标志设为 False
                all_loaded = False
            # 如果插件名称不在插件名称列表中
            if plugin["name"] not in self.plugin_names:
                # 将插件名称添加到插件名称列表中
                self.plugin_names.append(plugin["name"])

        # 记录调试信息，显示加载插件所花费的时间
        self.log.debug("Plugins loaded in %.3fs" % (time.time() - s))
        # 遍历所有加载后的函数并执行
        for func in self.after_load:
            func()
        # 返回所有插件是否都加载成功的标志
        return all_loaded

    # 重新加载所有插件
plugin_manager = PluginManager()  # 创建插件管理器实例，作为单例模式使用

# -- Decorators --

# 接受插件到类装饰器
def acceptPlugins(base_class):
    class_name = base_class.__name__
    plugin_manager.pluggable[class_name] = base_class  # 将基类添加到插件管理器的可插拔类字典中
    if class_name in plugin_manager.plugins:  # 如果存在插件
        classes = plugin_manager.plugins[class_name][:]  # 复制当前的插件列表

        # 在重新加载后恢复子类的顺序
        if class_name in plugin_manager.subclass_order:
            classes = sorted(
                classes,
                key=lambda key:
                    plugin_manager.subclass_order[class_name].index(str(key))
                    if str(key) in plugin_manager.subclass_order[class_name]
                    else 9999
            )
        plugin_manager.subclass_order[class_name] = list(map(str, classes))  # 更新子类的顺序

        classes.reverse()
        classes.append(base_class)  # 将类本身添加到继承线的末尾
        plugined_class = type(class_name, tuple(classes), dict())  # 创建插件化的类
        plugin_manager.log.debug("New class accepts plugins: %s (Loaded plugins: %s)" % (class_name, classes))  # 记录日志
    else:  # 如果没有插件，直接使用原始类
        plugined_class = base_class
    return plugined_class


# 注册插件到类名装饰器
def registerTo(class_name):
    if config.debug and not plugin_manager.reloading:  # 如果是调试模式且不是重新加载
        import gc
        for obj in gc.get_objects():
            if type(obj).__name__ == class_name:
                raise Exception("Class %s instances already present in memory" % class_name)  # 抛出异常
                break

    plugin_manager.log.debug("New plugin registered to: %s" % class_name)  # 记录日志
    if class_name not in plugin_manager.plugins:
        plugin_manager.plugins[class_name] = []  # 如果类名不在插件管理器的插件字典中，添加空列表

    def classDecorator(self):
        plugin_manager.plugins[class_name].append(self)  # 将插件添加到对应类名的插件列表中
        return self
    return classDecorator


def afterLoad(func):
    plugin_manager.after_load.append(func)  # 将函数添加到插件管理器的加载后函数列表中
    return func
# - Example usage -

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 将 RequestPlugin 类注册到 Request 插件中
    @registerTo("Request")
    class RequestPlugin(object):

        # 定义 actionMainPage 方法，返回 "Hello MainPage!"
        def actionMainPage(self, path):
            return "Hello MainPage!"

    # 接受插件注册
    @acceptPlugins
    class Request(object):

        # 路由请求路径
        def route(self, path):
            # 获取请求路径对应的方法
            func = getattr(self, "action" + path, None)
            # 如果方法存在，则调用并返回结果
            if func:
                return func(path)
            # 如果方法不存在，则返回无法路由到该路径
            else:
                return "Can't route to", path

    # 打印请求路径为 "MainPage" 的路由结果
    print(Request().route("MainPage"))
```