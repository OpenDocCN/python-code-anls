# `ZeroNet\plugins\UiPluginManager\UiPluginManagerPlugin.py`

```
# 导入所需的模块
import io
import os
import json
import shutil
import time

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Config 模块中导入 config 变量
from Config import config
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 Translate 模块中导入 Translate 类
from Translate import Translate
# 从 util.Flag 模块中导入 flag 变量
from util.Flag import flag

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 如果 "_" 不在当前作用域中
if "_" not in locals():
    # 使用 Translate 类翻译指定路径下的语言文件
    _ = Translate(plugin_dir + "/languages/")

# 将非字符串、整数、浮点数类型的值转换为字符串类型
def restrictDictValues(input_dict):
    allowed_types = (int, str, float)
    return {
        key: val if type(val) in allowed_types else str(val)
        for key, val in input_dict.items()
    }

# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 定义 actionWrapper 方法
    def actionWrapper(self, path, extra_headers=None):
        # 如果路径去除首尾斜杠后不等于 "Plugins"
        if path.strip("/") != "Plugins":
            # 调用父类的 actionWrapper 方法
            return super(UiRequestPlugin, self).actionWrapper(path, extra_headers)

        # 如果 extra_headers 为空
        if not extra_headers:
            # 初始化 extra_headers 为空字典
            extra_headers = {}

        # 获取脚本的 nonce 值
        script_nonce = self.getScriptNonce()

        # 发送头部信息
        self.sendHeader(extra_headers=extra_headers, script_nonce=script_nonce)
        # 获取站点信息
        site = self.server.site_manager.get(config.homepage)
        # 返回迭代器
        return iter([super(UiRequestPlugin, self).renderWrapper(
            site, path, "uimedia/plugins/plugin_manager/plugin_manager.html",
            "Plugin Manager", extra_headers, show_loadingscreen=False, script_nonce=script_nonce
        )])
    # 定义处理 UI 媒体文件的方法，接受路径和其他参数
    def actionUiMedia(self, path, *args, **kwargs):
        # 如果路径以指定字符串开头
        if path.startswith("/uimedia/plugins/plugin_manager/"):
            # 替换路径中的字符串，得到文件路径
            file_path = path.replace("/uimedia/plugins/plugin_manager/", plugin_dir + "/media/")
            # 如果处于调试模式且文件路径以特定后缀结尾
            if config.debug and (file_path.endswith("all.js") or file_path.endswith("all.css")):
                # 如果处于调试模式，将 *.css 合并到 all.css，将 *.js 合并到 all.js
                from Debug import DebugMedia
                DebugMedia.merge(file_path)

            # 如果文件路径以 .js 结尾
            if file_path.endswith("js"):
                # 读取文件内容并进行 JS 模式的数据转换，编码为 utf8
                data = _.translateData(open(file_path).read(), mode="js").encode("utf8")
            # 如果文件路径以 .html 结尾
            elif file_path.endswith("html"):
                # 读取文件内容并进行 HTML 模式的数据转换，编码为 utf8
                data = _.translateData(open(file_path).read(), mode="html").encode("utf8")
            else:
                # 否则直接读取文件内容，以二进制形式
                data = open(file_path, "rb").read()

            # 调用 actionFile 方法，传入文件路径、文件对象和文件大小，并返回结果
            return self.actionFile(file_path, file_obj=io.BytesIO(data), file_size=len(data))
        else:
            # 否则调用父类的 actionUiMedia 方法处理
            return super(UiRequestPlugin, self).actionUiMedia(path)
# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 标记为管理员权限的方法，用于获取插件列表
    @flag.admin
    def actionPluginList(self, to):
        # 初始化插件列表
        plugins = []
        # 遍历所有插件，包括禁用的插件
        for plugin in PluginManager.plugin_manager.listPlugins(list_disabled=True):
            # 获取插件信息文件的路径
            plugin_info_path = plugin["dir_path"] + "/plugin_info.json"
            # 初始化插件信息字典
            plugin_info = {}
            # 如果插件信息文件存在
            if os.path.isfile(plugin_info_path):
                try:
                    # 读取插件信息文件内容
                    plugin_info = json.load(open(plugin_info_path))
                except Exception as err:
                    # 如果读取出错，记录错误日志
                    self.log.error(
                        "Error loading plugin info for %s: %s" %
                        (plugin["name"], Debug.formatException(err))
                    )
            # 如果插件信息存在
            if plugin_info:
                # 限制插件信息字典的值，以确保安全性
                plugin_info = restrictDictValues(plugin_info)
                # 将插件信息添加到插件字典中
                plugin["info"] = plugin_info

            # 如果插件来源不是内置的
            if plugin["source"] != "builtin":
                # 获取插件来源的站点
                plugin_site = self.server.sites.get(plugin["source"])
                # 如果插件站点存在
                if plugin_site:
                    try:
                        # 加载插件站点的信息文件
                        plugin_site_info = plugin_site.storage.loadJson(plugin["inner_path"] + "/plugin_info.json")
                        # 限制插件站点信息字典的值，以确保安全性
                        plugin_site_info = restrictDictValues(plugin_site_info)
                        # 将插件站点信息添加到插件字典中
                        plugin["site_info"] = plugin_site_info
                        # 获取插件站点的标题
                        plugin["site_title"] = plugin_site.content_manager.contents["content.json"].get("title")
                        # 构建插件键值
                        plugin_key = "%s/%s" % (plugin["source"], plugin["inner_path"])
                        # 检查插件是否已更新
                        plugin["updated"] = plugin_key in PluginManager.plugin_manager.plugins_updated
                    except Exception:
                        pass

            # 将插件添加到插件列表中
            plugins.append(plugin)

        # 返回包含插件列表的字典
        return {"plugins": plugins}

    # 标记为管理员权限和不支持多用户的方法
    @flag.admin
    @flag.no_multiuser
    # 设置动作插件的配置信息
    def actionPluginConfigSet(self, to, source, inner_path, key, value):
        # 获取插件管理器实例
        plugin_manager = PluginManager.plugin_manager
        # 获取所有插件列表，包括禁用的
        plugins = plugin_manager.listPlugins(list_disabled=True)
        # 初始化插件变量
        plugin = None
        # 遍历插件列表，查找匹配的插件
        for item in plugins:
            if item["source"] == source and item["inner_path"] in (inner_path, "disabled-" + inner_path):
                plugin = item
                break

        # 如果未找到匹配的插件，返回错误信息
        if not plugin:
            return {"error": "Plugin not found"}

        # 获取插件配置信息
        config_source = plugin_manager.config.setdefault(source, {})
        config_plugin = config_source.setdefault(inner_path, {})

        # 如果配置键存在且值为 None，则删除该键
        if key in config_plugin and value is None:
            del config_plugin[key]
        # 否则，设置或更新配置键值对
        else:
            config_plugin[key] = value

        # 保存配置信息
        plugin_manager.saveConfig()

        # 返回操作成功信息
        return "ok"
    # 定义一个插件操作方法，接受操作类型、地址和内部路径作为参数
    def pluginAction(self, action, address, inner_path):
        # 获取指定地址的站点对象
        site = self.server.sites.get(address)
        # 获取插件管理器对象
        plugin_manager = PluginManager.plugin_manager

        # 检查安装/更新路径是否存在
        if action in ("add", "update", "add_request"):
            # 如果站点对象不存在，则抛出异常
            if not site:
                raise Exception("Site not found")
            # 如果内部路径在站点存储中不存在，则抛出异常
            if not site.storage.isDir(inner_path):
                raise Exception("Directory not found on the site")
            try:
                # 加载内部路径下的 plugin_info.json 文件，获取插件信息
                plugin_info = site.storage.loadJson(inner_path + "/plugin_info.json")
                plugin_data = (plugin_info["rev"], plugin_info["description"], plugin_info["name"])
            except Exception as err:
                # 如果加载失败，则抛出异常
                raise Exception("Invalid plugin_info.json: %s" % Debug.formatExceptionMessage(err))
            # 获取源路径
            source_path = site.storage.getPath(inner_path)

        # 设置目标路径为已安装插件的路径
        target_path = plugin_manager.path_installed_plugins + "/" + address + "/" + inner_path
        # 获取插件配置信息
        plugin_config = plugin_manager.config.setdefault(site.address, {}).setdefault(inner_path, {})

        # 确保插件已(未)安装
        if action in ("add", "add_request") and os.path.isdir(target_path):
            # 如果是添加操作且目标路径已存在，则抛出异常
            raise Exception("Plugin already installed")

        if action in ("update", "remove") and not os.path.isdir(target_path):
            # 如果是更新或移除操作且目标路径不存在，则抛出异常
            raise Exception("Plugin not installed")

        # 执行操作
        if action == "add":
            # 复制源路径下的文件到目标路径
            shutil.copytree(source_path, target_path)
            # 更新插件配置信息
            plugin_config["date_added"] = int(time.time())
            plugin_config["rev"] = plugin_info["rev"]
            plugin_config["enabled"] = True

        if action == "update":
            # 删除目标路径下的文件
            shutil.rmtree(target_path)
            # 重新复制源路径下的文件到目标路径
            shutil.copytree(source_path, target_path)
            # 更新插件配置信息
            plugin_config["rev"] = plugin_info["rev"]
            plugin_config["date_updated"] = time.time()

        if action == "remove":
            # 从插件管理器配置中删除指定地址和内部路径的插件配置信息
            del plugin_manager.config[address][inner_path]
            # 删除目标路径下的文件
            shutil.rmtree(target_path)
    # 执行插件添加操作，传入目标地址、内部路径和结果
    def doPluginAdd(self, to, inner_path, res):
        # 如果结果为空，返回空
        if not res:
            return None

        # 执行插件添加动作，传入"add"、站点地址和内部路径
        self.pluginAction("add", self.site.address, inner_path)
        # 保存插件管理器的配置
        PluginManager.plugin_manager.saveConfig()

        # 执行命令，传入"confirm"、消息列表和回调函数
        self.cmd(
            "confirm",
            ["Plugin installed!<br>You have to restart the client to load the plugin", "Restart"],
            # 定义回调函数，传入结果并执行服务器关闭动作，重启客户端
            lambda res: self.actionServerShutdown(to, restart=True)
        )

        # 响应消息，传入目标地址和消息内容
        self.response(to, "ok")

    # 标记为不支持多用户的插件添加请求动作，传入目标地址和内部路径
    @flag.no_multiuser
    def actionPluginAddRequest(self, to, inner_path):
        # 执行插件添加请求动作，传入"add_request"、站点地址和内部路径
        self.pluginAction("add_request", self.site.address, inner_path)
        # 加载内部路径下的插件信息
        plugin_info = self.site.storage.loadJson(inner_path + "/plugin_info.json")
        # 创建警告消息
        warning = "<b>Warning!<br/>Plugins has the same permissions as the ZeroNet client.<br/>"
        warning += "Do not install it if you don't trust the developer.</b>"

        # 执行命令，传入"confirm"、消息列表和回调函数
        self.cmd(
            "confirm",
            ["Install new plugin: %s?<br>%s" % (plugin_info["name"], warning), "Trust & Install"],
            # 定义回调函数，传入结果并执行插件添加操作
            lambda res: self.doPluginAdd(to, inner_path, res)
        )

    # 标记为管理员权限和不支持多用户的插件移除动作，传入目标地址、地址和内部路径
    @flag.admin
    @flag.no_multiuser
    def actionPluginRemove(self, to, address, inner_path):
        # 执行插件移除动作，传入"remove"、地址和内部路径
        self.pluginAction("remove", address, inner_path)
        # 保存插件管理器的配置
        PluginManager.plugin_manager.saveConfig()
        # 返回"ok"
        return "ok"

    # 标记为管理员权限和不支持多用户的插件更新动作，传入目标地址、地址和内部路径
    @flag.admin
    @flag.no_multiuser
    def actionPluginUpdate(self, to, address, inner_path):
        # 执行插件更新动作，传入"update"、地址和内部路径
        self.pluginAction("update", address, inner_path)
        # 保存插件管理器的配置
        PluginManager.plugin_manager.saveConfig()
        # 将插件更新标记为已更新
        PluginManager.plugin_manager.plugins_updated["%s/%s" % (address, inner_path)] = True
        # 返回"ok"
        return "ok"
```