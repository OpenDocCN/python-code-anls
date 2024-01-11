# `ZeroNet\plugins\Cors\CorsPlugin.py`

```
# 导入 re 模块，用于正则表达式操作
import re
# 导入 html 模块，用于 HTML 转义操作
import html
# 导入 copy 模块，用于复制操作
import copy
# 导入 os 模块，用于操作系统相关功能
import os
# 导入 gevent 模块，用于协程操作
import gevent

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Translate 模块中导入 Translate 类
from Translate import Translate

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 如果当前作用域中不存在变量 "_"
if "_" not in locals():
    # 创建一个 Translate 对象，并将其赋值给变量 "_"
    _ = Translate(plugin_dir + "/languages/")


# 定义函数 getCorsPath，用于解析 cors 路径
def getCorsPath(site, inner_path):
    # 使用正则表达式匹配 cors 路径
    match = re.match("^cors-([A-Za-z0-9]{26,35})/(.*)", inner_path)
    # 如果匹配失败，则抛出异常
    if not match:
        raise Exception("Invalid cors path: %s" % inner_path)
    # 获取 cors 地址和内部路径
    cors_address = match.group(1)
    cors_inner_path = match.group(2)

    # 如果当前站点没有访问 cors 地址的权限，则抛出异常
    if not "Cors:%s" % cors_address in site.settings["permissions"]:
        raise Exception("This site has no permission to access site %s" % cors_address)

    # 返回 cors 地址和内部路径
    return cors_address, cors_inner_path


# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 检查站点是否有权限执行指定命令
    def hasSitePermission(self, address, cmd=None):
        # 如果父类 UiWebsocketPlugin 有站点权限，则返回 True
        if super(UiWebsocketPlugin, self).hasSitePermission(address, cmd=cmd):
            return True

        # 允许执行的命令列表
        allowed_commands = [
            "fileGet", "fileList", "dirList", "fileRules", "optionalFileInfo",
            "fileQuery", "dbQuery", "userGetSettings", "siteInfo"
        ]
        # 如果当前站点没有 cors 地址的权限，或者指定命令不在允许列表中，则返回 False，否则返回 True
        if not "Cors:%s" % address in self.site.settings["permissions"] or cmd not in allowed_commands:
            return False
        else:
            return True

    # 为文件命令添加 cors 支持
    # 定义一个包装函数，用于处理跨域请求
    def corsFuncWrapper(self, func_name, to, inner_path, *args, **kwargs):
        # 如果路径以"cors-"开头，则获取跨域地址和内部路径
        if inner_path.startswith("cors-"):
            cors_address, cors_inner_path = getCorsPath(self.site, inner_path)

            # 复制当前对象，修改其中的 site 属性为合并后的地址
            req_self = copy.copy(self)
            req_self.site = self.server.sites.get(cors_address)  # 修改 site 为合并后的地址
            # 如果找不到对应的 site，则返回错误信息
            if not req_self.site:
                return {"error": "No site found"}

            # 获取指定函数的引用，并调用该函数
            func = getattr(super(UiWebsocketPlugin, req_self), func_name)
            back = func(to, cors_inner_path, *args, **kwargs)
            return back
        else:
            # 如果路径不以"cors-"开头，则直接调用指定函数
            func = getattr(super(UiWebsocketPlugin, self), func_name)
            return func(to, inner_path, *args, **kwargs)

    # 定义 actionFileGet 方法，调用 corsFuncWrapper 处理跨域请求
    def actionFileGet(self, to, inner_path, *args, **kwargs):
        return self.corsFuncWrapper("actionFileGet", to, inner_path, *args, **kwargs)

    # 定义 actionFileList 方法，调用 corsFuncWrapper 处理跨域请求
    def actionFileList(self, to, inner_path, *args, **kwargs):
        return self.corsFuncWrapper("actionFileList", to, inner_path, *args, **kwargs)

    # 定义 actionDirList 方法，调用 corsFuncWrapper 处理跨域请求
    def actionDirList(self, to, inner_path, *args, **kwargs):
        return self.corsFuncWrapper("actionDirList", to, inner_path, *args, **kwargs)

    # 定义 actionFileRules 方法，调用 corsFuncWrapper 处理跨域请求
    def actionFileRules(self, to, inner_path, *args, **kwargs):
        return self.corsFuncWrapper("actionFileRules", to, inner_path, *args, **kwargs)

    # 定义 actionOptionalFileInfo 方法，调用 corsFuncWrapper 处理跨域请求
    def actionOptionalFileInfo(self, to, inner_path, *args, **kwargs):
        return self.corsFuncWrapper("actionOptionalFileInfo", to, inner_path, *args, **kwargs)
    # 处理跨域请求的权限
    def actionCorsPermission(self, to, address):
        # 如果地址是列表，则直接赋值给addresses，否则将地址放入列表中
        if isinstance(address, list):
            addresses = address
        else:
            addresses = [address]

        # 设置按钮标题为"Grant"
        button_title = _["Grant"]
        site_names = []
        site_addresses = []
        # 遍历地址列表
        for address in addresses:
            # 获取服务器中对应地址的站点
            site = self.server.sites.get(address)
            if site:
                # 获取站点的名称，如果没有则使用地址作为名称
                site_name = site.content_manager.contents.get("content.json", {}).get("title", address)
            else:
                site_name = address
                # 如果至少有一个站点尚未下载，则将按钮标题设置为"Grant & Add"
                button_title = _["Grant & Add"]

            if not (site and "Cors:" + address in self.permissions):
                # 如果站点不存在或者没有权限，则将站点名称和地址添加到对应列表中
                site_names.append(site_name)
                site_addresses.append(address)

        if len(site_names) == 0:
            return "ignored"

        # 弹出确认框，询问用户是否授予权限
        self.cmd(
            "confirm",
            [_["This site requests <b>read</b> permission to: <b>%s</b>"] % ", ".join(map(html.escape, site_names)), button_title],
            lambda res: self.cbCorsPermission(to, site_addresses)
        )

    # 处理跨域请求权限的回调函数
    def cbCorsPermission(self, to, addresses):
        # 添加权限
        for address in addresses:
            permission = "Cors:" + address
            if permission not in self.site.settings["permissions"]:
                self.site.settings["permissions"].append(permission)

        # 保存站点设置
        self.site.saveSettings()
        # 更新站点的websocket，通知权限已添加
        self.site.updateWebsocket(permission_added=permission)

        # 响应请求，返回"ok"
        self.response(to, "ok")

        # 遍历地址列表
        for address in addresses:
            # 获取服务器中对应地址的站点
            site = self.server.sites.get(address)
            if not site:
                # 如果站点不存在，则异步下载站点内容
                gevent.spawn(self.server.site_manager.need, address)
# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 解析路径，允许使用 /cors-address/file.jpg 加载跨域文件
    def parsePath(self, path):
        # 调用父类的 parsePath 方法，获取路径的各个部分
        path_parts = super(UiRequestPlugin, self).parsePath(path)
        # 如果路径中不包含 "cors-"，则直接返回路径的各个部分
        if "cors-" not in path:  # Optimization
            return path_parts
        # 获取路径对应的站点对象
        site = self.server.sites[path_parts["address"]]
        # 尝试获取跨域路径，并更新地址和内部路径
        try:
            path_parts["address"], path_parts["inner_path"] = getCorsPath(site, path_parts["inner_path"])
        # 如果出现异常，则返回空值
        except Exception:
            return None
        # 返回更新后的路径的各个部分
        return path_parts
```