# `ZeroNet\plugins\MergerSite\MergerSitePlugin.py`

```py
# 导入 re 模块，用于正则表达式操作
import re
# 导入 time 模块，用于时间操作
import time
# 导入 copy 模块，用于复制操作
import copy
# 导入 os 模块，用于与操作系统交互
import os
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Translate 模块中导入 Translate 类
from Translate import Translate
# 从 util 模块中导入 RateLimit 类
from util import RateLimit
# 从 util 模块中导入 helper 函数
from util import helper
# 从 util.Flag 模块中导入 flag 对象
from util.Flag import flag
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 尝试导入 OptionalManager.UiWebsocketPlugin 模块
try:
    import OptionalManager.UiWebsocketPlugin  # To make optioanlFileInfo merger sites compatible
except Exception:
    pass

# 如果当前作用域中不存在名为 "merger_db" 的变量
if "merger_db" not in locals().keys():  # To keep merger_sites between module reloads
    # 创建名为 merger_db 的空字典，用于存储允许列出其他站点的站点信息
    merger_db = {}  # Sites that allowed to list other sites {address: [type1, type2...]}
    # 创建名为 merged_db 的空字典，用于存储允许合并到其他站点的站点信息
    merged_db = {}  # Sites that allowed to be merged to other sites {address: type, ...}
    # 创建名为 merged_to_merger 的空字典，用于缓存站点合并信息
    merged_to_merger = {}  # {address: [site1, site2, ...]} cache
    # 创建名为 site_manager 的空值，用于存储站点管理器的信息
    site_manager = None  # Site manager for merger sites

# 获取当前文件所在目录的路径，赋值给 plugin_dir 变量
plugin_dir = os.path.dirname(__file__)

# 如果当前作用域中不存在名为 "_" 的变量
if "_" not in locals():
    # 创建名为 "_" 的实例，用于多语言翻译
    _ = Translate(plugin_dir + "/languages/")


# 检查站点是否有权限访问合并站点
def checkMergerPath(address, inner_path):
    # 使用正则表达式匹配 inner_path 是否符合合并站点的格式
    merged_match = re.match("^merged-(.*?)/([A-Za-z0-9]{26,35})/", inner_path)
    if merged_match:
        # 获取合并站点的类型
        merger_type = merged_match.group(1)
        # 检查合并站点是否允许包含其他站点
        if merger_type in merger_db.get(address, []):
            # 获取被包含站点的地址
            merged_address = merged_match.group(2)
            # 检查被包含站点是否允许被包含
            if merged_db.get(merged_address) == merger_type:
                # 如果允许，则去除合并站点信息，返回被包含站点地址和剩余路径
                inner_path = re.sub("^merged-(.*?)/([A-Za-z0-9]{26,35})/", "", inner_path)
                return merged_address, inner_path
            else:
                # 如果不允许，则抛出异常
                raise Exception(
                    "Merger site (%s) does not have permission for merged site: %s (%s)" %
                    (merger_type, merged_address, merged_db.get(merged_address))
                )
        else:
            # 如果不允许，则抛出异常
            raise Exception("No merger (%s) permission to load: <br>%s (%s not in %s)" % (
                address, inner_path, merger_type, merger_db.get(address, []))
            )
    # 如果不满足前面的条件，抛出异常，异常信息包含无效的合并路径
    else:
        raise Exception("Invalid merger path: %s" % inner_path)
# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 添加新站点
    def actionMergerSiteAdd(self, to, addresses):
        # 如果 addresses 不是列表，则将其转换为列表
        if type(addresses) != list:
            # 单个站点添加
            addresses = [addresses]
        # 检查站点是否具有合并权限
        merger_types = merger_db.get(self.site.address)
        if not merger_types:
            return self.response(to, {"error": "Not a merger site"})

        # 如果站点地址没有在过去10秒内被调用，并且只有一个站点地址，则无需确认直接添加
        if RateLimit.isAllowed(self.site.address + "-MergerSiteAdd", 10) and len(addresses) == 1:
            self.cbMergerSiteAdd(to, addresses)
        else:
            # 否则需要确认添加新站点
            self.cmd(
                "confirm",
                [_["Add <b>%s</b> new site?"] % len(addresses), "Add"],
                lambda res: self.cbMergerSiteAdd(to, addresses)
            )
        # 返回响应
        self.response(to, "ok")

    # 添加新站点确认的回调
    def cbMergerSiteAdd(self, to, addresses):
        added = 0
        for address in addresses:
            try:
                # 尝试添加站点
                site_manager.need(address)
                added += 1
            except Exception as err:
                # 如果添加失败，则发送错误通知
                self.cmd("notification", ["error", _["Adding <b>%s</b> failed: %s"] % (address, err)])
        # 如果有站点被成功添加，则发送成功通知
        if added:
            self.cmd("notification", ["done", _["Added <b>%s</b> new site"] % added, 5000])
        # 更新调用限制
        RateLimit.called(self.site.address + "-MergerSiteAdd")
        # 更新合并站点
        site_manager.updateMergerSites()

    # 删除合并站点
    @flag.no_multiuser
    # 删除合并站点
    def actionMergerSiteDelete(self, to, address):
        # 获取指定地址的站点对象
        site = self.server.sites.get(address)
        # 如果站点对象不存在，则返回错误响应
        if not site:
            return self.response(to, {"error": "No site found: %s" % address})

        # 获取合并类型
        merger_types = merger_db.get(self.site.address)
        # 如果合并类型不存在，则返回错误响应
        if not merger_types:
            return self.response(to, {"error": "Not a merger site"})
        # 如果地址对应的合并类型不在合并类型列表中，则返回错误响应
        if merged_db.get(address) not in merger_types:
            return self.response(to, {"error": "Merged type (%s) not in %s" % (merged_db.get(address), merger_types)})

        # 发送通知命令
        self.cmd("notification", ["done", _["Site deleted: <b>%s</b>"] % address, 5000])
        # 返回成功响应
        self.response(to, "ok")

    # 列出合并站点
    def actionMergerSiteList(self, to, query_site_info=False):
        # 获取合并类型
        merger_types = merger_db.get(self.site.address)
        ret = {}
        # 如果合并类型不存在，则返回错误响应
        if not merger_types:
            return self.response(to, {"error": "Not a merger site"})
        # 遍历已合并站点的地址和合并类型
        for address, merged_type in merged_db.items():
            # 如果合并类型不在合并类型列表中，则跳过
            if merged_type not in merger_types:
                continue  # Site not for us
            # 如果需要查询站点信息，则获取站点信息并添加到返回结果中，否则只添加合并类型
            if query_site_info:
                site = self.server.sites.get(address)
                ret[address] = self.formatSiteInfo(site, create_user=False)
            else:
                ret[address] = merged_type
        # 返回结果
        self.response(to, ret)

    # 检查站点权限
    def hasSitePermission(self, address, *args, **kwargs):
        # 调用父类方法检查站点权限
        if super(UiWebsocketPlugin, self).hasSitePermission(address, *args, **kwargs):
            return True
        else:
            # 如果当前站点地址在已合并到的站点列表中，则返回 True，否则返回 False
            if self.site.address in [merger_site.address for merger_site in merged_to_merger.get(address, [])]:
                return True
            else:
                return False

    # 为文件命令添加合并站点支持
    # 定义一个包装器函数，用于调用指定函数名的方法，并处理合并路径的情况
    def mergerFuncWrapper(self, func_name, to, inner_path, *args, **kwargs):
        # 如果内部路径以"merged-"开头，则表示是合并路径
        if inner_path.startswith("merged-"):
            # 检查合并路径，并获取合并地址和内部路径
            merged_address, merged_inner_path = checkMergerPath(self.site.address, inner_path)

            # 为合并站点设置相同的证书
            merger_cert = self.user.getSiteData(self.site.address).get("cert")
            if merger_cert and self.user.getSiteData(merged_address).get("cert") != merger_cert:
                self.user.setCert(merged_address, merger_cert)

            # 复制当前对象，修改站点为合并站点
            req_self = copy.copy(self)
            req_self.site = self.server.sites.get(merged_address)  # 修改站点为合并站点

            # 获取指定函数名对应的方法
            func = getattr(super(UiWebsocketPlugin, req_self), func_name)
            # 调用指定方法，并返回结果
            return func(to, merged_inner_path, *args, **kwargs)
        else:
            # 获取指定函数名对应的方法
            func = getattr(super(UiWebsocketPlugin, self), func_name)
            # 调用指定方法，并返回结果
            return func(to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，执行 actionFileList 操作
    def actionFileList(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileList", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，执行 actionDirList 操作
    def actionDirList(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionDirList", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，执行 actionFileGet 操作
    def actionFileGet(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileGet", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，执行 actionFileWrite 操作
    def actionFileWrite(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileWrite", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，执行 actionFileDelete 操作
    def actionFileDelete(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileDelete", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，执行 actionFileRules 操作
    def actionFileRules(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileRules", to, inner_path, *args, **kwargs)
    # 调用 mergerFuncWrapper 方法，传入参数 "actionFileNeed"，to，inner_path，*args 和 **kwargs，并返回结果
    def actionFileNeed(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionFileNeed", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，传入参数 "actionOptionalFileInfo"，to，inner_path，*args 和 **kwargs，并返回结果
    def actionOptionalFileInfo(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionOptionalFileInfo", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，传入参数 "actionOptionalFileDelete"，to，inner_path，*args 和 **kwargs，并返回结果
    def actionOptionalFileDelete(self, to, inner_path, *args, **kwargs):
        return self.mergerFuncWrapper("actionOptionalFileDelete", to, inner_path, *args, **kwargs)

    # 调用 mergerFuncWrapper 方法，传入参数 "actionBigfileUploadInit"，to，inner_path，*args 和 **kwargs，并返回结果
    def actionBigfileUploadInit(self, to, inner_path, *args, **kwargs):
        # 调用 mergerFuncWrapper 方法，传入参数 "actionBigfileUploadInit"，to，inner_path，*args 和 **kwargs，并将结果赋给 back
        back = self.mergerFuncWrapper("actionBigfileUploadInit", to, inner_path, *args, **kwargs)
        # 如果 inner_path 以 "merged-" 开头
        if inner_path.startswith("merged-"):
            # 调用 checkMergerPath 方法，传入参数 self.site.address 和 inner_path，并将结果赋给 merged_address 和 merged_inner_path
            merged_address, merged_inner_path = checkMergerPath(self.site.address, inner_path)
            # 修改 back 字典中的 "inner_path" 键对应的值
            back["inner_path"] = "merged-%s/%s/%s" % (merged_db[merged_address], merged_address, back["inner_path"])
        # 返回 back
        return back

    # 为带有 privatekey 参数的文件命令添加合并站点支持
    # 定义一个包装器函数，用于在私钥下执行指定函数
    def mergerFuncWrapperWithPrivatekey(self, func_name, to, privatekey, inner_path, *args, **kwargs):
        # 获取父类中指定名称的函数
        func = getattr(super(UiWebsocketPlugin, self), func_name)
        # 如果内部路径以"merged-"开头
        if inner_path.startswith("merged-"):
            # 检查合并路径，获取合并地址和合并内部路径
            merged_address, merged_inner_path = checkMergerPath(self.site.address, inner_path)
            # 获取合并站点
            merged_site = self.server.sites.get(merged_address)

            # 为合并站点设置相同的证书
            merger_cert = self.user.getSiteData(self.site.address).get("cert")
            if merger_cert:
                self.user.setCert(merged_address, merger_cert)

            site_before = self.site  # 保存当前站点以便在运行命令后能够切换回来
            self.site = merged_site  # 将站点切换为合并站点
            try:
                back = func(to, privatekey, merged_inner_path, *args, **kwargs)  # 执行指定函数
            finally:
                self.site = site_before  # 切换回原始站点
            return back  # 返回执行结果
        else:
            return func(to, privatekey, inner_path, *args, **kwargs)  # 在当前站点下执行指定函数

    # 执行站点签名操作
    def actionSiteSign(self, to, privatekey=None, inner_path="content.json", *args, **kwargs):
        return self.mergerFuncWrapperWithPrivatekey("actionSiteSign", to, privatekey, inner_path, *args, **kwargs)

    # 执行站点发布操作
    def actionSitePublish(self, to, privatekey=None, inner_path="content.json", *args, **kwargs):
        return self.mergerFuncWrapperWithPrivatekey("actionSitePublish", to, privatekey, inner_path, *args, **kwargs)

    # 添加权限操作
    def actionPermissionAdd(self, to, permission):
        super(UiWebsocketPlugin, self).actionPermissionAdd(to, permission)  # 调用父类的添加权限操作
        # 如果权限以"Merger"开头
        if permission.startswith("Merger"):
            self.site.storage.rebuildDb()  # 重建数据库
    # 定义一个方法，用于处理权限详情
    def actionPermissionDetails(self, to, permission):
        # 如果权限不是以"Merger"开头，则调用父类的actionPermissionDetails方法
        if not permission.startswith("Merger"):
            return super(UiWebsocketPlugin, self).actionPermissionDetails(to, permission)

        # 获取合并类型
        merger_type = permission.replace("Merger:", "")
        # 如果合并类型不符合指定格式，则抛出异常
        if not re.match("^[A-Za-z0-9-]+$", merger_type):
            raise Exception("Invalid merger_type: %s" % merger_type)
        
        # 初始化合并站点列表
        merged_sites = []
        # 遍历merged_db中的地址和合并类型
        for address, merged_type in merged_db.items():
            # 如果合并类型不等于指定的合并类型，则跳过当前循环
            if merged_type != merger_type:
                continue
            # 获取站点对象
            site = self.server.sites.get(address)
            try:
                # 尝试获取站点内容管理器中的content.json文件的标题，如果失败则将地址添加到合并站点列表中
                merged_sites.append(site.content_manager.contents.get("content.json").get("title", address))
            except Exception:
                merged_sites.append(address)

        # 构建权限详情信息
        details = _["Read and write permissions to sites with merged type of <b>%s</b> "] % merger_type
        details += _["(%s sites)"] % len(merged_sites)
        details += "<div style='white-space: normal; max-width: 400px'>%s</div>" % ", ".join(merged_sites)
        # 发送权限详情信息
        self.response(to, details)
# 将 UiRequestPlugin 类注册到 PluginManager 的 "UiRequest" 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 解析路径，允许使用 /merged-ZeroMe/address/file.jpg 加载合并的站点文件
    def parsePath(self, path):
        # 调用父类的 parsePath 方法，获取路径的各个部分
        path_parts = super(UiRequestPlugin, self).parsePath(path)
        # 如果路径中不包含 "merged-"，则直接返回路径的各个部分
        if "merged-" not in path:  # 优化
            return path_parts
        # 如果路径中包含 "merged-"，则调用 checkMergerPath 方法处理地址和内部路径
        path_parts["address"], path_parts["inner_path"] = checkMergerPath(path_parts["address"], path_parts["inner_path"])
        return path_parts


# 将 SiteStoragePlugin 类注册到 PluginManager 的 "SiteStorage" 插件中
@PluginManager.registerTo("SiteStorage")
class SiteStoragePlugin(object):
    # 当合并站点文件发生变化时，重新构建
    # 注意合并站点文件的变化
    def onUpdated(self, inner_path, file=None):
        # 调用父类的 onUpdated 方法，处理内部路径和文件
        super(SiteStoragePlugin, self).onUpdated(inner_path, file)

        # 获取合并类型
        merged_type = merged_db.get(self.site.address)

        # 遍历与当前站点地址相关的合并站点
        for merger_site in merged_to_merger.get(self.site.address, []):
            # 避免无限循环，如果合并站点地址与当前站点地址相同，则跳过
            if merger_site.address == self.site.address:
                continue
            # 构建虚拟路径
            virtual_path = "merged-%s/%s/%s" % (merged_type, self.site.address, inner_path)
            # 如果内部路径以 ".json" 结尾
            if inner_path.endswith(".json"):
                # 如果文件不为空，则调用合并站点的 storage 的 onUpdated 方法
                if file is not None:
                    merger_site.storage.onUpdated(virtual_path, file=file)
                # 如果文件为空，则调用合并站点的 storage 的 onUpdated 方法，并打开内部路径的文件
                else:
                    merger_site.storage.onUpdated(virtual_path, file=self.open(inner_path))
            else:
                # 调用合并站点的 storage 的 onUpdated 方法
                merger_site.storage.onUpdated(virtual_path)


# 将 SitePlugin 类注册到 PluginManager 的 "Site" 插件中
@PluginManager.registerTo("Site")
class SitePlugin(object):
    # 当文件处理完成时
    def fileDone(self, inner_path):
        # 调用父类的 fileDone 方法，处理内部路径
        super(SitePlugin, self).fileDone(inner_path)

        # 遍历与当前地址相关的合并站点
        for merger_site in merged_to_merger.get(self.address, []):
            # 如果合并站点地址与当前地址相同，则跳过
            if merger_site.address == self.address:
                continue
            # 遍历合并站点的 websockets，触发事件
            for ws in merger_site.websockets:
                ws.event("siteChanged", self, {"event": ["file_done", inner_path]})
    # 定义一个方法，用于处理文件加载失败的情况，接收内部路径作为参数
    def fileFailed(self, inner_path):
        # 调用父类的fileFailed方法，传入内部路径参数
        super(SitePlugin, self).fileFailed(inner_path)

        # 遍历与当前地址相关的合并站点列表
        for merger_site in merged_to_merger.get(self.address, []):
            # 如果合并站点的地址与当前地址相同，则跳过本次循环
            if merger_site.address == self.address:
                continue
            # 遍历合并站点的websockets列表
            for ws in merger_site.websockets:
                # 向websockets发送站点变更事件，包括当前站点、事件类型和内部路径
                ws.event("siteChanged", self, {"event": ["file_failed", inner_path]})
# 将 SiteManagerPlugin 类注册到 PluginManager 的 SiteManager 中
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # 更新合并站点的站点类型
    def load(self, *args, **kwags):
        # 调用父类的 load 方法
        super(SiteManagerPlugin, self).load(*args, **kwags)
        # 更新合并站点
        self.updateMergerSites()

    # 延迟保存方法
    def saveDelayed(self, *args, **kwags):
        # 调用父类的 saveDelayed 方法
        super(SiteManagerPlugin, self).saveDelayed(*args, **kwags)
        # 更新合并站点
        self.updateMergerSites()
```