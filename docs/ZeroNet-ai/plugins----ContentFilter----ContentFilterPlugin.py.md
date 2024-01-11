# `ZeroNet\plugins\ContentFilter\ContentFilterPlugin.py`

```
# 导入时间模块
import time
# 导入正则表达式模块
import re
# 导入 HTML 模块
import html
# 导入操作系统模块
import os

# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Translate 模块中导入 Translate 类
from Translate import Translate
# 从 Config 模块中导入 config 变量
from Config import config
# 从 util.Flag 模块中导入 flag 变量
from util.Flag import flag

# 从当前目录下的 ContentFilterStorage 模块中导入 ContentFilterStorage 类
from .ContentFilterStorage import ContentFilterStorage

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 如果当前作用域中不存在下划线变量
if "_" not in locals():
    # 使用 Translate 类创建 _ 变量，指定语言文件路径
    _ = Translate(plugin_dir + "/languages/")

# 将 SiteManagerPlugin 类注册到 PluginManager 的 SiteManager 插件中
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # 加载方法
    def load(self, *args, **kwargs):
        # 声明全局变量 filter_storage
        global filter_storage
        # 调用父类的 load 方法
        super(SiteManagerPlugin, self).load(*args, **kwargs)
        # 创建 ContentFilterStorage 实例并赋值给 filter_storage
        filter_storage = ContentFilterStorage(site_manager=self)

    # 添加方法
    def add(self, address, *args, **kwargs):
        # 获取是否忽略阻止标志
        should_ignore_block = kwargs.get("ignore_block") or kwargs.get("settings")
        # 如果需要忽略阻止
        if should_ignore_block:
            # 设置 block_details 为 None
            block_details = None
        # 如果不需要忽略阻止
        elif filter_storage.isSiteblocked(address):
            # 获取阻止详情
            block_details = filter_storage.getSiteblockDetails(address)
        else:
            # 获取地址的哈希值
            address_hashed = filter_storage.getSiteAddressHashed(address)
            # 如果地址的哈希值被阻止
            if filter_storage.isSiteblocked(address_hashed):
                # 获取阻止详情
                block_details = filter_storage.getSiteblockDetails(address_hashed)
            else:
                # 设置 block_details 为 None
                block_details = None

        # 如果存在阻止详情
        if block_details:
            # 抛出异常，指明网站被阻止
            raise Exception("Site blocked: %s" % html.escape(block_details.get("reason", "unknown reason")))
        else:
            # 调用父类的 add 方法
            return super(SiteManagerPlugin, self).add(address, *args, **kwargs)

# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # Mute 添加方法
    def cbMuteAdd(self, to, auth_address, cert_user_id, reason):
        # 将静音信息添加到文件内容中
        filter_storage.file_content["mutes"][auth_address] = {
            "cert_user_id": cert_user_id, "reason": reason, "source": self.site.address, "date_added": time.time()
        }
        # 保存文件内容
        filter_storage.save()
        # 更改数据库
        filter_storage.changeDbs(auth_address, "remove")
        # 响应消息
        self.response(to, "ok")

    # 标记为不支持多用户
    @flag.no_multiuser
    # 添加禁言操作，将用户加入禁言列表
    def actionMuteAdd(self, to, auth_address, cert_user_id, reason):
        # 如果用户具有管理员权限
        if "ADMIN" in self.getPermissions(to):
            # 调用禁言添加回调函数
            self.cbMuteAdd(to, auth_address, cert_user_id, reason)
        else:
            # 否则，发送确认消息，等待用户确认后再调用禁言添加回调函数
            self.cmd(
                "confirm",
                [_["Hide all content from <b>%s</b>?"] % html.escape(cert_user_id), _["Mute"]],
                lambda res: self.cbMuteAdd(to, auth_address, cert_user_id, reason)
            )

    # 禁言移除回调函数
    @flag.no_multiuser
    def cbMuteRemove(self, to, auth_address):
        # 从禁言列表中删除指定用户
        del filter_storage.file_content["mutes"][auth_address]
        # 保存修改后的禁言列表
        filter_storage.save()
        # 更改数据库
        filter_storage.changeDbs(auth_address, "load")
        # 发送响应消息
        self.response(to, "ok")

    # 移除禁言操作
    @flag.no_multiuser
    def actionMuteRemove(self, to, auth_address):
        # 如果用户具有管理员权限
        if "ADMIN" in self.getPermissions(to):
            # 调用禁言移除回调函数
            self.cbMuteRemove(to, auth_address)
        else:
            # 否则，发送确认消息，等待用户确认后再调用禁言移除回调函数
            cert_user_id = html.escape(filter_storage.file_content["mutes"][auth_address]["cert_user_id"])
            self.cmd(
                "confirm",
                [_["Unmute <b>%s</b>?"] % cert_user_id, _["Unmute"]],
                lambda res: self.cbMuteRemove(to, auth_address)
            )

    # 禁言列表操作
    @flag.admin
    def actionMuteList(self, to):
        # 返回禁言列表
        self.response(to, filter_storage.file_content["mutes"])

    # 添加站点忽略操作
    @flag.no_multiuser
    @flag.admin
    def actionSiteblockIgnoreAddSite(self, to, site_address):
        # 如果站点已经在管理列表中
        if site_address in filter_storage.site_manager.sites:
            return {"error": "Site already added"}
        else:
            # 如果站点需要被忽略
            if filter_storage.site_manager.need(site_address, ignore_block=True):
                return "ok"
            else:
                return {"error": "Invalid address"}

    # 添加站点屏蔽操作
    @flag.no_multiuser
    @flag.admin
    def actionSiteblockAdd(self, to, site_address, reason=None):
        # 将站点添加到屏蔽列表中
        filter_storage.file_content["siteblocks"][site_address] = {"date_added": time.time(), "reason": reason}
        # 保存修改后的屏蔽列表
        filter_storage.save()
        # 发送响应消息
        self.response(to, "ok")
    # 标记该方法不支持多用户
    @flag.no_multiuser
    # 标记该方法需要管理员权限
    @flag.admin
    # 移除指定站点的过滤规则
    def actionSiteblockRemove(self, to, site_address):
        # 从过滤器存储中删除指定站点的内容
        del filter_storage.file_content["siteblocks"][site_address]
        # 保存修改后的过滤器存储
        filter_storage.save()
        # 响应操作结果
        self.response(to, "ok")

    # 标记该方法需要管理员权限
    @flag.admin
    # 列出所有站点的过滤规则
    def actionSiteblockList(self, to):
        # 响应包含所有站点过滤规则的内容
        self.response(to, filter_storage.file_content["siteblocks"])

    # 标记该方法需要管理员权限
    @flag.admin
    # 获取指定站点的过滤规则
    def actionSiteblockGet(self, to, site_address):
        # 如果指定站点被阻止，则获取站点的详细信息
        if filter_storage.isSiteblocked(site_address):
            res = filter_storage.getSiteblockDetails(site_address)
        else:
            # 如果指定站点未被阻止，则获取站点地址的哈希值
            site_address_hashed = filter_storage.getSiteAddressHashed(site_address)
            # 如果哈希值对应的站点被阻止，则获取站点的详细信息
            if filter_storage.isSiteblocked(site_address_hashed):
                res = filter_storage.getSiteblockDetails(site_address_hashed)
            else:
                # 如果站点未找到，则返回错误信息
                res = {"error": "Site block not found"}
        # 响应获取到的站点过滤规则信息
        self.response(to, res)

    # 标记该方法不支持多用户
    @flag.no_multiuser
    # 添加过滤器包含规则
    def actionFilterIncludeAdd(self, to, inner_path, description=None, address=None):
        # 如果指定地址存在
        if address:
            # 如果当前用户没有管理员权限，则返回错误信息
            if "ADMIN" not in self.getPermissions(to):
                return self.response(to, {"error": "Forbidden: Only ADMIN sites can manage different site include"})
            # 获取指定地址对应的站点
            site = self.server.sites[address]
        else:
            # 如果地址不存在，则使用当前站点的地址和站点
            address = self.site.address
            site = self.site

        # 如果当前用户有管理员权限
        if "ADMIN" in self.getPermissions(to):
            # 调用过滤器包含规则添加的回调函数
            self.cbFilterIncludeAdd(to, True, address, inner_path, description)
        else:
            # 获取站点存储中指定路径的内容
            content = site.storage.loadJson(inner_path)
            # 生成标题信息
            title = _["New shared global content filter: <b>%s</b> (%s sites, %s users)"] % (
                html.escape(inner_path), len(content.get("siteblocks", {})), len(content.get("mutes", {}))
            )
            # 发送确认命令，根据结果调用过滤器包含规则添加的回调函数
            self.cmd(
                "confirm",
                [title, "Add"],
                lambda res: self.cbFilterIncludeAdd(to, res, address, inner_path, description)
            )
    # 定义一个方法，用于向过滤器中添加包含规则
    def cbFilterIncludeAdd(self, to, res, address, inner_path, description):
        # 如果 res 为空，则返回空响应并结束方法
        if not res:
            self.response(to, res)
            return False

        # 向过滤器存储中添加包含规则
        filter_storage.includeAdd(address, inner_path, description)
        # 返回成功响应
        self.response(to, "ok")

    # 标记为不支持多用户的方法，用于移除过滤器中的包含规则
    @flag.no_multiuser
    def actionFilterIncludeRemove(self, to, inner_path, address=None):
        # 如果指定了地址，并且当前用户没有管理员权限，则返回禁止访问的错误响应
        if address:
            if "ADMIN" not in self.getPermissions(to):
                return self.response(to, {"error": "Forbidden: Only ADMIN sites can manage different site include"})
        else:
            address = self.site.address

        # 根据地址和内部路径构建键值
        key = "%s/%s" % (address, inner_path)
        # 如果键值不在过滤器存储的包含规则中，则返回包含未找到的错误响应
        if key not in filter_storage.file_content["includes"]:
            self.response(to, {"error": "Include not found"})
        # 从过滤器存储中移除包含规则
        filter_storage.includeRemove(address, inner_path)
        # 返回成功响应
        self.response(to, "ok")

    # 定义一个方法，用于列出所有包含规则
    def actionFilterIncludeList(self, to, all_sites=False, filters=False):
        # 如果需要列出所有站点的包含规则，并且当前用户不是管理员，则返回禁止访问的错误响应
        if all_sites and "ADMIN" not in self.getPermissions(to):
            return self.response(to, {"error": "Forbidden: Only ADMIN sites can list all sites includes"})

        # 初始化返回结果列表
        back = []
        # 获取过滤器存储中的所有包含规则
        includes = filter_storage.file_content.get("includes", {}).values()
        # 遍历所有包含规则
        for include in includes:
            # 如果不需要列出所有站点的包含规则，并且当前包含规则的地址不是当前站点的地址，则跳过当前包含规则
            if not all_sites and include["address"] != self.site.address:
                continue
            # 如果需要获取过滤器规则，则获取规则内容并添加到包含规则中
            if filters:
                include = dict(include)  # 不修改原始 file_content
                include_site = filter_storage.site_manager.get(include["address"])
                if not include_site:
                    continue
                content = include_site.storage.loadJson(include["inner_path"])
                include["mutes"] = content.get("mutes", {})
                include["siteblocks"] = content.get("siteblocks", {})
            # 将包含规则添加到返回结果列表中
            back.append(include)
        # 返回包含规则列表
        self.response(to, back)
# 将 SiteStoragePlugin 类注册到 PluginManager 的 SiteStorage 插件中
@PluginManager.registerTo("SiteStorage")
class SiteStoragePlugin(object):
    # 更新数据库文件
    def updateDbFile(self, inner_path, file=None, cur=None):
        # 如果文件不是 False，则允许删除文件
        if file is not False:  # File deletion always allowed
            # 在文件路径中查找比特币地址
            matches = re.findall("/(1[A-Za-z0-9]{26,35})/", inner_path)
            # 检查地址是否在静音列表中
            for auth_address in matches:
                if filter_storage.isMuted(auth_address):
                    self.log.debug("Mute match: %s, ignoring %s" % (auth_address, inner_path))
                    return False

        return super(SiteStoragePlugin, self).updateDbFile(inner_path, file=file, cur=cur)

    # 当文件更新时
    def onUpdated(self, inner_path, file=None):
        # 构建文件路径
        file_path = "%s/%s" % (self.site.address, inner_path)
        # 如果文件路径在包含列表中
        if file_path in filter_storage.file_content["includes"]:
            self.log.debug("Filter file updated: %s" % inner_path)
            # 更新所有包含文件
            filter_storage.includeUpdateAll()
        return super(SiteStoragePlugin, self).onUpdated(inner_path, file=file)


# 将 UiRequestPlugin 类注册到 PluginManager 的 UiRequest 插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 定义一个方法，用于包装处理请求的动作
    def actionWrapper(self, path, extra_headers=None):
        # 使用正则表达式匹配路径中的地址和内部路径
        match = re.match(r"/(?P<address>[A-Za-z0-9\._-]+)(?P<inner_path>/.*|$)", path)
        # 如果没有匹配到地址，则返回 False
        if not match:
            return False
        # 获取匹配到的地址
        address = match.group("address")

        # 如果站点管理器中已经存在该地址，则调用父类的 actionWrapper 方法处理请求
        if self.server.site_manager.get(address):  # Site already exists
            return super(UiRequestPlugin, self).actionWrapper(path, extra_headers)

        # 如果地址是一个域名，则解析域名
        if self.isDomain(address):
            address = self.resolveDomain(address)

        # 如果地址存在，则获取其哈希值
        if address:
            address_hashed = filter_storage.getSiteAddressHashed(address)
        else:
            address_hashed = None

        # 如果地址被阻止访问，则返回被阻止页面
        if filter_storage.isSiteblocked(address) or filter_storage.isSiteblocked(address_hashed):
            # 获取首页站点
            site = self.server.site_manager.get(config.homepage)
            # 如果没有额外的头部信息，则创建一个空字典
            if not extra_headers:
                extra_headers = {}
            # 获取脚本的 nonce 值
            script_nonce = self.getScriptNonce()
            # 发送头部信息
            self.sendHeader(extra_headers=extra_headers, script_nonce=script_nonce)
            # 返回被阻止页面的迭代器
            return iter([super(UiRequestPlugin, self).renderWrapper(
                site, path, "uimedia/plugins/contentfilter/blocklisted.html?address=" + address,
                "Blacklisted site", extra_headers, show_loadingscreen=False, script_nonce=script_nonce
            )])
        # 如果地址未被阻止，则调用父类的 actionWrapper 方法处理请求
        else:
            return super(UiRequestPlugin, self).actionWrapper(path, extra_headers)

    # 定义一个方法，用于处理 UI 媒体请求
    def actionUiMedia(self, path, *args, **kwargs):
        # 如果路径以指定的前缀开头，则替换路径并调用 actionFile 方法处理文件请求
        if path.startswith("/uimedia/plugins/contentfilter/"):
            file_path = path.replace("/uimedia/plugins/contentfilter/", plugin_dir + "/media/")
            return self.actionFile(file_path)
        # 否则调用父类的 actionUiMedia 方法处理 UI 媒体请求
        else:
            return super(UiRequestPlugin, self).actionUiMedia(path)
```