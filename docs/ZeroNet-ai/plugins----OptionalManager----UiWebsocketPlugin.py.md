# `ZeroNet\plugins\OptionalManager\UiWebsocketPlugin.py`

```py
# 导入正则表达式、时间、HTML处理和操作系统模块
import re
import time
import html
import os

# 导入协程模块
import gevent

# 从插件管理器中导入插件管理器和配置
from Plugin import PluginManager
from Config import config
from util import helper
from util.Flag import flag
from Translate import Translate

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 如果当前作用域中不存在下划线变量，则创建一个翻译对象
if "_" not in locals():
    _ = Translate(plugin_dir + "/languages/")

# 创建一个空的大文件 SHA512 缓存字典
bigfile_sha512_cache = {}

# 将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    def __init__(self, *args, **kwargs):
        # 初始化函数，设置时间戳为0
        self.time_peer_numbers_updated = 0
        super(UiWebsocketPlugin, self).__init__(*args, **kwargs)

    def actionSiteSign(self, to, privatekey=None, inner_path="content.json", *args, **kwargs):
        # 将文件添加到 content.db 并将其设置为 pinned
        content_db = self.site.content_manager.contents.db
        content_inner_dir = helper.getDirname(inner_path)
        content_db.my_optional_files[self.site.address + "/" + content_inner_dir] = time.time()
        # 如果 my_optional_files 中的文件数量超过50个，则保留最新的50个
        if len(content_db.my_optional_files) > 50:
            oldest_key = min(
                iter(content_db.my_optional_files.keys()),
                key=(lambda key: content_db.my_optional_files[key])
            )
            del content_db.my_optional_files[oldest_key]

        return super(UiWebsocketPlugin, self).actionSiteSign(to, privatekey, inner_path, *args, **kwargs)

    def updatePeerNumbers(self):
        # 更新站点的哈希字段
        self.site.updateHashfield()
        content_db = self.site.content_manager.contents.db
        # 更新内容数据库中的对等数量
        content_db.updatePeerNumbers()
        # 更新站点的 WebSocket 连接，标记对等数量已更新
        self.site.updateWebsocket(peernumber_updated=True)

    # 可选文件函数
    # 定义一个方法，用于获取可选文件的信息
    def actionOptionalFileInfo(self, to, inner_path):
        # 获取内容数据库
        content_db = self.site.content_manager.contents.db
        # 获取站点 ID
        site_id = content_db.site_ids[self.site.address]

        # 如果距离上次更新节点数的时间超过1分钟，并且距离上次更新节点数的时间超过5分钟，则更新节点数
        if time.time() - content_db.time_peer_numbers_updated > 60 * 1 and time.time() - self.time_peer_numbers_updated > 60 * 5:
            # 在新线程中启动以避免阻塞
            self.time_peer_numbers_updated = time.time()
            gevent.spawn(self.updatePeerNumbers)

        # 查询文件可选信息
        query = "SELECT * FROM file_optional WHERE site_id = :site_id AND inner_path = :inner_path LIMIT 1"
        res = content_db.execute(query, {"site_id": site_id, "inner_path": inner_path})
        row = next(res, None)
        if row:
            row = dict(row)
            # 如果文件大小大于1MB，则添加大文件信息
            if row["size"] > 1024 * 1024:
                row["address"] = self.site.address
                self.addBigfileInfo(row)
            # 响应结果
            self.response(to, row)
        else:
            self.response(to, None)

    # 设置文件的固定状态
    def setPin(self, inner_path, is_pinned, address=None):
        if not address:
            address = self.site.address

        # 如果没有站点权限，则返回错误信息
        if not self.hasSitePermission(address):
            return {"error": "Forbidden"}

        # 获取站点并设置文件固定状态
        site = self.server.sites[address]
        site.content_manager.setPin(inner_path, is_pinned)

        return "ok"

    # 处理文件的固定状态
    @flag.no_multiuser
    def actionOptionalFilePin(self, to, inner_path, address=None):
        # 如果内部路径不是列表，则转换为列表
        if type(inner_path) is not list:
            inner_path = [inner_path]
        # 设置文件固定状态为固定
        back = self.setPin(inner_path, 1, address)
        num_file = len(inner_path)
        # 如果设置成功，则发送通知
        if back == "ok":
            if num_file == 1:
                self.cmd("notification", ["done", _["Pinned %s"] % html.escape(helper.getFilename(inner_path[0])), 5000])
            else:
                self.cmd("notification", ["done", _["Pinned %s files"] % num_file, 5000])
        # 响应结果
        self.response(to, back)

    # 处理文件的固定状态
    @flag.no_multiuser
    # 定义一个方法，用于取消文件的固定操作
    def actionOptionalFileUnpin(self, to, inner_path, address=None):
        # 如果inner_path不是列表，则将其转换为列表
        if type(inner_path) is not list:
            inner_path = [inner_path]
        # 调用setPin方法，将inner_path对应的文件取消固定
        back = self.setPin(inner_path, 0, address)
        # 获取inner_path的文件数量
        num_file = len(inner_path)
        # 如果取消固定操作成功
        if back == "ok":
            # 如果只取消了一个文件的固定
            if num_file == 1:
                # 发送通知，提示取消了某个文件的固定
                self.cmd("notification", ["done", _["Removed pin from %s"] % html.escape(helper.getFilename(inner_path[0])), 5000])
            else:
                # 发送通知，提示取消了多个文件的固定
                self.cmd("notification", ["done", _["Removed pin from %s files"] % num_file, 5000])
        # 将取消固定的结果返回
        self.response(to, back)
    
    # 标记该方法不支持多用户操作
    @flag.no_multiuser
    # 定义一个方法，用于删除可选文件
    def actionOptionalFileDelete(self, to, inner_path, address=None):
        # 如果地址为空，则使用站点地址
        if not address:
            address = self.site.address

        # 如果没有站点权限，则返回禁止访问的错误信息
        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        # 获取站点对象
        site = self.server.sites[address]

        # 获取内容数据库
        content_db = site.content_manager.contents.db
        # 获取站点 ID
        site_id = content_db.site_ids[site.address]

        # 从文件可选表中查询指定站点和内部路径的文件信息
        res = content_db.execute("SELECT * FROM file_optional WHERE ? LIMIT 1", {"site_id": site_id, "inner_path": inner_path, "is_downloaded": 1})
        # 获取查询结果的第一行
        row = next(res, None)

        # 如果查询结果为空，则返回未在内容数据库中找到的错误信息
        if not row:
            return self.response(to, {"error": "Not found in content.db"})

        # 调用站点内容管理器的optionalRemoved方法，标记文件为已删除
        removed = site.content_manager.optionalRemoved(inner_path, row["hash_id"], row["size"])

        # 更新文件可选表中的is_downloaded、is_pinned和peer字段
        content_db.execute("UPDATE file_optional SET is_downloaded = 0, is_pinned = 0, peer = peer - 1 WHERE ?", {"site_id": site_id, "inner_path": inner_path})

        # 尝试删除文件
        try:
            site.storage.delete(inner_path)
        # 捕获异常并返回文件删除错误信息
        except Exception as err:
            return self.response(to, {"error": "File delete error: %s" % err})
        # 更新站点的websocket，通知文件已删除
        site.updateWebsocket(file_delete=inner_path)

        # 如果内部路径在站点内容管理器的cache_is_pinned中，则清空cache_is_pinned
        if inner_path in site.content_manager.cache_is_pinned:
            site.content_manager.cache_is_pinned = {}

        # 返回成功信息
        self.response(to, "ok")

    # 限制函数

    # 标记为管理员权限，定义一个方法，用于获取可选文件的限制统计信息
    @flag.admin
    def actionOptionalLimitStats(self, to):
        back = {}
        # 获取可选文件的限制
        back["limit"] = config.optional_limit
        # 获取已使用的可选文件字节数
        back["used"] = self.site.content_manager.contents.db.getOptionalUsedBytes()
        # 获取剩余可用空间
        back["free"] = helper.getFreeSpace()

        # 返回统计信息
        self.response(to, back)

    # 标记为非多用户和管理员权限
    @flag.no_multiuser
    @flag.admin
    # 设置可选限制值，并保存到配置中
    def actionOptionalLimitSet(self, to, limit):
        config.optional_limit = re.sub(r"\.0+$", "", limit)  # 从末尾删除不必要的数字
        config.saveValue("optional_limit", limit)
        self.response(to, "ok")

    # 分发帮助函数

    # 获取可选帮助列表
    def actionOptionalHelpList(self, to, address=None):
        if not address:
            address = self.site.address

        # 如果没有站点权限，则返回错误信息
        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        site = self.server.sites[address]

        self.response(to, site.settings.get("optional_help", {}))

    # 设置可选帮助
    @flag.no_multiuser
    def actionOptionalHelp(self, to, directory, title, address=None):
        if not address:
            address = self.site.address

        # 如果没有站点权限，则返回错误信息
        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        site = self.server.sites[address]
        content_db = site.content_manager.contents.db
        site_id = content_db.site_ids[address]

        # 如果站点设置中没有可选帮助，则创建一个空字典
        if "optional_help" not in site.settings:
            site.settings["optional_help"] = {}

        # 查询文件可选帮助的数量和大小
        stats = content_db.execute(
            "SELECT COUNT(*) AS num, SUM(size) AS size FROM file_optional WHERE site_id = :site_id AND inner_path LIKE :inner_path",
            {"site_id": site_id, "inner_path": directory + "%"}
        ).fetchone()
        stats = dict(stats)

        # 如果大小为0，则设置为0
        if not stats["size"]:
            stats["size"] = 0
        # 如果数量为0，则设置为0
        if not stats["num"]:
            stats["num"] = 0

        # 发送通知消息
        self.cmd("notification", [
            "done",
            _["You started to help distribute <b>%s</b>.<br><small>Directory: %s</small>"] %
            (html.escape(title), html.escape(directory)),
            10000
        ])

        # 将可选帮助添加到站点设置中
        site.settings["optional_help"][directory] = title

        # 返回统计信息
        self.response(to, dict(stats))

    # 设置可选帮助
    @flag.no_multiuser
    # 定义一个方法，用于移除指定目录下的可选帮助信息
    def actionOptionalHelpRemove(self, to, directory, address=None):
        # 如果地址为空，则使用默认地址
        if not address:
            address = self.site.address

        # 如果没有站点权限，则返回错误信息
        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        # 获取指定地址的站点对象
        site = self.server.sites[address]

        # 尝试删除站点设置中的可选帮助信息
        try:
            del site.settings["optional_help"][directory]
            # 返回成功信息
            self.response(to, "ok")
        except Exception:
            # 返回未找到错误信息
            self.response(to, {"error": "Not found"})

    # 定义一个方法，用于设置站点的自动下载可选帮助信息的值
    def cbOptionalHelpAll(self, to, site, value):
        site.settings["autodownloadoptional"] = value
        # 返回设置的值
        self.response(to, value)

    # 定义一个方法，用于设置所有站点的自动下载可选帮助信息的值
    @flag.no_multiuser
    def actionOptionalHelpAll(self, to, value, address=None):
        # 如果地址为空，则使用默认地址
        if not address:
            address = self.site.address

        # 如果没有站点权限，则返回错误信息
        if not self.hasSitePermission(address):
            return self.response(to, {"error": "Forbidden"})

        # 获取指定地址的站点对象
        site = self.server.sites[address]

        # 如果值为真
        if value:
            # 如果当前站点具有管理员权限，则设置所有站点的自动下载可选帮助信息的值为真
            if "ADMIN" in self.site.settings["permissions"]:
                self.cbOptionalHelpAll(to, site, True)
            else:
                # 否则，弹出确认框，确认是否帮助分发所有新的可选文件
                site_title = site.content_manager.contents["content.json"].get("title", address)
                self.cmd(
                    "confirm",
                    [
                        _["Help distribute all new optional files on site <b>%s</b>"] % html.escape(site_title),
                        _["Yes, I want to help!"]
                    ],
                    lambda res: self.cbOptionalHelpAll(to, site, True)
                )
        else:
            # 否则，设置所有站点的自动下载可选帮助信息的值为假
            site.settings["autodownloadoptional"] = False
            # 返回假
            self.response(to, False)
```