# `ZeroNet\src\Ui\UiWebsocket.py`

```
# 导入所需的模块
import json
import time
import sys
import os
import shutil
import re
import copy
import logging
import stat

# 导入 gevent 模块
import gevent

# 从 Config 模块中导入 config 对象
from Config import config
# 从 Site 模块中导入 SiteManager 类
from Site import SiteManager
# 从 Crypt 模块中导入 CryptBitcoin 类
from Crypt import CryptBitcoin
# 从 Debug 模块中导入 Debug 类
from Debug import Debug
# 从 util 模块中导入 QueryJson, RateLimit, helper, SafeRe 模块
from util import QueryJson, RateLimit, helper, SafeRe
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Translate 模块中导入 translate 函数
from Translate import translate as _
# 从 util.Flag 模块中导入 flag 对象
from util.Flag import flag
# 从 Content.ContentManager 模块中导入 VerifyError, SignError 类
from Content.ContentManager import VerifyError, SignError

# 使用 PluginManager.acceptPlugins 装饰器注册插件
@PluginManager.acceptPlugins
# 定义 UiWebsocket 类
class UiWebsocket(object):
    # 初始化方法
    def __init__(self, ws, site, server, user, request):
        # 初始化实例变量
        self.ws = ws
        self.site = site
        self.user = user
        self.log = site.log
        self.request = request
        self.permissions = []
        self.server = server
        self.next_message_id = 1
        self.waiting_cb = {}  # Waiting for callback. Key: message_id, Value: function pointer
        self.channels = []  # Channels joined to
        self.state = {"sending": False}  # Shared state of websocket connection
        self.send_queue = []  # Messages to send to client

    # 关闭方法
    def onClosed(self):
        pass

    # 缩进方法，用于去除文本中的空格和换行符
    def dedent(self, text):
        return re.sub("[\\r\\n\\x20\\t]+", " ", text.strip().replace("<br>", " "))

    # 添加主页通知方法
    def addHomepageNotifications(self):
        # 如果没有 Multiuser 插件和 UiPassword 插件
        if not(self.hasPlugin("Multiuser")) and not(self.hasPlugin("UiPassword")):
            # 获取配置中的 ui_ip 和 ui_restrict
            bind_ip = getattr(config, "ui_ip", "")
            whitelist = getattr(config, "ui_restrict", [])
            # 如果绑定到互联网，没有 IP 白名单，没有 UiPassword，没有 Multiuser
            if ("0.0.0.0" == bind_ip or "*" == bind_ip) and (not whitelist):
                # 向站点通知列表中添加警告信息
                self.site.notifications.append([
                    "error",
                    _("You are not going to set up a public gateway. However, <b>your Web UI is<br>" +
                        "open to the whole Internet.</b> " +
                        "Please check your configuration.")
                ])
    # 检查给定的插件名是否在插件管理器的插件名列表中
    def hasPlugin(self, name):
        return name in PluginManager.plugin_manager.plugin_names

    # 检查是否有权限运行指定的命令
    def hasCmdPermission(self, cmd):
        # 获取指定命令对应的标志列表
        flags = flag.db.get(self.getCmdFuncName(cmd), ())
        # 如果标志列表中包含"admin"，但是当前用户权限中不包含"ADMIN"，则返回 False，否则返回 True
        if "admin" in flags and "ADMIN" not in self.permissions:
            return False
        else:
            return True

    # 检查是否有权限访问指定站点
    def hasSitePermission(self, address, cmd=None):
        # 如果地址不等于当前站点地址，并且当前站点设置中不包含"ADMIN"权限，则返回 False，否则返回 True
        if address != self.site.address and "ADMIN" not in self.site.settings["permissions"]:
            return False
        else:
            return True

    # 检查是否有权限访问指定文件
    def hasFilePermission(self, inner_path):
        # 获取有效签名者列表
        valid_signers = self.site.content_manager.getValidSigners(inner_path)
        # 如果当前站点设置中包含"own"权限，或者当前用户的授权地址在有效签名者列表中，则返回 True，否则返回 False
        return self.site.settings["own"] or self.user.getAuthAddress(self.site.address) in valid_signers

    # 处理频道中的事件
    def event(self, channel, *params):
        if channel in self.channels:  # 检查是否已加入频道
            if channel == "siteChanged":
                site = params[0]
                site_info = self.formatSiteInfo(site, create_user=False)
                if len(params) > 1 and params[1]:  # 检查是否有额外数据
                    site_info.update(params[1])
                self.cmd("setSiteInfo", site_info)
            elif channel == "serverChanged":
                server_info = self.formatServerInfo()
                if len(params) > 0 and params[0]:  # 检查是否有额外数据
                    server_info.update(params[0])
                self.cmd("setServerInfo", server_info)
            elif channel == "announcerChanged":
                site = params[0]
                announcer_info = self.formatAnnouncerInfo(site)
                if len(params) > 1 and params[1]:  # 检查是否有额外数据
                    announcer_info.update(params[1])
                self.cmd("setAnnouncerInfo", announcer_info)

    # 向客户端发送响应（to = message.id）
    # 发送响应消息给指定的客户端
    def response(self, to, result):
        self.send({"cmd": "response", "to": to, "result": result})

    # 发送一个命令
    def cmd(self, cmd, params={}, cb=None):
        self.send({"cmd": cmd, "params": params}, cb)

    # 编码为 JSON 格式并发送消息
    def send(self, message, cb=None):
        message["id"] = self.next_message_id  # 添加消息 ID 以便进行响应
        self.next_message_id += 1
        if cb:  # 客户端响应后的回调
            self.waiting_cb[message["id"]] = cb
        self.send_queue.append(message)
        if self.state["sending"]:
            return  # 已经在发送中
        try:
            while self.send_queue:
                self.state["sending"] = True
                message = self.send_queue.pop(0)
                self.ws.send(json.dumps(message))
                self.state["sending"] = False
        except Exception as err:
            self.log.debug("Websocket send error: %s" % Debug.formatException(err))
            self.state["sending"] = False

    # 获取权限
    def getPermissions(self, req_id):
        permissions = self.site.settings["permissions"]
        if req_id >= 1000000:  # 这是一个包装命令，允许管理员命令
            permissions = permissions[:]
            permissions.append("ADMIN")
        return permissions
    # 定义一个装饰器函数，用于处理异步函数的错误和结果
    def asyncWrapper(self, func):
        # 定义一个异步错误监视器函数，用于捕获异步函数的错误并处理
        def asyncErrorWatcher(func, *args, **kwargs):
            try:
                # 调用被装饰的异步函数，并获取其结果
                result = func(*args, **kwargs)
                # 如果结果不为空，则向客户端发送响应
                if result is not None:
                    self.response(args[0], result)
            except Exception as err:
                # 如果配置为调试模式，则允许将 websocket 错误显示在 /Debug 页面上
                if config.debug:
                    import main
                    main.DebugHook.handleError()
                # 记录 WebSocket handleRequest 出现的错误
                self.log.error("WebSocket handleRequest error: %s" % Debug.formatException(err))
                # 向客户端发送错误信息
                self.cmd("error", "Internal error: %s" % Debug.formatException(err, "html"))

        # 定义一个装饰器函数，用于创建一个新的协程来执行被装饰的函数
        def wrapper(*args, **kwargs):
            gevent.spawn(asyncErrorWatcher, func, *args, **kwargs)
        return wrapper

    # 根据命令获取对应的函数名
    def getCmdFuncName(self, cmd):
        # 根据命令拼接出对应的函数名
        func_name = "action" + cmd[0].upper() + cmd[1:]
        return func_name

    # 处理传入的消息
    # 处理请求的方法，接收一个请求对象作为参数
    def handleRequest(self, req):
        # 从请求对象中获取命令和参数
        cmd = req.get("cmd")
        params = req.get("params")
        # 获取请求对象的权限
        self.permissions = self.getPermissions(req["id"])

        # 如果命令是"response"，表示这是对一个命令的响应
        if cmd == "response":  # It's a response to a command
            # 调用actionResponse方法处理响应，并返回结果
            return self.actionResponse(req["to"], req["result"])
        else:  # Normal command
            # 获取命令对应的方法名
            func_name = self.getCmdFuncName(cmd)
            # 获取对应的方法
            func = getattr(self, func_name, None)
            # 如果站点设置中包含"deleting"，返回站点正在删除的错误信息
            if self.site.settings.get("deleting"):
                return self.response(req["id"], {"error": "Site is deleting"})

            # 如果方法不存在，返回未知命令的错误信息
            if not func:  # Unknown command
                return self.response(req["id"], {"error": "Unknown command: %s" % cmd})

            # 如果没有权限运行该命令，返回权限错误信息
            if not self.hasCmdPermission(cmd):  # Admin commands
                return self.response(req["id"], {"error": "You don't have permission to run %s" % cmd})

        # 并行执行
        func_flags = flag.db.get(self.getCmdFuncName(cmd), ())
        if func_flags and "async_run" in func_flags:
            func = self.asyncWrapper(func)

        # 支持命名参数、无名参数和原始第一个参数的调用
        if type(params) is dict:
            result = func(req["id"], **params)
        elif type(params) is list:
            result = func(req["id"], *params)
        elif params:
            result = func(req["id"], params)
        else:
            result = func(req["id"])

        # 如果结果不为空，返回响应
        if result is not None:
            self.response(req["id"], result)

    # 格式化站点信息
    # 格式化站点信息，包括内容和设置
    def formatSiteInfo(self, site, create_user=True):
        # 获取站点内容管理器中的 content.json 文件内容
        content = site.content_manager.contents.get("content.json", {})
        # 如果 content 不为空，则进行数据处理
        if content:  # Remove unnecessary data transfer
            # 复制 content 对象，避免修改原对象
            content = content.copy()
            # 统计文件数量并更新 content 对象
            content["files"] = len(content.get("files", {}))
            content["files_optional"] = len(content.get("files_optional", {}))
            content["includes"] = len(content.get("includes", {}))
            # 如果 content 中包含 "sign"，则删除该键值对
            if "sign" in content:
                del(content["sign"])
            # 如果 content 中包含 "signs"，则删除该键值对
            if "signs" in content:
                del(content["signs"])
            # 如果 content 中包含 "signers_sign"，则删除该键值对
            if "signers_sign" in content:
                del(content["signers_sign"])

        # 复制站点设置对象
        settings = site.settings.copy()
        # 删除 settings 中的 "wrapper_key" 键值对，避免暴露密钥
        del settings["wrapper_key"]  # Dont expose wrapper key

        # 构建返回的站点信息字典
        ret = {
            "auth_address": self.user.getAuthAddress(site.address, create=create_user),
            "cert_user_id": self.user.getCertUserId(site.address),
            "address": site.address,
            "address_short": site.address_short,
            "address_hash": site.address_hash.hex(),
            "settings": settings,
            "content_updated": site.content_updated,
            "bad_files": len(site.bad_files),
            "size_limit": site.getSizeLimit(),
            "next_size_limit": site.getNextSizeLimit(),
            "peers": max(site.settings.get("peers", 0), len(site.peers)),
            "started_task_num": site.worker_manager.started_task_num,
            "tasks": len(site.worker_manager.tasks),
            "workers": len(site.worker_manager.workers),
            "content": content
        }
        # 如果站点设置中包含 "own"，则添加 "privatekey" 到返回字典中
        if site.settings["own"]:
            ret["privatekey"] = bool(self.user.getSiteData(site.address, create=create_user).get("privatekey"))
        # 如果站点正在提供服务并且 content 不为空，则增加 "peers" 的值
        if site.isServing() and content:
            ret["peers"] += 1  # Add myself if serving
        # 返回构建的站点信息字典
        return ret
    # 格式化服务器信息，返回包含服务器信息的字典
    def formatServerInfo(self):
        # 导入 main 模块
        import main
        # 获取 main 模块中的 file_server 对象
        file_server = main.file_server
        # 如果 file_server 的 port_opened 属性为空字典
        if file_server.port_opened == {}:
            # 将 ip_external 设置为 None
            ip_external = None
        else:
            # 否则，ip_external 设置为 file_server.port_opened 中的任意值
            ip_external = any(file_server.port_opened.values())
        # 返回包含服务器信息的字典
        back = {
            "ip_external": ip_external,
            "port_opened": file_server.port_opened,
            "platform": sys.platform,
            "fileserver_ip": config.fileserver_ip,
            "fileserver_port": config.fileserver_port,
            "tor_enabled": file_server.tor_manager.enabled,
            "tor_status": file_server.tor_manager.status,
            "tor_has_meek_bridges": file_server.tor_manager.has_meek_bridges,
            "tor_use_bridges": config.tor_use_bridges,
            "ui_ip": config.ui_ip,
            "ui_port": config.ui_port,
            "version": config.version,
            "rev": config.rev,
            "timecorrection": file_server.timecorrection,
            "language": config.language,
            "debug": config.debug,
            "offline": config.offline,
            "plugins": PluginManager.plugin_manager.plugin_names,
            "plugins_rev": PluginManager.plugin_manager.plugins_rev,
            "user_settings": self.user.settings
        }
        # 如果当前用户具有 "ADMIN" 权限
        if "ADMIN" in self.site.settings["permissions"]:
            # 添加额外的信息到返回的字典中
            back["updatesite"] = config.updatesite
            back["dist_type"] = config.dist_type
            back["lib_verify_best"] = CryptBitcoin.lib_verify_best
        # 返回包含服务器信息的字典
        return back

    # 格式化通告者信息，返回包含通告者信息的字典
    def formatAnnouncerInfo(self, site):
        # 返回包含通告者地址和统计信息的字典
        return {"address": site.address, "stats": site.announcer.stats}

    # - Actions -
    # 将指定命令和参数发送到指定地址的站点，如果没有权限则返回相应消息
    def actionAs(self, to, address, cmd, params=[]):
        if not self.hasSitePermission(address, cmd=cmd):
            return self.response(to, "No permission for site %s" % address)
        # 复制当前对象，设置站点属性为指定地址的站点，使用相同的权限
        req_self = copy.copy(self)
        req_self.site = self.server.sites.get(address)
        req_self.hasCmdPermission = self.hasCmdPermission  # 使用与当前站点相同的权限
        req_obj = super(UiWebsocket, req_self)
        req = {"id": to, "cmd": cmd, "params": params}
        req_obj.handleRequest(req)

    # 在收到响应时进行回调 {"cmd": "response", "to": message_id, "result": result}
    def actionResponse(self, to, result):
        if to in self.waiting_cb:
            self.waiting_cb[to](result)  # 调用回调函数
        else:
            self.log.error("Websocket callback not found: %s, %s" % (to, result))

    # 发送简单的 pong 回复
    def actionPing(self, to):
        self.response(to, "pong")

    # 发送站点详情
    def actionSiteInfo(self, to, file_status=None):
        ret = self.formatSiteInfo(self.site)
        if file_status:  # 客户端查询文件状态
            if self.site.storage.isFile(file_status):  # 文件存在，添加事件完成
                ret["event"] = ("file_done", file_status)
        self.response(to, ret)

    # 获取站点的坏文件列表
    def actionSiteBadFiles(self, to):
        return list(self.site.bad_files.keys())

    # 加入事件频道
    def actionChannelJoin(self, to, channels):
        if type(channels) != list:
            channels = [channels]

        for channel in channels:
            if channel not in self.channels:
                self.channels.append(channel)

        self.response(to, "ok")

    # 服务器变量
    def actionServerInfo(self, to):
        back = self.formatServerInfo()
        self.response(to, back)

    # 创建一个新的包装 nonce，允许加载 html 文件
    @flag.admin
    # 从服务器获取包装器的随机数，然后将其发送给指定的目标
    def actionServerGetWrapperNonce(self, to):
        wrapper_nonce = self.request.getWrapperNonce()
        self.response(to, wrapper_nonce)

    # 获取广播者信息，并将其格式化后发送给指定的目标
    def actionAnnouncerInfo(self, to):
        back = self.formatAnnouncerInfo(self.site)
        self.response(to, back)

    # 管理员权限装饰器，用于获取广播者的统计信息
    @flag.admin
    def actionAnnouncerStats(self, to):
        back = {}
        trackers = self.site.announcer.getTrackers()
        for site in list(self.server.sites.values()):
            for tracker, stats in site.announcer.stats.items():
                if tracker not in trackers:
                    continue
                if tracker not in back:
                    back[tracker] = {}
                is_latest_data = bool(stats["time_request"] > back[tracker].get("time_request", 0) and stats["status"])
                for key, val in stats.items():
                    if key.startswith("num_"):
                        back[tracker][key] = back[tracker].get(key, 0) + val
                    elif is_latest_data:
                        back[tracker][key] = val

        # 返回广播者的统计信息
        return back

    # 签署 content.json
    # 签署并发布 content.json
    # 发布站点内容到指定地址
    def actionSitePublish(self, to, privatekey=None, inner_path="content.json", sign=True, remove_missing_optional=False, update_changed_files=False):
        # 如果需要签名，则调用 actionSiteSign 方法对内容进行签名
        if sign:
            inner_path = self.actionSiteSign(
                to, privatekey, inner_path, response_ok=False,
                remove_missing_optional=remove_missing_optional, update_changed_files=update_changed_files
            )
            # 如果签名失败，则返回
            if not inner_path:
                return
        # 发布站点内容
        if not self.site.settings["serving"]:  # 如果站点处于暂停状态，则启用站点
            self.site.settings["serving"] = True
            self.site.saveSettings()
            self.site.announce()

        # 如果指定的文件路径不在站点内容管理器的内容中，则返回错误信息
        if inner_path not in self.site.content_manager.contents:
            return self.response(to, {"error": "File %s not found" % inner_path})

        # 生成事件名称
        event_name = "publish %s %s" % (self.site.address, inner_path)
        # 检查是否允许立即调用
        called_instantly = RateLimit.isAllowed(event_name, 30)
        # 调用异步方法进行站点内容发布
        thread = RateLimit.callAsync(event_name, 30, self.doSitePublish, self.site, inner_path)  # 每30秒只能发布一次
        # 只在第一次回调时显示通知
        notification = "linked" not in dir(thread)
        thread.linked = True
        # 如果允许立即调用，则在结束时回调并显示进度
        if called_instantly:
            self.cmd("progress", ["publish", _["Content published to {0}/{1} peers."].format(0, 5), 0])
            thread.link(lambda thread: self.cbSitePublish(to, self.site, thread, notification, callback=notification))
        # 如果不允许立即调用，则显示通知并返回 "ok"
        else:
            self.cmd(
                "notification",
                ["info", _["Content publish queued for {0:.0f} seconds."].format(RateLimit.delayLeft(event_name, 30)), 5000]
            )
            self.response(to, "ok")
            # 在结束时显示通知
            thread.link(lambda thread: self.cbSitePublish(to, self.site, thread, notification, callback=False))
    # 定义一个方法用于发布站点内容
    def doSitePublish(self, site, inner_path):
        # 定义一个回调函数，用于显示发布进度
        def cbProgress(published, limit):
            # 计算发布进度百分比
            progress = int(float(published) / limit * 100)
            # 发送进度信息给命令行
            self.cmd("progress", [
                "publish",
                _["Content published to {0}/{1} peers."].format(published, limit),
                progress
            ])
        # 获取指定路径下的内容差异
        diffs = site.content_manager.getDiffs(inner_path)
        # 发布站点内容，限制为5个，传入路径、差异和进度回调函数
        back = site.publish(limit=5, inner_path=inner_path, diffs=diffs, cb_progress=cbProgress)
        # 如果发布失败
        if back == 0:
            # 发送发布失败信息给命令行
            self.cmd("progress", ["publish", _["Content publish failed."], -100])
        else:
            # 发送全部发布信息给命令行
            cbProgress(back, back)
        # 返回发布结果
        return back

    # 站点发布的回调函数
    # 定义一个方法，用于发布站点内容
    def cbSitePublish(self, to, site, thread, notification=True, callback=True):
        # 获取发布的内容数量
        published = thread.value
        # 如果发布数量大于0，表示成功发布
        if published > 0:  # Successfully published
            # 如果需要通知
            if notification:
                # 发送通知，告知内容已发布到指定的对等节点数量
                # self.cmd("notification", ["done", _["Content published to {0} peers."].format(published), 5000])
                # 更新本地 WebSocket 客户端的站点数据
                site.updateWebsocket()  # Send updated site data to local websocket clients
            # 如果需要回调
            if callback:
                # 响应消息，表示发布成功
                self.response(to, "ok")
        else:
            # 如果站点对等节点数量为0
            if len(site.peers) == 0:
                # 导入主模块
                import main
                # 如果文件服务器的端口已经打开，或者 Tor 管理器已经启动
                if any(main.file_server.port_opened.values()) or main.file_server.tor_manager.start_onions:
                    # 如果需要通知
                    if notification:
                        # 发送通知，告知没有找到对等节点，但内容已准备好可以访问
                        self.cmd("notification", ["info", _["No peers found, but your content is ready to access."]])
                    # 如果需要回调
                    if callback:
                        # 响应消息，表示发布成功
                        self.response(to, "ok")
                else:
                    # 如果需要通知
                    if notification:
                        # 发送通知，告知网络连接受限，需要打开指定端口才能让站点对所有人可访问
                        self.cmd("notification", [
                            "info",
                            _("""{_[Your network connection is restricted. Please, open <b>{0}</b> port]}<br>
                            {_[on your router to make your site accessible for everyone.]}""").format(config.fileserver_port)
                        ])
                    # 如果需要回调
                    if callback:
                        # 响应消息，表示端口未打开
                        self.response(to, {"error": "Port not opened."})
            else:
                # 如果需要通知
                if notification:
                    # 响应消息，表示内容发布失败
                    self.response(to, {"error": "Content publish failed."})

    # 重新加载站点内容
    def actionSiteReload(self, to, inner_path):
        # 加载指定路径的站点内容，不添加坏文件
        self.site.content_manager.loadContent(inner_path, add_bad_files=False)
        # 验证文件，快速检查
        self.site.storage.verifyFiles(quick_check=True)
        # 更新本地 WebSocket 客户端的站点数据
        self.site.updateWebsocket()
        # 返回消息，表示操作成功
        return "ok"

    # 将文件写入磁盘
    # 定义一个方法，用于删除指定路径下的文件
    def actionFileDelete(self, to, inner_path):
        # 检查当前用户是否有权限删除指定路径下的文件
        if not self.hasFilePermission(inner_path):
            # 如果没有权限，则记录错误信息并返回相应的响应
            self.log.error("File delete error: you don't own this site & you are not approved by the owner.")
            return self.response(to, {"error": "Forbidden, you can only modify your own files"})

        # 默认需要删除文件
        need_delete = True
        # 获取文件信息
        file_info = self.site.content_manager.getFileInfo(inner_path)
        # 如果文件信息存在并且文件是可选的
        if file_info and file_info.get("optional"):
            # 非存在的可选文件不会从 content.json 中删除，因此我们需要手动处理
            self.log.debug("Deleting optional file: %s" % inner_path)
            # 获取相对路径
            relative_path = file_info["relative_path"]
            # 加载 content.json 文件
            content_json = self.site.storage.loadJson(file_info["content_inner_path"])
            # 如果相对路径在 files_optional 中
            if relative_path in content_json.get("files_optional", {}):
                # 从 files_optional 中删除相对路径
                del content_json["files_optional"][relative_path]
                # 将更新后的 content.json 写回存储
                self.site.storage.writeJson(file_info["content_inner_path"], content_json)
                # 重新加载 content.json 文件
                self.site.content_manager.loadContent(file_info["content_inner_path"], add_bad_files=False, force=True)
                # 检查文件是否仍然存在（从 content.json 中删除后仍然存在的文件）
                need_delete = self.site.storage.isFile(inner_path)  # File sill exists after removing from content.json (owned site)

        # 如果需要删除文件
        if need_delete:
            try:
                # 尝试删除文件
                self.site.storage.delete(inner_path)
            except Exception as err:
                # 如果删除文件时出现异常，记录错误信息并返回相应的响应
                self.log.error("File delete error: %s" % err)
                return self.response(to, {"error": "Delete error: %s" % Debug.formatExceptionMessage(err)})

        # 返回成功的响应
        self.response(to, "ok")

        # 向其他本地用户发送 siteChanged 事件
        for ws in self.site.websockets:
            if ws != self:
                ws.event("siteChanged", self.site, {"event": ["file_deleted", inner_path]})

    # 在 json 文件中查找数据
    # 查询指定目录下的文件，返回查询结果
    def actionFileQuery(self, to, dir_inner_path, query=None):
        # 获取目录的完整路径
        dir_path = self.site.storage.getPath(dir_inner_path)
        # 使用 QueryJson.query 方法查询目录下符合条件的文件，返回结果列表
        rows = list(QueryJson.query(dir_path, query or ""))
        # 返回查询结果
        return self.response(to, rows)

    # 列出目录中的文件
    @flag.async_run
    def actionFileList(self, to, inner_path):
        try:
            # 使用 self.site.storage.walk 方法列出目录中的文件，返回文件列表
            return list(self.site.storage.walk(inner_path))
        except Exception as err:
            # 如果出现异常，记录错误日志并返回错误信息
            self.log.error("fileList %s error: %s" % (inner_path, Debug.formatException(err)))
            return {"error": Debug.formatExceptionMessage(err)}

    # 列出目录中的子目录
    @flag.async_run
    def actionDirList(self, to, inner_path, stats=False):
        try:
            if stats:
                # 如果需要获取文件统计信息
                back = []
                # 遍历目录中的文件，获取文件统计信息并添加到列表中
                for file_name in self.site.storage.list(inner_path):
                    file_stats = os.stat(self.site.storage.getPath(inner_path + "/" + file_name))
                    is_dir = stat.S_ISDIR(file_stats.st_mode)
                    back.append(
                        {"name": file_name, "size": file_stats.st_size, "is_dir": is_dir}
                    )
                return back
            else:
                # 如果不需要获取文件统计信息，直接列出目录中的子目录
                return list(self.site.storage.list(inner_path))
        except Exception as err:
            # 如果出现异常，记录错误日志并返回错误信息
            self.log.error("dirList %s error: %s" % (inner_path, Debug.formatException(err)))
            return {"error": Debug.formatExceptionMessage(err)}

    # Sql query
    # 执行数据库查询操作
    def actionDbQuery(self, to, query, params=None, wait_for=None):
        # 如果处于调试模式或者详细模式，则记录当前时间
        if config.debug or config.verbose:
            s = time.time()
        # 初始化结果集
        rows = []
        try:
            # 执行数据库查询操作
            res = self.site.storage.query(query, params)
        except Exception as err:  # 处理异常情况，返回错误信息给客户端
            self.log.error("DbQuery error: %s" % Debug.formatException(err))
            return self.response(to, {"error": Debug.formatExceptionMessage(err)})
        # 将查询结果转换为字典形式
        for row in res:
            rows.append(dict(row))
        # 如果处于详细模式并且查询时间超过0.1秒，则记录慢查询日志
        if config.verbose and time.time() - s > 0.1:  
            self.log.debug("Slow query: %s (%.3fs)" % (query, time.time() - s))
        # 返回查询结果给客户端
        return self.response(to, rows)

    # 返回文件内容
    @flag.async_run
    def actionFileGet(self, to, inner_path, required=True, format="text", timeout=300, priority=6):
        try:
            # 如果文件是必需的或者在坏文件列表中，则需要获取文件内容
            if required or inner_path in self.site.bad_files:
                with gevent.Timeout(timeout):
                    self.site.needFile(inner_path, priority=priority)
            # 读取文件内容
            body = self.site.storage.read(inner_path, "rb")
        except (Exception, gevent.Timeout) as err:
            # 处理异常情况，记录错误信息
            self.log.debug("%s fileGet error: %s" % (inner_path, Debug.formatException(err)))
            body = None

        # 根据指定格式对文件内容进行处理
        if not body:
            body = None
        elif format == "base64":
            import base64
            body = base64.b64encode(body).decode()
        else:
            try:
                body = body.decode()
            except Exception as err:
                # 处理异常情况，返回错误信息给客户端
                self.response(to, {"error": "Error decoding text: %s" % err})
        # 返回处理后的文件内容给客户端
        self.response(to, body)

    @flag.async_run
    # 定义一个方法，用于请求需要文件的操作
    def actionFileNeed(self, to, inner_path, timeout=300, priority=6):
        try:
            # 设置超时时间，如果超时则抛出异常
            with gevent.Timeout(timeout):
                # 调用 site 对象的 needFile 方法请求文件
                self.site.needFile(inner_path, priority=priority)
        except (Exception, gevent.Timeout) as err:
            # 如果出现异常，则返回错误信息
            return self.response(to, {"error": Debug.formatExceptionMessage(err)})
        # 如果没有出现异常，则返回成功信息
        return self.response(to, "ok")

    # 定义一个方法，用于获取文件规则
    def actionFileRules(self, to, inner_path, use_my_cert=False, content=None):
        # 如果没有定义内容，则从 site.content_manager.contents 中获取内容
        if not content:  # No content defined by function call
            content = self.site.content_manager.contents.get(inner_path)

        # 如果内容仍然为空，则文件尚未创建
        if not content:  # File not created yet
            # 获取用户在该站点的证书
            cert = self.user.getCert(self.site.address)
            # 如果证书存在且有效，则将其添加到查询规则中
            if cert and cert["auth_address"] in self.site.content_manager.getValidSigners(inner_path):
                content = {}
                content["cert_auth_type"] = cert["auth_type"]
                content["cert_user_id"] = self.user.getCertUserId(self.site.address)
                content["cert_sign"] = cert["cert_sign"]

        # 获取文件规则
        rules = self.site.content_manager.getRules(inner_path, content)
        # 如果文件路径以 "content.json" 结尾且存在规则，则计算当前大小
        if inner_path.endswith("content.json") and rules:
            if content:
                rules["current_size"] = len(json.dumps(content)) + sum([file["size"] for file in list(content.get("files", {}).values())])
            else:
                rules["current_size"] = 0
        # 返回规则信息
        return self.response(to, rules)

    # 添加证书到用户
    # 添加证书操作，接收参数包括目标地址、域名、认证类型、认证用户名和证书
    def actionCertAdd(self, to, domain, auth_type, auth_user_name, cert):
        try:
            # 调用用户对象的addCert方法，向指定地址添加证书
            res = self.user.addCert(self.user.getAuthAddress(self.site.address), domain, auth_type, auth_user_name, cert)
            # 如果添加成功
            if res is True:
                # 发送通知消息，证书添加成功
                self.cmd(
                    "notification",
                    ["done", _("{_[New certificate added]:} <b>{auth_type}/{auth_user_name}@{domain}</b>.")]
                )
                # 设置用户证书
                self.user.setCert(self.site.address, domain)
                # 更新网页套接字，证书已更改
                self.site.updateWebsocket(cert_changed=domain)
                # 响应目标地址，返回"ok"
                self.response(to, "ok")
            # 如果添加失败
            elif res is False:
                # 显示更改确认
                cert_current = self.user.certs[domain]
                body = _("{_[Your current certificate]:} <b>{cert_current[auth_type]}/{cert_current[auth_user_name]}@{domain}</b>")
                self.cmd(
                    "confirm",
                    [body, _("Change it to {auth_type}/{auth_user_name}@{domain}")],
                    # 确认更改后的回调函数
                    lambda res: self.cbCertAddConfirm(to, domain, auth_type, auth_user_name, cert)
                )
            # 如果出现其他情况
            else:
                # 响应目标地址，返回"Not changed"
                self.response(to, "Not changed")
        # 捕获异常
        except Exception as err:
            # 记录错误日志
            self.log.error("CertAdd error: Exception - %s (%s)" % (err.message, Debug.formatException(err)))
            # 响应目标地址，返回错误消息
            self.response(to, {"error": err.message})

    # 更改证书确认的回调函数
    def cbCertAddConfirm(self, to, domain, auth_type, auth_user_name, cert):
        # 删除指定域名的证书
        self.user.deleteCert(domain)
        # 添加新的证书
        self.user.addCert(self.user.getAuthAddress(self.site.address), domain, auth_type, auth_user_name, cert)
        # 发送通知消息，证书已更改
        self.cmd(
            "notification",
            ["done", _("Certificate changed to: <b>{auth_type}/{auth_user_name}@{domain}</b>.")]
        )
        # 设置用户证书
        self.user.setCert(self.site.address, domain)
        # 更新网页套接字，证书已更改
        self.site.updateWebsocket(cert_changed=domain)
        # 响应目标地址，返回"ok"
        self.response(to, "ok")

    # 选择站点的证书
    # - 管理员操作 -
    @flag.admin
    # 添加权限到指定用户
    def actionPermissionAdd(self, to, permission):
        # 如果权限不在站点设置的权限列表中，则添加权限并保存设置
        if permission not in self.site.settings["permissions"]:
            self.site.settings["permissions"].append(permission)
            self.site.saveSettings()
            # 更新 WebSocket，通知权限已添加
            self.site.updateWebsocket(permission_added=permission)
        # 响应操作结果
        self.response(to, "ok")

    # 管理员权限装饰器，用于标识该方法需要管理员权限
    @flag.admin
    def actionPermissionRemove(self, to, permission):
        # 从站点设置的权限列表中移除指定权限
        self.site.settings["permissions"].remove(permission)
        # 保存设置
        self.site.saveSettings()
        # 更新 WebSocket，通知权限已移除
        self.site.updateWebsocket(permission_removed=permission)
        # 响应操作结果
        self.response(to, "ok")

    # 管理员权限装饰器，用于标识该方法需要管理员权限
    def actionPermissionDetails(self, to, permission):
        # 根据不同的权限返回不同的信息
        if permission == "ADMIN":
            self.response(to, _["Modify your client's configuration and access all site"] + " <span style='color: red'>" + _["(Dangerous!)"] + "</span>")
        elif permission == "NOSANDBOX":
            self.response(to, _["Modify your client's configuration and access all site"] + " <span style='color: red'>" + _["(Dangerous!)"] + "</span>")
        elif permission == "PushNotification":
            self.response(to, _["Send notifications"])
        else:
            self.response(to, "")

    # 管理员权限装饰器，用于标识该方法需要管理员权限
    def actionCertSet(self, to, domain):
        # 设置用于站点用户认证的证书
        self.user.setCert(self.site.address, domain)
        # 更新 WebSocket，通知证书已更改
        self.site.updateWebsocket(cert_changed=domain)
        # 响应操作结果
        self.response(to, "ok")

    # 管理员权限装饰器，用于标识该方法需要管理员权限
    def actionCertList(self, to):
        # 返回用户的证书列表
        back = []
        auth_address = self.user.getAuthAddress(self.site.address)
        for domain, cert in list(self.user.certs.items()):
            back.append({
                "auth_address": cert["auth_address"],
                "auth_type": cert["auth_type"],
                "auth_user_name": cert["auth_user_name"],
                "domain": domain,
                "selected": cert["auth_address"] == auth_address
            })
        return back
    # 列出所有站点信息
    @flag.admin  # 标记为管理员权限
    def actionSiteList(self, to, connecting_sites=False):
        ret = []  # 初始化返回结果列表
        for site in list(self.server.sites.values()):  # 遍历所有站点
            if not site.content_manager.contents.get("content.json") and not connecting_sites:  # 如果站点内容不完整且不需要连接站点
                continue  # 跳过当前站点
            ret.append(self.formatSiteInfo(site, create_user=False))  # 将站点信息格式化后添加到返回结果列表中，不生成认证地址
        self.response(to, ret)  # 发送返回结果列表

    # 加入所有站点的事件频道
    @flag.admin  # 标记为管理员权限
    def actionChannelJoinAllsite(self, to, channel):
        if channel not in self.channels:  # 如果频道不在频道列表中
            self.channels.append(channel)  # 将频道添加到频道列表中

        for site in list(self.server.sites.values()):  # 遍历所有站点
            if self not in site.websockets:  # 如果当前对象不在站点的 WebSocket 列表中
                site.websockets.append(self)  # 将当前对象添加到站点的 WebSocket 列表中

        self.response(to, "ok")  # 发送响应消息

    # 更新站点的 content.json
    def actionSiteUpdate(self, to, address, check_files=False, since=None, announce=False):
        def updateThread():  # 定义更新线程函数
            site.update(announce=announce, check_files=check_files, since=since)  # 更新站点内容
            self.response(to, "Updated")  # 发送更新完成的消息

        site = self.server.sites.get(address)  # 获取指定地址的站点
        if site and (site.address == self.site.address or "ADMIN" in self.site.settings["permissions"]):  # 如果站点存在且具有权限
            if not site.settings["serving"]:  # 如果站点未提供服务
                site.settings["serving"] = True  # 设置站点提供服务
                site.saveSettings()  # 保存站点设置

            gevent.spawn(updateThread)  # 使用协程启动更新线程
        else:
            self.response(to, {"error": "Unknown site: %s" % address})  # 发送错误消息，站点不存在

    # 暂停站点服务
    @flag.admin  # 标记为管理员权限
    # 暂停站点服务
    def actionSitePause(self, to, address):
        # 获取指定地址的站点对象
        site = self.server.sites.get(address)
        # 如果站点对象存在
        if site:
            # 设置站点服务状态为False
            site.settings["serving"] = False
            # 保存站点设置
            site.saveSettings()
            # 更新站点的websocket连接
            site.updateWebsocket()
            # 停止站点的工作线程
            site.worker_manager.stopWorkers()
            # 响应暂停操作
            self.response(to, "Paused")
        else:
            # 响应未知站点错误
            self.response(to, {"error": "Unknown site: %s" % address})

    # 恢复站点服务
    @flag.admin
    def actionSiteResume(self, to, address):
        # 获取指定地址的站点对象
        site = self.server.sites.get(address)
        # 如果站点对象存在
        if site:
            # 设置站点服务状态为True
            site.settings["serving"] = True
            # 保存站点设置
            site.saveSettings()
            # 异步启动站点更新，同时进行websocket连接更新
            gevent.spawn(site.update, announce=True)
            # 等待更新线程启动
            time.sleep(0.001)
            # 更新站点的websocket连接
            site.updateWebsocket()
            # 响应恢复操作
            self.response(to, "Resumed")
        else:
            # 响应未知站点错误
            self.response(to, {"error": "Unknown site: %s" % address})

    # 删除站点
    @flag.admin
    @flag.no_multiuser
    def actionSiteDelete(self, to, address):
        # 获取指定地址的站点对象
        site = self.server.sites.get(address)
        # 如果站点对象存在
        if site:
            # 删除站点
            site.delete()
            # 删除用户站点数据
            self.user.deleteSiteData(address)
            # 响应删除成功
            self.response(to, "Deleted")
            # 手动触发垃圾回收
            import gc
            gc.collect(2)
        else:
            # 响应未知站点错误
            self.response(to, {"error": "Unknown site: %s" % address})
    # 定义一个方法，用于克隆网站
    def cbSiteClone(self, to, address, root_inner_path="", target_address=None, redirect=True):
        # 发送通知，提示正在克隆网站
        self.cmd("notification", ["info", _["Cloning site..."]])
        # 获取指定地址的网站对象
        site = self.server.sites.get(address)
        # 初始化响应字典
        response = {}
        # 如果有目标地址
        if target_address:
            # 获取目标网站对象
            target_site = self.server.sites.get(target_address)
            # 获取目标网站的私钥
            privatekey = self.user.getSiteData(target_site.address).get("privatekey")
            # 克隆网站到目标地址
            site.clone(target_address, privatekey, root_inner_path=root_inner_path)
            # 发送通知，提示网站源代码已升级
            self.cmd("notification", ["done", _["Site source code upgraded!"]])
            # 发布网站
            site.publish()
            # 更新响应字典
            response = {"address": target_address}
        else:
            # 从用户的bip32种子生成一个新的网站
            new_address, new_address_index, new_site_data = self.user.getNewSiteData()
            # 克隆网站到新地址
            new_site = site.clone(new_address, new_site_data["privatekey"], address_index=new_address_index, root_inner_path=root_inner_path)
            # 设置新网站为自己的网站
            new_site.settings["own"] = True
            new_site.saveSettings()
            # 发送通知，提示网站已克隆
            self.cmd("notification", ["done", _["Site cloned"]])
            # 如果需要重定向，则执行重定向
            if redirect:
                self.cmd("redirect", "/%s" % new_address)
            # 异步执行新网站的公告
            gevent.spawn(new_site.announce)
            # 更新响应字典
            response = {"address": new_address}
        # 响应请求
        self.response(to, response)
        # 返回"ok"
        return "ok"

    # 标记方法为不支持多用户的方法
    @flag.no_multiuser
    # 克隆站点，将站点克隆到指定位置
    def actionSiteClone(self, to, address, root_inner_path="", target_address=None, redirect=True):
        # 如果地址不是站点地址，则返回错误信息
        if not SiteManager.site_manager.isAddress(address):
            self.response(to, {"error": "Not a site: %s" % address})
            return

        # 如果服务器中不存在该站点，则不暴露站点存在
        if not self.server.sites.get(address):
            return

        # 获取指定地址的站点
        site = self.server.sites.get(address)
        # 如果站点存在损坏文件
        if site.bad_files:
            # 遍历站点的损坏文件
            for bad_inner_path in list(site.bad_files.keys()):
                # 检查是否是用户文件
                is_user_file = "cert_signers" in site.content_manager.getRules(bad_inner_path)
                # 如果不是用户文件且不是content.json，则返回站点仍在同步中的错误信息
                if not is_user_file and bad_inner_path != "content.json":
                    self.cmd("notification", ["error", _["Clone error: Site still in sync"]])
                    return {"error": "Site still in sync"}

        # 如果当前用户有管理员权限
        if "ADMIN" in self.getPermissions(to):
            # 调用站点克隆方法
            self.cbSiteClone(to, address, root_inner_path, target_address, redirect)
        else:
            # 否则，发送确认信息，确认后调用站点克隆方法
            self.cmd(
                "confirm",
                [_["Clone site <b>%s</b>?"] % address, _["Clone"]],
                lambda res: self.cbSiteClone(to, address, root_inner_path, target_address, redirect)
            )

    # 设置站点大小限制
    @flag.admin
    @flag.no_multiuser
    def actionSiteSetLimit(self, to, size_limit):
        # 设置站点大小限制
        self.site.settings["size_limit"] = int(size_limit)
        # 保存站点设置
        self.site.saveSettings()
        # 响应结果
        self.response(to, "ok")
        # 更新站点的 WebSocket 连接
        self.site.updateWebsocket()
        # 下载站点内容，包括盲目包含的文件
        self.site.download(blind_includes=True)

    # 添加站点
    @flag.admin
    def actionSiteAdd(self, to, address):
        # 获取站点管理器
        site_manager = SiteManager.site_manager
        # 如果地址已经存在于站点管理器中，则返回站点已经添加的错误信息
        if address in site_manager.sites:
            return {"error": "Site already added"}
        else:
            # 如果地址需要添加，则返回"ok"，否则返回无效地址的错误信息
            if site_manager.need(address):
                return "ok"
            else:
                return {"error": "Invalid address"}

    # 异步运行标记，管理员权限标记
    @flag.async_run
    @flag.admin
    # 设置站点特定设置的数值
    def actionSiteSetSettingsValue(self, to, key, value):
        # 如果键不在指定的列表中，则返回错误信息
        if key not in ["modified_files_notification"]:
            return {"error": "Can't change this key"}

        # 设置站点特定设置的数值
        self.site.settings[key] = value

        # 返回成功信息
        return "ok"

    # 获取用户的站点设置
    def actionUserGetSettings(self, to):
        # 获取用户站点的设置
        settings = self.user.sites.get(self.site.address, {}).get("settings", {})
        # 返回设置信息
        self.response(to, settings)

    # 设置用户的站点设置
    def actionUserSetSettings(self, to, settings):
        # 设置用户的站点设置
        self.user.setSiteSettings(self.site.address, settings)
        # 返回成功信息
        self.response(to, "ok")

    # 获取用户的全局设置
    def actionUserGetGlobalSettings(self, to):
        # 获取用户的全局设置
        settings = self.user.settings
        # 返回设置信息
        self.response(to, settings)

    # 设置用户的全局设置
    @flag.admin
    def actionUserSetGlobalSettings(self, to, settings):
        # 设置用户的全局设置
        self.user.settings = settings
        self.user.save()
        # 返回成功信息
        self.response(to, "ok")

    # 获取服务器错误信息
    @flag.admin
    @flag.no_multiuser
    def actionServerErrors(self, to):
        return config.error_logger.lines

    # 更新服务器
    @flag.admin
    @flag.no_multiuser
    def actionServerUpdate(self, to):
        # 回调函数，用于更新服务器
        def cbServerUpdate(res):
            self.response(to, res)
            if not res:
                return False
            for websocket in self.server.websockets:
                websocket.cmd(
                    "notification",
                    ["info", _["Updating ZeroNet client, will be back in a few minutes..."], 20000]
                )
                websocket.cmd("updating")

            import main
            main.update_after_shutdown = True
            main.restart_after_shutdown = True
            SiteManager.site_manager.save()
            main.file_server.stop()
            main.ui_server.stop()

        # 发送确认消息，执行更新服务器的回调函数
        self.cmd(
            "confirm",
            [_["Update <b>ZeroNet client</b> to latest version?"], _["Update"]],
            cbServerUpdate
        )

    # 异步运行，不支持多用户的服务器操作
    @flag.admin
    @flag.async_run
    @flag.no_multiuser
    # 检查服务器端口是否开放
    def actionServerPortcheck(self, to):
        # 导入 main 模块
        import main
        # 获取 main 模块中的 file_server 对象
        file_server = main.file_server
        # 调用 file_server 对象的 portCheck 方法
        file_server.portCheck()
        # 向指定目标发送 file_server.port_opened 的响应
        self.response(to, file_server.port_opened)

    # 服务器关闭操作，需要管理员权限，且不允许多用户同时操作
    @flag.admin
    @flag.no_multiuser
    def actionServerShutdown(self, to, restart=False):
        # 导入 main 模块
        import main
        # 定义服务器关闭回调函数
        def cbServerShutdown(res):
            # 向指定目标发送服务器关闭结果的响应
            self.response(to, res)
            # 如果关闭失败，则返回 False
            if not res:
                return False
            # 如果需要重启，则设置 main 模块的 restart_after_shutdown 为 True
            if restart:
                main.restart_after_shutdown = True
            # 停止文件服务器和 UI 服务器
            main.file_server.stop()
            main.ui_server.stop()

        # 如果需要重启，则设置消息为重启消息，否则为关闭消息
        if restart:
            message = [_["Restart <b>ZeroNet client</b>?"], _["Restart"]]
        else:
            message = [_["Shut down <b>ZeroNet client</b>?"], _["Shut down"]]
        # 发送确认消息，确认后执行 cbServerShutdown 回调函数
        self.cmd("confirm", message, cbServerShutdown)

    # 显示指定目录的内容，需要管理员权限，且不允许多用户同时操作
    @flag.admin
    @flag.no_multiuser
    def actionServerShowdirectory(self, to, directory="backup", inner_path=""):
        # 如果请求不是来自本地，则返回错误响应
        if self.request.env["REMOTE_ADDR"] != "127.0.0.1":
            return self.response(to, {"error": "Only clients from 127.0.0.1 allowed to run this command"})

        # 导入 webbrowser 模块
        import webbrowser
        # 根据不同的目录参数设置不同的路径
        if directory == "backup":
            path = os.path.abspath(config.data_dir)
        elif directory == "log":
            path = os.path.abspath(config.log_dir)
        elif directory == "site":
            path = os.path.abspath(self.site.storage.getPath(helper.getDirname(inner_path)))

        # 如果路径是一个目录，则打开该目录，并返回 "ok" 响应
        if os.path.isdir(path):
            self.log.debug("Opening: %s" % path)
            webbrowser.open('file://' + path)
            return self.response(to, "ok")
        # 如果路径不是一个目录，则返回错误响应
        else:
            return self.response(to, {"error": "Not a directory"})

    # 需要管理员权限，且不允许多用户同时操作
    @flag.admin
    @flag.no_multiuser
```