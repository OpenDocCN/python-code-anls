# `ZeroNet\plugins\disabled-UiPassword\UiPasswordPlugin.py`

```
# 导入必要的模块
import string
import random
import time
import json
import re
import os

# 从Config模块中导入config变量
from Config import config
# 从Plugin模块中导入PluginManager类
from Plugin import PluginManager
# 从util模块中导入helper函数
from util import helper

# 获取当前文件所在目录
plugin_dir = os.path.dirname(__file__)

# 如果sessions变量不在局部变量中，则创建sessions和whitelisted_client_ids字典，用于在模块重新加载时保持会话
if "sessions" not in locals().keys():
    sessions = {}
    whitelisted_client_ids = {}

# 定义一个函数，用于显示密码建议
def showPasswordAdvice(password):
    error_msgs = []
    # 如果密码为空或不是字符串类型，则添加错误信息到列表中
    if not password or not isinstance(password, str):
        error_msgs.append("You have enabled <b>UiPassword</b> plugin, but you forgot to set a password!")
    # 如果密码长度小于8，则添加错误信息到列表中
    elif len(password) < 8:
        error_msgs.append("You are using a very short UI password!")
    return error_msgs

# 将UiRequest类注册到PluginManager的UiRequest插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 将sessions和whitelisted_client_ids变量赋值给类属性
    sessions = sessions
    whitelisted_client_ids = whitelisted_client_ids
    # 初始化last_cleanup变量为当前时间戳
    last_cleanup = time.time()

    # 定义一个方法，用于获取客户端ID
    def getClientId(self):
        return self.env["REMOTE_ADDR"] + " - " + self.env["HTTP_USER_AGENT"]

    # 定义一个方法，用于将客户端ID添加到白名单
    def whitelistClientId(self, session_id=None):
        # 如果session_id为空，则从cookies中获取session_id
        if not session_id:
            session_id = self.getCookies().get("session_id")
        # 获取客户端ID
        client_id = self.getClientId()
        # 如果客户端ID在whitelisted_client_ids中，则更新其更新时间并返回False
        if client_id in self.whitelisted_client_ids:
            self.whitelisted_client_ids[client_id]["updated"] = time.time()
            return False
        # 否则将客户端ID添加到whitelisted_client_ids中
        self.whitelisted_client_ids[client_id] = {
            "added": time.time(),
            "updated": time.time(),
            "session_id": session_id
        }
    # 定义路由方法，根据路径进行处理
    def route(self, path):
        # 如果配置了 UI 限制，并且请求的 IP 不在限制列表中，则返回 403 错误
        if config.ui_restrict and self.env['REMOTE_ADDR'] not in config.ui_restrict:
            return self.error403(details=False)
        # 如果请求的路径以 "favicon.ico" 结尾，则返回对应的图标文件
        if path.endswith("favicon.ico"):
            return self.actionFile("src/Ui/media/img/favicon.ico")
        else:
            # 如果配置了 UI 密码
            if config.ui_password:
                # 如果距离上次清理会话已经过去了一个小时，则进行会话清理
                if time.time() - self.last_cleanup > 60 * 60:  # Cleanup expired sessions every hour
                    self.sessionCleanup()
                # 验证会话
                session_id = self.getCookies().get("session_id")
                # 如果会话 ID 不在会话列表中，并且客户端 ID 不在白名单中，则显示登录页面
                if session_id not in self.sessions and self.getClientId() not in self.whitelisted_client_ids:
                    return self.actionLogin()
            # 调用父类的路由方法处理请求
            return super(UiRequestPlugin, self).route(path)

    # 定义动作包装方法，用于处理特定路径的请求
    def actionWrapper(self, path, *args, **kwargs):
        # 如果配置了 UI 密码，并且需要包装器
        if config.ui_password and self.isWrapperNecessary(path):
            session_id = self.getCookies().get("session_id")
            # 如果会话 ID 不在会话列表中，则返回登录页面
            if session_id not in self.sessions:
                return self.actionLogin()
            else:
                # 将客户端 ID 加入白名单
                self.whitelistClientId()

        # 调用父类的动作包装方法处理请求
        return super().actionWrapper(path, *args, **kwargs)

    # 定义动作：登录
    @helper.encodeResponse
    # 登录操作，返回登录页面模板
    def actionLogin(self):
        # 读取登录页面模板内容
        template = open(plugin_dir + "/login.html").read()
        # 发送 HTTP 头部
        self.sendHeader()
        # 获取 HTTP POST 数据
        posted = self.getPosted()
        if posted:  # 验证 HTTP POST 数据
            if self.sessionCheckPassword(posted.get("password")):
                # 有效密码，创建会话
                session_id = self.randomString(26)
                self.sessions[session_id] = {
                    "added": time.time(),
                    "keep": posted.get("keep")
                }
                self.whitelistClientId(session_id)

                # 重定向到主页或引用页
                url = self.env.get("HTTP_REFERER", "")
                if not url or re.sub(r"\?.*", "", url).endswith("/Login"):
                    url = "/" + config.homepage
                cookie_header = ('Set-Cookie', "session_id=%s;path=/;max-age=2592000;" % session_id)  # 最大有效期 = 30 天
                self.start_response('301 Redirect', [('Location', url), cookie_header])
                yield "Redirecting..."

            else:
                # 无效密码，再次显示登录表单
                template = template.replace("{result}", "bad_password")
        yield template

    # 生成指定长度的随机字符串
    def randomString(self, nchars):
        return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(nchars))

    # 验证会话密码
    def sessionCheckPassword(self, password):
        return password == config.ui_password

    # 删除会话
    def sessionDelete(self, session_id):
       del self.sessions[session_id]

       # 删除白名单中与会话相关的客户端 ID
       for client_id in list(self.whitelisted_client_ids):
            if self.whitelisted_client_ids[client_id]["session_id"] == session_id:
                del self.whitelisted_client_ids[client_id]
    # 清理会话，删除过期的会话
    def sessionCleanup(self):
        # 记录最后一次清理的时间戳
        self.last_cleanup = time.time()
        # 遍历会话字典中的每个会话
        for session_id, session in list(self.sessions.items()):
            # 如果会话需要保留且已经存在超过60天，则删除该会话
            if session["keep"] and time.time() - session["added"] > 60 * 60 * 24 * 60:  # Max 60days for keep sessions
                self.sessionDelete(session_id)
            # 如果会话不需要保留且已经存在超过24小时，则删除该会话
            elif not session["keep"] and time.time() - session["added"] > 60 * 60 * 24:  # Max 24h for non-keep sessions
                self.sessionDelete(session_id)

    # 动作：显示会话
    @helper.encodeResponse
    def actionSessions(self):
        # 发送 HTTP 头部
        self.sendHeader()
        # 生成会话信息的 JSON 格式并返回
        yield "<pre>"
        yield json.dumps(self.sessions, indent=4)
        yield "\r\n"
        yield json.dumps(self.whitelisted_client_ids, indent=4)

    # 动作：注销
    @helper.encodeResponse
    def actionLogout(self):
        # 会话 ID 必须作为 GET 参数传递，或者在没有引用的情况下调用，以避免远程注销
        session_id = self.getCookies().get("session_id")
        if not self.env.get("HTTP_REFERER") or session_id == self.get.get("session_id"):
            # 如果会话 ID 存在于会话字典中，则删除该会话
            if session_id in self.sessions:
                self.sessionDelete(session_id)

            # 发送 301 重定向响应头，并设置 cookie 为删除状态
            self.start_response('301 Redirect', [
                ('Location', "/"),
                ('Set-Cookie', "session_id=deleted; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT")
            ])
            yield "Redirecting..."
        else:
            # 发送 HTTP 头部
            self.sendHeader()
            yield "Error: Invalid session id"
# 将 ConfigPlugin 类注册到 PluginManager 的 ConfigPlugin 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数
    def createArguments(self):
        # 添加一个参数组到解析器中
        group = self.parser.add_argument_group("UiPassword plugin")
        # 添加一个参数到参数组中，用于指定访问 UiServer 的密码
        group.add_argument('--ui_password', help='Password to access UiServer', default=None, metavar="password")

        # 调用父类的 createArguments 方法
        return super(ConfigPlugin, self).createArguments()


# 从 Translate 模块中导入 translate 函数，并将 UiWebsocketPlugin 类注册到 PluginManager 的 UiWebsocket 插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 执行 Ui 注销操作
    def actionUiLogout(self, to):
        # 获取用户权限
        permissions = self.getPermissions(to)
        # 如果用户不是管理员，返回权限不足的提示
        if "ADMIN" not in permissions:
            return self.response(to, "You don't have permission to run this command")

        # 获取 session_id
        session_id = self.request.getCookies().get("session_id", "")
        # 发送重定向命令到 '/Logout?session_id=%s' % session_id
        self.cmd("redirect", '/Logout?session_id=%s' % session_id)

    # 添加主页通知
    def addHomepageNotifications(self):
        # 显示密码建议的错误消息
        error_msgs = showPasswordAdvice(config.ui_password)
        # 遍历错误消息列表，将翻译后的消息添加到站点通知中
        for msg in error_msgs:
            self.site.notifications.append(["error", lang[msg]])

        # 调用父类的 addHomepageNotifications 方法
        return super(UiWebsocketPlugin, self).addHomepageNotifications()
```