# `ZeroNet\plugins\disabled-Multiuser\MultiuserPlugin.py`

```py
# 导入 re 模块，用于正则表达式操作
import re
# 导入 sys 模块，用于系统相关操作
import sys
# 导入 json 模块，用于 JSON 数据的解析和生成
import json

# 从 Config 模块中导入 config 对象
from Config import config
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Crypt 模块中导入 CryptBitcoin 类
from Crypt import CryptBitcoin
# 从当前目录下的 UserPlugin 模块中导入所有内容
from . import UserPlugin
# 从 util.Flag 模块中导入 flag 对象
from util.Flag import flag
# 从 Translate 模块中导入 translate 函数，并重命名为 _
from Translate import translate as _

# 在插件加载后执行的函数，用于导入插件主机类
@PluginManager.afterLoad
def importPluginnedClasses():
    # 声明全局变量 UserManager
    global UserManager
    # 从 User 模块中导入 UserManager 类
    from User import UserManager

# 尝试从 users.json 文件中加载本地主地址集合，如果失败则将本地主地址集合设为空集合
try:
    local_master_addresses = set(json.load(open("%s/users.json" % config.data_dir)).keys())  # Users in users.json
except Exception as err:
    local_master_addresses = set()

# 将 UiRequestPlugin 类注册到 PluginManager 的"UiRequest"插件中
@PluginManager.registerTo("UiRequest")
class UiRequestPlugin(object):
    # 初始化函数，接受任意参数
    def __init__(self, *args, **kwargs):
        # 获取用户管理器对象
        self.user_manager = UserManager.user_manager
        # 调用父类的初始化函数
        super(UiRequestPlugin, self).__init__(*args, **kwargs)

    # 创建新用户并注入用户欢迎消息（如果需要）
    # 返回：包含注入的 Html 主体
    # 根据请求的 cookies 获取当前用户
    # 返回：用户对象或如果没有匹配则返回 None
    def getCurrentUser(self):
        # 获取请求的 cookies
        cookies = self.getCookies()
        user = None
        # 如果 cookies 中包含"master_address"
        if "master_address" in cookies:
            # 获取用户列表
            users = self.user_manager.list()
            # 根据 cookies 中的"master_address"获取用户对象
            user = users.get(cookies["master_address"])
        # 返回用户对象
        return user

# 将 UiWebsocketPlugin 类注册到 PluginManager 的"UiWebsocket"插件中
@PluginManager.registerTo("UiWebsocket")
class UiWebsocketPlugin(object):
    # 初始化函数，接受任意参数
    def __init__(self, *args, **kwargs):
        # 如果配置中禁止新站点的多用户模式
        if config.multiuser_no_new_sites:
            # 标记为不支持多用户
            flag.no_multiuser(self.actionMergerSiteAdd)

        # 调用父类的初始化函数
        super(UiWebsocketPlugin, self).__init__(*args, **kwargs)

    # 让页面知道我们正在多用户模式下运行
    # 格式化服务器信息，包括是否多用户、管理员地址等
    def formatServerInfo(self):
        # 调用父类方法获取服务器信息
        server_info = super(UiWebsocketPlugin, self).formatServerInfo()
        # 设置多用户标志为True
        server_info["multiuser"] = True
        # 如果当前用户具有管理员权限
        if "ADMIN" in self.site.settings["permissions"]:
            # 设置管理员地址为当前用户的主地址
            server_info["master_address"] = self.user.master_address
            # 判断当前用户是否为多用户管理员
            is_multiuser_admin = config.multiuser_local or self.user.master_address in local_master_addresses
            server_info["multiuser_admin"] = is_multiuser_admin
        # 返回服务器信息
        return server_info

    # 显示当前用户的主种子
    @flag.admin
    def actionUserShowMasterSeed(self, to):
        # 构建显示主种子的消息
        message = "<b style='padding-top: 5px; display: inline-block'>Your unique private key:</b>"
        message += "<div style='font-size: 84%%; background-color: #FFF0AD; padding: 5px 8px; margin: 9px 0px'>%s</div>" % self.user.master_seed
        message += "<small>(Save it, you can access your account using this information)</small>"
        # 发送通知消息
        self.cmd("notification", ["info", message])

    # 注销用户
    @flag.admin
    def actionUserLogout(self, to):
        # 构建注销用户的消息
        message = "<b>You have been logged out.</b> <a href='#Login' class='button' id='button_notification'>Login to another account</a>"
        # 发送通知消息，设置显示时间为永久
        self.cmd("notification", ["done", message, 1000000])  # 1000000 = Show ~forever :)

        # 构建注销用户的脚本
        script = "document.cookie = 'master_address=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/';"
        script += "$('#button_notification').on('click', function() { zeroframe.cmd(\"userLoginForm\", []); });"
        # 注入脚本
        self.cmd("injectScript", script)
        # 从用户管理器中删除用户
        user_manager = UserManager.user_manager
        if self.user.master_address in user_manager.users:
            if not config.multiuser_local:
                del user_manager.users[self.user.master_address]
            self.response(to, "Successful logout")
        else:
            self.response(to, "User not found")

    @flag.admin
    # 定义一个名为 actionUserSet 的方法，接受两个参数：to 和 master_address
    def actionUserSet(self, to, master_address):
        # 从 UserManager 类中获取用户管理器实例
        user_manager = UserManager.user_manager
        # 通过 master_address 获取用户信息
        user = user_manager.get(master_address)
        # 如果用户不存在，则抛出异常
        if not user:
            raise Exception("No user found")

        # 构建 JavaScript 脚本，设置 cookie 信息和执行 zeroframe 命令
        script = "document.cookie = 'master_address=%s;path=/;max-age=2592000;';" % master_address
        script += "zeroframe.cmd('wrapperReload', ['login=done']);"
        # 发送通知命令，提示用户登录成功并重新加载页面
        self.cmd("notification", ["done", "Successful login, reloading page..."])
        # 注入 JavaScript 脚本
        self.cmd("injectScript", script)

        # 响应请求，返回 "ok"
        self.response(to, "ok")

    # 标记该方法需要管理员权限
    @flag.admin
    # 用户选择表单操作，接收参数 to
    def actionUserSelectForm(self, to):
        # 如果不是多用户本地模式，则抛出异常
        if not config.multiuser_local:
            raise Exception("Only allowed in multiuser local mode")
        # 获取用户管理器
        user_manager = UserManager.user_manager
        # 构建表单内容
        body = "<span style='padding-bottom: 5px; display: inline-block'>" + "Change account:" + "</span>"
        # 遍历用户管理器中的用户列表
        for master_address, user in user_manager.list().items():
            # 判断当前用户是否为活跃用户
            is_active = self.user.master_address == master_address
            # 如果用户有证书
            if user.certs:
                # 获取第一个证书的信息
                first_cert = next(iter(user.certs.keys()))
                title = "%s@%s" % (user.certs[first_cert]["auth_user_name"], first_cert)
            else:
                title = user.master_address
                # 如果用户站点数量小于2且不是活跃用户，则跳过
                if len(user.sites) < 2 and not is_active:  # Avoid listing ad-hoc created users
                    continue
            # 根据用户活跃状态设置 CSS 类
            if is_active:
                css_class = "active"
            else:
                css_class = "noclass"
            # 构建用户选择链接
            body += "<a href='#Select+user' class='select select-close user %s' title='%s'>%s</a>" % (css_class, user.master_address, title)

        # 构建 JavaScript 脚本
        script = """
             $(".notification .select.user").on("click", function() {
                $(".notification .select").removeClass('active')
                zeroframe.response(%s, this.title)
                return false
             })
        """ % self.next_message_id

        # 发送通知，询问用户选择
        self.cmd("notification", ["ask", body], lambda master_address: self.actionUserSet(to, master_address))
        # 注入 JavaScript 脚本
        self.cmd("injectScript", script)

    # 显示登录表单
    def actionUserLoginForm(self, to):
        self.cmd("prompt", ["<b>Login</b><br>Your private key:", "password", "Login"], self.responseUserLogin)

    # 登录表单提交
    # 响应用户登录请求，使用主密钥生成用户地址，并进行相应操作
    def responseUserLogin(self, master_seed):
        # 获取用户管理器实例
        user_manager = UserManager.user_manager
        # 根据主密钥生成用户地址
        user = user_manager.get(CryptBitcoin.privatekeyToAddress(master_seed))
        # 如果用户不存在，则创建用户
        if not user:
            user = user_manager.create(master_seed=master_seed)
        # 如果用户存在主地址
        if user.master_address:
            # 生成设置 cookie 的 JavaScript 脚本，用于设置 master_address，并刷新页面
            script = "document.cookie = 'master_address=%s;path=/;max-age=2592000;';" % user.master_address
            script += "zeroframe.cmd('wrapperReload', ['login=done']);"
            # 发送成功登录的通知，并执行设置 cookie 的 JavaScript 脚本
            self.cmd("notification", ["done", "Successful login, reloading page..."])
            self.cmd("injectScript", script)
        else:
            # 发送错误通知，提示主密钥无效
            self.cmd("notification", ["error", "Error: Invalid master seed"])
            # 执行用户登录表单操作
            self.actionUserLoginForm(0)

    # 检查命令是否有权限执行
    def hasCmdPermission(self, cmd):
        # 获取命令对应的标志
        flags = flag.db.get(self.getCmdFuncName(cmd), ())
        # 判断是否为公共代理用户，并且命令不支持多用户模式
        is_public_proxy_user = not config.multiuser_local and self.user.master_address not in local_master_addresses
        if is_public_proxy_user and "no_multiuser" in flags:
            # 发送信息通知，提示该功能在代理上被禁用
            self.cmd("notification", ["info", _("This function ({cmd}) is disabled on this proxy!"])
            return False
        else:
            # 调用父类方法，检查命令是否有权限执行
            return super(UiWebsocketPlugin, self).hasCmdPermission(cmd)

    # 添加权限操作
    def actionPermissionAdd(self, to, permission):
        # 判断是否为公共代理用户
        is_public_proxy_user = not config.multiuser_local and self.user.master_address not in local_master_addresses
        # 如果权限为 NOSANDBOX 并且为公共代理用户
        if permission == "NOSANDBOX" and is_public_proxy_user:
            # 发送信息通知，提示在代理上无法禁用沙盒
            self.cmd("notification", ["info", "You can't disable sandbox on this proxy!"])
            # 响应操作，返回错误信息
            self.response(to, {"error": "Denied by proxy"})
            return False
        else:
            # 调用父类方法，执行添加权限操作
            return super(UiWebsocketPlugin, self).actionPermissionAdd(to, permission)
# 将 ConfigPlugin 类注册到 PluginManager 中的 ConfigPlugin 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数
    def createArguments(self):
        # 添加一个参数组到解析器中，用于多用户插件
        group = self.parser.add_argument_group("Multiuser plugin")
        # 添加一个参数到参数组中，用于启用不安全的 UI 功能并将用户写入磁盘
        group.add_argument('--multiuser_local', help="Enable unsafe Ui functions and write users to disk", action='store_true')
        # 添加一个参数到参数组中，用于拒绝普通用户添加新站点
        group.add_argument('--multiuser_no_new_sites', help="Denies adding new sites by normal users", action='store_true')

        # 调用父类的 createArguments 方法
        return super(ConfigPlugin, self).createArguments()
```