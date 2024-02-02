# `ZeroNet\src\Ui\UiRequest.py`

```py
# 导入时间模块
import time
# 导入正则表达式模块
import re
# 导入操作系统模块
import os
# 导入 MIME 类型模块
import mimetypes
# 导入 JSON 模块
import json
# 导入 HTML 模块
import html
# 导入 URL 编码模块
import urllib
# 导入套接字模块
import socket

# 导入协程模块
import gevent

# 从 Config 模块中导入 config 对象
from Config import config
# 从 Site 模块中导入 SiteManager 类
from Site import SiteManager
# 从 User 模块中导入 UserManager 类
from User import UserManager
# 从 Plugin 模块中导入 PluginManager 类
from Plugin import PluginManager
# 从 Ui.UiWebsocket 模块中导入 UiWebsocket 类
from Ui.UiWebsocket import UiWebsocket
# 从 Crypt 模块中导入 CryptHash 类
from Crypt import CryptHash
# 从 util 模块中导入 helper 模块
from util import helper

# 定义状态码对应的文本信息
status_texts = {
    200: "200 OK",
    206: "206 Partial Content",
    400: "400 Bad Request",
    403: "403 Forbidden",
    404: "404 Not Found",
    500: "500 Internal Server Error",
}

# 定义文件扩展名对应的 MIME 类型
content_types = {
    "asc": "application/pgp-keys",
    "css": "text/css",
    "gpg": "application/pgp-encrypted",
    "html": "text/html",
    "js": "application/javascript",
    "json": "application/json",
    "oga": "audio/ogg",
    "ogg": "application/ogg",
    "ogv": "video/ogg",
    "sig": "application/pgp-signature",
    "txt": "text/plain",
    "webmanifest": "application/manifest+json",
    "wasm": "application/wasm",
    "webp": "image/webp"
}

# 定义安全错误的异常类
class SecurityError(Exception):
    pass

# 使用 PluginManager.acceptPlugins 装饰器注册 UiRequest 类
@PluginManager.acceptPlugins
class UiRequest(object):
    # 初始化函数，接受服务器、GET参数、环境设置和开始响应函数作为参数
    def __init__(self, server, get, env, start_response):
        # 如果有服务器对象，则将其赋值给self.server，并将其日志对象赋值给self.log
        if server:
            self.server = server
            self.log = server.log
        # 将GET参数赋值给self.get
        self.get = get  # Get parameters
        # 将环境设置赋值给self.env
        self.env = env  # Enviroment settings
        # 开始响应函数
        self.start_response = start_response  # Start response function
        # 用户对象初始化为None
        self.user = None
        # 脚本标签在包装HTML中的随机数初始化为None
        self.script_nonce = None  # Nonce for script tags in wrapper html

    # 学习主机函数，将主机添加到允许的主机集合中
    def learnHost(self, host):
        self.server.allowed_hosts.add(host)
        self.server.log.info("Added %s as allowed host" % host)

    # 检查主机是否允许访问
    def isHostAllowed(self, host):
        # 如果主机在允许的主机集合中，则返回True
        if host in self.server.allowed_hosts:
            return True
        # 如果主机是IP地址，则学习该主机并返回True
        if helper.isIp(host):
            self.learnHost(host)
            return True
        # 如果主机带有端口号且主机地址是IP地址，则学习该主机并返回True
        if ":" in host and helper.isIp(host.rsplit(":", 1)[0]):  # Test without port
            self.learnHost(host)
            return True
        # 如果是代理请求，则支持Chrome扩展代理
        if self.isProxyRequest():
            # 如果是域名，则返回True，否则返回False
            if self.isDomain(host):
                return True
            else:
                return False
        # 默认返回False
        return False

    # 检查地址是否为域名
    def isDomain(self, address):
        return self.server.site_manager.isDomainCached(address)

    # 解析域名
    def resolveDomain(self, domain):
        return self.server.site_manager.resolveDomainCached(domain)
    # 根据路径调用请求处理函数
    # 请求由 Chrome 扩展或透明代理代理
    def isProxyRequest(self):
        return self.env["PATH_INFO"].startswith("http://") or (self.server.allow_trans_proxy and self.isDomain(self.env.get("HTTP_HOST")))
    
    # 检查是否为 WebSocket 请求
    def isWebSocketRequest(self):
        return self.env.get("HTTP_UPGRADE") == "websocket"
    
    # 检查是否为 Ajax 请求
    def isAjaxRequest(self):
        return self.env.get("HTTP_X_REQUESTED_WITH") == "XMLHttpRequest"
    
    # 根据文件名获取 MIME 类型
    def getContentType(self, file_name):
        file_name = file_name.lower()
        ext = file_name.rsplit(".", 1)[-1]
    
        if ext in content_types:
            content_type = content_types[ext]
        elif ext in ("ttf", "woff", "otf", "woff2", "eot", "sfnt", "collection"):
            content_type = "font/%s" % ext
        else:
            content_type = mimetypes.guess_type(file_name)[0]
    
        if not content_type:
            content_type = "application/octet-stream"
    
        return content_type.lower()
    
    # 返回： <dict> 发布的变量
    def getPosted(self):
        if self.env['REQUEST_METHOD'] == "POST":
            return dict(urllib.parse.parse_qsl(
                self.env['wsgi.input'].readline().decode()
            ))
        else:
            return {}
    
    # 返回： <dict> 基于 self.env 的 Cookies
    def getCookies(self):
        raw_cookies = self.env.get('HTTP_COOKIE')
        if raw_cookies:
            cookies = urllib.parse.parse_qsl(raw_cookies)
            return {key.strip(): val for key, val in cookies}
        else:
            return {}
    
    # 获取当前用户
    def getCurrentUser(self):
        if self.user:
            return self.user  # 缓存
        self.user = UserManager.user_manager.get()  # 获取用户
        if not self.user:
            self.user = UserManager.user_manager.create()
        return self.user
    # 获取请求的 URL
    def getRequestUrl(self):
        # 如果是代理请求
        if self.isProxyRequest():
            # 如果路径以"http://zero/"开头，则直接返回路径
            if self.env["PATH_INFO"].startswith("http://zero/"):
                return self.env["PATH_INFO"]
            else:  # 否则在路径前加上"http://zero/"
                return self.env["PATH_INFO"].replace("http://", "http://zero/", 1)
        else:
            # 如果不是代理请求，则返回完整的 URL
            return self.env["wsgi.url_scheme"] + "://" + self.env["HTTP_HOST"] + self.env["PATH_INFO"]

    # 获取引用页面的 URL
    def getReferer(self):
        referer = self.env.get("HTTP_REFERER")
        # 如果有引用页面，并且是代理请求，并且引用页面不是以"http://zero/"开头，则在引用页面前加上"http://zero/"
        if referer and self.isProxyRequest() and not referer.startswith("http://zero/"):
            return referer.replace("http://", "http://zero/", 1)
        else:
            return referer

    # 检查是否支持脚本随机数
    def isScriptNonceSupported(self):
        user_agent = self.env.get("HTTP_USER_AGENT")
        # 根据用户代理判断是否支持脚本随机数
        if "Edge/" in user_agent:
            is_script_nonce_supported = False
        elif "Safari/" in user_agent and "Chrome/" not in user_agent:
            is_script_nonce_supported = False
        else:
            is_script_nonce_supported = True
        return is_script_nonce_supported

    # 发送响应头
    # 渲染模板
    def render(self, template_path, *args, **kwargs):
        # 读取模板文件内容
        template = open(template_path, encoding="utf8").read()

        # 定义替换函数，用于替换模板中的变量
        def renderReplacer(m):
            if m.group(1) in kwargs:
                return "%s" % kwargs.get(m.group(1), "")
            else:
                return m.group(0)

        # 使用替换函数替换模板中的变量
        template_rendered = re.sub("{(.*?)}", renderReplacer, template)

        # 返回编码后的渲染结果
        return template_rendered.encode("utf8")
    # 检查是否需要包装路径
    def isWrapperNecessary(self, path):
        # 使用正则表达式匹配路径，提取地址和内部路径
        match = re.match(r"/(?P<address>[A-Za-z0-9\._-]+)(?P<inner_path>/.*|$)", path)

        # 如果没有匹配到，则需要包装路径
        if not match:
            return True

        # 提取内部路径并去除开头的斜杠
        inner_path = match.group("inner_path").lstrip("/")
        # 如果内部路径为空或者路径以斜杠结尾，则为目录
        if not inner_path or path.endswith("/"):  # It's a directory
            # 获取 index.html 的内容类型
            content_type = self.getContentType("index.html")
        else:  # It's a file
            # 获取内部路径的内容类型
            content_type = self.getContentType(inner_path)

        # 判断内容类型是否为 HTML 文件
        is_html_file = "html" in content_type or "xhtml" in content_type

        return is_html_file

    # 格式化重定向页面
    @helper.encodeResponse
    def formatRedirect(self, url):
        return """
            <html>
            <body>
            Redirecting to <a href="{0}" target="_top">{0}</a>
            <script>
            window.top.location = "{0}"
            </script>
            </body>
            </html>
        """.format(html.escape(url))

    # 重定向到指定 URL
    def actionRedirect(self, url):
        # 发送 301 重定向响应
        self.start_response('301 Redirect', [('Location', str(url))])
        # 返回格式化后的重定向页面
        yield self.formatRedirect(url)

    # 执行首页重定向
    def actionIndex(self):
        # 执行重定向到首页
        return self.actionRedirect("/" + config.homepage + "/")

    # 获取站点 URL
    def getSiteUrl(self, address):
        # 如果是代理请求，则返回 zero 站点的地址
        if self.isProxyRequest():
            return "http://zero/" + address
        else:
            return "/" + address

    # 获取 WebSocket 服务器地址
    def getWsServerUrl(self):
        # 如果是代理请求
        if self.isProxyRequest():
            # 如果是本地客户端，则服务器地址也应该是 127.0.0.1
            if self.env["REMOTE_ADDR"] == "127.0.0.1":
                server_url = "http://127.0.0.1:%s" % self.env["SERVER_PORT"]
            else:  # 远程客户端，使用 SERVER_NAME 作为服务器的真实地址
                server_url = "http://%s:%s" % (self.env["SERVER_NAME"], self.env["SERVER_PORT"])
        else:
            server_url = ""
        return server_url
    # 处理查询字符串，将其中的zeronet_peers参数解析并添加到指定站点中
    def processQueryString(self, site, query_string):
        # 从查询字符串中匹配zeronet_peers参数
        match = re.search("zeronet_peers=(.*?)(&|$)")
        if match:
            # 从查询字符串中移除匹配到的zeronet_peers参数
            query_string = query_string.replace(match.group(0), "")
            num_added = 0
            # 遍历匹配到的zeronet_peers参数中的每个peer
            for peer in match.group(1).split(","):
                # 如果peer不符合IP:端口的格式，则跳过
                if not re.match(".*?:[0-9]+$", peer):
                    continue
                # 将peer按冒号分割为IP和端口，并将其添加到指定站点中
                ip, port = peer.rsplit(":", 1)
                if site.addPeer(ip, int(port), source="query_string"):
                    num_added += 1
            # 记录通过查询字符串添加的peer数量
            site.log.debug("%s peers added by query string" % num_added)

        return query_string

    # 创建一个新的包装器nonce，允许获取一个不带包装器的html文件
    def getWrapperNonce(self):
        # 生成一个随机的包装器nonce
        wrapper_nonce = CryptHash.random()
        # 将包装器nonce添加到服务器的包装器nonce列表中
        self.server.wrapper_nonces.append(wrapper_nonce)
        return wrapper_nonce

    # 获取脚本nonce
    def getScriptNonce(self):
        if not self.script_nonce:
            # 如果脚本nonce不存在，则生成一个base64编码的随机脚本nonce
            self.script_nonce = CryptHash.random(encoding="base64")

        return self.script_nonce

    # 创建一个新的添加nonce，允许获取一个站点
    def getAddNonce(self):
        # 生成一个随机的添加nonce
        add_nonce = CryptHash.random()
        # 将添加nonce添加到服务器的添加nonce列表中
        self.server.add_nonces.append(add_nonce)
        return add_nonce

    # 判断两个URL是否属于同源
    def isSameOrigin(self, url_a, url_b):
        if not url_a or not url_b:
            return False

        # 将URL中的/raw/替换为/
        url_a = url_a.replace("/raw/", "/")
        url_b = url_b.replace("/raw/", "/")

        # 定义同源的正则表达式模式
        origin_pattern = "http[s]{0,1}://(.*?/.*?/).*"
        is_origin_full = re.match(origin_pattern, url_a)
        if not is_origin_full:  # 如果Origin看起来被修剪到主机，只需要相同的主机
            origin_pattern = "http[s]{0,1}://(.*?/).*"

        # 从URL中提取Origin
        origin_a = re.sub(origin_pattern, "\\1", url_a)
        origin_b = re.sub(origin_pattern, "\\1", url_b)

        # 判断两个URL的Origin是否相同
        return origin_a == origin_b

    # 从URL路径中返回{address: 1Site.., inner_path: /data/users.json}
    # 解析路径，将反斜杠替换为斜杠
    def parsePath(self, path):
        path = path.replace("\\", "/")
        # 将路径中的 "/index.html/" 替换为 "/"
        path = path.replace("/index.html/", "/")  # Base Backward compatibility fix
        # 如果路径以斜杠结尾，则将其修改为以 "index.html" 结尾
        if path.endswith("/"):
            path = path + "index.html"

        # 如果路径中包含 "../" 或 "./"，则抛出安全错误
        if "../" in path or "./" in path:
            raise SecurityError("Invalid path")

        # 使用正则表达式匹配路径中的特定格式
        match = re.match(r"/media/(?P<address>[A-Za-z0-9]+[A-Za-z0-9\._-]+)(?P<inner_path>/.*|$)", path)
        if match:
            # 将匹配的部分提取出来
            path_parts = match.groupdict()
            # 如果地址是域名，则解析域名
            if self.isDomain(path_parts["address"]):
                path_parts["address"] = self.resolveDomain(path_parts["address"])
            # 设置请求地址为原始地址（用于合并站点）
            path_parts["request_address"] = path_parts["address"]  # Original request address (for Merger sites)
            # 去除内部路径中的开头斜杠
            path_parts["inner_path"] = path_parts["inner_path"].lstrip("/")
            # 如果内部路径为空，则设置为 "index.html"
            if not path_parts["inner_path"]:
                path_parts["inner_path"] = "index.html"
            # 返回路径部分
            return path_parts
        else:
            # 如果路径不匹配，则返回 None
            return None

    # 为站点提供媒体服务
    # 为 UI 提供媒体服务
    # 处理 UiMedia 请求，根据路径返回对应的媒体文件
    def actionUiMedia(self, path):
        # 使用正则表达式匹配路径中的 uimedia 子路径
        match = re.match("/uimedia/(?P<inner_path>.*)", path)
        if match:  # Looks like a valid path
            # 构建文件路径
            file_path = "src/Ui/media/%s" % match.group("inner_path")
            # 获取允许访问的目录绝对路径
            allowed_dir = os.path.abspath("src/Ui/media")  # Only files within data/sitehash allowed
            # 检查文件路径是否在允许访问的目录内，以及是否包含非法路径
            if "../" in file_path or not os.path.dirname(os.path.abspath(file_path)).startswith(allowed_dir):
                # File not in allowed path
                return self.error403()  # 返回 403 错误
            else:
                if (config.debug or config.merge_media) and match.group("inner_path").startswith("all."):
                    # If debugging merge *.css to all.css and *.js to all.js
                    # 如果处于调试模式或合并媒体文件模式，并且路径以 "all." 开头，则合并对应的文件
                    from Debug import DebugMedia
                    DebugMedia.merge(file_path)
                return self.actionFile(file_path, header_length=False)  # Dont's send site to allow plugins append content

        else:  # Bad url
            return self.error400()  # 返回 400 错误，URL 不合法

    # 处理 SiteAdd 请求，添加站点
    def actionSiteAdd(self):
        # 读取 POST 数据并解析成字典
        post_data = self.env["wsgi.input"].read().decode()
        post = dict(urllib.parse.parse_qsl(post_data))
        # 检查 add_nonce 是否在服务器的 add_nonces 列表中
        if post["add_nonce"] not in self.server.add_nonces:
            return self.error403("Add nonce error.")  # 返回 403 错误，add_nonce 错误
        # 如果 add_nonce 正确，从 add_nonces 列表中移除该 add_nonce
        self.server.add_nonces.remove(post["add_nonce"])
        # 调用 SiteManager 添加站点
        SiteManager.site_manager.need(post["address"])
        # 重定向到指定的 URL
        return self.actionRedirect(post["url"])

    # 对响应进行编码处理
    @helper.encodeResponse
    # 在网站上添加提示，根据给定路径
    def actionSiteAddPrompt(self, path):
        # 解析路径，获取路径部分
        path_parts = self.parsePath(path)
        # 如果路径部分不存在或者地址不在站点管理器中
        if not path_parts or not self.server.site_manager.isAddress(path_parts["address"]):
            # 返回 404 错误
            return self.error404(path)
        
        # 发送头部信息，状态码为 200，内容类型为 text/html，禁用脚本
        self.sendHeader(200, "text/html", noscript=True)
        # 读取站点添加模板文件内容
        template = open("src/Ui/template/site_add.html").read()
        # 替换模板中的占位符为实际值
        template = template.replace("{url}", html.escape(self.env["PATH_INFO"]))
        template = template.replace("{address}", path_parts["address"])
        template = template.replace("{add_nonce}", self.getAddNonce())
        # 返回替换后的模板内容
        return template
    
    # 替换 HTML 变量
    def replaceHtmlVariables(self, block, path_parts):
        # 获取当前用户
        user = self.getCurrentUser()
        # 根据用户设置的主题，生成主题类名
        themeclass = "theme-%-6s" % re.sub("[^a-z]", "", user.settings.get("theme", "light"))
        block = block.replace(b"{themeclass}", themeclass.encode("utf8"))
        
        # 如果路径部分存在
        if path_parts:
            # 获取站点对象
            site = self.server.sites.get(path_parts.get("address"))
            # 如果站点拥有者为当前用户
            if site.settings["own"]:
                modified = int(time.time())
            else:
                modified = int(site.content_manager.contents["content.json"]["modified"])
            # 替换模板中的站点修改时间占位符为实际值
            block = block.replace(b"{site_modified}", str(modified).encode("utf8"))
        
        # 返回替换后的内容
        return block
    
    # 在 WebSocket 连接上向客户端流式传输文件
    # 调试最后一个错误
    def actionDebug(self):
        # 导入 main 模块
        import main
        # 获取 DebugHook 中的最后一个错误
        last_error = main.DebugHook.last_error
        # 如果存在最后一个错误
        if last_error:
            # 抛出最后一个错误
            raise last_error[0](last_error[1]).with_traceback(last_error[2])
        else:
            # 发送头部信息
            self.sendHeader()
            # 返回无错误提示
            return [b"No error! :)"]
    
    # 抛出错误以获取控制台
    # 定义一个方法，用于在控制台执行操作
    def actionConsole(self):
        # 导入 sys 模块
        import sys
        # 获取服务器的站点信息
        sites = self.server.sites
        # 获取 main 模块
        main = sys.modules["main"]

        # 定义一个函数，用于对代码进行性能测试
        def bench(code, times=100, init=None):
            # 获取服务器的站点信息
            sites = self.server.sites
            # 获取 main 模块
            main = sys.modules["main"]
            # 获取当前时间
            s = time.time()
            # 如果有初始化代码，则执行
            if init:
                eval(compile(init, '<string>', 'exec'), globals(), locals())
            # 循环执行指定次数的代码，并计算执行时间
            for _ in range(times):
                back = eval(code, globals(), locals())
            # 返回执行时间和结果
            return ["%s run: %.3fs" % (times, time.time() - s), back]
        # 抛出异常，显示控制台
        raise Exception("Here is your console")

    # - Tests -

    # 定义一个方法，用于测试流
    def actionTestStream(self):
        # 发送 HTTP 头部
        self.sendHeader()
        # 产生一个 1080 个空格的字符串，用于溢出浏览器缓冲区
        yield " " * 1080  # Overflow browser's buffer
        # 产生字符串 "He"
        yield "He"
        # 休眠 1 秒
        time.sleep(1)
        # 产生字符串 "llo!"
        yield "llo!"
        # 产生字符串 "Running websockets: %s" % len(self.server.websockets)
        # self.server.sendMessage("Hello!")

    # - Errors -

    # 发送 400 错误
    def error400(self, message=""):
        # 发送 400 错误的 HTTP 头部
        self.sendHeader(400, noscript=True)
        # 记录错误日志
        self.log.error("Error 400: %s" % message)
        # 返回格式化后的错误信息
        return self.formatError("Bad Request", message)

    # 禁止访问错误
    def error403(self, message="", details=True):
        # 发送 403 错误的 HTTP 头部
        self.sendHeader(403, noscript=True)
        # 记录警告日志
        self.log.warning("Error 403: %s" % message)
        # 返回格式化后的错误信息
        return self.formatError("Forbidden", message, details=details)

    # 发送 404 错误
    def error404(self, path=""):
        # 发送 404 错误的 HTTP 头部
        self.sendHeader(404, noscript=True)
        # 返回格式化后的错误信息
        return self.formatError("Not Found", path, details=False)

    # 服务器内部错误
    def error500(self, message=":("):
        # 发送 500 错误的 HTTP 头部
        self.sendHeader(500, noscript=True)
        # 记录错误日志
        self.log.error("Error 500: %s" % message)
        # 返回格式化后的错误信息
        return self.formatError("Server error", message)

    # 对响应进行编码
    @helper.encodeResponse
    # 定义一个方法，用于格式化错误信息
    def formatError(self, title, message, details=True):
        # 导入系统模块和协程模块
        import sys
        import gevent

        # 如果需要显示详细信息并且处于调试模式
        if details and config.debug:
            # 从环境变量中提取具有特定属性的键值对，排除包含"COOKIE"的键
            details = {key: val for key, val in list(self.env.items()) if hasattr(val, "endswith") and "COOKIE" not in key}
            # 添加版本信息到details字典中
            details["version_zeronet"] = "%s r%s" % (config.version, config.rev)
            details["version_python"] = sys.version
            details["version_gevent"] = gevent.__version__
            details["plugins"] = PluginManager.plugin_manager.plugin_names
            # 从配置参数中提取键值对，排除包含"password"的键
            arguments = {key: val for key, val in vars(config.arguments).items() if "password" not in key}
            details["arguments"] = arguments
            # 返回格式化后的错误信息，包括标题、消息和详细信息
            return """
                <style>
                * { font-family: Consolas, Monospace; color: #333 }
                pre { padding: 10px; background-color: #EEE }
                </style>
                <h1>%s</h1>
                <h2>%s</h3>
                <h3>Please <a href="https://github.com/HelloZeroNet/ZeroNet/issues" target="_top">report it</a> if you think this an error.</h3>
                <h4>Details:</h4>
                <pre>%s</pre>
            """ % (title, html.escape(message), html.escape(json.dumps(details, indent=4, sort_keys=True)))
        # 如果不需要显示详细信息或者不处于调试模式
        else:
            # 返回格式化后的错误信息，只包括标题和消息
            return """
                <style>
                * { font-family: Consolas, Monospace; color: #333; }
                code { font-family: Consolas, Monospace; background-color: #EEE }
                </style>
                <h1>%s</h1>
                <h2>%s</h3>
            """ % (title, html.escape(message))
```