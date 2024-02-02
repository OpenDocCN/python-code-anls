# `ZeroNet\src\main.py`

```py
# Included modules
# 导入所需的模块
import os
import sys
import stat
import time
import logging

# 存储启动时的错误信息
startup_errors = []

# 添加启动错误信息
def startupError(msg):
    startup_errors.append(msg)
    print("Startup error: %s" % msg)

# Third party modules
# 导入第三方模块
import gevent

# 如果 gevent 的主版本号小于等于 1，则进行以下操作
if gevent.version_info.major <= 1:  # Workaround for random crash when libuv used with threads
    try:
        # 如果 gevent 配置的事件循环不是 libev，则将其切换为 libev-cext
        if "libev" not in str(gevent.config.loop):
            gevent.config.loop = "libev-cext"
    except Exception as err:
        # 记录切换事件循环失败的错误信息
        startupError("Unable to switch gevent loop to libev: %s" % err)

# 打补丁，使得 gevent 能够在不阻塞主线程的情况下运行
import gevent.monkey
gevent.monkey.patch_all(thread=False, subprocess=False)

# 在主循环结束后是否更新并重启 zeronet
update_after_shutdown = False
# 在主循环结束后是否重启 zeronet
restart_after_shutdown = False

# Load config
# 加载配置文件
from Config import config
config.parse(silent=True)  # Plugins need to access the configuration
# 如果配置文件解析失败，则显示帮助信息并退出
if not config.arguments:
    config.parse()

# 如果数据目录不存在，则创建
if not os.path.isdir(config.data_dir):
    os.mkdir(config.data_dir)
    try:
        # 尝试修改数据目录的权限
        os.chmod(config.data_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
    except Exception as err:
        # 记录修改权限失败的错误信息
        startupError("Can't change permission of %s: %s" % (config.data_dir, err))

# 如果 sites.json 文件不存在，则创建
if not os.path.isfile("%s/sites.json" % config.data_dir):
    open("%s/sites.json" % config.data_dir, "w").write("{}")
# 如果 users.json 文件不存在，则创建
if not os.path.isfile("%s/users.json" % config.data_dir):
    open("%s/users.json" % config.data_dir, "w").write("{}")

# 如果配置动作为 "main"，则执行以下操作
if config.action == "main":
    # 导入 helper 模块
    from util import helper
    try:
        # 打开一个锁文件，并写入当前进程的 PID
        lock = helper.openLocked("%s/lock.pid" % config.data_dir, "w")
        lock.write("%s" % os.getpid())
    # 捕获阻塞 I/O 错误，并输出错误信息
    except BlockingIOError as err:
        startupError("Can't open lock file, your ZeroNet client is probably already running, exiting... (%s)" % err)
        # 如果配置中指定了打开浏览器，并且不等于 "False"
        if config.open_browser and config.open_browser != "False":
            # 打印正在打开的浏览器信息
            print("Opening browser: %s...", config.open_browser)
            # 导入 webbrowser 模块
            import webbrowser
            try:
                # 如果配置中指定了使用默认浏览器
                if config.open_browser == "default_browser":
                    # 获取默认浏览器
                    browser = webbrowser.get()
                else:
                    # 根据配置指定的浏览器类型获取浏览器
                    browser = webbrowser.get(config.open_browser)
                # 打开浏览器并访问指定的 URL
                browser.open("http://%s:%s/%s" % (
                    config.ui_ip if config.ui_ip != "*" else "127.0.0.1", config.ui_port, config.homepage
                ), new=2)
            # 捕获异常并输出错误信息
            except Exception as err:
                startupError("Error starting browser: %s" % err)
        # 退出程序
        sys.exit()
# 初始化日志记录
config.initLogging()

# 导入 Debug 模块中的 DebugHook 类
from Debug import DebugHook

# 加载插件管理器
from Plugin import PluginManager
PluginManager.plugin_manager.loadPlugins()
# 加载配置
config.loadPlugins()
# 解析配置，添加插件配置选项
config.parse()

# 记录当前配置信息
logging.debug("Config: %s" % config)

# 在特殊硬件上修改堆栈大小
if config.stack_size:
    import threading
    threading.stack_size(config.stack_size)

# 使用纯 Python 实现的 msgpack 以节省 CPU
if config.msgpack_purepython:
    os.environ["MSGPACK_PUREPYTHON"] = "True"

# 修复 Windows 上的控制台编码
if sys.platform.startswith("win"):
    import subprocess
    try:
        # 更改控制台编码为 utf8
        chcp_res = subprocess.check_output("chcp 65001", shell=True).decode(errors="ignore").strip()
        logging.debug("Changed console encoding to utf8: %s" % chcp_res)
    except Exception as err:
        logging.error("Error changing console encoding to utf8: %s" % err)

# Socket 修补
if config.proxy:
    # 导入 SocksProxy 类和 urllib.request 模块
    from util import SocksProxy
    import urllib.request
    # 打补丁，将套接字连接到 socks 代理
    logging.info("Patching sockets to socks proxy: %s" % config.proxy)
    if config.fileserver_ip == "*":
        config.fileserver_ip = '127.0.0.1'  # 仅允许本地连接
    config.disable_udp = True  # 目前不支持 UDP
    SocksProxy.monkeyPatch(*config.proxy.split(":"))
elif config.tor == "always":
    from util import SocksProxy
    import urllib.request
    # 打补丁，将套接字连接到 tor socks 代理
    logging.info("Patching sockets to tor socks proxy: %s" % config.tor_proxy)
    if config.fileserver_ip == "*":
        config.fileserver_ip = '127.0.0.1'  # 仅允许本地连接
    SocksProxy.monkeyPatch(*config.tor_proxy.split(":"))
    config.disable_udp = True
elif config.bind:
    bind = config.bind
    if ":" not in config.bind:
        bind += ":0"
    from util import helper
    # 打补丁，绑定套接字到指定地址和端口
    helper.socketBindMonkeyPatch(*bind.split(":"))

# -- Actions --

# 接受插件管理器中的插件
@PluginManager.acceptPlugins
class Actions(object):
    # 调用指定函数，并记录系统信息
    def call(self, function_name, kwargs):
        logging.info("Version: %s r%s, Python %s, Gevent: %s" % (config.version, config.rev, sys.version, gevent.__version__))

        # 获取指定函数的引用
        func = getattr(self, function_name, None)
        # 调用指定函数，并传入参数
        back = func(**kwargs)
        # 如果有返回结果，则打印出来
        if back:
            print(back)

    # 默认操作：启动 UiServer 和 FileServer
    def main(self):
        global ui_server, file_server
        # 导入 FileServer 和 UiServer 模块
        from File import FileServer
        from Ui import UiServer
        logging.info("Creating FileServer....")
        # 创建 FileServer 实例
        file_server = FileServer()
        logging.info("Creating UiServer....")
        # 创建 UiServer 实例
        ui_server = UiServer()
        # 将 FileServer 的 ui_server 属性设置为 UiServer 实例
        file_server.ui_server = ui_server

        # 遍历启动错误列表，并记录日志
        for startup_error in startup_errors:
            logging.error("Startup error: %s" % startup_error)

        # 移除旧的 SSL 证书
        logging.info("Removing old SSL certs...")
        from Crypt import CryptConnection
        CryptConnection.manager.removeCerts()

        # 启动服务器
        logging.info("Starting servers....")
        gevent.joinall([gevent.spawn(ui_server.start), gevent.spawn(file_server.start)])
        logging.info("All server stopped")

    # 网站命令
    # 对站点进行签名操作
    def siteSign(self, address, privatekey=None, inner_path="content.json", publish=False, remove_missing_optional=False):
        # 导入 Site 和 SiteManager 类
        from Site.Site import Site
        from Site import SiteManager
        # 导入 Debug 模块
        from Debug import Debug
        # 加载站点管理器
        SiteManager.site_manager.load()
        # 记录日志信息
        logging.info("Signing site: %s..." % address)
        # 创建站点对象
        site = Site(address, allow_create=False)

        # 如果没有定义私钥
        if not privatekey:
            # 导入 UserManager 类
            from User import UserManager
            # 获取当前用户
            user = UserManager.user_manager.get()
            if user:
                # 获取用户的站点数据
                site_data = user.getSiteData(address)
                privatekey = site_data.get("privatekey")
            else:
                privatekey = None
            if not privatekey:
                # 在 users.json 中未找到私钥，从控制台输入
                import getpass
                privatekey = getpass.getpass("Private key (input hidden):")
        try:
            # 对内容进行签名
            succ = site.content_manager.sign(
                inner_path=inner_path, privatekey=privatekey,
                update_changed_files=True, remove_missing_optional=remove_missing_optional
            )
        except Exception as err:
            # 记录签名错误日志
            logging.error("Sign error: %s" % Debug.formatException(err))
            succ = False
        # 如果签名成功且需要发布，则执行站点发布操作
        if succ and publish:
            self.sitePublish(address, inner_path=inner_path)
    # 对指定地址的站点进行验证
    def siteVerify(self, address):
        # 导入时间模块
        import time
        # 从Site.Site模块中导入Site类
        from Site.Site import Site
        # 从Site模块中导入SiteManager模块
        from Site import SiteManager
        # 加载站点管理器
        SiteManager.site_manager.load()

        # 记录当前时间
        s = time.time()
        # 记录日志，验证站点
        logging.info("Verifing site: %s..." % address)
        # 创建站点对象
        site = Site(address)
        # 初始化坏文件列表
        bad_files = []

        # 遍历站点内容管理器中的内容
        for content_inner_path in site.content_manager.contents:
            # 记录当前时间
            s = time.time()
            # 记录日志，验证内容的签名
            logging.info("Verifing %s signature..." % content_inner_path)
            # 初始化错误变量
            err = None
            try:
                # 验证文件的正确性
                file_correct = site.content_manager.verifyFile(
                    content_inner_path, site.storage.open(content_inner_path, "rb"), ignore_same=False
                )
            except Exception as err:
                # 如果出现异常，则文件不正确
                file_correct = False

            # 如果文件正确，则记录日志
            if file_correct is True:
                logging.info("[OK] %s (Done in %.3fs)" % (content_inner_path, time.time() - s))
            else:
                # 如果文件不正确，则记录错误日志，并等待用户输入
                logging.error("[ERROR] %s: invalid file: %s!" % (content_inner_path, err))
                input("Continue?")
                # 将错误文件添加到坏文件列表中
                bad_files += content_inner_path

        # 记录日志，验证站点文件
        logging.info("Verifying site files...")
        # 将站点存储中的坏文件添加到坏文件列表中
        bad_files += site.storage.verifyFiles()["bad_files"]
        # 如果没有坏文件，则记录日志
        if not bad_files:
            logging.info("[OK] All file sha512sum matches! (%.3fs)" % (time.time() - s))
        else:
            # 如果有坏文件，则记录错误日志
            logging.error("[ERROR] Error during verifying site files!")

    # 重建指定地址站点的数据库
    def dbRebuild(self, address):
        # 从Site.Site模块中导入Site类
        from Site.Site import Site
        # 从Site模块中导入SiteManager模块
        from Site import SiteManager
        # 加载站点管理器
        SiteManager.site_manager.load()

        # 记录日志，重建站点的SQL缓存
        logging.info("Rebuilding site sql cache: %s..." % address)
        # 获取指定地址的站点对象
        site = SiteManager.site_manager.get(address)
        # 记录当前时间
        s = time.time()
        try:
            # 重建站点存储的数据库
            site.storage.rebuildDb()
            # 记录日志，重建完成所花费的时间
            logging.info("Done in %.3fs" % (time.time() - s))
        except Exception as err:
            # 如果出现异常，则记录错误日志
            logging.error(err)
    # 执行数据库查询操作
    def dbQuery(self, address, query):
        # 从Site.Site模块中导入Site类
        from Site.Site import Site
        # 从Site模块中导入SiteManager模块
        from Site import SiteManager
        # 加载SiteManager中的站点信息
        SiteManager.site_manager.load()

        # 导入json模块
        import json
        # 创建Site对象
        site = Site(address)
        # 初始化结果列表
        result = []
        # 遍历查询结果，将每行数据转换为字典并添加到结果列表中
        for row in site.storage.query(query):
            result.append(dict(row))
        # 将结果列表以缩进格式打印为JSON格式
        print(json.dumps(result, indent=4))

    # 发布站点信息
    def siteAnnounce(self, address):
        # 从Site.Site模块中导入Site类
        from Site.Site import Site
        # 从Site模块中导入SiteManager模块
        from Site import SiteManager
        # 加载SiteManager中的站点信息
        SiteManager.site_manager.load()

        # 导入日志模块
        logging.info("Opening a simple connection server")
        # 声明全局变量file_server
        global file_server
        # 从File模块中导入FileServer类
        from File import FileServer
        # 创建FileServer对象并启动
        file_server = FileServer("127.0.0.1", 1234)
        file_server.start()

        # 记录日志，宣布站点信息到追踪器
        logging.info("Announcing site %s to tracker..." % address)
        # 创建Site对象
        site = Site(address)

        # 记录当前时间
        s = time.time()
        # 发布站点信息
        site.announce()
        # 打印响应时间
        print("Response time: %.3fs" % (time.time() - s))
        # 打印站点的对等节点信息
        print(site.peers)

    # 下载站点内容
    def siteDownload(self, address):
        # 从Site.Site模块中导入Site类
        from Site.Site import Site
        # 从Site模块中导入SiteManager模块
        from Site import SiteManager
        # 加载SiteManager中的站点信息
        SiteManager.site_manager.load()

        # 导入日志模块
        logging.info("Opening a simple connection server")
        # 声明全局变量file_server
        global file_server
        # 从File模块中导入FileServer类
        from File import FileServer
        # 创建FileServer对象并启动，使用协程方式
        file_server = FileServer("127.0.0.1", 1234)
        file_server_thread = gevent.spawn(file_server.start, check_sites=False)

        # 创建Site对象
        site = Site(address)

        # 创建异步结果对象
        on_completed = gevent.event.AsyncResult()

        # 定义完成时的回调函数
        def onComplete(evt):
            evt.set(True)

        # 注册完成时的回调函数
        site.onComplete.once(lambda: onComplete(on_completed))
        # 打印提示信息
        print("Announcing...")
        # 发布站点信息
        site.announce()

        # 记录当前时间
        s = time.time()
        # 打印提示信息
        print("Downloading...")
        # 下载站点内容，检查是否有修改
        site.downloadContent("content.json", check_modifications=True)

        # 打印下载所用时间
        print("Downloaded in %.3fs" % (time.time()-s))
    # 定义一个方法，用于检查站点是否需要文件
    def siteNeedFile(self, address, inner_path):
        # 导入Site和SiteManager模块
        from Site.Site import Site
        from Site import SiteManager
        # 加载站点管理器
        SiteManager.site_manager.load()

        # 定义一个检查器函数，用于定时检查站点
        def checker():
            while 1:
                s = time.time()
                time.sleep(1)
                print("Switch time:", time.time() - s)
        # 使用gevent创建一个协程来执行检查器函数
        gevent.spawn(checker)

        # 记录信息到日志
        logging.info("Opening a simple connection server")
        # 全局变量file_server，导入FileServer模块并创建一个FileServer对象
        global file_server
        from File import FileServer
        file_server = FileServer("127.0.0.1", 1234)
        # 使用gevent创建一个协程来启动文件服务器
        file_server_thread = gevent.spawn(file_server.start, check_sites=False)

        # 创建一个站点对象
        site = Site(address)
        # 发布站点信息
        site.announce()
        # 打印站点是否需要文件的结果
        print(site.needFile(inner_path, update=True))

    # 定义一个方法，用于执行站点命令
    def siteCmd(self, address, cmd, parameters):
        # 导入json和SiteManager模块
        import json
        from Site import SiteManager

        # 获取指定地址的站点对象
        site = SiteManager.site_manager.get(address)

        # 如果站点不存在，则记录错误信息并返回空值
        if not site:
            logging.error("Site not found: %s" % address)
            return None

        # 获取站点的websocket连接
        ws = self.getWebsocket(site)

        # 发送命令和参数到站点的websocket连接
        ws.send(json.dumps({"cmd": cmd, "params": parameters, "id": 1}))
        # 接收站点返回的结果
        res_raw = ws.recv()

        try:
            # 尝试解析返回结果
            res = json.loads(res_raw)
        except Exception as err:
            # 如果解析失败，则返回错误信息和原始结果
            return {"error": "Invalid result: %s" % err, "res_raw": res_raw}

        # 如果返回结果中包含"result"字段，则返回该字段的值，否则返回整个结果
        if "result" in res:
            return res["result"]
        else:
            return res

    # 定义一个方法，用于获取站点的websocket连接
    def getWebsocket(self, site):
        # 导入websocket模块
        import websocket

        # 构建websocket连接地址
        ws_address = "ws://%s:%s/Websocket?wrapper_key=%s" % (config.ui_ip, config.ui_port, site.settings["wrapper_key"])
        # 记录信息到日志
        logging.info("Connecting to %s" % ws_address)
        # 创建websocket连接并返回
        ws = websocket.create_connection(ws_address)
        return ws
    # 发布网站到指定地址
    def sitePublish(self, address, peer_ip=None, peer_port=15441, inner_path="content.json"):
        # 引入全局变量 file_server
        global file_server
        # 引入 Site 和 SiteManager 模块
        from Site.Site import Site
        from Site import SiteManager
        # 引入 FileServer 模块，用于处理传入的文件请求
        from File import FileServer  
        # 引入 Peer 模块
        from Peer import Peer
        # 创建 FileServer 实例
        file_server = FileServer()
        # 获取指定地址的网站
        site = SiteManager.site_manager.get(address)
        # 记录日志，加载网站
        logging.info("Loading site...")
        # 设置网站的 serving 属性为 True，即使网站被禁用也要提供服务
        site.settings["serving"] = True  

        try:
            # 获取网站的 WebSocket 连接
            ws = self.getWebsocket(site)
            # 记录日志，发送 siteReload 命令
            logging.info("Sending siteReload")
            self.siteCmd(address, "siteReload", inner_path)

            # 记录日志，发送 sitePublish 命令
            logging.info("Sending sitePublish")
            self.siteCmd(address, "sitePublish", {"inner_path": inner_path, "sign": False})
            logging.info("Done.")

        except Exception as err:
            # 记录日志，无法连接到本地 WebSocket 客户端
            logging.info("Can't connect to local websocket client: %s" % err)
            logging.info("Creating FileServer....")
            # 创建 FileServer 线程
            file_server_thread = gevent.spawn(file_server.start, check_sites=False)  # 不检查每个网站的完整性
            time.sleep(0.001)

            # 启动文件服务器
            file_server.portCheck()
            if peer_ip:  # 如果指定了 peer_ip，则通告该 IP
                site.addPeer(peer_ip, peer_port)
            else:  # 否则，从追踪器获取 peers
                logging.info("Gathering peers from tracker")
                site.announce()  # 收集 peers
            # 发布网站到 peers
            published = site.publish(5, inner_path)  
            if published > 0:
                time.sleep(3)
                logging.info("Serving files (max 60s)...")
                # 等待所有协程完成，最长等待时间为 60 秒
                gevent.joinall([file_server_thread], timeout=60)
                logging.info("Done.")
            else:
                # 记录日志，未找到 peers，sitePublish 命令只在已有访问者为您的网站提供服务时才有效
                logging.info("No peers found, sitePublish command only works if you already have visitors serving your site")

    # Crypto commands
    # 将私钥加密为地址
    def cryptPrivatekeyToAddress(self, privatekey=None):
        # 导入 CryptBitcoin 模块
        from Crypt import CryptBitcoin
        # 如果没有传入私钥，则从用户输入中获取私钥
        if not privatekey:  # If no privatekey in args then ask it now
            import getpass
            privatekey = getpass.getpass("Private key (input hidden):")

        # 打印私钥对应的地址
        print(CryptBitcoin.privatekeyToAddress(privatekey))

    # 对消息进行签名
    def cryptSign(self, message, privatekey):
        # 导入 CryptBitcoin 模块
        from Crypt import CryptBitcoin
        # 打印消息的签名
        print(CryptBitcoin.sign(message, privatekey))

    # 验证签名
    def cryptVerify(self, message, sign, address):
        # 导入 CryptBitcoin 模块
        from Crypt import CryptBitcoin
        # 打印消息、地址和签名的验证结果
        print(CryptBitcoin.verify(message, address, sign))

    # 获取私钥
    def cryptGetPrivatekey(self, master_seed, site_address_index=None):
        # 导入 CryptBitcoin 模块
        from Crypt import CryptBitcoin
        # 如果主密钥长度不为64，则打印错误信息并返回 False
        if len(master_seed) != 64:
            logging.error("Error: Invalid master seed length: %s (required: 64)" % len(master_seed))
            return False
        # 使用主密钥和地址索引生成私钥
        privatekey = CryptBitcoin.hdPrivatekey(master_seed, site_address_index)
        # 打印生成的私钥
        print("Requested private key: %s" % privatekey)

    # Peer
    # 定义一个方法用于向指定的对等方 IP 发送 ping 请求，peer_port 默认为 15441
    def peerPing(self, peer_ip, peer_port=None):
        # 如果未指定 peer_port，则默认使用 15441
        if not peer_port:
            peer_port = 15441
        # 记录信息到日志
        logging.info("Opening a simple connection server")
        # 引入全局变量 file_server
        global file_server
        # 从 Connection 模块中引入 ConnectionServer 类
        from Connection import ConnectionServer
        # 创建 ConnectionServer 实例，监听本地 IP 地址和端口 1234
        file_server = ConnectionServer("127.0.0.1", 1234)
        # 启动连接服务器，不检查连接
        file_server.start(check_connections=False)
        # 从 Crypt 模块中引入 CryptConnection 类
        from Crypt import CryptConnection
        # 加载证书
        CryptConnection.manager.loadCerts()

        # 从 Peer 模块中引入 Peer 类
        from Peer import Peer
        # 记录信息到日志
        logging.info("Pinging 5 times peer: %s:%s..." % (peer_ip, int(peer_port)))
        # 记录当前时间
        s = time.time()
        # 创建 Peer 实例，指定对等方 IP 和端口，进行连接
        peer = Peer(peer_ip, peer_port)
        peer.connect()

        # 如果无法连接到对等方，则打印错误信息并返回 False
        if not peer.connection:
            print("Error: Can't connect to peer (connection error: %s)" % peer.connection_error)
            return False
        # 如果连接对象中存在 shared_ciphers 属性，则打印共享密码
        if "shared_ciphers" in dir(peer.connection.sock):
            print("Shared ciphers:", peer.connection.sock.shared_ciphers())
        # 如果连接对象中存在 cipher 属性，则打印密码
        if "cipher" in dir(peer.connection.sock):
            print("Cipher:", peer.connection.sock.cipher()[0])
        # 如果连接对象中存在 version 属性，则打印 TLS 版本
        if "version" in dir(peer.connection.sock):
            print("TLS version:", peer.connection.sock.version())
        # 打印连接时间和连接错误信息
        print("Connection time: %.3fs  (connection error: %s)" % (time.time() - s, peer.connection_error))

        # 发送 5 次 ping 请求，打印响应时间，每次间隔 1 秒
        for i in range(5):
            ping_delay = peer.ping()
            print("Response time: %.3fs" % ping_delay)
            time.sleep(1)
        # 移除对等方连接
        peer.remove()
        # 打印重新连接测试信息
        print("Reconnect test...")
        # 重新创建 Peer 实例，再次发送 5 次 ping 请求，打印响应时间，每次间隔 1 秒
        peer = Peer(peer_ip, peer_port)
        for i in range(5):
            ping_delay = peer.ping()
            print("Response time: %.3fs" % ping_delay)
            time.sleep(1)
    # 定义一个方法，用于从对等节点获取文件
    def peerGetFile(self, peer_ip, peer_port, site, filename, benchmark=False):
        # 记录日志信息
        logging.info("Opening a simple connection server")
        # 导入 Connection 模块中的 ConnectionServer 类
        global file_server
        from Connection import ConnectionServer
        # 创建 ConnectionServer 实例并启动
        file_server = ConnectionServer("127.0.0.1", 1234)
        file_server.start(check_connections=False)
        # 导入 Crypt 模块中的 CryptConnection 类
        from Crypt import CryptConnection
        # 加载证书
        CryptConnection.manager.loadCerts()

        # 导入 Peer 模块中的 Peer 类
        from Peer import Peer
        # 记录日志信息
        logging.info("Getting %s/%s from peer: %s:%s..." % (site, filename, peer_ip, peer_port))
        # 创建 Peer 实例
        peer = Peer(peer_ip, peer_port)
        # 记录开始时间
        s = time.time()
        # 如果需要进行基准测试
        if benchmark:
            # 进行 10 次文件获取操作
            for i in range(10):
                peer.getFile(site, filename),
            # 打印响应时间
            print("Response time: %.3fs" % (time.time() - s))
            # 检查内存
            input("Check memory")
        else:
            # 打印获取到的文件内容
            print(peer.getFile(site, filename).read())

    # 定义一个方法，用于向对等节点发送命令
    def peerCmd(self, peer_ip, peer_port, cmd, parameters):
        # 记录日志信息
        logging.info("Opening a simple connection server")
        # 全局变量 file_server
        global file_server
        # 导入 Connection 模块中的 ConnectionServer 类
        from Connection import ConnectionServer
        # 创建 ConnectionServer 实例并启动
        file_server = ConnectionServer()
        file_server.start(check_connections=False)
        # 导入 Crypt 模块中的 CryptConnection 类
        from Crypt import CryptConnection
        # 加载证书
        CryptConnection.manager.loadCerts()

        # 导入 Peer 模块中的 Peer 类
        from Peer import Peer
        # 创建 Peer 实例
        peer = Peer(peer_ip, peer_port)

        # 导入 json 模块
        import json
        # 如果存在参数，则将参数转换为 JSON 格式
        if parameters:
            parameters = json.loads(parameters.replace("'", '"'))
        else:
            parameters = {}
        try:
            # 向对等节点发送请求并获取响应
            res = peer.request(cmd, parameters)
            # 打印响应内容
            print(json.dumps(res, indent=2, ensure_ascii=False))
        except Exception as err:
            # 打印异常信息
            print("Unknown response (%s): %s" % (err, res))

    # 定义一个方法，用于获取配置信息
    def getConfig(self):
        # 导入 json 模块
        import json
        # 打印服务器信息的 JSON 格式
        print(json.dumps(config.getServerInfo(), indent=2, ensure_ascii=False))
    # 定义一个测试方法，接受测试名称、位置参数和关键字参数
    def test(self, test_name, *args, **kwargs):
        # 导入 types 模块
        import types
        # 定义一个内部函数，用于将测试方法名转换为测试名称
        def funcToName(func_name):
            test_name = func_name.replace("test", "")
            return test_name[0].lower() + test_name[1:]

        # 获取所有以 "test" 开头的方法名，并将其转换为测试名称
        test_names = [funcToName(name) for name in dir(self) if name.startswith("test") and name != "test"]
        # 如果没有指定测试名称，则列出所有可能的测试
        if not test_name:
            print("\nNo test specified, possible tests:")
            for test_name in test_names:
                func_name = "test" + test_name[0].upper() + test_name[1:]
                func = getattr(self, func_name)
                # 如果测试方法有文档字符串，则打印测试名称和文档字符串
                if func.__doc__:
                    print("- %s: %s" % (test_name, func.__doc__.strip()))
                else:
                    print("- %s" % test_name)
            return None

        # 运行测试
        func_name = "test" + test_name[0].upper() + test_name[1:]
        if hasattr(self, func_name):
            func = getattr(self, func_name)
            print("- Running test: %s" % test_name, end="")
            s = time.time()
            ret = func(*args, **kwargs)
            # 如果返回值是生成器类型，则逐个打印生成器的值
            if type(ret) is types.GeneratorType:
                for progress in ret:
                    print(progress, end="")
                    sys.stdout.flush()
            print("\n* Test %s done in %.3fs" % (test_name, time.time() - s))
        else:
            # 如果指定的测试名称不存在，则打印错误信息
            print("Unknown test: %r (choose from: %s)" % (
                test_name, test_names
            ))
# 创建一个名为actions的Actions对象实例
actions = Actions()
# 当运行zeronet.py时，从这里开始

# 定义一个名为start的函数
def start():
    # 调用config.getActionArguments()函数，并将返回的结果赋值给action_kwargs变量
    action_kwargs = config.getActionArguments()
    # 调用actions对象的call方法，传入config.action和action_kwargs作为参数
    actions.call(config.action, action_kwargs)
```