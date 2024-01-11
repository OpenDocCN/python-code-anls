# `ZeroNet\src\Test\conftest.py`

```
# 导入必要的模块
import os
import sys
import urllib.request
import time
import logging
import json
import shutil
import gc
import datetime
import atexit
import threading
import socket

# 导入 pytest 和 mock 模块
import pytest
import mock

# 导入 gevent 模块并进行配置
import gevent
if "libev" not in str(gevent.config.loop):
    # 解决 libuv 与线程一起使用时的随机崩溃问题
    gevent.config.loop = "libev-cext"

# 注册 atexit 函数
atexit_register = atexit.register
atexit.register = lambda func: ""  # 避免注册关闭函数以避免退出时的 IO 错误

# 添加 pytest 选项
def pytest_addoption(parser):
    parser.addoption("--slow", action='store_true', default=False, help="Also run slow tests")

# 修改 pytest 集合项
def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        # 在命令行中使用 --slow 选项：不跳过慢速测试
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

# 配置
if sys.platform == "win32":
    CHROMEDRIVER_PATH = "tools/chrome/chromedriver.exe"
else:
    CHROMEDRIVER_PATH = "chromedriver"
SITE_URL = "http://127.0.0.1:43110"

TEST_DATA_PATH = 'src/Test/testdata'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../lib"))  # 外部模块目录
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))  # 相对于 src 目录的导入

# 导入 Config 模块并配置参数
from Config import config
config.argv = ["none"]  # 不传递任何参数给配置解析器
config.parse(silent=True, parse_config=False)  # 插件需要访问配置
config.action = "test"

# 加载插件
from Plugin import PluginManager

config.data_dir = TEST_DATA_PATH  # 用于单元测试的测试数据
config.debug = True

os.chdir(os.path.abspath(os.path.dirname(__file__) + "/../.."))  # 设置工作目录

# 加载所有插件并断言是否全部加载成功
all_loaded = PluginManager.plugin_manager.loadPlugins()
assert all_loaded, "Not all plugin loaded successfully"
# 加载插件
config.loadPlugins()
# 再次解析配置，以添加插件配置选项
config.parse(parse_config=False)
# 设置动作为测试
config.action = "test"
# 开启调试模式
config.debug = True
# 开启调试套接字
config.debug_socket = True
# 设置为详细模式
config.verbose = True
# 禁用 Tor 客户端
config.tor = "disable"
# 清空追踪器列表
config.trackers = []
# 设置数据目录为测试数据路径
config.data_dir = TEST_DATA_PATH
# 如果环境变量中存在 ZERONET_LOG_DIR，则设置日志目录为该环境变量的值
if "ZERONET_LOG_DIR" in os.environ:
    config.log_dir = os.environ["ZERONET_LOG_DIR"]
# 初始化日志记录，禁用控制台日志
config.initLogging(console_logging=False)

# 设置自定义格式化程序，使用相对时间格式
time_start = time.time()
# 定义时间过滤器类
class TimeFilter(logging.Filter):
    def __init__(self, *args, **kwargs):
        self.time_last = time.time()
        self.main_thread_id = threading.current_thread().ident
        super().__init__(*args, **kwargs)

    def filter(self, record):
        # 如果当前线程不是主线程，则设置线程标记和标题
        if threading.current_thread().ident != self.main_thread_id:
            record.thread_marker = "T"
            record.thread_title = "(Thread#%s)" % self.main_thread_id
        else:
            record.thread_marker = " "
            record.thread_title = ""

        # 计算距上次日志的时间间隔
        since_last = time.time() - self.time_last
        if since_last > 0.1:
            line_marker = "!"
        elif since_last > 0.02:
            line_marker = "*"
        elif since_last > 0.01:
            line_marker = "-"
        else:
            line_marker = " "

        # 计算距开始的时间
        since_start = time.time() - time_start
        record.since_start = "%s%.3fs" % (line_marker, since_start)

        # 更新上次日志时间
        self.time_last = time.time()
        return True

# 获取日志记录器
log = logging.getLogger()
# 设置日志格式
fmt = logging.Formatter(fmt='%(since_start)s %(thread_marker)s %(levelname)-8s %(name)s %(message)s %(thread_title)s')
# 为所有日志处理器添加时间过滤器
[hndl.addFilter(TimeFilter()) for hndl in log.handlers]
# 为所有日志处理器设置格式化程序
[hndl.setFormatter(fmt) for hndl in log.handlers]

# 导入 Site 和 SiteManager 模块
from Site.Site import Site
from Site import SiteManager
# 导入 UserManager 模块
from User import UserManager
# 从 File 模块中导入 FileServer 类
from File import FileServer
# 从 Connection 模块中导入 ConnectionServer 类
from Connection import ConnectionServer
# 从 Crypt 模块中导入 CryptConnection 和 CryptBitcoin 类
from Crypt import CryptConnection
from Crypt import CryptBitcoin
# 从 Ui 模块中导入 UiWebsocket 类
from Ui import UiWebsocket
# 从 Tor 模块中导入 TorManager 类
from Tor import TorManager
# 从 Content 模块中导入 ContentDb 类
from Content import ContentDb
# 从 util 模块中导入 RateLimit 类
from util import RateLimit
# 从 Db 模块中导入 Db 类
from Db import Db
# 从 Debug 模块中导入 Debug 类
from Debug import Debug

# 将 Debug.Notify 方法添加到 gevent 的 NOT_ERROR 事件中
gevent.get_hub().NOT_ERROR += (Debug.Notify,)

# 定义 cleanup 函数
def cleanup():
    # 关闭所有数据库连接
    Db.dbCloseAll()
    # 遍历数据目录和临时数据目录
    for dir_path in [config.data_dir, config.data_dir + "-temp"]:
        # 如果目录存在
        if os.path.isdir(dir_path):
            # 遍历目录下的文件
            for file_name in os.listdir(dir_path):
                # 获取文件扩展名
                ext = file_name.rsplit(".", 1)[-1]
                # 如果扩展名不在指定的列表中，则跳过
                if ext not in ["csr", "pem", "srl", "db", "json", "tmp"]:
                    continue
                # 构建文件路径
                file_path = dir_path + "/" + file_name
                # 如果是文件，则删除
                if os.path.isfile(file_path):
                    os.unlink(file_path)

# 在程序退出时执行 cleanup 函数
atexit_register(cleanup)

# 定义 resetSettings 会话级别的 fixture
@pytest.fixture(scope="session")
def resetSettings(request):
    # 清空 sites.json 文件
    open("%s/sites.json" % config.data_dir, "w").write("{}")
    # 清空 filters.json 文件
    open("%s/filters.json" % config.data_dir, "w").write("{}")
    # 写入默认的 users.json 文件内容
    open("%s/users.json" % config.data_dir, "w").write("""
        {
            "15E5rhcAUD69WbiYsYARh4YHJ4sLm2JEyc": {
                "certs": {},
                "master_seed": "024bceac1105483d66585d8a60eaf20aa8c3254b0f266e0d626ddb6114e2949a",
                "sites": {}
            }
        }
    """)

# 定义 resetTempSettings 会话级别的 fixture
@pytest.fixture(scope="session")
def resetTempSettings(request):
    # 构建临时数据目录路径
    data_dir_temp = config.data_dir + "-temp"
    # 如果临时数据目录不存在，则创建
    if not os.path.isdir(data_dir_temp):
        os.mkdir(data_dir_temp)
    # 清空临时数据目录下的 sites.json 文件
    open("%s/sites.json" % data_dir_temp, "w").write("{}")
    # 清空临时数据目录下的 filters.json 文件
    open("%s/filters.json" % data_dir_temp, "w").write("{}")
    # 写入默认的临时数据目录下的 users.json 文件内容
    open("%s/users.json" % data_dir_temp, "w").write("""
        {
            "15E5rhcAUD69WbiYsYARh4YHJ4sLm2JEyc": {
                "certs": {},
                "master_seed": "024bceac1105483d66585d8a60eaf20aa8c3254b0f266e0d626ddb6114e2949a",
                "sites": {}
            }
        }
    """)
    # 定义一个清理函数，用于删除临时数据目录中的特定文件
    def cleanup():
        # 删除临时数据目录中的 sites.json 文件
        os.unlink("%s/sites.json" % data_dir_temp)
        # 删除临时数据目录中的 users.json 文件
        os.unlink("%s/users.json" % data_dir_temp)
        # 删除临时数据目录中的 filters.json 文件
        os.unlink("%s/filters.json" % data_dir_temp)
    
    # 在测试用例执行完成后执行 cleanup 函数，清理临时数据目录中的文件
    request.addfinalizer(cleanup)
# 使用 pytest 的 fixture 装饰器定义一个名为 site 的测试环境
@pytest.fixture()
def site(request):
    # 获取当前所有的 greenlet 对象，用于后续清理
    threads_before = [obj for obj in gc.get_objects() if isinstance(obj, gevent.Greenlet)]
    # 重置 RateLimit 的队列和调用记录
    RateLimit.queue_db = {}
    RateLimit.called_db = {}

    # 创建一个名为 "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT" 的 Site 对象
    site = Site("1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT")

    # 确保原始数据一直存在，避免删除所有内容
    assert "1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT" in site.storage.getPath("")  
    # 递归删除当前路径下的所有文件和文件夹
    shutil.rmtree(site.storage.getPath(""), True)
    # 复制原始数据到当前路径
    shutil.copytree(site.storage.getPath("") + "-original", site.storage.getPath(""))

    # 将当前 Site 对象添加到 SiteManager 中
    SiteManager.site_manager.get("1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT")
    # 使用 mock.MagicMock 创建一个虚拟的 announce 方法，返回 True
    site.announce = mock.MagicMock(return_value=True)  

    # 定义清理函数
    def cleanup():
        # 删除当前 Site 对象
        site.delete()
        # 关闭内容管理器的数据库连接
        site.content_manager.contents.db.close("Test cleanup")
        # 停止内容管理器的定时检查任务
        site.content_manager.contents.db.timer_check_optional.kill()
        # 清空 SiteManager 中的所有 Site 对象
        SiteManager.site_manager.sites.clear()
        # 删除指定路径下的数据库文件
        db_path = "%s/content.db" % config.data_dir
        os.unlink(db_path)
        # 从 ContentDb 的字典中删除指定路径的数据库对象
        del ContentDb.content_dbs[db_path]
        # 杀死所有非线程创建前的 greenlet 对象
        gevent.killall([obj for obj in gc.get_objects() if isinstance(obj, gevent.Greenlet) and obj not in threads_before])
    # 将清理函数添加到测试环境的最终化操作中
    request.addfinalizer(cleanup)

    # 停止当前 Site 对象的所有 greenlet 对象
    site.greenlet_manager.stopGreenlets()
    # 创建一个新的 Site 对象，用于加载 content.json 文件
    site = Site("1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT")  
    # 如果 SiteManager 中没有 Site 对象，则创建一个空的字典
    if not SiteManager.site_manager.sites:
        SiteManager.site_manager.sites = {}
    # 将当前 Site 对象添加到 SiteManager 中
    SiteManager.site_manager.sites["1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT"] = site
    # 设置当前 Site 对象的 serving 属性为 True
    site.settings["serving"] = True
    # 返回当前 Site 对象
    return site

# 使用 pytest 的 fixture 装饰器定义一个名为 site_temp 的测试环境
@pytest.fixture()
def site_temp(request):
    # 获取当前所有的 greenlet 对象，用于后续清理
    threads_before = [obj for obj in gc.get_objects() if isinstance(obj, gevent.Greenlet)]
    # 使用 mock.patch 方法修改配置文件中的 data_dir，添加 "-temp" 后缀
    with mock.patch("Config.config.data_dir", config.data_dir + "-temp"):
        # 创建一个临时的站点对象
        site_temp = Site("1TeSTvb4w2PWE81S2rEELgmX2GCCExQGT")
        # 设置临时站点的 serving 属性为 True
        site_temp.settings["serving"] = True
        # 用 mock.MagicMock 方法模拟 announce 方法的返回值为 True
        site_temp.announce = mock.MagicMock(return_value=True)  # Don't try to find peers from the net
    
    # 定义一个清理函数
    def cleanup():
        # 删除临时站点
        site_temp.delete()
        # 关闭临时站点的内容管理器中的数据库连接
        site_temp.content_manager.contents.db.close("Test cleanup")
        # 停止临时站点内容管理器中的定时器
        site_temp.content_manager.contents.db.timer_check_optional.kill()
        # 获取临时站点数据库的路径，并删除该文件
        db_path = "%s-temp/content.db" % config.data_dir
        os.unlink(db_path)
        # 从 ContentDb.content_dbs 字典中删除临时站点数据库的引用
        del ContentDb.content_dbs[db_path]
        # 终止所有未完成的协程
        gevent.killall([obj for obj in gc.get_objects() if isinstance(obj, gevent.Greenlet) and obj not in threads_before])
    
    # 注册清理函数，用于在测试结束时执行清理操作
    request.addfinalizer(cleanup)
    # 为临时站点设置日志记录器
    site_temp.log = logging.getLogger("Temp:%s" % site_temp.address_short)
    # 返回临时站点对象
    return site_temp
# 定义一个会话级别的 fixture，用于获取用户信息
@pytest.fixture(scope="session")
def user():
    # 从 UserManager 中获取用户信息
    user = UserManager.user_manager.get()
    # 如果用户信息不存在，则创建一个新用户
    if not user:
        user = UserManager.user_manager.create()
    # 重置用户数据
    user.sites = {}
    # 返回用户信息
    return user


# 定义一个会话级别的 fixture，用于获取浏览器对象
@pytest.fixture(scope="session")
def browser(request):
    try:
        # 导入 selenium 的 webdriver 模块
        from selenium import webdriver
        # 打印信息，表示正在启动 chromedriver
        print("Starting chromedriver...")
        # 设置 chromedriver 的参数
        options = webdriver.chrome.options.Options()
        options.add_argument("--headless")
        options.add_argument("--window-size=1920x1080")
        options.add_argument("--log-level=1")
        # 创建 Chrome 浏览器对象
        browser = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH, service_log_path=os.path.devnull, options=options)

        # 定义一个关闭浏览器的函数，并注册为测试结束时的清理函数
        def quit():
            browser.quit()
        request.addfinalizer(quit)
    except Exception as err:
        # 如果出现异常，则抛出 pytest.skip 异常
        raise pytest.skip("Test requires selenium + chromedriver: %s" % err)
    # 返回浏览器对象
    return browser


# 定义一个会话级别的 fixture，用于获取站点 URL
@pytest.fixture(scope="session")
def site_url():
    try:
        # 打开站点 URL 并读取内容
        urllib.request.urlopen(SITE_URL).read()
    except Exception as err:
        # 如果出现异常，则抛出 pytest.skip 异常
        raise pytest.skip("Test requires zeronet client running: %s" % err)
    # 返回站点 URL
    return SITE_URL


# 定义一个参数化的 fixture，用于获取文件服务器
@pytest.fixture(params=['ipv4', 'ipv6'])
def file_server(request):
    # 根据参数值选择不同的 fixture
    if request.param == "ipv4":
        return request.getfixturevalue("file_server4")
    else:
        return request.getfixturevalue("file_server6")


# 定义一个普通 fixture，用于获取 IPv4 文件服务器
@pytest.fixture
def file_server4(request):
    # 等待一段时间
    time.sleep(0.1)
    # 创建一个 IPv4 文件服务器对象
    file_server = FileServer("127.0.0.1", 1544)
    # 设置外部 IP 地址为假的外部 IP
    file_server.ip_external = "1.2.3.4"

    # 定义一个监听函数，并使用 gevent 异步执行
    def listen():
        ConnectionServer.start(file_server)
        ConnectionServer.listen(file_server)

    gevent.spawn(listen)
    # 等待端口打开
    for retry in range(10):
        time.sleep(0.1)  # 等待端口打开
        try:
            # 尝试连接文件服务器
            conn = file_server.getConnection("127.0.0.1", 1544)
            conn.close()
            break
        except Exception as err:
            # 打印错误信息
            print("FileServer6 startup error", Debug.formatException(err))
    # 断言文件服务器正在运行
    assert file_server.running
    file_server.ip_incoming = {}  # 重置防洪保护，清空IP地址的入站连接记录

    def stop():
        file_server.stop()  # 停止文件服务器
    request.addfinalizer(stop)  # 在测试用例执行完成后执行stop函数，用于停止文件服务器
    return file_server  # 返回文件服务器对象
# 定义一个名为file_server6的fixture，用于测试IPv6支持的文件服务器
@pytest.fixture
def file_server6(request):
    # 尝试创建一个IPv6的UDP套接字并连接到本地的::1地址的80端口
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
        sock.connect(("::1", 80, 1, 1))
        has_ipv6 = True
    except OSError:
        has_ipv6 = False
    # 如果没有IPv6支持，则跳过测试
    if not has_ipv6:
        pytest.skip("Ipv6 not supported")

    # 等待0.1秒
    time.sleep(0.1)
    # 创建一个IPv6地址为::1，端口为1544的文件服务器对象
    file_server6 = FileServer("::1", 1544)
    file_server6.ip_external = 'fca5:95d6:bfde:d902:8951:276e:1111:a22c'  # 设置外部IP地址为假地址

    # 定义一个监听函数，用于启动文件服务器并监听连接
    def listen():
        ConnectionServer.start(file_server6)
        ConnectionServer.listen(file_server6)

    # 使用gevent异步启动监听函数
    gevent.spawn(listen)
    # 等待端口打开
    for retry in range(10):
        time.sleep(0.1)  # 等待端口打开
        try:
            # 尝试连接到文件服务器
            conn = file_server6.getConnection("::1", 1544)
            conn.close()
            break
        except Exception as err:
            # 打印文件服务器启动错误信息
            print("FileServer6 startup error", Debug.formatException(err))
    # 断言文件服务器正在运行
    assert file_server6.running
    # 重置防洪保护
    file_server6.ip_incoming = {}

    # 定义一个停止函数，用于停止文件服务器
    def stop():
        file_server6.stop()
    # 将停止函数添加到测试用例的收尾执行器中
    request.addfinalizer(stop)
    # 返回文件服务器对象
    return file_server6


# 定义一个名为ui_websocket的fixture，用于模拟UI的WebSocket连接
@pytest.fixture()
def ui_websocket(site, user):
    # 定义一个名为WsMock的类，用于模拟WebSocket连接
    class WsMock:
        def __init__(self):
            self.result = gevent.event.AsyncResult()

        def send(self, data):
            # 发送数据并设置结果
            logging.debug("WsMock: Set result (data: %s) called by %s" % (data, Debug.formatStack()))
            self.result.set(json.loads(data)["result"])

        def getResult(self):
            # 获取结果
            logging.debug("WsMock: Get result")
            back = self.result.get()
            logging.debug("WsMock: Got result (data: %s)" % back)
            self.result = gevent.event.AsyncResult()
            return back

    # 创建一个WsMock对象，并使用site和user参数初始化UiWebsocket对象
    ws_mock = WsMock()
    ui_websocket = UiWebsocket(ws_mock, site, None, user, None)

    # 定义一个名为testAction的函数，用于处理UI的WebSocket请求
    def testAction(action, *args, **kwargs):
        ui_websocket.handleRequest({"id": 0, "cmd": action, "params": list(args) if args else kwargs})
        return ui_websocket.ws.getResult()
    # 将testAction赋值给ui_websocket的testAction属性
    ui_websocket.testAction = testAction
    # 返回ui_websocket对象
    return ui_websocket
# 定义一个会话级别的 fixture，用于管理 Tor
@pytest.fixture(scope="session")
def tor_manager():
    try:
        # 创建 TorManager 对象，并指定文件服务器端口
        tor_manager = TorManager(fileserver_port=1544)
        # 启动 TorManager
        tor_manager.start()
        # 断言 TorManager 的连接不为空
        assert tor_manager.conn is not None
        # 启动 Tor 的 Onion 服务
        tor_manager.startOnions()
    except Exception as err:
        # 如果出现异常，则跳过测试，并抛出异常信息
        raise pytest.skip("Test requires Tor with ControlPort: %s, %s" % (config.tor_controller, err))
    # 返回 TorManager 对象
    return tor_manager

# 定义一个 fixture，用于处理数据库
@pytest.fixture()
def db(request):
    # 设置数据库路径
    db_path = "%s/zeronet.db" % config.data_dir
    # 定义数据库的结构和映射关系
    schema = {
        # 数据库名称和文件路径
        "db_name": "TestDb",
        "db_file": "%s/zeronet.db" % config.data_dir,
        "maps": {
            "data.json": {
                "to_table": [
                    "test",
                    {"node": "test", "table": "test_importfilter", "import_cols": ["test_id", "title"]}
                ]
            }
        },
        "tables": {
            # 定义 test 表的结构和索引
            "test": {
                "cols": [
                    ["test_id", "INTEGER"],
                    ["title", "TEXT"],
                    ["json_id", "INTEGER REFERENCES json (json_id)"]
                ],
                "indexes": ["CREATE UNIQUE INDEX test_id ON test(test_id)"],
                "schema_changed": 1426195822
            },
            # 定义 test_importfilter 表的结构和索引
            "test_importfilter": {
                "cols": [
                    ["test_id", "INTEGER"],
                    ["title", "TEXT"],
                    ["json_id", "INTEGER REFERENCES json (json_id)"]
                ],
                "indexes": ["CREATE UNIQUE INDEX test_importfilter_id ON test_importfilter(test_id)"],
                "schema_changed": 1426195822
            }
        }
    }

    # 如果数据库文件存在，则删除
    if os.path.isfile(db_path):
        os.unlink(db_path)
    # 创建 Db 对象，并传入数据库结构和路径
    db = Db.Db(schema, db_path)
    # 检查数据库表是否存在，如果不存在则创建
    db.checkTables()

    # 定义一个函数用于清理数据库和文件
    def stop():
        db.close("Test db cleanup")
        os.unlink(db_path)

    # 注册清理函数，确保在测试结束后执行
    request.addfinalizer(stop)
    # 返回 Db 对象
    return db

# 定义一个参数化的 fixture，用于处理加密库
@pytest.fixture(params=["sslcrypto", "sslcrypto_fallback", "libsecp256k1"])
def crypt_bitcoin_lib(request, monkeypatch):
    # 使用 monkeypatch 来设置 CryptBitcoin 类的 lib_verify_best 属性为 request.param
    monkeypatch.setattr(CryptBitcoin, "lib_verify_best", request.param)
    # 调用 CryptBitcoin 类的 loadLib 方法，传入 request.param 作为参数
    CryptBitcoin.loadLib(request.param)
    # 返回 CryptBitcoin 对象
    return CryptBitcoin
# 定义一个 pytest 的装饰器，作用域为 function，自动使用
@pytest.fixture(scope='function', autouse=True)
def logCaseStart(request):
    # 声明全局变量 time_start，并赋值为当前时间
    global time_start
    time_start = time.time()
    # 记录测试用例开始的日志信息
    logging.debug("---- Start test case: %s ----" % request._pyfuncitem)
    # 等待所有测试完成后继续执行
    yield None  # Wait until all test done

# 解决 pytest 中的一个 bug，当在 atexit/post-fixture 处理程序中记录日志时出现 I/O 操作关闭文件的错误
def workaroundPytestLogError():
    # 导入 _pytest.capture 模块
    import _pytest.capture
    # 保存 _pytest.capture.EncodedFile.write 的原始方法
    write_original = _pytest.capture.EncodedFile.write

    # 重写 _pytest.capture.EncodedFile.write 方法
    def write_patched(obj, *args, **kwargs):
        try:
            # 尝试调用原始的 write 方法
            write_original(obj, *args, **kwargs)
        except ValueError as err:
            # 捕获 ValueError 异常，如果错误信息为 "I/O operation on closed file"，则忽略
            if str(err) == "I/O operation on closed file":
                pass
            else:
                raise err

    # 重写 _pytest.capture.EncodedFile.flush 方法
    def flush_patched(obj, *args, **kwargs):
        try:
            # 尝试调用原始的 flush 方法
            obj.buffer.flush(*args, **kwargs)
        except ValueError as err:
            # 捕获 ValueError 异常，如果错误信息以 "I/O operation on closed file" 开头，则忽略
            if str(err).startswith("I/O operation on closed file"):
                pass
            else:
                raise err

    # 将重写后的方法赋值给 _pytest.capture.EncodedFile.write 和 _pytest.capture.EncodedFile.flush
    _pytest.capture.EncodedFile.write = write_patched
    _pytest.capture.EncodedFile.flush = flush_patched

# 调用解决 pytest 日志错误的方法
workaroundPytestLogError()

# 定义一个 pytest 的装饰器，作用域为 session，自动使用
@pytest.fixture(scope='session', autouse=True)
def disableLog():
    # 等待所有测试完成后继续执行
    yield None  # Wait until all test done
    # 设置根日志记录器的日志级别为 CRITICAL
    logging.getLogger('').setLevel(logging.getLevelName(logging.CRITICAL))
```