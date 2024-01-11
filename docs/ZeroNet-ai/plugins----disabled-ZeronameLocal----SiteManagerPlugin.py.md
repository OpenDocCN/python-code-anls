# `ZeroNet\plugins\disabled-ZeronameLocal\SiteManagerPlugin.py`

```
# 导入所需的模块
import logging, json, os, re, sys, time, socket
from Plugin import PluginManager
from Config import config
from Debug import Debug
from http.client import HTTPSConnection, HTTPConnection, HTTPException
from base64 import b64encode

# 设置不支持重新加载
allow_reload = False 

# 将该类注册到"SiteManager"插件管理器中
@PluginManager.registerTo("SiteManager")
class SiteManagerPlugin(object):
    # 调用父类的 load 方法，传入所有位置参数和关键字参数
    def load(self, *args, **kwargs):
        super(SiteManagerPlugin, self).load(*args, **kwargs)
        # 获取名为 "ZeronetLocal Plugin" 的日志记录器
        self.log = logging.getLogger("ZeronetLocal Plugin")
        self.error_message = None
        # 检查是否缺少连接到 namecoin 节点所需的参数
        if not config.namecoin_host or not config.namecoin_rpcport or not config.namecoin_rpcuser or not config.namecoin_rpcpassword:
            self.error_message = "Missing parameters"
            self.log.error("Missing parameters to connect to namecoin node. Please check all the arguments needed with '--help'. Zeronet will continue working without it.")
            return

        # 构建连接 namecoin 节点的 URL
        url = "%(host)s:%(port)s" % {"host": config.namecoin_host, "port": config.namecoin_rpcport}
        # 创建 HTTP 连接对象
        self.c = HTTPConnection(url, timeout=3)
        # 构建用户名和密码的 base64 编码
        user_pass = "%(user)s:%(password)s" % {"user": config.namecoin_rpcuser, "password": config.namecoin_rpcpassword}
        userAndPass = b64encode(bytes(user_pass, "utf-8")).decode("ascii")
        # 设置请求头部信息
        self.headers = {"Authorization" : "Basic %s" %  userAndPass, "Content-Type": " application/json " }

        # 构建 JSON 格式的请求数据
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": "zeronet",
            "method": "ping",
            "params": []
        })

        try:
            # 发送 POST 请求到 namecoin 节点
            self.c.request("POST", "/", payload, headers=self.headers)
            response = self.c.getresponse()
            data = response.read()
            self.c.close()
            # 检查响应状态码，如果为 200 则解析响应数据
            if response.status == 200:
                result = json.loads(data.decode())["result"]
            else:
                raise Exception(response.reason)
        except Exception as err:
            # 捕获异常并记录错误信息
            self.log.error("The Namecoin node is unreachable. Please check the configuration value are correct. Zeronet will continue working without it.")
            self.error_message = err
        # 初始化缓存字典
        self.cache = dict()

    # 检查是否为有效地址
    def isAddress(self, address):
        return self.isBitDomain(address) or super(SiteManagerPlugin, self).isAddress(address)
    # 如果地址是域名，则返回True
    def isDomain(self, address):
        return self.isBitDomain(address) or super(SiteManagerPlugin, self).isDomain(address)
    
    # 如果地址是.bit域名，则返回True
    def isBitDomain(self, address):
        return re.match(r"(.*?)([A-Za-z0-9_-]+\.bit)$", address)
    
    # 获取网站对象，如果找不到则返回None
    def get(self, address):
        if self.isBitDomain(address):  # 看起来像是一个域名
            address_resolved = self.resolveDomain(address)
            if address_resolved:  # 找到了域名
                site = self.sites.get(address_resolved)
                if site:
                    site_domain = site.settings.get("domain")
                    if site_domain != address:
                        site.settings["domain"] = address
            else:  # 域名未找到
                site = self.sites.get(address)
        else:  # 通过网站地址访问
            site = super(SiteManagerPlugin, self).get(address)
        return site
    
    # 返回或创建网站并开始下载网站文件
    # 如果DNS解析失败，则返回Site或None
    def need(self, address, *args, **kwargs):
        if self.isBitDomain(address):  # 看起来像是一个域名
            address_resolved = self.resolveDomain(address)
            if address_resolved:
                address = address_resolved
            else:
                return None
        return super(SiteManagerPlugin, self).need(address, *args, **kwargs)
    
    # 解析域名
    # 返回：地址或None
# 将 ConfigPlugin 类注册到 PluginManager 中的 ConfigPlugin 插件中
@PluginManager.registerTo("ConfigPlugin")
class ConfigPlugin(object):
    # 创建参数
    def createArguments(self):
        # 添加一个参数组到解析器中，用于 Zeroname 本地插件
        group = self.parser.add_argument_group("Zeroname Local plugin")
        # 添加一个参数，用于指定 namecoin 节点的主机地址
        group.add_argument('--namecoin_host', help="Host to namecoin node (eg. 127.0.0.1)")
        # 添加一个参数，用于指定连接的端口
        group.add_argument('--namecoin_rpcport', help="Port to connect (eg. 8336)")
        # 添加一个参数，用于指定连接到 namecoin 节点的 RPC 用户
        group.add_argument('--namecoin_rpcuser', help="RPC user to connect to the namecoin node (eg. nofish)")
        # 添加一个参数，用于指定连接到 namecoin 节点的 RPC 密码
        group.add_argument('--namecoin_rpcpassword', help="RPC password to connect to namecoin node")

        # 调用父类的 createArguments 方法
        return super(ConfigPlugin, self).createArguments()
```