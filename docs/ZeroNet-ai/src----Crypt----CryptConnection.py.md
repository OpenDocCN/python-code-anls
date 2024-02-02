# `ZeroNet\src\Crypt\CryptConnection.py`

```py
# 导入所需的模块
import sys
import logging
import os
import ssl
import hashlib
import random

# 从Config模块中导入config对象
from Config import config
# 从util模块中导入helper函数
from util import helper

# 创建CryptConnectionManager类
class CryptConnectionManager:
    # 初始化方法
    def __init__(self):
        # 根据配置文件设置openssl_bin的数值
        if config.openssl_bin_file:
            self.openssl_bin = config.openssl_bin_file
        # 如果是Windows平台，设置openssl_bin为指定路径
        elif sys.platform.startswith("win"):
            self.openssl_bin = "tools\\openssl\\openssl.exe"
        # 如果是Linux平台，设置openssl_bin为指定路径
        elif config.dist_type.startswith("bundle_linux"):
            self.openssl_bin = "../runtime/bin/openssl"
        # 其他情况下，设置openssl_bin为默认路径
        else:
            self.openssl_bin = "openssl"

        # 初始化SSL客户端和服务器的上下文
        self.context_client = None
        self.context_server = None

        # 设置openssl配置文件的模板路径和实际路径
        self.openssl_conf_template = "src/lib/openssl/openssl.cnf"
        self.openssl_conf = config.data_dir + "/openssl.cnf"

        # 设置openssl环境变量
        self.openssl_env = {
            "OPENSSL_CONF": self.openssl_conf,
            "RANDFILE": config.data_dir + "/openssl-rand.tmp"
        }

        # 初始化支持的加密算法列表
        self.crypt_supported = []  # Supported cryptos

        # 设置证书和密钥文件的路径
        self.cacert_pem = config.data_dir + "/cacert-rsa.pem"
        self.cakey_pem = config.data_dir + "/cakey-rsa.pem"
        self.cert_pem = config.data_dir + "/cert-rsa.pem"
        self.cert_csr = config.data_dir + "/cert-rsa.csr"
        self.key_pem = config.data_dir + "/key-rsa.pem"

        # 初始化日志记录器
        self.log = logging.getLogger("CryptConnectionManager")
        # 记录SSL库的版本信息
        self.log.debug("Version: %s" % ssl.OPENSSL_VERSION)

        # 初始化虚假域名列表
        self.fakedomains = [
            "yahoo.com", "amazon.com", "live.com", "microsoft.com", "mail.ru", "csdn.net", "bing.com",
            "amazon.co.jp", "office.com", "imdb.com", "msn.com", "samsung.com", "huawei.com", "ztedevices.com",
            "godaddy.com", "w3.org", "gravatar.com", "creativecommons.org", "hatena.ne.jp",
            "adobe.com", "opera.com", "apache.org", "rambler.ru", "one.com", "nationalgeographic.com",
            "networksolutions.com", "php.net", "python.org", "phoca.cz", "debian.org", "ubuntu.com",
            "nazwa.pl", "symantec.com"
        ]
    # 创建 SSL 上下文
    def createSslContexts(self):
        # 如果服务端和客户端上下文已经存在，则返回 False
        if self.context_server and self.context_client:
            return False
        # 定义加密算法
        ciphers = "ECDHE-RSA-CHACHA20-POLY1305:ECDHE-RSA-AES128-GCM-SHA256:AES128-SHA256:AES256-SHA:"
        ciphers += "!aNULL:!eNULL:!EXPORT:!DSS:!DES:!RC4:!3DES:!MD5:!PSK"

        # 根据 Python 版本选择 SSL 协议
        if hasattr(ssl, "PROTOCOL_TLS"):
            protocol = ssl.PROTOCOL_TLS
        else:
            protocol = ssl.PROTOCOL_TLSv1_2
        # 创建客户端 SSL 上下文
        self.context_client = ssl.SSLContext(protocol)
        self.context_client.check_hostname = False
        self.context_client.verify_mode = ssl.CERT_NONE

        # 创建服务端 SSL 上下文
        self.context_server = ssl.SSLContext(protocol)
        self.context_server.load_cert_chain(self.cert_pem, self.key_pem)

        # 针对每个上下文设置加密算法和选项
        for ctx in (self.context_client, self.context_server):
            ctx.set_ciphers(ciphers)
            ctx.options |= ssl.OP_NO_COMPRESSION
            try:
                ctx.set_alpn_protocols(["h2", "http/1.1"])
                ctx.set_npn_protocols(["h2", "http/1.1"])
            except Exception:
                pass

    # 选择双方都支持的加密算法
    # 返回：加密算法的名称
    def selectCrypt(self, client_supported):
        for crypt in self.crypt_supported:
            if crypt in client_supported:
                return crypt
        return False

    # 为加密包装套接字
    # 返回：包装后的套接字
    # 对给定的套接字进行包装，使用指定的加密方式和证书验证
    def wrapSocket(self, sock, crypt, server=False, cert_pin=None):
        # 如果加密方式是 "tls-rsa"
        if crypt == "tls-rsa":
            # 如果是服务器端，使用服务器端的上下文进行套接字包装
            if server:
                sock_wrapped = self.context_server.wrap_socket(sock, server_side=True)
            # 如果是客户端，使用客户端的上下文进行套接字包装，并指定服务器主机名
            else:
                sock_wrapped = self.context_client.wrap_socket(sock, server_hostname=random.choice(self.fakedomains))
            # 如果有证书 pin，计算套接字的证书哈希值，并与给定的证书 pin 进行比较
            if cert_pin:
                cert_hash = hashlib.sha256(sock_wrapped.getpeercert(True)).hexdigest()
                if cert_hash != cert_pin:
                    raise Exception("Socket certificate does not match (%s != %s)" % (cert_hash, cert_pin))
            # 返回包装后的套接字
            return sock_wrapped
        # 如果加密方式不是 "tls-rsa"，直接返回原始套接字
        else:
            return sock

    # 删除证书文件
    def removeCerts(self):
        # 如果配置中指定保留 SSL 证书，则返回 False
        if config.keep_ssl_cert:
            return False
        # 遍历需要删除的证书文件列表，如果文件存在则删除
        for file_name in ["cert-rsa.pem", "key-rsa.pem", "cacert-rsa.pem", "cakey-rsa.pem", "cacert-rsa.srl", "cert-rsa.csr", "openssl-rand.tmp"]:
            file_path = "%s/%s" % (config.data_dir, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    # 加载和创建证书文件（如果需要的话）
    def loadCerts(self):
        # 如果禁用加密，返回 False
        if config.disable_encryption:
            return False
        # 如果成功创建 SSL RSA 证书，并且 "tls-rsa" 不在支持的加密方式列表中，则将其添加到列表中
        if self.createSslRsaCert() and "tls-rsa" not in self.crypt_supported:
            self.crypt_supported.append("tls-rsa")

    # 尝试创建 RSA 服务器证书并签名以用于连接加密
    # 返回：成功返回 True
# 创建一个加密连接管理器的实例
manager = CryptConnectionManager()
```