# `ZeroNet\src\util\SocksProxy.py`

```py
# 导入 socket 模块
import socket

# 导入 socks 模块
import socks
# 从 Config 模块中导入 config 变量
from Config import config

# 创建连接函数，接受地址、超时和源地址作为参数
def create_connection(address, timeout=None, source_address=None):
    # 如果地址在本地 IP 列表中
    if address in config.ip_local:
        # 创建一个不使用代理的套接字对象，使用 IPv4 和 TCP 协议
        sock = socket.socket_noproxy(socket.AF_INET, socket.SOCK_STREAM)
        # 连接到指定地址
        sock.connect(address)
    else:
        # 创建一个使用代理的套接字对象
        sock = socks.socksocket()
        # 连接到指定地址
        sock.connect(address)
    # 返回套接字对象
    return sock

# 使用代理进行 DNS 查询
def getaddrinfo(*args):
    # 返回一个包含地址族、套接字类型、协议号、标志、地址元组的列表
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (args[0], args[1]))]

# 为了使用代理，对 socket 进行修改
def monkeyPatch(proxy_ip, proxy_port):
    # 设置默认代理类型为 SOCKS5，代理 IP 和端口
    socks.setdefaultproxy(socks.PROXY_TYPE_SOCKS5, proxy_ip, int(proxy_port))
    # 将 socket.socket_noproxy 设置为 socket
    socket.socket_noproxy = socket.socket
    # 将 socket.socket 设置为使用代理的套接字对象
    socket.socket = socks.socksocket
    # 重写 socket.create_connection 函数
    socket.create_connection = create_connection
    # 重写 socket.getaddrinfo 函数
    socket.getaddrinfo = getaddrinfo
```