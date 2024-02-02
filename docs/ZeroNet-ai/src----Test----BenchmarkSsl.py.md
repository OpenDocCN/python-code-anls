# `ZeroNet\src\Test\BenchmarkSsl.py`

```py
#!/usr/bin/python2
# 指定脚本解释器为 Python2

from gevent import monkey
monkey.patch_all()
# 使用 gevent monkey patch 来实现协程

import os
import time
import sys
import socket
import ssl
sys.path.append(os.path.abspath(".."))  # Imports relative to src dir
# 将当前路径添加到系统路径中，以便导入相对于 src 目录的模块

import io as StringIO
import gevent
# 导入所需的模块

from gevent.server import StreamServer
from gevent.pool import Pool
from Config import config
config.parse()
# 从 Config 模块中导入 config 对象，并解析配置

from util import SslPatch
# 从 util 模块中导入 SslPatch 类

# Server
socks = []
data = os.urandom(1024 * 100)
data += "\n"
# 初始化变量 socks 和 data

def handle(sock_raw, addr):
    socks.append(sock_raw)
    sock = sock_raw
    # 将原始套接字添加到 socks 列表中，并赋值给 sock 变量
    # sock = ctx.wrap_socket(sock, server_side=True)
    # if sock_raw.recv( 1, gevent.socket.MSG_PEEK ) == "\x16":
    #   sock = gevent.ssl.wrap_socket(sock_raw, server_side=True, keyfile='key-cz.pem',
    #          certfile='cert-cz.pem', ciphers=ciphers, ssl_version=ssl.PROTOCOL_TLSv1)
    # fp = os.fdopen(sock.fileno(), 'rb', 1024*512)
    # 尝试使用 SSL 包装套接字

    try:
        while True:
            line = sock.recv(16 * 1024)
            if not line:
                break
            if line == "bye\n":
                break
            elif line == "gotssl\n":
                sock.sendall("yes\n")
                sock = gevent.ssl.wrap_socket(
                    sock_raw, server_side=True, keyfile='../../data/key-rsa.pem', certfile='../../data/cert-rsa.pem',
                    ciphers=ciphers, ssl_version=ssl.PROTOCOL_TLSv1
                )
            else:
                sock.sendall(data)
    except Exception as err:
        print(err)
    # 接收和处理套接字数据，如果出现异常则打印错误信息

    try:
        sock.shutdown(gevent.socket.SHUT_WR)
        sock.close()
    except:
        pass
    # 尝试关闭套接字

    socks.remove(sock_raw)
    # 从 socks 列表中移除原始套接字

pool = Pool(1000)  # do not accept more than 10000 connections
# 创建协程池，最多接受 1000 个连接
server = StreamServer(('127.0.0.1', 1234), handle)
server.start()
# 创建流式服务器，并启动

# Client

total_num = 0
total_bytes = 0
clipher = None
ciphers = "ECDHE-ECDSA-AES128-GCM-SHA256:ECDH+AES128:ECDHE-RSA-AES128-GCM-SHA256:AES128-GCM-SHA256:AES128-SHA256:AES128-SHA:HIGH:" + \
    "!aNULL:!eNULL:!EXPORT:!DSS:!DES:!RC4:!3DES:!MD5:!PSK"
# 初始化变量 total_num, total_bytes, clipher 和 ciphers
# 创建 SSL 上下文对象，使用 SSLv23 协议
# ctx = ssl.SSLContext(ssl.PROTOCOL_SSLv23)

# 定义获取数据的函数
def getData():
    global total_num, total_bytes, clipher
    data = None
    # 创建套接字对象
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到指定的地址和端口
    sock.connect(("127.0.0.1", 1234)
    # 发送数据
    sock.send("gotssl\n")
    # 接收数据并判断是否为 "yes\n"
    if sock.recv(128) == "yes\n":
        # 使用指定的密码和 SSL 版本包装套接字
        sock = ssl.wrap_socket(sock, ciphers=ciphers, ssl_version=ssl.PROTOCOL_TLSv1)
        # 进行 SSL 握手
        sock.do_handshake()
        # 获取 SSL 密钥信息
        clipher = sock.cipher()

    # 循环发送请求并接收数据
    for req in range(20):
        sock.sendall("req\n")
        buff = StringIO.StringIO()
        data = sock.recv(16 * 1024)
        buff.write(data)
        if not data:
            break
        while not data.endswith("\n"):
            data = sock.recv(16 * 1024)
            if not data:
                break
            buff.write(data)
        total_num += 1
        total_bytes += buff.tell()
        if not data:
            print("No data")

    # 关闭写入端并关闭套接字
    sock.shutdown(gevent.socket.SHUT_WR)
    sock.close()

# 获取当前时间
s = time.time()

# 定义信息函数
def info():
    import psutil
    import os
    # 获取当前进程的 PID
    process = psutil.Process(os.getpid())
    # 获取内存信息
    if "memory_info" in dir(process):
        memory_info = process.memory_info
    else:
        memory_info = process.get_memory_info
    # 循环打印信息
    while 1:
        print(total_num, "req", (total_bytes / 1024), "kbytes", "transfered in", time.time() - s, end=' ')
        print("using", clipher, "Mem:", memory_info()[0] / float(2 ** 20))
        time.sleep(1)

# 使用协程创建信息函数
gevent.spawn(info)

# 循环创建客户端并发送请求
for test in range(1):
    clients = []
    for i in range(500):  # Thread
        clients.append(gevent.spawn(getData))
    gevent.joinall(clients)

# 打印总请求数和传输的数据量
print(total_num, "req", (total_bytes / 1024), "kbytes", "transfered in", time.time() - s)

# 分离客户端/服务器进程：
# 10*10*100:
# Raw:      10000 req 1000009 kbytes transfered in 5.39999985695
# RSA 2048: 10000 req 1000009 kbytes transfered in 27.7890000343 using ('ECDHE-RSA-AES256-SHA', 'TLSv1/SSLv3', 256)
# ECC:      10000 req 1000009 kbytes transfered in 26.1959998608 using ('ECDHE-ECDSA-AES256-SHA', 'TLSv1/SSLv3', 256)
# ECC:      10000 req 1000009 kbytes transfered in 28.2410001755 using ('ECDHE-ECDSA-AES256-GCM-SHA384', 'TLSv1/SSLv3', 256) Mem: 13.3828125
#
# 10*100*10:
# Raw:      10000 req 1000009 kbytes transfered in 7.02700018883 Mem: 14.328125
# RSA 2048: 10000 req 1000009 kbytes transfered in 44.8860001564 using ('ECDHE-RSA-AES256-GCM-SHA384', 'TLSv1/SSLv3', 256) Mem: 20.078125
# ECC:      10000 req 1000009 kbytes transfered in 37.9430000782 using ('ECDHE-ECDSA-AES256-GCM-SHA384', 'TLSv1/SSLv3', 256) Mem: 20.0234375
#
# 1*100*100:
# Raw:      10000 req 1000009 kbytes transfered in 4.64400005341 Mem: 14.06640625
# RSA:      10000 req 1000009 kbytes transfered in 24.2300000191 using ('ECDHE-RSA-AES256-GCM-SHA384', 'TLSv1/SSLv3', 256) Mem: 19.7734375
# ECC:      10000 req 1000009 kbytes transfered in 22.8849999905 using ('ECDHE-ECDSA-AES256-GCM-SHA384', 'TLSv1/SSLv3', 256) Mem: 17.8125
# AES128:   10000 req 1000009 kbytes transfered in 21.2839999199 using ('AES128-GCM-SHA256', 'TLSv1/SSLv3', 128) Mem: 14.1328125
# ECC+128:  10000 req 1000009 kbytes transfered in 20.496999979  using ('ECDHE-ECDSA-AES128-GCM-SHA256', 'TLSv1/SSLv3', 128) Mem: 14.40234375
#
#
# Single process:
# 1*100*100
# RSA:      10000 req 1000009 kbytes transfered in 41.7899999619 using ('ECDHE-RSA-AES128-GCM-SHA256', 'TLSv1/SSLv3', 128) Mem: 26.91015625
#
# 10*10*100
# RSA:      10000 req 1000009 kbytes transfered in 40.1640000343 using ('ECDHE-RSA-AES128-GCM-SHA256', 'TLSv1/SSLv3', 128) Mem: 14.94921875
```