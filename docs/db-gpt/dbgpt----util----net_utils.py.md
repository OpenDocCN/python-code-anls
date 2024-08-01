# `.\DB-GPT-src\dbgpt\util\net_utils.py`

```py
import errno
import socket


# 定义一个函数用于获取本地网络接口的 IP 地址
def _get_ip_address(address: str = "10.254.254.254:1") -> str:
    # 从传入的地址中提取 IP 和端口号
    ip, port = address.split(":")
    # 创建一个 UDP 套接字
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 设置套接字超时时间为0，即非阻塞模式
    s.settimeout(0)
    # 初始化当前 IP 地址为本地回环地址
    curr_address = "127.0.0.1"
    try:
        # 尝试连接到指定的 IP 和端口，实际上并不需要连接成功
        s.connect((ip, int(port)))
        # 获取套接字的本地地址，即当前系统对外通信的 IP 地址
        curr_address = s.getsockname()[0]
    except OSError as e:
        # 处理操作系统抛出的异常
        IP = "127.0.0.1"
        # 如果出现网络不可达的异常
        if e.errno == errno.ENETUNREACH:
            try:
                # 获取本地主机的全限定域名，并尝试获取其 IP 地址
                hostname = socket.getfqdn(socket.gethostname())
                curr_address = socket.gethostbyname(hostname)
            except Exception:
                pass
    finally:
        # 关闭套接字
        s.close()
    # 返回获取到的当前 IP 地址
    return curr_address
```